"""
TurboQuant Triton kernels for KV cache quantization.

Implements the TurboQuant algorithm from "TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate" (Zandieh et al., ICLR 2026).

The algorithm works in two stages:
  Stage 1 (PolarQuant): Random rotation via Hadamard transform + per-coordinate
           scalar quantization using precomputed optimal centroids.
  Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on the residual for
           unbiased inner product estimation.

For KV cache compression at b total bits per coordinate:
  - TurboQuant_mse uses all b bits for MSE-optimal quantization (Stage 1 only)
  - TurboQuant_prod uses (b-1) bits for Stage 1 + 1 bit QJL for Stage 2
"""

import logging
import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Profiling: set TURBOQUANT_PROFILE=1 to measure per-kernel GPU time.
# This adds CUDA event overhead — never leave on in production.
_PROFILE_TQ = os.environ.get("TURBOQUANT_PROFILE", "0") == "1"
_profile_timings: dict = {
    "prepare": [],
    "hadamard": [],
    "quantize_pack": [],
    "total": [],
}


def _profile_report():
    """Print profiling summary and reset counters."""
    if not _profile_timings["total"]:
        return
    n = len(_profile_timings["total"])
    total_ms = sum(_profile_timings["total"])
    print(
        f"\n=== TurboQuant Profiling ({n} quantize calls, {total_ms:.1f}ms total) ==="
    )
    for key in ["prepare", "hadamard", "quantize_pack"]:
        vals = _profile_timings[key]
        if vals:
            avg = sum(vals) / len(vals)
            total = sum(vals)
            pct = total / total_ms * 100 if total_ms > 0 else 0
            print(f"  {key:>15s}: avg={avg:.3f}ms, total={total:.1f}ms ({pct:.1f}%)")
    print(f"  {'TOTAL':>15s}: avg={total_ms/n:.3f}ms, total={total_ms:.1f}ms")
    # Reset
    for k in _profile_timings:
        _profile_timings[k].clear()


# ---------------------------------------------------------------------------
# Precomputed optimal centroids for the Beta-distributed coordinates after
# random rotation. These are the MSE-optimal scalar quantizer centroids for
# a standard normal distribution (the high-dimensional limit of the Beta
# distribution after rotation), computed via Lloyd-Max algorithm.
#
# For b bits we have 2^b centroids.  The values below are for a zero-mean,
# unit-variance Gaussian (the limiting distribution in high dimensions).
# At quantization time they are scaled by 1/sqrt(d) to match the actual
# coordinate distribution.
# ---------------------------------------------------------------------------

# 1-bit (2 centroids): optimal for N(0,1) -> +/- 0.7979 (= sqrt(2/pi))
CENTROIDS_1BIT = [-0.7978845608, 0.7978845608]

# 2-bit (4 centroids): Lloyd-Max for N(0,1)
CENTROIDS_2BIT = [-1.510, -0.4528, 0.4528, 1.510]

# 3-bit (8 centroids): Lloyd-Max for N(0,1)
CENTROIDS_3BIT = [
    -2.152,
    -1.344,
    -0.7560,
    -0.2451,
    0.2451,
    0.7560,
    1.344,
    2.152,
]

# 4-bit (16 centroids): Lloyd-Max for N(0,1)
CENTROIDS_4BIT = [
    -2.733,
    -2.069,
    -1.618,
    -1.256,
    -0.9424,
    -0.6568,
    -0.3881,
    -0.1284,
    0.1284,
    0.3881,
    0.6568,
    0.9424,
    1.256,
    1.618,
    2.069,
    2.733,
]


_centroids_cache: dict = {}
_scaled_centroids_cache: dict = {}
_scaled_boundaries_cache: dict = {}

# JIT Hadamard kernel availability
try:
    from sglang.jit_kernel.hadamard import hadamard_transform as _jit_hadamard_transform

    _HAS_JIT_HADAMARD = True
except ImportError:
    _HAS_JIT_HADAMARD = False

# Track which (dtype, dim) combos have been JIT-warmed to avoid redundant compilation
_jit_warmed: set = set()


class QuantizeWorkspace:
    """Pre-allocated scratch buffers for zero-allocation quantization.

    Used during CUDA graph capture to avoid temporary tensor allocations
    that would inflate graph memory pools.  A single workspace is reused
    across all layers since they execute sequentially.
    """

    def __init__(self, max_rows: int, padded_dim: int, device: torch.device):
        self.max_rows = max_rows
        self.padded_dim = padded_dim
        packed_dim = padded_dim // 2  # 4-bit nibble packing

        # Norms buffer (prepare output / quantize_pack input)
        self.norms = torch.empty(max_rows, dtype=torch.float32, device=device)
        # Prepare output / Hadamard input (float32, signs-applied + padded)
        self.rotated = torch.empty(
            max_rows, padded_dim, dtype=torch.float32, device=device
        )
        # JIT Hadamard output buffer / quantize_pack input
        self.fwht_out = torch.empty(
            max_rows, padded_dim, dtype=torch.float32, device=device
        )
        # Quantize+pack output (4-bit packed uint8)
        self.packed = torch.empty(
            max_rows, packed_dim, dtype=torch.uint8, device=device
        )

    def memory_bytes(self) -> int:
        total = 0
        for attr in (
            "norms",
            "rotated",
            "fwht_out",
            "packed",
        ):
            t = getattr(self, attr)
            total += t.numel() * t.element_size()
        return total


def _get_centroids_tensor(bits: int, device: torch.device) -> torch.Tensor:
    """Return the centroid tensor for the given bit-width (cached per device)."""
    key = (bits, device)
    if key not in _centroids_cache:
        table = {
            1: CENTROIDS_1BIT,
            2: CENTROIDS_2BIT,
            3: CENTROIDS_3BIT,
            4: CENTROIDS_4BIT,
        }
        if bits not in table:
            raise ValueError(f"TurboQuant supports 1-4 bits, got {bits}")
        _centroids_cache[key] = torch.tensor(
            table[bits], dtype=torch.float32, device=device
        )
    return _centroids_cache[key]


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT) — used as the random rotation.
#
# We use a randomized Hadamard transform: H_d * diag(s) where s_i ~ Rademacher
# (random +/-1).  This is O(d log d) and a near-isometry, matching the paper's
# requirement of a random rotation that makes coordinates near-independent.
# ---------------------------------------------------------------------------


def _generate_random_signs(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a deterministic Rademacher vector (+1/-1) for the randomized Hadamard."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1).to(device)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class HadamardTransform:
    """Manages the randomized Hadamard transform for TurboQuant.

    The transform is:  y = (1/sqrt(d)) * H_d * diag(signs) * x

    where H_d is the Walsh-Hadamard matrix and signs are random +/-1.
    """

    def __init__(self, dim: int, seed: int = 42, device: torch.device = None):
        if device is None:
            device = torch.device("cuda")
        self.dim = dim
        self.padded_dim = _next_power_of_2(dim)
        self.signs = _generate_random_signs(self.padded_dim, seed, device)
        self.scale = 1.0 / math.sqrt(self.padded_dim)
        self.device = device
        self._use_jit = _HAS_JIT_HADAMARD

        # JIT warmup: trigger compilation before CUDA graph capture.
        # If the JIT build/load fails at runtime (e.g. missing compiler,
        # unsupported GPU), fall back to the Python butterfly silently.
        if self._use_jit and device.type == "cuda":
            key = (torch.float32, self.padded_dim)
            if key not in _jit_warmed:
                try:
                    dummy = torch.zeros(
                        1, self.padded_dim, dtype=torch.float32, device=device
                    )
                    _jit_hadamard_transform(dummy, scale=1.0)
                    _jit_warmed.add(key)
                except Exception:
                    self._use_jit = False

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor = None,
        fwht_out: Optional[torch.Tensor] = None,
        skip_signs: bool = False,
    ) -> torch.Tensor:
        """Apply randomized Hadamard: y = scale * H * diag(signs) * x.

        Args:
            x: (..., dim) tensor
            out: optional pre-allocated output tensor (..., padded_dim).
                 When provided, x is copied into out and transformed in-place.
            fwht_out: optional pre-allocated output buffer for the JIT Hadamard
                      kernel (shape matches x after padding). When using the Python
                      fallback, this is used as a butterfly scratch buffer (only
                      needs padded_dim//2 cols, but padded_dim works too).
            skip_signs: if True, skip the signs multiplication (caller already
                        applied signs, e.g. in the fused prepare kernel).
        Returns:
            (..., padded_dim) tensor of rotated coordinates
        """
        shape = x.shape
        d = shape[-1]

        # Pad to power-of-2 if needed
        if d < self.padded_dim:
            if out is not None:
                out[..., :d] = x
                out[..., d:] = 0
                x = out
            else:
                x = torch.nn.functional.pad(x, (0, self.padded_dim - d))
        elif out is not None:
            out.copy_(x)
            x = out

        # Apply random signs (unless already applied by fused prepare kernel)
        if not skip_signs:
            x.mul_(self.signs)

        # Hadamard transform
        if self._use_jit:
            # Single CUDA kernel, zero-alloc when fwht_out is provided
            x = _jit_hadamard_transform(x, scale=self.scale, out=fwht_out)
        else:
            # Python butterfly fallback (fwht_out doubles as scratch buffer)
            x = self._fwht_inplace(x, tmp=fwht_out)
            x.mul_(self.scale)
        return x

    def inverse(
        self,
        y: torch.Tensor,
        out: torch.Tensor = None,
        fwht_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply inverse randomized Hadamard: x = diag(signs) * H * scale * y.

        Since H is symmetric and orthogonal: H^{-1} = H / d.
        So full inverse = diag(signs) * (1/d) * H * (y / scale)
        But scale = 1/sqrt(d), so (1/d) * (1/scale) = 1/sqrt(d) = scale.
        """
        if self._use_jit:
            if out is not None:
                out.copy_(y)
                x = _jit_hadamard_transform(out, scale=self.scale, out=fwht_out)
            else:
                x = _jit_hadamard_transform(y, scale=self.scale, out=fwht_out)
        else:
            if out is not None:
                out.copy_(y)
                x = self._fwht_inplace(out, tmp=fwht_out)
            else:
                x = self._fwht_inplace(y, tmp=fwht_out)
            x.mul_(self.scale)
        x.mul_(self.signs)
        return x[..., : self.dim]

    @staticmethod
    def _fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform along the last dimension."""
        return HadamardTransform._fwht_inplace(x)

    @staticmethod
    def _fwht_inplace(
        x: torch.Tensor, tmp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform — in-place butterfly, minimal allocs.

        Args:
            x: input tensor (last dim must be power-of-2), modified in-place.
            tmp: optional pre-allocated (max_rows, n//2) float32 buffer.
                 When provided, no memory is allocated (CUDA-graph safe).
        """
        orig_shape = x.shape
        n = orig_shape[-1]
        x = x.reshape(-1, n).float()
        rows = x.shape[0]
        # Single temp buffer reused across all butterfly stages
        if tmp is None:
            tmp = torch.empty(rows, n // 2, dtype=x.dtype, device=x.device)
        else:
            tmp = tmp[:rows, : n // 2]
        h = 1
        while h < n:
            x_view = x.view(rows, n // (2 * h), 2, h)
            a = x_view[:, :, 0, :]
            b = x_view[:, :, 1, :]
            tmp_view = tmp.view(rows, n // (2 * h), h)
            # Save b, compute a+b and a-b in-place
            tmp_view.copy_(b)
            b.copy_(a).sub_(tmp_view)  # b = a - b_orig
            a.add_(tmp_view)  # a = a_orig + b_orig
            h *= 2
        return x.view(orig_shape)


# ---------------------------------------------------------------------------
# Bit-packing helpers
#
# For b-bit quantization, pack multiple indices per byte:
#   4-bit: 2 per byte (nibble packing)   -> packed_dim = padded_dim / 2
#   3-bit: 8 per 3 bytes (24-bit groups)  -> packed_dim = padded_dim * 3 / 8
#   2-bit: 4 per byte                     -> packed_dim = padded_dim / 4
#   1-bit: 8 per byte                     -> packed_dim = padded_dim / 8
# ---------------------------------------------------------------------------


def compute_packed_dim(padded_dim: int, bits: int) -> int:
    """Compute the byte size of a packed index buffer."""
    if bits == 4:
        return padded_dim // 2
    elif bits == 3:
        assert (
            padded_dim % 8 == 0
        ), "padded_dim must be divisible by 8 for 3-bit packing"
        return (padded_dim * 3) // 8
    elif bits == 2:
        return padded_dim // 4
    elif bits == 1:
        return padded_dim // 8
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 centroid indices into sub-byte representation.

    Args:
        indices: (..., padded_dim) uint8 tensor with values in [0, 2^bits)
        bits: 1, 2, 3, or 4
    Returns:
        (..., packed_dim) uint8 tensor
    """
    if bits == 4:
        even = indices[..., 0::2].to(torch.int32)
        odd = indices[..., 1::2].to(torch.int32)
        return ((odd << 4) | (even & 0x0F)).to(torch.uint8)
    elif bits == 2:
        i0 = indices[..., 0::4].to(torch.int32)
        i1 = indices[..., 1::4].to(torch.int32)
        i2 = indices[..., 2::4].to(torch.int32)
        i3 = indices[..., 3::4].to(torch.int32)
        return ((i3 << 6) | (i2 << 4) | (i1 << 2) | (i0 & 0x03)).to(torch.uint8)
    elif bits == 1:
        result = indices[..., 0::8].to(torch.int32) & 1
        for i in range(1, 8):
            result = result | ((indices[..., i::8].to(torch.int32) & 1) << i)
        return result.to(torch.uint8)
    elif bits == 3:
        padded_dim = indices.shape[-1]
        batch_shape = indices.shape[:-1]
        num_groups = padded_dim // 8
        groups = indices.reshape(*batch_shape, num_groups, 8).to(torch.int32)
        # Pack 8 x 3-bit = 24 bits into a 32-bit int, then split into 3 bytes
        packed_24 = groups[..., 0] & 0x07
        for i in range(1, 8):
            packed_24 = packed_24 | ((groups[..., i] & 0x07) << (i * 3))
        b0 = (packed_24 & 0xFF).to(torch.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(*batch_shape, num_groups * 3)
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def _pack_4bit_inplace(
    indices: torch.Tensor,
    even_buf: torch.Tensor,
    odd_buf: torch.Tensor,
    packed_buf: torch.Tensor,
) -> torch.Tensor:
    """Pack 4-bit indices using pre-allocated int32 buffers.  Zero allocation.

    Args:
        indices: (n, padded_dim) uint8, values in [0, 15]
        even_buf: (n, padded_dim//2) int32, scratch
        odd_buf:  (n, padded_dim//2) int32, scratch
        packed_buf: (n, padded_dim//2) uint8, output
    Returns:
        packed_buf with nibble-packed results
    """
    even_buf.copy_(indices[..., 0::2])  # uint8 → int32
    odd_buf.copy_(indices[..., 1::2])  # uint8 → int32
    odd_buf.mul_(16)  # << 4
    odd_buf.add_(even_buf)  # high_nibble | low_nibble
    packed_buf.copy_(odd_buf)  # int32 → uint8 (truncates to low byte)
    return packed_buf


def unpack_indices(packed: torch.Tensor, bits: int, padded_dim: int) -> torch.Tensor:
    """Unpack sub-byte indices back to uint8.

    Args:
        packed: (..., packed_dim) uint8 tensor
        bits: 1, 2, 3, or 4
        padded_dim: original dimension before packing
    Returns:
        (..., padded_dim) uint8 tensor
    """
    if bits == 4:
        p = packed.to(torch.int32)
        even = (p & 0x0F).to(torch.uint8)
        odd = ((p >> 4) & 0x0F).to(torch.uint8)
        return torch.stack([even, odd], dim=-1).reshape(*packed.shape[:-1], padded_dim)
    elif bits == 2:
        p = packed.to(torch.int32)
        i0 = (p & 0x03).to(torch.uint8)
        i1 = ((p >> 2) & 0x03).to(torch.uint8)
        i2 = ((p >> 4) & 0x03).to(torch.uint8)
        i3 = ((p >> 6) & 0x03).to(torch.uint8)
        return torch.stack([i0, i1, i2, i3], dim=-1).reshape(
            *packed.shape[:-1], padded_dim
        )
    elif bits == 1:
        p = packed.to(torch.int32)
        parts = [((p >> i) & 1).to(torch.uint8) for i in range(8)]
        return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], padded_dim)
    elif bits == 3:
        batch_shape = packed.shape[:-1]
        num_groups = packed.shape[-1] // 3
        bytes_g = packed.reshape(*batch_shape, num_groups, 3).to(torch.int32)
        packed_24 = bytes_g[..., 0] | (bytes_g[..., 1] << 8) | (bytes_g[..., 2] << 16)
        parts = [((packed_24 >> (i * 3)) & 0x07).to(torch.uint8) for i in range(8)]
        return torch.stack(parts, dim=-1).reshape(*batch_shape, padded_dim)
    else:
        raise ValueError(f"Unsupported bits: {bits}")


# ---------------------------------------------------------------------------
# Triton kernels
#
# _turboquant_quantize_kernel: find nearest centroid, output unpacked uint8
# _turboquant_dequantize_packed_4bit_kernel: fused unpack + centroid lookup
# _turboquant_dequantize_kernel: legacy unpacked dequant (used for prod mode
#   residual computation which needs unpacked indices)
# ---------------------------------------------------------------------------


@triton.jit
def _turboquant_prepare_kernel(
    # Pointers
    x_ptr,  # [N, dim] bf16/f32 input
    signs_ptr,  # [padded_dim] f32 signs
    out_ptr,  # [N, padded_dim] f32 output (signs-applied, padded)
    norms_ptr,  # [N] f32 output (L2 norms)
    # Strides
    x_stride_0,
    out_stride_0,
    # Constants
    DIM: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused bf16→f32 + zero-pad + L2 norm + signs multiply."""
    row = tl.program_id(0)
    norm_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, PADDED_DIM, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        in_bounds = offs < DIM
        pad_bounds = offs < PADDED_DIM
        # Load bf16 → f32 (zero for padded region)
        vals = tl.load(x_ptr + row * x_stride_0 + offs, mask=in_bounds, other=0.0).to(
            tl.float32
        )
        # Accumulate L2 norm (only real dims)
        norm_acc += tl.where(in_bounds, vals * vals, 0.0)
        # Multiply by signs
        signs = tl.load(signs_ptr + offs, mask=pad_bounds, other=0.0)
        out_vals = vals * signs
        tl.store(out_ptr + row * out_stride_0 + offs, out_vals, mask=pad_bounds)
    # Store norm
    tl.store(norms_ptr + row, tl.sqrt(tl.sum(norm_acc, axis=0)))


@triton.jit
def _turboquant_quantize_pack_kernel(
    # Pointers
    rotated_ptr,  # [N, padded_dim] f32
    norms_ptr,  # [N] f32
    boundaries_ptr,  # [num_centroids - 1] f32 decision boundaries
    packed_ptr,  # [N, packed_dim] u8 output
    # Strides
    rotated_stride_0,
    packed_stride_0,
    # Direct pool write params (pool_packed_ptr == 0 means disabled)
    pool_packed_ptr,  # pool's packed buffer for this layer
    pool_norms_ptr,  # pool's norms buffer for this layer
    loc_ptr,  # [num_tokens] int64 location indices
    pool_packed_stride_0,  # stride(0) of pool packed buffer
    pool_packed_stride_1,  # stride(1) of pool packed buffer
    pool_norms_stride_0,  # stride(0) of pool norms buffer
    # Constants
    PADDED_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    NUM_CENTROIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,  # 0 = no pool write
):
    """Fused normalize + quantize to nearest centroid + 4-bit pack.

    Uses precomputed decision boundaries (midpoints between adjacent centroids)
    instead of linear search. Index = count of boundaries the value exceeds.
    This is 15 comparisons + 15 additions vs 16 × (sub + mul + 2 cmp + 2 select).

    When HEAD_NUM > 0, also writes packed data and norms directly to pool
    buffers at scatter locations, eliminating separate scatter writes.
    """
    row = tl.program_id(0)
    block_id = tl.program_id(1)

    norm = tl.maximum(tl.load(norms_ptr + row), 1e-10)
    inv_norm = 1.0 / norm

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < PACKED_DIM

    # Load even and odd coordinates
    even_offs = offs * 2
    odd_offs = offs * 2 + 1
    even_vals = (
        tl.load(rotated_ptr + row * rotated_stride_0 + even_offs, mask=mask, other=0.0)
        * inv_norm
    )
    odd_vals = (
        tl.load(rotated_ptr + row * rotated_stride_0 + odd_offs, mask=mask, other=0.0)
        * inv_norm
    )

    # Boundary-based nearest centroid: index = number of boundaries exceeded
    best_even = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    best_odd = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for b in tl.static_range(NUM_CENTROIDS - 1):
        boundary = tl.load(boundaries_ptr + b)
        best_even += (even_vals >= boundary).to(tl.int32)
        best_odd += (odd_vals >= boundary).to(tl.int32)

    # Pack: low nibble = even, high nibble = odd
    packed = ((best_odd << 4) | (best_even & 0x0F)).to(tl.uint8)

    # Write to workspace (always)
    tl.store(packed_ptr + row * packed_stride_0 + offs, packed, mask=mask)

    # Write directly to pool buffer if enabled
    if HEAD_NUM > 0:
        token_id = row // HEAD_NUM
        head_id = row % HEAD_NUM
        pool_loc = tl.load(loc_ptr + token_id)
        pool_base = pool_loc * pool_packed_stride_0 + head_id * pool_packed_stride_1
        tl.store(pool_packed_ptr + pool_base + offs, packed, mask=mask)

        # Write norms (only from first block to avoid duplicate writes)
        if block_id == 0:
            pool_norms_base = pool_loc * pool_norms_stride_0 + head_id
            tl.store(pool_norms_ptr + pool_norms_base, norm)


@triton.jit
def _turboquant_fused_quantize_kernel(
    # Input
    x_ptr,  # [N, dim] bf16/f32 input
    signs_ptr,  # [padded_dim] f32 random signs
    boundaries_ptr,  # [NUM_CENTROIDS - 1] f32 decision boundaries
    norms_ptr,  # [N] f32 output (L2 norms)
    packed_ptr,  # [N, packed_dim] u8 output
    # Scratch buffer for FWHT butterfly (reuses workspace.rotated)
    scratch_ptr,  # [N, padded_dim] f32
    # Direct pool write params (pool_packed_ptr == 0 means disabled)
    pool_packed_ptr,
    pool_norms_ptr,
    loc_ptr,
    pool_packed_stride_0,
    pool_packed_stride_1,
    pool_norms_stride_0,
    # Strides
    x_stride_0,
    packed_stride_0,
    scratch_stride_0,
    # Constants
    DIM: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    LOG2_DIM: tl.constexpr,
    NUM_CENTROIDS: tl.constexpr,
    HEAD_NUM: tl.constexpr,  # 0 = no pool write
    SCALE: tl.constexpr,  # 1.0 / sqrt(padded_dim)
):
    """Fused prepare + FWHT + normalize + quantize + pack in a single kernel.

    Eliminates two global memory round-trips between the prepare→hadamard→quantize
    pipeline by doing the FWHT butterfly in-kernel using scratch memory (hot in L1
    cache). For dim=128 this is 7 butterfly stages, each writing/reading 512 bytes
    that stay in L1.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, PADDED_DIM)

    # === Stage 1: Load bf16→f32, pad, compute L2 norm, apply signs ===
    in_bounds = offs < DIM
    vals = tl.load(x_ptr + row * x_stride_0 + offs, mask=in_bounds, other=0.0).to(
        tl.float32
    )

    # L2 norm (padding positions are 0, don't affect sum)
    norm = tl.sqrt(tl.sum(vals * vals, axis=0))
    tl.store(norms_ptr + row, norm)

    # Apply random signs
    signs = tl.load(signs_ptr + offs)
    vals = vals * signs

    # === Stage 2: FWHT butterfly using scratch memory (L1-resident) ===
    scratch_row = scratch_ptr + row * scratch_stride_0

    for s in tl.static_range(LOG2_DIM):
        tl.store(scratch_row + offs, vals)
        tl.debug_barrier()
        partner = offs ^ (1 << s)
        partner_vals = tl.load(scratch_row + partner)
        is_top = (offs & (1 << s)) == 0
        vals = tl.where(is_top, vals + partner_vals, partner_vals - vals)
        tl.debug_barrier()

    # Apply Hadamard scale: y = vals / sqrt(d)
    vals = vals * SCALE

    # === Stage 3: Normalize + quantize + 4-bit pack + optional scatter ===
    # Store rotated values for even/odd reindexing
    tl.store(scratch_row + offs, vals)
    tl.debug_barrier()

    inv_norm = 1.0 / tl.maximum(norm, 1e-10)

    # Load even/odd pairs for nibble packing
    pack_offs = tl.arange(0, PACKED_DIM)
    pack_mask = pack_offs < PACKED_DIM
    even_vals = (
        tl.load(scratch_row + pack_offs * 2, mask=pack_mask, other=0.0) * inv_norm
    )
    odd_vals = (
        tl.load(scratch_row + pack_offs * 2 + 1, mask=pack_mask, other=0.0) * inv_norm
    )

    # Boundary-based nearest centroid search
    best_even = tl.zeros([PACKED_DIM], dtype=tl.int32)
    best_odd = tl.zeros([PACKED_DIM], dtype=tl.int32)
    for b in tl.static_range(NUM_CENTROIDS - 1):
        boundary = tl.load(boundaries_ptr + b)
        best_even += (even_vals >= boundary).to(tl.int32)
        best_odd += (odd_vals >= boundary).to(tl.int32)

    # Pack: low nibble = even, high nibble = odd
    packed = ((best_odd << 4) | (best_even & 0x0F)).to(tl.uint8)

    # Write to workspace
    tl.store(packed_ptr + row * packed_stride_0 + pack_offs, packed, mask=pack_mask)

    # Write directly to pool buffer if enabled
    if HEAD_NUM > 0:
        token_id = row // HEAD_NUM
        head_id = row % HEAD_NUM
        pool_loc = tl.load(loc_ptr + token_id)
        pool_base = pool_loc * pool_packed_stride_0 + head_id * pool_packed_stride_1
        tl.store(pool_packed_ptr + pool_base + pack_offs, packed, mask=pack_mask)
        # Write norms (use head_id == 0 guard to avoid redundant writes from other heads;
        # actually each head has its own norm, so all heads must write)
        pool_norms_base = pool_loc * pool_norms_stride_0 + head_id
        tl.store(pool_norms_ptr + pool_norms_base, norm)


@triton.jit
def _turboquant_quantize_kernel(
    # Pointers
    rotated_ptr,  # [num_tokens, dim] float32 input (already rotated)
    indices_ptr,  # [num_tokens, padded_dim] uint8 output (unpacked)
    centroids_ptr,  # [num_centroids] float32
    # Strides
    rotated_stride_0: tl.constexpr,
    indices_stride_0: tl.constexpr,
    # Constants
    DIM: tl.constexpr,
    NUM_CENTROIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize rotated coordinates to nearest centroid indices (unpacked)."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    vals = tl.load(
        rotated_ptr + token_id * rotated_stride_0 + offs, mask=mask, other=0.0
    )

    best_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    best_dist = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)

    for c in range(NUM_CENTROIDS):
        centroid = tl.load(centroids_ptr + c)
        dist = (vals - centroid) * (vals - centroid)
        closer = dist < best_dist
        best_idx = tl.where(closer, c, best_idx)
        best_dist = tl.where(closer, dist, best_dist)

    tl.store(
        indices_ptr + token_id * indices_stride_0 + offs,
        best_idx.to(tl.uint8),
        mask=mask,
    )


@triton.jit
def _turboquant_dequantize_packed_4bit_kernel(
    packed_ptr,  # [num_tokens, packed_dim] uint8 input (nibble-packed)
    output_ptr,  # [num_tokens, padded_dim] float32 output
    centroids_ptr,  # [num_centroids] float32
    packed_stride_0: tl.constexpr,
    output_stride_0: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused unpack + centroid lookup for 4-bit packed indices."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Each element in packed buffer holds 2 indices
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < PACKED_DIM

    packed = tl.load(
        packed_ptr + token_id * packed_stride_0 + offs, mask=mask, other=0
    ).to(tl.int32)

    idx_even = packed & 0x0F
    idx_odd = (packed >> 4) & 0x0F

    val_even = tl.load(centroids_ptr + idx_even, mask=mask, other=0.0)
    val_odd = tl.load(centroids_ptr + idx_odd, mask=mask, other=0.0)

    coord_even = offs * 2
    coord_odd = offs * 2 + 1
    tl.store(
        output_ptr + token_id * output_stride_0 + coord_even,
        val_even,
        mask=mask & (coord_even < PADDED_DIM),
    )
    tl.store(
        output_ptr + token_id * output_stride_0 + coord_odd,
        val_odd,
        mask=mask & (coord_odd < PADDED_DIM),
    )


@triton.jit
def _turboquant_dequantize_kernel(
    indices_ptr,  # [num_tokens, padded_dim] uint8 input (unpacked)
    output_ptr,  # [num_tokens, padded_dim] float32 output
    centroids_ptr,  # [num_centroids] float32
    indices_stride_0: tl.constexpr,
    output_stride_0: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize unpacked indices to centroid values."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    idx = tl.load(
        indices_ptr + token_id * indices_stride_0 + offs, mask=mask, other=0
    ).to(tl.int32)
    vals = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    tl.store(output_ptr + token_id * output_stride_0 + offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def turboquant_quantize(
    x: torch.Tensor,
    hadamard: HadamardTransform,
    bits: int = 4,
    mode: str = "mse",
    workspace: Optional[QuantizeWorkspace] = None,
    pool_packed: Optional[torch.Tensor] = None,
    pool_norms: Optional[torch.Tensor] = None,
    loc: Optional[torch.Tensor] = None,
    head_num: int = 0,
) -> dict:
    """Quantize input vectors using TurboQuant with bit-packed storage.

    Args:
        x: (num_tokens, dim) input tensor (K or V cache entries)
        hadamard: HadamardTransform instance for this dimension
        bits: quantization bit-width (1-4)
        mode: "mse" for MSE-optimal, "prod" for inner-product-optimal (uses QJL)
        workspace: optional pre-allocated QuantizeWorkspace.  When provided and
                   large enough, ALL temporary allocations are eliminated —
                   critical for CUDA graph capture.
        pool_packed: optional pool packed buffer [pool_size, head_num, packed_dim]
                     for direct scatter writes from the kernel.
        pool_norms: optional pool norms buffer [pool_size, head_num] for direct
                    scatter writes from the kernel.
        loc: optional [num_tokens] int64 location indices for scatter writes.
        head_num: number of heads (>0 enables direct pool writes).

    Returns:
        dict with keys:
            - "packed_indices": (num_tokens, packed_dim) uint8, bit-packed centroid indices
            - "norms": (num_tokens,) float32, L2 norms of original vectors
            - "padded_dim": int, the padded dimension (needed for unpacking)
            - "qjl_signs": (num_tokens, packed_dim_qjl) uint8, packed QJL sign bits (mode="prod")
            - "residual_norms": (num_tokens,) float32 (mode="prod")
    """
    num_tokens, dim = x.shape
    device = x.device

    mse_bits = bits - 1 if mode == "prod" else bits
    padded_dim = hadamard.padded_dim

    # Decide whether to use the zero-allocation fused workspace path.
    # Fused path requires 4-bit MSE (the common KV cache case).
    use_ws = workspace is not None and num_tokens <= workspace.max_rows
    use_fused = use_ws and mse_bits == 4

    # Get centroids scaled by 1/sqrt(d) for the coordinate distribution
    sc_key = (mse_bits, padded_dim, device)
    scaled_centroids = _scaled_centroids_cache.get(sc_key)
    if scaled_centroids is None:
        centroids = _get_centroids_tensor(mse_bits, device)
        scaled_centroids = centroids / math.sqrt(padded_dim)
        _scaled_centroids_cache[sc_key] = scaled_centroids

    # Decision boundaries: midpoints between adjacent centroids (for boundary search)
    sb_key = (mse_bits, padded_dim, device)
    scaled_boundaries = _scaled_boundaries_cache.get(sb_key)
    if scaled_boundaries is None:
        sc = scaled_centroids
        scaled_boundaries = (sc[:-1] + sc[1:]) / 2.0
        _scaled_boundaries_cache[sb_key] = scaled_boundaries

    if use_fused:
        # === Single fused kernel: prepare + FWHT + quantize + pack ===
        # Eliminates 2 kernel launches and 2 global memory round-trips vs
        # the previous 3-kernel pipeline. FWHT butterfly runs in-kernel
        # using scratch memory that stays L1-resident.
        n = num_tokens
        norms = workspace.norms[:n]
        packed_buf = workspace.packed[:n]
        scratch = workspace.rotated[:n]  # repurposed as butterfly scratch
        packed_dim = padded_dim // 2
        log2_dim = padded_dim.bit_length() - 1

        use_pool = (
            pool_packed is not None
            and pool_norms is not None
            and loc is not None
            and head_num > 0
        )
        _turboquant_fused_quantize_kernel[(n,)](
            x,
            hadamard.signs,
            scaled_boundaries,
            norms,
            packed_buf,
            scratch,
            # Pool write params
            pool_packed if use_pool else 0,
            pool_norms if use_pool else 0,
            loc if use_pool else 0,
            pool_packed.stride(0) if use_pool else 0,
            pool_packed.stride(1) if use_pool else 0,
            pool_norms.stride(0) if use_pool else 0,
            # Strides
            x.stride(0),
            packed_buf.stride(0),
            scratch.stride(0),
            # Constants
            DIM=dim,
            PADDED_DIM=padded_dim,
            PACKED_DIM=packed_dim,
            LOG2_DIM=log2_dim,
            NUM_CENTROIDS=len(scaled_centroids),
            HEAD_NUM=head_num if use_pool else 0,
            SCALE=hadamard.scale,
        )

        packed_indices = packed_buf

    else:
        # === Legacy per-step path (dynamic alloc or non-4bit workspace) ===
        if use_ws:
            n = num_tokens
            rotated_buf = workspace.rotated[:n]
            rotated_buf[:, :dim].copy_(x)  # bf16 → float32
            if dim < padded_dim:
                rotated_buf[:, dim:].zero_()
            norms = workspace.norms[:n]
            fwht_out = workspace.fwht_out[:n]
        else:
            float_x = x.float()
            if dim < padded_dim:
                float_x = torch.nn.functional.pad(float_x, (0, padded_dim - dim))
            rotated_buf = torch.empty(
                num_tokens, padded_dim, dtype=torch.float32, device=device
            )
            norms = torch.empty(num_tokens, dtype=torch.float32, device=device)
            fwht_out = None

        # Compute L2 norms before rotation (preserved by orthogonal transform)
        if use_ws:
            torch.norm(rotated_buf[:, :dim], dim=-1, out=norms)
        else:
            torch.norm(x.float(), dim=-1, out=norms)

        # Step 1: Rotate via randomized Hadamard
        if use_ws:
            rotated = hadamard.forward(rotated_buf, fwht_out=fwht_out)
        else:
            rotated = hadamard.forward(float_x, out=rotated_buf, fwht_out=fwht_out)

        # Normalize (clamp norms in-place to avoid +1e-10 temp allocation)
        norms.clamp_min_(1e-10)
        rot_norm_buf = torch.empty(
            num_tokens, padded_dim, dtype=torch.float32, device=device
        )
        torch.div(rotated, norms.unsqueeze(-1), out=rot_norm_buf)

        # Step 2: Quantize each coordinate to nearest centroid (unpacked first)
        indices_buf = torch.empty(
            num_tokens, padded_dim, dtype=torch.uint8, device=device
        )
        indices_buf.zero_()

        BLOCK_SIZE = 128
        num_blocks = triton.cdiv(padded_dim, BLOCK_SIZE)

        _turboquant_quantize_kernel[(num_tokens, num_blocks)](
            rot_norm_buf,
            indices_buf,
            scaled_centroids,
            rot_norm_buf.stride(0),
            indices_buf.stride(0),
            DIM=padded_dim,
            NUM_CENTROIDS=len(scaled_centroids),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Step 3: Bit-pack the indices
        packed_indices = pack_indices(indices_buf, mse_bits)

    result = {
        "packed_indices": packed_indices,
        "norms": norms,
        "padded_dim": padded_dim,
    }

    # Step 4 (mode="prod" only): QJL on residual
    # Note: prod mode is uncommon and not optimized for graph capture
    if mode == "prod":
        # Reconstruct MSE approximation to compute residual
        dequant_normalized = torch.zeros_like(rot_norm_buf)
        _turboquant_dequantize_kernel[(num_tokens, num_blocks)](
            indices_buf,
            dequant_normalized,
            scaled_centroids,
            indices_buf.stride(0),
            dequant_normalized.stride(0),
            DIM=padded_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        residual = rot_norm_buf - dequant_normalized
        residual_norms = torch.norm(residual, dim=-1)

        # QJL: store sign bits of residual (1 bit per coordinate)
        qjl_signs_raw = (residual >= 0).to(torch.uint8)
        # Pack QJL signs at 1-bit
        result["qjl_signs"] = pack_indices(qjl_signs_raw, 1)
        result["residual_norms"] = residual_norms

    return result


def turboquant_dequantize(
    quantized: dict,
    hadamard: HadamardTransform,
    bits: int = 4,
    mode: str = "mse",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize TurboQuant bit-packed compressed vectors.

    Args:
        quantized: dict from turboquant_quantize()
        hadamard: same HadamardTransform used for quantization
        bits: same bit-width used for quantization
        mode: same mode used for quantization
        output_dtype: desired output dtype

    Returns:
        (num_tokens, dim) reconstructed tensor in original (unpadded) space
    """
    packed_indices = quantized["packed_indices"]
    norms = quantized["norms"]
    padded_dim = quantized["padded_dim"]
    num_tokens = packed_indices.shape[0]
    device = packed_indices.device

    mse_bits = bits - 1 if mode == "prod" else bits
    centroids = _get_centroids_tensor(mse_bits, device)
    scaled_centroids = centroids / math.sqrt(padded_dim)

    packed_dim = packed_indices.shape[-1]
    dequant = torch.zeros(num_tokens, padded_dim, dtype=torch.float32, device=device)

    # Use fused Triton kernel for 4-bit (the common case), PyTorch unpack for others
    if mse_bits == 4:
        BLOCK_SIZE = 128
        num_blocks = triton.cdiv(packed_dim, BLOCK_SIZE)
        _turboquant_dequantize_packed_4bit_kernel[(num_tokens, num_blocks)](
            packed_indices,
            dequant,
            scaled_centroids,
            packed_indices.stride(0),
            dequant.stride(0),
            PADDED_DIM=padded_dim,
            PACKED_DIM=packed_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Unpack then dequant via existing kernel
        indices = unpack_indices(packed_indices, mse_bits, padded_dim)
        BLOCK_SIZE = 128
        num_blocks = triton.cdiv(padded_dim, BLOCK_SIZE)
        _turboquant_dequantize_kernel[(num_tokens, num_blocks)](
            indices,
            dequant,
            scaled_centroids,
            indices.stride(0),
            dequant.stride(0),
            DIM=padded_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Add QJL correction if mode="prod"
    if mode == "prod" and "qjl_signs" in quantized:
        qjl_packed = quantized["qjl_signs"]
        residual_norms = quantized["residual_norms"]
        # Unpack 1-bit QJL signs and convert to +1/-1
        qjl_unpacked = unpack_indices(qjl_packed, 1, padded_dim).float()
        qjl_signs = qjl_unpacked * 2.0 - 1.0  # 0/1 -> -1/+1
        qjl_scale = math.sqrt(math.pi / 2) / padded_dim
        dequant += qjl_scale * residual_norms.unsqueeze(-1) * qjl_signs

    # Rescale by original norm
    dequant = dequant * norms.unsqueeze(-1)

    # Inverse Hadamard to get back to original space
    reconstructed = hadamard.inverse(dequant)

    return reconstructed.to(output_dtype)


# ---------------------------------------------------------------------------
# Mixed-precision quantization (paper's 2.5-bit and 3.5-bit configs)
#
# Paper: "splitting channels into outlier and non-outlier sets, and applying
# two independent instances of TurboQuant to each, allocating higher bit
# precision to outliers."
#
# Implementation: split raw channels BEFORE rotation into two groups, each
# getting its own independent Hadamard rotation and quantization.  The
# split is a fixed 50/50 by default.  A per-layer outlier-aware split can
# be configured by passing channel indices (see `outlier_indices` param).
# ---------------------------------------------------------------------------

# Allowed effective bit-widths and their (high, low) decomposition
MIXED_PRECISION_CONFIGS = {
    2.5: (3, 2),  # split_dim coords @ 3-bit + rest @ 2-bit
    3.5: (4, 3),  # split_dim coords @ 4-bit + rest @ 3-bit
}


def parse_bits(bits) -> tuple:
    """Parse bit-width spec into (is_mixed, bits_hi, bits_lo).

    Args:
        bits: int (1-4) for uniform, or float (2.5, 3.5) for mixed-precision
    Returns:
        (is_mixed, bits_hi, bits_lo)
    """
    if isinstance(bits, float) and bits in MIXED_PRECISION_CONFIGS:
        hi, lo = MIXED_PRECISION_CONFIGS[bits]
        return (True, hi, lo)
    bits = int(bits)
    return (False, bits, bits)


def compute_packed_dim_mixed(head_dim: int, bits) -> int:
    """Compute total packed byte size for uniform or mixed-precision.

    For mixed-precision, each channel group is independently padded to
    a power of 2 and packed at its own bit-width.  The returned size
    includes both groups' packed indices but NOT norms (those are
    accounted for separately in memory accounting).
    """
    is_mixed, bits_hi, bits_lo = parse_bits(bits)
    if not is_mixed:
        padded = _next_power_of_2(head_dim)
        return compute_packed_dim(padded, bits_hi)
    split = head_dim // 2
    hi_padded = _next_power_of_2(split)
    lo_padded = _next_power_of_2(head_dim - split)
    return compute_packed_dim(hi_padded, bits_hi) + compute_packed_dim(
        lo_padded, bits_lo
    )


def turboquant_quantize_mixed(
    x: torch.Tensor,
    hadamard_hi: HadamardTransform,
    hadamard_lo: HadamardTransform,
    bits_hi: int,
    bits_lo: int,
    split_dim: Optional[int] = None,
) -> dict:
    """Mixed-precision quantization with two independent TurboQuant instances.

    Splits raw channels BEFORE rotation.  Each group gets its own
    Hadamard rotation and quantization at a different bit-width.

    Args:
        x: (num_tokens, dim) input tensor
        hadamard_hi: HadamardTransform for the first (outlier) channel group
        hadamard_lo: HadamardTransform for the second channel group
        bits_hi: bit-width for the first group
        bits_lo: bit-width for the second group
        split_dim: number of channels in the first group (default: dim // 2)

    Returns dict with:
        - "packed_hi": packed indices for the high-bit group
        - "packed_lo": packed indices for the low-bit group
        - "norms_hi": float32 L2 norms of the high-bit group
        - "norms_lo": float32 L2 norms of the low-bit group
        - "padded_dim_hi", "padded_dim_lo": padded dims per group
        - "split_dim": channel split point
        - "bits_hi", "bits_lo": for dequantization
    """
    num_tokens, dim = x.shape
    device = x.device
    if split_dim is None:
        split_dim = dim // 2

    # Split raw channels BEFORE rotation
    x_hi = x[:, :split_dim]
    x_lo = x[:, split_dim:]

    # Independent TurboQuant instance for each group
    q_hi = turboquant_quantize(x_hi, hadamard_hi, bits_hi, mode="mse")
    q_lo = turboquant_quantize(x_lo, hadamard_lo, bits_lo, mode="mse")

    return {
        "packed_hi": q_hi["packed_indices"],
        "packed_lo": q_lo["packed_indices"],
        "norms_hi": q_hi["norms"],
        "norms_lo": q_lo["norms"],
        "padded_dim_hi": q_hi["padded_dim"],
        "padded_dim_lo": q_lo["padded_dim"],
        "split_dim": split_dim,
        "bits_hi": bits_hi,
        "bits_lo": bits_lo,
    }


def turboquant_dequantize_mixed(
    quantized: dict,
    hadamard_hi: HadamardTransform,
    hadamard_lo: HadamardTransform,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize mixed-precision data from two independent TurboQuant instances."""
    bits_hi = quantized["bits_hi"]
    bits_lo = quantized["bits_lo"]
    split_dim = quantized["split_dim"]

    # Reconstruct each group independently
    q_hi = {
        "packed_indices": quantized["packed_hi"],
        "norms": quantized["norms_hi"],
        "padded_dim": quantized["padded_dim_hi"],
    }
    q_lo = {
        "packed_indices": quantized["packed_lo"],
        "norms": quantized["norms_lo"],
        "padded_dim": quantized["padded_dim_lo"],
    }

    recon_hi = turboquant_dequantize(q_hi, hadamard_hi, bits_hi, "mse", output_dtype)
    recon_lo = turboquant_dequantize(q_lo, hadamard_lo, bits_lo, "mse", output_dtype)

    # turboquant_dequantize already trims to hadamard.dim (the original
    # unpadded dimension for each group), so just concatenate.
    return torch.cat([recon_hi, recon_lo], dim=-1)


# ---------------------------------------------------------------------------
# KV Cache specific quantize/dequantize wrappers
# ---------------------------------------------------------------------------


def turboquant_quantize_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_hadamard: HadamardTransform,
    v_hadamard: HadamardTransform,
    bits: int = 4,
    mode: str = "mse",
) -> Tuple[dict, dict]:
    """Quantize both K and V cache entries.

    Args:
        k: (num_tokens, num_heads, head_dim) key tensor
        v: (num_tokens, num_heads, head_dim) value tensor
        k_hadamard: HadamardTransform for key dimension
        v_hadamard: HadamardTransform for value dimension
        bits: quantization bits
        mode: "mse" or "prod"

    Returns:
        (k_quantized, v_quantized) dicts
    """
    num_tokens, num_heads, head_dim = k.shape

    # Reshape to (num_tokens * num_heads, head_dim) for quantization
    k_flat = k.reshape(-1, head_dim)
    v_flat = v.reshape(-1, head_dim)

    k_q = turboquant_quantize(k_flat, k_hadamard, bits, mode)
    v_q = turboquant_quantize(v_flat, v_hadamard, bits, mode)

    return k_q, v_q


def turboquant_dequantize_kv_cache(
    k_quantized: dict,
    v_quantized: dict,
    k_hadamard: HadamardTransform,
    v_hadamard: HadamardTransform,
    num_heads: int,
    bits: int = 4,
    mode: str = "mse",
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize both K and V cache entries.

    Returns:
        (k, v) tensors of shape (num_tokens, num_heads, head_dim)
    """
    k_recon = turboquant_dequantize(k_quantized, k_hadamard, bits, mode, output_dtype)
    v_recon = turboquant_dequantize(v_quantized, v_hadamard, bits, mode, output_dtype)

    total = k_recon.shape[0]
    num_tokens = total // num_heads
    head_dim = k_hadamard.dim

    k_recon = k_recon[:, :head_dim].reshape(num_tokens, num_heads, head_dim)
    v_recon = v_recon[:, :head_dim].reshape(num_tokens, num_heads, head_dim)

    return k_recon, v_recon


# ---------------------------------------------------------------------------
# Compression ratio calculation
# ---------------------------------------------------------------------------


def compute_compression_ratio(
    head_dim: int, bits, mode: str = "mse", dtype_bytes: int = 2
) -> float:
    """Compute the theoretical compression ratio vs baseline dtype.

    Args:
        head_dim: original head dimension
        bits: quantization bits (1-4 int, or 2.5/3.5 for mixed-precision)
        mode: "mse" or "prod"
        dtype_bytes: bytes per element for baseline (2 for bf16/fp16)
    Returns:
        compression ratio (e.g., 3.77 means 3.77x smaller)
    """
    padded_dim = _next_power_of_2(head_dim)
    is_mixed, bits_hi, bits_lo = parse_bits(bits)

    if is_mixed:
        index_bytes = compute_packed_dim_mixed(padded_dim, bits)
    else:
        mse_bits = bits_hi - 1 if mode == "prod" else bits_hi
        index_bytes = compute_packed_dim(padded_dim, mse_bits)

    # Norm: 1 float32 per token-head
    norm_bytes = 4
    # QJL for prod mode: 1 bit per coord (packed) + 1 float32 residual norm
    qjl_bytes = 0
    if mode == "prod" and not is_mixed:
        qjl_bytes = compute_packed_dim(padded_dim, 1) + 4

    tq_bytes_per_head = index_bytes + norm_bytes + qjl_bytes
    baseline_bytes_per_head = head_dim * dtype_bytes
    return baseline_bytes_per_head / tq_bytes_per_head
