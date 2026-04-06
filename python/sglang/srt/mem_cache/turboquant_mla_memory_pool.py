"""
TurboQuant memory pool for MLA (Multi-head Latent Attention) KV cache compression.

Extends TurboQuant (ICLR 2026) to DeepSeek-V3 family models which use MLA
instead of standard MHA.  MLA stores a single latent vector per token
consisting of k_nope (kv_lora_rank=512) and k_rope (qk_rope_head_dim=64).

Strategy: Quantize k_nope and k_rope separately with independent Hadamard
transforms.  Both dims are power-of-2 so there is zero padding waste.
"""

import logging
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.quantization.turboquant_kernels import (
    HadamardTransform,
    _get_centroids_tensor,
    _next_power_of_2,
    compute_packed_dim,
    compute_packed_dim_mixed,
    parse_bits,
    turboquant_dequantize,
    turboquant_dequantize_mixed,
    turboquant_quantize,
    turboquant_quantize_mixed,
)
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

# Target peak memory for float32 intermediates during chunked dequantization.
_DEQUANT_CHUNK_MEMORY_BUDGET = 256 * 1024 * 1024  # 256 MB

# Deterministic seeds for the randomized Hadamard rotation.
# Different seeds for nope vs rope (and hi vs lo in mixed-precision).
_HADAMARD_SEED_NOPE = 42
_HADAMARD_SEED_NOPE_LO = 43
_HADAMARD_SEED_ROPE = 137
_HADAMARD_SEED_ROPE_LO = 138


class MLATokenToKVPoolTurboQuant(MLATokenToKVPool):
    """Memory pool that stores MLA KV cache compressed via TurboQuant.

    Storage per token per layer:
      - Bit-packed centroid indices for k_nope (uint8)
      - Bit-packed centroid indices for k_rope (uint8)
      - L2 norms: 1 per component (uniform) or 2 per component (mixed)

    A shared workspace buffer of shape (max_tokens, 1, kv_cache_dim) in the
    working dtype is pre-allocated and reused across layers.

    On set_mla_kv_buffer: quantize nope/rope independently, store compressed.
    On get_key_buffer: dequantize both into workspace, return full kv_cache_dim.
    On get_value_buffer: dequantize nope only, return kv_lora_rank slice.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        bits: float = 4,
        mode: str = "mse",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
    ):
        self._tq_use_nsa = use_nsa
        self.bits = bits
        self.mode = mode
        self.is_mixed, self.bits_hi, self.bits_lo = parse_bits(bits)
        self.mse_bits = (
            int(bits) - 1 if mode == "prod" and not self.is_mixed else int(bits)
        )

        # Cache padded dimensions (both are power-of-2 for DeepSeek, so no waste)
        self.nope_padded_dim = _next_power_of_2(kv_lora_rank)
        self.rope_padded_dim = _next_power_of_2(qk_rope_head_dim)

        # Initialize Hadamard transforms
        torch_device = torch.device(device)
        if self.is_mixed:
            nope_split = kv_lora_rank // 2
            rope_split = qk_rope_head_dim // 2
            self.nope_hadamard_hi = HadamardTransform(
                nope_split, seed=_HADAMARD_SEED_NOPE, device=torch_device
            )
            self.nope_hadamard_lo = HadamardTransform(
                kv_lora_rank - nope_split,
                seed=_HADAMARD_SEED_NOPE_LO,
                device=torch_device,
            )
            self.rope_hadamard_hi = HadamardTransform(
                rope_split, seed=_HADAMARD_SEED_ROPE, device=torch_device
            )
            self.rope_hadamard_lo = HadamardTransform(
                qk_rope_head_dim - rope_split,
                seed=_HADAMARD_SEED_ROPE_LO,
                device=torch_device,
            )
            self._nope_split_dim = nope_split
            self._rope_split_dim = rope_split
            # Compatibility placeholders
            self.nope_hadamard = self.nope_hadamard_hi
            self.rope_hadamard = self.rope_hadamard_hi
        else:
            self.nope_hadamard = HadamardTransform(
                kv_lora_rank, seed=_HADAMARD_SEED_NOPE, device=torch_device
            )
            self.rope_hadamard = HadamardTransform(
                qk_rope_head_dim, seed=_HADAMARD_SEED_ROPE, device=torch_device
            )

        # Compute chunk size for dequantization based on memory budget.
        # MLA has 1 head, so bytes_per_token = 1 * max_padded * 4
        max_padded = max(self.nope_padded_dim, self.rope_padded_dim)
        bytes_per_token = 1 * max_padded * 4  # float32, 1 head
        self._dequant_chunk_tokens = max(
            1, _DEQUANT_CHUNK_MEMORY_BUDGET // bytes_per_token
        )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            use_nsa=use_nsa,
        )

        # Pre-scaled centroid tables for the fused decode kernel.
        # The centroids are scaled by 1/sqrt(dim) so the kernel only needs to
        # multiply by the per-token norm (no additional 1/sqrt(d) factor).
        if self.can_use_fused_kernel:
            raw_centroids = _get_centroids_tensor(self.mse_bits, torch_device)
            self.nope_centroids_scaled = raw_centroids / math.sqrt(self.nope_padded_dim)
            self.rope_centroids_scaled = raw_centroids / math.sqrt(self.rope_padded_dim)
        else:
            self.nope_centroids_scaled = None
            self.rope_centroids_scaled = None

    def _create_buffers(self):
        """Allocate bit-packed compressed storage buffers + shared workspace."""
        self.store_dtype = torch.uint8

        m = self.size + self.page_size
        # For "prod" mode, the MSE stage uses bits-1 (remaining bit goes to QJL).
        # Buffer allocation must match what turboquant_quantize actually packs.
        alloc_bits = (
            self.mse_bits if (self.mode == "prod" and not self.is_mixed) else self.bits
        )
        nope_packed_dim = compute_packed_dim_mixed(self.kv_lora_rank, alloc_bits)
        rope_packed_dim = compute_packed_dim_mixed(self.qk_rope_head_dim, alloc_bits)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Bit-packed centroid indices — per layer, shape (m, 1, packed_dim)
                self.nope_packed_buffer = [
                    torch.zeros(
                        (m, 1, nope_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.rope_packed_buffer = [
                    torch.zeros(
                        (m, 1, rope_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # L2 norms — per layer.
                # Uniform: (m, 1) per component.  Mixed: (m, 1, 2) per component.
                norm_shape = (m, 1, 2) if self.is_mixed else (m, 1)
                self.nope_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.rope_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]

                # QJL sign bits — only for "prod" mode
                if self.mode == "prod":
                    nope_qjl_dim = compute_packed_dim(self.nope_padded_dim, 1)
                    rope_qjl_dim = compute_packed_dim(self.rope_padded_dim, 1)
                    self.nope_qjl_buffer = [
                        torch.zeros(
                            (m, 1, nope_qjl_dim),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.rope_qjl_buffer = [
                        torch.zeros(
                            (m, 1, rope_qjl_dim),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.nope_residual_norms_buffer = [
                        torch.zeros((m, 1), dtype=torch.float32, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.rope_residual_norms_buffer = [
                        torch.zeros((m, 1), dtype=torch.float32, device=self.device)
                        for _ in range(self.layer_num)
                    ]

                # Shared workspace for dequantized data — reused across layers.
                # Full KV workspace: (m, 1, kv_lora_rank + qk_rope_head_dim)
                self._kv_workspace = torch.zeros(
                    (m, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )

        # Override kv_buffer to an empty list so base class code that
        # iterates over self.kv_buffer (e.g. data_ptrs) doesn't break.
        # We don't use it; all storage is in packed buffers.
        self.kv_buffer = []

    def _clear_buffers(self):
        del self.nope_packed_buffer
        del self.rope_packed_buffer
        del self.nope_norms_buffer
        del self.rope_norms_buffer
        del self._kv_workspace
        if self.mode == "prod":
            del self.nope_qjl_buffer
            del self.rope_qjl_buffer
            del self.nope_residual_norms_buffer
            del self.rope_residual_norms_buffer

    def get_kv_size_bytes(self):
        size = 0
        size += sum(get_tensor_size_bytes(b) for b in self.nope_packed_buffer)
        size += sum(get_tensor_size_bytes(b) for b in self.rope_packed_buffer)
        size += sum(get_tensor_size_bytes(b) for b in self.nope_norms_buffer)
        size += sum(get_tensor_size_bytes(b) for b in self.rope_norms_buffer)
        size += get_tensor_size_bytes(self._kv_workspace)
        if self.mode == "prod":
            size += sum(get_tensor_size_bytes(b) for b in self.nope_qjl_buffer)
            size += sum(get_tensor_size_bytes(b) for b in self.rope_qjl_buffer)
            size += sum(
                get_tensor_size_bytes(b) for b in self.nope_residual_norms_buffer
            )
            size += sum(
                get_tensor_size_bytes(b) for b in self.rope_residual_norms_buffer
            )
        return size

    def _dequant_component_chunked(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        workspace: torch.Tensor,
        hadamard,
        padded_dim: int,
        out_dim: int,
        qjl_buf: Optional[torch.Tensor] = None,
        residual_norms_buf: Optional[torch.Tensor] = None,
        hadamard_hi: Optional[HadamardTransform] = None,
        hadamard_lo: Optional[HadamardTransform] = None,
        split_dim: int = 0,
    ):
        """Dequantize one component (nope or rope) of one layer in chunks."""
        total_tokens = packed.shape[0]
        num_heads = packed.shape[1]  # always 1 for MLA
        chunk = self._dequant_chunk_tokens

        for start in range(0, total_tokens, chunk):
            end = min(start + chunk, total_tokens)
            c_packed = packed[start:end]
            c_norms = norms[start:end]
            n_chunk = end - start

            if self.is_mixed:
                hi_packed_dim = compute_packed_dim(
                    _next_power_of_2(split_dim), self.bits_hi
                )
                flat_packed = c_packed.reshape(-1, c_packed.shape[-1])
                flat_norms_hi = c_norms[..., 0].reshape(-1)
                flat_norms_lo = c_norms[..., 1].reshape(-1)

                quantized = {
                    "packed_hi": flat_packed[:, :hi_packed_dim],
                    "packed_lo": flat_packed[:, hi_packed_dim:],
                    "norms_hi": flat_norms_hi,
                    "norms_lo": flat_norms_lo,
                    "padded_dim_hi": _next_power_of_2(split_dim),
                    "padded_dim_lo": _next_power_of_2(out_dim - split_dim),
                    "split_dim": split_dim,
                    "bits_hi": self.bits_hi,
                    "bits_lo": self.bits_lo,
                }
                result = turboquant_dequantize_mixed(
                    quantized, hadamard_hi, hadamard_lo, self.dtype
                )
                workspace[start:end] = result[:, :out_dim].reshape(
                    n_chunk, num_heads, out_dim
                )
            else:
                quantized = {
                    "packed_indices": c_packed.reshape(-1, c_packed.shape[-1]),
                    "norms": c_norms.reshape(-1),
                    "padded_dim": padded_dim,
                }
                if self.mode == "prod" and qjl_buf is not None:
                    c_qjl = qjl_buf[start:end]
                    quantized["qjl_signs"] = c_qjl.reshape(-1, c_qjl.shape[-1])
                    quantized["residual_norms"] = residual_norms_buf[start:end].reshape(
                        -1
                    )
                result = turboquant_dequantize(
                    quantized, hadamard, int(self.bits), self.mode, self.dtype
                )
                workspace[start:end] = result[:, :out_dim].reshape(
                    n_chunk, num_heads, out_dim
                )

    def _dequant_full_key(self, layer_id: int):
        """Dequantize nope + rope into workspace for a full key buffer."""
        idx = layer_id - self.start_layer

        # Dequant nope into workspace[:, :, :kv_lora_rank]
        nope_ws = self._kv_workspace[:, :, : self.kv_lora_rank]
        nope_qjl = self.nope_qjl_buffer[idx] if self.mode == "prod" else None
        nope_res = self.nope_residual_norms_buffer[idx] if self.mode == "prod" else None
        self._dequant_component_chunked(
            self.nope_packed_buffer[idx],
            self.nope_norms_buffer[idx],
            nope_ws,
            self.nope_hadamard,
            self.nope_padded_dim,
            self.kv_lora_rank,
            nope_qjl,
            nope_res,
            hadamard_hi=getattr(self, "nope_hadamard_hi", None),
            hadamard_lo=getattr(self, "nope_hadamard_lo", None),
            split_dim=getattr(self, "_nope_split_dim", 0),
        )

        # Dequant rope into workspace[:, :, kv_lora_rank:]
        rope_ws = self._kv_workspace[:, :, self.kv_lora_rank :]
        rope_qjl = self.rope_qjl_buffer[idx] if self.mode == "prod" else None
        rope_res = self.rope_residual_norms_buffer[idx] if self.mode == "prod" else None
        self._dequant_component_chunked(
            self.rope_packed_buffer[idx],
            self.rope_norms_buffer[idx],
            rope_ws,
            self.rope_hadamard,
            self.rope_padded_dim,
            self.qk_rope_head_dim,
            rope_qjl,
            rope_res,
            hadamard_hi=getattr(self, "rope_hadamard_hi", None),
            hadamard_lo=getattr(self, "rope_hadamard_lo", None),
            split_dim=getattr(self, "_rope_split_dim", 0),
        )

    def get_key_buffer(self, layer_id: int):
        """Dequantize and return full key buffer (nope + rope) for a layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._dequant_full_key(layer_id)
        return self._kv_workspace

    def get_value_buffer(self, layer_id: int):
        """Dequantize and return value buffer (nope only) for a layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        # Value is just the nope portion
        idx = layer_id - self.start_layer
        nope_ws = self._kv_workspace[:, :, : self.kv_lora_rank]
        nope_qjl = self.nope_qjl_buffer[idx] if self.mode == "prod" else None
        nope_res = self.nope_residual_norms_buffer[idx] if self.mode == "prod" else None
        self._dequant_component_chunked(
            self.nope_packed_buffer[idx],
            self.nope_norms_buffer[idx],
            nope_ws,
            self.nope_hadamard,
            self.nope_padded_dim,
            self.kv_lora_rank,
            nope_qjl,
            nope_res,
            hadamard_hi=getattr(self, "nope_hadamard_hi", None),
            hadamard_lo=getattr(self, "nope_hadamard_lo", None),
            split_dim=getattr(self, "_nope_split_dim", 0),
        )
        return nope_ws

    def get_kv_buffer(self, layer_id: int, **kwargs):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    @property
    def can_use_fused_kernel(self):
        """True iff the fused dequant-attention Triton kernel can be used."""
        return self.mode == "mse" and self.mse_bits == 4 and not self.is_mixed

    def get_nope_packed_buffer(self, layer_id: int):
        """Raw uint8 packed nope indices for a layer (no dequant)."""
        return self.nope_packed_buffer[layer_id - self.start_layer]

    def get_rope_packed_buffer(self, layer_id: int):
        """Raw uint8 packed rope indices for a layer (no dequant)."""
        return self.rope_packed_buffer[layer_id - self.start_layer]

    def get_nope_norms_buffer(self, layer_id: int):
        """Float32 nope norms for a layer."""
        return self.nope_norms_buffer[layer_id - self.start_layer]

    def get_rope_norms_buffer(self, layer_id: int):
        """Float32 rope norms for a layer."""
        return self.rope_norms_buffer[layer_id - self.start_layer]

    def _quantize_component(
        self, data_flat, hadamard, hadamard_hi=None, hadamard_lo=None, split_dim=0
    ):
        """Quantize a flat tensor using TurboQuant."""
        if self.is_mixed:
            return turboquant_quantize_mixed(
                data_flat,
                hadamard_hi,
                hadamard_lo,
                self.bits_hi,
                self.bits_lo,
                split_dim,
            )
        else:
            return turboquant_quantize(data_flat, hadamard, int(self.bits), self.mode)

    def _store_quantized(
        self,
        q_result,
        packed_buf,
        norms_buf,
        num_tokens,
        qjl_buf=None,
        residual_norms_buf=None,
    ):
        """Store quantized result into the appropriate buffers at loc."""
        if self.is_mixed:
            packed = torch.cat([q_result["packed_hi"], q_result["packed_lo"]], dim=-1)
            norms_stacked = torch.stack(
                [q_result["norms_hi"], q_result["norms_lo"]], dim=-1
            ).reshape(num_tokens, 1, 2)
            return packed.reshape(num_tokens, 1, -1), norms_stacked
        else:
            packed = q_result["packed_indices"].reshape(num_tokens, 1, -1)
            norms = q_result["norms"].reshape(num_tokens, 1)
            return packed, norms

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """Quantize and store concatenated KV cache (k = [nope | rope])."""
        layer_id = layer.layer_id
        idx = layer_id - self.start_layer
        num_tokens = cache_k.shape[0]

        # Split into nope and rope
        cache_k_nope = cache_k[:, :, : self.kv_lora_rank].reshape(-1, self.kv_lora_rank)
        cache_k_rope = cache_k[:, :, self.kv_lora_rank :].reshape(
            -1, self.qk_rope_head_dim
        )

        self._quantize_and_store(idx, loc, cache_k_nope, cache_k_rope, num_tokens)

    def set_mla_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """Quantize and store separate nope/rope KV cache entries."""
        layer_id = layer.layer_id
        idx = layer_id - self.start_layer
        num_tokens = cache_k_nope.shape[0]

        nope_flat = cache_k_nope.reshape(-1, self.kv_lora_rank)
        rope_flat = cache_k_rope.reshape(-1, self.qk_rope_head_dim)

        self._quantize_and_store(idx, loc, nope_flat, rope_flat, num_tokens)

    def _quantize_and_store(self, idx, loc, nope_flat, rope_flat, num_tokens):
        """Core quantization + storage for both set_kv_buffer and set_mla_kv_buffer."""
        # Quantize nope
        nope_q = self._quantize_component(
            nope_flat,
            self.nope_hadamard,
            hadamard_hi=getattr(self, "nope_hadamard_hi", None),
            hadamard_lo=getattr(self, "nope_hadamard_lo", None),
            split_dim=getattr(self, "_nope_split_dim", 0),
        )

        # Quantize rope
        rope_q = self._quantize_component(
            rope_flat,
            self.rope_hadamard,
            hadamard_hi=getattr(self, "rope_hadamard_hi", None),
            hadamard_lo=getattr(self, "rope_hadamard_lo", None),
            split_dim=getattr(self, "_rope_split_dim", 0),
        )

        # Store nope
        if self.is_mixed:
            packed_nope = torch.cat([nope_q["packed_hi"], nope_q["packed_lo"]], dim=-1)
            self.nope_packed_buffer[idx][loc] = packed_nope.reshape(num_tokens, 1, -1)
            nope_norms = torch.stack(
                [nope_q["norms_hi"], nope_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, 1, 2)
            self.nope_norms_buffer[idx][loc] = nope_norms
        else:
            self.nope_packed_buffer[idx][loc] = nope_q["packed_indices"].reshape(
                num_tokens, 1, -1
            )
            self.nope_norms_buffer[idx][loc] = nope_q["norms"].reshape(num_tokens, 1)

        # Store rope
        if self.is_mixed:
            packed_rope = torch.cat([rope_q["packed_hi"], rope_q["packed_lo"]], dim=-1)
            self.rope_packed_buffer[idx][loc] = packed_rope.reshape(num_tokens, 1, -1)
            rope_norms = torch.stack(
                [rope_q["norms_hi"], rope_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, 1, 2)
            self.rope_norms_buffer[idx][loc] = rope_norms
        else:
            self.rope_packed_buffer[idx][loc] = rope_q["packed_indices"].reshape(
                num_tokens, 1, -1
            )
            self.rope_norms_buffer[idx][loc] = rope_q["norms"].reshape(num_tokens, 1)

        # Store prod-mode extras
        if not self.is_mixed and self.mode == "prod":
            self.nope_qjl_buffer[idx][loc] = nope_q["qjl_signs"].reshape(
                num_tokens, 1, -1
            )
            self.rope_qjl_buffer[idx][loc] = rope_q["qjl_signs"].reshape(
                num_tokens, 1, -1
            )
            self.nope_residual_norms_buffer[idx][loc] = nope_q[
                "residual_norms"
            ].reshape(num_tokens, 1)
            self.rope_residual_norms_buffer[idx][loc] = rope_q[
                "residual_norms"
            ].reshape(num_tokens, 1)

    def get_mla_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        """Dequantize at specific locations, return (nope, rope) tuple."""
        layer_id = layer.layer_id
        # Dequant full key first, then gather at locations
        self._dequant_full_key(layer_id)
        dst_dtype = dst_dtype or self.dtype
        kv_buf = self._kv_workspace
        cache_k_nope = kv_buf[loc, :, : self.kv_lora_rank].to(dst_dtype)
        cache_k_rope = kv_buf[loc, :, self.kv_lora_rank :].to(dst_dtype)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Copy KV cache entries between locations."""
        if tgt_loc.numel() == 0:
            return
        for i in range(self.layer_num):
            self.nope_packed_buffer[i][tgt_loc] = self.nope_packed_buffer[i][src_loc]
            self.rope_packed_buffer[i][tgt_loc] = self.rope_packed_buffer[i][src_loc]
            self.nope_norms_buffer[i][tgt_loc] = self.nope_norms_buffer[i][src_loc]
            self.rope_norms_buffer[i][tgt_loc] = self.rope_norms_buffer[i][src_loc]
            if self.mode == "prod":
                self.nope_qjl_buffer[i][tgt_loc] = self.nope_qjl_buffer[i][src_loc]
                self.rope_qjl_buffer[i][tgt_loc] = self.rope_qjl_buffer[i][src_loc]
                self.nope_residual_norms_buffer[i][tgt_loc] = (
                    self.nope_residual_norms_buffer[i][src_loc]
                )
                self.rope_residual_norms_buffer[i][tgt_loc] = (
                    self.rope_residual_norms_buffer[i][src_loc]
                )

    def get_contiguous_buf_infos(self):
        # TurboQuant MLA doesn't use contiguous kv_buffer layout
        raise NotImplementedError(
            "get_contiguous_buf_infos not supported for TurboQuant MLA"
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offloading not yet supported for TurboQuant MLA")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offloading not yet supported for TurboQuant MLA")
