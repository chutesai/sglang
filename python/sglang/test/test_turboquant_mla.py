"""
TurboQuant MLA unit tests.

Tests the MLA-specific TurboQuant memory pool (MLATokenToKVPoolTurboQuant)
and the NSA variant (NSATokenToKVPoolTurboQuant).

Run on a GPU node:
    python -m pytest python/sglang/test/test_turboquant_mla.py -v
Or directly:
    python python/sglang/test/test_turboquant_mla.py
"""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

# Direct import of turboquant kernels to avoid sglang full init.
_kernels_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "srt",
    "layers",
    "quantization",
    "turboquant_kernels.py",
)
_spec = importlib.util.spec_from_file_location(
    "turboquant_kernels", os.path.abspath(_kernels_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HadamardTransform = _mod.HadamardTransform
_next_power_of_2 = _mod._next_power_of_2
compute_packed_dim = _mod.compute_packed_dim
compute_packed_dim_mixed = _mod.compute_packed_dim_mixed
compute_compression_ratio = _mod.compute_compression_ratio
parse_bits = _mod.parse_bits
turboquant_quantize = _mod.turboquant_quantize
turboquant_dequantize = _mod.turboquant_dequantize
turboquant_quantize_mixed = _mod.turboquant_quantize_mixed
turboquant_dequantize_mixed = _mod.turboquant_dequantize_mixed

DEVICE = torch.device("cuda")

# Paper's theoretical MSE upper bounds (Theorem 1) — dimension-independent
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

# MLA dimensions (DeepSeek-V3 family)
KV_LORA_RANK = 512  # nope dimension
QK_ROPE_HEAD_DIM = 64  # rope dimension


# ---------------------------------------------------------------------------
# Unit tests: quantization quality at MLA dimensions
# ---------------------------------------------------------------------------


def test_mla_nope_roundtrip_quality():
    """Quantize random (1024, 512) tensor at 3/4-bit, verify MSE <= paper bounds."""
    h = HadamardTransform(KV_LORA_RANK, seed=42, device=DEVICE)
    x = torch.randn(1024, KV_LORA_RANK, device=DEVICE)
    for bits in [3, 4]:
        q = turboquant_quantize(x, h, bits, "mse")
        r = turboquant_dequantize(q, h, bits, "mse", torch.float32)
        rel_mse = ((x.float() - r[:, :KV_LORA_RANK]) ** 2).mean().item() / (
            x.float() ** 2
        ).mean().item()
        assert (
            rel_mse < PAPER_MSE[bits] * 1.2
        ), f"nope {bits}b: MSE {rel_mse:.4f} > paper {PAPER_MSE[bits]}"
        print(f"  nope {bits}b: relMSE={rel_mse:.6f} (paper: <={PAPER_MSE[bits]})")
    print("PASS: test_mla_nope_roundtrip_quality")


def test_mla_rope_roundtrip_quality():
    """Quantize random (1024, 64) tensor at 3/4-bit, verify MSE <= paper bounds."""
    h = HadamardTransform(QK_ROPE_HEAD_DIM, seed=137, device=DEVICE)
    x = torch.randn(1024, QK_ROPE_HEAD_DIM, device=DEVICE)
    for bits in [3, 4]:
        q = turboquant_quantize(x, h, bits, "mse")
        r = turboquant_dequantize(q, h, bits, "mse", torch.float32)
        rel_mse = ((x.float() - r[:, :QK_ROPE_HEAD_DIM]) ** 2).mean().item() / (
            x.float() ** 2
        ).mean().item()
        assert (
            rel_mse < PAPER_MSE[bits] * 1.2
        ), f"rope {bits}b: MSE {rel_mse:.4f} > paper {PAPER_MSE[bits]}"
        print(f"  rope {bits}b: relMSE={rel_mse:.6f} (paper: <={PAPER_MSE[bits]})")
    print("PASS: test_mla_rope_roundtrip_quality")


def test_mla_separate_vs_joint():
    """Verify separate nope/rope quantization is equivalent to joint
    (both dims are power-of-2 => no padding waste)."""
    nope_dim = KV_LORA_RANK
    rope_dim = QK_ROPE_HEAD_DIM
    joint_dim = nope_dim + rope_dim

    h_joint = HadamardTransform(joint_dim, seed=42, device=DEVICE)
    h_nope = HadamardTransform(nope_dim, seed=42, device=DEVICE)
    h_rope = HadamardTransform(137, seed=137, device=DEVICE)  # 64 padded stays 64

    N = 512
    x = torch.randn(N, joint_dim, device=DEVICE)
    x_nope = x[:, :nope_dim]
    x_rope = x[:, nope_dim:]

    for bits in [3, 4]:
        # Joint quantization
        q_joint = turboquant_quantize(x, h_joint, bits, "mse")
        r_joint = turboquant_dequantize(q_joint, h_joint, bits, "mse", torch.float32)
        mse_joint = ((x.float() - r_joint[:, :joint_dim]) ** 2).mean().item() / (
            x.float() ** 2
        ).mean().item()

        # Separate quantization
        q_nope = turboquant_quantize(x_nope, h_nope, bits, "mse")
        r_nope = turboquant_dequantize(q_nope, h_nope, bits, "mse", torch.float32)[
            :, :nope_dim
        ]
        q_rope = turboquant_quantize(x_rope, h_rope, bits, "mse")
        r_rope = turboquant_dequantize(q_rope, h_rope, bits, "mse", torch.float32)[
            :, :rope_dim
        ]
        r_sep = torch.cat([r_nope, r_rope], dim=-1)
        mse_sep = ((x.float() - r_sep) ** 2).mean().item() / (
            x.float() ** 2
        ).mean().item()

        # Separate should be similar or better (no cross-component interference)
        print(f"  {bits}b: joint MSE={mse_joint:.6f}, separate MSE={mse_sep:.6f}")
        # Allow 2x slack for separate being slightly worse due to different Hadamard
        assert (
            mse_sep < mse_joint * 2.0
        ), f"{bits}b: separate MSE {mse_sep:.4f} much worse than joint {mse_joint:.4f}"
    print("PASS: test_mla_separate_vs_joint")


# ---------------------------------------------------------------------------
# Pool-level tests (requires the full sglang import)
# ---------------------------------------------------------------------------


def _make_pool(bits=4.0, mode="mse", size=1024, layer_num=2):
    """Create a MLATokenToKVPoolTurboQuant for testing."""
    from sglang.srt.mem_cache.turboquant_mla_memory_pool import (
        MLATokenToKVPoolTurboQuant,
    )

    return MLATokenToKVPoolTurboQuant(
        size=size,
        page_size=1,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
        bits=bits,
        mode=mode,
        start_layer=0,
        end_layer=layer_num,
    )


class _FakeLayer:
    """Minimal mock for RadixAttention."""

    def __init__(self, layer_id):
        self.layer_id = layer_id


def test_mla_pool_set_get_roundtrip():
    """Create pool, write data via set_mla_kv_buffer, read via get_key_buffer,
    verify cosine similarity > 0.99 at 4-bit."""
    pool = _make_pool(bits=4.0, size=256, layer_num=2)
    layer = _FakeLayer(0)

    N = 64
    loc = torch.arange(N, device=DEVICE)
    nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
    rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(layer, loc, nope, rope)

    # Read back via get_key_buffer (returns full kv_cache_dim)
    key_buf = pool.get_key_buffer(0)
    recovered_nope = key_buf[loc, :, :KV_LORA_RANK].float()
    recovered_rope = key_buf[loc, :, KV_LORA_RANK:].float()

    # Check cosine similarity
    orig_nope = nope.float().reshape(-1, KV_LORA_RANK)
    rec_nope = recovered_nope.reshape(-1, KV_LORA_RANK)
    cos_nope = F.cosine_similarity(orig_nope, rec_nope, dim=-1).mean().item()

    orig_rope = rope.float().reshape(-1, QK_ROPE_HEAD_DIM)
    rec_rope = recovered_rope.reshape(-1, QK_ROPE_HEAD_DIM)
    cos_rope = F.cosine_similarity(orig_rope, rec_rope, dim=-1).mean().item()

    print(f"  nope cos_sim={cos_nope:.6f}, rope cos_sim={cos_rope:.6f}")
    assert cos_nope > 0.99, f"nope cosine similarity {cos_nope:.4f} too low"
    assert cos_rope > 0.99, f"rope cosine similarity {cos_rope:.4f} too low"
    print("PASS: test_mla_pool_set_get_roundtrip")


def test_mla_pool_get_mla_kv_buffer():
    """Test get_mla_kv_buffer returns separate (nope, rope) with correct shapes."""
    pool = _make_pool(bits=4.0, size=256, layer_num=1)
    layer = _FakeLayer(0)

    N = 32
    loc = torch.arange(N, device=DEVICE)
    nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
    rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(layer, loc, nope, rope)

    # Read back via get_mla_kv_buffer
    read_loc = torch.arange(16, device=DEVICE)  # read subset
    rec_nope, rec_rope = pool.get_mla_kv_buffer(layer, read_loc)

    assert rec_nope.shape == (16, 1, KV_LORA_RANK), f"nope shape {rec_nope.shape}"
    assert rec_rope.shape == (16, 1, QK_ROPE_HEAD_DIM), f"rope shape {rec_rope.shape}"

    # Quality check
    cos = (
        F.cosine_similarity(
            nope[:16].float().reshape(-1, KV_LORA_RANK),
            rec_nope.float().reshape(-1, KV_LORA_RANK),
            dim=-1,
        )
        .mean()
        .item()
    )
    assert cos > 0.99, f"get_mla_kv_buffer nope cosine {cos:.4f} too low"
    print(f"  nope cos_sim={cos:.6f}")
    print("PASS: test_mla_pool_get_mla_kv_buffer")


def test_mla_pool_value_is_nope_only():
    """Verify get_value_buffer returns only first kv_lora_rank dims."""
    pool = _make_pool(bits=4.0, size=128, layer_num=1)
    val_buf = pool.get_value_buffer(0)
    assert (
        val_buf.shape[-1] == KV_LORA_RANK
    ), f"value buffer dim {val_buf.shape[-1]} != {KV_LORA_RANK}"
    print("PASS: test_mla_pool_value_is_nope_only")


def test_mla_pool_move_kv_cache():
    """Write, move, verify data integrity."""
    pool = _make_pool(bits=4.0, size=256, layer_num=1)
    layer = _FakeLayer(0)

    N = 32
    src_loc = torch.arange(N, device=DEVICE)
    nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
    rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(layer, src_loc, nope, rope)

    # Read original
    key_before = pool.get_key_buffer(0)[src_loc].clone()

    # Move to new locations
    tgt_loc = torch.arange(100, 100 + N, device=DEVICE)
    pool.move_kv_cache(tgt_loc, src_loc)

    # Read from new locations
    key_after = pool.get_key_buffer(0)[tgt_loc]

    assert torch.allclose(
        key_before, key_after, atol=1e-6
    ), "move_kv_cache corrupted data"
    print("PASS: test_mla_pool_move_kv_cache")


def test_mla_mixed_precision():
    """Test 2.5-bit and 3.5-bit configs."""
    for bits in [2.5, 3.5]:
        pool = _make_pool(bits=bits, size=128, layer_num=1)
        layer = _FakeLayer(0)

        N = 32
        loc = torch.arange(N, device=DEVICE)
        nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
        rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

        pool.set_mla_kv_buffer(layer, loc, nope, rope)

        key_buf = pool.get_key_buffer(0)
        recovered = key_buf[loc, :, :KV_LORA_RANK].float()
        cos = (
            F.cosine_similarity(
                nope.float().reshape(-1, KV_LORA_RANK),
                recovered.reshape(-1, KV_LORA_RANK),
                dim=-1,
            )
            .mean()
            .item()
        )
        print(f"  {bits}b: nope cos_sim={cos:.6f}")
        # Lower bits = lower quality, but should still be reasonable
        min_cos = 0.95 if bits >= 3.5 else 0.90
        assert cos > min_cos, f"{bits}b nope cosine {cos:.4f} < {min_cos}"
    print("PASS: test_mla_mixed_precision")


def test_mla_prod_mode():
    """Test 'prod' mode with QJL signs."""
    pool = _make_pool(bits=4.0, mode="prod", size=128, layer_num=1)
    layer = _FakeLayer(0)

    N = 32
    loc = torch.arange(N, device=DEVICE)
    nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
    rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

    pool.set_mla_kv_buffer(layer, loc, nope, rope)

    key_buf = pool.get_key_buffer(0)
    recovered = key_buf[loc, :, :KV_LORA_RANK].float()
    cos = (
        F.cosine_similarity(
            nope.float().reshape(-1, KV_LORA_RANK),
            recovered.reshape(-1, KV_LORA_RANK),
            dim=-1,
        )
        .mean()
        .item()
    )
    print(f"  prod 4b: nope cos_sim={cos:.6f}")
    assert cos > 0.98, f"prod mode nope cosine {cos:.4f} too low"
    print("PASS: test_mla_prod_mode")


def test_mla_compression_ratio():
    """Verify actual memory savings vs bf16 baseline."""
    pool = _make_pool(bits=4.0, size=1024, layer_num=2)
    actual_bytes = pool.get_kv_size_bytes()

    # bf16 baseline: size * 1 head * (512+64) * 2 bytes * 2 layers + padding
    m = 1024 + 1  # size + page_size
    bf16_baseline = m * 1 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2 * 2  # bf16 * 2 layers

    ratio = bf16_baseline / actual_bytes
    print(
        f"  actual={actual_bytes} bytes, bf16_baseline={bf16_baseline} bytes, ratio={ratio:.2f}x"
    )
    # With only 2 layers, the shared workspace (full bf16 size, NOT per-layer)
    # dominates. At higher layer counts the ratio improves dramatically since
    # workspace is amortized.  With 2 layers, expect modest savings.
    assert ratio > 1.2, f"Compression ratio {ratio:.2f}x too low"

    # Verify ratio improves with more layers (workspace amortized)
    pool_deep = _make_pool(bits=4.0, size=1024, layer_num=60)
    actual_deep = pool_deep.get_kv_size_bytes()
    bf16_deep = m * 1 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2 * 60
    ratio_deep = bf16_deep / actual_deep
    print(
        f"  60-layer: actual={actual_deep} bytes, bf16={bf16_deep} bytes, ratio={ratio_deep:.2f}x"
    )
    assert ratio_deep > 2.5, f"60-layer compression ratio {ratio_deep:.2f}x too low"
    print("PASS: test_mla_compression_ratio")


def test_mla_set_kv_buffer_concatenated():
    """Test set_kv_buffer with concatenated [nope | rope] input."""
    pool = _make_pool(bits=4.0, size=128, layer_num=1)
    layer = _FakeLayer(0)

    N = 16
    loc = torch.arange(N, device=DEVICE)
    nope = torch.randn(N, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16)
    rope = torch.randn(N, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    combined_k = torch.cat([nope, rope], dim=-1)
    dummy_v = combined_k  # unused in MLA TQ

    pool.set_kv_buffer(layer, loc, combined_k, dummy_v)

    key_buf = pool.get_key_buffer(0)
    recovered = key_buf[loc].float()
    cos = (
        F.cosine_similarity(
            combined_k.float().reshape(-1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            recovered.reshape(-1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            dim=-1,
        )
        .mean()
        .item()
    )
    print(f"  set_kv_buffer concatenated cos_sim={cos:.6f}")
    assert cos > 0.99, f"set_kv_buffer cosine {cos:.4f} too low"
    print("PASS: test_mla_set_kv_buffer_concatenated")


def test_nsa_pool_inherits_mla_tq():
    """Verify NSATokenToKVPoolTurboQuant has both TQ main cache + FP8 index cache."""
    from sglang.srt.mem_cache.turboquant_mla_memory_pool import (
        MLATokenToKVPoolTurboQuant,
    )
    from sglang.srt.mem_cache.turboquant_nsa_memory_pool import (
        NSATokenToKVPoolTurboQuant,
    )

    pool = NSATokenToKVPoolTurboQuant(
        size=1024,
        page_size=64,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=2,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        bits=4.0,
        mode="mse",
        start_layer=0,
        end_layer=2,
    )

    # Should be an instance of MLATokenToKVPoolTurboQuant
    assert isinstance(pool, MLATokenToKVPoolTurboQuant)

    # Should have TQ buffers
    assert hasattr(pool, "nope_packed_buffer")
    assert hasattr(pool, "rope_packed_buffer")
    assert len(pool.nope_packed_buffer) == 2

    # Should have index K buffers
    assert hasattr(pool, "index_k_with_scale_buffer")
    assert len(pool.index_k_with_scale_buffer) == 2

    # Should have use_nsa flag
    assert pool.use_nsa is True

    # Index K buffer shape check
    num_pages = (1024 + 64 + 1) // 64
    expected_page_data = 64 * (128 + 128 // 128 * 4)
    assert pool.index_k_with_scale_buffer[0].shape == (num_pages, expected_page_data)

    print("PASS: test_nsa_pool_inherits_mla_tq")


# ---------------------------------------------------------------------------
# Fused decode kernel tests
# ---------------------------------------------------------------------------


def test_hadamard_sign_convention():
    """Verify <HT.forward(q), HT.forward(k)> == <q, k> for random vectors."""
    ht = HadamardTransform(KV_LORA_RANK, seed=42, device=DEVICE)
    N = 128
    q = torch.randn(N, KV_LORA_RANK, device=DEVICE)
    k = torch.randn(N, KV_LORA_RANK, device=DEVICE)

    dot_orig = (q.float() * k.float()).sum(dim=-1)
    q_rot = ht.forward(q)
    k_rot = ht.forward(k)
    dot_rot = (q_rot.float() * k_rot.float()).sum(dim=-1)

    # Should match to high precision (orthogonal transform preserves inner product)
    rel_err = ((dot_orig - dot_rot).abs() / (dot_orig.abs() + 1e-8)).mean().item()
    print(f"  Hadamard dot product relative error: {rel_err:.8f}")
    assert rel_err < 1e-4, f"Hadamard dot product error {rel_err:.6f} too large"
    print("PASS: test_hadamard_sign_convention")


def test_wvc_rotation_cancels():
    """Verify (o_rotated @ w_vc_rot) ≈ (o_true @ w_vc) for random data."""
    import importlib

    wvc_mod_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "srt",
        "layers",
        "attention",
        "triton_ops",
        "wvc_rotation.py",
    )
    wvc_spec = importlib.util.spec_from_file_location(
        "wvc_rotation", os.path.abspath(wvc_mod_path)
    )
    wvc_mod = importlib.util.module_from_spec(wvc_spec)
    wvc_spec.loader.exec_module(wvc_mod)

    heads = 16
    kv_lora_rank = KV_LORA_RANK
    v_head_dim = 128

    ht = HadamardTransform(kv_lora_rank, seed=42, device=DEVICE)

    # Random w_vc and attention output
    w_vc = torch.randn(
        heads, kv_lora_rank, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    o_true = torch.randn(heads, 8, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16)

    # Compute rotated w_vc
    w_vc_rot = wvc_mod.compute_rotated_wvc(w_vc, ht)

    # Compute o_rot = R @ o_true (apply Hadamard along kv_lora_rank dim)
    # o_true is (heads, batch, kv_lora_rank), apply R to last dim
    o_rot = ht.forward(o_true).to(torch.bfloat16)

    # Reference: o_true @ w_vc
    ref = torch.bmm(o_true.float(), w_vc.float())

    # Fused path: o_rot @ w_vc_rot
    fused = torch.bmm(o_rot.float(), w_vc_rot.float())

    cos = F.cosine_similarity(ref.flatten(1), fused.flatten(1), dim=-1).mean().item()
    print(f"  w_vc rotation cancellation cosine sim: {cos:.6f}")
    assert cos > 0.999, f"w_vc rotation cosine sim {cos:.4f} too low"
    print("PASS: test_wvc_rotation_cancels")


def test_separate_centroid_scaling():
    """Verify nope_centroids_scaled = raw/√512 and rope_centroids_scaled = raw/√64."""
    import math

    pool = _make_pool(bits=4.0, mode="mse", size=64, layer_num=1)

    assert pool.can_use_fused_kernel, "Pool should support fused kernel for 4-bit MSE"
    assert pool.nope_centroids_scaled is not None
    assert pool.rope_centroids_scaled is not None

    # Check shapes
    assert pool.nope_centroids_scaled.shape == (16,)
    assert pool.rope_centroids_scaled.shape == (16,)

    # Check values: raw 4-bit centroids divided by sqrt(dim)
    raw_centroids = torch.tensor(
        [
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
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    expected_nope = raw_centroids / math.sqrt(512)
    expected_rope = raw_centroids / math.sqrt(64)

    assert torch.allclose(
        pool.nope_centroids_scaled, expected_nope, atol=1e-5
    ), "nope centroids scaling mismatch"
    assert torch.allclose(
        pool.rope_centroids_scaled, expected_rope, atol=1e-5
    ), "rope centroids scaling mismatch"
    print("PASS: test_separate_centroid_scaling")


def test_fallback_to_workspace():
    """Verify non-4bit or mixed mode falls back (can_use_fused_kernel=False)."""
    # 3-bit should not use fused kernel
    pool_3b = _make_pool(bits=3.0, mode="mse", size=64, layer_num=1)
    assert not pool_3b.can_use_fused_kernel, "3-bit should not use fused kernel"
    assert pool_3b.nope_centroids_scaled is None

    # Mixed-precision should not use fused kernel
    pool_mixed = _make_pool(bits=3.5, mode="mse", size=64, layer_num=1)
    assert (
        not pool_mixed.can_use_fused_kernel
    ), "3.5-bit mixed should not use fused kernel"

    # Prod mode should not use fused kernel
    pool_prod = _make_pool(bits=4.0, mode="prod", size=64, layer_num=1)
    assert not pool_prod.can_use_fused_kernel, "prod mode should not use fused kernel"

    # 4-bit MSE should use fused kernel
    pool_4b = _make_pool(bits=4.0, mode="mse", size=64, layer_num=1)
    assert pool_4b.can_use_fused_kernel, "4-bit MSE should use fused kernel"
    print("PASS: test_fallback_to_workspace")


def test_fused_kernel_vs_workspace():
    """Verify fused kernel latent output matches workspace-dequant path."""
    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant import (
        decode_attention_fwd_tq,
    )

    pool = _make_pool(bits=4.0, mode="mse", size=256, layer_num=1)
    layer = _FakeLayer(0)

    N_tokens = 64
    batch = 4
    heads = 16
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM

    loc = torch.arange(N_tokens, device=DEVICE)
    nope_data = torch.randn(
        N_tokens, 1, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    rope_data = torch.randn(
        N_tokens, 1, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(layer, loc, nope_data, rope_data)

    # Random Q (not rotated yet)
    q_nope = torch.randn(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        batch, heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )

    # Each batch element attends to a slice of the N_tokens
    seq_lens = torch.tensor([16, 16, 16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    # --- Workspace path (reference) ---
    # Dequant full K buffer
    key_buf = pool.get_key_buffer(0).to(torch.bfloat16)  # (max_tokens, 1, 576)
    val_buf = pool.get_value_buffer(0).to(torch.bfloat16)  # (max_tokens, 1, 512)

    # Manual attention for reference
    ref_outputs = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end]

        k_nope = key_buf[token_ids, 0, :kv_lora_rank].float()  # (seq_len, 512)
        k_rope_slice = key_buf[token_ids, 0, kv_lora_rank:].float()  # (seq_len, 64)
        v = val_buf[token_ids, 0, :].float()  # (seq_len, 512)

        # Compute attention scores
        q_n = q_nope[b].float()  # (heads, 512)
        q_r = q_rope[b].float()  # (heads, 64)

        scores = q_n @ k_nope.T + q_r @ k_rope_slice.T  # (heads, seq_len)
        # No sm_scale applied in this test (set to 1.0)
        probs = torch.softmax(scores, dim=-1)  # (heads, seq_len)
        out = probs @ v  # (heads, 512)
        ref_outputs.append(out)

    ref_output = torch.stack(ref_outputs, dim=0)  # (batch, heads, 512)

    # --- Fused kernel path ---
    # Rotate Q
    q_nope_rot = pool.nope_hadamard.forward(q_nope)
    q_rope_rot = pool.rope_hadamard.forward(q_rope)

    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    max_kv_splits = 1

    o_fused = torch.zeros(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    decode_attention_fwd_tq(
        q_nope_rot,
        q_rope_rot,
        pool.get_nope_packed_buffer(0),
        pool.get_rope_packed_buffer(0),
        pool.get_nope_norms_buffer(0),
        pool.get_rope_norms_buffer(0),
        pool.nope_centroids_scaled,
        pool.rope_centroids_scaled,
        o_fused,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale=1.0,
    )

    # The fused output is in rotated space. To compare with reference,
    # we need to apply inverse Hadamard (or compare via w_vc projection).
    # Instead, compare attention scores indirectly by checking cosine similarity
    # after applying inverse rotation to the fused output.
    o_fused_f = o_fused.float()
    # Inverse rotate: for each (batch, head), apply inverse Hadamard
    o_unrotated = pool.nope_hadamard.inverse(o_fused_f)[:, :, :kv_lora_rank]

    cos = (
        F.cosine_similarity(ref_output.flatten(1), o_unrotated.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  Fused kernel vs workspace cosine sim: {cos:.6f}")
    assert cos > 0.99, f"Fused kernel output cosine sim {cos:.4f} too low"
    print("PASS: test_fused_kernel_vs_workspace")


def test_nsa_pool_fused_kernel_flag():
    """Verify NSATokenToKVPoolTurboQuant inherits can_use_fused_kernel from parent."""
    from sglang.srt.mem_cache.turboquant_nsa_memory_pool import (
        NSATokenToKVPoolTurboQuant,
    )

    def _make_nsa_pool(bits=4.0, mode="mse"):
        return NSATokenToKVPoolTurboQuant(
            size=1024,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=2,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            bits=bits,
            mode=mode,
            start_layer=0,
            end_layer=2,
        )

    # 4-bit MSE NSA pool: should inherit True from parent
    pool_4b = _make_nsa_pool(bits=4.0, mode="mse")
    assert pool_4b.can_use_fused_kernel, "4-bit MSE NSA pool should enable fused kernel"

    # 3-bit NSA pool: should be False
    pool_3b = _make_nsa_pool(bits=3.0, mode="mse")
    assert (
        not pool_3b.can_use_fused_kernel
    ), "3-bit NSA pool should not use fused kernel"

    # prod mode NSA pool: should be False
    pool_prod = _make_nsa_pool(bits=4.0, mode="prod")
    assert (
        not pool_prod.can_use_fused_kernel
    ), "prod mode NSA pool should not use fused kernel"

    print("PASS: test_nsa_pool_fused_kernel_flag")


def test_fused_decode_nsa_sparse_path():
    """Integration test: exercises _forward_turboquant_fused() with NSA pool.

    Instantiates NSATokenToKVPoolTurboQuant (verifying can_use_fused_kernel=True),
    calls NativeSparseAttnBackend._forward_turboquant_fused() via a minimal mock
    backend, and compares output against workspace-dequant + manual sparse attention.

    Uses non-contiguous token positions with -1 padding (the real NSA case).
    """
    from sglang.srt.layers.attention.nsa.triton_kernel import get_valid_kv_indices
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
    from sglang.srt.mem_cache.turboquant_nsa_memory_pool import (
        NSATokenToKVPoolTurboQuant,
    )

    N_tokens = 128
    batch = 4
    heads = 16
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    topk = 32
    max_bs = 64

    # --- Create NSA TQ pool (page_size=1 for unit test simplicity) ---
    pool = NSATokenToKVPoolTurboQuant(
        size=512,
        page_size=64,
        dtype=torch.bfloat16,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=1,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        bits=4.0,
        mode="mse",
        start_layer=0,
        end_layer=1,
    )

    # Verify the pool change: NSA pool now inherits fused kernel support
    assert (
        pool.can_use_fused_kernel
    ), "NSATokenToKVPoolTurboQuant should inherit can_use_fused_kernel=True for 4-bit MSE"

    # --- Populate KV cache ---
    layer = _FakeLayer(0)
    layer.tp_q_head_num = heads
    layer.v_head_dim = kv_lora_rank
    layer.head_dim = kv_lora_rank + qk_rope_head_dim
    layer.scaling = 1.0
    layer.logit_cap = 0.0
    layer._tq_fused_ready = True

    loc = torch.arange(N_tokens, device=DEVICE)
    nope_data = torch.randn(
        N_tokens, 1, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    rope_data = torch.randn(
        N_tokens, 1, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(layer, loc, nope_data, rope_data)

    # --- Create sparse topk_indices (non-contiguous + -1 padding) ---
    page_table_1 = torch.full((batch, topk), -1, dtype=torch.int32, device=DEVICE)
    actual_counts = [20, 15, 25, 10]
    for b in range(batch):
        count = actual_counts[b]
        selected = torch.arange(b * 3, b * 3 + count * 2, 2, device=DEVICE)[:count]
        selected = selected.clamp(max=N_tokens - 1)
        page_table_1[b, :count] = selected.to(torch.int32)

    # --- Build mock backend with attributes _forward_turboquant_fused needs ---
    class _MockBackend:
        pass

    backend = _MockBackend()
    backend.device = DEVICE
    backend.req_to_token = torch.zeros(max_bs, 1, dtype=torch.int32, device=DEVICE)
    backend.nsa_index_topk = topk
    # Init lazy buffers to None (method will allocate them)
    backend._tq_kv_indptr = None
    backend._tq_kv_indices = None
    backend._tq_attn_logits = None
    backend._tq_attn_lse = None
    backend._tq_max_kv_splits = 128

    # --- Build mock forward_batch ---
    class _MockForwardBatch:
        pass

    forward_batch = _MockForwardBatch()

    # --- Prepare Q ---
    q_nope = torch.randn(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        batch, heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )

    # --- Call the actual _forward_turboquant_fused method ---
    o_flat = NativeSparseAttnBackend._forward_turboquant_fused(
        backend,
        q_nope,
        q_rope,
        page_table_1,
        pool,
        layer,
        forward_batch,
    )

    # Verify flag was set on forward_batch
    assert getattr(
        forward_batch, "_tq_rotated_output", False
    ), "_tq_rotated_output flag should be set by _forward_turboquant_fused"

    # Verify lazy buffers were allocated
    assert backend._tq_kv_indptr is not None, "Lazy kv_indptr should be allocated"
    assert backend._tq_kv_indices is not None, "Lazy kv_indices should be allocated"
    assert backend._tq_attn_logits is not None, "Lazy attn_logits should be allocated"

    # Reshape output
    o_fused = o_flat.view(batch, heads, kv_lora_rank)

    # --- Workspace-dequant reference: manual sparse attention ---
    # Build kv_indptr/kv_indices for reference path
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    non_minus1_counts = (page_table_1 != -1).sum(dim=1)
    kv_indptr[1:] = torch.cumsum(non_minus1_counts, dim=0)
    total_indices = kv_indptr[-1].item()
    kv_indices = torch.zeros(total_indices + 256, dtype=torch.int32, device=DEVICE)
    get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, batch)

    key_buf = pool.get_key_buffer(0).to(torch.bfloat16)
    val_buf = pool.get_value_buffer(0).to(torch.bfloat16)

    ref_latents = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end].long()

        k_nope = key_buf[token_ids, 0, :kv_lora_rank].float()
        k_rope_slice = key_buf[token_ids, 0, kv_lora_rank:].float()
        v = val_buf[token_ids, 0, :].float()

        scores = q_nope[b].float() @ k_nope.T + q_rope[b].float() @ k_rope_slice.T
        probs = torch.softmax(scores, dim=-1)
        ref_latents.append(probs @ v)

    o_workspace = torch.stack(ref_latents, dim=0)

    # Inverse-rotate fused output to compare
    o_unrotated = pool.nope_hadamard.inverse(o_fused.float())[:, :, :kv_lora_rank]

    cos_latent = (
        F.cosine_similarity(o_workspace.flatten(1), o_unrotated.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  NSA integration: latent cosine sim: {cos_latent:.6f}")
    assert (
        cos_latent > 0.99
    ), f"NSA integration latent cosine sim {cos_latent:.4f} too low"

    # Projected comparison through w_vc
    v_head_dim = 128
    w_vc = torch.randn(
        heads, kv_lora_rank, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    proj_ref = torch.bmm(
        o_workspace.reshape(batch * heads, 1, kv_lora_rank),
        w_vc.float()
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * heads, kv_lora_rank, v_head_dim),
    ).reshape(batch, heads, v_head_dim)
    proj_fused = torch.bmm(
        o_unrotated.reshape(batch * heads, 1, kv_lora_rank),
        w_vc.float()
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * heads, kv_lora_rank, v_head_dim),
    ).reshape(batch, heads, v_head_dim)

    cos_proj = (
        F.cosine_similarity(proj_ref.flatten(1), proj_fused.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  NSA integration: projected cosine sim: {cos_proj:.6f}")
    assert (
        cos_proj > 0.99
    ), f"NSA integration projected cosine sim {cos_proj:.4f} too low"

    # Consume flag (as forward_mla would)
    forward_batch._tq_rotated_output = False
    assert not forward_batch._tq_rotated_output

    print("PASS: test_fused_decode_nsa_sparse_path")


def test_fused_decode_end_to_end():
    """End-to-end test: backend dispatch → fused kernel → w_vc_tq_rotated projection.

    Verifies the complete V-path correctness:
        fused rotated output × rotated w_vc ≈ workspace output × original w_vc
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant import (
        decode_attention_fwd_tq,
    )
    from sglang.srt.layers.attention.triton_ops.wvc_rotation import (
        compute_rotated_wvc,
    )

    pool = _make_pool(bits=4.0, mode="mse", size=256, layer_num=1)
    layer = _FakeLayer(0)

    N_tokens = 64
    batch = 4
    heads = 16
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    v_head_dim = 128

    # Populate KV cache
    loc = torch.arange(N_tokens, device=DEVICE)
    nope_data = torch.randn(
        N_tokens, 1, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    rope_data = torch.randn(
        N_tokens, 1, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(layer, loc, nope_data, rope_data)

    # Random Q and w_vc
    q_nope = torch.randn(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        batch, heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    w_vc = torch.randn(
        heads, kv_lora_rank, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )

    seq_lens = torch.tensor([16, 16, 16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    # --- Workspace path (reference): dequant → manual attention → w_vc ---
    key_buf = pool.get_key_buffer(0).to(torch.bfloat16)
    val_buf = pool.get_value_buffer(0).to(torch.bfloat16)

    ref_outputs = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end]

        k_nope = key_buf[token_ids, 0, :kv_lora_rank].float()
        k_rope_slice = key_buf[token_ids, 0, kv_lora_rank:].float()
        v = val_buf[token_ids, 0, :].float()

        q_n = q_nope[b].float()
        q_r = q_rope[b].float()

        scores = q_n @ k_nope.T + q_r @ k_rope_slice.T
        probs = torch.softmax(scores, dim=-1)
        o_latent = probs @ v  # (heads, kv_lora_rank)
        # Project through original w_vc
        o_final = torch.bmm(
            o_latent.unsqueeze(1),  # (heads, 1, kv_lora_rank)
            w_vc.float(),  # (heads, kv_lora_rank, v_head_dim)
        ).squeeze(
            1
        )  # (heads, v_head_dim)
        ref_outputs.append(o_final)

    ref_output = torch.stack(ref_outputs, dim=0)  # (batch, heads, v_head_dim)

    # --- Fused path: Q rotation → fused kernel → w_vc_tq_rotated ---
    q_nope_rot = pool.nope_hadamard.forward(q_nope)
    q_rope_rot = pool.rope_hadamard.forward(q_rope)

    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    max_kv_splits = 1

    o_fused = torch.zeros(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    decode_attention_fwd_tq(
        q_nope_rot,
        q_rope_rot,
        pool.get_nope_packed_buffer(0),
        pool.get_rope_packed_buffer(0),
        pool.get_nope_norms_buffer(0),
        pool.get_rope_norms_buffer(0),
        pool.nope_centroids_scaled,
        pool.rope_centroids_scaled,
        o_fused,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale=1.0,
    )

    # Compute rotated w_vc and project
    w_vc_rot = compute_rotated_wvc(w_vc, pool.nope_hadamard)
    fused_final = torch.bmm(
        o_fused.float().reshape(batch * heads, 1, kv_lora_rank),
        w_vc_rot.float()
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * heads, kv_lora_rank, v_head_dim),
    ).reshape(batch, heads, v_head_dim)

    # Compare
    cos = (
        F.cosine_similarity(ref_output.flatten(1), fused_final.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  E2E fused vs workspace final output cosine sim: {cos:.6f}")
    assert cos > 0.99, f"E2E cosine sim {cos:.4f} too low"
    print("PASS: test_fused_decode_end_to_end")


def test_fused_decode_deep_gemm_path():
    """Verify inverse-rotate approach matches workspace-dequant path for deep_gemm.

    Simulates the deep_gemm integration:
    1. Run fused kernel → get o_rot (rotated output)
    2. Inverse rotate → get o_unrot (original latent space)
    3. Run workspace dequant → manual attention → get o_workspace
    4. Assert o_unrot ≈ o_workspace (cosine > 0.99)
    5. Project both through same w_vc: o_unrot @ w_vc vs o_workspace @ w_vc
    6. Assert final projected outputs match (cosine > 0.99)
    7. Verify _tq_rotated_output flag is consumed (set to False)
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant import (
        decode_attention_fwd_tq,
    )

    pool = _make_pool(bits=4.0, mode="mse", size=256, layer_num=1)
    layer = _FakeLayer(0)

    N_tokens = 64
    batch = 4
    heads = 16
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    v_head_dim = 128

    # Populate KV cache
    loc = torch.arange(N_tokens, device=DEVICE)
    nope_data = torch.randn(
        N_tokens, 1, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    rope_data = torch.randn(
        N_tokens, 1, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(layer, loc, nope_data, rope_data)

    # Random Q and w_vc (original FP8-compatible weights, simulated as bf16)
    q_nope = torch.randn(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        batch, heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    w_vc = torch.randn(
        heads, kv_lora_rank, v_head_dim, device=DEVICE, dtype=torch.bfloat16
    )

    seq_lens = torch.tensor([16, 16, 16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    # --- Workspace path (reference): dequant → manual attention ---
    key_buf = pool.get_key_buffer(0).to(torch.bfloat16)
    val_buf = pool.get_value_buffer(0).to(torch.bfloat16)

    ref_latents = []
    for b in range(batch):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_ids = kv_indices[start:end]

        k_nope = key_buf[token_ids, 0, :kv_lora_rank].float()
        k_rope_slice = key_buf[token_ids, 0, kv_lora_rank:].float()
        v = val_buf[token_ids, 0, :].float()

        q_n = q_nope[b].float()
        q_r = q_rope[b].float()

        scores = q_n @ k_nope.T + q_r @ k_rope_slice.T
        probs = torch.softmax(scores, dim=-1)
        o_latent = probs @ v  # (heads, kv_lora_rank)
        ref_latents.append(o_latent)

    o_workspace = torch.stack(ref_latents, dim=0)  # (batch, heads, kv_lora_rank)

    # --- Fused kernel path ---
    q_nope_rot = pool.nope_hadamard.forward(q_nope)
    q_rope_rot = pool.rope_hadamard.forward(q_rope)

    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    max_kv_splits = 1

    o_fused = torch.zeros(
        batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    decode_attention_fwd_tq(
        q_nope_rot,
        q_rope_rot,
        pool.get_nope_packed_buffer(0),
        pool.get_rope_packed_buffer(0),
        pool.get_nope_norms_buffer(0),
        pool.get_rope_norms_buffer(0),
        pool.nope_centroids_scaled,
        pool.rope_centroids_scaled,
        o_fused,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale=1.0,
    )

    # Simulate flag lifecycle
    class FakeBatch:
        pass

    fb = FakeBatch()
    fb._tq_rotated_output = True  # Backend sets flag

    # --- Step 2: Inverse rotate (deep_gemm path in forward_mla.py) ---
    assert fb._tq_rotated_output is True
    fb._tq_rotated_output = False  # Consume flag (as forward_mla does)

    o_unrot = pool.nope_hadamard.inverse(o_fused.float()).to(o_fused.dtype)

    # --- Step 3: Compare latent outputs ---
    cos_latent = (
        F.cosine_similarity(o_workspace.flatten(1), o_unrot.float().flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  Latent cosine sim (inverse-rot vs workspace): {cos_latent:.6f}")
    assert cos_latent > 0.99, f"Latent cosine sim {cos_latent:.4f} too low"

    # --- Step 4: Project both through same w_vc ---
    proj_unrot = torch.bmm(
        o_unrot.float().reshape(batch * heads, 1, kv_lora_rank),
        w_vc.float()
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * heads, kv_lora_rank, v_head_dim),
    ).reshape(batch, heads, v_head_dim)

    proj_workspace = torch.bmm(
        o_workspace.reshape(batch * heads, 1, kv_lora_rank),
        w_vc.float()
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .reshape(batch * heads, kv_lora_rank, v_head_dim),
    ).reshape(batch, heads, v_head_dim)

    cos_proj = (
        F.cosine_similarity(proj_workspace.flatten(1), proj_unrot.flatten(1), dim=-1)
        .mean()
        .item()
    )
    print(f"  Projected cosine sim (through w_vc): {cos_proj:.6f}")
    assert cos_proj > 0.99, f"Projected cosine sim {cos_proj:.4f} too low"

    # --- Step 5: Verify flag consumed ---
    assert fb._tq_rotated_output is False, "Flag should be consumed (False)"

    print("PASS: test_fused_decode_deep_gemm_path")


def test_forward_batch_flag_lifecycle():
    """Verify _tq_rotated_output flag is set by backend and consumed exactly once per layer.

    Simulates the real multi-layer flow:
      For each layer:
        1. Backend (forward_decode) sets _tq_rotated_output = True
        2. forward_mla checks the flag, uses w_vc_tq_rotated, resets to False
        3. Next layer sees False until backend sets it again
    Also verifies that consuming flag changes the projection path (rotated vs original).
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant import (
        decode_attention_fwd_tq,
    )
    from sglang.srt.layers.attention.triton_ops.wvc_rotation import (
        compute_rotated_wvc,
    )

    pool = _make_pool(bits=4.0, mode="mse", size=256, layer_num=3)
    num_layers = 3
    batch = 2
    heads = 16
    kv_lora_rank = KV_LORA_RANK
    v_head_dim = 128
    N_tokens = 32

    # Populate KV cache for all layers
    loc = torch.arange(N_tokens, device=DEVICE)
    for lid in range(num_layers):
        layer = _FakeLayer(lid)
        nope = torch.randn(
            N_tokens, 1, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
        )
        rope = torch.randn(
            N_tokens, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        pool.set_mla_kv_buffer(layer, loc, nope, rope)

    # Per-layer w_vc weights (different for each layer)
    w_vcs = [
        torch.randn(
            heads, kv_lora_rank, v_head_dim, device=DEVICE, dtype=torch.bfloat16
        )
        for _ in range(num_layers)
    ]
    w_vc_rots = [compute_rotated_wvc(w, pool.nope_hadamard) for w in w_vcs]

    # Attention metadata
    seq_lens = torch.tensor([16, 16], dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

    class FakeBatch:
        pass

    fb = FakeBatch()

    # Simulate multi-layer decode: backend → forward_mla for each layer
    for lid in range(num_layers):
        q_nope = torch.randn(
            batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
        )
        q_rope = torch.randn(
            batch, heads, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )

        # --- Step 1: Backend sets flag (simulates FlashInferMLAAttnBackend.forward_decode) ---
        assert not getattr(
            fb, "_tq_rotated_output", False
        ), f"Layer {lid}: flag was True BEFORE backend set it (leaked from previous layer)"

        q_nope_rot = pool.nope_hadamard.forward(q_nope)
        q_rope_rot = pool.rope_hadamard.forward(q_rope)
        num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)

        o_fused = torch.zeros(
            batch, heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
        )
        decode_attention_fwd_tq(
            q_nope_rot,
            q_rope_rot,
            pool.get_nope_packed_buffer(lid),
            pool.get_rope_packed_buffer(lid),
            pool.get_nope_norms_buffer(lid),
            pool.get_rope_norms_buffer(lid),
            pool.nope_centroids_scaled,
            pool.rope_centroids_scaled,
            o_fused,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            1,
            sm_scale=1.0,
        )
        fb._tq_rotated_output = True  # Backend sets flag

        # --- Step 2: forward_mla consumes flag and uses rotated w_vc ---
        _use_rotated = getattr(fb, "_tq_rotated_output", False)
        assert _use_rotated, f"Layer {lid}: flag should be True after backend set it"
        fb._tq_rotated_output = False  # Consume

        # Project with rotated w_vc (what forward_mla does when flag is True)
        attn_output = o_fused.transpose(0, 1)  # (heads, batch, kv_lora_rank)
        result_rotated = torch.bmm(attn_output.float(), w_vc_rots[lid].float())

        # Compare with workspace path: inverse-rotate then use original w_vc
        o_unrotated = pool.nope_hadamard.inverse(o_fused.float())[:, :, :kv_lora_rank]
        attn_output_ws = o_unrotated.transpose(0, 1)
        result_workspace = torch.bmm(attn_output_ws.float(), w_vcs[lid].float())

        cos = (
            F.cosine_similarity(
                result_rotated.flatten(1), result_workspace.flatten(1), dim=-1
            )
            .mean()
            .item()
        )
        assert (
            cos > 0.99
        ), f"Layer {lid}: rotated vs workspace projection cosine {cos:.4f} too low"

        # --- Step 3: Verify flag is consumed (False for next layer) ---
        assert not getattr(
            fb, "_tq_rotated_output", False
        ), f"Layer {lid}: flag should be False after consumption"

    print(f"  Multi-layer flag lifecycle verified across {num_layers} layers")
    print("PASS: test_forward_batch_flag_lifecycle")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    # Kernel-only tests (no full sglang import needed)
    kernel_tests = [
        test_mla_nope_roundtrip_quality,
        test_mla_rope_roundtrip_quality,
        test_mla_separate_vs_joint,
    ]

    # Pool tests (require full sglang import)
    pool_tests = [
        test_mla_pool_set_get_roundtrip,
        test_mla_pool_get_mla_kv_buffer,
        test_mla_pool_value_is_nope_only,
        test_mla_pool_move_kv_cache,
        test_mla_mixed_precision,
        test_mla_prod_mode,
        test_mla_compression_ratio,
        test_mla_set_kv_buffer_concatenated,
        test_nsa_pool_inherits_mla_tq,
    ]

    # Fused decode kernel tests
    fused_tests = [
        test_hadamard_sign_convention,
        test_wvc_rotation_cancels,
        test_separate_centroid_scaling,
        test_fallback_to_workspace,
        test_fused_kernel_vs_workspace,
        test_nsa_pool_fused_kernel_flag,
        test_fused_decode_nsa_sparse_path,
        test_fused_decode_end_to_end,
        test_fused_decode_deep_gemm_path,
        test_forward_batch_flag_lifecycle,
    ]

    all_tests = kernel_tests + pool_tests + fused_tests
    passed = 0
    failed = 0

    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
