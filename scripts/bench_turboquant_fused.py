"""
Benchmark: TurboQuant fused decode kernel vs workspace-dequant path.

Measures latency of both paths across different context lengths and batch sizes,
reports speedup ratio.

Usage:
    python scripts/bench_turboquant_fused.py
"""

import time

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda")
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEADS = 16
WARMUP = 10
ITERS = 100


def _make_pool(size, layer_num=1):
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
        bits=4.0,
        mode="mse",
        start_layer=0,
        end_layer=layer_num,
    )


class FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


def bench_config(batch_size, ctx_len):
    """Benchmark one (batch_size, ctx_len) configuration."""
    total_tokens = batch_size * ctx_len
    pool = _make_pool(size=total_tokens + 64)
    layer = FakeLayer(0)

    # Populate KV cache
    loc = torch.arange(total_tokens, device=DEVICE)
    nope = torch.randn(
        total_tokens, 1, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16
    )
    rope = torch.randn(
        total_tokens, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    pool.set_mla_kv_buffer(layer, loc, nope, rope)

    # Setup attention metadata
    seq_lens = torch.full((batch_size,), ctx_len, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

    q_nope = torch.randn(
        batch_size, HEADS, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16
    )
    q_rope = torch.randn(
        batch_size, HEADS, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )

    # --- Workspace path ---
    def workspace_step():
        key_buf = pool.get_key_buffer(0).to(torch.bfloat16)
        # Just the dequant + gather cost (primary bottleneck)
        _ = key_buf[kv_indices]

    # Warmup
    for _ in range(WARMUP):
        workspace_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        workspace_step()
    torch.cuda.synchronize()
    ws_ms = (time.perf_counter() - t0) / ITERS * 1000

    # --- Fused path ---
    from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant import (
        decode_attention_fwd_tq,
    )

    q_nope_rot = pool.nope_hadamard.forward(q_nope)
    q_rope_rot = pool.rope_hadamard.forward(q_rope)

    BLOCK_N = 32
    max_kv_splits = 128
    num_kv_splits = torch.clamp(
        (seq_lens + BLOCK_N * 4 - 1) // (BLOCK_N * 4),
        min=1,
        max=max_kv_splits,
    ).to(torch.int32)
    o_fused = torch.zeros(
        batch_size, HEADS, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16
    )

    def fused_step():
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
            sm_scale=0.1,
        )

    # Warmup
    for _ in range(WARMUP):
        fused_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_step()
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / ITERS * 1000

    speedup = ws_ms / fused_ms if fused_ms > 0 else float("inf")
    return ws_ms, fused_ms, speedup


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    ctx_lengths = [1024, 4096, 16384, 65536]
    batch_sizes = [1, 8, 32]

    print(
        f"{'Batch':>6} {'CtxLen':>8} {'Workspace(ms)':>14} {'Fused(ms)':>12} {'Speedup':>9}"
    )
    print("-" * 55)

    for bs in batch_sizes:
        for ctx in ctx_lengths:
            total = bs * ctx
            if total > 512 * 1024:
                # Skip configs that would OOM on typical GPUs
                print(f"{bs:>6} {ctx:>8}    (skipped - too large)")
                continue
            try:
                ws_ms, fused_ms, speedup = bench_config(bs, ctx)
                print(
                    f"{bs:>6} {ctx:>8} {ws_ms:>13.3f} {fused_ms:>11.3f} {speedup:>8.2f}x"
                )
            except Exception as e:
                print(f"{bs:>6} {ctx:>8}    ERROR: {e}")
            # Free GPU memory between configs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
