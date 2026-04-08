#!/usr/bin/env bash
#
# TurboQuant TPOT optimization: test, profile, benchmark
#
# Usage: ssh durbstation 'cd ~/git/sglang && bash scripts/test_tq_tpot.sh'
#
# Steps:
#   1. Run unit tests (fused MHA kernel tests)
#   2. Run fused Hadamard rotation correctness test
#   3. Profile per-component TPOT timings (SGLANG_TQ_PROFILE=1)
#   4. Benchmark TPOT with bench_serving (TQ vs baseline)
#
set -uo pipefail
# Note: not using -e so pre-existing test failures don't abort the whole script

VENV="${HOME}/git/sglang/venv/bin/activate"
if [ -f "$VENV" ]; then
    source "$VENV"
fi

OUTDIR="$(pwd)/tq_tpot_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "=== Results dir: $OUTDIR ==="
echo ""

# -------------------------------------------------------------------
# Step 1: Run existing TQ unit tests (fused MHA kernel tests)
# -------------------------------------------------------------------
echo "=== Step 1: Existing TQ fused MHA kernel tests ==="
python -c "
import sys, torch, traceback

sys.path.insert(0, 'python')
from sglang.test.test_turboquant import (
    test_mha_pool_fused_kernel_flag,
    test_fused_kernel_vs_workspace_mha,
    test_fused_kernel_mha_head_configs,
)

tests = [
    test_mha_pool_fused_kernel_flag,
    test_fused_kernel_vs_workspace_mha,
    test_fused_kernel_mha_head_configs,
]

passed = failed = 0
for t in tests:
    try:
        t()
        passed += 1
    except Exception as e:
        print(f'FAIL: {t.__name__}: {e}')
        traceback.print_exc()
        failed += 1

print(f'\nUnit tests: {passed} passed, {failed} failed')
if failed:
    sys.exit(1)
" 2>&1 | tee "$OUTDIR/01_unit_tests.log"
echo ""

# -------------------------------------------------------------------
# Step 2: Test fused Hadamard rotation correctness
# -------------------------------------------------------------------
echo "=== Step 2: Fused Hadamard rotation correctness ==="
python -c "
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, 'python')

DEVICE = torch.device('cuda')

# Test rotation matrix correctness
from sglang.srt.layers.quantization.turboquant_kernels import HadamardTransform

for dim in [64, 128, 256]:
    h = HadamardTransform(dim, seed=42, device=DEVICE)

    # Forward rotation matrix
    M_fwd = h.get_fwd_rotation_matrix_bf16()
    x = torch.randn(32, dim, device=DEVICE, dtype=torch.float32)
    ref = h.forward(x.clone())
    # Pad x for matmul
    if dim < h.padded_dim:
        x_padded = F.pad(x, (0, h.padded_dim - dim))
    else:
        x_padded = x
    got = x_padded.to(torch.bfloat16) @ M_fwd
    err = (ref.float() - got.float()).norm() / ref.float().norm()
    status = 'PASS' if err < 0.02 else 'FAIL'
    print(f'  {status}: fwd rotation dim={dim}, rel_err={err:.6f}')
    assert err < 0.02, f'Forward rotation error too high for dim={dim}'

    # Inverse rotation matrix
    M_inv = h.get_inv_rotation_matrix_bf16()
    y = torch.randn(32, h.padded_dim, device=DEVICE, dtype=torch.float32)
    ref_inv = h.inverse(y.clone())
    got_inv_full = y.to(torch.bfloat16) @ M_inv
    got_inv = got_inv_full[..., :dim]
    err_inv = (ref_inv.float() - got_inv.float()).norm() / ref_inv.float().norm()
    status = 'PASS' if err_inv < 0.02 else 'FAIL'
    print(f'  {status}: inv rotation dim={dim}, rel_err={err_inv:.6f}')
    assert err_inv < 0.02, f'Inverse rotation error too high for dim={dim}'

print()

# Test fused kernel path (K-rot + V-inv fused into Triton kernels)
from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant_mha import (
    decode_attention_fwd_tq_mha,
)
from sglang.srt.mem_cache.turboquant_memory_pool import MHATokenToKVPoolTurboQuant

head_dim = 128
v_head_dim = 128
kv_heads = 8
q_heads = 64
batch = 4
N_tokens = 64

pool = MHATokenToKVPoolTurboQuant(
    size=256, page_size=1, dtype=torch.bfloat16,
    head_num=kv_heads, head_dim=head_dim, layer_num=1,
    device='cuda', enable_memory_saver=False,
    bits=4.0, mode='mse', v_head_dim=v_head_dim,
)

class _FL:
    layer_id = 0

loc = torch.arange(N_tokens, device=DEVICE)
cache_k = torch.randn(N_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
cache_v = torch.randn(N_tokens, kv_heads, v_head_dim, device=DEVICE, dtype=torch.bfloat16)
pool.set_kv_buffer(_FL(), loc, cache_k, cache_v)

q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
seq_lens = torch.full((batch,), 16, dtype=torch.int32, device=DEVICE)
kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
kv_indices = torch.arange(N_tokens, dtype=torch.int32, device=DEVICE)

# Reference: workspace-dequant path
key_buf = pool._get_key_buffer(0)
val_buf = pool._get_value_buffer(0)
ref_outputs = []
for b in range(batch):
    start = kv_indptr[b].item()
    end = kv_indptr[b + 1].item()
    token_ids = kv_indices[start:end]
    per_head = []
    for qh in range(q_heads):
        kvh = qh // (q_heads // kv_heads)
        k = key_buf[token_ids, kvh, :].float()
        v = val_buf[token_ids, kvh, :v_head_dim].float()
        scores = q[b, qh].float() @ k.T
        probs = torch.softmax(scores, dim=-1)
        per_head.append(probs @ v)
    ref_outputs.append(torch.stack(per_head))
ref_output = torch.stack(ref_outputs)

# Legacy path (external Hadamard kernels)
q_rot = pool.k_hadamard.forward(q.clone())
num_kv_splits = torch.ones(batch, dtype=torch.int32, device=DEVICE)
padded_v = pool.v_padded_head_dim
o_legacy = torch.zeros(batch, q_heads, padded_v, device=DEVICE, dtype=torch.bfloat16)
decode_attention_fwd_tq_mha(
    q_rot,
    pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
    pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
    pool.k_centroids_scaled, pool.v_centroids_scaled,
    o_legacy, kv_indptr, kv_indices, num_kv_splits, 1, sm_scale=1.0,
)
o_legacy_unrot = pool.v_hadamard.inverse(o_legacy.float())[:, :, :v_head_dim]

cos_legacy = F.cosine_similarity(
    ref_output.flatten(1), o_legacy_unrot.flatten(1), dim=-1
).mean().item()
print(f'  Legacy path vs ref cosine: {cos_legacy:.6f}')

# Fused path (Hadamard inside Triton kernels)
o_fused = torch.zeros(batch, q_heads, v_head_dim, device=DEVICE, dtype=torch.bfloat16)
decode_attention_fwd_tq_mha(
    q.clone().view(batch, q_heads, head_dim),
    pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
    pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
    pool.k_centroids_scaled, pool.v_centroids_scaled,
    o_fused, kv_indptr, kv_indices, num_kv_splits, 1, sm_scale=1.0,
    k_rot_even=pool.k_fwd_rot_even,
    k_rot_odd=pool.k_fwd_rot_odd,
    v_inv_rot=pool.v_inv_rot_matrix,
)

cos_fused = F.cosine_similarity(
    ref_output.flatten(1), o_fused.float().flatten(1), dim=-1
).mean().item()
print(f'  Fused path vs ref cosine:  {cos_fused:.6f}')

# Fused vs legacy should be very close
cos_fused_legacy = F.cosine_similarity(
    o_legacy_unrot.flatten(1), o_fused.float().flatten(1), dim=-1
).mean().item()
print(f'  Fused vs legacy cosine:    {cos_fused_legacy:.6f}')

assert cos_legacy > 0.99, f'Legacy path cosine {cos_legacy:.4f} too low'
assert cos_fused > 0.98, f'Fused path cosine {cos_fused:.4f} too low'
print()
print('PASS: Fused Hadamard rotation correctness')
" 2>&1 | tee "$OUTDIR/02_fused_hadamard_test.log"
echo ""

# -------------------------------------------------------------------
# Step 3: Micro-benchmark fused vs legacy kernel latency
# -------------------------------------------------------------------
echo "=== Step 3: Micro-benchmark fused vs legacy kernel latency ==="
python -c "
import sys, time, torch
sys.path.insert(0, 'python')

DEVICE = torch.device('cuda')
WARMUP = 50
ITERS = 200

from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant_mha import (
    decode_attention_fwd_tq_mha,
)
from sglang.srt.mem_cache.turboquant_memory_pool import MHATokenToKVPoolTurboQuant

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Warmup={WARMUP}, Iters={ITERS}')
print()

configs = [
    # (batch, ctx_len, kv_heads, q_heads, head_dim)
    (1, 1024, 8, 64, 128),
    (1, 4096, 8, 64, 128),
    (1, 16384, 8, 64, 128),
    (8, 4096, 8, 64, 128),
    (20, 4096, 8, 64, 128),
]

print(f'{\"Batch\":>6} {\"CtxLen\":>8} {\"Legacy(ms)\":>12} {\"Fused(ms)\":>12} {\"Speedup\":>9}')
print('-' * 55)

for batch, ctx_len, kv_heads, q_heads, head_dim in configs:
    total_tokens = batch * ctx_len
    if total_tokens > 512 * 1024:
        print(f'{batch:>6} {ctx_len:>8}    (skipped - too large)')
        continue

    pool = MHATokenToKVPoolTurboQuant(
        size=total_tokens + 64, page_size=1, dtype=torch.bfloat16,
        head_num=kv_heads, head_dim=head_dim, layer_num=1,
        device='cuda', enable_memory_saver=False,
        bits=4.0, mode='mse',
    )

    class _FL:
        layer_id = 0

    loc = torch.arange(total_tokens, device=DEVICE)
    cache_k = torch.randn(total_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    cache_v = torch.randn(total_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    pool.set_kv_buffer(_FL(), loc, cache_k, cache_v)

    q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

    BLOCK_N = 32
    max_kv_splits = 128
    num_kv_splits = torch.clamp(
        (seq_lens + BLOCK_N * 4 - 1) // (BLOCK_N * 4), min=1, max=max_kv_splits,
    ).to(torch.int32)

    padded_v = pool.v_padded_head_dim

    # --- Legacy path: external Hadamard + kernel ---
    def legacy_step():
        q_rot = pool.k_hadamard.forward(q.clone())
        o = torch.zeros(batch, q_heads, padded_v, device=DEVICE, dtype=torch.bfloat16)
        decode_attention_fwd_tq_mha(
            q_rot, pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
            pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
            pool.k_centroids_scaled, pool.v_centroids_scaled,
            o, kv_indptr, kv_indices, num_kv_splits, max_kv_splits, sm_scale=0.1,
        )
        pool.v_hadamard.inverse(o)

    for _ in range(WARMUP):
        legacy_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        legacy_step()
    torch.cuda.synchronize()
    legacy_ms = (time.perf_counter() - t0) / ITERS * 1000

    # --- Fused path: Hadamard inside Triton kernels ---
    def fused_step():
        o = torch.zeros(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        decode_attention_fwd_tq_mha(
            q.clone().view(batch, q_heads, head_dim),
            pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
            pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
            pool.k_centroids_scaled, pool.v_centroids_scaled,
            o, kv_indptr, kv_indices, num_kv_splits, max_kv_splits, sm_scale=0.1,
            k_rot_even=pool.k_fwd_rot_even,
            k_rot_odd=pool.k_fwd_rot_odd,
            v_inv_rot=pool.v_inv_rot_matrix,
        )

    for _ in range(WARMUP):
        fused_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_step()
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / ITERS * 1000

    speedup = legacy_ms / fused_ms if fused_ms > 0 else float('inf')
    print(f'{batch:>6} {ctx_len:>8} {legacy_ms:>11.3f} {fused_ms:>11.3f} {speedup:>8.2f}x')
    torch.cuda.empty_cache()
" 2>&1 | tee "$OUTDIR/03_micro_benchmark.log"
echo ""

# -------------------------------------------------------------------
# Step 4: Per-component profiling (SGLANG_TQ_PROFILE)
# -------------------------------------------------------------------
echo "=== Step 4: Per-component profiling ==="
python -c "
import sys, os, torch, time
sys.path.insert(0, 'python')

DEVICE = torch.device('cuda')

# Enable profiling
os.environ['SGLANG_TQ_PROFILE'] = '1'
os.environ['SGLANG_TQ_PROFILE_INTERVAL'] = '50'

from sglang.srt.layers.attention.triton_ops.decode_attention_turboquant_mha import (
    decode_attention_fwd_tq_mha,
)
from sglang.srt.mem_cache.turboquant_memory_pool import MHATokenToKVPoolTurboQuant
from sglang.srt.layers.quantization.turboquant_kernels import HadamardTransform

# Profile setup
head_dim = 128
kv_heads = 8
q_heads = 64
batch = 1
ctx_len = 4096
total_tokens = batch * ctx_len

pool = MHATokenToKVPoolTurboQuant(
    size=total_tokens + 64, page_size=1, dtype=torch.bfloat16,
    head_num=kv_heads, head_dim=head_dim, layer_num=1,
    device='cuda', enable_memory_saver=False,
    bits=4.0, mode='mse',
)

class _FL:
    layer_id = 0

loc = torch.arange(total_tokens, device=DEVICE)
cache_k = torch.randn(total_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
cache_v = torch.randn(total_tokens, kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
pool.set_kv_buffer(_FL(), loc, cache_k, cache_v)

q = torch.randn(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=DEVICE)
kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
num_kv_splits = torch.clamp(
    (seq_lens + 127) // 128, min=1, max=128
).to(torch.int32)

ITERS = 100
padded_v = pool.v_padded_head_dim

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Config: batch={batch}, ctx_len={ctx_len}, kv_heads={kv_heads}, q_heads={q_heads}')
print()

# Profile legacy path
print('--- Legacy path (5 kernel launches) ---')
events_legacy = []
for i in range(ITERS):
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

    evs[0].record()
    pool.set_kv_buffer(_FL(), loc[:1], cache_k[:1], cache_v[:1])  # set_kv
    evs[1].record()
    q_rot = pool.k_hadamard.forward(q.clone())  # k_hadamard
    evs[2].record()
    o = torch.zeros(batch, q_heads, padded_v, device=DEVICE, dtype=torch.bfloat16)
    decode_attention_fwd_tq_mha(  # stage1 + stage2
        q_rot, pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
        pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
        pool.k_centroids_scaled, pool.v_centroids_scaled,
        o, kv_indptr, kv_indices, num_kv_splits, 128, sm_scale=0.1,
    )
    evs[3].record()
    pool.v_hadamard.inverse(o)  # v_hadamard
    evs[4].record()
    events_legacy.append(evs)

torch.cuda.synchronize()
# Aggregate
set_kv_ms = sum(events_legacy[i][0].elapsed_time(events_legacy[i][1]) for i in range(ITERS)) / ITERS
k_had_ms = sum(events_legacy[i][1].elapsed_time(events_legacy[i][2]) for i in range(ITERS)) / ITERS
attn_ms = sum(events_legacy[i][2].elapsed_time(events_legacy[i][3]) for i in range(ITERS)) / ITERS
v_had_ms = sum(events_legacy[i][3].elapsed_time(events_legacy[i][4]) for i in range(ITERS)) / ITERS
total_ms = set_kv_ms + k_had_ms + attn_ms + v_had_ms
print(f'  set_kv:     {set_kv_ms:.4f} ms')
print(f'  k_hadamard: {k_had_ms:.4f} ms')
print(f'  attn_s1s2:  {attn_ms:.4f} ms')
print(f'  v_hadamard: {v_had_ms:.4f} ms')
print(f'  TOTAL:      {total_ms:.4f} ms')
print()

# Profile fused path
print('--- Fused path (3 kernel launches) ---')
events_fused = []
for i in range(ITERS):
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    evs[0].record()
    pool.set_kv_buffer(_FL(), loc[:1], cache_k[:1], cache_v[:1])  # set_kv
    evs[1].record()
    o = torch.zeros(batch, q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    decode_attention_fwd_tq_mha(  # fused stage1+k_rot + fused stage2+v_inv
        q.clone().view(batch, q_heads, head_dim),
        pool.get_k_packed_buffer(0), pool.get_v_packed_buffer(0),
        pool.get_k_norms_buffer(0), pool.get_v_norms_buffer(0),
        pool.k_centroids_scaled, pool.v_centroids_scaled,
        o, kv_indptr, kv_indices, num_kv_splits, 128, sm_scale=0.1,
        k_rot_even=pool.k_fwd_rot_even,
        k_rot_odd=pool.k_fwd_rot_odd,
        v_inv_rot=pool.v_inv_rot_matrix,
    )
    evs[2].record()
    events_fused.append(evs)

torch.cuda.synchronize()
set_kv_fused = sum(events_fused[i][0].elapsed_time(events_fused[i][1]) for i in range(ITERS)) / ITERS
attn_fused = sum(events_fused[i][1].elapsed_time(events_fused[i][2]) for i in range(ITERS)) / ITERS
total_fused = set_kv_fused + attn_fused
print(f'  set_kv:          {set_kv_fused:.4f} ms')
print(f'  attn+rot (fused): {attn_fused:.4f} ms')
print(f'  TOTAL:           {total_fused:.4f} ms')
print()

savings = total_ms - total_fused
print(f'--- Savings: {savings:.4f} ms/layer ({savings/total_ms*100:.1f}%) ---')
print(f'--- Projected 28-layer savings: {savings*28:.2f} ms ---')
" 2>&1 | tee "$OUTDIR/04_per_component_profile.log"
echo ""

echo "=== All results saved to $OUTDIR ==="
echo ""
echo "Files:"
ls -la "$OUTDIR/"
