"""
Fused dequant-attention Triton decode kernel for TurboQuant MHA/GQA.

Reads compressed 4-bit uint8 K and V buffers directly during attention,
eliminating the full-buffer dequant workspace.  Two key insights:

K-side: Hadamard R_k is orthogonal => <R_k(q), R_k(k)> = <q, k>.
        Rotate Q once (O(d)) instead of dequanting every K token.

V-side: The kernel accumulates sum(p_i * R_v(v_i)) = R_v(sum(p_i * v_i)).
        Caller applies R_v^{-1} to recover the true attention output.

Key difference from MLA kernel: K and V are separate buffers with separate
Hadamard transforms, separate centroids, and multiple KV heads.

Scope: 4-bit uniform MSE mode only (matches MLA fused kernel scope).

Based on _fwd_grouped_kernel_stage1_tq from decode_attention_turboquant.py.

Fused Hadamard rotation support:
  - FUSE_K_ROT=True: K-Hadamard rotation is fused into Stage 1 via
    pre-computed dense rotation matrices (eliminates external kernel launch).
  - Grouped Stage 2 with V-inverse rotation: eliminates external V-Hadamard
    inverse kernel launch.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _fwd_kernel_stage2,
    tanh,
)

# ---------------------------------------------------------------------------
# Tuning configuration for Stage 1 kernel.
# Set SGLANG_TQ_DECODE_CONFIG=<BLOCK_N>/<num_warps>/<num_stages> to override.
# Example: SGLANG_TQ_DECODE_CONFIG=64/8/3
# Default: 32/4/2 (current production config)
# ---------------------------------------------------------------------------
_TQ_DECODE_CONFIG = os.environ.get("SGLANG_TQ_DECODE_CONFIG", "32/4/2")
_TQ_BLOCK_N, _TQ_NUM_WARPS, _TQ_NUM_STAGES = (
    int(x) for x in _TQ_DECODE_CONFIG.split("/")
)

# ---------------------------------------------------------------------------
# Stage 1: Fused dequant-attention kernel for MHA/GQA
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_grouped_kernel_stage1_tq_mha(
    # Q input — either pre-rotated or raw (when FUSE_K_ROT=True)
    Q,  # (batch, q_heads, head_dim)
    # Compressed KV buffers — separate K and V
    K_Packed,  # uint8, (max_tokens, kv_heads, k_packed_dim)
    V_Packed,  # uint8, (max_tokens, kv_heads, v_packed_dim)
    K_Norms,  # float32, (max_tokens, kv_heads)
    V_Norms,  # float32, (max_tokens, kv_heads)
    # Pre-scaled centroid tables: raw_centroids / sqrt(padded_dim)
    K_Centroids,  # float32, (16,) — scaled by 1/sqrt(k_padded_dim)
    V_Centroids,  # float32, (16,) — scaled by 1/sqrt(v_padded_dim)
    # Attention params
    sm_scale,
    kv_indptr,
    kv_indices,
    # Outputs
    Att_Out,
    Att_Lse,
    num_kv_splits,
    # K-Hadamard rotation matrices (used when FUSE_K_ROT=True)
    K_Rot_Even,  # bf16, (padded_dim, k_packed_dim) — even columns of rotation matrix
    K_Rot_Odd,  # bf16, (padded_dim, k_packed_dim) — odd columns of rotation matrix
    # Strides for Q
    stride_q_bs,
    stride_q_h,
    # Strides for K packed (tokens, kv_heads, packed_dim)
    stride_kp_bs,
    stride_kp_h,
    # Strides for V packed (tokens, kv_heads, packed_dim)
    stride_vp_bs,
    stride_vp_h,
    # Strides for K norms (tokens, kv_heads)
    stride_kn_bs,
    stride_kn_h,
    # Strides for V norms (tokens, kv_heads)
    stride_vn_bs,
    stride_vn_h,
    # Strides for output
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Constexprs
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    K_PACKED_DIM: tl.constexpr,  # head_dim // 2 for 4-bit
    V_PACKED_DIM: tl.constexpr,  # v_head_dim // 2 for 4-bit
    BLOCK_DV: tl.constexpr,  # v_head_dim (output dim)
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    FUSE_K_ROT: tl.constexpr,  # fuse K-Hadamard rotation into this kernel
    PADDED_DIM: tl.constexpr,  # padded head dim (power of 2), used when FUSE_K_ROT
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # V accumulation split into even/odd halves (matches nibble packing layout)
    acc_even = tl.zeros([BLOCK_H, V_PACKED_DIM], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, V_PACKED_DIM], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        if FUSE_K_ROT:
            # ---- Fused K-Hadamard: load raw Q, rotate via dense matmul ----
            # Load full Q: (BLOCK_H, PADDED_DIM)
            offs_full = tl.arange(0, PADDED_DIM)
            q_full = tl.load(
                Q
                + cur_batch * stride_q_bs
                + cur_head[:, None] * stride_q_h
                + offs_full[None, :],
                mask=mask_h[:, None],
                other=0.0,
            ).to(tl.bfloat16)

            # Load rotation matrix halves: (PADDED_DIM, K_PACKED_DIM) each
            offs_rot_k = tl.arange(0, K_PACKED_DIM)
            rot_even = tl.load(
                K_Rot_Even + offs_full[:, None] * K_PACKED_DIM + offs_rot_k[None, :]
            ).to(tl.bfloat16)
            rot_odd = tl.load(
                K_Rot_Odd + offs_full[:, None] * K_PACKED_DIM + offs_rot_k[None, :]
            ).to(tl.bfloat16)

            # Q_rot_even = Q @ Rot_Even, Q_rot_odd = Q @ Rot_Odd
            q_even = tl.dot(q_full, rot_even).to(tl.bfloat16)
            q_odd = tl.dot(q_full, rot_odd).to(tl.bfloat16)
        else:
            # ---- Legacy path: Q already K-Hadamard-rotated by caller ----
            offs_k_even = tl.arange(0, K_PACKED_DIM) * 2
            offs_k_odd = tl.arange(0, K_PACKED_DIM) * 2 + 1

            offs_q_even = (
                cur_batch * stride_q_bs
                + cur_head[:, None] * stride_q_h
                + offs_k_even[None, :]
            )
            offs_q_odd = (
                cur_batch * stride_q_bs
                + cur_head[:, None] * stride_q_h
                + offs_k_odd[None, :]
            )
            q_even = tl.load(Q + offs_q_even, mask=mask_h[:, None], other=0.0).to(
                tl.bfloat16
            )
            q_odd = tl.load(Q + offs_q_odd, mask=mask_h[:, None], other=0.0).to(
                tl.bfloat16
            )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # ----------------------------------------------------------
            # Dequant K: load packed bytes, nibble unpack, centroid lookup
            # ----------------------------------------------------------
            offs_k_packed = tl.arange(0, K_PACKED_DIM)
            offs_buf_k = (
                kv_loc[:, None] * stride_kp_bs
                + cur_kv_head * stride_kp_h
                + offs_k_packed[None, :]
            )
            k_bytes = tl.load(
                K_Packed + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end),
                other=0,
            ).to(tl.int32)

            k_lo_idx = k_bytes & 0x0F
            k_hi_idx = (k_bytes >> 4) & 0x0F

            # Centroid gather (pre-scaled by 1/sqrt(k_padded_dim))
            k_lo = tl.load(K_Centroids + k_lo_idx)  # (BLOCK_N, K_PACKED_DIM)
            k_hi = tl.load(K_Centroids + k_hi_idx)

            # Scale by per-token norm
            k_norm = tl.load(
                K_Norms + kv_loc * stride_kn_bs + cur_kv_head * stride_kn_h,
                mask=offs_n < split_kv_end,
                other=0.0,
            )
            k_lo = k_lo * k_norm[:, None]
            k_hi = k_hi * k_norm[:, None]

            # Cast to bf16 for tensor-core dot
            k_lo_bf = k_lo.to(tl.bfloat16)
            k_hi_bf = k_hi.to(tl.bfloat16)

            # K dot: qk = q_even @ k_lo^T + q_odd @ k_hi^T
            qk = tl.dot(q_even, tl.trans(k_lo_bf))
            qk += tl.dot(q_odd, tl.trans(k_hi_bf))

            # ----------------------------------------------------------
            # Attention score processing
            # ----------------------------------------------------------
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            # ----------------------------------------------------------
            # Dequant V: separate buffer, separate centroids
            # ----------------------------------------------------------
            offs_v_packed = tl.arange(0, V_PACKED_DIM)
            offs_buf_v = (
                kv_loc[:, None] * stride_vp_bs
                + cur_kv_head * stride_vp_h
                + offs_v_packed[None, :]
            )
            v_bytes = tl.load(
                V_Packed + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end),
                other=0,
            ).to(tl.int32)

            v_lo_idx = v_bytes & 0x0F
            v_hi_idx = (v_bytes >> 4) & 0x0F

            v_lo = tl.load(V_Centroids + v_lo_idx)  # (BLOCK_N, V_PACKED_DIM)
            v_hi = tl.load(V_Centroids + v_hi_idx)

            v_norm = tl.load(
                V_Norms + kv_loc * stride_vn_bs + cur_kv_head * stride_vn_h,
                mask=offs_n < split_kv_end,
                other=0.0,
            )
            v_lo = v_lo * v_norm[:, None]
            v_hi = v_hi * v_norm[:, None]

            v_lo_bf = v_lo.to(tl.bfloat16)
            v_hi_bf = v_hi.to(tl.bfloat16)

            # ----------------------------------------------------------
            # Online softmax + V accumulation
            # ----------------------------------------------------------
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            acc_even *= re_scale[:, None]
            acc_odd *= re_scale[:, None]

            # V accumulation: p @ v_lo, p @ v_hi
            acc_even += tl.dot(p.to(tl.bfloat16), v_lo_bf)
            acc_odd += tl.dot(p.to(tl.bfloat16), v_hi_bf)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # ----------------------------------------------------------
        # Store results: interleave even/odd into standard layout
        # ----------------------------------------------------------
        result_even = acc_even / e_sum[:, None]
        result_odd = acc_odd / e_sum[:, None]

        # Store even positions (dims 0, 2, 4, ...)
        offs_dv_even = tl.arange(0, V_PACKED_DIM) * 2
        offs_mid_o_even = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_even[None, :]
        )
        tl.store(Att_Out + offs_mid_o_even, result_even, mask=mask_h[:, None])

        # Store odd positions (dims 1, 3, 5, ...)
        offs_dv_odd = tl.arange(0, V_PACKED_DIM) * 2 + 1
        offs_mid_o_odd = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_odd[None, :]
        )
        tl.store(Att_Out + offs_mid_o_odd, result_odd, mask=mask_h[:, None])

        # Store LSE
        offs_mid_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // BLOCK_DV
        tl.store(
            Att_Lse + offs_mid_lse,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# ---------------------------------------------------------------------------
# Stage 2 with fused V-Hadamard inverse rotation (grouped heads)
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_stage2_tq_v_inv(
    Mid_O,
    Mid_O_1,
    O,
    V_Inv_Rot,  # bf16, (v_padded_dim, v_head_dim) — truncated V-inverse rotation
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_vr_row,  # stride for V_Inv_Rot rows
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,  # v_padded_dim (padded, power of 2)
    Lv: tl.constexpr,  # v_head_dim (actual output dim)
    BLOCK_H_S2: tl.constexpr,  # heads per thread block
    q_head_num: tl.constexpr,
):
    """Stage 2: reduce KV splits + apply V-inverse Hadamard rotation.

    Processes BLOCK_H_S2 heads per thread block. After reducing across KV
    splits, applies the dense V-inverse rotation matrix via tl.dot to
    eliminate the external V-Hadamard inverse kernel launch.
    """
    cur_batch = tl.program_id(0)
    head_block_id = tl.program_id(1)

    cur_head = head_block_id * BLOCK_H_S2 + tl.arange(0, BLOCK_H_S2)
    mask_h = cur_head < q_head_num

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < BLOCK_DV  # always True for padded dim

    # Per-head accumulators: (BLOCK_H_S2, BLOCK_DV)
    e_sum = tl.zeros([BLOCK_H_S2], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_H_S2], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_H_S2, BLOCK_DV], dtype=tl.float32)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            # Load mid_o: (BLOCK_H_S2, BLOCK_DV)
            offs_v = (
                cur_batch * stride_mid_ob
                + cur_head[:, None] * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_d[None, :]
            )
            tv = tl.load(Mid_O + offs_v, mask=mask_h[:, None], other=0.0)

            # Load LSE: (BLOCK_H_S2,)
            offs_logic = (
                cur_batch * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // BLOCK_DV
            tlogic = tl.load(Mid_O_1 + offs_logic, mask=mask_h, other=-float("inf"))

            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale[:, None]
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic[:, None] * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Normalize: (BLOCK_H_S2, BLOCK_DV)
    acc = acc / e_sum[:, None]

    # Apply V-inverse Hadamard rotation via dense matmul:
    # result = acc @ V_Inv_Rot  where V_Inv_Rot is (BLOCK_DV, Lv)
    # This fuses the V-Hadamard inverse into Stage 2, eliminating a kernel launch.
    offs_lv = tl.arange(0, Lv)
    rot_mat = tl.load(
        V_Inv_Rot + offs_d[:, None] * stride_vr_row + offs_lv[None, :]
    ).to(tl.bfloat16)
    result = tl.dot(acc.to(tl.bfloat16), rot_mat)  # (BLOCK_H_S2, Lv)

    # Store final output
    offs_out = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_lv[None, :]
    tl.store(O + offs_out, result, mask=mask_h[:, None])


# ---------------------------------------------------------------------------
# Python wrapper — launches stage1 + stage2
# ---------------------------------------------------------------------------


def decode_attention_fwd_tq_mha(
    q: torch.Tensor,  # (batch, q_heads, head_dim) — already K-Hadamard-rotated
    k_packed: torch.Tensor,  # (max_tokens, kv_heads, k_packed_dim) uint8
    v_packed: torch.Tensor,  # (max_tokens, kv_heads, v_packed_dim) uint8
    k_norms: torch.Tensor,  # (max_tokens, kv_heads) float32
    v_norms: torch.Tensor,  # (max_tokens, kv_heads) float32
    k_centroids: torch.Tensor,  # (16,) float32 — raw / sqrt(k_padded_head_dim)
    v_centroids: torch.Tensor,  # (16,) float32 — raw / sqrt(v_padded_head_dim)
    o: torch.Tensor,  # (batch, q_heads, v_head_dim) output
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    attn_logits: torch.Tensor = None,
    attn_lse: torch.Tensor = None,
    # Fused Hadamard rotation matrices (None = legacy path)
    k_rot_even: torch.Tensor = None,  # bf16, (padded_dim, k_packed_dim)
    k_rot_odd: torch.Tensor = None,  # bf16, (padded_dim, k_packed_dim)
    v_inv_rot: torch.Tensor = None,  # bf16, (v_padded_dim, v_head_dim)
):
    """Launch the fused TurboQuant dequant-attention decode kernel for MHA/GQA."""
    batch = q.shape[0]
    q_head_num = q.shape[1]
    head_dim = q.shape[2]
    v_head_dim = o.shape[2]
    kv_head_num = k_packed.shape[1]
    k_packed_dim = k_packed.shape[2]
    v_packed_dim = v_packed.shape[2]

    kv_group_num = q_head_num // kv_head_num

    fuse_k_rot = k_rot_even is not None and k_rot_odd is not None
    fuse_v_inv = v_inv_rot is not None

    BLOCK_N = _TQ_BLOCK_N
    BLOCK_H = triton.next_power_of_2(min(16, kv_group_num))
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    PADDED_DIM = head_dim if not fuse_k_rot else k_rot_even.shape[0]
    NUM_WARPS = _TQ_NUM_WARPS
    NUM_STAGES = _TQ_NUM_STAGES

    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(q_head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    # Use pre-allocated buffers if provided, otherwise allocate
    if attn_logits is not None:
        attn_logits = attn_logits[:batch, :q_head_num, :MAX_KV_SPLITS, :BLOCK_DV]
        attn_logits.zero_()
    else:
        attn_logits = torch.zeros(
            (batch, q_head_num, MAX_KV_SPLITS, BLOCK_DV),
            dtype=torch.float32,
            device=q.device,
        )
    if attn_lse is not None:
        attn_lse = attn_lse[:batch, :q_head_num, :MAX_KV_SPLITS]
        attn_lse.zero_()
    else:
        attn_lse = torch.zeros(
            (batch, q_head_num, MAX_KV_SPLITS),
            dtype=torch.float32,
            device=q.device,
        )

    _fwd_grouped_kernel_stage1_tq_mha[grid](
        q,
        k_packed,
        v_packed,
        k_norms,
        v_norms,
        k_centroids,
        v_centroids,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        # K-rotation matrices (None when not fusing)
        k_rot_even if fuse_k_rot else k_packed,  # dummy ptr when not used
        k_rot_odd if fuse_k_rot else k_packed,  # dummy ptr when not used
        # Q strides
        q.stride(0),
        q.stride(1),
        # K packed strides
        k_packed.stride(0),
        k_packed.stride(1),
        # V packed strides
        v_packed.stride(0),
        v_packed.stride(1),
        # K norms strides
        k_norms.stride(0),
        k_norms.stride(1),
        # V norms strides
        v_norms.stride(0),
        v_norms.stride(1),
        # Output strides
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        # Constexprs
        kv_group_num=kv_group_num,
        q_head_num=q_head_num,
        K_PACKED_DIM=k_packed_dim,
        V_PACKED_DIM=v_packed_dim,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        FUSE_K_ROT=fuse_k_rot,
        PADDED_DIM=PADDED_DIM,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Stage 2: reduce across KV splits
    Lv = v_head_dim
    BLOCK_DV_S2 = triton.next_power_of_2(Lv)

    if fuse_v_inv:
        # Grouped Stage 2 with fused V-inverse Hadamard rotation.
        # tl.dot requires all dims to be multiples of 16; fall back to legacy
        # if v_head_dim is not a power of 2 (rare in practice).
        assert Lv == BLOCK_DV_S2, (
            f"Fused V-inverse requires v_head_dim to be a power of 2, "
            f"got {Lv} (padded to {BLOCK_DV_S2})"
        )
        BLOCK_H_S2 = triton.next_power_of_2(min(16, q_head_num))
        grid_s2 = (batch, triton.cdiv(q_head_num, BLOCK_H_S2))

        _fwd_kernel_stage2_tq_v_inv[grid_s2](
            attn_logits,
            attn_lse,
            o,
            v_inv_rot,
            kv_indptr,
            num_kv_splits,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            o.stride(0),
            o.stride(1),
            v_inv_rot.stride(0),
            MAX_KV_SPLITS=MAX_KV_SPLITS,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            BLOCK_DV=BLOCK_DV_S2,
            Lv=Lv,
            BLOCK_H_S2=BLOCK_H_S2,
            q_head_num=q_head_num,
            num_warps=4,
            num_stages=2,
        )
    else:
        # Legacy Stage 2: one head per thread block, no V-inverse
        grid_s2 = (batch, q_head_num)

        _fwd_kernel_stage2[grid_s2](
            attn_logits,
            attn_lse,
            o,
            1.0,  # v_scale = 1.0 (no FP8 scaling)
            kv_indptr,
            num_kv_splits,
            None,  # no sinks
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            o.stride(0),
            o.stride(1),
            MAX_KV_SPLITS=MAX_KV_SPLITS,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            BLOCK_DV=BLOCK_DV_S2,
            Lv=Lv,
            HAS_SINK=False,
            num_warps=4,
            num_stages=2,
        )

    return o
