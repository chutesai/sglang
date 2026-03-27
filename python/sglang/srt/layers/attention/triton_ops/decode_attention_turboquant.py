"""
Fused dequant-attention Triton decode kernel for TurboQuant MLA.

Reads compressed 4-bit uint8 buffers directly during attention, eliminating the
full-buffer dequant workspace.  Two key mathematical insights:

K-side: Hadamard R is orthogonal → <Rq, Rk> = <q, k>.  Rotate Q once (O(d))
        instead of inverse-rotating every K token (O(context_len × d)).

V-side: The kernel accumulates Σ p_i (R v_i) which is in rotated space.
        Downstream uses a separate w_vc_tq_rotated = R @ w_vc to cancel out.

Scope: 4-bit uniform MSE mode only (the primary production config).

Based on _fwd_grouped_kernel_stage1 from decode_attention.py.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _MIN_BLOCK_KV,
    _fwd_kernel_stage2,
    tanh,
)

# ---------------------------------------------------------------------------
# Stage 1: Fused dequant-attention kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_grouped_kernel_stage1_tq(
    # Q inputs (already Hadamard-rotated by caller)
    Q_Nope,  # (batch, heads, kv_lora_rank) — rotated Q nope
    Q_Rope,  # (batch, heads, qk_rope_head_dim) — rotated Q rope
    # Compressed KV buffers
    Nope_Packed,  # uint8, (max_tokens, 1, nope_packed_dim)
    Rope_Packed,  # uint8, (max_tokens, 1, rope_packed_dim)
    Nope_Norms,  # float32, (max_tokens, 1)
    Rope_Norms,  # float32, (max_tokens, 1)
    # Pre-scaled centroid tables: raw_centroids / sqrt(dim)
    Nope_Centroids,  # float32, (16,) — scaled by 1/√nope_dim
    Rope_Centroids,  # float32, (16,) — scaled by 1/√rope_dim
    # Attention params
    sm_scale,
    kv_indptr,
    kv_indices,
    # Outputs
    Att_Out,
    Att_Lse,
    num_kv_splits,
    # Strides for Q
    stride_qn_bs,
    stride_qn_h,
    stride_qr_bs,
    stride_qr_h,
    # Strides for compressed buffers (dim 0 = tokens)
    stride_nope_p_bs,  # nope_packed stride along tokens
    stride_rope_p_bs,  # rope_packed stride along tokens
    stride_nope_n_bs,  # nope_norms stride along tokens
    stride_rope_n_bs,  # rope_norms stride along tokens
    # Strides for output
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Constexprs
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    NOPE_PACKED_DIM: tl.constexpr,  # 256 for 4-bit nope (512/2)
    ROPE_PACKED_DIM: tl.constexpr,  # 32 for 4-bit rope (64/2)
    BLOCK_DV: tl.constexpr,  # 512 (kv_lora_rank = nope dim)
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
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
    acc_even = tl.zeros([BLOCK_H, NOPE_PACKED_DIM], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, NOPE_PACKED_DIM], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Load Q nope (even/odd for nibble-packed K layout)
        offs_nope_even = tl.arange(0, NOPE_PACKED_DIM) * 2
        offs_nope_odd = tl.arange(0, NOPE_PACKED_DIM) * 2 + 1

        offs_qn_even = (
            cur_batch * stride_qn_bs
            + cur_head[:, None] * stride_qn_h
            + offs_nope_even[None, :]
        )
        offs_qn_odd = (
            cur_batch * stride_qn_bs
            + cur_head[:, None] * stride_qn_h
            + offs_nope_odd[None, :]
        )
        q_nope_even = tl.load(
            Q_Nope + offs_qn_even, mask=mask_h[:, None], other=0.0
        ).to(tl.bfloat16)
        q_nope_odd = tl.load(Q_Nope + offs_qn_odd, mask=mask_h[:, None], other=0.0).to(
            tl.bfloat16
        )

        # Load Q rope (even/odd)
        offs_rope_even = tl.arange(0, ROPE_PACKED_DIM) * 2
        offs_rope_odd = tl.arange(0, ROPE_PACKED_DIM) * 2 + 1

        offs_qr_even = (
            cur_batch * stride_qr_bs
            + cur_head[:, None] * stride_qr_h
            + offs_rope_even[None, :]
        )
        offs_qr_odd = (
            cur_batch * stride_qr_bs
            + cur_head[:, None] * stride_qr_h
            + offs_rope_odd[None, :]
        )
        q_rope_even = tl.load(
            Q_Rope + offs_qr_even, mask=mask_h[:, None], other=0.0
        ).to(tl.bfloat16)
        q_rope_odd = tl.load(Q_Rope + offs_qr_odd, mask=mask_h[:, None], other=0.0).to(
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
            # Dequant K nope: load packed bytes, nibble unpack, centroid lookup
            # ----------------------------------------------------------
            offs_nope_packed = tl.arange(0, NOPE_PACKED_DIM)
            offs_buf_nope = (
                kv_loc[:, None] * stride_nope_p_bs + offs_nope_packed[None, :]
            )
            nope_bytes = tl.load(
                Nope_Packed + offs_buf_nope,
                mask=(offs_n[:, None] < split_kv_end),
                other=0,
            ).to(tl.int32)

            lo_idx = nope_bytes & 0x0F
            hi_idx = (nope_bytes >> 4) & 0x0F

            # Centroid gather (pre-scaled by 1/sqrt(nope_dim))
            k_nope_lo = tl.load(Nope_Centroids + lo_idx)  # (BLOCK_N, NOPE_PACKED_DIM)
            k_nope_hi = tl.load(Nope_Centroids + hi_idx)

            # Scale by per-token norm
            nope_norm = tl.load(
                Nope_Norms + kv_loc * stride_nope_n_bs,
                mask=offs_n < split_kv_end,
                other=0.0,
            )
            k_nope_lo = k_nope_lo * nope_norm[:, None]
            k_nope_hi = k_nope_hi * nope_norm[:, None]

            # Cast to bf16 for tensor-core dot
            k_nope_lo_bf = k_nope_lo.to(tl.bfloat16)
            k_nope_hi_bf = k_nope_hi.to(tl.bfloat16)

            # K-nope dot: qk = q_even @ k_lo^T + q_odd @ k_hi^T
            qk = tl.dot(q_nope_even, tl.trans(k_nope_lo_bf))
            qk += tl.dot(q_nope_odd, tl.trans(k_nope_hi_bf))

            # ----------------------------------------------------------
            # Dequant K rope: same process
            # ----------------------------------------------------------
            offs_rope_packed = tl.arange(0, ROPE_PACKED_DIM)
            offs_buf_rope = (
                kv_loc[:, None] * stride_rope_p_bs + offs_rope_packed[None, :]
            )
            rope_bytes = tl.load(
                Rope_Packed + offs_buf_rope,
                mask=(offs_n[:, None] < split_kv_end),
                other=0,
            ).to(tl.int32)

            rope_lo_idx = rope_bytes & 0x0F
            rope_hi_idx = (rope_bytes >> 4) & 0x0F

            k_rope_lo = tl.load(Rope_Centroids + rope_lo_idx)
            k_rope_hi = tl.load(Rope_Centroids + rope_hi_idx)

            rope_norm = tl.load(
                Rope_Norms + kv_loc * stride_rope_n_bs,
                mask=offs_n < split_kv_end,
                other=0.0,
            )
            k_rope_lo = k_rope_lo * rope_norm[:, None]
            k_rope_hi = k_rope_hi * rope_norm[:, None]

            k_rope_lo_bf = k_rope_lo.to(tl.bfloat16)
            k_rope_hi_bf = k_rope_hi.to(tl.bfloat16)

            qk += tl.dot(q_rope_even, tl.trans(k_rope_lo_bf))
            qk += tl.dot(q_rope_odd, tl.trans(k_rope_hi_bf))

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
            # Online softmax + V accumulation
            # V = nope in MLA, so reuse k_nope_lo/hi (already dequanted)
            # ----------------------------------------------------------
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            acc_even *= re_scale[:, None]
            acc_odd *= re_scale[:, None]

            # V accumulation: p @ v_lo, p @ v_hi
            acc_even += tl.dot(p.to(tl.bfloat16), k_nope_lo_bf)
            acc_odd += tl.dot(p.to(tl.bfloat16), k_nope_hi_bf)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # ----------------------------------------------------------
        # Store results: interleave even/odd into standard layout
        # ----------------------------------------------------------
        result_even = acc_even / e_sum[:, None]
        result_odd = acc_odd / e_sum[:, None]

        # Store even positions (dims 0, 2, 4, ...)
        offs_dv_even = tl.arange(0, NOPE_PACKED_DIM) * 2
        offs_mid_o_even = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv_even[None, :]
        )
        tl.store(Att_Out + offs_mid_o_even, result_even, mask=mask_h[:, None])

        # Store odd positions (dims 1, 3, 5, ...)
        offs_dv_odd = tl.arange(0, NOPE_PACKED_DIM) * 2 + 1
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
# Python wrapper — launches stage1 + stage2
# ---------------------------------------------------------------------------


def decode_attention_fwd_tq(
    q_nope: torch.Tensor,  # (batch, heads, kv_lora_rank) — already Hadamard-rotated
    q_rope: torch.Tensor,  # (batch, heads, qk_rope_head_dim) — already Hadamard-rotated
    nope_packed: torch.Tensor,  # (max_tokens, 1, nope_packed_dim) uint8
    rope_packed: torch.Tensor,  # (max_tokens, 1, rope_packed_dim) uint8
    nope_norms: torch.Tensor,  # (max_tokens, 1) float32
    rope_norms: torch.Tensor,  # (max_tokens, 1) float32
    nope_centroids: torch.Tensor,  # (16,) float32, pre-scaled by 1/√nope_dim
    rope_centroids: torch.Tensor,  # (16,) float32, pre-scaled by 1/√rope_dim
    o: torch.Tensor,  # (batch, heads, kv_lora_rank) output
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    attn_logits: torch.Tensor = None,  # pre-allocated (batch, heads, max_kv_splits, kv_lora_rank) f32
    attn_lse: torch.Tensor = None,  # pre-allocated (batch, heads, max_kv_splits) f32
):
    """Launch the fused TurboQuant dequant-attention decode kernel."""
    batch = q_nope.shape[0]
    head_num = q_nope.shape[1]
    kv_lora_rank = q_nope.shape[2]
    nope_packed_dim = nope_packed.shape[2]
    rope_packed_dim = rope_packed.shape[2]

    # MLA: all Q heads share 1 KV head
    kv_group_num = head_num  # kv_heads = 1 for MLA

    BLOCK_N = 32
    BLOCK_H = 16
    BLOCK_DV = kv_lora_rank  # 512

    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    # Use pre-allocated buffers if provided, otherwise allocate
    if attn_logits is not None:
        attn_logits = attn_logits[:batch, :head_num, :MAX_KV_SPLITS, :BLOCK_DV]
        attn_logits.zero_()
    else:
        attn_logits = torch.zeros(
            (batch, head_num, MAX_KV_SPLITS, BLOCK_DV),
            dtype=torch.float32,
            device=q_nope.device,
        )
    if attn_lse is not None:
        attn_lse = attn_lse[:batch, :head_num, :MAX_KV_SPLITS]
        attn_lse.zero_()
    else:
        attn_lse = torch.zeros(
            (batch, head_num, MAX_KV_SPLITS),
            dtype=torch.float32,
            device=q_nope.device,
        )

    # Flatten packed buffers: remove the head=1 dim for simpler striding
    # nope_packed: (max_tokens, 1, nope_packed_dim) → stride[0] = 1 * nope_packed_dim
    # We pass the base pointer with the head dim folded in
    nope_packed_flat = nope_packed.view(-1, nope_packed_dim)
    rope_packed_flat = rope_packed.view(-1, rope_packed_dim)
    nope_norms_flat = nope_norms.view(-1)
    rope_norms_flat = rope_norms.view(-1)

    _fwd_grouped_kernel_stage1_tq[grid](
        q_nope,
        q_rope,
        nope_packed_flat,
        rope_packed_flat,
        nope_norms_flat,
        rope_norms_flat,
        nope_centroids,
        rope_centroids,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        # Q strides
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        # Compressed buffer strides (flattened: token stride)
        nope_packed_flat.stride(0),
        rope_packed_flat.stride(0),
        nope_norms_flat.stride(0),
        rope_norms_flat.stride(0),
        # Output strides
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        # Constexprs
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        NOPE_PACKED_DIM=nope_packed_dim,
        ROPE_PACKED_DIM=rope_packed_dim,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2: reduce across KV splits (reuse existing kernel)
    Lv = BLOCK_DV
    BLOCK_DV_S2 = triton.next_power_of_2(Lv)
    grid_s2 = (batch, head_num)

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
