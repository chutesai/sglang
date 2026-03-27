"""
Utility to compute a rotated w_vc for TurboQuant fused decode.

The fused decode kernel accumulates V in the Hadamard-rotated space:
    o_rot = sum_i p_i * (R @ v_i) = R @ (sum_i p_i * v_i) = R @ o_true

To recover the correct projection:
    o_rot @ w_vc_rot = (o_true @ R^T) @ (R @ w_vc) = o_true @ w_vc

where R^T @ R = I (orthogonal).

This module computes w_vc_rot = R @ w_vc once at model load time.
"""

import math

import torch


def compute_rotated_wvc(
    w_vc: torch.Tensor,
    hadamard_transform,
    w_scale=None,
) -> torch.Tensor:
    """
    Compute w_vc_tq_rotated = R @ w_vc where R is the nope Hadamard transform.

    w_vc shape: (heads, kv_lora_rank=512, v_head_dim=128)
    R operates on the kv_lora_rank dimension (dim 1).

    Args:
        w_vc: Weight matrix, shape (heads, kv_lora_rank, v_head_dim).
              May be bf16, fp8, or uint8.
        hadamard_transform: HadamardTransform instance for nope.
        w_scale: Optional scale factor for fp8 weights.

    Returns:
        A NEW bf16 tensor of shape (heads, kv_lora_rank, v_head_dim).
        The original w_vc is NOT modified.
        Returns None if w_vc is uint8 (quark) — cannot rotate quantized weights.
    """
    if w_vc.dtype == torch.uint8:
        # Quark quantized weights cannot be rotated without loss
        return None

    # Dequant fp8 to bf16 if needed
    _fp8_dtypes = (torch.float8_e4m3fn,)
    if hasattr(torch, "float8_e4m3fnuz"):
        _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if w_vc.dtype in _fp8_dtypes:
        w_vc_f = w_vc.to(torch.bfloat16)
        if w_scale is not None:
            w_vc_f = w_vc_f * w_scale
    else:
        w_vc_f = w_vc.to(torch.bfloat16)

    # w_vc_f: (heads, kv_lora_rank, v_head_dim)
    # HadamardTransform.forward operates on the LAST dimension.
    # We want to apply R along kv_lora_rank (dim 1).
    # Transpose → apply → transpose back.
    w_vc_t = w_vc_f.transpose(1, 2)  # (heads, v_head_dim, kv_lora_rank)
    w_vc_t_rot = hadamard_transform.forward(w_vc_t)  # R applied along last dim
    w_vc_rot = w_vc_t_rot.transpose(1, 2).contiguous()  # (heads, kv_lora_rank, v_head_dim)

    return w_vc_rot.to(torch.bfloat16)
