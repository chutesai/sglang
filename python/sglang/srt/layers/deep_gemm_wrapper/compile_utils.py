import gc
import logging
import os
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto
from typing import Dict, List, Tuple

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div, get_available_gpu_memory, is_musa

logger = logging.getLogger(__name__)

_is_musa = is_musa()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()

# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:
        # In fast warmup mode, only compile a small set of typical Ms

        # First cover all the small bs to ensure decode performance
        _BUILTIN_M_LIST += list(range(1, 1025))

        # Then cover larger batch sizes with gradually increasing steps
        # For example, when chunekd prefill size is 16384
        # The sampled Ms would be:
        #   1024, 1026, ... 2046 (step 2)
        #   2048, 2052, ... 4092 (step 4)
        #   4096, 5004, ... 8184 (step 8)
        #   8192, 9008, ... 16384 (step 16)
        # Totally 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072 kernels
        next_m, sample_step = 1024, 2
        max_prefill_bs = (
            min(server_args.chunked_prefill_size, 32 * 1024)
            if server_args.chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:
        # When fast warmup isn't enabled, generate m_max and compile all the covered Ms.
        m_max = 1024 * 16
        if server_args.chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif server_args.chunked_prefill_size > 8192:
            m_max = server_args.chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


# TODO improve code
def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        # TODO maybe improve logs
        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


# Map SGLang kernel type enum to DeepGEMM warmup kernel name string
_KERNEL_NAME_MAP = {
    DeepGemmKernelType.GEMM_NT_F8F8BF16: "fp8_gemm_nt",
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: "m_grouped_fp8_gemm_nt_masked",
    DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: "m_grouped_fp8_gemm_nt_contiguous",
    DeepGemmKernelType.GEMM_NT_BF16BF16F32: "bf16_gemm_nt",
}


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))

        if hasattr(deep_gemm, "warmup_kernels"):
            kernel_name = _KERNEL_NAME_MAP[kernel_type]
            num_unique = deep_gemm.warmup_kernels(kernel_name, m_list, n, k, num_groups)
            logger.info(
                f"Compiled {num_unique} unique kernels for {kernel_name} N={n} K={k}"
            )
        else:
            logger.warning(
                "deep_gemm.warmup_kernels not available, "
                "falling back to legacy per-M warmup. "
                "Update DeepGEMM for faster warmup."
            )
            _compile_deep_gemm_legacy(kernel_type, n, k, num_groups, m_list)
    finally:
        # Restore symmetric memory context
        restore_symmetric_memory_context(saved_context)


_BLOCK_SIZE = 128


def _empty_token_fp8(size):
    import torch

    from sglang.srt.utils import ceil_div

    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    import torch

    from sglang.srt.utils import ceil_div

    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _compile_deep_gemm_legacy(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    """Legacy per-M warmup for DeepGEMM versions without warmup_kernels API."""
    import torch
    from tqdm import tqdm

    from sglang.srt.utils import get_available_gpu_memory

    # Determine max_m and check memory budget
    max_m = max(m_list)
    memory_budget = get_available_gpu_memory(device="cuda", gpu_id=0)
    _GB = 1 << 30

    if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
        required_memory = (max_m * k + n * k + max_m * n * 2) / _GB
    elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
        required_memory = (
            max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2
        ) / _GB
    elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
        required_memory = (
            num_groups * max_m * k
            + num_groups * n * k
            + num_groups * 4
            + num_groups * max_m * n * 2
        ) / _GB
    elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:
        required_memory = (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
    else:
        raise ValueError(f"Invalid kernel type: {kernel_type}")

    logger.info(
        f"Required memory for warmup: {required_memory:.1f}GB, Available memory: {memory_budget:.1f}GB"
    )
    if memory_budget < required_memory:
        while max_m > 4096:
            max_m = max_m // 2
            if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
                req = (max_m * k + n * k + max_m * n * 2) / _GB
            elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
                req = (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
            elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
                req = (
                    num_groups * max_m * k
                    + num_groups * n * k
                    + num_groups * 4
                    + num_groups * max_m * n * 2
                ) / _GB
            else:
                req = (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
            if req <= memory_budget:
                break
        logger.warning(
            f"Available memory {memory_budget:.1f}GB is less than required memory "
            f"{required_memory:.1f}GB for warmup, reducing max_m to {max_m}"
        )
        m_list = [m for m in m_list if m <= max_m]

    # Create executor and pre-allocate tensors
    if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
        lhs_q, lhs_s = _empty_token_fp8((max_m, k))
        rhs_q, rhs_s = _empty_block_fp8((n, k))
        out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

        def execute(m):
            deep_gemm.fp8_gemm_nt((lhs_q[:m], lhs_s[:m]), (rhs_q, rhs_s), out[:m])

    elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
        lhs_q, lhs_s = _empty_token_fp8((max_m, k))
        rhs_q, rhs_s = _empty_block_fp8((num_groups, n, k))
        m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

        def execute(m):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (lhs_q[:m], lhs_s[:m]),
                (rhs_q, rhs_s),
                out[:m],
                m_indices=m_indices[:m],
            )

    elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
        lhs_q, lhs_s = _empty_token_fp8((num_groups, max_m, k))
        rhs_q, rhs_s = _empty_block_fp8((num_groups, n, k))
        masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        out = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)

        def execute(m):
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                (lhs_q, lhs_s),
                (rhs_q, rhs_s),
                out,
                masked_m=masked_m,
                expected_m=m,
            )

    elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:
        lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)

        def execute(m):
            deep_gemm.bf16_gemm_nt(lhs[:m], rhs, out[:m])

    else:
        raise ValueError(f"Invalid kernel type: {kernel_type}")

    old_compile_mode = deep_gemm.get_compile_mode()
    deep_gemm.set_compile_mode(1)
    for m in tqdm(m_list, desc="DeepGEMM warmup"):
        execute(m=m)
    deep_gemm.set_compile_mode(old_compile_mode)

    torch.cuda.current_stream().synchronize()
    # Deleting execute drops the closure which holds refs to all warmup
    # tensors (lhs_q, lhs_s, rhs_q, rhs_s, out, m_indices, etc.).
    del execute
    gc.collect()
    torch.cuda.empty_cache()


def _shape_type_to_kernel_type(shape_type: str) -> DeepGemmKernelType:
    return {
        "MASKED": DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED,
        "CONTIG": DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
        "NORMAL": DeepGemmKernelType.GEMM_NT_F8F8BF16,
    }[shape_type]


def precompile_deep_gemm_shapes(hf_config, tp_size: int, server_args) -> None:
    """Precompile all DeepGEMM kernels at startup from HF model config.

    Called during ModelRunner.__init__() before any forward passes, so each TP
    rank compiles independently without NCCL collectives blocking.
    """
    if not _ENABLE_JIT_DEEPGEMM_PRECOMPILE or not _DO_COMPILE_ALL:
        return

    config = {}
    for key in [
        "hidden_size",
        "num_attention_heads",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "q_lora_rank",
        "n_routed_experts",
        "n_shared_experts",
        "moe_intermediate_size",
        "intermediate_size",
        "first_k_dense_replace",
    ]:
        val = getattr(hf_config, key, None)
        if val is not None:
            config[key] = val

    if "hidden_size" not in config:
        return

    has_mla = config.get("kv_lora_rank", 0) > 0
    has_moe = config.get("n_routed_experts", 0) > 0
    if not has_mla and not has_moe:
        return

    # Compute effective attention TP size (accounts for dp_attention)
    dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
    attn_tp_size = tp_size // dp_size

    try:
        shapes = _compute_deepseek_shapes(config, tp_size, attn_tp_size)
    except Exception as e:
        logger.warning(
            f"Failed to derive DeepGEMM shapes from config, skipping precompilation: {e}"
        )
        return

    if not shapes:
        return

    logger.info(
        f"Precompiling DeepGEMM kernels for {len(shapes)} shapes "
        f"(tp={tp_size}, attn_tp={attn_tp_size}, {len(_BUILTIN_M_LIST)} M values)"
    )

    for shape_type, n, k, num_groups in shapes:
        kernel_type = _shape_type_to_kernel_type(shape_type)
        query_key = (kernel_type, n, k, num_groups)
        if _INITIALIZATION_DICT.get(query_key) is not None:
            continue
        _INITIALIZATION_DICT[query_key] = True

        logger.info(
            f"Precompiling <{kernel_type.name}> N={n}, K={k}, num_groups={num_groups}"
        )
        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )

    logger.info("DeepGEMM precompilation complete")


def _compute_deepseek_shapes(config: dict, tp: int, attn_tp: int):
    shapes = []

    hidden_size = config["hidden_size"]
    num_attention_heads = config.get("num_attention_heads", 128)
    kv_lora_rank = config.get("kv_lora_rank", 512)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 128)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
    v_head_dim = config.get("v_head_dim", 128)
    q_lora_rank = config.get("q_lora_rank", 0)
    n_routed_experts = config.get("n_routed_experts", 0)
    n_shared_experts = config.get("n_shared_experts", 0)
    moe_intermediate_size = config.get("moe_intermediate_size", 0)
    intermediate_size = config.get("intermediate_size", 0)
    first_k_dense_replace = config.get("first_k_dense_replace", 1)

    # Attention heads are sharded by attn_tp (which accounts for dp_attention)
    num_local_heads = num_attention_heads // attn_tp
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # MoE experts are sharded by regular tp
    num_local_experts = n_routed_experts + n_shared_experts

    # --- MoE expert GEMM shapes (MASKED/CONTIG, sharded by tp) ---
    if n_routed_experts > 0 and moe_intermediate_size > 0:
        moe_inter_per_tp = moe_intermediate_size // tp
        shapes.append(("MASKED", moe_inter_per_tp * 2, hidden_size, num_local_experts))
        shapes.append(("CONTIG", moe_inter_per_tp * 2, hidden_size, num_local_experts))
        shapes.append(("MASKED", hidden_size, moe_inter_per_tp, num_local_experts))
        shapes.append(("CONTIG", hidden_size, moe_inter_per_tp, num_local_experts))

    # --- MLA grouped GEMM shapes (MASKED, sharded by attn_tp) ---
    if kv_lora_rank > 0 and num_local_heads > 0:
        # Q_nope -> compressed K
        shapes.append(("MASKED", kv_lora_rank, qk_nope_head_dim, num_local_heads))
        # Attention output -> V
        shapes.append(("MASKED", v_head_dim, kv_lora_rank, num_local_heads))

    # --- GEMM_NT (non-grouped FP8) shapes for all linear layers ---
    if kv_lora_rank > 0 and num_local_heads > 0:
        # kv_b_proj: ColumnParallelLinear(kv_lora_rank, num_heads*(qk_nope+v_head_dim))
        kv_b_proj_n = num_local_heads * (qk_nope_head_dim + v_head_dim)
        shapes.append(("NORMAL", kv_b_proj_n, kv_lora_rank, 1))

        # o_proj: RowParallelLinear(num_heads*v_head_dim, hidden_size)
        o_proj_k = num_local_heads * v_head_dim
        shapes.append(("NORMAL", hidden_size, o_proj_k, 1))

    if q_lora_rank > 0:
        # fused_qkv_a_proj_with_mqa: ReplicatedLinear (no TP sharding)
        # N = q_lora_rank + kv_lora_rank + qk_rope_head_dim, K = hidden_size
        fused_n = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        shapes.append(("NORMAL", fused_n, hidden_size, 1))

        # q_b_proj: ColumnParallelLinear(q_lora_rank, num_heads*qk_head_dim)
        q_b_proj_n = num_local_heads * qk_head_dim
        shapes.append(("NORMAL", q_b_proj_n, q_lora_rank, 1))

    # --- Dense MLP layers (first_k_dense_replace layers, sharded by tp) ---
    if first_k_dense_replace > 0 and intermediate_size > 0:
        dense_inter_per_tp = intermediate_size // tp
        # gate_up_proj: MergedColumnParallelLinear(hidden_size, [inter, inter])
        shapes.append(("NORMAL", dense_inter_per_tp * 2, hidden_size, 1))
        # down_proj: RowParallelLinear(inter, hidden_size)
        shapes.append(("NORMAL", hidden_size, dense_inter_per_tp, 1))

    return shapes


def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if _is_musa:
        return nullcontext()

    return _deep_gemm_execution_hook(m, n, k, num_groups, kernel_type)


@contextmanager
def _deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
