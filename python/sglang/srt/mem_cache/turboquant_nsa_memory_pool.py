"""
TurboQuant memory pool for NSA (Native Sparse Attention) KV cache compression.

Extends MLATokenToKVPoolTurboQuant with the FP8 index K cache required by
DeepSeek-V3.2's NSA architecture.  The main MLA KV cache is TurboQuant-
compressed; the indexer cache remains in FP8 (128-dim, negligible size).
"""

import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import (
    NSATokenToKVPool,
    get_tensor_size_bytes,
    index_buf_accessor,
)
from sglang.srt.mem_cache.turboquant_mla_memory_pool import MLATokenToKVPoolTurboQuant

try:
    from sglang.srt.utils.common import is_hip as _is_hip_fn

    _is_hip = _is_hip_fn()
except (ImportError, AttributeError):
    _is_hip = False

logger = logging.getLogger(__name__)


class NSATokenToKVPoolTurboQuant(MLATokenToKVPoolTurboQuant):
    """TurboQuant-compressed MLA cache + FP8 index K cache for NSA models.

    Inherits all TurboQuant MLA behavior for the main KV cache.
    Adds the paged FP8 index K + scale buffer used by the NSA indexer.
    The indexer continues to use FP8 index keys directly — no TQ on index
    cache since it's only 128-dim FP8 with negligible memory footprint.
    """

    @property
    def can_use_fused_kernel(self):
        """NSA decode uses dynamic sparse selection — incompatible with fused TQ kernel."""
        return False

    # Match NSATokenToKVPool class-level constants
    quant_block_size = NSATokenToKVPool.quant_block_size  # 128
    index_k_with_scale_buffer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
    rope_storage_dtype = NSATokenToKVPool.rope_storage_dtype

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        bits: float = 4,
        mode: str = "mse",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # Initialize TurboQuant MLA base with use_nsa=True to suppress
        # _finalize_allocation_log until we allocate index_k buffers below.
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            bits=bits,
            mode=mode,
            start_layer=start_layer,
            end_layer=end_layer,
            use_nsa=True,
        )

        self.index_head_dim = index_head_dim
        assert index_head_dim == 128

        if _is_hip:
            assert self.page_size == 1
        else:
            assert self.page_size == 64

        # Allocate FP8 index K + scale buffer (same as NSATokenToKVPool)
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    (
                        (size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim
                            + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]

        self._finalize_allocation_log(size)

    # --- Index K cache methods (delegated from NSATokenToKVPool pattern) ---

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        size = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            size += get_tensor_size_bytes(index_k_cache)
        return size
