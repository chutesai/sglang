# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Encrypted, size-capped, LRU-evicting file-based HiCache storage backend.

On startup, rank 0 within each TP group purges its group-scoped storage
subdirectory, generates a fresh AES-256-GCM key, and broadcasts it to all
ranks in the TP group.  The key lives only in process memory -- never disk.

Storage isolation:
  Each TP group gets its own subdirectory identified by the group's first
  global rank (unique regardless of PP, DP, DP-attention, or EP topology).

Capacity:
  SGLANG_HICACHE_DISK_CAPACITY_GB  --  disk budget **per TP group** in GB
  (default: 0 = unlimited).  Within a group the budget is divided among
  writers: for MLA models only rank 0 writes (full budget); for non-MLA
  every TP rank writes its own files (budget / tp_size each).  If multiple
  TP groups share the same physical disk, total consumption is budget *
  num_groups.

MLA shared-writer mode:
  For MLA models only tp_rank 0 performs backup writes (see backup_skip in
  cache_controller.py).  Non-writing ranks answer exists()/batch_exists()
  queries by checking the filesystem directly (not their empty local index),
  so the MIN all-reduce during prefetch correctly sees the files rank 0
  wrote.

On-disk layout per file:
  [12-byte nonce | ciphertext | 16-byte GCM tag]
  with the suffixed cache key bound as GCM AAD.

Requires:  pip install 'sglang[encrypted-cache]'
"""

import hashlib
import logging
import os
import secrets
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Set

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_NONCE_LEN = 12
_TAG_LEN = 16
_ENCRYPT_OVERHEAD = _NONCE_LEN + _TAG_LEN
_SHARD_PREFIX_LEN = 2  # 256 subdirectories
_IO_WORKERS = 4


# ---------------------------------------------------------------------------
# Crypto helpers
# ---------------------------------------------------------------------------


def _make_aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError(
            "The encrypted_file HiCache backend requires the 'cryptography' "
            "package.  Install it with:  pip install 'sglang[encrypted-cache]'"
        )
    return AESGCM(key)


def _encrypt_into(aesgcm, src: torch.Tensor, aad: bytes) -> bytes:
    """Encrypt tensor data.  Returns nonce || ciphertext || tag."""
    np_buf = src.contiguous().view(dtype=torch.uint8).numpy()
    nonce = secrets.token_bytes(_NONCE_LEN)
    ct = aesgcm.encrypt(nonce, bytes(memoryview(np_buf)), aad)
    return nonce + ct


def _decrypt_into(aesgcm, blob: bytes, aad: bytes, target: torch.Tensor) -> None:
    """Decrypt blob directly into target tensor."""
    plaintext = aesgcm.decrypt(blob[:_NONCE_LEN], blob[_NONCE_LEN:], aad)
    flat = target.view(torch.uint8).contiguous()
    flat.copy_(torch.frombuffer(bytearray(plaintext), dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Key generation + broadcast
# ---------------------------------------------------------------------------


def _generate_and_broadcast_key(tp_rank: int) -> bytes:
    """Generate or broadcast a 32-byte AES key across the attention TP group.

    In DP-attention mode each scheduler process is its own attention TP group
    (size 1), so each generates an independent key — no broadcast needed.
    In normal TP mode, rank 0 generates and broadcasts to the full TP group.
    """
    try:
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        if is_dp_attention_enabled():
            # Each DP-attention worker is independent; no shared key needed.
            return secrets.token_bytes(32)
    except ImportError:
        pass

    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return secrets.token_bytes(32)

    key_tensor = torch.zeros(32, dtype=torch.uint8, device="cuda")
    if tp_rank == 0:
        key_bytes = secrets.token_bytes(32)
        key_tensor.copy_(torch.frombuffer(bytearray(key_bytes), dtype=torch.uint8))

    tp_group.broadcast(key_tensor, src=0)
    return bytes(key_tensor.cpu().tolist())


# ---------------------------------------------------------------------------
# Group directory naming
# ---------------------------------------------------------------------------


def _compute_group_dir(storage_config: HiCacheStorageConfig) -> str:
    """Return a subdirectory name unique to this storage writer group.

    Uses the TP group's first global rank as the base identifier, then
    appends the DP-attention rank when DP-attention is enabled.  In
    DP-attention mode all ranks share one TP group (for MoE/FFN) but each
    DP-attention rank runs its own cache controller and must write to its
    own isolated directory.
    """
    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    group_id = tp_group.ranks[0]

    model_name = storage_config.model_name or ""
    model_name_safe = "-".join(model_name.split("/"))
    dir_name = f"{model_name_safe}_g{group_id}"

    try:
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            is_dp_attention_enabled,
        )

        if is_dp_attention_enabled():
            dir_name += f"_dp{get_attention_dp_rank()}"
    except ImportError:
        pass

    return dir_name


# ---------------------------------------------------------------------------
# LRU index with atomic space reservation
# ---------------------------------------------------------------------------


class _LRUIndex:
    """Thread-safe LRU index with size accounting and reservation protocol."""

    def __init__(self, capacity_bytes: int):
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._total_bytes: int = 0
        self._capacity_bytes = capacity_bytes

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def touch(self, key: str) -> None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def reserve(self, key: str, estimated_size: int) -> tuple[bool, list[str]]:
        """Atomically check-and-reserve space for *key*.

        Returns (is_new, evicted_keys).
        """
        with self._lock:
            if key in self._entries:
                return False, []

            evicted: list[str] = []
            if self._capacity_bytes > 0:
                overshoot = (self._total_bytes + estimated_size) - self._capacity_bytes
                if overshoot > 0:
                    evicted = self._evict_locked(overshoot)

            self._entries[key] = estimated_size
            self._entries.move_to_end(key)
            self._total_bytes += estimated_size
            return True, evicted

    def confirm(self, key: str, actual_size: int) -> None:
        with self._lock:
            old = self._entries.get(key)
            if old is None:
                return
            self._entries[key] = actual_size
            self._total_bytes += actual_size - old

    def cancel(self, key: str) -> None:
        self.remove(key)

    def remove(self, key: str) -> int:
        with self._lock:
            size = self._entries.pop(key, None)
            if size is not None:
                self._total_bytes -= size
                return size
            return 0

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def _evict_locked(self, needed_bytes: int) -> list[str]:
        victims = []
        freed = 0
        while freed < needed_bytes and self._entries:
            key, size = self._entries.popitem(last=False)
            freed += size
            self._total_bytes -= size
            victims.append(key)
        return victims


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------


class HiCacheEncryptedFile(HiCacheStorage):
    """Encrypted, size-capped, LRU-evicting file backend for HiCache L3."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache",
    ):
        from sglang.srt.environ import envs

        base_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path
        self._tp_rank = storage_config.tp_rank
        self._is_mla = storage_config.is_mla_model

        # Group-scoped subdirectory: unique per TP group via first global rank.
        group_dir = _compute_group_dir(storage_config)
        self.file_path = os.path.join(base_path, group_dir)

        model_name = storage_config.model_name
        model_name_safe = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = storage_config.pp_size > 1
        self.config_suffix = f"_{model_name_safe}"
        if not self._is_mla:
            self.config_suffix += f"_{storage_config.tp_rank}_{storage_config.tp_size}"
        if enable_pp:
            self.config_suffix += f"_{storage_config.pp_size}_{storage_config.pp_rank}"

        # --- Coordinated startup: rank 0 purges, then broadcast syncs all ---
        if self._tp_rank == 0:
            self._purge_and_create_dirs()

        key = _generate_and_broadcast_key(self._tp_rank)
        self._key = key
        self._aesgcm = _make_aesgcm(key)

        if self._tp_rank != 0:
            os.makedirs(self.file_path, exist_ok=True)
            for i in range(256):
                os.makedirs(os.path.join(self.file_path, f"{i:02x}"), exist_ok=True)

        # Capacity (per-group budget, divided among writers).
        cap_gb = float(os.environ.get("SGLANG_HICACHE_DISK_CAPACITY_GB", "0"))
        if cap_gb > 0:
            num_writers = 1 if self._is_mla else storage_config.tp_size
            per_writer_gb = cap_gb / num_writers
            cap_bytes = int(per_writer_gb * (1024**3))
        else:
            cap_bytes = 0
            per_writer_gb = 0
        self._index = _LRUIndex(cap_bytes)

        self._io_pool = ThreadPoolExecutor(
            max_workers=_IO_WORKERS, thread_name_prefix="hicache_io"
        )

        logger.info(
            "HiCacheEncryptedFile initialized: AES-256-GCM (ephemeral key), "
            "capacity=%.1f GB/writer (%.1f GB/group), group=%s, tp_rank=%d, "
            "mla=%s",
            per_writer_gb if cap_bytes > 0 else float("inf"),
            cap_gb if cap_gb > 0 else float("inf"),
            group_dir,
            self._tp_rank,
            self._is_mla,
        )

    def _purge_and_create_dirs(self) -> None:
        if os.path.isdir(self.file_path):
            for entry in os.scandir(self.file_path):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path, ignore_errors=True)
                    elif entry.is_file(follow_symlinks=False):
                        os.remove(entry.path)
                except OSError:
                    pass
        os.makedirs(self.file_path, exist_ok=True)
        for i in range(256):
            os.makedirs(os.path.join(self.file_path, f"{i:02x}"), exist_ok=True)

    def close(self) -> None:
        """Shut down the I/O thread pool.  Called by runtime detach."""
        self._io_pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _shard_for_key(self, suffixed_key: str) -> str:
        h = hashlib.md5(suffixed_key.encode(), usedforsecurity=False).hexdigest()
        return h[:_SHARD_PREFIX_LEN]

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _path_for_suffixed_key(self, suffixed_key: str) -> str:
        shard = self._shard_for_key(suffixed_key)
        return os.path.join(self.file_path, shard, f"{suffixed_key}.bin")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return self._path_for_suffixed_key(self._get_component_key(key, component_name))

    def _ensure_shard_dir(self, path: str) -> None:
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _make_aad(suffixed_key: str) -> bytes:
        return suffixed_key.encode("utf-8")

    # ------------------------------------------------------------------
    # Existence checks -- filesystem-backed for MLA shared-writer mode
    # ------------------------------------------------------------------

    def _key_exists(self, suffixed_key: str) -> bool:
        """Check whether a key exists.

        For MLA models, non-writing ranks (tp_rank != 0) don't populate the
        index (they skip backup), but rank 0 writes shared files to disk.
        These ranks must check the filesystem so the MIN all-reduce during
        prefetch correctly reflects rank 0's writes.

        For non-MLA, each rank writes its own suffixed files and the local
        index is authoritative.
        """
        if self._index.contains(suffixed_key):
            return True
        if self._is_mla:
            # Rank 0's index has it, but this rank's doesn't.  Check disk.
            return os.path.isfile(self._path_for_suffixed_key(suffixed_key))
        return False

    # ------------------------------------------------------------------
    # Eviction file cleanup
    # ------------------------------------------------------------------

    def _delete_evicted_files(self, victims: list[str]) -> None:
        for key in victims:
            try:
                os.remove(self._path_for_suffixed_key(key))
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Failed to delete evicted file %s: %s", key, e)

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        suffixed = self._get_suffixed_key(key)
        path = self._path_for_suffixed_key(suffixed)
        try:
            with open(path, "rb") as f:
                blob = f.read()
            _decrypt_into(self._aesgcm, blob, self._make_aad(suffixed), target_location)
            self._index.touch(suffixed)
            return target_location
        except FileNotFoundError:
            self._index.remove(suffixed)
            return None
        except Exception as e:
            logger.error("Failed to read/decrypt %s: %s", key, e)
            self._index.remove(suffixed)
            return None

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        suffixed = self._get_suffixed_key(key)

        plaintext_size = value.numel() * value.element_size()
        estimated_blob_size = plaintext_size + _ENCRYPT_OVERHEAD

        is_new, evicted = self._index.reserve(suffixed, estimated_blob_size)
        if not is_new:
            return True

        self._delete_evicted_files(evicted)

        try:
            aad = self._make_aad(suffixed)
            blob = _encrypt_into(self._aesgcm, value, aad)

            path = self._path_for_suffixed_key(suffixed)
            self._ensure_shard_dir(path)
            with open(path, "wb") as f:
                f.write(blob)

            if len(blob) != estimated_blob_size:
                self._index.confirm(suffixed, len(blob))
            return True
        except Exception as e:
            logger.error("Failed to encrypt/save %s: %s", key, e)
            self._index.cancel(suffixed)
            return False

    def exists(self, key: str) -> bool:
        return self._key_exists(self._get_suffixed_key(key))

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[Optional[torch.Tensor]]:
        return [
            self.get(k, loc)
            for k, loc in zip(keys, target_locations or [None] * len(keys))
        ]

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for k, v in zip(keys, values):
            if not self.set(k, v):
                return False
        return True

    # ------------------------------------------------------------------
    # V2 batch interface
    # ------------------------------------------------------------------

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        """Check which component files exist.

        Uses _key_exists (index + filesystem fallback for MLA) so non-writing
        MLA ranks correctly report files written by rank 0.
        """
        existing = set()
        for key in keys:
            sk = self._get_component_key(key)
            if self._key_exists(sk):
                existing.add(f"{sk}.bin")
        for transfer in pool_transfers or []:
            for key in keys:
                sk = self._get_component_key(key, transfer.name)
                if self._key_exists(sk):
                    existing.add(f"{sk}.bin")
        return existing

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)),
                    kv_pages,
                )
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            futures = [
                self._io_pool.submit(
                    op_fn,
                    transfer.name,
                    key,
                    host_pool,
                    host_indices[i * page_size].item(),
                )
                for i, key in enumerate(keys)
            ]
            results[transfer.name] = [f.result() for f in futures]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            for shard in range(256):
                shard_dir = os.path.join(self.file_path, f"{shard:02x}")
                if not os.path.isdir(shard_dir):
                    continue
                for fname in os.listdir(shard_dir):
                    fpath = os.path.join(shard_dir, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
            self._index.reset()
            logger.info("Cleared all entries in HiCacheEncryptedFile storage.")
            return True
        except Exception as e:
            logger.error("Failed to clear HiCacheEncryptedFile storage: %s", e)
            return False

    def get_stats(self):
        return {
            "total_entries": self._index.count(),
            "total_bytes": self._index.total_bytes,
            "capacity_bytes": self._index.capacity_bytes,
            "utilization": (
                self._index.total_bytes / self._index.capacity_bytes
                if self._index.capacity_bytes > 0
                else 0.0
            ),
        }
