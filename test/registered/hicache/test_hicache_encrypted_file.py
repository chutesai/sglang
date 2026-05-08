# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Unit tests for HiCacheEncryptedFile backend.

Covers:
  - Multi-rank startup: group directory isolation via _compute_group_dir
  - Key broadcast: _generate_and_broadcast_key for single and multi-rank
  - MLA exists() filesystem fallback
  - LRU eviction reservation protocol
  - AAD tamper detection (ciphertext + key-swap rejection)
  - Basic encrypt/decrypt round-trip
"""

import os
import secrets
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig


def _make_config(
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    is_mla_model=False,
    model_name="test/model",
):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


def _mock_tp_group(world_size=1, ranks=None):
    """Create a mock TP group coordinator."""
    group = MagicMock()
    group.world_size = world_size
    group.ranks = ranks or list(range(world_size))
    group.broadcast = MagicMock(side_effect=lambda t, src: None)
    return group


# ---------------------------------------------------------------------------
# Test _compute_group_dir isolation
# ---------------------------------------------------------------------------


class TestComputeGroupDir(unittest.TestCase):
    """Ensure each TP group gets a unique subdirectory, even with tp_size=1."""

    def _call(self, global_rank_0, model_name="test/model"):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _compute_group_dir,
        )

        config = _make_config(model_name=model_name)
        group = _mock_tp_group(world_size=1, ranks=[global_rank_0])
        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ):
            return _compute_group_dir(config)

    def test_different_global_ranks_give_different_dirs(self):
        dirs = {self._call(r) for r in range(8)}
        self.assertEqual(len(dirs), 8, "Each global rank should produce a unique dir")

    def test_single_rank_not_collapsed_to_g0(self):
        """Regression: tp_size=1 groups must NOT all map to g0."""
        d = self._call(global_rank_0=5)
        self.assertIn("_g5", d)

    def test_model_name_sanitized(self):
        d = self._call(0, model_name="deepseek-ai/DeepSeek-V3")
        self.assertIn("deepseek-ai-DeepSeek-V3", d)

    def test_multi_rank_group(self):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _compute_group_dir,
        )

        config = _make_config()
        group = _mock_tp_group(world_size=4, ranks=[8, 9, 10, 11])
        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ):
            d = _compute_group_dir(config)
        self.assertIn("_g8", d)


# ---------------------------------------------------------------------------
# Test _generate_and_broadcast_key
# ---------------------------------------------------------------------------


class TestGenerateAndBroadcastKey(unittest.TestCase):
    def test_single_rank_returns_32_bytes(self):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _generate_and_broadcast_key,
        )

        group = _mock_tp_group(world_size=1, ranks=[0])
        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ):
            key = _generate_and_broadcast_key(tp_rank=0)
        self.assertEqual(len(key), 32)
        self.assertIsInstance(key, bytes)

    def test_multi_rank_rank0_broadcasts(self):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _generate_and_broadcast_key,
        )

        group = _mock_tp_group(world_size=2, ranks=[0, 1])
        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ):
            key = _generate_and_broadcast_key(tp_rank=0)
        self.assertEqual(len(key), 32)
        group.broadcast.assert_called_once()
        call_args = group.broadcast.call_args
        tensor_arg = call_args[0][0]
        self.assertEqual(tensor_arg.dtype, torch.uint8)
        self.assertEqual(tensor_arg.shape, (32,))

    def test_multi_rank_non_rank0_broadcasts_zeros(self):
        """Non-rank-0 starts with zeros; after broadcast the key comes from rank 0."""
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _generate_and_broadcast_key,
        )

        group = _mock_tp_group(world_size=2, ranks=[0, 1])
        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ):
            key = _generate_and_broadcast_key(tp_rank=1)
        # broadcast was called but mock doesn't mutate the tensor,
        # so the key will be all zeros (since rank 1 doesn't fill it).
        self.assertEqual(len(key), 32)
        group.broadcast.assert_called_once()


# ---------------------------------------------------------------------------
# Test LRU index reservation protocol
# ---------------------------------------------------------------------------


class TestLRUIndex(unittest.TestCase):
    def _make_index(self, capacity_bytes):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import _LRUIndex

        return _LRUIndex(capacity_bytes)

    def test_reserve_new_key(self):
        idx = self._make_index(1000)
        is_new, evicted = idx.reserve("k1", 100)
        self.assertTrue(is_new)
        self.assertEqual(evicted, [])
        self.assertEqual(idx.total_bytes, 100)

    def test_reserve_duplicate_key_returns_false(self):
        idx = self._make_index(1000)
        idx.reserve("k1", 100)
        is_new, evicted = idx.reserve("k1", 100)
        self.assertFalse(is_new)
        self.assertEqual(idx.total_bytes, 100)

    def test_eviction_frees_oldest(self):
        idx = self._make_index(200)
        idx.reserve("k1", 100)
        idx.reserve("k2", 100)
        # This should evict k1 (oldest) to make room
        is_new, evicted = idx.reserve("k3", 100)
        self.assertTrue(is_new)
        self.assertIn("k1", evicted)
        self.assertNotIn("k2", evicted)

    def test_touch_updates_lru_order(self):
        idx = self._make_index(200)
        idx.reserve("k1", 100)
        idx.reserve("k2", 100)
        # Touch k1 so k2 is now oldest
        idx.touch("k1")
        is_new, evicted = idx.reserve("k3", 100)
        self.assertTrue(is_new)
        self.assertIn("k2", evicted)
        self.assertNotIn("k1", evicted)

    def test_confirm_adjusts_size(self):
        idx = self._make_index(1000)
        idx.reserve("k1", 100)
        idx.confirm("k1", 150)
        self.assertEqual(idx.total_bytes, 150)

    def test_cancel_removes_entry(self):
        idx = self._make_index(1000)
        idx.reserve("k1", 100)
        idx.cancel("k1")
        self.assertEqual(idx.total_bytes, 0)
        self.assertFalse(idx.contains("k1"))

    def test_unlimited_capacity(self):
        """capacity_bytes=0 means unlimited -- no eviction ever."""
        idx = self._make_index(0)
        for i in range(100):
            is_new, evicted = idx.reserve(f"k{i}", 1000)
            self.assertTrue(is_new)
            self.assertEqual(evicted, [])
        self.assertEqual(idx.total_bytes, 100_000)

    def test_concurrent_reserves(self):
        """Multiple threads reserving keys should not corrupt accounting."""
        idx = self._make_index(0)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    idx.reserve(f"t{thread_id}_k{i}", 10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(idx.count(), 400)
        self.assertEqual(idx.total_bytes, 4000)


# ---------------------------------------------------------------------------
# Test encrypt/decrypt round-trip and AAD tamper detection
# ---------------------------------------------------------------------------


class TestEncryptDecrypt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cryptography package not installed")

    def test_round_trip(self):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _decrypt_into,
            _encrypt_into,
            _make_aesgcm,
        )

        key = secrets.token_bytes(32)
        aesgcm = _make_aesgcm(key)
        original = torch.randn(64, 128, dtype=torch.float16)
        aad = b"test_key_suffix"

        blob = _encrypt_into(aesgcm, original, aad)
        target = torch.zeros_like(original)
        _decrypt_into(aesgcm, blob, aad, target)
        self.assertTrue(torch.equal(original, target))

    def test_wrong_aad_rejects(self):
        """Decrypting with wrong AAD must raise (InvalidTag)."""
        from cryptography.exceptions import InvalidTag

        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _decrypt_into,
            _encrypt_into,
            _make_aesgcm,
        )

        key = secrets.token_bytes(32)
        aesgcm = _make_aesgcm(key)
        data = torch.ones(32, dtype=torch.float32)

        blob = _encrypt_into(aesgcm, data, b"correct_key")
        target = torch.zeros_like(data)
        with self.assertRaises(InvalidTag):
            _decrypt_into(aesgcm, blob, b"wrong_key", target)

    def test_wrong_encryption_key_rejects(self):
        """Decrypting with a different AES key must raise."""
        from cryptography.exceptions import InvalidTag

        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _decrypt_into,
            _encrypt_into,
            _make_aesgcm,
        )

        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        aesgcm1 = _make_aesgcm(key1)
        aesgcm2 = _make_aesgcm(key2)
        data = torch.ones(16, dtype=torch.float32)
        aad = b"same_aad"

        blob = _encrypt_into(aesgcm1, data, aad)
        target = torch.zeros_like(data)
        with self.assertRaises(InvalidTag):
            _decrypt_into(aesgcm2, blob, aad, target)

    def test_tampered_ciphertext_rejects(self):
        """Flipping a bit in the ciphertext must be detected."""
        from cryptography.exceptions import InvalidTag

        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            _decrypt_into,
            _encrypt_into,
            _make_aesgcm,
        )

        key = secrets.token_bytes(32)
        aesgcm = _make_aesgcm(key)
        data = torch.ones(16, dtype=torch.float32)
        aad = b"key"

        blob = _encrypt_into(aesgcm, data, aad)
        # Tamper with ciphertext (after the 12-byte nonce)
        tampered = bytearray(blob)
        tampered[15] ^= 0xFF
        tampered = bytes(tampered)

        target = torch.zeros_like(data)
        with self.assertRaises(InvalidTag):
            _decrypt_into(aesgcm, tampered, aad, target)


# ---------------------------------------------------------------------------
# Test MLA exists() filesystem fallback
# ---------------------------------------------------------------------------


class TestMLAExistsFallback(unittest.TestCase):
    """Non-writing MLA ranks must see files written by rank 0 via filesystem check."""

    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cryptography package not installed")

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.key = secrets.token_bytes(32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_backend(self, tp_rank=0, tp_size=2, is_mla=True, global_ranks=None):
        """Construct an HiCacheEncryptedFile with mocked distributed state."""
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            HiCacheEncryptedFile,
        )

        if global_ranks is None:
            global_ranks = list(range(tp_size))
        config = _make_config(
            tp_rank=tp_rank, tp_size=tp_size, is_mla_model=is_mla
        )
        group = _mock_tp_group(world_size=tp_size, ranks=global_ranks)

        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file._generate_and_broadcast_key",
            return_value=self.key,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.envs"
        ) as mock_envs:
            mock_envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get.return_value = (
                self.tmpdir
            )
            backend = HiCacheEncryptedFile(config)
        return backend

    def test_rank0_set_rank1_exists(self):
        """Rank 0 writes a key; rank 1 (MLA, non-writer) should find it via filesystem."""
        rank0 = self._make_backend(tp_rank=0, tp_size=2, is_mla=True)
        data = torch.randn(32, dtype=torch.float16)
        self.assertTrue(rank0.set("page_abc", data))
        self.assertTrue(rank0.exists("page_abc"))

        # Rank 1 has the same file_path (MLA: no tp_rank in suffix)
        rank1 = self._make_backend(tp_rank=1, tp_size=2, is_mla=True)
        # rank1's index is empty, but filesystem fallback should find the file
        self.assertTrue(rank1.exists("page_abc"))

    def test_rank1_exists_returns_false_for_missing(self):
        rank1 = self._make_backend(tp_rank=1, tp_size=2, is_mla=True)
        self.assertFalse(rank1.exists("nonexistent_key"))

    def test_non_mla_no_filesystem_fallback(self):
        """Non-MLA: each rank has its own suffixed files, no filesystem fallback."""
        rank0 = self._make_backend(tp_rank=0, tp_size=2, is_mla=False)
        data = torch.randn(32, dtype=torch.float16)
        rank0.set("page_abc", data)

        # Rank 1 has a different suffix so won't find rank 0's file
        rank1 = self._make_backend(tp_rank=1, tp_size=2, is_mla=False)
        self.assertFalse(rank1.exists("page_abc"))


# ---------------------------------------------------------------------------
# Test full backend set/get round-trip
# ---------------------------------------------------------------------------


class TestBackendSetGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cryptography package not installed")

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.key = secrets.token_bytes(32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_backend(self, **kwargs):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            HiCacheEncryptedFile,
        )

        defaults = dict(tp_rank=0, tp_size=1, is_mla_model=False)
        defaults.update(kwargs)
        config = _make_config(**defaults)
        group = _mock_tp_group(world_size=1, ranks=[0])

        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file._generate_and_broadcast_key",
            return_value=self.key,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.envs"
        ) as mock_envs:
            mock_envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get.return_value = (
                self.tmpdir
            )
            return HiCacheEncryptedFile(config)

    def test_set_get_round_trip(self):
        backend = self._make_backend()
        original = torch.randn(128, 64, dtype=torch.float16)
        self.assertTrue(backend.set("key1", original))
        target = torch.zeros_like(original)
        result = backend.get("key1", target)
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(original, target))

    def test_get_missing_returns_none(self):
        backend = self._make_backend()
        target = torch.zeros(16, dtype=torch.float32)
        self.assertIsNone(backend.get("missing", target))

    def test_set_duplicate_is_noop(self):
        backend = self._make_backend()
        data = torch.randn(32, dtype=torch.float32)
        self.assertTrue(backend.set("k", data))
        self.assertTrue(backend.set("k", data))  # no error, returns True
        self.assertEqual(backend._index.count(), 1)

    def test_batch_set_get(self):
        backend = self._make_backend()
        keys = [f"k{i}" for i in range(5)]
        values = [torch.randn(32, dtype=torch.float16) for _ in range(5)]
        self.assertTrue(backend.batch_set(keys, values))
        targets = [torch.zeros(32, dtype=torch.float16) for _ in range(5)]
        results = backend.batch_get(keys, targets)
        for i, r in enumerate(results):
            self.assertIsNotNone(r)
            self.assertTrue(torch.equal(values[i], targets[i]))

    def test_clear_removes_all(self):
        backend = self._make_backend()
        for i in range(10):
            backend.set(f"k{i}", torch.randn(16, dtype=torch.float32))
        self.assertEqual(backend._index.count(), 10)
        self.assertTrue(backend.clear())
        self.assertEqual(backend._index.count(), 0)
        for i in range(10):
            self.assertFalse(backend.exists(f"k{i}"))

    def test_close_shuts_down_pool(self):
        backend = self._make_backend()
        backend.close()
        # After close, the thread pool should be shut down
        self.assertTrue(backend._io_pool._shutdown)


# ---------------------------------------------------------------------------
# Test disk capacity eviction
# ---------------------------------------------------------------------------


class TestDiskCapacityEviction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cryptography package not installed")

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.key = secrets.token_bytes(32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("SGLANG_HICACHE_DISK_CAPACITY_GB", None)

    def _make_backend(self, capacity_gb):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            HiCacheEncryptedFile,
        )

        config = _make_config(tp_rank=0, tp_size=1)
        group = _mock_tp_group(world_size=1, ranks=[0])
        os.environ["SGLANG_HICACHE_DISK_CAPACITY_GB"] = str(capacity_gb)

        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file._generate_and_broadcast_key",
            return_value=self.key,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.envs"
        ) as mock_envs:
            mock_envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get.return_value = (
                self.tmpdir
            )
            return HiCacheEncryptedFile(config)

    def test_eviction_deletes_oldest_files(self):
        # 1 KB capacity — very small to force eviction
        cap_gb = 1024 / (1024**3)  # 1 KB
        backend = self._make_backend(cap_gb)

        # Each tensor is 256 bytes + encryption overhead ≈ 284 bytes
        data = torch.randn(64, dtype=torch.float32)  # 256 bytes
        backend.set("k1", data)
        backend.set("k2", data)
        backend.set("k3", data)

        # With ~1KB cap and ~284 bytes each, should have evicted earliest
        self.assertTrue(backend.exists("k3"))
        # k1 should have been evicted to make room
        self.assertFalse(backend.exists("k1"))

    def test_evicted_files_removed_from_disk(self):
        cap_gb = 512 / (1024**3)  # 512 bytes
        backend = self._make_backend(cap_gb)

        data = torch.randn(64, dtype=torch.float32)  # 256 bytes
        backend.set("k1", data)
        path_k1 = backend._get_component_path("k1")
        self.assertTrue(os.path.isfile(path_k1))

        # Write enough to evict k1
        backend.set("k2", data)
        self.assertFalse(os.path.isfile(path_k1), "Evicted file should be deleted")


# ---------------------------------------------------------------------------
# Test multi-rank startup (purge coordination)
# ---------------------------------------------------------------------------


class TestMultiRankStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cryptography package not installed")

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.key = secrets.token_bytes(32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_backend(self, tp_rank, tp_size, global_ranks):
        from sglang.srt.mem_cache.storage.file.hicache_encrypted_file import (
            HiCacheEncryptedFile,
        )

        config = _make_config(tp_rank=tp_rank, tp_size=tp_size)
        group = _mock_tp_group(world_size=tp_size, ranks=global_ranks)

        with patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.get_tp_group",
            return_value=group,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file._generate_and_broadcast_key",
            return_value=self.key,
        ), patch(
            "sglang.srt.mem_cache.storage.file.hicache_encrypted_file.envs"
        ) as mock_envs:
            mock_envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get.return_value = (
                self.tmpdir
            )
            return HiCacheEncryptedFile(config)

    def test_rank0_purges_stale_files(self):
        """Rank 0 should purge existing files on init."""
        # Pre-create a stale file in the expected group directory
        group_dir = os.path.join(self.tmpdir, "test-model_g0")
        shard_dir = os.path.join(group_dir, "ab")
        os.makedirs(shard_dir)
        stale_path = os.path.join(shard_dir, "stale.bin")
        with open(stale_path, "wb") as f:
            f.write(b"stale data")

        self._make_backend(tp_rank=0, tp_size=2, global_ranks=[0, 1])
        self.assertFalse(os.path.isfile(stale_path), "Stale file should be purged")

    def test_shard_dirs_created(self):
        """All 256 shard dirs should exist after init."""
        backend = self._make_backend(tp_rank=0, tp_size=1, global_ranks=[0])
        for i in range(256):
            shard = os.path.join(backend.file_path, f"{i:02x}")
            self.assertTrue(os.path.isdir(shard), f"Missing shard dir {i:02x}")

    def test_pp_groups_isolated(self):
        """Two TP groups in a PP layout with tp_size=1 must get separate dirs."""
        b1 = self._make_backend(tp_rank=0, tp_size=1, global_ranks=[0])
        b2 = self._make_backend(tp_rank=0, tp_size=1, global_ranks=[1])
        self.assertNotEqual(
            b1.file_path, b2.file_path,
            "PP groups with different global ranks must have different dirs"
        )

    def test_non_rank0_creates_dirs_too(self):
        """Non-rank-0 must also create shard dirs (they may need to write)."""
        # First create rank 0 to purge
        self._make_backend(tp_rank=0, tp_size=2, global_ranks=[0, 1])
        # Now create rank 1
        backend = self._make_backend(tp_rank=1, tp_size=2, global_ranks=[0, 1])
        for i in range(256):
            shard = os.path.join(backend.file_path, f"{i:02x}")
            self.assertTrue(os.path.isdir(shard))


if __name__ == "__main__":
    unittest.main()
