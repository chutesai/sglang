# SPDX-License-Identifier: Apache-2.0
"""
Verify the integrity of a HuggingFace model cache before loading.

Checks that all files in the HF cache match the remote repository metadata:
- LFS files have correct SHA256 (via symlink target name)
- Non-LFS files have correct blob_id / SHA1 (via symlink target name)
- File sizes match
- No missing or extraneous files

Exits the process hard on any discrepancy.
"""

import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHUTES_PROXY_URL = "https://proxy.chutes.ai/misc/hf_repo_info"


def _get_symlink_hash(file_path: Path) -> Optional[str]:
    """Extract hash from symlink target (blob filename).

    HF cache stores files as symlinks pointing to blobs named by hash:
    - 64 chars = SHA256 (LFS files)
    - 40 chars = SHA1 (git blob / non-LFS files)
    """
    if file_path.is_symlink():
        target = os.readlink(file_path)
        blob_name = Path(target).name
        if len(blob_name) in (40, 64):
            return blob_name
    return None


def _git_blob_hash(filepath: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute git blob SHA-1 for a file (streaming, memory efficient).

    Git blob format: "blob {size}\\0{content}"
    """
    size = filepath.stat().st_size
    sha1 = hashlib.sha1()
    sha1.update(f"blob {size}\0".encode())
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha1.update(chunk)
    return sha1.hexdigest()


def _compute_sha256(filepath: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 hash of a file (streaming, memory efficient)."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _fetch_repo_info_from_hf(
    repo_id: str,
    revision: str,
    hf_token: Optional[str] = None,
) -> dict:
    """Fetch repository file metadata directly from HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    repo_items = api.list_repo_tree(
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
        recursive=True,
    )

    files = []
    directories = []
    for item in repo_items:
        if not hasattr(item, "size"):
            directories.append(item.path)
            continue
        file_info: dict = {
            "path": item.path,
            "size": getattr(item, "size", None),
        }
        if hasattr(item, "lfs") and item.lfs:
            file_info["sha256"] = item.lfs.sha256
            file_info["is_lfs"] = True
        else:
            file_info["blob_id"] = getattr(item, "blob_id", None)
            file_info["is_lfs"] = False
        files.append(file_info)

    return {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "files": files,
        "directories": directories,
    }


def _fetch_repo_info_from_proxy(
    repo_id: str,
    revision: str,
    hf_token: Optional[str] = None,
) -> dict:
    """Fetch repository file metadata from chutes proxy (fallback)."""
    params: dict[str, str] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
    }
    if hf_token:
        params["hf_token"] = hf_token

    url = f"{CHUTES_PROXY_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            body = resp.read().decode(errors="replace")
            raise RuntimeError(
                f"Chutes proxy returned {resp.status}: {body}"
            )
        return json.loads(resp.read().decode())


def _get_repo_info(
    repo_id: str,
    revision: str,
    hf_token: Optional[str] = None,
) -> dict:
    """Get repo info from HF directly, falling back to chutes proxy."""
    # Try HuggingFace directly first.
    hf_error = None
    try:
        logger.info(
            "Fetching repo info from HuggingFace for %s@%s",
            repo_id,
            revision,
        )
        return _fetch_repo_info_from_hf(repo_id, revision, hf_token)
    except Exception as e:
        hf_error = e
        logger.warning(
            "Failed to fetch repo info from HuggingFace directly: %s. "
            "Falling back to chutes proxy.",
            e,
        )

    # Fallback: chutes proxy.
    try:
        logger.info(
            "Fetching repo info from chutes proxy for %s@%s",
            repo_id,
            revision,
        )
        return _fetch_repo_info_from_proxy(repo_id, revision, hf_token)
    except Exception as proxy_error:
        logger.fatal(
            "Failed to fetch repo info from both HuggingFace and chutes "
            "proxy for %s@%s. Cannot verify cache integrity.\n"
            "  HF error: %s\n"
            "  Proxy error: %s",
            repo_id,
            revision,
            hf_error,
            proxy_error,
        )
        os._exit(99)


def _find_snapshot_dir(
    repo_id: str,
    revision: str,
    cache_dir: Optional[str],
) -> Path:
    """Locate the HF cache snapshot directory for a given repo/revision.

    HF stores snapshots under:
        {cache_dir}/hub/models--{org}--{name}/snapshots/{commit_hash}

    The revision may be a branch name, in which case we resolve it via
    the refs file to get the actual commit hash.
    """
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        hf_cache = Path(HF_HUB_CACHE)
    else:
        hf_cache = Path(cache_dir)
        # If the cache_dir contains a "hub" subdirectory, descend into it.
        if (hf_cache / "hub").is_dir():
            hf_cache = hf_cache / "hub"

    repo_folder = f"models--{repo_id.replace('/', '--')}"
    repo_dir = hf_cache / repo_folder

    if not repo_dir.is_dir():
        logger.fatal(
            "HF cache repo directory not found: %s", repo_dir
        )
        os._exit(99)

    snapshots_dir = repo_dir / "snapshots"

    # Try the revision directly (works for commit hashes).
    direct = snapshots_dir / revision
    if direct.is_dir():
        return direct

    # Resolve branch/tag name via refs file.
    refs_file = repo_dir / "refs" / revision
    if refs_file.is_file():
        commit_hash = refs_file.read_text().strip()
        resolved = snapshots_dir / commit_hash
        if resolved.is_dir():
            return resolved

    # If there's exactly one snapshot, use that.
    if snapshots_dir.is_dir():
        snapshot_dirs = [
            d for d in snapshots_dir.iterdir() if d.is_dir()
        ]
        if len(snapshot_dirs) == 1:
            logger.info(
                "Could not resolve revision '%s' from refs, using "
                "only available snapshot: %s",
                revision,
                snapshot_dirs[0].name,
            )
            return snapshot_dirs[0]

    logger.fatal(
        "Could not find snapshot directory for %s@%s in %s",
        repo_id,
        revision,
        snapshots_dir,
    )
    os._exit(99)


def _verify_cache(
    repo_id: str,
    revision: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    full_hash_check: bool = False,
    max_workers: int = 4,
) -> None:
    """Core verification logic."""
    repo_info = _get_repo_info(repo_id, revision, hf_token)

    # Build remote files dict: {path: (hash, size, is_lfs)}
    remote_files: dict[str, tuple] = {}
    for item in repo_info["files"]:
        is_lfs = item.get("is_lfs", False)
        if is_lfs:
            remote_files[item["path"]] = (
                item.get("sha256"),
                item.get("size"),
                True,
            )
        else:
            remote_files[item["path"]] = (
                item.get("blob_id"),
                item.get("size"),
                False,
            )

    # Track directories from remote.
    directories = repo_info.get("directories")
    if directories is not None:
        for dir_path in directories:
            if dir_path not in remote_files:
                remote_files[dir_path] = (None, None, False)
    else:
        for item in repo_info["files"]:
            parts = item["path"].split("/")
            for i in range(1, len(parts)):
                dir_path = "/".join(parts[:i])
                if dir_path not in remote_files:
                    remote_files[dir_path] = (None, None, False)

    snapshot_dir = _find_snapshot_dir(repo_id, revision, cache_dir)
    logger.info("Verifying HF cache at: %s", snapshot_dir)

    # Index local files.
    local_files: dict[str, Path] = {}
    for path in snapshot_dir.rglob("*"):
        if path.is_file() or path.is_symlink() or path.is_dir():
            rel_path = str(path.relative_to(snapshot_dir))
            local_files[rel_path] = path

    verified = 0
    skipped = 0
    mismatches: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    # Files needing hash computation:
    # (remote_path, resolved_path, expected_hash, hash_type)
    files_to_hash: list[tuple[str, Path, str, str]] = []

    for remote_path, (remote_hash, remote_size, is_lfs) in (
        remote_files.items()
    ):
        local_path = local_files.get(remote_path)

        if not local_path or (
            not local_path.exists() and not local_path.is_symlink()
        ):
            missing.append(remote_path)
            continue

        if remote_hash is None:
            skipped += 1
            continue

        resolved_path = local_path.resolve()

        # Check size first (quick sanity check).
        if remote_size is not None:
            try:
                actual_size = resolved_path.stat().st_size
                if actual_size != remote_size:
                    mismatches.append(
                        f"{remote_path}: size {actual_size} "
                        f"!= expected {remote_size}"
                    )
                    continue
            except OSError as e:
                errors.append(f"{remote_path}: cannot stat: {e}")
                continue

        if is_lfs:
            if full_hash_check:
                files_to_hash.append(
                    (remote_path, resolved_path, remote_hash, "sha256")
                )
            else:
                symlink_hash = _get_symlink_hash(local_path)
                if symlink_hash:
                    if symlink_hash != remote_hash:
                        mismatches.append(
                            f"{remote_path}: hash {symlink_hash} "
                            f"!= expected {remote_hash}"
                        )
                    else:
                        verified += 1
                else:
                    errors.append(
                        f"{remote_path}: LFS file not a symlink, "
                        "cannot fast-verify"
                    )
        else:
            if full_hash_check:
                files_to_hash.append(
                    (remote_path, resolved_path, remote_hash, "git_blob")
                )
            else:
                symlink_hash = _get_symlink_hash(local_path)
                if symlink_hash:
                    if symlink_hash != remote_hash:
                        mismatches.append(
                            f"{remote_path}: hash {symlink_hash} "
                            f"!= expected {remote_hash}"
                        )
                    else:
                        verified += 1
                else:
                    # Not a symlink, must compute hash.
                    files_to_hash.append(
                        (
                            remote_path,
                            resolved_path,
                            remote_hash,
                            "git_blob",
                        )
                    )

    # Compute hashes in parallel using thread pool.
    if files_to_hash:

        def _compute_hash(
            item: tuple[str, Path, str, str],
        ) -> tuple[str, Optional[str], str, Optional[str]]:
            rpath, resolved, expected, hash_type = item
            try:
                if hash_type == "sha256":
                    computed = _compute_sha256(resolved)
                else:
                    computed = _git_blob_hash(resolved)
                return (rpath, computed, expected, None)
            except Exception as e:
                return (rpath, None, expected, str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_compute_hash, files_to_hash))

        for remote_path, computed, expected, error in results:
            if error:
                errors.append(
                    f"{remote_path}: hash computation failed: {error}"
                )
            elif computed != expected:
                mismatches.append(
                    f"{remote_path}: hash {computed} "
                    f"!= expected {expected}"
                )
            else:
                verified += 1

    # Check for extraneous local files not in remote.
    extra = [
        p
        for p in local_files
        if p not in remote_files
        and not any(part.startswith("_") for part in Path(p).parts)
        and not any(part.startswith(".") for part in Path(p).parts)
    ]

    if mismatches or missing or extra or errors:
        msg_parts = [
            f"Cache verification FAILED for {repo_id}@{revision}"
        ]
        if mismatches:
            msg_parts.append(f"  Mismatches ({len(mismatches)}):")
            for m in mismatches:
                msg_parts.append(f"    - {m}")
        if missing:
            msg_parts.append(f"  Missing ({len(missing)}):")
            for m in missing:
                msg_parts.append(f"    - {m}")
        if extra:
            msg_parts.append(f"  Extra ({len(extra)}):")
            for e in extra:
                msg_parts.append(f"    - {e}")
        if errors:
            msg_parts.append(f"  Errors ({len(errors)}):")
            for e in errors:
                msg_parts.append(f"    - {e}")
        logger.fatal("\n".join(msg_parts))
        os._exit(99)

    logger.info(
        "HF cache verified for %s@%s: %d files verified, %d skipped",
        repo_id,
        revision,
        verified,
        skipped,
    )


def verify_model_cache(
    model: str,
    revision: Optional[str],
    download_dir: Optional[str],
    hf_token: Optional[str] = None,
    full_hash_check: bool = False,
) -> None:
    """Verify HuggingFace model cache integrity before loading.

    Exits the process if any integrity issues are found or if repo
    metadata cannot be fetched.

    Args:
        model: HuggingFace model ID (e.g. "meta-llama/Llama-2-7b").
        revision: Model revision (branch, tag, or commit hash).
        download_dir: HF cache directory override (None = HF default).
        hf_token: HuggingFace API token.
        full_hash_check: If True, compute full file hashes instead of
            checking symlink names. Much slower but more thorough.
    """
    # Skip verification for local model paths.
    if os.path.isdir(model):
        logger.info(
            "Skipping HF cache verification for local model path: %s",
            model,
        )
        return

    if revision is None:
        revision = "main"

    logger.info(
        "Starting HF cache verification for %s@%s", model, revision
    )

    try:
        _verify_cache(
            repo_id=model,
            revision=revision,
            cache_dir=download_dir,
            hf_token=hf_token,
            full_hash_check=full_hash_check,
        )
    except Exception as e:
        logger.fatal(
            "Unexpected error during cache verification for "
            "%s@%s: %s",
            model,
            revision,
            e,
        )
        os._exit(99)
