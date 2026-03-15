"""Calibration for IndexCache layer assignment.

Uses the greedy LM-loss approach from the THUDM reference implementation:
start with all layers Full, then iteratively convert the layer whose removal
causes the least perplexity increase. Each candidate flip is tested by
toggling `index_cache_is_shared` on the attention layer — no engine restart
needed.

Supports block-wise search (--num-blocks) matching the THUDM paper's pipeline
parallelism acceleration: layers are divided into P blocks, each block's first
layer is protected as Full, and the best candidate in each block is committed
per step. This reduces forward passes by ~P times.

Usage:

1. **Uniform spacing** (no GPU needed):
   python scripts/index_cache_calibrate.py \\
       --model zai-org/GLM-5 --uniform --target-ratio 0.25 -o config.json

2. **Greedy LM-loss calibration** (requires GPU, matches THUDM reference):
   python scripts/index_cache_calibrate.py \\
       --model deepseek-ai/DeepSeek-V3.2 --tp 8 --target-ratio 0.25 -o config.json

3. **Fast calibration with block-wise search + DP attention** (~Nx faster with DP):
   python scripts/index_cache_calibrate.py \\
       --model deepseek-ai/DeepSeek-V3.2 --tp 8 --dp 8 --enable-dp-attention \\
       --target-ratio 0.25 --num-blocks 8 --context-length 131072 -o config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Calibration datasets — long-document sources for prefill loss measurement.
# Only the input tokens matter (max_new_tokens=1), so raw text is fine.
CALIBRATION_DATASETS = {
    "pg19": {
        "hf_path": "deepmind/pg19",
        "hf_name": None,
        "split": "test",
        "text_column": "text",
    },
    "infinitebench": {
        "hf_path": "xinrongzhang2022/InfiniteBench",
        "hf_name": None,
        "split": "longbook_qa_eng",
        "text_column": "context",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_single_dataset(ds_name: str):
    from datasets import load_dataset

    if ds_name in CALIBRATION_DATASETS:
        cfg = CALIBRATION_DATASETS[ds_name]
        ds = load_dataset(
            cfg["hf_path"],
            cfg["hf_name"],
            split=cfg["split"],
            streaming=True,
            trust_remote_code=True,
        )
        return ds, cfg["text_column"]
    else:
        ds = load_dataset(ds_name, split="train", streaming=True)
        return ds, "text"


def _cache_key(
    num_samples: int, context_length: int, tokenizer_name: Optional[str]
) -> str:
    key_data = json.dumps(
        {
            "num_samples": num_samples,
            "context_length": context_length,
            "datasets": {k: v["hf_path"] for k, v in CALIBRATION_DATASETS.items()},
            "tokenizer": tokenizer_name or "char_approx",
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _cache_path(cache_key: str) -> Path:
    cache_dir = Path("/cache") / "index_cache_calibration"
    if not cache_dir.parent.exists():
        cache_dir = Path.home() / ".cache" / "index_cache_calibration"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"prompts_{cache_key}.jsonl"


def _get_tokenizer(tokenizer_name: Optional[str]):
    if not tokenizer_name:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        logger.info(f"Using tokenizer {tokenizer_name}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using char approximation.")
        return None


def _truncate_to_length(text: str, target_tokens: int, tokenizer=None) -> Optional[str]:
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < target_tokens:
            return None
        return tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)
    else:
        char_len = target_tokens * 4
        if len(text) < char_len:
            return None
        return text[:char_len]


def load_calibration_prompts(
    num_samples: int,
    context_length: int,
    tokenizer_name: Optional[str] = None,
    no_cache: bool = False,
) -> List[str]:
    """Load calibration prompts at target context length.

    Following THUDM reference: calibrate at max context because if indices
    are similar at max context, they're similar at shorter context too.
    """
    key = _cache_key(num_samples, context_length, tokenizer_name)
    cached = _cache_path(key)
    if not no_cache and cached.exists():
        logger.info(f"Loading cached calibration prompts from {cached}")
        prompts = [json.loads(line) for line in open(cached)]
        logger.info(f"Loaded {len(prompts)} cached prompts")
        return prompts

    tokenizer = _get_tokenizer(tokenizer_name)
    prompts = []

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    dataset_names = list(CALIBRATION_DATASETS.keys())
    per_dataset = max(1, num_samples // len(dataset_names))

    logger.info(
        f"Loading {num_samples} prompts at {context_length} tokens "
        f"from: {dataset_names}"
    )

    for ds_name in dataset_names:
        if len(prompts) >= num_samples:
            break
        try:
            ds, text_column = _load_single_dataset(ds_name)
        except Exception as e:
            logger.warning(f"  Skipping {ds_name}: {e}")
            continue

        ds_count = 0
        skipped = 0
        ds_target = min(per_dataset, num_samples - len(prompts))
        ds_iter = (
            tqdm(ds, desc=f"  {ds_name}", leave=False, unit="doc") if has_tqdm else ds
        )

        for item in ds_iter:
            if ds_count >= ds_target:
                break
            text = item.get(text_column, "")
            if not text or len(text.strip()) < 100:
                continue
            if len(text) // 4 < context_length:
                skipped += 1
                if has_tqdm and hasattr(ds_iter, "set_postfix"):
                    ds_iter.set_postfix(found=ds_count, skipped=skipped)
                continue
            prompt = _truncate_to_length(text, context_length, tokenizer)
            if prompt is not None:
                prompts.append(prompt)
                ds_count += 1

        logger.info(f"  {ds_name}: {ds_count} prompts (skipped {skipped} too-short)")

    if len(prompts) < num_samples:
        logger.warning(
            f"Only found {len(prompts)}/{num_samples} prompts >= {context_length} tokens"
        )

    # Cache
    with open(cached, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Cached to {cached}")
    return prompts


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------


def _load_raw_model_config(model_path: str) -> Optional[dict]:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
    except Exception:
        pass
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_path, "config.json")) as f:
            return json.load(f)
    except Exception:
        pass
    try:
        p = Path(model_path) / "config.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def get_num_layers(model_path: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    cfg = _load_raw_model_config(model_path)
    if cfg and "num_hidden_layers" in cfg:
        n = cfg["num_hidden_layers"]
        logger.info(f"Auto-detected num_layers={n}")
        return n
    logger.error(f"Could not detect num_layers for {model_path}. Use --num-layers.")
    sys.exit(1)


def get_max_context_length(model_path: str) -> Optional[int]:
    cfg = _load_raw_model_config(model_path)
    if not cfg:
        return None
    for key in ["max_position_embeddings", "max_sequence_length", "seq_length"]:
        if key in cfg:
            logger.info(f"Auto-detected context length={cfg[key]} from {key}")
            return cfg[key]
    return None


# ---------------------------------------------------------------------------
# Core calibration: greedy LM-loss (matches THUDM reference Algorithm 1)
# ---------------------------------------------------------------------------


def _set_layer_pattern(engine, shared_layers: Set[int]):
    """Toggle IndexCache layers between Full/Shared via Engine IPC.

    Also flushes the KV cache (RadixCache) to prevent stale prefix matches.
    Without this, SGLang's prefix caching would reuse KV values computed
    under a previous IndexCache pattern, giving incorrect loss measurements.
    """
    engine.flush_cache()
    result = engine.update_index_cache(sorted(shared_layers))
    if not result.success:
        raise RuntimeError("Failed to update IndexCache layer pattern via IPC")
    return result


def _extract_logprob_nll(output) -> tuple:
    """Extract NLL from a single generate output. Returns (nll, num_tokens)."""
    meta = output.get("meta_info", {})
    input_token_logprobs = meta.get("input_token_logprobs", None)
    nll = 0.0
    tokens = 0
    if input_token_logprobs is not None and len(input_token_logprobs) > 0:
        for entry in input_token_logprobs:
            if entry is not None:
                # Each entry is (logprob, token_id, token_text)
                lp = entry[0] if isinstance(entry, (list, tuple)) else entry
                if lp is not None:
                    nll -= lp
                    tokens += 1
    return nll, tokens


def _measure_loss(
    engine,
    prompts: List[str],
    max_new_tokens: int = 1,
    batch_size: int = 1,
) -> float:
    """Measure average LM loss (negative log-likelihood) on prompts.

    Uses engine.generate with return_logprob to get per-token log probs,
    then averages across all tokens and prompts.

    Args:
        batch_size: Number of prompts to send per engine.generate call.
            With --dp N, set to N to utilize all DP workers in parallel.
    """
    total_nll = 0.0
    total_tokens = 0

    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
    }

    # Process prompts in batches for DP parallelism
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        if len(batch) == 1:
            # Single prompt — pass as string, not list
            outputs = engine.generate(
                batch[0],
                sampling_params,
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=0,
            )
            if isinstance(outputs, list):
                outputs = outputs[0]
            nll, tokens = _extract_logprob_nll(outputs)
            total_nll += nll
            total_tokens += tokens
        else:
            # Batch of prompts — engine distributes across DP workers
            outputs = engine.generate(
                batch,
                [sampling_params] * len(batch),
                return_logprob=[True] * len(batch),
                logprob_start_len=[0] * len(batch),
                top_logprobs_num=[0] * len(batch),
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            for output in outputs:
                nll, tokens = _extract_logprob_nll(output)
                total_nll += nll
                total_tokens += tokens

    if total_tokens == 0:
        logger.warning("No log-probs returned. Is return_logprob supported?")
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = np.exp(avg_nll)
    logger.info(
        f"  Loss: {avg_nll:.4f} (perplexity: {perplexity:.2f}, tokens: {total_tokens})"
    )
    return avg_nll


def _make_blocks(
    num_layers: int, num_blocks: int, protected: Set[int]
) -> tuple:
    """Divide layers into blocks for block-wise greedy search.

    Each block's first layer is added to the protected set (always Full).

    Returns:
        (blocks, all_protected) where blocks is a list of candidate layer
        lists per block, and all_protected is the full set of protected layers.
    """
    block_size = max(1, num_layers // num_blocks)
    blocks = []
    all_protected = set(protected)

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size if b < num_blocks - 1 else num_layers
        if start >= num_layers:
            break
        # First layer of each block is protected
        all_protected.add(start)
        # Candidates are non-protected layers in this block
        candidates = [l for l in range(start, end) if l not in all_protected]
        if candidates:
            blocks.append(candidates)

    logger.info(
        f"Block-wise search: {len(blocks)} blocks, "
        f"protected layers: {sorted(all_protected)}"
    )
    return blocks, all_protected


def greedy_loss_calibration(
    engine,
    calibration_prompts: List[str],
    num_layers: int,
    target_num_full: int,
    num_blocks: int = 1,
    validation_prompts: Optional[List[str]] = None,
    batch_size: int = 1,
) -> Set[int]:
    """Greedy calibration matching THUDM reference Algorithm 1.

    With num_blocks=1 (default): pure greedy, one layer committed per step.
    With num_blocks=P: block-wise search matching THUDM paper's PP
    acceleration. Layers divided into P blocks, best candidate in each
    block committed per step. ~P times fewer forward passes.

    Args:
        engine: Running SGLang Engine instance.
        calibration_prompts: Prompts for loss measurement.
        num_layers: Total number of model layers.
        target_num_full: Target number of Full layers to keep.
        num_blocks: Number of blocks for block-wise search (1=pure greedy).
        validation_prompts: If provided, used for final validation loss.
        batch_size: Number of prompts per generate call (set to dp_size).

    Returns:
        Set of layer IDs that should remain Full.
    """
    # Layers 0 and 1 are always Full (layer 0 = NextN/dense, layer 1 = first DSA)
    full_layers = set(range(num_layers))
    protected = {0, 1}
    shared_layers: Set[int] = set()

    layers_to_remove = len(full_layers) - target_num_full

    # Verify IPC works and measure baseline loss (all Full)
    result = _set_layer_pattern(engine, shared_layers)
    logger.info(f"IndexCache IPC ready (toggled {result.num_toggled} layers)")
    logger.info("Measuring baseline loss (all layers Full)...")
    baseline_loss = _measure_loss(engine, calibration_prompts, batch_size=batch_size)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    if num_blocks > 1:
        # Block-wise greedy search (THUDM paper's PP acceleration)
        blocks, all_protected = _make_blocks(num_layers, num_blocks, protected)

        step = 0
        while len(full_layers) > target_num_full:
            step += 1
            committed_this_step = []

            for block_idx, block_candidates in enumerate(blocks):
                if len(full_layers) <= target_num_full:
                    break

                # Filter to layers still Full and not protected
                available = [
                    l for l in block_candidates
                    if l in full_layers and l not in all_protected
                ]
                if not available:
                    continue

                best_layer = None
                best_loss = float("inf")

                desc = f"Step {step} block {block_idx + 1}/{len(blocks)}"
                iter_candidates = (
                    tqdm(available, desc=desc, leave=False, unit="layer")
                    if has_tqdm
                    else available
                )

                for layer_id in iter_candidates:
                    trial_shared = shared_layers | {layer_id}
                    _set_layer_pattern(engine, trial_shared)
                    loss = _measure_loss(engine, calibration_prompts, batch_size=batch_size)

                    if loss < best_loss:
                        best_loss = loss
                        best_layer = layer_id

                if best_layer is not None:
                    committed_this_step.append(best_layer)
                    shared_layers.add(best_layer)
                    full_layers.remove(best_layer)
                    # Apply immediately so next block sees updated state
                    # (matches paper: "best flip in each block is committed
                    # before the next block is searched")
                    _set_layer_pattern(engine, shared_layers)

                    loss_delta = best_loss - baseline_loss
                    logger.info(
                        f"[Step {step}, block {block_idx + 1}] "
                        f"Layer {best_layer} -> Shared "
                        f"(loss={best_loss:.4f}, delta={loss_delta:+.4f}, "
                        f"remaining Full={len(full_layers)})"
                    )

            if not committed_this_step:
                logger.warning("No layers committed this step, stopping early.")
                break

            logger.info(
                f"Step {step} complete: committed {len(committed_this_step)} layers "
                f"({committed_this_step}), "
                f"remaining Full={len(full_layers)}/{num_layers}"
            )
    else:
        # Pure greedy search (original Algorithm 1)
        candidates = set(range(2, num_layers))

        for step in range(layers_to_remove):
            best_layer = None
            best_loss = float("inf")

            step_candidates = sorted((candidates & full_layers) - protected)
            if not step_candidates:
                break

            desc = f"Step {step + 1}/{layers_to_remove}"
            iter_candidates = (
                tqdm(step_candidates, desc=desc, leave=False, unit="layer")
                if has_tqdm
                else step_candidates
            )

            for layer_id in iter_candidates:
                trial_shared = shared_layers | {layer_id}
                _set_layer_pattern(engine, trial_shared)
                loss = _measure_loss(engine, calibration_prompts, batch_size=batch_size)

                if loss < best_loss:
                    best_loss = loss
                    best_layer = layer_id

            if best_layer is None:
                break

            shared_layers.add(best_layer)
            full_layers.remove(best_layer)
            candidates.discard(best_layer)
            _set_layer_pattern(engine, shared_layers)

            loss_delta = best_loss - baseline_loss
            logger.info(
                f"[{step + 1}/{layers_to_remove}] Layer {best_layer} -> Shared "
                f"(loss={best_loss:.4f}, delta={loss_delta:+.4f}, "
                f"remaining Full={len(full_layers)})"
            )

    # Final validation
    if validation_prompts:
        logger.info("Running final validation with full prompt set...")
        _set_layer_pattern(engine, shared_layers)
        final_loss = _measure_loss(engine, validation_prompts, batch_size=batch_size)
        _set_layer_pattern(engine, set())  # all Full
        baseline_final = _measure_loss(engine, validation_prompts, batch_size=batch_size)
        loss_increase = (final_loss - baseline_final) / baseline_final * 100
        logger.info(
            f"Final validation: baseline={baseline_final:.4f}, "
            f"IndexCache={final_loss:.4f}, "
            f"increase={loss_increase:+.2f}%"
        )

    return full_layers


# ---------------------------------------------------------------------------
# Uniform config
# ---------------------------------------------------------------------------


def generate_uniform_config(num_layers: int, target_ratio: float, output_path: str):
    # Match THUDM reference formula: layers 0,1 always Full, then
    # every step-th layer starting from layer 1: {0, 1, 1+step, 1+2*step, ...}
    step = max(1, int(round(1.0 / target_ratio)))
    full_layers = sorted({0, 1} | set(range(1, num_layers, step)))
    config = {
        "full_layers": full_layers,
        "num_layers": num_layers,
        "target_ratio": target_ratio,
        "actual_ratio": len(full_layers) / num_layers,
        "method": "uniform",
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig written to: {output_path}")
    print(f"Full layers ({len(full_layers)}/{num_layers}): {full_layers}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate IndexCache Full/Shared layer assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Greedy LM-loss calibration (matches THUDM reference):
  python scripts/index_cache_calibrate.py \\
      --model deepseek-ai/DeepSeek-V3.2 --tp 8 --target-ratio 0.25 -o config.json

  # Fast block-wise calibration with DP (~8x blocks + ~4x DP = ~32x faster):
  python scripts/index_cache_calibrate.py \\
      --model deepseek-ai/DeepSeek-V3.2 --tp 8 --dp 8 --enable-dp-attention \\
      --target-ratio 0.25 --num-blocks 8 --context-length 131072 -o config.json

  # Uniform config (no GPU needed):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --uniform --target-ratio 0.25 -o config.json

        """,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HuggingFace ID"
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data parallel size (default: 1). Set to N to process N prompts "
        "in parallel per loss measurement, giving ~Nx speedup.",
    )
    parser.add_argument(
        "--enable-dp-attention",
        action="store_true",
        help="Enable DP attention (required when using --dp with MLA models).",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.25,
        help="Target fraction of Full layers (default: 0.25)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="index_cache_config.json",
        help="Output JSON config path",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (auto-detected if not set)",
    )
    parser.add_argument(
        "--uniform", action="store_true", help="Use uniform spacing (no GPU needed)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length for calibration (auto-detected from model). "
        "Use your production context length (e.g. 131072) for best results.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=768,
        help="Number of calibration prompts (default: 768, matching THUDM paper). "
        "eval-prompts used per greedy step, rest for final validation.",
    )
    parser.add_argument(
        "--eval-prompts",
        type=int,
        default=8,
        help="Number of prompts used per greedy step (default: 8). "
        "Set to a multiple of --dp for full utilization. "
        "Lower = faster but noisier. Must be <= calibration-samples.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of blocks for block-wise greedy search (default: 1 = pure greedy). "
        "Set to 8 for ~8x speedup matching THUDM paper's PP acceleration. "
        "Each block's first layer is protected as Full.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="GPU memory fraction (default: 0.85)",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
        help="Chunked prefill size (larger = faster prefill, more memory). "
        "Recommend matching your production config (e.g. 16384).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regeneration of calibration prompts",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    num_layers = get_num_layers(args.model, args.num_layers)

    if args.uniform:
        generate_uniform_config(num_layers, args.target_ratio, args.output)
        return

    # Determine context length
    context_length = args.context_length
    if context_length is None:
        context_length = get_max_context_length(args.model)
        if context_length is None:
            logger.error("Could not auto-detect context length. Use --context-length.")
            sys.exit(1)
    # Reserve a small margin so prompts fit strictly within context window
    # (SGLang requires input_tokens < context_length).
    prompt_length = context_length - 32
    logger.info(
        f"Calibration context length: {context_length}, "
        f"prompt length: {prompt_length}"
    )

    # Load calibration data
    calibration_prompts = load_calibration_prompts(
        num_samples=args.calibration_samples,
        context_length=prompt_length,
        tokenizer_name=args.model,
        no_cache=args.no_cache,
    )

    # --- Greedy LM-loss calibration ---
    from sglang.srt.entrypoints.engine import Engine

    target_num_full = max(1, int(round(num_layers * args.target_ratio)))

    # Estimate forward passes
    layers_to_remove = num_layers - target_num_full
    if args.num_blocks > 1:
        avg_candidates_per_block = (num_layers - 2) // args.num_blocks
        steps = max(1, layers_to_remove // args.num_blocks)
        est_passes = steps * args.num_blocks * avg_candidates_per_block * args.eval_prompts
    else:
        est_passes = sum(range(num_layers - 2, num_layers - 2 - layers_to_remove, -1)) * args.eval_prompts
    dp_info = f", dp={args.dp}" if args.dp > 1 else ""
    logger.info(
        f"Calibrating {args.model}: {num_layers} layers, "
        f"target {target_num_full} Full, context={context_length}, "
        f"samples={len(calibration_prompts)}, eval_prompts={args.eval_prompts}, "
        f"blocks={args.num_blocks}{dp_info}, ~{est_passes} forward passes"
    )

    # Build engine kwargs
    engine_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp,
        index_cache_ratio=1.0,  # Start all Full — we toggle manually
        disable_cuda_graph=True,
        mem_fraction_static=args.mem_fraction_static,
        log_level="info",
    )
    if args.dp > 1:
        engine_kwargs["dp_size"] = args.dp
    if args.enable_dp_attention:
        engine_kwargs["enable_dp_attention"] = True
    if args.chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = args.chunked_prefill_size
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True
    if args.context_length is not None:
        engine_kwargs["context_length"] = args.context_length

    engine = Engine(**engine_kwargs)

    try:
        # Split prompts: eval for greedy steps, rest for final validation
        n_eval = min(args.eval_prompts, len(calibration_prompts))
        eval_prompts = calibration_prompts[:n_eval]
        val_prompts = calibration_prompts[n_eval:] if len(calibration_prompts) > n_eval else None

        # With DP, batch prompts to utilize all workers in parallel
        batch_size = args.dp if args.dp > 1 else 1

        full_layers = greedy_loss_calibration(
            engine=engine,
            calibration_prompts=eval_prompts,
            num_layers=num_layers,
            target_num_full=target_num_full,
            num_blocks=args.num_blocks,
            validation_prompts=val_prompts,
            batch_size=batch_size,
        )
    finally:
        engine.shutdown()

    config = {
        "full_layers": sorted(full_layers),
        "num_layers": num_layers,
        "target_ratio": args.target_ratio,
        "actual_ratio": len(full_layers) / num_layers,
        "method": "greedy_lm_loss" + (f"_blocks{args.num_blocks}" if args.num_blocks > 1 else ""),
        "calibration_context_length": context_length,
        "calibration_samples": len(calibration_prompts),
        "eval_prompts": n_eval,
        "num_blocks": args.num_blocks,
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig written to: {args.output}")
    print(f"Full layers ({len(full_layers)}/{num_layers}): {sorted(full_layers)}")
    print(
        f"\nTo use: python -m sglang.launch_server --model MODEL "
        f"--index-cache-config {args.output}"
    )


if __name__ == "__main__":
    main()
