"""Training-free calibration for IndexCache layer assignment.

Supports three modes:

1. **Uniform spacing** (no GPU needed):
   Evenly spaces Full layers. Simple and effective.

   python scripts/index_cache_calibrate.py \\
       --model zai-org/GLM-5 --uniform --target-ratio 0.25 -o config.json

2. **Greedy calibration** (requires GPU, runs model offline):
   Collects per-layer top-k indices on calibration data, then greedily
   removes Full layers that have highest Jaccard similarity to neighbors.

   python scripts/index_cache_calibrate.py \\
       --model zai-org/GLM-5 --tp 8 --target-ratio 0.25 -o config.json

3. **Similarity analysis** (requires GPU, runs model offline):
   Measures and reports pairwise Jaccard similarity between consecutive
   layers to validate cross-layer index reuse assumptions.

   python scripts/index_cache_calibrate.py \\
       --model zai-org/GLM-5 --tp 8 --measure-similarity

Calibration uses a stratified mix of datasets across domains (code, books,
web, math, chat) and sequence lengths (2K to 120K tokens), biased toward
longer sequences to ensure IndexCache correctness at long context.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Individual calibration datasets, organized by domain.
# Each entry has: hf_path, hf_name (subset), split, text_column,
# and max_doc_tokens (approximate max usable length per document).
CALIBRATION_DATASETS = {
    # General web text (short-medium)
    "slimpajama": {
        "hf_path": "DKYoon/SlimPajama-6B",
        "hf_name": None,
        "split": "validation",
        "text_column": "text",
        "max_doc_tokens": 8192,
        "domain": "web",
    },
    "c4": {
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "split": "validation",
        "text_column": "text",
        "max_doc_tokens": 4096,
        "domain": "web",
    },
    # Code (public, no auth needed)
    "code_python": {
        "hf_path": "codeparrot/github-code",
        "hf_name": "Python-all",
        "split": "train",
        "text_column": "code",
        "max_doc_tokens": 32768,
        "domain": "code",
    },
    # Books (long context)
    "pg19": {
        "hf_path": "deepmind/pg19",
        "hf_name": None,
        "split": "test",
        "text_column": "text",
        "max_doc_tokens": 200000,
        "domain": "books",
    },
    # Academic/mixed (long docs)
    "long_data": {
        "hf_path": "emozilla/Long-Data-Collections-Fine-Tune",
        "hf_name": None,
        "split": "train",
        "text_column": "text",
        "max_doc_tokens": 65536,
        "domain": "academic",
    },
    # Long-context QA
    "longbench": {
        "hf_path": "THUDM/LongBench",
        "hf_name": "qasper",
        "split": "test",
        "text_column": "context",
        "max_doc_tokens": 32768,
        "domain": "long_qa",
    },
    # Very long context (100K+)
    "infinitebench": {
        "hf_path": "xinrongzhang2022/InfiniteBench",
        "hf_name": None,
        "split": "longbook_qa_eng",
        "text_column": "context",
        "max_doc_tokens": 200000,
        "domain": "books",
    },
}

# Stratified length distribution for comprehensive calibration.
# Biased toward longer sequences since that's where IndexCache matters most
# and where calibration failures are most damaging.
# Format: (min_tokens, max_tokens, fraction_of_total)
LENGTH_STRATA = [
    (1024, 4096, 0.10),       # 10% short
    (4096, 16384, 0.15),      # 15% medium
    (16384, 32768, 0.20),     # 20% medium-long
    (32768, 65536, 0.25),     # 25% long
    (65536, 120000, 0.30),    # 30% very long
]

# Which datasets to use for each length stratum.
# Short strata can use any dataset; long strata need datasets with long docs.
LENGTH_DATASET_MAP = {
    (1024, 4096): ["slimpajama", "c4", "code_python", "long_data"],
    (4096, 16384): ["slimpajama", "code_python", "long_data", "longbench"],
    (16384, 32768): ["code_python", "long_data", "longbench", "pg19"],
    (32768, 65536): ["pg19", "long_data", "infinitebench"],
    (65536, 120000): ["pg19", "long_data", "infinitebench"],
}


def _load_single_dataset(ds_name: str):
    """Load a single dataset in streaming mode, return (iterator, text_column)."""
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


def load_calibration_prompts(
    dataset_name: str,
    num_samples: int = 128,
    seq_len: int = 2048,
    tokenizer_name: Optional[str] = None,
) -> List[str]:
    """Load calibration prompts from a single dataset at a fixed length.

    Used when --calibration-dataset is specified (legacy single-dataset mode).
    """
    from datasets import load_dataset

    ds, text_column = _load_single_dataset(dataset_name)
    tokenizer = _get_tokenizer(tokenizer_name)

    prompts = []
    for item in ds:
        if len(prompts) >= num_samples:
            break

        text = item.get(text_column, "")
        if not text or len(text.strip()) < 100:
            continue

        prompt = _truncate_to_length(text, seq_len, tokenizer)
        if prompt is not None:
            prompts.append(prompt)

    if len(prompts) < num_samples:
        logger.warning(
            f"Only found {len(prompts)} suitable samples "
            f"(requested {num_samples})"
        )

    logger.info(
        f"Loaded {len(prompts)} calibration prompts "
        f"(target seq_len={seq_len} tokens)"
    )
    return prompts


def load_stratified_calibration_prompts(
    num_samples: int = 20000,
    tokenizer_name: Optional[str] = None,
    length_strata: Optional[List] = None,
) -> List[str]:
    """Load calibration prompts stratified across datasets and sequence lengths.

    Samples from multiple domains (code, books, web, academic, QA) at multiple
    sequence lengths (2K-120K), biased toward longer sequences.

    Args:
        num_samples: Total number of calibration prompts to collect.
        tokenizer_name: Tokenizer for accurate length measurement.
        length_strata: Override default LENGTH_STRATA.

    Returns:
        List of prompts covering diverse domains and lengths.
    """
    if length_strata is None:
        length_strata = LENGTH_STRATA

    tokenizer = _get_tokenizer(tokenizer_name)
    all_prompts = []
    rng = random.Random(42)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    for min_tok, max_tok, fraction in length_strata:
        target_count = max(1, int(num_samples * fraction))
        datasets_for_stratum = LENGTH_DATASET_MAP.get(
            (min_tok, max_tok),
            list(CALIBRATION_DATASETS.keys()),
        )
        # Divide target evenly across datasets, then round-robin
        per_dataset = max(1, target_count // len(datasets_for_stratum))
        stratum_prompts = []

        logger.info(
            f"Loading stratum [{min_tok}-{max_tok} tok] "
            f"({fraction:.0%}, target={target_count}): "
            f"datasets={datasets_for_stratum}"
        )

        for ds_name in datasets_for_stratum:
            if len(stratum_prompts) >= target_count:
                break

            try:
                ds, text_column = _load_single_dataset(ds_name)
            except Exception as e:
                logger.warning(f"  Skipping {ds_name}: {e}")
                continue

            ds_count = 0
            skipped = 0
            # Target the middle of the stratum range for this dataset
            target_len = (min_tok + max_tok) // 2

            ds_iter = ds
            if has_tqdm:
                ds_iter = tqdm(
                    ds, desc=f"  {ds_name}", leave=False,
                    unit="doc",
                )

            for item in ds_iter:
                if ds_count >= per_dataset:
                    break
                if len(stratum_prompts) >= target_count:
                    break

                text = item.get(text_column, "")
                if not text or len(text.strip()) < 100:
                    continue

                # Check document is long enough for this stratum
                approx_tokens = len(text) // 4
                if approx_tokens < min_tok:
                    skipped += 1
                    if has_tqdm and hasattr(ds_iter, 'set_postfix'):
                        ds_iter.set_postfix(found=ds_count, skipped=skipped)
                    continue

                # Sample a random length within the stratum range
                sample_len = rng.randint(min_tok, min(max_tok, approx_tokens))
                prompt = _truncate_to_length(text, sample_len, tokenizer)
                if prompt is not None:
                    stratum_prompts.append(prompt)
                    ds_count += 1

            logger.info(f"  {ds_name}: {ds_count} prompts")

        all_prompts.extend(stratum_prompts)
        logger.info(
            f"  Stratum [{min_tok}-{max_tok}]: "
            f"{len(stratum_prompts)}/{target_count} prompts collected"
        )

    # Shuffle so strata are interleaved during inference
    rng.shuffle(all_prompts)

    # Log distribution summary
    if tokenizer:
        lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts[:100]]
    else:
        lengths = [len(p) // 4 for p in all_prompts[:100]]
    logger.info(
        f"Loaded {len(all_prompts)} total calibration prompts. "
        f"Length distribution (sample of 100): "
        f"min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, "
        f"max={max(lengths)} tokens"
    )
    return all_prompts


def _get_tokenizer(tokenizer_name: Optional[str]):
    """Load tokenizer for length normalization, or return None."""
    if not tokenizer_name:
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        logger.info(f"Using tokenizer {tokenizer_name} for length normalization")
        return tokenizer
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using char approximation.")
        return None


def _truncate_to_length(
    text: str, target_tokens: int, tokenizer=None
) -> Optional[str]:
    """Truncate text to target token length. Returns None if too short."""
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < target_tokens // 2:
            return None
        tokens = tokens[:target_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        char_len = target_tokens * 4
        if len(text) < char_len // 2:
            return None
        return text[:char_len]


def compute_jaccard_similarity(
    indices_a: np.ndarray, indices_b: np.ndarray
) -> float:
    """Compute average Jaccard similarity between two sets of top-k indices.

    Args:
        indices_a: (num_tokens, topk) int array
        indices_b: (num_tokens, topk) int array

    Returns:
        Average Jaccard similarity across all tokens.
    """
    similarities = []
    n = min(indices_a.shape[0], indices_b.shape[0])
    for i in range(n):
        set_a = set(indices_a[i].tolist())
        set_b = set(indices_b[i].tolist())
        if len(set_a) == 0 and len(set_b) == 0:
            similarities.append(1.0)
        else:
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            similarities.append(intersection / union)
    return float(np.mean(similarities))


def greedy_layer_assignment(
    layer_indices: Dict[int, np.ndarray],
    num_layers: int,
    target_num_full: int,
) -> Set[int]:
    """Greedily assign layers as Full or Shared to maximize index reuse quality.

    Starts with all layers as Full, then iteratively converts the layer whose
    removal causes the least quality degradation (highest Jaccard similarity to
    its nearest remaining Full layer).
    """
    full_layers = set(range(num_layers))
    # Layer 0 (NextN/dense) and layer 1 (first DSA layer) are always Full.
    # Matches THUDM reference: layer 0 is dense attention (not DSA), so its
    # indices are meaningless for DSA layers. Layer 1 must compute its own.
    candidates = set(range(2, num_layers))

    layers_to_remove = len(full_layers) - target_num_full
    removed = 0
    while len(full_layers) > target_num_full and candidates:
        best_layer = None
        best_similarity = -1.0

        for layer_id in sorted(candidates):
            remaining = full_layers - {layer_id}
            preceding = [f for f in remaining if f < layer_id]
            if not preceding:
                following = [f for f in remaining if f > layer_id]
                if not following:
                    continue
                nearest = min(following)
            else:
                nearest = max(preceding)

            if layer_id not in layer_indices or nearest not in layer_indices:
                continue

            sim = compute_jaccard_similarity(
                layer_indices[layer_id], layer_indices[nearest]
            )
            if sim > best_similarity:
                best_similarity = sim
                best_layer = layer_id

        if best_layer is None:
            break

        full_layers.remove(best_layer)
        candidates.remove(best_layer)
        removed += 1
        logger.info(
            f"[{removed}/{layers_to_remove}] Converted layer {best_layer} to Shared "
            f"(similarity={best_similarity:.4f}, "
            f"remaining Full={len(full_layers)})"
        )

    return full_layers


def collect_indices(
    model_path: str,
    tp_size: int,
    calibration_prompts: List[str],
    max_new_tokens: int = 1,
    batch_size: int = 8,
    mem_fraction_static: float = 0.80,
) -> Dict[int, np.ndarray]:
    """Collect top-k indices from all layers using SGLang's offline Engine.

    Launches the engine with --index-cache-capture-dir pointing to a temp
    directory. During inference, each layer's indexer writes its topk_indices
    tensor to disk. After inference completes, we read them back and
    concatenate across forward passes.

    Args:
        model_path: Model path or HuggingFace ID.
        tp_size: Tensor parallel size.
        calibration_prompts: List of calibration text prompts.
        max_new_tokens: Tokens to generate per prompt (1 is sufficient for
            capturing prefill indices).
        batch_size: Number of prompts to send per batch (prevents OOM with
            long prompts by limiting concurrent prefill).
        mem_fraction_static: GPU memory fraction for model weights/KV cache.
            Lower than default to leave room for long-context prefill.

    Returns:
        Dict mapping layer_id -> np.ndarray of shape (total_tokens, topk).
    """
    import tempfile

    import torch

    from sglang.srt.entrypoints.engine import Engine

    capture_dir = tempfile.mkdtemp(prefix="index_cache_capture_")
    logger.info(f"Capturing indices to: {capture_dir}")

    # Launch engine with capture mode enabled
    engine = Engine(
        model_path=model_path,
        tp_size=tp_size,
        index_cache_capture_dir=capture_dir,
        # Disable CUDA graphs — incompatible with per-pass file I/O
        disable_cuda_graph=True,
        mem_fraction_static=mem_fraction_static,
        log_level="info",
    )

    try:
        # Send prompts in batches to prevent OOM on long sequences.
        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        total = len(calibration_prompts)
        num_batches = (total + batch_size - 1) // batch_size
        logger.info(
            f"Running {total} calibration prompts in {num_batches} batches "
            f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})..."
        )

        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                range(0, total, batch_size),
                desc="Calibration",
                total=num_batches,
                unit="batch",
            )
        except ImportError:
            batch_iter = range(0, total, batch_size)

        for i in batch_iter:
            batch = calibration_prompts[i : i + batch_size]
            # Log approximate token count for this batch
            batch_chars = sum(len(p) for p in batch)
            approx_tokens = batch_chars // 4
            if not isinstance(batch_iter, range):
                batch_iter.set_postfix(
                    prompts=f"{min(i + batch_size, total)}/{total}",
                    approx_tok=f"~{approx_tokens:,}",
                )
            else:
                done = min(i + batch_size, total)
                logger.info(
                    f"  Batch {i // batch_size + 1}/{num_batches}: "
                    f"{done}/{total} prompts (~{approx_tokens:,} tokens)"
                )
            engine.generate(batch, sampling_params)
        logger.info("Inference complete. Reading captured indices...")
    finally:
        engine.shutdown()

    # Read captured indices from disk
    # Structure: {capture_dir}/layer_{id}/pass_{n}.pt
    layer_indices: Dict[int, List[np.ndarray]] = {}
    capture_path = Path(capture_dir)

    for layer_dir in sorted(capture_path.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_id = int(layer_dir.name.split("_")[1])
        indices_list = []
        for pt_file in sorted(layer_dir.iterdir()):
            if pt_file.suffix != ".pt":
                continue
            tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
            indices_list.append(tensor.numpy())
        if indices_list:
            layer_indices[layer_id] = indices_list

    if not layer_indices:
        raise RuntimeError(
            f"No indices captured in {capture_dir}. "
            f"This may indicate the model does not use the DSA indexer, "
            f"or the prompts were too short to trigger index computation."
        )

    # Concatenate across passes per layer
    result = {}
    for layer_id, arrays in layer_indices.items():
        result[layer_id] = np.concatenate(arrays, axis=0)

    # Clean up
    import shutil

    shutil.rmtree(capture_dir, ignore_errors=True)

    logger.info(
        f"Captured indices for {len(result)} layers. "
        f"Tokens per layer: {next(iter(result.values())).shape[0]}"
    )
    return result


def get_num_layers(model_path: str, num_layers_override: Optional[int]) -> int:
    """Get the number of layers from model config."""
    if num_layers_override is not None:
        return num_layers_override

    # Try AutoConfig first (works for most models)
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config.num_hidden_layers
    except Exception:
        pass

    # Fallback: load raw config.json directly (handles models like DeepSeek-V3.2
    # whose model_type isn't registered in transformers yet)
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, "config.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        num_layers = raw_config.get("num_hidden_layers")
        if num_layers is not None:
            logger.info(
                f"Auto-detected num_layers={num_layers} from raw config.json"
            )
            return num_layers
    except Exception:
        pass

    # Fallback: try local path
    try:
        local_config = Path(model_path) / "config.json"
        if local_config.exists():
            with open(local_config) as f:
                raw_config = json.load(f)
            num_layers = raw_config.get("num_hidden_layers")
            if num_layers is not None:
                logger.info(
                    f"Auto-detected num_layers={num_layers} from local config.json"
                )
                return num_layers
    except Exception:
        pass

    logger.error(
        f"Could not auto-detect num_layers for {model_path}. "
        f"Please specify --num-layers."
    )
    sys.exit(1)


def generate_uniform_config(
    num_layers: int, target_ratio: float, output_path: str
):
    """Generate a uniform-spacing IndexCache config."""
    step = max(1, int(round(1.0 / target_ratio)))
    full_layers = sorted(set(range(0, num_layers, step)) | {0, 1})

    config = {
        "full_layers": full_layers,
        "num_layers": num_layers,
        "target_ratio": target_ratio,
        "actual_ratio": len(full_layers) / num_layers,
        "method": "uniform",
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Wrote uniform IndexCache config to {output_path}: "
        f"{len(full_layers)}/{num_layers} Full layers "
        f"(actual ratio={len(full_layers)/num_layers:.2%})"
    )
    print(f"\nConfig written to: {output_path}")
    print(f"Full layers ({len(full_layers)}/{num_layers}): {full_layers}")
    print(
        f"\nTo use: python -m sglang.launch_server --model MODEL "
        f"--index-cache-config {output_path}"
    )


def measure_similarity_report(
    layer_indices: Dict[int, np.ndarray], num_layers: int
):
    """Print a similarity report between consecutive layers."""
    print("\n" + "=" * 70)
    print("IndexCache Layer Similarity Analysis")
    print("=" * 70)
    print(f"\n{'Layer i':<10} {'Layer j':<10} {'Jaccard Similarity':>20}")
    print("-" * 40)

    similarities = []
    for i in range(num_layers - 1):
        j = i + 1
        if i in layer_indices and j in layer_indices:
            sim = compute_jaccard_similarity(layer_indices[i], layer_indices[j])
            similarities.append(sim)
            print(f"{i:<10} {j:<10} {sim:>20.4f}")

    if similarities:
        print("-" * 40)
        print(f"{'Mean':<20} {np.mean(similarities):>20.4f}")
        print(f"{'Median':<20} {np.median(similarities):>20.4f}")
        print(f"{'Min':<20} {np.min(similarities):>20.4f}")
        print(f"{'Max':<20} {np.max(similarities):>20.4f}")
        print(f"\nLayers with <80% similarity: ", end="")
        low_sim = [(i, s) for i, s in enumerate(similarities) if s < 0.8]
        if low_sim:
            print(", ".join(f"({i},{i+1})={s:.2f}" for i, s in low_sim))
        else:
            print("None (all layers have >80% similarity)")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate IndexCache Full/Shared layer assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production calibration (2K prompts, all domains, 2K-120K tokens):
  python scripts/index_cache_calibrate.py \\
      --model deepseek-ai/DeepSeek-V3.2 --tp 8 --target-ratio 0.25 \\
      --stratified -o config.json

  # Thorough calibration (more samples):
  python scripts/index_cache_calibrate.py \\
      --model deepseek-ai/DeepSeek-V3.2 --tp 8 --target-ratio 0.25 \\
      --stratified --calibration-samples 5000 -o config.json

  # Single-dataset calibration at specific length:
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --tp 8 --target-ratio 0.25 \\
      --calibration-dataset pg19 --calibration-seq-len 65536 \\
      --calibration-samples 128 -o config.json

  # Uniform config (no GPU needed):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --uniform --target-ratio 0.25 -o config.json

  # Analyze layer similarity:
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --tp 8 --measure-similarity --stratified
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HuggingFace ID"
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
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
        help="Number of layers (auto-detected from model config if not set)",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Use uniform spacing (no GPU needed)",
    )
    parser.add_argument(
        "--measure-similarity",
        action="store_true",
        help="Measure and report inter-layer index similarity (requires GPU)",
    )

    # Stratified calibration (recommended for production)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified multi-dataset, multi-length calibration. "
        "Samples from code, books, web, academic, QA datasets at lengths "
        "from 2K to 120K tokens, biased toward longer sequences. "
        "Recommended for production configs.",
    )

    # Single-dataset calibration (legacy / quick testing)
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="slimpajama",
        help=(
            f"Single calibration dataset (used when --stratified is not set). "
            f"Built-in: {', '.join(CALIBRATION_DATASETS.keys())}. "
            f"Default: slimpajama"
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Number of calibration samples. "
        "Default: 2000 for --stratified, 512 for single-dataset.",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=2048,
        help="Target sequence length per sample in tokens (single-dataset mode only, default: 2048)",
    )

    # Engine resource args
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to send per batch during calibration (default: 8). "
        "Lower this if you hit OOM with long prompts.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.80,
        help="GPU memory fraction for model weights/KV cache (default: 0.80). "
        "Lower than serving default to leave room for long-context prefill.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    num_layers = get_num_layers(args.model, args.num_layers)

    if args.uniform:
        generate_uniform_config(num_layers, args.target_ratio, args.output)
        return

    # Determine sample count
    if args.calibration_samples is not None:
        num_samples = args.calibration_samples
    elif args.stratified:
        num_samples = 2000
    else:
        num_samples = 512

    # Load calibration data
    if args.stratified:
        calibration_prompts = load_stratified_calibration_prompts(
            num_samples=num_samples,
            tokenizer_name=args.model,
        )
    else:
        calibration_prompts = load_calibration_prompts(
            dataset_name=args.calibration_dataset,
            num_samples=num_samples,
            seq_len=args.calibration_seq_len,
            tokenizer_name=args.model,
        )

    if args.measure_similarity:
        layer_indices = collect_indices(
            model_path=args.model,
            tp_size=args.tp,
            calibration_prompts=calibration_prompts,
            batch_size=args.batch_size,
            mem_fraction_static=args.mem_fraction_static,
        )
        measure_similarity_report(layer_indices, num_layers)
        return

    # Default: greedy calibration
    target_num_full = max(1, int(round(num_layers * args.target_ratio)))
    logger.info(
        f"Calibrating IndexCache for {args.model}: "
        f"{num_layers} layers, target {target_num_full} Full layers, "
        f"mode={'stratified' if args.stratified else args.calibration_dataset}, "
        f"samples={len(calibration_prompts)}"
    )

    layer_indices = collect_indices(
        model_path=args.model,
        tp_size=args.tp,
        calibration_prompts=calibration_prompts,
        batch_size=args.batch_size,
        mem_fraction_static=args.mem_fraction_static,
    )

    full_layers = greedy_layer_assignment(layer_indices, num_layers, target_num_full)

    config = {
        "full_layers": sorted(full_layers),
        "num_layers": num_layers,
        "target_ratio": args.target_ratio,
        "actual_ratio": len(full_layers) / num_layers,
        "method": "greedy_calibration",
        "calibration_mode": "stratified" if args.stratified else "single_dataset",
        "calibration_samples": len(calibration_prompts),
    }
    if args.stratified:
        config["calibration_length_strata"] = [
            {"min_tokens": s[0], "max_tokens": s[1], "fraction": s[2]}
            for s in LENGTH_STRATA
        ]
    else:
        config["calibration_dataset"] = args.calibration_dataset
        config["calibration_seq_len"] = args.calibration_seq_len

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Wrote calibrated config to {args.output}: "
        f"{len(full_layers)}/{num_layers} Full layers"
    )
    print(f"\nConfig written to: {args.output}")
    print(f"Full layers ({len(full_layers)}/{num_layers}): {sorted(full_layers)}")
    print(
        f"\nTo use: python -m sglang.launch_server --model MODEL "
        f"--index-cache-config {args.output}"
    )


if __name__ == "__main__":
    main()
