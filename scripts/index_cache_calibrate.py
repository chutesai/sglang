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
       --model zai-org/GLM-5 --tp 8 --target-ratio 0.25 \\
       --calibration-samples 128 -o config.json

3. **Similarity analysis** (requires GPU, runs model offline):
   Measures and reports pairwise Jaccard similarity between consecutive
   layers to validate cross-layer index reuse assumptions.

   python scripts/index_cache_calibrate.py \\
       --model zai-org/GLM-5 --tp 8 --measure-similarity \\
       --calibration-samples 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Supported calibration datasets.
# Default is SlimPajama-6B which is a multi-source sample (C4, CommonCrawl,
# StackExchange, GitHub, Wikipedia) with proper val/test splits and manageable
# size. Research (EMNLP 2024, "Is C4 Dataset Optimal for Pruning?") shows
# source diversity matters for sparsity calibration — SlimPajama's multi-source
# nature makes it a good default.
CALIBRATION_DATASETS = {
    "slimpajama": {
        "hf_path": "DKYoon/SlimPajama-6B",
        "hf_name": None,
        "split": "validation",
        "text_column": "text",
    },
    "c4": {
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "split": "validation",
        "text_column": "text",
    },
    "pile": {
        "hf_path": "monology/pile-uncopyrighted",
        "hf_name": None,
        "split": "validation",
        "text_column": "text",
    },
    "wikitext": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
    },
    "redpajama": {
        "hf_path": "togethercomputer/RedPajama-Data-1T-Sample",
        "hf_name": None,
        "split": "train",
        "text_column": "text",
    },
}


def load_calibration_prompts(
    dataset_name: str,
    num_samples: int = 128,
    seq_len: int = 2048,
    tokenizer_name: Optional[str] = None,
) -> List[str]:
    """Load calibration prompts from a standard dataset.

    Samples documents, tokenizes to seq_len tokens, and returns as strings.
    This ensures consistent-length inputs for fair cross-layer comparison.

    Args:
        dataset_name: One of the CALIBRATION_DATASETS keys, or a HuggingFace
            dataset path (uses "text" column from "train" split).
        num_samples: Number of calibration samples to use.
        seq_len: Target sequence length in tokens per sample.
        tokenizer_name: Tokenizer to use for length normalization. If None,
            uses character-based approximation (4 chars ~= 1 token).
    """
    from datasets import load_dataset

    if dataset_name in CALIBRATION_DATASETS:
        ds_config = CALIBRATION_DATASETS[dataset_name]
        logger.info(
            f"Loading calibration dataset: {ds_config['hf_path']} "
            f"(split={ds_config['split']})"
        )
        ds = load_dataset(
            ds_config["hf_path"],
            ds_config["hf_name"],
            split=ds_config["split"],
            streaming=True,
        )
        text_column = ds_config["text_column"]
    else:
        # Treat as a HuggingFace dataset path
        logger.info(f"Loading custom calibration dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train", streaming=True)
        text_column = "text"

    # Try to use tokenizer for accurate length normalization
    tokenizer = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            logger.info(f"Using tokenizer {tokenizer_name} for length normalization")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using char approximation.")

    prompts = []
    for item in ds:
        if len(prompts) >= num_samples:
            break

        text = item.get(text_column, "")
        if not text or len(text.strip()) < 100:
            continue

        if tokenizer:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < seq_len // 2:
                continue  # Skip very short documents
            tokens = tokens[:seq_len]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            # Approximate: 4 chars per token
            char_len = seq_len * 4
            if len(text) < char_len // 2:
                continue
            prompt = text[:char_len]

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
    candidates = set(range(1, num_layers))  # Layer 0 always Full

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
        logger.info(
            f"Converted layer {best_layer} to Shared "
            f"(similarity={best_similarity:.4f}, "
            f"remaining Full={len(full_layers)})"
        )

    return full_layers


def collect_indices(
    model_path: str,
    tp_size: int,
    calibration_prompts: List[str],
    max_new_tokens: int = 1,
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
        log_level="info",
    )

    try:
        # Run calibration prompts through the engine
        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        logger.info(
            f"Running {len(calibration_prompts)} calibration prompts "
            f"(max_new_tokens={max_new_tokens})..."
        )
        engine.generate(calibration_prompts, sampling_params)
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
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config.num_hidden_layers
    except Exception as e:
        logger.error(
            f"Could not auto-detect num_layers: {e}. "
            f"Please specify --num-layers."
        )
        sys.exit(1)


def generate_uniform_config(
    num_layers: int, target_ratio: float, output_path: str
):
    """Generate a uniform-spacing IndexCache config."""
    step = max(1, int(round(1.0 / target_ratio)))
    full_layers = sorted(set(range(0, num_layers, step)))

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
Calibration datasets (--calibration-dataset):
  slimpajama  DKYoon/SlimPajama-6B validation split (default, multi-source)
  c4          allenai/c4 validation split
  pile        monology/pile-uncopyrighted validation split
  wikitext    wikitext-2-raw-v1 test split (simple, small)
  redpajama   RedPajama-Data-1T-Sample (diverse, large)
  <hf_path>   Any HuggingFace dataset with a "text" column

Examples:
  # Generate uniform config (no GPU needed):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --uniform --target-ratio 0.25 -o config.json

  # Greedy calibration with SlimPajama (default, requires GPU):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --tp 8 --target-ratio 0.25 \\
      --calibration-samples 128 -o config.json

  # Use Pile instead (better for sparsity per EMNLP 2024 findings):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --tp 8 --target-ratio 0.25 \\
      --calibration-dataset pile -o config.json

  # Analyze layer similarity (requires GPU):
  python scripts/index_cache_calibrate.py \\
      --model zai-org/GLM-5 --tp 8 --measure-similarity
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
        help="Use uniform spacing (recommended, no GPU needed)",
    )
    parser.add_argument(
        "--measure-similarity",
        action="store_true",
        help="Measure and report inter-layer index similarity (requires GPU)",
    )

    # Calibration data arguments
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="slimpajama",
        help=(
            f"Calibration dataset name or HuggingFace path. "
            f"Built-in: {', '.join(CALIBRATION_DATASETS.keys())}. "
            f"Default: slimpajama (DKYoon/SlimPajama-6B)"
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=2048,
        help="Target sequence length per sample in tokens (default: 2048)",
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

    # Load calibration data for GPU-based modes
    calibration_prompts = load_calibration_prompts(
        dataset_name=args.calibration_dataset,
        num_samples=args.calibration_samples,
        seq_len=args.calibration_seq_len,
        tokenizer_name=args.model,
    )

    if args.measure_similarity:
        layer_indices = collect_indices(
            model_path=args.model,
            tp_size=args.tp,
            calibration_prompts=calibration_prompts,
        )
        measure_similarity_report(layer_indices, num_layers)
        return

    # Default: greedy calibration
    target_num_full = max(1, int(round(num_layers * args.target_ratio)))
    logger.info(
        f"Calibrating IndexCache for {args.model}: "
        f"{num_layers} layers, target {target_num_full} Full layers, "
        f"dataset={args.calibration_dataset}, "
        f"samples={len(calibration_prompts)}"
    )

    layer_indices = collect_indices(
        model_path=args.model,
        tp_size=args.tp,
        calibration_prompts=calibration_prompts,
    )

    full_layers = greedy_layer_assignment(layer_indices, num_layers, target_num_full)

    config = {
        "full_layers": sorted(full_layers),
        "num_layers": num_layers,
        "target_ratio": args.target_ratio,
        "actual_ratio": len(full_layers) / num_layers,
        "method": "greedy_calibration",
        "calibration_dataset": args.calibration_dataset,
        "calibration_samples": len(calibration_prompts),
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Wrote calibrated config to {args.output}: "
        f"{len(full_layers)}/{num_layers} Full layers"
    )


if __name__ == "__main__":
    main()
