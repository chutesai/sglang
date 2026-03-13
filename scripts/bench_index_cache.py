"""Benchmark IndexCache quality and performance impact.

Launches an SGLang server in two configurations (baseline vs IndexCache),
runs lm-eval quality benchmarks and bench_serving latency/throughput tests,
and produces a comparison report.

Uses lm-eval-harness for all quality benchmarks:
  - local-completions: for loglikelihood tasks (MMLU, HellaSwag, ARC, etc.)
  - local-chat-completions: for generate_until tasks with chat models
    (GSM8K, GPQA Diamond CoT, IFEval, MATH, RULER NIAH, etc.)

Usage:
    # Chat model quality (GPQA Diamond zero-shot + GSM8K)
    python scripts/bench_index_cache.py \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --index-cache-ratio 0.25 \
        --chat-model \
        --lm-eval-tasks gpqa_diamond_cot_zeroshot gsm8k

    # Chat model with long-context RULER NIAH tasks
    python scripts/bench_index_cache.py \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --index-cache-ratio 0.25 \
        --chat-model \
        --lm-eval-tasks niah_single_1 niah_single_2 niah_single_3

    # Base model quality (MMLU, HellaSwag via loglikelihood)
    python scripts/bench_index_cache.py \
        --model zai-org/GLM-5 --tp 8 \
        --index-cache-ratio 0.25 \
        --lm-eval-tasks mmlu hellaswag

    # Latency-only benchmark
    python scripts/bench_index_cache.py \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --index-cache-ratio 0.25 \
        --skip-lm-eval --run-latency \
        --input-lens 1024 4096 16384 65536 131072

    # Calibrated config with full eval suite
    python scripts/bench_index_cache.py \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --index-cache-config path/to/config.json \
        --chat-model \
        --lm-eval-tasks gpqa_diamond_cot_zeroshot gsm8k ifeval

    # Task presets (shorthand for common task sets)
    python scripts/bench_index_cache.py \
        --model deepseek-ai/DeepSeek-V3.2 --tp 8 \
        --index-cache-ratio 0.25 --chat-model \
        --preset chat-quality          # gsm8k + gpqa_diamond_cot_zeroshot + ifeval
        --preset chat-long-context     # RULER NIAH tasks
        --preset chat-full             # all of the above

Requirements:
    pip install lm-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_PORT = 30000
HEALTH_ENDPOINT = "/health"
FLUSH_CACHE_ENDPOINT = "/flush_cache"

# Task presets for common evaluation scenarios.
# generate_until tasks work with both local-completions and local-chat-completions.
# loglikelihood tasks (mmlu, hellaswag, arc, etc.) only work with local-completions.
TASK_PRESETS = {
    # Chat model presets (generate_until only — work with local-chat-completions)
    "chat-quality": [
        "gsm8k",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
    ],
    "chat-long-context": [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
    ],
    "chat-full": [
        "gsm8k",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
    ],
    # Base model presets (loglikelihood — require local-completions)
    "base-quality": [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "truthfulqa_mc2",
    ],
    "base-full": [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "truthfulqa_mc2",
        "gsm8k",
    ],
}


@dataclass
class BenchResult:
    config_name: str
    lm_eval_results: Dict[str, Any] = field(default_factory=dict)
    latency_results: Dict[str, Any] = field(default_factory=dict)
    server_args: List[str] = field(default_factory=list)


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    """Wait for the server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}{HEALTH_ENDPOINT}", timeout=5)
            if resp.status_code == 200:
                logger.info(f"Server ready at {base_url}")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    logger.error(f"Server failed to start within {timeout}s")
    return False


def launch_server(
    model: str,
    base_url: str,
    tp: int,
    extra_args: List[str],
    timeout: int = 600,
) -> subprocess.Popen:
    """Launch an SGLang server and wait for it to be ready."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--tp",
        str(tp),
        "--port",
        str(base_url.split(":")[-1]),
        "--host",
        "127.0.0.1",
    ] + extra_args

    logger.info(f"Launching server: {' '.join(cmd)}")

    # Write server output to a log file to avoid pipe buffer deadlock.
    log_path = Path(f"/tmp/bench_index_cache_server_{os.getpid()}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    logger.info(f"Server PID={proc.pid}, log={log_path}")

    if not wait_for_server(base_url, timeout=timeout):
        kill_server(proc)
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            logger.error("Server log (last 50 lines):\n" + "\n".join(lines[-50:]))
        raise RuntimeError("Server failed to start")

    return proc


def kill_server(proc: subprocess.Popen):
    """Kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait(timeout=10)
    time.sleep(5)


def run_lm_eval(
    base_url: str,
    model_name: str,
    tasks: List[str],
    chat_model: bool = False,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    num_concurrent: int = 24,
    gen_kwargs: Optional[str] = None,
) -> Dict[str, Any]:
    """Run lm-eval harness against a running server.

    Args:
        chat_model: If True, use local-chat-completions (chat API).
                    If False, use local-completions (completions API).
        gen_kwargs: Generation kwargs string for lm-eval
                    (e.g. "max_gen_toks=65536,temperature=1.0,top_p=0.95").
                    When chat_model=True and gen_kwargs is None, defaults to
                    sensible values for CoT tasks.
    """
    try:
        import lm_eval
    except ImportError:
        logger.error(
            "lm_eval not installed. Install with: pip install lm-eval\n"
            "Skipping quality benchmark. Use --skip-lm-eval to suppress this."
        )
        return {}

    # Flush cache before evaluation
    requests.get(f"{base_url}{FLUSH_CACHE_ENDPOINT}")

    if chat_model:
        model_type = "local-chat-completions"
        model_args = {
            "model": model_name,
            "base_url": f"{base_url}/v1/chat/completions",
            "num_concurrent": num_concurrent,
            "tokenized_requests": False,
            "max_retries": 25,
            "max_length": 100000,
        }
        # Default gen_kwargs for chat models — CoT tasks need large
        # max_gen_toks or the response gets truncated before the answer.
        if gen_kwargs is None:
            gen_kwargs = "max_gen_toks=100000,temperature=1.0,top_p=0.95"
    else:
        model_type = "local-completions"
        model_args = {
            "model": model_name,
            "base_url": f"{base_url}/v1/completions",
            "num_concurrent": num_concurrent,
        }

    kwargs = dict(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size="auto",
    )
    if chat_model:
        kwargs["apply_chat_template"] = True
    if gen_kwargs:
        kwargs["gen_kwargs"] = gen_kwargs
    if limit is not None:
        kwargs["limit"] = limit

    logger.info(
        f"Running lm-eval: model_type={model_type}, tasks={tasks}, "
        f"num_fewshot={num_fewshot}, limit={limit}, gen_kwargs={gen_kwargs}"
    )

    # Set dummy API key for local server
    old_key = os.environ.get("OPENAI_API_KEY")
    if not old_key:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    try:
        results = lm_eval.simple_evaluate(**kwargs)
    finally:
        if not old_key:
            os.environ.pop("OPENAI_API_KEY", None)
        elif old_key:
            os.environ["OPENAI_API_KEY"] = old_key

    # Extract key metrics
    summary = {}
    for task_name, task_results in results.get("results", {}).items():
        summary[task_name] = {}
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)) and "stderr" not in metric_name:
                summary[task_name][metric_name] = value

    return summary


def run_latency_bench(
    base_url: str,
    input_lens: List[int],
    output_len: int = 128,
    num_prompts: int = 64,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Run bench_serving for latency measurements."""
    results = {}
    for input_len in input_lens:
        logger.info(f"Benchmarking latency: input_len={input_len}")
        output_file = Path(f"/tmp/bench_index_cache_{input_len}_{os.getpid()}.json")
        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--base-url",
            base_url,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(num_prompts),
            "--random-input",
            str(input_len),
            "--random-output",
            str(output_len),
            "--request-rate",
            str(concurrency),
            "--output-file",
            str(output_file),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if output_file.exists():
            with open(output_file) as f:
                bench_data = json.load(f)
            results[f"input_{input_len}"] = {
                "input_len": input_len,
                "output_len": output_len,
                "median_ttft_ms": bench_data.get("median_ttft_ms"),
                "median_tpot_ms": bench_data.get("median_tpot_ms"),
                "median_itl_ms": bench_data.get("median_itl_ms"),
                "output_throughput": bench_data.get("output_throughput"),
                "total_throughput": bench_data.get("total_throughput"),
            }
            output_file.unlink()
        else:
            logger.warning(
                f"No output for input_len={input_len}. "
                f"stderr: {proc.stderr[-500:] if proc.stderr else 'none'}"
            )

    return results


def run_benchmark_config(
    model: str,
    tp: int,
    base_url: str,
    extra_server_args: List[str],
    config_name: str,
    lm_eval_tasks: List[str],
    lm_eval_limit: Optional[int],
    lm_eval_num_fewshot: int,
    lm_eval_gen_kwargs: Optional[str],
    chat_model: bool,
    run_latency: bool,
    input_lens: List[int],
    output_len: int,
    server_timeout: int,
) -> BenchResult:
    """Run full benchmark for a single server configuration."""
    result = BenchResult(config_name=config_name, server_args=extra_server_args)

    proc = launch_server(model, base_url, tp, extra_server_args, timeout=server_timeout)
    try:
        # Warmup: send a few requests to trigger CUDA graph capture,
        # speculative decoding warmup, etc. before benchmarking.
        logger.info(f"Warming up server ({config_name})...")
        warmup_cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--base-url",
            base_url,
            "--dataset-name",
            "random",
            "--num-prompts",
            "16",
            "--random-input",
            "512",
            "--random-output",
            "64",
            "--request-rate",
            "1",
        ]
        subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=300)
        requests.get(f"{base_url}{FLUSH_CACHE_ENDPOINT}")
        logger.info("Warmup complete.")

        # Quality benchmark via lm-eval
        if lm_eval_tasks:
            result.lm_eval_results = run_lm_eval(
                base_url=base_url,
                model_name=model,
                tasks=lm_eval_tasks,
                chat_model=chat_model,
                num_fewshot=lm_eval_num_fewshot,
                limit=lm_eval_limit,
                gen_kwargs=lm_eval_gen_kwargs,
            )

        # Latency benchmark via bench_serving
        if run_latency:
            result.latency_results = run_latency_bench(
                base_url=base_url,
                input_lens=input_lens,
                output_len=output_len,
            )
    finally:
        kill_server(proc)

    return result


def print_comparison(baseline: BenchResult, index_cache: BenchResult):
    """Print a formatted comparison of baseline vs IndexCache results."""
    print("\n" + "=" * 80)
    print("IndexCache Benchmark Comparison")
    print("=" * 80)

    print(f"\nBaseline args:    {' '.join(baseline.server_args)}")
    print(f"IndexCache args:  {' '.join(index_cache.server_args)}")

    # Quality comparison
    if baseline.lm_eval_results and index_cache.lm_eval_results:
        print("\n--- Quality (lm-eval) ---")
        print(
            f"{'Task':<30} {'Metric':<30} {'Baseline':>10} "
            f"{'IndexCache':>10} {'Delta':>10}"
        )
        print("-" * 90)
        for task in baseline.lm_eval_results:
            if task not in index_cache.lm_eval_results:
                continue
            for metric in baseline.lm_eval_results[task]:
                if metric not in index_cache.lm_eval_results[task]:
                    continue
                b_val = baseline.lm_eval_results[task][metric]
                ic_val = index_cache.lm_eval_results[task][metric]
                delta = ic_val - b_val
                print(
                    f"{task:<30} {metric:<30} {b_val:>10.4f} "
                    f"{ic_val:>10.4f} {delta:>+10.4f}"
                )

    # Latency comparison
    if baseline.latency_results and index_cache.latency_results:
        print("\n--- Latency ---")
        print(
            f"{'Input Len':<12} {'Metric':<25} {'Baseline':>12} "
            f"{'IndexCache':>12} {'Speedup':>10}"
        )
        print("-" * 71)
        for key in baseline.latency_results:
            if key not in index_cache.latency_results:
                continue
            b_data = baseline.latency_results[key]
            ic_data = index_cache.latency_results[key]
            input_len = b_data["input_len"]
            for metric in [
                "median_ttft_ms",
                "median_tpot_ms",
                "output_throughput",
            ]:
                b_val = b_data.get(metric)
                ic_val = ic_data.get(metric)
                if b_val is None or ic_val is None:
                    continue
                if "throughput" in metric:
                    speedup = ic_val / b_val if b_val > 0 else float("inf")
                else:
                    speedup = b_val / ic_val if ic_val > 0 else float("inf")
                print(
                    f"{input_len:<12} {metric:<25} {b_val:>12.2f} "
                    f"{ic_val:>12.2f} {speedup:>9.2f}x"
                )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IndexCache quality and performance impact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task presets (use with --preset):
  chat-quality       gsm8k, gpqa_diamond_cot_zeroshot, ifeval
  chat-long-context  RULER NIAH tasks (niah_single_1/2/3, niah_multikey_1)
  chat-full          all of the above
  base-quality       mmlu, hellaswag, arc_challenge, winogrande, truthfulqa_mc2
  base-full          base-quality + gsm8k

Examples:
  # GPQA Diamond zero-shot (chat model)
  %(prog)s --model deepseek-ai/DeepSeek-V3.2 --tp 8 \\
      --index-cache-ratio 0.25 --chat-model \\
      --lm-eval-tasks gpqa_diamond_cot_zeroshot

  # Full chat eval suite
  %(prog)s --model deepseek-ai/DeepSeek-V3.2 --tp 8 \\
      --index-cache-ratio 0.25 --chat-model --preset chat-full

  # Latency only
  %(prog)s --model deepseek-ai/DeepSeek-V3.2 --tp 8 \\
      --index-cache-ratio 0.25 --skip-lm-eval --run-latency \\
      --input-lens 1024 4096 16384 65536 131072
""",
    )

    # Model / server args
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HF ID"
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Server port"
    )
    parser.add_argument(
        "--extra-server-args",
        type=str,
        default=None,
        help="Additional args passed to both baseline and IndexCache servers, "
        "as a single quoted string (e.g. '--reasoning-parser glm45 --mem-fraction-static 0.9')",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Timeout for server startup (seconds)",
    )

    # IndexCache config
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index-cache-ratio",
        type=float,
        help="IndexCache ratio (e.g. 0.25 = 25%% Full layers)",
    )
    group.add_argument(
        "--index-cache-config",
        type=str,
        help="Path to calibrated IndexCache JSON config",
    )

    # Model type
    parser.add_argument(
        "--chat-model",
        action="store_true",
        help="Use local-chat-completions (chat API) for lm-eval. "
        "Required for chat/instruct models. Only supports generate_until "
        "tasks (gsm8k, gpqa_diamond_cot_zeroshot, ifeval, RULER NIAH, etc.)",
    )

    # Quality benchmark args
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(TASK_PRESETS.keys()),
        default=None,
        help="Use a predefined task set (overrides --lm-eval-tasks)",
    )
    parser.add_argument(
        "--lm-eval-tasks",
        type=str,
        nargs="*",
        default=None,
        help="lm-eval task names (default: gsm8k). See lm-eval docs for full list.",
    )
    parser.add_argument(
        "--lm-eval-limit",
        type=int,
        default=None,
        help="Max examples per task (default: unlimited, set for faster runs)",
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0 = zero-shot)",
    )
    parser.add_argument(
        "--skip-lm-eval",
        action="store_true",
        help="Skip lm-eval quality benchmarks entirely",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=str,
        default=None,
        help="Generation kwargs for lm-eval (e.g. "
        "'max_gen_toks=65536,temperature=1.0,top_p=0.95'). "
        "For --chat-model, defaults to max_gen_toks=65536,temperature=1.0,top_p=0.95 "
        "if not specified (required for CoT tasks).",
    )

    # Latency benchmark args
    parser.add_argument(
        "--run-latency",
        action="store_true",
        help="Run latency/throughput benchmarks via bench_serving",
    )
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="*",
        default=[1024, 4096, 16384, 32768, 65536],
        help="Input lengths for latency tests",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length for latency tests",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (useful if you already have baseline numbers)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve tasks
    if args.skip_lm_eval:
        lm_eval_tasks = []
    elif args.preset:
        lm_eval_tasks = TASK_PRESETS[args.preset]
        logger.info(f"Using preset '{args.preset}': {lm_eval_tasks}")
    elif args.lm_eval_tasks:
        lm_eval_tasks = args.lm_eval_tasks
    else:
        # Default based on model type
        lm_eval_tasks = ["gsm8k", "gpqa_diamond_cot_zeroshot"] if args.chat_model else ["gsm8k"]

    base_url = f"http://127.0.0.1:{args.port}"
    extra_server_args = args.extra_server_args.split() if args.extra_server_args else []

    # Build IndexCache server args
    ic_extra_args = list(extra_server_args)
    if args.index_cache_ratio is not None:
        ic_extra_args += ["--index-cache-ratio", str(args.index_cache_ratio)]
        ic_name = f"IndexCache(ratio={args.index_cache_ratio})"
    else:
        ic_extra_args += ["--index-cache-config", args.index_cache_config]
        ic_name = f"IndexCache(config={args.index_cache_config})"

    results = {}

    # Run baseline
    if not args.skip_baseline:
        logger.info("=" * 40 + " BASELINE " + "=" * 40)
        baseline = run_benchmark_config(
            model=args.model,
            tp=args.tp,
            base_url=base_url,
            extra_server_args=list(extra_server_args),
            config_name="baseline",
            lm_eval_tasks=lm_eval_tasks,
            lm_eval_limit=args.lm_eval_limit,
            lm_eval_num_fewshot=args.lm_eval_num_fewshot,
            lm_eval_gen_kwargs=args.gen_kwargs,
            chat_model=args.chat_model,
            run_latency=args.run_latency,
            input_lens=args.input_lens,
            output_len=args.output_len,
            server_timeout=args.server_timeout,
        )
        results["baseline"] = {
            "lm_eval": baseline.lm_eval_results,
            "latency": baseline.latency_results,
        }
    else:
        baseline = BenchResult(config_name="baseline")

    # Run IndexCache
    logger.info("=" * 40 + f" {ic_name} " + "=" * 40)
    index_cache = run_benchmark_config(
        model=args.model,
        tp=args.tp,
        base_url=base_url,
        extra_server_args=ic_extra_args,
        config_name=ic_name,
        lm_eval_tasks=lm_eval_tasks,
        lm_eval_limit=args.lm_eval_limit,
        lm_eval_num_fewshot=args.lm_eval_num_fewshot,
        lm_eval_gen_kwargs=args.gen_kwargs,
        chat_model=args.chat_model,
        run_latency=args.run_latency,
        input_lens=args.input_lens,
        output_len=args.output_len,
        server_timeout=args.server_timeout,
    )
    results["index_cache"] = {
        "config": ic_name,
        "lm_eval": index_cache.lm_eval_results,
        "latency": index_cache.latency_results,
    }

    # Print comparison
    if not args.skip_baseline:
        print_comparison(baseline, index_cache)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
