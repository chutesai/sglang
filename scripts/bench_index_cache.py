"""Benchmark IndexCache quality and performance impact.

Launches an SGLang server in two configurations (baseline vs IndexCache),
runs lm-eval quality benchmarks and bench_serving latency/throughput tests,
and produces a comparison report.

Usage:
    # Quick quality check (GSM8K subset)
    python scripts/bench_index_cache.py \
        --model zai-org/GLM-5 --tp 8 \
        --index-cache-ratio 0.25

    # Full benchmark with latency tests
    python scripts/bench_index_cache.py \
        --model zai-org/GLM-5 --tp 8 \
        --index-cache-ratio 0.25 \
        --run-latency \
        --input-lens 1024 4096 16384 65536

    # Use calibrated config
    python scripts/bench_index_cache.py \
        --model zai-org/GLM-5 --tp 8 \
        --index-cache-config path/to/config.json

    # Custom lm-eval tasks
    python scripts/bench_index_cache.py \
        --model zai-org/GLM-5 --tp 8 \
        --index-cache-ratio 0.25 \
        --lm-eval-tasks gsm8k mmlu hellaswag

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
COMPLETIONS_ENDPOINT = "/v1/completions"
FLUSH_CACHE_ENDPOINT = "/flush_cache"


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
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(base_url, timeout=timeout):
        proc.kill()
        raise RuntimeError("Server failed to start")

    return proc


def kill_server(proc: subprocess.Popen):
    """Kill the server process tree."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.kill()
        proc.wait(timeout=30)
    except Exception:
        pass
    time.sleep(5)


def run_lm_eval(
    base_url: str,
    model_name: str,
    tasks: List[str],
    num_fewshot: int = 5,
    limit: Optional[int] = None,
    num_concurrent: int = 128,
    gen_kwargs: Optional[str] = None,
) -> Dict[str, Any]:
    """Run lm-eval harness against a running server."""
    import lm_eval

    # Flush cache before evaluation
    requests.get(f"{base_url}{FLUSH_CACHE_ENDPOINT}")

    model_args = {
        "model": model_name,
        "base_url": f"{base_url}{COMPLETIONS_ENDPOINT}",
        "num_concurrent": num_concurrent,
    }

    kwargs = dict(
        model="local-completions",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size="auto",
    )
    if limit is not None:
        kwargs["limit"] = limit
    if gen_kwargs is not None:
        kwargs["gen_kwargs"] = gen_kwargs

    logger.info(f"Running lm-eval: tasks={tasks}, limit={limit}")
    results = lm_eval.simple_evaluate(**kwargs)

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
            f"/tmp/bench_index_cache_{input_len}.json",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output_file = Path(f"/tmp/bench_index_cache_{input_len}.json")
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
    run_latency: bool,
    input_lens: List[int],
    output_len: int,
    server_timeout: int,
) -> BenchResult:
    """Run full benchmark for a single server configuration."""
    result = BenchResult(config_name=config_name, server_args=extra_server_args)

    proc = launch_server(model, base_url, tp, extra_server_args, timeout=server_timeout)
    try:
        # Quality benchmark
        if lm_eval_tasks:
            result.lm_eval_results = run_lm_eval(
                base_url=base_url,
                model_name=model,
                tasks=lm_eval_tasks,
                num_fewshot=lm_eval_num_fewshot,
                limit=lm_eval_limit,
                gen_kwargs=lm_eval_gen_kwargs,
            )

        # Latency benchmark
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
        print(f"{'Task':<20} {'Metric':<35} {'Baseline':>10} {'IndexCache':>10} {'Delta':>10}")
        print("-" * 85)
        for task in baseline.lm_eval_results:
            if task not in index_cache.lm_eval_results:
                continue
            for metric in baseline.lm_eval_results[task]:
                if metric not in index_cache.lm_eval_results[task]:
                    continue
                b_val = baseline.lm_eval_results[task][metric]
                ic_val = index_cache.lm_eval_results[task][metric]
                delta = ic_val - b_val
                print(f"{task:<20} {metric:<35} {b_val:>10.4f} {ic_val:>10.4f} {delta:>+10.4f}")

    # Latency comparison
    if baseline.latency_results and index_cache.latency_results:
        print("\n--- Latency ---")
        print(f"{'Input Len':<12} {'Metric':<25} {'Baseline':>12} {'IndexCache':>12} {'Speedup':>10}")
        print("-" * 71)
        for key in baseline.latency_results:
            if key not in index_cache.latency_results:
                continue
            b_data = baseline.latency_results[key]
            ic_data = index_cache.latency_results[key]
            input_len = b_data["input_len"]
            for metric in ["median_ttft_ms", "median_tpot_ms", "output_throughput"]:
                b_val = b_data.get(metric)
                ic_val = ic_data.get(metric)
                if b_val is None or ic_val is None:
                    continue
                if "throughput" in metric:
                    speedup = ic_val / b_val if b_val > 0 else float("inf")
                    print(f"{input_len:<12} {metric:<25} {b_val:>12.2f} {ic_val:>12.2f} {speedup:>9.2f}x")
                else:
                    speedup = b_val / ic_val if ic_val > 0 else float("inf")
                    print(f"{input_len:<12} {metric:<25} {b_val:>12.2f} {ic_val:>12.2f} {speedup:>9.2f}x")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IndexCache quality and performance impact"
    )

    # Model / server args
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument(
        "--extra-server-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional args passed to both baseline and IndexCache servers",
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

    # Quality benchmark args
    parser.add_argument(
        "--lm-eval-tasks",
        type=str,
        nargs="*",
        default=["gsm8k"],
        help="lm-eval tasks to run (default: gsm8k)",
    )
    parser.add_argument(
        "--lm-eval-limit",
        type=int,
        default=200,
        help="Max examples per task (default: 200, set 0 for unlimited)",
    )
    parser.add_argument(
        "--lm-eval-num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--lm-eval-gen-kwargs",
        type=str,
        default=None,
        help="Generation kwargs for lm-eval (e.g. 'max_gen_toks=2048')",
    )
    parser.add_argument(
        "--skip-lm-eval",
        action="store_true",
        help="Skip lm-eval quality benchmarks",
    )

    # Latency benchmark args
    parser.add_argument(
        "--run-latency",
        action="store_true",
        help="Run latency/throughput benchmarks",
    )
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="*",
        default=[1024, 4096, 16384],
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

    base_url = f"http://127.0.0.1:{args.port}"
    lm_eval_tasks = [] if args.skip_lm_eval else args.lm_eval_tasks
    lm_eval_limit = None if args.lm_eval_limit == 0 else args.lm_eval_limit

    # Build IndexCache server args
    ic_extra_args = list(args.extra_server_args)
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
            extra_server_args=list(args.extra_server_args),
            config_name="baseline",
            lm_eval_tasks=lm_eval_tasks,
            lm_eval_limit=lm_eval_limit,
            lm_eval_num_fewshot=args.lm_eval_num_fewshot,
            lm_eval_gen_kwargs=args.lm_eval_gen_kwargs,
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
        lm_eval_limit=lm_eval_limit,
        lm_eval_num_fewshot=args.lm_eval_num_fewshot,
        lm_eval_gen_kwargs=args.lm_eval_gen_kwargs,
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
