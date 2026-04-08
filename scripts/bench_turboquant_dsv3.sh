#!/bin/bash
# =============================================================================
# TurboQuant Benchmark — DeepSeek-V3-0324 (MLA, multi-GPU)
#
# Compares bf16 baseline vs TurboQuant 4-bit KV cache compression.
# Requires 8 GPUs (e.g. 8xH100/A100). Uses dp-attention for throughput.
#
# Tests:
#   1. GSM8K           — Math reasoning (short context)
#   2. GPQA Diamond    — Science QA with CoT (short context)
#   3. IFEval          — Instruction following (short context)
#   4. RULER 4K        — Needle-in-haystack at 4K (sanity)
#   5. RULER 32K       — Needle-in-haystack at 32K
#   6. RULER 128K      — Needle-in-haystack at 128K (long context stress)
#   7. BABILong        — Reasoning-in-haystack qa1-qa5
#   8. Latency         — TTFT/throughput at various input lengths
#
# Usage:
#   ./scripts/bench_turboquant_dsv3.sh
#   TP=8 DP=8 PORT=30000 ./scripts/bench_turboquant_dsv3.sh
# =============================================================================

set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V3-0324}"
TP="${TP:-8}"
DP="${DP:-8}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.80}"
OUTDIR="${OUTDIR:-/tmp/bench_tq_dsv3}"
CONTEXT="${CONTEXT:-131072}"

# Common server args
BASE_SERVER_ARGS="--dp ${DP} --enable-dp-attention --context-length ${CONTEXT} --chunked-prefill-size 65536 --mem-fraction-static ${MEM} --trust-remote-code --attention-backend triton"

mkdir -p "${OUTDIR}"

passed=0
failed=0
declare -a results=()

# ---- Server management ----

wait_for_server() {
    local url="$1"
    local timeout="${2:-900}"
    local start=$(date +%s)
    while true; do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            return 0
        fi
        if (( $(date +%s) - start > timeout )); then
            return 1
        fi
        sleep 5
    done
}

launch_server() {
    local tag="$1"
    shift
    local logfile="${OUTDIR}/server_${tag}.log"
    echo "  Launching server [${tag}]..."
    python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --tp "${TP}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        ${BASE_SERVER_ARGS} \
        "$@" \
        > "${logfile}" 2>&1 &
    SERVER_PID=$!
    echo "  PID=${SERVER_PID}, log=${logfile}"

    if ! wait_for_server "http://127.0.0.1:${PORT}" 900; then
        echo "  ERROR: Server failed to start. Last 30 lines:"
        tail -30 "${logfile}"
        kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null
        return 1
    fi
    echo "  Server ready."
    return 0
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null
        sleep 5
        SERVER_PID=""
    fi
}

# ---- Eval helpers ----

run_lm_eval() {
    local tasks="$1"
    local outfile="$2"
    shift 2
    python -m lm_eval \
        --model local-chat-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=24,tokenized_requests=False" \
        --tasks "${tasks}" \
        --batch_size 1 \
        --apply_chat_template \
        --output_path "${outfile}" \
        --log_samples \
        "$@"
}

run_bench_serving() {
    local tag="$1"
    local input_len="$2"
    local output_len="${3:-128}"
    local num_prompts="${4:-200}"
    python -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 --port "${PORT}" \
        --dataset-name random \
        --random-input "${input_len}" \
        --random-output "${output_len}" \
        --num-prompts "${num_prompts}" \
        --output-file "${OUTDIR}/latency_${tag}.jsonl" 2>&1 | tee "${OUTDIR}/latency_${tag}.txt"
}

# ---- Test runner ----

run_comparison() {
    local name="$1"
    local description="$2"
    shift 2

    echo ""
    echo "================================================================"
    echo "  TEST: ${name}"
    echo "  ${description}"
    echo "================================================================"

    local baseline_out="${OUTDIR}/${name}_baseline"
    local tq_out="${OUTDIR}/${name}_turboquant"
    local ok=true

    # --- Baseline ---
    echo "  >> Baseline (bf16 KV cache)..."
    if launch_server "baseline_${name}"; then
        if "$@" "${baseline_out}"; then
            echo "  >> Baseline: done"
        else
            echo "  >> Baseline: eval FAILED"
            ok=false
        fi
        kill_server
    else
        ok=false
    fi

    # --- TurboQuant ---
    echo "  >> TurboQuant 4-bit..."
    if launch_server "tq_${name}" --kv-cache-dtype turboquant --turboquant-bits 4 --turboquant-mode mse; then
        if "$@" "${tq_out}"; then
            echo "  >> TurboQuant: done"
        else
            echo "  >> TurboQuant: eval FAILED"
            ok=false
        fi
        kill_server
    else
        ok=false
    fi

    if ${ok}; then
        echo "  >> ${name}: PASSED"
        results+=("PASS  ${name}")
        passed=$((passed + 1))
    else
        echo "  >> ${name}: FAILED"
        results+=("FAIL  ${name}")
        failed=$((failed + 1))
    fi
}

# ---- Dependency check ----
echo "Checking dependencies..."
python -c "import lm_eval" 2>/dev/null || { echo "ERROR: pip install lm-eval"; exit 1; }
python -c "import langdetect" 2>/dev/null || echo "WARNING: pip install langdetect immutabledict (needed for ifeval)"
echo "OK"

# =============================================================================
# Short-context quality benchmarks
# =============================================================================

run_comparison "gsm8k" \
    "Math reasoning (short context)" \
    run_lm_eval gsm8k

run_comparison "gpqa" \
    "Graduate-level science QA with CoT (short context)" \
    run_lm_eval gpqa_diamond_cot_zeroshot

run_comparison "ifeval" \
    "Instruction following (short context)" \
    run_lm_eval ifeval

# =============================================================================
# Long-context quality benchmarks
# =============================================================================

_ruler_4k() {
    run_lm_eval ruler "$1" --metadata "{\"max_seq_lengths\":[4096],\"pretrained\":\"${MODEL}\"}"
}
run_comparison "ruler_4k" \
    "RULER needle-in-haystack at 4K context (sanity)" \
    _ruler_4k

_ruler_32k() {
    run_lm_eval ruler "$1" --metadata "{\"max_seq_lengths\":[32768],\"pretrained\":\"${MODEL}\"}"
}
run_comparison "ruler_32k" \
    "RULER needle-in-haystack at 32K context" \
    _ruler_32k

_ruler_128k() {
    # Lower concurrency for 128K — each request is huge
    python -m lm_eval \
        --model local-chat-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=4,tokenized_requests=False" \
        --tasks ruler \
        --batch_size 1 \
        --apply_chat_template \
        --output_path "$1" \
        --log_samples \
        --metadata "{\"max_seq_lengths\":[128000],\"pretrained\":\"${MODEL}\"}"
}
run_comparison "ruler_128k" \
    "RULER needle-in-haystack at 128K context (long context stress)" \
    _ruler_128k

_babilong() {
    python -m lm_eval \
        --model local-chat-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=8,tokenized_requests=False" \
        --tasks babilong_longctx \
        --batch_size 1 \
        --apply_chat_template \
        --output_path "$1" \
        --log_samples
}
run_comparison "babilong" \
    "BABILong reasoning-in-haystack qa1-qa5 (long context)" \
    _babilong

# =============================================================================
# Latency / throughput benchmark
# =============================================================================
echo ""
echo "================================================================"
echo "  TEST: latency"
echo "  TTFT/throughput at various input lengths"
echo "================================================================"

latency_ok=true
for input_len in 1024 4096 16384 65536 131072; do
    echo "  >> Baseline input_len=${input_len}..."
    if launch_server "baseline_lat_${input_len}"; then
        run_bench_serving "baseline_${input_len}" "${input_len}"
        kill_server
    else
        latency_ok=false
    fi

    echo "  >> TurboQuant input_len=${input_len}..."
    if launch_server "tq_lat_${input_len}" --kv-cache-dtype turboquant --turboquant-bits 4 --turboquant-mode mse; then
        run_bench_serving "tq_${input_len}" "${input_len}"
        kill_server
    else
        latency_ok=false
    fi
done

if ${latency_ok}; then
    results+=("PASS  latency")
    passed=$((passed + 1))
else
    results+=("FAIL  latency")
    failed=$((failed + 1))
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  SUMMARY — DeepSeek-V3-0324 (MLA) TurboQuant Benchmark"
echo "================================================================"
echo ""
for r in "${results[@]}"; do
    echo "  ${r}"
done
echo ""
echo "  Passed: ${passed}  Failed: ${failed}"
echo "  Results saved to: ${OUTDIR}/"
echo "================================================================"
