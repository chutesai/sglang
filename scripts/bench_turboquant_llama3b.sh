#!/bin/bash
# =============================================================================
# TurboQuant Benchmark — Llama-3.2-3B-Instruct (MHA, single GPU)
#
# Compares bf16 baseline vs TurboQuant 4-bit KV cache compression.
# Designed for a single GPU (e.g. RTX 4090 24GB).
#
# Tests:
#   1. GSM8K           — Math reasoning (short context)
#   2. GPQA Diamond    — Science QA (short context)
#   3. IFEval          — Instruction following (short context)
#   4. RULER 4K        — Needle-in-haystack at 4K
#   5. RULER 32K       — Needle-in-haystack at 32K
#   6. Latency         — TTFT/throughput at various input lengths
#
# Usage:
#   ./scripts/bench_turboquant_llama3b.sh
#   PORT=30001 OUTDIR=/tmp/results ./scripts/bench_turboquant_llama3b.sh
# =============================================================================

set -uo pipefail

MODEL="${MODEL:-unsloth/Llama-3.2-3B-Instruct}"
REVISION="${REVISION:-006f5dcd1393c3add266de40994ba96225e9689d}"
TP="${TP:-1}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.85}"
OUTDIR="${OUTDIR:-/tmp/bench_tq_llama3b}"
CONTEXT="${CONTEXT:-32768}"

mkdir -p "${OUTDIR}"

passed=0
failed=0
declare -a results=()

# ---- Server management ----

wait_for_server() {
    local url="$1"
    local timeout="${2:-300}"
    local start=$(date +%s)
    while true; do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            return 0
        fi
        if (( $(date +%s) - start > timeout )); then
            return 1
        fi
        sleep 2
    done
}

launch_server() {
    local tag="$1"
    shift
    local logfile="${OUTDIR}/server_${tag}.log"
    echo "  Launching server [${tag}]..."
    python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --revision "${REVISION}" \
        --tp "${TP}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --context-length "${CONTEXT}" \
        --mem-fraction-static "${MEM}" \
        --trust-remote-code \
        --attention-backend triton \
        "$@" \
        > "${logfile}" 2>&1 &
    SERVER_PID=$!
    echo "  PID=${SERVER_PID}, log=${logfile}"

    if ! wait_for_server "http://127.0.0.1:${PORT}" 300; then
        echo "  ERROR: Server failed to start. Last 20 lines:"
        tail -20 "${logfile}"
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
        sleep 3
        SERVER_PID=""
    fi
}

# ---- Test runner ----

run_lm_eval() {
    local tasks="$1"
    local outfile="$2"
    shift 2
    python -m lm_eval \
        --model local-chat-completions \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=16,tokenized_requests=False" \
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
    local num_prompts="${4:-100}"
    python -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 --port "${PORT}" \
        --dataset-name random \
        --random-input "${input_len}" \
        --random-output "${output_len}" \
        --num-prompts "${num_prompts}" \
        --output-file "${OUTDIR}/latency_${tag}.jsonl" 2>&1 | tee "${OUTDIR}/latency_${tag}.txt"
}

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
# 1. GSM8K — Math word problems
# =============================================================================
run_comparison "gsm8k" \
    "Math reasoning (short context)" \
    run_lm_eval gsm8k

# =============================================================================
# 2. GPQA Diamond — Graduate-level science QA
# =============================================================================
run_comparison "gpqa" \
    "Graduate-level science QA (short context)" \
    run_lm_eval gpqa_diamond_cot_zeroshot

# =============================================================================
# 3. IFEval — Instruction following
# =============================================================================
run_comparison "ifeval" \
    "Instruction following (short context)" \
    run_lm_eval ifeval

# =============================================================================
# 4. RULER 4K — Needle-in-a-haystack at 4K
# =============================================================================
_ruler_4k() {
    run_lm_eval ruler "$1" --metadata "{\"max_seq_lengths\":[4096],\"pretrained\":\"${MODEL}\"}"
}
run_comparison "ruler_4k" \
    "RULER needle-in-haystack at 4K context" \
    _ruler_4k

# =============================================================================
# 5. RULER 32K — Needle-in-a-haystack at 32K
# =============================================================================
_ruler_32k() {
    run_lm_eval ruler "$1" --metadata "{\"max_seq_lengths\":[32768],\"pretrained\":\"${MODEL}\"}"
}
run_comparison "ruler_32k" \
    "RULER needle-in-haystack at 32K context" \
    _ruler_32k

# =============================================================================
# 6. Latency / throughput benchmark
# =============================================================================
echo ""
echo "================================================================"
echo "  TEST: latency"
echo "  TTFT/throughput at various input lengths"
echo "================================================================"

latency_ok=true
for input_len in 512 2048 8192 16384; do
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
echo "  SUMMARY — Llama-3.2-3B-Instruct TurboQuant Benchmark"
echo "================================================================"
echo ""
for r in "${results[@]}"; do
    echo "  ${r}"
done
echo ""
echo "  Passed: ${passed}  Failed: ${failed}"
echo "  Results saved to: ${OUTDIR}/"
echo "================================================================"
