#!/bin/bash
#
# Comprehensive TQ vs Baseline benchmark suite
# Starts servers, runs benchmarks, stops servers — fully automated.
#
# Usage: bash scripts/multi_bench_tq.sh
#

set -uo pipefail

cd ~/git/sglang
source venv/bin/activate

MODEL="Qwen/Qwen3-8B"
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
PORT=30000
HOST="127.0.0.1"
BENCH_COMMON="--backend sglang --host ${HOST} --port ${PORT} --request-rate 20 --max-concurrency 20"
BENCH_CHAT="--backend sglang-oai-chat --host ${HOST} --port ${PORT} --request-rate 20 --max-concurrency 20"

SERVER_COMMON="--model-path ${MODEL} --revision ${REVISION} --tp 1 \
  --port ${PORT} --host ${HOST} --context-length 32768 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 24"

# ===================================================================
# Helper functions
# ===================================================================

start_server() {
    local label="$1"
    shift
    echo "=== Starting server: ${label} ==="
    echo "=== Server args: $* ==="
    python -m sglang.launch_server $* > "server_${label}.log" 2>&1 &
    SERVER_PID=$!
    echo "=== Server PID: ${SERVER_PID} ==="

    # Wait for server to be ready
    echo "=== Waiting for server to be ready... ==="
    local elapsed=0
    local timeout=900
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "=== Server ready after ${elapsed}s ==="
            return 0
        fi
        # Check server hasn't crashed
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "=== ERROR: Server process died. Check server_${label}.log ==="
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "=== ERROR: Server not ready after ${timeout}s ==="
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "=== Stopping server (PID ${SERVER_PID}) ==="
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null || true
        sleep 5
        echo "=== Server stopped ==="
    fi
}

run_benchmarks() {
    local label="$1"
    local log="bench_${label}.log"

    echo "=== Benchmark suite: ${label} ===" | tee "$log"
    echo "=== Started: $(date) ===" | tee -a "$log"

    # 1. Single batch, short context — pure TPOT measurement
    echo -e "\n=== Test 1: Single batch, short context (TPOT focus) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_COMMON} \
        --dataset-name random \
        --num-prompts 50 \
        --random-input-len 512 \
        --random-output-len 512 \
        --max-concurrency 1 \
        --request-rate 1 2>&1 | tee -a "$log"

    # 2. Short prompts, moderate output — chatbot-like
    echo -e "\n=== Test 2: Short prompts, moderate output (chatbot) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_COMMON} \
        --dataset-name random \
        --num-prompts 300 \
        --random-input-len 256 \
        --random-output-len 1024 2>&1 | tee -a "$log"

    # 3. Medium context, long output — code generation / CoT
    echo -e "\n=== Test 3: Medium context, long output (code gen) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_COMMON} \
        --dataset-name random \
        --num-prompts 200 \
        --random-input-len 4096 \
        --random-output-len 2048 2>&1 | tee -a "$log"

    # 4. Long context, long output — document QA
    echo -e "\n=== Test 4: Long context, long output (doc QA) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_COMMON} \
        --dataset-name random \
        --num-prompts 100 \
        --random-input-len 8192 \
        --random-output-len 2048 2>&1 | tee -a "$log"

    # 5. Shared prefix, multi-turn — prefix cache stress test
    echo -e "\n=== Test 5: Shared prefix, multi-turn (prefix cache) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_CHAT} \
        --dataset-name generated-shared-prefix \
        --gsp-num-groups 10 \
        --gsp-prompts-per-group 20 \
        --gsp-system-prompt-len 4096 \
        --gsp-question-len 256 \
        --gsp-output-len 1024 \
        --gsp-num-turns 5 2>&1 | tee -a "$log"

    # 6. Shared prefix, single-turn, many groups — prefix cache hit rate
    echo -e "\n=== Test 6: Shared prefix, single-turn (cache hit rate) ===" | tee -a "$log"
    python -m sglang.bench_serving ${BENCH_COMMON} \
        --dataset-name generated-shared-prefix \
        --gsp-num-groups 5 \
        --gsp-prompts-per-group 40 \
        --gsp-system-prompt-len 2048 \
        --gsp-question-len 512 \
        --gsp-output-len 512 2>&1 | tee -a "$log"

    echo -e "\n=== Finished: $(date) ===" | tee -a "$log"
    echo "=== Results saved to ${log} ===" | tee -a "$log"
}

# Clean up server on exit
trap stop_server EXIT

# ===================================================================
# Phase 1: TurboQuant
# ===================================================================

echo ""
echo "###############################################################"
echo "# Phase 1: TurboQuant"
echo "###############################################################"
echo ""

start_server "tq" ${SERVER_COMMON} \
    --kv-cache-dtype turboquant --turboquant-bits 4 --turboquant-mode mse \
    --trust-remote-code

run_benchmarks "tq"
stop_server

# ===================================================================
# Phase 2: Baseline (no TQ)
# ===================================================================

echo ""
echo "###############################################################"
echo "# Phase 2: Baseline"
echo "###############################################################"
echo ""

start_server "baseline" ${SERVER_COMMON}

run_benchmarks "baseline"
stop_server

echo ""
echo "###############################################################"
echo "# All done! Results in bench_tq.log and bench_baseline.log"
echo "# Server logs in server_tq.log and server_baseline.log"
echo "###############################################################"
