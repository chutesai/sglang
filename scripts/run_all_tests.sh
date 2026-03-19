#!/bin/bash
# =============================================================================
# IndexCache Benchmark Suite — Run each test independently
#
# Tests:
#   1. GSM8K          — Math reasoning (generate_until, chat API, short context)
#   2. GPQA Diamond   — Graduate-level science QA (generate_until, chat API, short)
#   3. IFEval         — Instruction following (generate_until, chat API, short)
#   4. RULER 4K       — Needle-in-haystack retrieval at 4K context (generate_until)
#   5. RULER 32K      — RULER at 32K context
#   6. RULER 131K     — RULER at 131K context (full production length)
#   7. BABILong       — Reasoning-in-haystack, qa1-qa5 (generate_until, long ctx)
#   8. Latency bench  — TTFT/throughput at various input lengths
#
# Each test launches baseline + IndexCache servers, runs the eval, prints
# comparison, and saves results. If one test OOMs or fails, the next still runs.
# =============================================================================

set -euo pipefail

# ---- Configuration ----
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
IC_CONFIG="${IC_CONFIG:-/cache/index_cache_dsv3.2_r0.3.json}"
OUTDIR="${OUTDIR:-/cache/bench_results}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.75}"

# Common server args (shared by baseline and IndexCache)
SERVER_ARGS="--dp 8 --enable-dp-attention --context-length 131072 --chunked-prefill-size 65536 --mem-fraction-static ${MEM} --trust-remote-code"

BENCH="python scripts/bench_index_cache.py"
COMMON="--model ${MODEL} --tp ${TP} --index-cache-config ${IC_CONFIG} --chat-model --port ${PORT} --server-timeout 900"

mkdir -p "${OUTDIR}"

passed=0
failed=0
skipped=0
declare -a results=()

run_test() {
    local name="$1"
    local description="$2"
    shift 2

    echo ""
    echo "================================================================"
    echo "  TEST: ${name}"
    echo "  ${description}"
    echo "================================================================"
    echo ""

    local outfile="${OUTDIR}/${name}.json"

    if ${BENCH} ${COMMON} \
        --extra-server-args "${SERVER_ARGS}" \
        --output "${outfile}" \
        "$@"; then
        echo ""
        echo "  >> ${name}: PASSED (results in ${outfile})"
        results+=("PASS  ${name}")
        ((passed++))
    else
        echo ""
        echo "  >> ${name}: FAILED (exit code $?)"
        results+=("FAIL  ${name}")
        ((failed++))
    fi
}

# ---- Dependencies check ----
echo "Checking dependencies..."
python -c "import lm_eval" 2>/dev/null || { echo "ERROR: pip install lm-eval"; exit 1; }
python -c "import langdetect" 2>/dev/null || { echo "WARNING: pip install langdetect immutabledict (needed for ifeval)"; }
echo "OK"
echo ""

# =============================================================================
# 1. GSM8K — Math word problems, chain-of-thought generation
#    Tests basic arithmetic reasoning. Short context (~500 tokens).
#    Metric: exact_match (extract answer after "####")
# =============================================================================
run_test "gsm8k" \
    "Math reasoning (short context, generate_until)" \
    --lm-eval-tasks gsm8k

# =============================================================================
# 2. GPQA Diamond — Graduate-level science multiple choice with CoT
#    Tests deep reasoning on physics/chemistry/biology. Short context.
#    198 questions, zero-shot CoT, model generates reasoning then answer.
#    Metric: exact_match
# =============================================================================
run_test "gpqa" \
    "Graduate-level science QA (short context, generate_until)" \
    --lm-eval-tasks gpqa_diamond_cot_zeroshot

# =============================================================================
# 3. IFEval — Instruction following evaluation
#    Tests whether model follows specific formatting instructions
#    (e.g. "write in all caps", "include exactly 3 paragraphs").
#    541 prompts, short context, generate_until.
#    Metric: prompt_level_strict_acc, inst_level_strict_acc
#    Requires: pip install langdetect immutabledict
# =============================================================================
run_test "ifeval" \
    "Instruction following (short context, generate_until)" \
    --lm-eval-tasks ifeval

# =============================================================================
# 4. RULER 4K — Needle-in-a-haystack at 4K context (default)
#    13 subtasks: single/multi needle retrieval, variable tracking,
#    common/frequent word extraction, QA (HotPotQA, SQuAD).
#    Tests attention quality at short context (sanity check).
#    500 samples per subtask, generate_until.
#    Metric: string_match (exact match of retrieved value)
# =============================================================================
run_test "ruler_4k" \
    "RULER needle-in-haystack at 4K context (generate_until)" \
    --lm-eval-tasks ruler \
    --lm-eval-metadata '{"max_seq_lengths":[4096],"pretrained":"'"${MODEL}"'"}'

# =============================================================================
# 5. RULER 32K — RULER at 32K context
#    Same 13 subtasks but with longer haystacks. This is where IndexCache
#    impact starts to show — attention patterns over 32K tokens.
#    Reduced concurrency to avoid OOM with longer sequences.
# =============================================================================
run_test "ruler_32k" \
    "RULER needle-in-haystack at 32K context (generate_until)" \
    --lm-eval-tasks ruler --num-concurrent 8 \
    --lm-eval-metadata '{"max_seq_lengths":[32768],"pretrained":"'"${MODEL}"'"}'

# =============================================================================
# 6. RULER 131K — RULER at full production context length
#    Maximum stress test for IndexCache. If calibration is good, scores
#    should be close to baseline. The paper shows greedy calibration
#    outperforms uniform by ~7 points at 1/4 retention on long context.
#    Low concurrency to avoid OOM — each request is 131K tokens.
# =============================================================================
run_test "ruler_131k" \
    "RULER needle-in-haystack at 131K context (generate_until)" \
    --lm-eval-tasks ruler --num-concurrent 2 \
    --lm-eval-metadata '{"max_seq_lengths":[131072],"pretrained":"'"${MODEL}"'"}'

# =============================================================================
# 7. BABILong — Reasoning-in-haystack (qa1-qa5)
#    Tests multi-hop reasoning buried in long documents. Generate_until.
#    qa1: single supporting fact, qa2: two facts, qa3: three facts,
#    qa4: two argument relations, qa5: three argument relations.
#    ~1000 samples per task, context lengths from dataset.
#    Metric: exact_match
#    Reduced concurrency for longer context samples.
# =============================================================================
run_test "babilong" \
    "BABILong reasoning-in-haystack qa1-qa5 (generate_until)" \
    --lm-eval-tasks babilong_longctx --num-concurrent 8

# =============================================================================
# 8. Latency / throughput benchmark
#    Measures TTFT (time-to-first-token), TPOT (time-per-output-token),
#    and throughput at various input lengths.
#    Uses bench_serving with synthetic random prompts.
#    No quality eval — pure performance measurement.
# =============================================================================
run_test "latency" \
    "Latency/throughput at various input lengths" \
    --skip-lm-eval --run-latency \
    --input-lens 1024 4096 16384 65536 131072

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo ""
for r in "${results[@]}"; do
    echo "  ${r}"
done
echo ""
echo "  Passed: ${passed}  Failed: ${failed}"
echo "  Results saved to: ${OUTDIR}/"
echo "================================================================"
