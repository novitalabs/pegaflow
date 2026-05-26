#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:?set BASE_URL, for example http://127.0.0.1:8300}"
MODEL="${MODEL:?set MODEL, for example /data/models/Kimi-K2.5}"
LABEL="${LABEL:?set LABEL, for example tp8-baseline}"
RESULT_DIR="${RESULT_DIR:-/tmp/pegaflow-ttft-bench}"
VLLM_BIN="${VLLM_BIN:-vllm}"
SEED="${SEED:-20260521}"
NUM_PROMPTS="${NUM_PROMPTS:-30}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
LENGTHS="${LENGTHS:-1024 4096 8192 10000 16384 30000}"

mkdir -p "$RESULT_DIR"

for input_len in $LENGTHS; do
  run_label="${LABEL}-in${input_len}-out${OUTPUT_LEN}-c${MAX_CONCURRENCY}-n${NUM_PROMPTS}-seed${SEED}"
  echo "==> ${run_label}"
  "$VLLM_BIN" bench serve \
    --backend openai \
      --base-url "$BASE_URL" \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --trust-remote-code \
      --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --seed "$SEED" \
    --temperature 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "${run_label}.json" \
    --label "$run_label"
done
