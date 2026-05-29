#!/usr/bin/env bash
# Run a fixed-shape TTFT sweep for H20 Kimi P/D experiments.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "proxy" ]]; then
  cat >&2 <<EOF
Usage: $0 {baseline|proxy}

Run on h20-99 after starting the matching service shape:
  baseline: direct vLLM server, same non-connector flags as P/D, usually port 18102
  proxy:    P/D cluster and proxy, usually proxy port 18100
EOF
  exit 2
fi

MODEL="${MODEL:-/data/models/Kimi-K2.5}"
VLLM_BIN="${VLLM_BIN:-/root/develop/xingming/pegaflow-rdma-e2e/.venv/bin/vllm}"
RESULT_DIR="${RESULT_DIR:-pd_h20_logs/bench/ttft-sweep}"
LENGTHS="${LENGTHS:-1024 4096 8192 16384 30000}"
NUM_PROMPTS=50
NUM_WARMUPS=0
OUTPUT_LEN=1
REQUEST_RATE=inf
MAX_CONCURRENCY=1
SEED="${SEED:-20260528}"
RANDOM_RANGE_RATIO=0.0
DECODE_SSH_HOST="${DECODE_SSH_HOST:-10.96.191.100}"
DECODE_HOST="${DECODE_HOST:-10.96.191.100}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/root/develop/xingming/pegaflow}"
NIC_SAMPLE_INTERVAL_S="${NIC_SAMPLE_INTERVAL_S:-1}"
NICS="${NICS:-mlx5_1 mlx5_2 mlx5_3 mlx5_4}"

if [[ "$MODE" == "baseline" ]]; then
  BASE_URL="${BASE_URL:-http://$DECODE_HOST:18102}"
  LABEL="${LABEL:-kimi-baseline-fixed32k}"
  MONITOR_NICS="${MONITOR_NICS:-0}"
else
  BASE_URL="${BASE_URL:-http://127.0.0.1:18100}"
  LABEL="${LABEL:-kimi-proxy-fixed32k}"
  MONITOR_NICS="${MONITOR_NICS:-1}"
fi

mkdir -p "$RESULT_DIR"

active_local_monitor_pid=""
active_remote_monitor_pid=""

monitor_local_nics() {
  local out_file="$1"
  (
    while true; do
      local ts
      ts="$(date +%s.%N)"
      for nic in $NICS; do
        local counters="/sys/class/infiniband/$nic/ports/1/counters"
        printf "%s,h20-99,%s,%s,%s\n" \
          "$ts" \
          "$nic" \
          "$(cat "$counters/port_xmit_data")" \
          "$(cat "$counters/port_rcv_data")"
      done
      sleep "$NIC_SAMPLE_INTERVAL_S"
    done
  ) >"$out_file" &
  echo "$!"
}

monitor_remote_nics() {
  local out_file="$1"
  ssh "$DECODE_SSH_HOST" \
    "cd '$REMOTE_REPO_ROOT' && mkdir -p '$RESULT_DIR' && NICS='$NICS' NIC_SAMPLE_INTERVAL_S='$NIC_SAMPLE_INTERVAL_S' bash -lc 'while true; do ts=\$(date +%s.%N); for nic in \$NICS; do counters=/sys/class/infiniband/\$nic/ports/1/counters; printf \"%s,h20-100,%s,%s,%s\\n\" \"\$ts\" \"\$nic\" \"\$(cat \$counters/port_xmit_data)\" \"\$(cat \$counters/port_rcv_data)\"; done; sleep \"\$NIC_SAMPLE_INTERVAL_S\"; done' > '$out_file'" &
  echo "$!"
}

stop_monitor() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup_active_monitors() {
  stop_monitor "$active_local_monitor_pid"
  stop_monitor "$active_remote_monitor_pid"
}
trap cleanup_active_monitors EXIT

summarize_nic_csv() {
  local csv="$1"
  awk -F, '
    {
      nic=$3
      if (!(nic in seen)) {
        seen[nic]=1
        ft[nic]=$1
        fx[nic]=$4
        fr[nic]=$5
      }
      if (nic in pt) {
        dt=$1-pt[nic]
        if (dt > 0) {
          x=($4-px[nic])*32/dt/1e9
          r=($5-pr[nic])*32/dt/1e9
          if (x>mx[nic]) mx[nic]=x
          if (r>mr[nic]) mr[nic]=r
        }
      }
      lt[nic]=$1
      lx[nic]=$4
      lr[nic]=$5
      pt[nic]=$1
      px[nic]=$4
      pr[nic]=$5
    }
    END {
      for (nic in seen) {
        dt=lt[nic]-ft[nic]
        if (dt <= 0) {
          continue
        }
        printf "%s dt=%.1f avg_xmit_gbps=%.2f peak_xmit_gbps=%.2f avg_rcv_gbps=%.2f peak_rcv_gbps=%.2f xmit_GB=%.2f rcv_GB=%.2f\n",
          nic, dt,
          (lx[nic]-fx[nic])*32/dt/1e9, mx[nic],
          (lr[nic]-fr[nic])*32/dt/1e9, mr[nic],
          (lx[nic]-fx[nic])*4/1e9,
          (lr[nic]-fr[nic])*4/1e9
      }
    }
  ' "$csv" | sort
}

for input_len in $LENGTHS; do
  run_label="${LABEL}-in${input_len}-out${OUTPUT_LEN}-c${MAX_CONCURRENCY}-n${NUM_PROMPTS}-seed${SEED}"
  result_json="${run_label}.json"
  local_nic_csv="$RESULT_DIR/${run_label}-h20-99-nic.csv"
  remote_nic_csv="$RESULT_DIR/${run_label}-h20-100-nic.csv"
  local_monitor_pid=""
  remote_monitor_pid=""

  echo "==> ${run_label}"
  if [[ "$MONITOR_NICS" == "1" ]]; then
    local_monitor_pid="$(monitor_local_nics "$local_nic_csv")"
    remote_monitor_pid="$(monitor_remote_nics "$remote_nic_csv")"
    active_local_monitor_pid="$local_monitor_pid"
    active_remote_monitor_pid="$remote_monitor_pid"
    sleep 2
  fi

  "$VLLM_BIN" bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --trust-remote-code \
    --dataset-name random \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --random-input-len "$input_len" \
    --random-output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --seed "$SEED" \
    --temperature 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_json" \
    --label "$run_label"

  if [[ "$MONITOR_NICS" == "1" ]]; then
    stop_monitor "$local_monitor_pid"
    stop_monitor "$remote_monitor_pid"
    active_local_monitor_pid=""
    active_remote_monitor_pid=""
    summarize_nic_csv "$local_nic_csv" >"$RESULT_DIR/${run_label}-h20-99-nic-summary.txt"
    ssh "$DECODE_SSH_HOST" "cd '$REMOTE_REPO_ROOT' && $(declare -f summarize_nic_csv); summarize_nic_csv '$remote_nic_csv'" \
      >"$RESULT_DIR/${run_label}-h20-100-nic-summary.txt"
  fi
done
