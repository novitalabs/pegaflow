#!/usr/bin/env bash
# Start or manage a two-node H20 P/D smoke setup for Kimi K2.5.
set -euo pipefail

ROLE="${1:-}"
MODEL="${MODEL:-/data/models/Kimi-K2.5}"
TP_SIZE=8
GPU_MEMORY_UTILIZATION=0.90
LOAD_FORMAT=dummy
MAX_NUM_BATCHED_TOKENS=32768
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PREFILL_HOST="${PREFILL_HOST:-10.96.191.99}"
DECODE_HOST="${DECODE_HOST:-10.96.191.100}"
DECODE_SSH_HOST="${DECODE_SSH_HOST:-$DECODE_HOST}"
PREFILL_PORT="${PREFILL_PORT:-18101}"
DECODE_PORT="${DECODE_PORT:-18102}"
PROXY_PORT="${PROXY_PORT:-18100}"
TIMEOUT_S="${TIMEOUT_S:-1800}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${VENV:-/root/develop/xingming/pegaflow-rdma-e2e/.venv}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/pd_h20_logs}"
mkdir -p "$LOG_DIR"

if [[ -z "${RANK_MAP:-}" ]]; then
  RANK_MAP='{"0":{"nic":"mlx5_1","worker_cpu":16},"1":{"nic":"mlx5_1","worker_cpu":30},"2":{"nic":"mlx5_2","worker_cpu":60},"3":{"nic":"mlx5_2","worker_cpu":90},"4":{"nic":"mlx5_3","worker_cpu":120},"5":{"nic":"mlx5_3","worker_cpu":150},"6":{"nic":"mlx5_4","worker_cpu":180},"7":{"nic":"mlx5_4","worker_cpu":210}}'
fi

usage() {
  cat <<EOF
Usage: $0 {start-baseline|start-prefill|start-decode|start-proxy|status|stop|smoke}
       $0 start-cluster  # run on the prefill/proxy node

Defaults:
  MODEL=$MODEL
  VENV=$VENV
  TP_SIZE=$TP_SIZE
  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
  LOAD_FORMAT=$LOAD_FORMAT
  MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
  PREFILL=http://$PREFILL_HOST:$PREFILL_PORT
  DECODE=http://$DECODE_HOST:$DECODE_PORT
  DECODE_SSH_HOST=$DECODE_SSH_HOST
  PROXY=http://0.0.0.0:$PROXY_PORT
  LOG_DIR=$LOG_DIR

No --block-size is passed; vLLM/model defaults decide the KV cache block shape.
No --max-model-len is passed; baseline and P/D runs keep the same model limit.
EOF
}

kv_config() {
  local engine_id="$1"
  "$VENV/bin/python" - "$engine_id" "$RANK_MAP" <<'PY'
import json
import sys

engine_id = sys.argv[1]
rank_map = json.loads(sys.argv[2])
print(json.dumps({
    "kv_connector": "PdConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "pegaflow.pd_connector",
    "engine_id": engine_id,
    "kv_connector_extra_config": {
        "pegaflow.pd.rdma.rank_map": rank_map,
    },
}, separators=(",", ":")))
PY
}

wait_ready() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local deadline=$((SECONDS + TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$name] ready on port $port"
      return
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[$name] exited early; check $log_file" >&2
      exit 1
    fi
    sleep 2
  done
  echo "[$name] did not become ready; check $log_file" >&2
  exit 1
}

wait_stopped() {
  local name="$1"
  local pid="$2"
  local deadline=$((SECONDS + 60))
  while kill -0 "$pid" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[$name] did not stop; sending SIGKILL pid=$pid"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
}

start_vllm() {
  local name="$1"
  local port="$2"
  local engine_id="$3"
  local log_file="$LOG_DIR/$name.log"
  local pid_file="$LOG_DIR/$name.pid"
  local config
  config="$(kv_config "$engine_id")"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "[$name] already running pid=$(cat "$pid_file")"
    return
  fi

  echo "[$name] starting on port $port, logs=$log_file"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PEGAFLOW_INSTANCE_ID="$engine_id" \
  PYTHONHASHSEED=0 \
  VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  PATH="$VENV/bin:$PATH" \
  nohup setsid "$VENV/bin/vllm" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --load-format "$LOAD_FORMAT" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config "$config" \
    >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  wait_ready "$name" "$port" "$pid" "$log_file"
}

start_baseline() {
  local log_file="$LOG_DIR/baseline.log"
  local pid_file="$LOG_DIR/baseline.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "[baseline] already running pid=$(cat "$pid_file")"
    return
  fi

  echo "[baseline] starting on port $DECODE_PORT, logs=$log_file"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONHASHSEED=0 \
  VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  PATH="$VENV/bin:$PATH" \
  nohup setsid "$VENV/bin/vllm" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --load-format "$LOAD_FORMAT" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  wait_ready baseline "$DECODE_PORT" "$pid" "$log_file"
}

start_proxy() {
  local log_file="$LOG_DIR/proxy.log"
  local pid_file="$LOG_DIR/proxy.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "[proxy] already running pid=$(cat "$pid_file")"
    return
  fi

  echo "[proxy] starting on port $PROXY_PORT, logs=$log_file"
  PATH="$VENV/bin:$PATH" \
  nohup setsid "$VENV/bin/python" -m pegaflow.pd_connector.proxy \
    --listen-host 0.0.0.0 \
    --listen-port "$PROXY_PORT" \
    --prefill-url "http://$PREFILL_HOST:$PREFILL_PORT" \
    --decode-url "http://$DECODE_HOST:$DECODE_PORT" \
    --timeout-s "$TIMEOUT_S" \
    >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  sleep 1
}

start_cluster() {
  local prefill_launch_log="$LOG_DIR/start-prefill.launch.log"
  local decode_launch_log="$LOG_DIR/start-decode.launch.log"
  local quoted_repo_root
  printf -v quoted_repo_root "%q" "$REPO_ROOT"

  echo "[cluster] starting prefill locally and decode on $DECODE_SSH_HOST"
  "$0" start-prefill >"$prefill_launch_log" 2>&1 &
  local prefill_launcher=$!
  ssh "$DECODE_SSH_HOST" "cd $quoted_repo_root && scripts/run_pd_h20_kimi.sh start-decode" \
    >"$decode_launch_log" 2>&1 &
  local decode_launcher=$!

  local failed=0
  wait "$prefill_launcher" || failed=1
  wait "$decode_launcher" || failed=1
  if [[ "$failed" != 0 ]]; then
    echo "[cluster] start failed; check $prefill_launch_log and $decode_launch_log" >&2
    exit 1
  fi

  start_proxy
}

stop_all() {
  for name in proxy prefill decode baseline; do
    local pid_file="$LOG_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" >/dev/null 2>&1; then
        echo "[$name] stopping pid=$pid"
        kill -- "-$pid" 2>/dev/null || kill "$pid" || true
        wait_stopped "$name" "$pid"
      fi
      rm -f "$pid_file"
    fi
  done
}

status() {
  for name in proxy prefill decode baseline; do
    local pid_file="$LOG_DIR/$name.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
      echo "[$name] running pid=$(cat "$pid_file")"
    else
      echo "[$name] stopped"
    fi
  done
  echo "logs: $LOG_DIR"
}

smoke() {
  curl -sS "http://127.0.0.1:$PROXY_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Write one sentence about RDMA.\",\"max_tokens\":16,\"temperature\":0}"
  echo
}

case "$ROLE" in
  start-baseline)
    start_baseline
    ;;
  start-prefill)
    start_vllm prefill "$PREFILL_PORT" prefill
    ;;
  start-decode)
    start_vllm decode "$DECODE_PORT" decode
    ;;
  start-proxy)
    start_proxy
    ;;
  start-cluster)
    start_cluster
    ;;
  status)
    status
    ;;
  stop)
    stop_all
    ;;
  smoke)
    smoke
    ;;
  *)
    usage
    exit 2
    ;;
esac
