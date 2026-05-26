#!/usr/bin/env bash
# Quick local NIXL PD setup: proxy (CPU) + decode (GPU 1) + prefill (GPU 2)
# Mirrors run_pd_local.sh for A/B comparison on the same machine.
# Usage: bash scripts/run_nixl_local.sh [model_path]
set -euo pipefail

MODEL="${1:-/data/models/Qwen3-8B}"
PROXY_PORT=8100
DECODE_PORT=8101
PREFILL_PORT=8102
LOG_DIR="/tmp/nixl_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/python/.venv/bin"
PROXY_SCRIPT="$SCRIPT_DIR/nixl_proxy_server.py"

if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "ERROR: toy_proxy_server.py not found at $PROXY_SCRIPT"
    exit 1
fi

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

echo "============================================="
echo " NIXL P/D Local Setup"
echo "============================================="
echo " Model:   $MODEL"
echo " Proxy:   http://0.0.0.0:$PROXY_PORT"
echo " Decode:  http://0.0.0.0:$DECODE_PORT  (GPU 1)"
echo " Prefill: http://0.0.0.0:$PREFILL_PORT (GPU 2)"
echo " Logs:    $LOG_DIR"
echo "============================================="

cleanup() {
    echo ""
    echo "Shutting down..."
    kill 0 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Logs at: $LOG_DIR"
}
trap cleanup EXIT INT TERM

# --- Decode (GPU 1) ---
echo "[D] Starting decode on GPU 1, port $DECODE_PORT ..."
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
"$VENV/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$DECODE_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "[D] PID=$DECODE_PID"

# --- Prefill (GPU 2) ---
echo "[P] Starting prefill on GPU 2, port $PREFILL_PORT ..."
CUDA_VISIBLE_DEVICES=2 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
"$VENV/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$PREFILL_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "[P] PID=$PREFILL_PID"

# --- Wait for vLLM servers to be ready ---
echo ""
echo "Waiting for vLLM servers to start..."
for name_port in "decode:$DECODE_PORT:$DECODE_PID" "prefill:$PREFILL_PORT:$PREFILL_PID"; do
    IFS=: read -r name port pid <<< "$name_port"
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "  [$name] ready on port $port (${i}s)"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  [$name] process died. Check $LOG_DIR/${name}.log"
            exit 1
        fi
        sleep 1
    done
done

# --- Proxy (CPU, no GPU) ---
echo "[Proxy] Starting proxy on port $PROXY_PORT ..."
"$VENV/python" "$PROXY_SCRIPT" \
    --port "$PROXY_PORT" \
    --prefiller-host localhost --prefiller-port "$PREFILL_PORT" \
    --decoder-host localhost --decoder-port "$DECODE_PORT" \
    > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "[Proxy] PID=$PROXY_PID"

echo ""
echo "============================================="
echo " All processes started"
echo "============================================="
echo " Send requests to: http://0.0.0.0:$PROXY_PORT/v1/completions"
echo ""
echo " tail -f $LOG_DIR/decode.log"
echo " tail -f $LOG_DIR/prefill.log"
echo " tail -f $LOG_DIR/proxy.log"
echo ""
echo " Quick test:"
echo '   curl http://127.0.0.1:'"$PROXY_PORT"'/v1/completions -H "Content-Type: application/json" -d '\''{"model":"'"$MODEL"'","prompt":"San Francisco is a","max_tokens":32}'\'''
echo ""
echo " Ctrl+C to stop all."
echo "============================================="

wait
