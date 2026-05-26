#!/usr/bin/env bash
# Quick local PD setup: proxy (GPU 0) + decode (GPU 1) + prefill (GPU 2)
# Usage: bash scripts/run_pd_local.sh [model_path]
set -euo pipefail

MODEL="${1:-/data/models/Qwen3-8B}"
PROXY_PORT=8100
DECODE_PORT=8101
PREFILL_PORT=8102
LOG_DIR="/tmp/pd_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/python/.venv/bin"

# GPU 1 → NIC mlx5_1 (PIX), GPU 2 → NIC mlx5_2 (PIX)
# rank_map: tp_rank 0 for each single-GPU node
DECODE_KV_CONFIG='{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.pd_connector","engine_id":"d0","kv_connector_extra_config":{"pegaflow.pd.rdma.rank_map":{"0":{"nic":"mlx5_1","worker_cpu":30}}}}'
PREFILL_KV_CONFIG='{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.pd_connector","engine_id":"p0","kv_connector_extra_config":{"pegaflow.pd.rdma.rank_map":{"0":{"nic":"mlx5_2","worker_cpu":60}}}}'

echo "============================================="
echo " PegaFlow P/D Local Setup"
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
PEGAFLOW_INSTANCE_ID=d0 \
"$VENV/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$DECODE_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "[D] PID=$DECODE_PID"

# --- Prefill (GPU 2) ---
echo "[P] Starting prefill on GPU 2, port $PREFILL_PORT ..."
CUDA_VISIBLE_DEVICES=2 \
PEGAFLOW_INSTANCE_ID=p0 \
"$VENV/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$PREFILL_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "[P] PID=$PREFILL_PID"

# --- Wait for vLLM servers to be ready ---
echo ""
echo "Waiting for vLLM servers to start..."
for name_port in "decode:$DECODE_PORT" "prefill:$PREFILL_PORT"; do
    name="${name_port%%:*}"
    port="${name_port##*:}"
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "  [$name] ready on port $port (${i}s)"
            break
        fi
        if ! kill -0 "$( [ "$name" = "decode" ] && echo $DECODE_PID || echo $PREFILL_PID )" 2>/dev/null; then
            echo "  [$name] process died. Check $LOG_DIR/${name}.log"
            exit 1
        fi
        sleep 1
    done
done

# --- Proxy (CPU, no GPU) ---
echo "[Proxy] Starting proxy on port $PROXY_PORT ..."
"$VENV/python" -m pegaflow.pd_connector.proxy \
    --listen-host 0.0.0.0 --listen-port "$PROXY_PORT" \
    --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
    --decode-url "http://127.0.0.1:$DECODE_PORT" \
    --timeout-s 600 \
    > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "[Proxy] PID=$PROXY_PID"

echo ""
echo "============================================="
echo " All processes started"
echo "============================================="
echo " Send requests to: http://0.0.0.0:$PROXY_PORT/v1/chat/completions"
echo ""
echo " tail -f $LOG_DIR/decode.log"
echo " tail -f $LOG_DIR/prefill.log"
echo " tail -f $LOG_DIR/proxy.log"
echo ""
echo " Quick test:"
echo '   curl http://127.0.0.1:'"$PROXY_PORT"'/v1/chat/completions -H "Content-Type: application/json" -d '\''{"model":"'"$MODEL"'","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'\'''
echo ""
echo " Ctrl+C to stop all."
echo "============================================="

wait
