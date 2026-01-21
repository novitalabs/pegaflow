#!/bin/bash
# Start PegaFlow and vLLM servers for CI testing
# This script manages three server types:
#   1. PegaFlow server (KV cache engine)
#   2. vLLM + PegaFlow (with KV connector)
#   3. vLLM Baseline (vanilla, for comparison)

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Server State
# =============================================================================

PEGA_SERVER_PID=""
VLLM_PEGA_PID=""
VLLM_BASELINE_PID=""

LOG_DIR="$RESULTS_DIR/logs"

# =============================================================================
# Usage
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
  start-all       Start all servers (PegaFlow + vLLM with PegaFlow + vLLM baseline)
  start-pega      Start only PegaFlow server
  start-vllm      Start vLLM with PegaFlow
  start-baseline  Start vLLM baseline (without PegaFlow)
  stop-all        Stop all servers
  status          Check server status

Options:
  --no-baseline   Skip starting baseline server
  -h, --help      Show this help message

Environment variables:
  MODEL           Model path (required)
  SERVED_MODEL_NAME   Served model name
  CHAT_TEMPLATE   Path to chat template
  VLLM_PORT       vLLM with PegaFlow port (default: 8000)
  VLLM_BASELINE_PORT  vLLM baseline port (default: 9000)
  PEGA_PORT       PegaFlow gRPC port (default: 50055)
  PEGA_POOL_SIZE  PegaFlow pool size (default: 50gb)

Examples:
  MODEL=/models/Llama-3.1-8B $0 start-all
  $0 stop-all
EOF
    exit 0
}

# =============================================================================
# PegaFlow Server
# =============================================================================

start_pegaflow_server() {
    log_section "Starting PegaFlow Server"
    
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/pegaflow-server.log"
    
    log_info "Building PegaFlow server..."
    cd "$PROJECT_ROOT"
    cargo build --release --bin pegaflow-server 2>&1 | tail -5
    
    log_info "Starting PegaFlow server..."
    log_info "  Address: 0.0.0.0:$PEGA_PORT"
    log_info "  Metrics: 0.0.0.0:$PEGA_METRICS_PORT"
    log_info "  Pool size: $PEGA_POOL_SIZE"
    log_info "  Device: $PEGA_DEVICE"
    log_info "  SSD cache: $PEGA_SSD_CACHE_PATH ($PEGA_SSD_CACHE_CAPACITY)"
    
    # Ensure SSD cache directory exists
    mkdir -p "$PEGA_SSD_CACHE_PATH"
    
    "$PROJECT_ROOT/target/release/pegaflow-server" \
        --addr "0.0.0.0:$PEGA_PORT" \
        --device "$PEGA_DEVICE" \
        --pool-size "$PEGA_POOL_SIZE" \
        --metrics-addr "0.0.0.0:$PEGA_METRICS_PORT" \
        --ssd-cache-path "$PEGA_SSD_CACHE_PATH" \
        --ssd-cache-capacity "$PEGA_SSD_CACHE_CAPACITY" \
        2>&1 | tee "$log_file" &
    
    PEGA_SERVER_PID=$!
    register_pid $PEGA_SERVER_PID
    echo "$PEGA_SERVER_PID" > "$RESULTS_DIR/.pegaflow.pid"
    
    log_info "PegaFlow server started with PID: $PEGA_SERVER_PID"
    
    # Wait for server to be ready
    wait_for_metrics "localhost" "$PEGA_METRICS_PORT" "PegaFlow server"
}

# =============================================================================
# vLLM with PegaFlow
# =============================================================================

start_vllm_with_pegaflow() {
    log_section "Starting vLLM with PegaFlow"
    
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/vllm-pegaflow.log"
    
    local served_name="${SERVED_MODEL_NAME:-$MODEL}"
    
    log_info "Starting vLLM with PegaFlow connector..."
    log_info "  Model: $MODEL"
    log_info "  Served name: $served_name"
    log_info "  Port: $VLLM_PORT"
    log_info "  Max sequences: $MAX_NUM_SEQS"
    
    # Build command
    local cmd="vllm serve \"$MODEL\""
    cmd="$cmd --served-model-name \"$served_name\""
    cmd="$cmd --trust-remote-code"
    cmd="$cmd --disable-log-requests"
    cmd="$cmd --host localhost"
    cmd="$cmd --port $VLLM_PORT"
    cmd="$cmd --max-num-seqs $MAX_NUM_SEQS"
    cmd="$cmd --max-model-len $MAX_MODEL_LEN"
    cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    
    if [[ -n "$CHAT_TEMPLATE" && -f "$CHAT_TEMPLATE" ]]; then
        cmd="$cmd --chat-template \"$CHAT_TEMPLATE\""
    fi
    
    # Add PegaFlow KV connector
    local kv_config='{"kv_connector": "PegaKVConnector", "kv_role": "kv_both", "kv_connector_module_path": "pegaflow.connector", "kv_connector_extra_config": {"pegaflow.host": "http://localhost", "pegaflow.port": '$PEGA_PORT'}}'
    cmd="$cmd --kv-transfer-config '$kv_config'"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES="${PEGA_DEVICE}"
    
    eval "$cmd" 2>&1 | tee "$log_file" &
    
    VLLM_PEGA_PID=$!
    register_pid $VLLM_PEGA_PID
    echo "$VLLM_PEGA_PID" > "$RESULTS_DIR/.vllm-pega.pid"
    
    log_info "vLLM with PegaFlow started with PID: $VLLM_PEGA_PID"
    
    # Wait for server to be ready
    wait_for_http "http://localhost:$VLLM_PORT/health" "vLLM with PegaFlow"
}

# =============================================================================
# vLLM Baseline (without PegaFlow)
# =============================================================================

start_vllm_baseline() {
    log_section "Starting vLLM Baseline"
    
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/vllm-baseline.log"
    
    local served_name="${SERVED_MODEL_NAME:-$MODEL}"
    
    log_info "Starting vanilla vLLM (baseline)..."
    log_info "  Model: $MODEL"
    log_info "  Served name: $served_name"
    log_info "  Port: $VLLM_BASELINE_PORT"
    log_info "  Max sequences: $MAX_NUM_SEQS"
    
    # Use a different GPU for baseline if available
    local baseline_device=1
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
    if [[ $gpu_count -lt 2 ]]; then
        baseline_device=$PEGA_DEVICE
        log_warn "Only one GPU available, running baseline on same device"
    fi
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=$baseline_device vllm serve \"$MODEL\""
    cmd="$cmd --served-model-name \"$served_name\""
    cmd="$cmd --trust-remote-code"
    cmd="$cmd --disable-log-requests"
    cmd="$cmd --host localhost"
    cmd="$cmd --port $VLLM_BASELINE_PORT"
    cmd="$cmd --max-num-seqs $MAX_NUM_SEQS"
    cmd="$cmd --max-model-len $MAX_MODEL_LEN"
    cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    
    if [[ -n "$CHAT_TEMPLATE" && -f "$CHAT_TEMPLATE" ]]; then
        cmd="$cmd --chat-template \"$CHAT_TEMPLATE\""
    fi
    
    eval "$cmd" 2>&1 | tee "$log_file" &
    
    VLLM_BASELINE_PID=$!
    register_pid $VLLM_BASELINE_PID
    echo "$VLLM_BASELINE_PID" > "$RESULTS_DIR/.vllm-baseline.pid"
    
    log_info "vLLM baseline started with PID: $VLLM_BASELINE_PID"
    
    # Wait for server to be ready
    wait_for_http "http://localhost:$VLLM_BASELINE_PORT/health" "vLLM baseline"
}

# =============================================================================
# Server Control Functions
# =============================================================================

start_all_servers() {
    local no_baseline="${1:-false}"
    
    log_section "Starting All Servers"
    print_config
    
    ensure_results_dir
    
    # Start PegaFlow first
    start_pegaflow_server
    
    # Start vLLM with PegaFlow
    start_vllm_with_pegaflow
    
    # Start baseline unless disabled
    if [[ "$no_baseline" != "true" ]]; then
        start_vllm_baseline
    fi
    
    log_section "All Servers Ready"
    echo "PegaFlow:          http://localhost:$PEGA_METRICS_PORT/metrics"
    echo "vLLM + PegaFlow:   http://localhost:$VLLM_PORT/v1/chat/completions"
    if [[ "$no_baseline" != "true" ]]; then
        echo "vLLM Baseline:     http://localhost:$VLLM_BASELINE_PORT/v1/chat/completions"
    fi
}

stop_all_servers() {
    log_section "Stopping All Servers"
    
    # Try to read PIDs from files
    for pidfile in "$RESULTS_DIR"/.*.pid; do
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Stopping process $pid from $pidfile"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
    
    cleanup_processes
}

check_status() {
    log_section "Server Status"
    
    echo -n "PegaFlow server:    "
    if curl -s -f "http://localhost:$PEGA_METRICS_PORT/metrics" >/dev/null 2>&1; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${RED}NOT RUNNING${NC}"
    fi
    
    echo -n "vLLM + PegaFlow:    "
    if check_vllm_health "$VLLM_PORT"; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${RED}NOT RUNNING${NC}"
    fi
    
    echo -n "vLLM Baseline:      "
    if check_vllm_health "$VLLM_BASELINE_PORT"; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${RED}NOT RUNNING${NC}"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    local no_baseline=false
    local command=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-baseline)
                no_baseline=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            start-all|start-pega|start-vllm|start-baseline|stop-all|status)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                usage
                ;;
        esac
    done
    
    if [[ -z "$command" ]]; then
        log_error "No command specified"
        usage
    fi
    
    # Validate configuration for start commands
    if [[ "$command" == start-* && "$command" != "start-pega" ]]; then
        validate_config || exit 1
    fi
    
    # Setup cleanup trap
    setup_cleanup_trap
    
    # Execute command
    case "$command" in
        start-all)
            start_all_servers "$no_baseline"
            ;;
        start-pega)
            ensure_results_dir
            start_pegaflow_server
            ;;
        start-vllm)
            ensure_results_dir
            start_vllm_with_pegaflow
            ;;
        start-baseline)
            ensure_results_dir
            start_vllm_baseline
            ;;
        stop-all)
            stop_all_servers
            ;;
        status)
            check_status
            ;;
    esac
}

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

