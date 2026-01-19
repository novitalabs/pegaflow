#!/bin/bash
# Common utilities for PegaFlow CI tests
# This script is sourced by other test scripts to provide shared functions.

set -e
set -o pipefail

# =============================================================================
# Configuration (can be overridden via environment variables)
# =============================================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build ID for unique result directories
BUILD_ID="${BUILD_ID:-$(date +%Y%m%d-%H%M%S)}"

# Results directory
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/.ci/results/$BUILD_ID}"

# Virtual environment
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.ci/.venv}"

# Server ports
PEGA_PORT="${PEGA_PORT:-50055}"
PEGA_METRICS_PORT="${PEGA_METRICS_PORT:-9091}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"

# PegaFlow server configuration
PEGA_POOL_SIZE="${PEGA_POOL_SIZE:-50gb}"
PEGA_DEVICE="${PEGA_DEVICE:-0}"
PEGA_SSD_CACHE_PATH="${PEGA_SSD_CACHE_PATH:-/tmp/pegaflow_ci_ssd_cache}"
PEGA_SSD_CACHE_CAPACITY="${PEGA_SSD_CACHE_CAPACITY:-100gb}"

# vLLM configuration
MODEL="${MODEL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Timeouts
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-600}"  # 10 minutes
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-5}"

# Performance thresholds
MAX_PEGAFLOW_TTFT="${MAX_PEGAFLOW_TTFT:-0.22}"  # Max TTFT in seconds for PegaFlow
MIN_SPEEDUP_RATIO="${MIN_SPEEDUP_RATIO:-1.2}"    # Minimum expected speedup vs baseline

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_section() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# =============================================================================
# Process Management
# =============================================================================

# Global array to track spawned PIDs
declare -a SPAWNED_PIDS=()

# Register a PID for cleanup
register_pid() {
    local pid="$1"
    SPAWNED_PIDS+=("$pid")
}

# Cleanup all registered processes
cleanup_processes() {
    log_info "Cleaning up spawned processes..."
    
    for pid in "${SPAWNED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "  Terminating process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait for graceful shutdown
    sleep 2
    
    # Force kill if still running
    for pid in "${SPAWNED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "  Force killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    SPAWNED_PIDS=()
    log_success "All processes cleaned up"
}

# =============================================================================
# Virtual Environment Setup
# =============================================================================

setup_venv() {
    local packages=("$@")
    
    log_info "Setting up Python virtual environment at $VENV_DIR"
    
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --quiet --upgrade pip
    
    # Install requested packages
    if [[ ${#packages[@]} -gt 0 ]]; then
        log_info "Installing packages: ${packages[*]}"
        pip install --quiet "${packages[@]}"
    fi
    
    log_success "Virtual environment ready"
}

# Activate existing venv (assumes setup_venv was called)
activate_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
    else
        log_error "Virtual environment not found at $VENV_DIR"
        return 1
    fi
}

# =============================================================================
# Health Checks
# =============================================================================

# Wait for HTTP endpoint to become healthy
wait_for_http() {
    local url="$1"
    local name="${2:-service}"
    local timeout="${3:-$SERVER_STARTUP_TIMEOUT}"
    
    log_info "Waiting for $name at $url (timeout: ${timeout}s)..."
    
    local start_time
    start_time=$(date +%s)
    
    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -ge $timeout ]]; then
            log_error "Timeout waiting for $name after ${timeout}s"
            return 1
        fi
        
        if curl -s -f "$url" >/dev/null 2>&1; then
            log_success "$name is ready (took ${elapsed}s)"
            return 0
        fi
        
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

# Wait for gRPC/metrics endpoint
wait_for_metrics() {
    local host="$1"
    local port="$2"
    local name="${3:-service}"
    local timeout="${4:-$SERVER_STARTUP_TIMEOUT}"
    
    # Convert 0.0.0.0 to localhost for health checks
    [[ "$host" == "0.0.0.0" ]] && host="localhost"
    
    wait_for_http "http://${host}:${port}/metrics" "$name" "$timeout"
}

# Check if vLLM is healthy
check_vllm_health() {
    local port="$1"
    local host="${2:-localhost}"
    
    curl -s -f "http://${host}:${port}/health" >/dev/null 2>&1
}

# =============================================================================
# Result Collection
# =============================================================================

# Ensure results directory exists
ensure_results_dir() {
    mkdir -p "$RESULTS_DIR"
    log_info "Results will be saved to: $RESULTS_DIR"
}

# Save test result
save_result() {
    local test_name="$1"
    local status="$2"  # "pass" or "fail"
    local message="${3:-}"
    
    local result_file="$RESULTS_DIR/${test_name}.result"
    
    cat > "$result_file" <<EOF
test: $test_name
status: $status
timestamp: $(date -Iseconds)
message: $message
EOF
    
    if [[ "$status" == "pass" ]]; then
        log_success "Test $test_name: PASSED"
    else
        log_error "Test $test_name: FAILED - $message"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

# Parse memory size string (e.g., "10gb", "500mb") to bytes
parse_memory_size() {
    local size_str="$1"
    local size_lower
    size_lower=$(echo "$size_str" | tr '[:upper:]' '[:lower:]')
    
    local number
    local unit
    number=$(echo "$size_lower" | sed 's/[^0-9.]//g')
    unit=$(echo "$size_lower" | sed 's/[0-9.]//g')
    
    case "$unit" in
        tb) echo "$number * 1024 * 1024 * 1024 * 1024" | bc ;;
        gb) echo "$number * 1024 * 1024 * 1024" | bc ;;
        mb) echo "$number * 1024 * 1024" | bc ;;
        kb) echo "$number * 1024" | bc ;;
        b|"") echo "$number" ;;
        *) echo "0" ;;
    esac
}

# Check if required commands exist
require_command() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        log_error "Required command not found: $cmd"
        return 1
    fi
}

# Validate configuration
validate_config() {
    local errors=0
    
    if [[ -z "$MODEL" ]]; then
        log_error "MODEL environment variable is required"
        ((errors++))
    elif [[ ! -d "$MODEL" && ! -f "$MODEL" ]]; then
        log_warn "MODEL path does not exist locally: $MODEL (may be a HuggingFace model ID)"
    fi
    
    require_command "python3" || ((errors++))
    require_command "curl" || ((errors++))
    require_command "cargo" || ((errors++))
    
    if [[ $errors -gt 0 ]]; then
        log_error "Configuration validation failed with $errors error(s)"
        return 1
    fi
    
    log_success "Configuration validated"
    return 0
}

# Print configuration summary
print_config() {
    log_section "Configuration"
    echo "Project root:         $PROJECT_ROOT"
    echo "Results directory:    $RESULTS_DIR"
    echo "Build ID:             $BUILD_ID"
    echo ""
    echo "PegaFlow:"
    echo "  Port:               $PEGA_PORT"
    echo "  Metrics port:       $PEGA_METRICS_PORT"
    echo "  Pool size:          $PEGA_POOL_SIZE"
    echo "  Device:             $PEGA_DEVICE"
    echo "  SSD cache:          $PEGA_SSD_CACHE_PATH ($PEGA_SSD_CACHE_CAPACITY)"
    echo ""
    echo "vLLM:"
    echo "  PegaFlow port:      $VLLM_PORT"
    echo "  Baseline port:      $VLLM_BASELINE_PORT"
    echo "  Model:              $MODEL"
    echo "  Served model name:  ${SERVED_MODEL_NAME:-$MODEL}"
    echo "  Max sequences:      $MAX_NUM_SEQS"
    echo ""
    echo "Thresholds:"
    echo "  Max TTFT:           ${MAX_PEGAFLOW_TTFT}s"
    echo "  Min speedup:        ${MIN_SPEEDUP_RATIO}x"
    echo ""
}

# =============================================================================
# Trap Setup
# =============================================================================

# Setup cleanup trap (call this in test scripts)
setup_cleanup_trap() {
    trap 'cleanup_processes; exit 130' SIGINT SIGTERM
    trap 'cleanup_processes' EXIT
}

