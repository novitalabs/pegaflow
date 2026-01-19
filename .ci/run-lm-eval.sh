#!/bin/bash
# LM-Eval Consistency Test for PegaFlow
# 
# This test validates PegaFlow's caching behavior by running lm_eval twice:
#   1. First run: Populates the cache (cache miss)
#   2. Second run: Uses cached results (cache hit)
#   3. Verification: Compare output samples - must be identical
#
# Based on: https://github.com/LMCache/LMCache/blob/dev/.buildkite/scripts/multiprocessing-test/run-lm-eval.sh

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# LM-Eval specific settings
LM_EVAL_TASK="${LM_EVAL_TASK:-gsm8k}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"
NUM_CONCURRENT="${NUM_CONCURRENT:-10}"
SEED="${SEED:-0}"

# Output directories
LM_EVAL_DIR="$RESULTS_DIR/lm_eval"
FIRST_RUN_DIR="$LM_EVAL_DIR/first_run"
SECOND_RUN_DIR="$LM_EVAL_DIR/second_run"

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run LM-Eval consistency test against vLLM with PegaFlow.

Options:
  --task TASK         LM-Eval task (default: $LM_EVAL_TASK)
  --limit N           Number of samples to evaluate (default: $LM_EVAL_LIMIT)
  --concurrent N      Number of concurrent requests (default: $NUM_CONCURRENT)
  --seed N            Random seed (default: $SEED)
  -h, --help          Show this help message

Environment variables:
  MODEL               Model name for lm_eval
  VLLM_PORT           vLLM server port (default: 8000)

Example:
  MODEL=Qwen/Qwen2.5-7B VLLM_PORT=8000 $0 --limit 50
EOF
    exit 0
}

run_lm_eval() {
    local run_name="$1"
    local output_dir="$2"
    
    log_section "Running lm_eval ($run_name)"
    log_info "Output directory: $output_dir"
    log_info "Task: $LM_EVAL_TASK"
    log_info "Limit: $LM_EVAL_LIMIT"
    log_info "Concurrent: $NUM_CONCURRENT"
    
    mkdir -p "$output_dir"
    
    local model_name="${SERVED_MODEL_NAME:-$MODEL}"
    local base_url="http://127.0.0.1:${VLLM_PORT}/v1/completions"
    
    lm_eval --model local-completions \
        --tasks "$LM_EVAL_TASK" \
        --model_args "model=${model_name},base_url=${base_url},num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LM_EVAL_LIMIT" \
        --seed "$SEED" \
        -s \
        --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    
    log_success "$run_name completed"
}

find_samples_file() {
    local dir="$1"
    local task="$2"
    
    # lm_eval creates: output_path/model_name/samples_task_timestamp.jsonl
    find "$dir" -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -1
}

verify_samples_match() {
    local first_dir="$1"
    local second_dir="$2"
    
    log_section "Verifying Output Consistency"
    
    # Find samples files
    local first_samples
    local second_samples
    first_samples=$(find_samples_file "$first_dir" "$LM_EVAL_TASK")
    second_samples=$(find_samples_file "$second_dir" "$LM_EVAL_TASK")
    
    if [[ -z "$first_samples" ]]; then
        log_error "Could not find samples file in first run directory: $first_dir"
        log_info "Files in directory:"
        find "$first_dir" -type f -name "*.jsonl" || true
        return 1
    fi
    
    if [[ -z "$second_samples" ]]; then
        log_error "Could not find samples file in second run directory: $second_dir"
        log_info "Files in directory:"
        find "$second_dir" -type f -name "*.jsonl" || true
        return 1
    fi
    
    log_info "First run samples:  $first_samples"
    log_info "Second run samples: $second_samples"
    
    # Sort both files and compare
    local first_sorted
    local second_sorted
    first_sorted=$(mktemp)
    second_sorted=$(mktemp)
    
    sort "$first_samples" > "$first_sorted"
    sort "$second_samples" > "$second_sorted"
    
    if diff -q "$first_sorted" "$second_sorted" >/dev/null 2>&1; then
        log_success "Samples files are IDENTICAL!"
        log_info "Cache consistency verified - outputs match across runs"
        rm -f "$first_sorted" "$second_sorted"
        return 0
    else
        log_error "Samples files DIFFER!"
        echo ""
        echo "=== Diff (first 50 lines) ==="
        diff "$first_sorted" "$second_sorted" | head -50 || true
        echo ""
        
        # Save diff for analysis
        diff "$first_sorted" "$second_sorted" > "$LM_EVAL_DIR/samples_diff.txt" 2>/dev/null || true
        
        rm -f "$first_sorted" "$second_sorted"
        return 1
    fi
}

check_server_running() {
    if ! check_vllm_health "$VLLM_PORT"; then
        log_error "vLLM server is not running at port $VLLM_PORT"
        log_info "Please start servers first with: ./start-servers.sh start-all"
        return 1
    fi
    return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --task)
                LM_EVAL_TASK="$2"
                shift 2
                ;;
            --limit)
                LM_EVAL_LIMIT="$2"
                shift 2
                ;;
            --concurrent)
                NUM_CONCURRENT="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    log_section "LM-Eval Consistency Test"
    echo "Model:              ${SERVED_MODEL_NAME:-$MODEL}"
    echo "vLLM Port:          $VLLM_PORT"
    echo "Task:               $LM_EVAL_TASK"
    echo "Limit:              $LM_EVAL_LIMIT"
    echo "Concurrent:         $NUM_CONCURRENT"
    echo "Seed:               $SEED"
    echo "Results dir:        $LM_EVAL_DIR"
    echo ""
    
    # Check server is running
    check_server_running || exit 1
    
    # Setup virtual environment with lm_eval
    setup_venv 'lm-eval[api]' openai pandas
    
    # Ensure directories exist
    ensure_results_dir
    mkdir -p "$FIRST_RUN_DIR"
    mkdir -p "$SECOND_RUN_DIR"
    
    # First run - populates cache
    log_section "Phase 1: Cache Population"
    log_info "Running first lm_eval pass (cache miss expected)..."
    
    local start_time1
    start_time1=$(date +%s)
    run_lm_eval "first_run" "$FIRST_RUN_DIR"
    local end_time1
    end_time1=$(date +%s)
    local duration1=$((end_time1 - start_time1))
    
    log_info "First run completed in ${duration1}s"
    
    # Second run - should use cached results
    log_section "Phase 2: Cache Hit"
    log_info "Running second lm_eval pass (cache hit expected)..."
    
    local start_time2
    start_time2=$(date +%s)
    run_lm_eval "second_run" "$SECOND_RUN_DIR"
    local end_time2
    end_time2=$(date +%s)
    local duration2=$((end_time2 - start_time2))
    
    log_info "Second run completed in ${duration2}s"
    
    # Report timing comparison
    log_section "Timing Summary"
    echo "First run (cache miss):  ${duration1}s"
    echo "Second run (cache hit):  ${duration2}s"
    
    if [[ $duration2 -lt $duration1 ]]; then
        local speedup
        speedup=$(echo "scale=2; $duration1 / $duration2" | bc)
        log_success "Second run was ${speedup}x faster (cache effect)"
    else
        log_warn "Second run was not faster - cache may not have been effective"
    fi
    echo ""
    
    # Verify consistency
    log_section "Phase 3: Consistency Verification"
    if verify_samples_match "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"; then
        save_result "lm-eval-consistency" "pass" "Samples identical across runs"
        
        log_section "LM-Eval Consistency Test: PASSED"
        echo "Results saved to: $LM_EVAL_DIR"
        echo "  - First run:  $FIRST_RUN_DIR"
        echo "  - Second run: $SECOND_RUN_DIR"
        exit 0
    else
        save_result "lm-eval-consistency" "fail" "Samples differ between runs"
        
        log_section "LM-Eval Consistency Test: FAILED"
        log_error "Output samples do not match - cache may be corrupting results"
        exit 1
    fi
}

main "$@"

