#!/bin/bash
# PegaFlow CI Test Suite Orchestrator
#
# This script runs the complete CI test suite:
#   1. Starts PegaFlow and vLLM servers
#   2. Runs all test scripts sequentially
#   3. Collects and reports results
#   4. Cleans up servers on exit
#
# Usage:
#   MODEL=/path/to/model ./run-all-tests.sh
#   MODEL=/path/to/model ./run-all-tests.sh --quick  # Skip baseline comparisons

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Test selection
RUN_LM_EVAL="${RUN_LM_EVAL:-true}"
RUN_LONG_DOC_QA="${RUN_LONG_DOC_QA:-true}"
RUN_SHARE_GPT="${RUN_SHARE_GPT:-true}"

# Quick mode: skip baseline comparisons
QUICK_MODE="${QUICK_MODE:-false}"

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the complete PegaFlow CI test suite.

Options:
  --quick              Quick mode: skip baseline comparisons
  --lm-eval-only       Run only LM-Eval consistency test
  --long-doc-only      Run only Long-Doc QA performance test
  --share-gpt-only     Run only ShareGPT multi-turn test
  --skip-lm-eval       Skip LM-Eval test
  --skip-long-doc      Skip Long-Doc QA test
  --skip-share-gpt     Skip ShareGPT test
  -h, --help           Show this help message

Required environment variables:
  MODEL                Model path or HuggingFace model ID

Optional environment variables:
  SERVED_MODEL_NAME    Served model name (default: MODEL)
  CHAT_TEMPLATE        Path to chat template
  VLLM_PORT            vLLM with PegaFlow port (default: 8000)
  VLLM_BASELINE_PORT   vLLM baseline port (default: 9000)
  PEGA_PORT            PegaFlow gRPC port (default: 50055)
  RESULTS_DIR          Results directory (default: .ci/results/<timestamp>)

Examples:
  # Full test suite
  MODEL=Qwen/Qwen2.5-7B ./run-all-tests.sh

  # Quick mode (no baseline)
  MODEL=/models/Llama-3.1-8B ./run-all-tests.sh --quick

  # Specific test only
  MODEL=mistralai/Mistral-7B ./run-all-tests.sh --lm-eval-only
EOF
    exit 0
}

# Run a test script and capture result
run_test() {
    local test_name="$1"
    local script="$2"
    shift 2
    local args=("$@")
    
    log_section "Running: $test_name"
    
    local start_time
    start_time=$(date +%s)
    
    local exit_code=0
    "$SCRIPT_DIR/$script" "${args[@]}" || exit_code=$?
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        TEST_RESULTS+=("$test_name:PASS:${duration}s")
        log_success "$test_name completed in ${duration}s"
        return 0
    else
        TEST_RESULTS+=("$test_name:FAIL:${duration}s")
        log_error "$test_name failed after ${duration}s"
        return 1
    fi
}

# Print final summary
print_summary() {
    log_section "Test Summary"
    
    local total=0
    local passed=0
    local failed=0
    
    echo "Test Name                     Status    Duration"
    echo "-----------------------------------------------------"
    
    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r name status duration <<< "$result"
        ((total++))
        
        if [[ "$status" == "PASS" ]]; then
            ((passed++))
            printf "%-30s ${GREEN}%s${NC}     %s\n" "$name" "$status" "$duration"
        else
            ((failed++))
            printf "%-30s ${RED}%s${NC}     %s\n" "$name" "$status" "$duration"
        fi
    done
    
    echo "-----------------------------------------------------"
    echo "Total: $total  Passed: $passed  Failed: $failed"
    echo ""
    
    if [[ $failed -eq 0 ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "$failed test(s) failed"
        return 1
    fi
}

# Cleanup handler
cleanup_handler() {
    log_info "Cleaning up..."
    "$SCRIPT_DIR/start-servers.sh" stop-all 2>/dev/null || true
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Result tracking
    declare -a TEST_RESULTS=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --lm-eval-only)
                RUN_LONG_DOC_QA=false
                RUN_SHARE_GPT=false
                shift
                ;;
            --long-doc-only)
                RUN_LM_EVAL=false
                RUN_SHARE_GPT=false
                shift
                ;;
            --share-gpt-only)
                RUN_LM_EVAL=false
                RUN_LONG_DOC_QA=false
                shift
                ;;
            --skip-lm-eval)
                RUN_LM_EVAL=false
                shift
                ;;
            --skip-long-doc)
                RUN_LONG_DOC_QA=false
                shift
                ;;
            --skip-share-gpt)
                RUN_SHARE_GPT=false
                shift
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
    
    # Validate configuration
    validate_config || exit 1
    
    log_section "PegaFlow CI Test Suite"
    print_config
    
    echo "Test Selection:"
    echo "  LM-Eval:      $RUN_LM_EVAL"
    echo "  Long-Doc QA:  $RUN_LONG_DOC_QA"
    echo "  ShareGPT:     $RUN_SHARE_GPT"
    echo "  Quick mode:   $QUICK_MODE"
    echo ""
    
    # Setup cleanup trap
    trap cleanup_handler EXIT SIGINT SIGTERM
    
    # Ensure results directory exists
    ensure_results_dir
    
    # Start servers
    log_section "Starting Servers"
    
    if [[ "$QUICK_MODE" == "true" ]]; then
        "$SCRIPT_DIR/start-servers.sh" start-all --no-baseline
    else
        "$SCRIPT_DIR/start-servers.sh" start-all
    fi
    
    # Track overall success
    local all_passed=true
    
    # Run LM-Eval consistency test
    if [[ "$RUN_LM_EVAL" == "true" ]]; then
        if ! run_test "LM-Eval Consistency" "run-lm-eval.sh" --limit 50; then
            all_passed=false
        fi
    fi
    
    # Run Long-Doc QA performance test
    if [[ "$RUN_LONG_DOC_QA" == "true" ]]; then
        local long_doc_args=(--num-requests 30)
        if [[ "$QUICK_MODE" == "true" ]]; then
            long_doc_args+=(--skip-baseline)
        fi
        
        if ! run_test "Long-Doc QA Performance" "run-long-doc-qa.sh" "${long_doc_args[@]}"; then
            all_passed=false
        fi
    fi
    
    # Run ShareGPT multi-turn test
    if [[ "$RUN_SHARE_GPT" == "true" ]]; then
        local sharegpt_args=(--num-conversations 20 --max-turns 3)
        if [[ "$QUICK_MODE" == "true" ]]; then
            sharegpt_args+=(--skip-baseline)
        fi
        
        if ! run_test "ShareGPT Multi-Turn" "run-share-gpt.sh" "${sharegpt_args[@]}"; then
            all_passed=false
        fi
    fi
    
    # Print summary
    print_summary
    
    # Save overall result
    if [[ "$all_passed" == "true" ]]; then
        echo "PASSED" > "$RESULTS_DIR/overall.result"
        exit 0
    else
        echo "FAILED" > "$RESULTS_DIR/overall.result"
        exit 1
    fi
}

main "$@"

