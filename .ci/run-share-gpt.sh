#!/bin/bash
# ShareGPT Multi-Turn Benchmark for PegaFlow
#
# This test runs ShareGPT multi-turn conversations to evaluate:
#   1. Multi-turn conversation performance with PegaFlow caching
#   2. Comparison against baseline vLLM (optional)
#   3. Cache effectiveness across conversation turns
#
# Uses the existing share_gpt_bench.py from pegaflow/examples/

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# ShareGPT benchmark settings
NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-2}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-10}"
MAX_TURNS="${MAX_TURNS:-5}"
REQUEST_RATE_PER_CLIENT="${REQUEST_RATE_PER_CLIENT:-0.5}"
WARMUP="${WARMUP:-false}"

# Output directory
SHAREGPT_DIR="$RESULTS_DIR/share_gpt"

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run ShareGPT multi-turn benchmark comparing PegaFlow vs baseline.

Options:
  --num-conversations N   Number of conversations (default: $NUM_CONVERSATIONS)
  --num-clients N         Number of parallel clients (default: $NUM_CLIENTS)
  --max-active N          Max active conversations (default: $MAX_ACTIVE_CONVERSATIONS)
  --max-turns N           Max turns per conversation (default: $MAX_TURNS)
  --request-rate N        Request rate per client (default: $REQUEST_RATE_PER_CLIENT)
  --warmup                Run warmup step before benchmark
  --skip-baseline         Skip baseline comparison
  -h, --help              Show this help message

Environment variables:
  MODEL                   Model name/path
  SERVED_MODEL_NAME       Served model name
  VLLM_PORT               vLLM with PegaFlow port (default: 8000)
  VLLM_BASELINE_PORT      vLLM baseline port (default: 9000)

Example:
  MODEL=Qwen/Qwen2.5-7B $0 --num-conversations 100 --max-turns 10
EOF
    exit 0
}

run_sharegpt_benchmark() {
    local name="$1"
    local port="$2"
    local output_dir="$3"
    
    log_section "Running ShareGPT Benchmark: $name"
    
    mkdir -p "$output_dir"
    
    local model_name="${SERVED_MODEL_NAME:-$MODEL}"
    local benchmark_script="$PROJECT_ROOT/examples/share_gpt_bench.py"
    
    if [[ ! -f "$benchmark_script" ]]; then
        log_error "ShareGPT benchmark script not found: $benchmark_script"
        return 1
    fi
    
    log_info "Configuration:"
    log_info "  Model: $model_name"
    log_info "  Port: $port"
    log_info "  Conversations: $NUM_CONVERSATIONS"
    log_info "  Clients: $NUM_CLIENTS"
    log_info "  Max active: $MAX_ACTIVE_CONVERSATIONS"
    log_info "  Max turns: $MAX_TURNS"
    log_info "  Request rate: $REQUEST_RATE_PER_CLIENT/client"
    
    local cmd="python3 $benchmark_script"
    cmd="$cmd --model $model_name"
    cmd="$cmd --served-model-name $model_name"
    cmd="$cmd --host localhost"
    cmd="$cmd --port $port"
    cmd="$cmd --num-conversations $NUM_CONVERSATIONS"
    cmd="$cmd --num-clients $NUM_CLIENTS"
    cmd="$cmd --request-rate $REQUEST_RATE_PER_CLIENT"
    cmd="$cmd --output-dir $output_dir"
    cmd="$cmd --seed 42"
    
    if [[ -n "$MAX_ACTIVE_CONVERSATIONS" ]]; then
        cmd="$cmd --max-active-conversations $MAX_ACTIVE_CONVERSATIONS"
    fi
    
    if [[ -n "$MAX_TURNS" ]]; then
        cmd="$cmd --max-turns $MAX_TURNS"
    fi
    
    if [[ "$WARMUP" == "true" ]]; then
        cmd="$cmd --warmup-step"
    fi
    
    local log_file="$output_dir/benchmark.log"
    
    eval "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "$name benchmark completed"
        return 0
    else
        log_error "$name benchmark failed with exit code $exit_code"
        return 1
    fi
}

run_simplified_multi_turn() {
    local name="$1"
    local port="$2"
    local output_file="$3"
    
    log_info "Running simplified multi-turn benchmark: $name"
    
    local model_name="${SERVED_MODEL_NAME:-$MODEL}"
    local url="http://localhost:${port}/v1/chat/completions"
    
    python3 << EOF
import json
import time
import asyncio
import aiohttp
import statistics
import random

# Simulated multi-turn conversations
CONVERSATION_STARTERS = [
    "Tell me about the solar system.",
    "Explain how machine learning works.",
    "What is the history of the internet?",
    "Describe the process of photosynthesis.",
    "How do computers store data?",
]

FOLLOW_UPS = [
    "Can you elaborate on that?",
    "What are the key points?",
    "Give me an example.",
    "Why is that important?",
    "What happens next?",
]

async def run_conversation(session, url, model, conv_id, max_turns):
    """Run a multi-turn conversation."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    results = []
    
    # First message
    starter = random.choice(CONVERSATION_STARTERS)
    messages.append({"role": "user", "content": starter})
    
    for turn in range(max_turns):
        payload = {
            "model": model,
            "messages": messages.copy(),
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": True,
        }
        
        start_time = time.perf_counter()
        ttft = None
        response_text = ""
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    results.append({"success": False, "error": f"HTTP {resp.status}", "turn": turn})
                    continue
                
                async for chunk in resp.content.iter_any():
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    
                    # Try to parse SSE data
                    try:
                        chunk_str = chunk.decode('utf-8')
                        if 'data: ' in chunk_str:
                            for line in chunk_str.split('\n'):
                                if line.startswith('data: ') and line != 'data: [DONE]':
                                    data = json.loads(line[6:])
                                    if 'choices' in data and data['choices']:
                                        delta = data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        response_text += content
                    except:
                        pass
                
                total_time = time.perf_counter() - start_time
                
                results.append({
                    "success": True,
                    "turn": turn,
                    "ttft": ttft or total_time,
                    "total_time": total_time,
                    "conv_id": conv_id,
                })
                
                # Add assistant response to history
                if response_text:
                    messages.append({"role": "assistant", "content": response_text[:500]})
                else:
                    messages.append({"role": "assistant", "content": "Response received."})
                
                # Add follow-up for next turn (if not last)
                if turn < max_turns - 1:
                    follow_up = random.choice(FOLLOW_UPS)
                    messages.append({"role": "user", "content": follow_up})
                    response_text = ""
                
        except Exception as e:
            results.append({"success": False, "error": str(e), "turn": turn})
    
    return results

async def run_benchmark(url, model, num_conversations, max_turns, max_concurrent):
    """Run multi-turn benchmark."""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            run_conversation(session, url, model, i, max_turns)
            for i in range(num_conversations)
        ]
        all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    return [r for conv_results in all_results for r in conv_results]

# Run benchmark
print(f"Running multi-turn benchmark against $url")
print(f"Conversations: $NUM_CONVERSATIONS, Max turns: $MAX_TURNS")
print()

start = time.perf_counter()
results = asyncio.run(run_benchmark(
    "$url", 
    "$model_name",
    $NUM_CONVERSATIONS,
    $MAX_TURNS,
    $NUM_CLIENTS
))
total_duration = time.perf_counter() - start

# Analyze
successful = [r for r in results if r.get("success")]
failed = [r for r in results if not r.get("success")]

ttfts = [r["ttft"] for r in successful]
total_times = [r["total_time"] for r in successful]

# Analyze by turn
turns = {}
for r in successful:
    turn = r["turn"]
    if turn not in turns:
        turns[turn] = []
    turns[turn].append(r["ttft"])

print(f"=== Multi-Turn Benchmark Results ===")
print(f"Total turns: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
print(f"Duration: {total_duration:.2f}s")
print()

if ttfts:
    print(f"Overall TTFT:")
    print(f"  Mean: {statistics.mean(ttfts)*1000:.2f}ms")
    print(f"  Median: {statistics.median(ttfts)*1000:.2f}ms")
    print()
    
    print(f"TTFT by Turn:")
    for turn in sorted(turns.keys()):
        turn_ttfts = turns[turn]
        print(f"  Turn {turn}: mean={statistics.mean(turn_ttfts)*1000:.2f}ms, n={len(turn_ttfts)}")

# Save results
output = {
    "total_turns": len(results),
    "successful": len(successful),
    "failed": len(failed),
    "duration_s": total_duration,
    "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
    "ttft_median_s": statistics.median(ttfts) if ttfts else None,
    "by_turn": {
        str(t): {
            "mean_ttft_s": statistics.mean(turns[t]),
            "count": len(turns[t])
        } for t in turns
    }
}

with open("$output_file", "w") as f:
    json.dump(output, f, indent=2)

print(f"\\nResults saved to $output_file")
EOF
}

extract_metric() {
    local json_file="$1"
    local metric="$2"
    
    python3 -c "import json; data=json.load(open('$json_file')); print(data.get('$metric', 0) or 0)"
}

compare_results() {
    local pega_result="$1"
    local baseline_result="$2"
    
    log_section "Performance Comparison"
    
    local pega_ttft
    local baseline_ttft
    
    pega_ttft=$(extract_metric "$pega_result" "ttft_mean_s")
    baseline_ttft=$(extract_metric "$baseline_result" "ttft_mean_s")
    
    echo "Metric                    PegaFlow        Baseline"
    echo "--------------------------------------------------------"
    printf "TTFT (mean):              %.4fs          %.4fs\n" "$pega_ttft" "$baseline_ttft"
    echo ""
    
    if [[ $(echo "$pega_ttft > 0" | bc -l) -eq 1 && $(echo "$baseline_ttft > 0" | bc -l) -eq 1 ]]; then
        local speedup
        speedup=$(echo "scale=2; $baseline_ttft / $pega_ttft" | bc)
        
        if [[ $(echo "$speedup >= 1.0" | bc -l) -eq 1 ]]; then
            log_success "PegaFlow is ${speedup}x faster than baseline"
        else
            log_warn "PegaFlow is slower than baseline (${speedup}x)"
        fi
    fi
    
    return 0
}

check_servers_running() {
    local skip_baseline="$1"
    
    if ! check_vllm_health "$VLLM_PORT"; then
        log_error "vLLM with PegaFlow is not running at port $VLLM_PORT"
        return 1
    fi
    
    if [[ "$skip_baseline" != "true" ]]; then
        if ! check_vllm_health "$VLLM_BASELINE_PORT"; then
            log_error "vLLM baseline is not running at port $VLLM_BASELINE_PORT"
            return 1
        fi
    fi
    
    return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    local skip_baseline=false
    local use_simplified=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --num-conversations)
                NUM_CONVERSATIONS="$2"
                shift 2
                ;;
            --num-clients)
                NUM_CLIENTS="$2"
                shift 2
                ;;
            --max-active)
                MAX_ACTIVE_CONVERSATIONS="$2"
                shift 2
                ;;
            --max-turns)
                MAX_TURNS="$2"
                shift 2
                ;;
            --request-rate)
                REQUEST_RATE_PER_CLIENT="$2"
                shift 2
                ;;
            --warmup)
                WARMUP=true
                shift
                ;;
            --skip-baseline)
                skip_baseline=true
                shift
                ;;
            --simplified)
                use_simplified=true
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
    
    log_section "ShareGPT Multi-Turn Benchmark"
    echo "Configuration:"
    echo "  Model:              ${SERVED_MODEL_NAME:-$MODEL}"
    echo "  Conversations:      $NUM_CONVERSATIONS"
    echo "  Clients:            $NUM_CLIENTS"
    echo "  Max active:         $MAX_ACTIVE_CONVERSATIONS"
    echo "  Max turns:          $MAX_TURNS"
    echo "  Request rate:       $REQUEST_RATE_PER_CLIENT/client"
    echo "  Warmup:             $WARMUP"
    echo "  Skip baseline:      $skip_baseline"
    echo "  Results dir:        $SHAREGPT_DIR"
    echo ""
    
    # Check servers
    check_servers_running "$skip_baseline" || exit 1
    
    # Setup
    setup_venv aiohttp
    ensure_results_dir
    mkdir -p "$SHAREGPT_DIR"
    
    # Check if share_gpt_bench.py exists
    local benchmark_script="$PROJECT_ROOT/examples/share_gpt_bench.py"
    if [[ ! -f "$benchmark_script" ]] || [[ "$use_simplified" == "true" ]]; then
        log_warn "Using simplified multi-turn benchmark"
        use_simplified=true
    fi
    
    # Run PegaFlow benchmark
    log_section "Phase 1: PegaFlow Multi-Turn Benchmark"
    
    local pega_output_dir="$SHAREGPT_DIR/pegaflow"
    mkdir -p "$pega_output_dir"
    
    if [[ "$use_simplified" == "true" ]]; then
        run_simplified_multi_turn "PegaFlow" "$VLLM_PORT" "$pega_output_dir/results.json"
    else
        run_sharegpt_benchmark "PegaFlow" "$VLLM_PORT" "$pega_output_dir"
    fi
    
    # Run baseline benchmark (if not skipped)
    if [[ "$skip_baseline" != "true" ]]; then
        log_section "Phase 2: Baseline Multi-Turn Benchmark"
        
        local baseline_output_dir="$SHAREGPT_DIR/baseline"
        mkdir -p "$baseline_output_dir"
        
        if [[ "$use_simplified" == "true" ]]; then
            run_simplified_multi_turn "Baseline" "$VLLM_BASELINE_PORT" "$baseline_output_dir/results.json"
        else
            run_sharegpt_benchmark "Baseline" "$VLLM_BASELINE_PORT" "$baseline_output_dir"
        fi
        
        # Compare results
        if [[ "$use_simplified" == "true" ]]; then
            compare_results "$pega_output_dir/results.json" "$baseline_output_dir/results.json"
        fi
    fi
    
    save_result "share-gpt-multi-turn" "pass" "Multi-turn benchmark completed"
    
    log_section "ShareGPT Multi-Turn Benchmark: COMPLETED"
    echo "Results saved to: $SHAREGPT_DIR"
}

main "$@"

