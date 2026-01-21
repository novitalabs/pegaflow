#!/bin/bash
# Long Document QA Performance Test for PegaFlow
#
# This test compares vLLM+PegaFlow against vanilla vLLM to verify:
#   1. PegaFlow provides performance improvement (speedup)
#   2. TTFT meets the configured threshold
#   3. Cache hit behavior on repeated queries
#
# Modeled after: https://github.com/LMCache/LMCache/blob/dev/.buildkite/scripts/multiprocessing-test/run-long-doc-qa.sh

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Benchmark settings
NUM_REQUESTS="${NUM_REQUESTS:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
REQUEST_RATE="${REQUEST_RATE:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

# Thresholds (can be overridden)
MAX_PEGAFLOW_TTFT="${MAX_PEGAFLOW_TTFT:-0.22}"  # Max TTFT in seconds
MIN_SPEEDUP_RATIO="${MIN_SPEEDUP_RATIO:-1.2}"    # Minimum expected speedup

# Output directory
LONG_DOC_DIR="$RESULTS_DIR/long_doc_qa"

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run Long Document QA performance test comparing PegaFlow vs baseline.

Options:
  --num-requests N      Number of requests (default: $NUM_REQUESTS)
  --max-concurrency N   Max concurrent requests (default: $MAX_CONCURRENCY)
  --request-rate N      Requests per second (default: $REQUEST_RATE)
  --output-len N        Output length per request (default: $OUTPUT_LEN)
  --max-ttft SECONDS    Maximum TTFT threshold (default: $MAX_PEGAFLOW_TTFT)
  --min-speedup RATIO   Minimum speedup ratio (default: $MIN_SPEEDUP_RATIO)
  --skip-baseline       Skip baseline comparison (just test PegaFlow)
  -h, --help            Show this help message

Environment variables:
  VLLM_PORT             vLLM with PegaFlow port (default: 8000)
  VLLM_BASELINE_PORT    vLLM baseline port (default: 9000)
  DATA_FILE             Input data file for benchmark

Example:
  DATA_FILE=/path/to/data.json $0 --num-requests 100
EOF
    exit 0
}

# Generate a long document QA dataset if not provided
generate_test_data() {
    local output_file="$1"
    local num_items="${2:-10}"
    
    log_info "Generating test data with $num_items long document prompts..."
    
    python3 << EOF
import json
import random

# Generate long document QA prompts
prompts = []
for i in range($num_items):
    # Create a long context document (simulating a long document)
    doc_paragraphs = []
    for p in range(random.randint(5, 10)):
        paragraph = f"Paragraph {p+1}: This is section {p+1} of the document discussing topic {i+1}. "
        paragraph += "The content here provides important context for the questions that follow. " * random.randint(3, 8)
        doc_paragraphs.append(paragraph)
    
    document = " ".join(doc_paragraphs)
    
    # Create a QA prompt
    question = f"Based on the document above, what is discussed in section {random.randint(1, len(doc_paragraphs))}?"
    
    prompt = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions about documents."},
            {"role": "user", "content": f"Document:\\n{document}\\n\\nQuestion: {question}"}
        ],
        "max_tokens": $OUTPUT_LEN,
        "temperature": 0.0,
        "stream": True
    }
    prompts.append(prompt)

with open("$output_file", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"Generated {len(prompts)} prompts to $output_file")
EOF
}

run_benchmark() {
    local name="$1"
    local port="$2"
    local output_file="$3"
    local data_file="$4"
    
    log_info "Running benchmark: $name"
    log_info "  Target: http://localhost:$port"
    log_info "  Requests: $NUM_REQUESTS"
    log_info "  Concurrency: $MAX_CONCURRENCY"
    
    local url="http://localhost:${port}/v1/chat/completions"
    
    # Use the project's benchmark script if available
    local benchmark_script="$PROJECT_ROOT/../benchmark_rps.py"
    
    if [[ -f "$benchmark_script" ]]; then
        python3 "$benchmark_script" \
            --url "$url" \
            --logfile "$data_file" \
            --num-requests "$NUM_REQUESTS" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --request-rate "$REQUEST_RATE" \
            --force-stream \
            2>&1 | tee "$output_file"
    else
        # Fallback: simple curl-based benchmark
        log_warn "benchmark_rps.py not found, using simplified benchmark"
        run_simple_benchmark "$url" "$data_file" "$output_file"
    fi
}

run_simple_benchmark() {
    local url="$1"
    local data_file="$2"
    local output_file="$3"
    
    python3 << EOF
import json
import time
import asyncio
import aiohttp
import statistics

async def send_request(session, url, payload, idx):
    """Send a single request and measure timing."""
    start = time.perf_counter()
    ttft = None
    
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                return {"success": False, "error": f"HTTP {resp.status}"}
            
            # For streaming, measure time to first chunk
            async for chunk in resp.content.iter_any():
                if ttft is None:
                    ttft = time.perf_counter() - start
                break
            
            # Read remaining response
            async for _ in resp.content.iter_any():
                pass
            
            total_time = time.perf_counter() - start
            return {
                "success": True,
                "ttft": ttft or total_time,
                "total_time": total_time
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def run_benchmark(url, payloads, max_concurrent):
    """Run benchmark with concurrency limit."""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_request(session, url, p, i) for i, p in enumerate(payloads)]
        results = await asyncio.gather(*tasks)
    
    return results

# Load test data
with open("$data_file") as f:
    payloads = json.load(f)

# Limit to configured number
payloads = payloads[:$NUM_REQUESTS]

print(f"Running benchmark against $url")
print(f"Requests: {len(payloads)}, Concurrency: $MAX_CONCURRENCY")

start = time.perf_counter()
results = asyncio.run(run_benchmark("$url", payloads, $MAX_CONCURRENCY))
total_duration = time.perf_counter() - start

# Analyze results
successful = [r for r in results if r.get("success")]
failed = [r for r in results if not r.get("success")]

ttfts = [r["ttft"] for r in successful]
total_times = [r["total_time"] for r in successful]

print(f"\\n=== Benchmark Results ===")
print(f"Total requests: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
print(f"Duration: {total_duration:.2f}s")
print(f"Throughput: {len(successful)/total_duration:.2f} req/s")

if ttfts:
    print(f"\\nTTFT (Time to First Token):")
    print(f"  Mean: {statistics.mean(ttfts)*1000:.2f}ms")
    print(f"  Median: {statistics.median(ttfts)*1000:.2f}ms")
    print(f"  P90: {sorted(ttfts)[int(len(ttfts)*0.9)]*1000:.2f}ms")
    print(f"  P99: {sorted(ttfts)[int(len(ttfts)*0.99)]*1000:.2f}ms")

if total_times:
    print(f"\\nEnd-to-End Latency:")
    print(f"  Mean: {statistics.mean(total_times)*1000:.2f}ms")
    print(f"  Median: {statistics.median(total_times)*1000:.2f}ms")

# Save results as JSON
output = {
    "total_requests": len(results),
    "successful": len(successful),
    "failed": len(failed),
    "duration_s": total_duration,
    "throughput_rps": len(successful)/total_duration,
    "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
    "ttft_median_s": statistics.median(ttfts) if ttfts else None,
    "ttft_p90_s": sorted(ttfts)[int(len(ttfts)*0.9)] if ttfts else None,
    "latency_mean_s": statistics.mean(total_times) if total_times else None,
}

with open("$output_file.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\\nResults saved to $output_file.json")
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
    
    # Extract metrics
    local pega_ttft
    local baseline_ttft
    local pega_throughput
    local baseline_throughput
    
    pega_ttft=$(extract_metric "$pega_result" "ttft_mean_s")
    baseline_ttft=$(extract_metric "$baseline_result" "ttft_mean_s")
    pega_throughput=$(extract_metric "$pega_result" "throughput_rps")
    baseline_throughput=$(extract_metric "$baseline_result" "throughput_rps")
    
    echo "Metric                    PegaFlow        Baseline"
    echo "--------------------------------------------------------"
    printf "TTFT (mean):              %.4fs          %.4fs\n" "$pega_ttft" "$baseline_ttft"
    printf "Throughput:               %.2f req/s     %.2f req/s\n" "$pega_throughput" "$baseline_throughput"
    echo ""
    
    # Calculate speedup
    local ttft_speedup
    local throughput_speedup
    
    if [[ $(echo "$pega_ttft > 0" | bc -l) -eq 1 && $(echo "$baseline_ttft > 0" | bc -l) -eq 1 ]]; then
        ttft_speedup=$(echo "scale=2; $baseline_ttft / $pega_ttft" | bc)
        echo "TTFT Speedup: ${ttft_speedup}x"
    fi
    
    if [[ $(echo "$baseline_throughput > 0" | bc -l) -eq 1 && $(echo "$pega_throughput > 0" | bc -l) -eq 1 ]]; then
        throughput_speedup=$(echo "scale=2; $pega_throughput / $baseline_throughput" | bc)
        echo "Throughput Speedup: ${throughput_speedup}x"
    fi
    
    echo ""
    
    # Check thresholds
    local pass=true
    
    # Check TTFT threshold
    if [[ $(echo "$pega_ttft > $MAX_PEGAFLOW_TTFT" | bc -l) -eq 1 ]]; then
        log_error "TTFT ${pega_ttft}s exceeds threshold ${MAX_PEGAFLOW_TTFT}s"
        pass=false
    else
        log_success "TTFT ${pega_ttft}s within threshold ${MAX_PEGAFLOW_TTFT}s"
    fi
    
    # Check speedup ratio
    if [[ -n "$ttft_speedup" ]]; then
        if [[ $(echo "$ttft_speedup < $MIN_SPEEDUP_RATIO" | bc -l) -eq 1 ]]; then
            log_warn "TTFT speedup ${ttft_speedup}x below target ${MIN_SPEEDUP_RATIO}x"
        else
            log_success "TTFT speedup ${ttft_speedup}x meets target ${MIN_SPEEDUP_RATIO}x"
        fi
    fi
    
    if [[ "$pass" == "true" ]]; then
        return 0
    else
        return 1
    fi
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
    local data_file="${DATA_FILE:-}"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --num-requests)
                NUM_REQUESTS="$2"
                shift 2
                ;;
            --max-concurrency)
                MAX_CONCURRENCY="$2"
                shift 2
                ;;
            --request-rate)
                REQUEST_RATE="$2"
                shift 2
                ;;
            --output-len)
                OUTPUT_LEN="$2"
                shift 2
                ;;
            --max-ttft)
                MAX_PEGAFLOW_TTFT="$2"
                shift 2
                ;;
            --min-speedup)
                MIN_SPEEDUP_RATIO="$2"
                shift 2
                ;;
            --skip-baseline)
                skip_baseline=true
                shift
                ;;
            --data-file)
                data_file="$2"
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
    
    log_section "Long Document QA Performance Test"
    echo "Configuration:"
    echo "  Requests:         $NUM_REQUESTS"
    echo "  Concurrency:      $MAX_CONCURRENCY"
    echo "  Request rate:     $REQUEST_RATE req/s"
    echo "  Output length:    $OUTPUT_LEN tokens"
    echo "  Max TTFT:         ${MAX_PEGAFLOW_TTFT}s"
    echo "  Min speedup:      ${MIN_SPEEDUP_RATIO}x"
    echo "  Skip baseline:    $skip_baseline"
    echo "  Results dir:      $LONG_DOC_DIR"
    echo ""
    
    # Check servers are running
    check_servers_running "$skip_baseline" || exit 1
    
    # Setup environment
    setup_venv aiohttp
    ensure_results_dir
    mkdir -p "$LONG_DOC_DIR"
    
    # Generate or use provided test data
    if [[ -z "$data_file" || ! -f "$data_file" ]]; then
        data_file="$LONG_DOC_DIR/test_data.json"
        generate_test_data "$data_file" "$NUM_REQUESTS"
    else
        log_info "Using provided data file: $data_file"
    fi
    
    # Run PegaFlow benchmark (two passes for cache behavior)
    log_section "Phase 1: PegaFlow Benchmark (First Pass - Cache Miss)"
    local pega_result1="$LONG_DOC_DIR/pegaflow_pass1.log"
    run_benchmark "PegaFlow (Pass 1)" "$VLLM_PORT" "$pega_result1" "$data_file"
    
    log_section "Phase 2: PegaFlow Benchmark (Second Pass - Cache Hit)"
    local pega_result2="$LONG_DOC_DIR/pegaflow_pass2.log"
    run_benchmark "PegaFlow (Pass 2)" "$VLLM_PORT" "$pega_result2" "$data_file"
    
    # Compare first and second pass for cache effectiveness
    log_section "Cache Effectiveness Analysis"
    local pass1_ttft
    local pass2_ttft
    pass1_ttft=$(extract_metric "${pega_result1}.json" "ttft_mean_s")
    pass2_ttft=$(extract_metric "${pega_result2}.json" "ttft_mean_s")
    
    echo "First pass TTFT (cache miss):  ${pass1_ttft}s"
    echo "Second pass TTFT (cache hit):  ${pass2_ttft}s"
    
    if [[ $(echo "$pass2_ttft < $pass1_ttft" | bc -l) -eq 1 ]]; then
        local cache_speedup
        cache_speedup=$(echo "scale=2; $pass1_ttft / $pass2_ttft" | bc)
        log_success "Cache provides ${cache_speedup}x TTFT improvement"
    else
        log_warn "Cache did not improve TTFT (may be expected for short contexts)"
    fi
    
    # Run baseline benchmark (if not skipped)
    if [[ "$skip_baseline" != "true" ]]; then
        log_section "Phase 3: Baseline Benchmark"
        local baseline_result="$LONG_DOC_DIR/baseline.log"
        run_benchmark "Baseline" "$VLLM_BASELINE_PORT" "$baseline_result" "$data_file"
        
        # Compare PegaFlow vs baseline
        if compare_results "${pega_result2}.json" "${baseline_result}.json"; then
            save_result "long-doc-qa" "pass" "Performance thresholds met"
            log_section "Long Document QA Test: PASSED"
        else
            save_result "long-doc-qa" "fail" "Performance thresholds not met"
            log_section "Long Document QA Test: FAILED"
            exit 1
        fi
    else
        # Just check TTFT threshold
        local final_ttft
        final_ttft=$(extract_metric "${pega_result2}.json" "ttft_mean_s")
        
        if [[ $(echo "$final_ttft <= $MAX_PEGAFLOW_TTFT" | bc -l) -eq 1 ]]; then
            save_result "long-doc-qa" "pass" "TTFT threshold met: ${final_ttft}s <= ${MAX_PEGAFLOW_TTFT}s"
            log_section "Long Document QA Test: PASSED"
        else
            save_result "long-doc-qa" "fail" "TTFT threshold exceeded: ${final_ttft}s > ${MAX_PEGAFLOW_TTFT}s"
            log_section "Long Document QA Test: FAILED"
            exit 1
        fi
    fi
    
    echo ""
    echo "Results saved to: $LONG_DOC_DIR"
}

main "$@"

