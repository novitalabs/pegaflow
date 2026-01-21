# PegaFlow CI Test Suite

This directory contains CI test scripts for validating PegaFlow's functionality and performance. The test suite is modeled after [LMCache's multiprocessing-test](https://github.com/LMCache/LMCache/tree/dev/.buildkite/scripts/multiprocessing-test).

## Quick Start

```bash
# Run the full test suite
MODEL=/path/to/model ./run-all-tests.sh

# Quick mode (skip baseline comparisons)
MODEL=Qwen/Qwen2.5-7B ./run-all-tests.sh --quick

# Run individual tests
MODEL=/path/to/model ./run-lm-eval.sh
MODEL=/path/to/model ./run-long-doc-qa.sh
MODEL=/path/to/model ./run-share-gpt.sh
```

## Test Scripts

### `run-all-tests.sh` - Orchestrator

The main entry point that:
1. Starts all required servers (PegaFlow + vLLM)
2. Runs test scripts sequentially
3. Collects results and generates summary
4. Cleans up servers on exit

```bash
# Full suite
MODEL=/models/Llama-3.1-8B ./run-all-tests.sh

# Quick mode (no baseline comparisons, faster)
MODEL=/models/Llama-3.1-8B ./run-all-tests.sh --quick

# Run specific tests only
./run-all-tests.sh --lm-eval-only
./run-all-tests.sh --skip-share-gpt
```

### `run-lm-eval.sh` - Consistency Test

Validates cache correctness by running lm_eval twice:
- First run populates the cache
- Second run uses cached results
- Compares output samples - they must be **identical**

```bash
MODEL=Qwen/Qwen2.5-7B ./run-lm-eval.sh --task gsm8k --limit 100
```

**Pass Criteria**: Sample outputs from both runs are byte-identical.

### `run-long-doc-qa.sh` - Performance Test

Benchmarks long document Q&A performance:
- Runs benchmark against PegaFlow-enabled vLLM (twice for cache effect)
- Optionally compares against vanilla vLLM baseline
- Validates TTFT and speedup thresholds

```bash
MODEL=/models/Llama-3.1-8B ./run-long-doc-qa.sh --num-requests 100

# Skip baseline comparison
./run-long-doc-qa.sh --skip-baseline
```

**Pass Criteria**:
- TTFT ≤ `MAX_PEGAFLOW_TTFT` (default: 0.22s)
- Speedup ≥ `MIN_SPEEDUP_RATIO` (default: 1.2x)

### `run-share-gpt.sh` - Multi-Turn Benchmark

Tests multi-turn conversation performance:
- Uses ShareGPT dataset for realistic conversations
- Measures TTFT across conversation turns
- Compares PegaFlow vs baseline performance

```bash
MODEL=Qwen/Qwen2.5-7B ./run-share-gpt.sh --num-conversations 50 --max-turns 5
```

### `start-servers.sh` - Server Management

Manages PegaFlow and vLLM server lifecycle:

```bash
# Start all servers
MODEL=/path/to/model ./start-servers.sh start-all

# Start without baseline
MODEL=/path/to/model ./start-servers.sh start-all --no-baseline

# Check status
./start-servers.sh status

# Stop all servers
./start-servers.sh stop-all
```

### `common.sh` - Shared Utilities

Provides common functions (sourced by other scripts):
- `setup_venv`: Python virtual environment management
- `wait_for_http`: Server health checking
- `cleanup_processes`: Graceful shutdown
- Configuration variables and thresholds

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | (required) | Model path or HuggingFace model ID |
| `SERVED_MODEL_NAME` | `$MODEL` | Served model name for API |
| `CHAT_TEMPLATE` | | Path to chat template file |
| `VLLM_PORT` | `8000` | vLLM with PegaFlow port |
| `VLLM_BASELINE_PORT` | `9000` | Vanilla vLLM port (for comparison) |
| `PEGA_PORT` | `50055` | PegaFlow gRPC port |
| `PEGA_METRICS_PORT` | `9091` | PegaFlow metrics port |
| `PEGA_POOL_SIZE` | `50gb` | PegaFlow pinned memory pool |
| `PEGA_DEVICE` | `0` | GPU device for PegaFlow |
| `PEGA_SSD_CACHE_PATH` | `/tmp/pegaflow_ci_ssd_cache` | SSD cache path |
| `PEGA_SSD_CACHE_CAPACITY` | `100gb` | SSD cache capacity |
| `RESULTS_DIR` | `.ci/results/<timestamp>` | Output directory |
| `MAX_PEGAFLOW_TTFT` | `0.22` | Maximum TTFT threshold (seconds) |
| `MIN_SPEEDUP_RATIO` | `1.2` | Minimum speedup vs baseline |

### LM-Eval Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `gsm8k` | lm_eval task |
| `--limit` | `100` | Number of samples |
| `--concurrent` | `10` | Concurrent requests |
| `--seed` | `0` | Random seed |

### Long-Doc QA Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-requests` | `50` | Number of requests |
| `--max-concurrency` | `10` | Max concurrent requests |
| `--request-rate` | `2` | Requests per second |
| `--output-len` | `128` | Output tokens per request |
| `--max-ttft` | `0.22` | TTFT threshold (seconds) |
| `--min-speedup` | `1.2` | Minimum speedup ratio |
| `--skip-baseline` | | Skip baseline comparison |

### ShareGPT Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-conversations` | `50` | Number of conversations |
| `--num-clients` | `2` | Parallel clients |
| `--max-active` | `10` | Max active conversations |
| `--max-turns` | `5` | Max turns per conversation |
| `--request-rate` | `0.5` | Rate per client |
| `--warmup` | | Run warmup step |
| `--skip-baseline` | | Skip baseline comparison |

## Output Structure

```
.ci/results/<build-id>/
├── logs/
│   ├── pegaflow-server.log
│   ├── vllm-pegaflow.log
│   └── vllm-baseline.log
├── lm_eval/
│   ├── first_run/
│   └── second_run/
├── long_doc_qa/
│   ├── pegaflow_pass1.log
│   ├── pegaflow_pass2.log
│   └── baseline.log
├── share_gpt/
│   ├── pegaflow/
│   └── baseline/
├── *.result          # Individual test results
└── overall.result    # Overall pass/fail
```

## Integration

### GitHub Actions

```yaml
jobs:
  ci-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      
      - name: Run CI Tests
        run: |
          cd pegaflow
          MODEL=Qwen/Qwen2.5-7B ./.ci/run-all-tests.sh --quick
        
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: ci-results
          path: pegaflow/.ci/results/
```

### BuildKite

```yaml
steps:
  - label: ":test_tube: PegaFlow CI Tests"
    command: |
      cd pegaflow
      MODEL=\$MODEL ./.ci/run-all-tests.sh
    agents:
      queue: gpu
    artifact_paths:
      - "pegaflow/.ci/results/**/*"
```

### Local Development

```bash
# Build PegaFlow first
cargo build --release

# Run quick validation
MODEL=/path/to/small/model ./.ci/run-all-tests.sh --quick --skip-share-gpt

# Run full suite overnight
MODEL=/path/to/model ./.ci/run-all-tests.sh 2>&1 | tee ci-output.log
```

## Troubleshooting

### Server Startup Failures

```bash
# Check server logs
cat .ci/results/*/logs/pegaflow-server.log
cat .ci/results/*/logs/vllm-pegaflow.log

# Manually check server status
./.ci/start-servers.sh status

# Clean up stale processes
./.ci/start-servers.sh stop-all
```

### GPU Issues

```bash
# Verify CUDA availability
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Use specific GPU
PEGA_DEVICE=1 MODEL=/path/to/model ./run-all-tests.sh
```

### Test Failures

```bash
# Check individual test results
cat .ci/results/*/lm-eval-consistency.result
cat .ci/results/*/long-doc-qa.result

# Re-run specific test with more verbosity
MODEL=/path/to/model ./run-lm-eval.sh --limit 10 2>&1 | tee debug.log
```

## Requirements

- Python 3.10+
- CUDA-capable GPU
- Rust toolchain (for building PegaFlow)
- vLLM with PegaFlow connector support
- lm-eval (for LM-Eval tests)

