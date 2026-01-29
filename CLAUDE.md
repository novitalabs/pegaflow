# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PegaFlow is a high-performance KV cache transfer system for LLM inference, designed to work with vLLM and SGLang. It provides RDMA-first, high-bandwidth transport optimized for GPU-to-CPU KV cache offloading and loading.

## Build Commands

### Rust Core

```bash
cargo build              # Debug build
cargo build --release    # Release build
cargo test               # Run Rust tests
```

### CI Checks (Local)

Run all CI checks locally before committing:

```bash
./scripts/check.sh       # Run fmt, typos, clippy, and cargo check
```

### Python Bindings (PyO3 via maturin)

```bash
cd python
maturin develop          # Dev build
maturin develop --release  # Release build
```

**Important:** When modifying `python/src/lib.rs` (PyO3 bindings), update the type stub file `python/pegaflow/pegaflow.pyi` to keep type hints in sync.

### Running Benchmarks

```bash
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

### Running Examples

```bash
# Start the PegaEngine server first (required)
# Auto-detect all GPUs (default behavior)
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --pool-size 30gb

# Or specify specific devices (e.g., GPUs 0, 2, 4)
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --devices 0,2,4 --pool-size 30gb

# Then run examples
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
```

**Server Options:**

- `--addr`: Bind address (default: `127.0.0.1:50055`)
- `--devices`: CUDA device IDs, comma-separated (default: auto-detect all available GPUs, e.g., `--devices 0,1,2,3`)
- `--pool-size`: Pinned memory pool size (default: `30gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--hint-value-size`: Hint for typical value size to tune cache and allocator (optional, supports: `kb`, `mb`, `gb`, `tb`)
- `--use-hugepages`: Use huge pages for pinned memory (default: `false`, requires pre-configured `/proc/sys/vm/nr_hugepages`)
- `--disable-lfu-admission`: Disable TinyLFU cache admission, fallback to plain LRU (default: `false`)
- `--metrics-addr`: Prometheus metrics HTTP endpoint (default: `0.0.0.0:9091`, set to empty to disable)
- `--metrics-otel-endpoint`: **DEPRECATED** - OTLP metrics export endpoint (optional, leave unset to disable)
- `--metrics-period-secs`: **DEPRECATED** - Metrics export period in seconds (default: `5`, only used with OTLP)
- `--log-level`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--ssd-cache-path`: Enable SSD cache by providing cache file path (optional)
- `--ssd-cache-capacity`: SSD cache capacity (default: `512gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--ssd-write-queue-depth`: SSD write queue depth, max pending write batches (default: `8`)
- `--ssd-prefetch-queue-depth`: SSD prefetch queue depth, max pending prefetch batches (default: `2`)
- `--ssd-write-inflight`: SSD write inflight, max concurrent block writes (default: `2`)
- `--ssd-prefetch-inflight`: SSD prefetch inflight, max concurrent block reads (default: `16`)
- `--max-prefetch-blocks`: Max blocks allowed in prefetching state, backpressure for SSD prefetch (default: `800`)

## Architecture

### Four-Crate Design

1. **pegaflow-core** (Rust): Core storage engine
   - `PegaEngine`: Main engine managing GPU workers and KV cache storage
   - `storage.rs`: Block-based storage with content-addressed blocks
   - `pinned_pool.rs` / `pinned_mem.rs`: Pinned memory allocator
   - `transfer.rs`: GPU-CPU transfer operations via CUDA
   - `cache.rs`: LRU cache for blocks
   - `gpu_worker.rs`: Per-GPU worker handling async operations

2. **pegaflow-proto** (Rust): Protobuf definitions
   - gRPC service definitions built with prost/tonic

3. **pegaflow-server** (Rust): gRPC server
   - `service.rs`: Tonic gRPC service implementation
   - `registry.rs`: Instance/worker registration
   - `bin/pegaflow-router.rs`: P/D disaggregation router

4. **python/** (Rust/PyO3 + Python): Python package (`pegaflow-llm` on PyPI)
   - `src/lib.rs`: PyO3 bindings exposing `PegaEngine` and gRPC client
   - `pegaflow/connector/`: vLLM v1 KV connector (scheduler + worker split)
   - `pegaflow/sglang/`: SGLang integration
   - `pegaflow/ipc_wrapper.py`: CUDA IPC handle wrapper

### Data Flow

```
vLLM/SGLang Worker <--gRPC--> PegaEngine Server <--CUDA IPC--> GPU Memory
                                    |
                             Pinned CPU Memory (KV cache storage)
```

### Key Concepts

- **Instance**: A model instance with specific num_layers and tp_size
- **Worker**: A tensor-parallel rank within an instance
- **Block**: Unit of KV cache storage, identified by content hash
- **Split Storage**: K and V segments stored separately for efficient batching

## Code Conventions

### General

- Use English in comments
- Use `.venv` for Python virtual environment
- KV cache uses layer-first layout: all K blocks contiguous, followed by all V blocks

### Rust

- Prefer `NonNull` over `*mut` in unsafe code

### Python (3.10+)

- Use native generics (`list`, `dict`, `set`, `tuple`) instead of `typing.List`, `typing.Dict`, etc.
- Use PEP 604 union syntax (`X | Y`, `X | None`) instead of `typing.Union`, `typing.Optional`
- Logging: use `%s` formatting (`logger.info("x=%s", x)`) instead of f-strings to avoid evaluation overhead

## Environment Variables

- `PEGAFLOW_ENGINE_ENDPOINT`: gRPC endpoint (default: `127.0.0.1:50055`)
- `PEGAFLOW_INSTANCE_ID`: Override instance ID
- `RUST_LOG`: Control Rust logging (e.g., `info,pegaflow_core=debug,pegaflow_server=debug`)

## vLLM Integration

Configure vLLM to use PegaFlow:

```python
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="PegaKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="pegaflow.connector",
)
```

## Sglang integration

PegaFlow integrates with SGLang by providing a drop-in replacement for the default `RadixCache` that uses the high-throughput PegaEngine server for distributed KV cache management.

### How to Use

1. **Import and Instantiate PeagflowRadixCache**

In your SGLang-based project, in `scheduler.py`, swap out the usual `RadixCache` for `PeagflowRadixCache`:

```python
from pegaflow.sglang.peagflow_radix_cache import PeagflowRadixCache

# Example instantiation inside the cache initialization logic:
kv_cache = PeagflowRadixCache(
    params=cache_params,       # sglang.srt.mem_cache.cache_init_params.CacheInitParams
    model_config=model_config, # sglang.srt.configs.model_config.ModelConfig
    tp_size=tp_size,           # Tensor parallel world size
    rank=tp_rank,              # Local TP rank
)
```

2. **Environment Variables**

- `PEGAFLOW_ENGINE_ENDPOINT`: Override the gRPC endpoint for the PegaEngine server. Defaults to `http://127.0.0.1:50055`.
- `PEGAFLOW_INSTANCE_ID`: Optional override for instance identification across processes.

3. **Automatic CUDA IPC Handling**

`PeagflowRadixCache` automatically registers all layer KV cache tensors for CUDA IPC with the PegaEngine upon construction, making them available for RDMA/pinned memory transfer.

4. **Behavioral Differences vs. Default RadixCache**

- On prefix miss, it queries the PegaEngine for remote blocks and loads missing blocks directly into local GPU buffers.
- When a request finishes, changed blocks are saved back and announced to the engine.
- All KV operations (query/load/save) are batched per block and per layer for efficiency.
- Integration is non-intrusive: simply swap the class during initialization.

5. **Shutdown and Cleanup**

The class automatically unregisters contexts with the engine on deletion, interpreter shutdown, or Ctrl+C.

### Reference

See the implementation in [`peagflow_radix_cache.py`](python/pegaflow/sglang/peagflow_radix_cache.py) for full details and entry points:

- Custom `match_prefix`
- Remote block query/load/save
- Context (un)registration
- Layer-wise registration for MLA/TP
- GPU KV allocator evict

### Git commit message format

- We use commitizen commit message format.
- Do not commit directly to the master branch; create a feat/fix/chore/style/refactor/ci/... branch first.

## Key Files

- `pegaflow-core/src/lib.rs`: Main PegaEngine implementation
- `pegaflow-core/src/storage.rs`: Block storage engine
- `pegaflow-core/src/numa.rs`: NUMA topology detection and GPU affinity queries
- `pegaflow-server/src/service.rs`: gRPC service implementation
- `python/src/lib.rs`: PyO3 bindings (Rust side)
- `python/pegaflow/pegaflow.pyi`: Type stubs for PyO3 bindings
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector
- `python/pegaflow/sglang/pegaflow_radix_cache.py`: sglang radix cache class
