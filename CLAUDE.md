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

### Running Benchmarks
```bash
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

### Running Examples
```bash
# Start the PegaEngine server first (required)
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --device 0 --pool-size 30gb

# Then run examples
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
```

**Server Configuration:**
- `--addr`: Bind address (default: `127.0.0.1:50055`)
- `--device`: CUDA device ID (default: `0`)
- `--pool-size`: Pinned memory pool size (default: `30gb`, supports: `kb`, `mb`, `gb`, `tb`)

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

## Key Files

- `pegaflow-core/src/lib.rs`: Main PegaEngine implementation
- `pegaflow-core/src/storage.rs`: Block storage engine
- `pegaflow-server/src/service.rs`: gRPC service implementation
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector