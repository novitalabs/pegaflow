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

### Running Benchmarks

```bash
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

## Architecture

### Five-Crate Design

1. **pegaflow-core** (Rust): Core storage engine
   - `PegaEngine`: Main engine managing GPU workers and KV cache storage
   - `storage.rs`: Block-based storage with content-addressed blocks (`check_prefix_memory_only` for memory-only, `check_prefix_and_prefetch` for SSD)
   - `pinned_pool.rs` / `pinned_mem.rs`: Pinned memory allocator
   - `transfer.rs`: GPU-CPU transfer operations via CUDA
   - `cache.rs`: LRU cache for blocks
   - `gpu_worker.rs`: Per-GPU worker handling async operations
   - `internode/`: Cross-node communication — `PegaflowClient` for querying remote nodes, service discovery via metaserver

2. **pegaflow-proto** (Rust): Protobuf definitions
   - gRPC service definitions built with prost/tonic

3. **pegaflow-server** (Rust): gRPC server
   - `service.rs`: Tonic gRPC service implementation
   - `registry.rs`: Instance/worker registration
   - `bin/pegaflow-router.rs`: P/D disaggregation router

4. **pegaflow-metaserver** (Rust): Cross-node block hash registry
   - `service.rs`: gRPC MetaServer service (insert/query block hashes)
   - `store.rs`: LRU block hash store with configurable capacity and TTL (backed by moka)
   - Used for multi-node KV cache coordination — each pegaflow-server registers its block hashes here

5. **python/** (Rust/PyO3 + Python): Python package (`pegaflow-llm` on PyPI)
   - `src/lib.rs`: PyO3 bindings exposing `PegaEngine` and gRPC client
   - `pegaflow/connector/`: vLLM v1 KV connector (scheduler + worker split)
   - `pegaflow/sglang/`: SGLang integration
   - `pegaflow/ipc_wrapper.py`: CUDA IPC handle wrapper
   - CLI binaries: `pegaflow-server`, `pegaflow-metaserver` (installed via pip)

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

- **Visibility**: Prefer `fn` (private) > `pub(crate)` > `pub`; use the minimal necessary visibility
- Prefer `NonNull` over `*mut` in unsafe code

### Python (3.10+)

- Use native generics (`list`, `dict`, `set`, `tuple`) instead of `typing.List`, `typing.Dict`, etc.
- Use PEP 604 union syntax (`X | Y`, `X | None`) instead of `typing.Union`, `typing.Optional`
- Logging: use `%s` formatting (`logger.info("x=%s", x)`) instead of f-strings to avoid evaluation overhead

## Environment Variables

- `PEGAFLOW_ENGINE_ENDPOINT`: gRPC endpoint (default: `127.0.0.1:50055`)
- `PEGAFLOW_INSTANCE_ID`: Override instance ID
- `PEGAFLOW_HOST_IP`: Host IP used for metaserver advertise address (fallback when `--advertise-addr` is not set)
- `RUST_LOG`: Control Rust logging (e.g., `info,pegaflow_core=debug,pegaflow_server=debug`)

### Git commit message format

- We use commitizen commit message format.
- Do not commit directly to the master branch; create a feat/fix/chore/style/refactor/ci/... branch first.

## Key Files

- `pegaflow-core/src/lib.rs`: Main PegaEngine implementation
- `pegaflow-core/src/storage.rs`: Block storage engine
- `pegaflow-core/src/internode/`: Cross-node client and service discovery
- `pegaflow-core/src/numa.rs`: NUMA topology detection and GPU affinity queries
- `pegaflow-server/src/service.rs`: gRPC service implementation
- `pegaflow-metaserver/src/lib.rs`: MetaServer entry point and CLI
- `pegaflow-metaserver/src/service.rs`: MetaServer gRPC service
- `pegaflow-metaserver/src/store.rs`: Block hash store (LRU + TTL)
- `python/src/lib.rs`: PyO3 bindings (Rust side)
- `python/pegaflow/pegaflow.pyi`: Type stubs for PyO3 bindings
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector
- `python/pegaflow/sglang/pegaflow_radix_cache.py`: sglang radix cache class
