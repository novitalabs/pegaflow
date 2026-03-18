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

### Seven-Crate Design

1. **pegaflow-common** (Rust): Shared lightweight utilities
   - `logging.rs`: Unified log initialization (logforth-based)
   - `numa.rs`: NUMA topology detection and CPU affinity utilities
   - Depended on by all other crates to avoid heavy transitive dependencies

2. **pegaflow-core** (Rust): Core storage engine
   - `PegaEngine`: Main engine managing GPU workers and KV cache storage
   - `storage/`: Modular block storage engine
     - `mod.rs`: `StorageEngine` — aggregates allocator, read cache, prefetch, write pipeline, SSD store, remote fetch
     - `read_cache.rs`: Pin/unpin/consume operations on sealed blocks
     - `prefetch.rs`: Per-request SSD prefetch state machine
     - `remote_fetch.rs`: Per-request cross-node remote fetch state machine (oneshot + backpressure)
     - `transfer_lock.rs`: Transfer lock manager — prevents LRU eviction during RDMA transfer
     - `write_path.rs`: Async insert worker thread for batched writes
   - `backing/`: SSD backing store
     - `ssd.rs`: `SsdBackingStore` coordinator
     - `ssd_cache.rs`: SSD-backed block storage
     - `uring.rs`: io_uring async I/O
   - `pinned_pool.rs` / `pinned_mem.rs`: NUMA-aware pinned memory allocator
   - `transfer.rs`: GPU-CPU transfer operations via CUDA
   - `cache.rs`: LRU cache for blocks
   - `gpu_worker.rs`: Per-GPU worker handling async operations
   - `internode/`: Cross-node communication
     - `client.rs`: `PegaflowClient` / `PegaflowClientPool` for querying remote nodes
     - `metaserver_query.rs`: MetaServer query client for block location discovery
     - `remote_fetch_worker.rs`: Async RDMA fetch worker (MetaServer → gRPC → RDMA READ → SealedBlock)
     - `registrar.rs`: Fire-and-forget block hash registration with MetaServer
     - `service_discovery.rs`: Kubernetes pod watcher for PegaFlow instance discovery

3. **pegaflow-proto** (Rust): Protobuf definitions
   - gRPC service definitions built with prost/tonic

4. **pegaflow-server** (Rust): gRPC server
   - `service.rs`: Tonic gRPC service implementation
   - `registry.rs`: Instance/worker registration
   - `http_server.rs`: HTTP health check and Prometheus metrics endpoint
   - `bin/pegaflow-router.rs`: P/D request router (coordinates P/D nodes; PegaFlow itself is a KV store)

5. **pegaflow-metaserver** (Rust): Cross-node block hash registry
   - `service.rs`: gRPC MetaServer service (insert/query block hashes)
   - `store.rs`: LRU block hash store with configurable capacity and TTL (backed by moka)
   - Used for multi-node KV cache coordination — each pegaflow-server registers its block hashes here

6. **pegaflow-transfer** (Rust): RDMA-based inter-node memory transfer engine
   - `engine.rs`: `MooncakeTransferEngine` — Mooncake-compatible API for one-sided RDMA READ/WRITE
   - `sideway_backend.rs`: UD control plane + RC data plane with per-peer sessions
   - `rdma_topo.rs`: NUMA-aware topology detection (GPUs, RDMA NICs, CPUs)
   - CLI tools: `pegaflow_topo_cli` (topology display), `pegaflow_cpu_bench` (RDMA benchmark)

7. **python/** (Rust/PyO3 + Python): Python package (`pegaflow-llm` on PyPI)
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
                                   / \
                   SSD Cache (io_uring)  Remote Node (RDMA READ)
                                              |
                                        MetaServer (block discovery)
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
- `RUST_LOG`: Control Rust logging (e.g., `info,pegaflow_core=debug,pegaflow_server=debug`)

### Git commit message format

- We use commitizen commit message format.
- Do not commit directly to the master branch; create a feat/fix/chore/style/refactor/ci/... branch first.

## Key Files

- `pegaflow-common/src/logging.rs`: Unified log initialization
- `pegaflow-common/src/numa.rs`: NUMA topology detection and CPU affinity
- `pegaflow-core/src/lib.rs`: Main PegaEngine implementation
- `pegaflow-core/src/storage/mod.rs`: StorageEngine (allocator, read cache, prefetch, write pipeline, remote fetch)
- `pegaflow-core/src/storage/read_cache.rs`: Pin/unpin/consume operations
- `pegaflow-core/src/storage/prefetch.rs`: SSD prefetch state machine
- `pegaflow-core/src/storage/remote_fetch.rs`: Cross-node remote fetch state machine
- `pegaflow-core/src/storage/transfer_lock.rs`: Transfer lock manager for RDMA transfers
- `pegaflow-core/src/storage/write_path.rs`: Async insert worker thread
- `pegaflow-core/src/backing/ssd.rs`: SSD backing store coordinator
- `pegaflow-core/src/internode/client.rs`: PegaflowClient/PegaflowClientPool for remote nodes
- `pegaflow-core/src/internode/metaserver_query.rs`: MetaServer query client
- `pegaflow-core/src/internode/remote_fetch_worker.rs`: Async RDMA fetch worker
- `pegaflow-server/src/service.rs`: gRPC service implementation
- `pegaflow-metaserver/src/lib.rs`: MetaServer entry point and CLI
- `pegaflow-metaserver/src/service.rs`: MetaServer gRPC service
- `pegaflow-metaserver/src/store.rs`: Block hash store (LRU + TTL)
- `pegaflow-transfer/src/engine.rs`: RDMA transfer engine (MooncakeTransferEngine)
- `python/src/lib.rs`: PyO3 bindings (Rust side)
- `python/pegaflow/pegaflow.pyi`: Type stubs for PyO3 bindings
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector
- `python/pegaflow/sglang/pegaflow_radix_cache.py`: sglang radix cache class
