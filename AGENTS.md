# PegaFlow Agent Guide

A comprehensive guide for LLM agents working with the PegaFlow repository. PegaFlow is a high-performance KV cache transfer system for LLM inference acceleration, supporting vLLM and SGLang.

## Project Overview

PegaFlow enables efficient KV cache offloading and sharing for LLM inference workloads:

- **Single-node KV cache offloading** — offload KV cache to host memory and restore it back to GPU with minimal latency
- **P/D disaggregation** — separate prefill and decode phases across GPUs for better resource utilization
- **Prefix Caching** — reuse computed KV cache across requests
- **High-performance transport** — based on CUDA IPC and zero-copy techniques
- **~9x TTFT improvement** — warm-start requests achieve dramatically lower time-to-first-token

## Repository Layout

```
pegaflow/
├── pegaflow-core/          # Core Rust engine (storage/transfer/allocator)
├── pegaflow-proto/         # Protobuf/gRPC definitions
├── pegaflow-server/        # gRPC server + P/D Router
├── python/                 # PyO3 bindings + Python connector
│   ├── src/lib.rs          # PyO3 bindings entry point
│   ├── pegaflow/connector/ # vLLM v1 connector (scheduler/worker)
│   ├── pegaflow/sglang/    # SGLang RadixCache integration
│   └── pegaflow.pyi        # Python type stubs
├── examples/               # Python examples and benchmarks
└── scripts/check.sh        # Local CI check script
```

## Architecture Layers

### 1. pegaflow-core (Storage Engine)

Core storage and transfer logic, independent of the network layer.

| File | Responsibility |
|------|----------------|
| `lib.rs` | Main `PegaEngine` implementation. Top-level orchestrator that manages inference instances, coordinates GPU worker pools for async transfers, and interfaces with the storage engine. Provides the primary API for register, save, and load operations. |
| `storage.rs` | Core block storage engine. Manages the two-phase block lifecycle: inflight (being written) → sealed (immutable) → cached. Handles allocation from pinned memory pool, block lookup by hash, LRU eviction, and coordinates with SSD cache tier for overflow storage. |
| `block.rs` | Block type definitions including `BlockKey` (namespace + hash), `BlockHash`, `LayerBlock` (single layer data in pinned memory), `SealedBlock` (immutable complete block), and `InflightBlock` (partially-filled block being written). Also defines `PrefetchStatus` for SSD prefetch results. |
| `allocator.rs` | Byte-addressable offset allocator wrapper. Scales large memory regions into u32-addressable units for the underlying `offset-allocator` crate, enabling efficient allocation/deallocation of variable-sized blocks within the pinned memory pool. |
| `pinned_pool.rs` | High-level pinned memory pool manager. Wraps `PinnedMemory` to provide an allocator interface with RAII guards (`PinnedAllocation`). Tracks pool usage metrics and handles fragmentation by exposing largest-free-region statistics. |
| `pinned_mem.rs` | Low-level CUDA pinned memory allocation. Provides two strategies: write-combined via `cudaHostAlloc` (optimized for CPU→GPU transfers) and huge pages via `mmap(MAP_HUGETLB)` + `cudaHostRegister` (faster allocation for large buffers). |
| `transfer.rs` | GPU ↔ CPU data transfer primitives. Implements async CUDA memcpy operations (`cuMemcpyDtoHAsync`, `cuMemcpyHtoDAsync`) and batch copy optimization that merges contiguous segments to reduce PCIe transaction overhead. |
| `gpu_worker.rs` | Per-GPU worker pool spawning dedicated load and save threads. Each worker maintains its own CUDA context and stream, processing tasks from unbounded channels. Handles both contiguous and split K/V layout transfers. |
| `cache.rs` | TinyLFU-enhanced LRU cache for sealed blocks. Uses a Count-Min Sketch frequency estimator for admission control (rejecting one-hit wonders) and LRU for eviction. Windowed aging prevents counter saturation over time. |
| `ssd_cache.rs` | SSD cache configuration and types. Defines `SsdCacheConfig` with tunable queue depths and inflight limits, `SsdIndexEntry` for block location tracking in the ring buffer, and batch types for write and prefetch operations. |
| `sync_state.rs` | Shared-memory synchronization primitive for async load operations.
| `seal_offload.rs` | Slot metadata types for SSD serialization.
| `uring.rs` | io_uring-based async I/O engine. Manages a submission/completion queue pair with configurable depth, handling read, write, readv, and writev operations on the SSD cache file descriptor in a dedicated thread. |
| `metrics.rs` | OpenTelemetry metrics collection. Defines `CoreMetrics` with counters for pool usage, cache hits/misses, GPU transfer bytes/duration, and SSD queue depths. Uses histograms with custom boundaries for latency and throughput. |
| `instance.rs` | Hierarchical context management. `InstanceContext` represents a model inference process with topology (layers, TP size); `GpuContext` manages per-device CUDA context and worker pool; `KVCacheRegistration` validates and stores layer memory layouts. |
| `logging.rs` | Unified logging configuration for Rust components. Provides `LoggingConfig` builder for setting log level, output destination (stdout/stderr), and colored output. Filters noisy dependency modules (h2, hyper, tonic) by default. |

### 2. pegaflow-proto (Protocol Definitions)

```
src/lib.rs    # Prost-generated gRPC code
build.rs      # Protobuf build script
```

Proto files define service interfaces including instance registration, KV operations, etc.

### 3. pegaflow-server (Network Service Layer)

| File | Responsibility |
|------|----------------|
| `main.rs` | Server binary entry point |
| `service.rs` | Tonic gRPC service implementation |
| `registry.rs` | Instance/worker registry |
| `bin/pegaflow-router.rs` | P/D disaggregation router |
| `utils.rs` | Utility functions (e.g., memory size parsing) |
| `metric.rs` | Server metrics and monitoring |

### 4. python/ (Python Bindings and Connectors)

**PyO3 Bindings** (`src/lib.rs`):
- `PegaEngineClient`: Python-accessible engine client
- Exposes core APIs: register instance, save/load KV, prefetch, etc.

**vLLM Connector** (`pegaflow/connector/`):

| File | Responsibility |
|------|----------------|
| `scheduler.py` | vLLM Scheduler-side connector, manages prefix caching logic |
| `worker.py` | vLLM Worker-side connector, executes actual KV transfers |
| `state_manager.py` | Load/save intent state management |
| `common.py` | Shared types and constants |

**SGLang Integration** (`pegaflow/sglang/`):

| File | Responsibility |
|------|----------------|
| `peagflow_radix_cache.py` | Drop-in replacement for SGLang's `RadixCache` |

**Utility Modules**:

| File | Responsibility |
|------|----------------|
| `ipc_wrapper.py` | CUDA IPC handle wrapper |
| `_server.py` | Python-side server launcher |
| `logging_utils.py` | Logging utilities |

## Core Concepts

- **Instance**: An inference instance process (e.g., a `vllm serve` run), identified by `num_layers` and `tp_size`
- **Worker**: A tensor-parallel rank within an instance (TP/PP/DP)
- **Block**: KV cache storage unit, identified by content hash
- **Split Storage**: K and V segments stored separately — all K blocks contiguous, then all V blocks contiguous
- **Layer-First Layout**: vLLM's KV layout — all layers' K contiguous, then all layers' V contiguous

## Data Flow

```
┌─────────────────┐     gRPC      ┌─────────────────┐     CUDA IPC     ┌─────────────┐
│ vLLM/SGLang     │ ◄────────────► │ PegaEngine      │ ◄───────────────► │ GPU Memory  │
│ Worker          │                │ Server          │                   │             │
└─────────────────┘                └─────────────────┘                   └─────────────┘
                                          │
                                          │ manages
                                          ▼
                                   ┌─────────────────┐
                                   │ Pinned CPU      │
                                   │ Memory Pool     │
                                   └─────────────────┘
                                          │
                                          ▼
                                   ┌─────────────────┐
                                   │ SSD Cache       │
                                   │ (optional)      │
                                   └─────────────────┘
```

## Build, Lint, Test

### Rust Builds

```bash
cargo build
cargo build --release
```

### Python Bindings (PyO3)

```bash
cd python
maturin develop
maturin develop --release
```

### Run the Server

```bash
# Auto-detect all GPUs (default)
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --pool-size 30gb

# Or specify specific devices
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --devices 0,1,2,3 --pool-size 30gb
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

### Examples & Benchmarks

```bash
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

### Lint & Formatting

```bash
./scripts/check.sh                      # fmt, typos, clippy, cargo check
```

### Tests

```bash
cargo test
```

## Environment Variables

- `PEGAFLOW_ENGINE_ENDPOINT`: gRPC endpoint (default: `127.0.0.1:50055`)
- `PEGAFLOW_INSTANCE_ID`: Override instance ID
- `RUST_LOG`: Rust logging (e.g., `info,pegaflow_core=debug`)
- `PYO3_PYTHON`: Python interpreter for PyO3 builds

## Where to Make Changes

| Target | Location |
|--------|----------|
| Core engine logic | `pegaflow-core/` |
| gRPC/service/router changes | `pegaflow-server/` |
| Python API surface | `python/src/lib.rs` |
| vLLM connector logic | `python/pegaflow/connector/` |
| SGLang integration | `python/pegaflow/sglang/` |

## Key Files Quick Reference

| Functionality | File |
|---------------|------|
| Engine core | `pegaflow-core/src/lib.rs` |
| Storage engine | `pegaflow-core/src/storage.rs` |
| Block management | `pegaflow-core/src/block.rs` |
| NUMA topology | `pegaflow-core/src/numa.rs` |
| gRPC service | `pegaflow-server/src/service.rs` |
| PyO3 bindings | `python/src/lib.rs` |
| vLLM Scheduler | `python/pegaflow/connector/scheduler.py` |
| vLLM Worker | `python/pegaflow/connector/worker.py` |
| SGLang integration | `python/pegaflow/sglang/peagflow_radix_cache.py` |
| Type stubs | `python/pegaflow/pegaflow.pyi` |

## Code Style Guidelines

### Rust

- Run `cargo fmt` and keep formatting rustfmt-compatible
- Keep `use` blocks ordered: std → external crates → local crate
- Naming: `snake_case` for functions/modules, `CamelCase` for types/traits
- Prefer explicit error enums + `Display` impls; use `Result<T, E>` and `?`
- Avoid `unwrap`/`expect` outside tests; bubble errors instead
- Logging via `tracing` macros (`info!`, `warn!`, `error!`)
- Unsafe code: prefer `NonNull` over raw `*mut` where possible
- Keep `#[allow(...)]` scoped and justified (e.g., `too_many_arguments`)

### Python (3.10+)

- Use native generics (`list`, `dict`, `tuple`) instead of `typing.List`, `typing.Dict`, etc.
- Use PEP 604 unions (`X | None`) instead of `Optional`
- Logging uses `%s` formatting (avoid f-strings in log calls)
- Keep imports grouped: standard lib → third-party → local modules
- Type hint public APIs in `python/pegaflow/` when practical

### PyO3 Bindings

- Use `#[pyclass]` and `#[pymethods]` for Python-exposed types
- Convert Rust errors to `PyErr` cleanly; avoid panics crossing the FFI
- Keep Python-facing APIs thin; delegate core logic to `pegaflow-core`
- **Important:** When modifying `python/src/lib.rs`, update `python/pegaflow/pegaflow.pyi` to keep type hints in sync

### General

- Use English in comments
- Use `.venv` for Python virtual environment

### Git Commit Message Format

- We use [Commitizen](https://commitizen-tools.github.io/commitizen/) commit message format
- Do not commit directly to the master branch; create a `feat/`/`fix/`/`chore/`/`style/`/`refactor/`/`ci/`/... branch first
- Use `cz c` for interactive commit message creation
