# PegaFlow Agent Guide

## Project Summary

PegaFlow is a Rust-based KV cache offloading engine with Python bindings and a
vLLM connector. The repo has three main layers: core engine (`pegaflow-core`),
server (`pegaflow-server`), and Python bindings plus connector (`python/`).

## Repository Layout

```
pegaflow/
├── pegaflow-core/       # Core Rust engine (storage/transfer)
├── pegaflow-server/     # gRPC server + router binaries
├── python/              # PyO3 bindings + Python connector
├── examples/            # Python examples + benchmarks
└── scripts/check.sh     # Local CI checks (fmt/clippy/typos)
```

## Architecture

### Layers

- `pegaflow-core`: GPU-aware KV engine (allocator, block storage, transfer)
- `pegaflow-server`: gRPC server plus a router binary for P/D setups
- `python/`: PyO3 bindings and the vLLM connector package

### Data Flow (Happy Path)

```
vLLM worker -> PegaEngine server -> pinned CPU memory -> GPU restore
```

### Key Concepts

- **Instance**: A inference instance process (e.g., a `vllm serve` run).
- **Worker**: A worker abstraction inside an instance (could be TP/PP/DP).
- **Block**: KV cache storage unit, tracked by content hash.
- **Split storage**: K segments contiguous, then V segments.

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
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --device 0 --pool-size 30gb
```

**Server Options:**
- `--addr`: Bind address (default `127.0.0.1:50055`)
- `--device`: CUDA device ID (default `0`)
- `--pool-size`: Pinned memory pool size (e.g. `10gb`, `500mb`, `1.5tb`)

### Examples & Benchmarks

```bash
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
cargo bench --bench pinned_copy
cargo bench --bench uds_latency
```

### Lint & Formatting

```bash
./scripts/check.sh                      # fmt, typos (optional), clippy, cargo check
cargo fmt --all -- --check
cargo clippy -p pegaflow-core --all-targets -- -D warnings
cargo clippy -p pegaflow-server --all-targets -- -D warnings
```

### Tests

```bash
cargo test
cargo test -p pegaflow-core
cargo test -p pegaflow-server
```

#### Single-Test Examples

```bash
cargo test -p pegaflow-core allocator::tests::creates_allocator_with_scaled_capacity
cargo test -p pegaflow-core sync_state::tests::create_and_attach_round_trip
cargo test -p pegaflow-core pinned_mem::tests::test_zero_size_fails
cargo test -p pegaflow-server utils::tests::test_parse_memory_size_basic
```

## Environment Variables

- `PEGAFLOW_ENGINE_ENDPOINT`: ZMQ endpoint (default `ipc:///tmp/pega_engine.sock`)
- `PEGAFLOW_INSTANCE_ID`: Override instance ID
- `RUST_LOG`: Rust logging (e.g. `info,pegaflow_core=debug`)
- `PYO3_PYTHON`: Python interpreter for PyO3 builds

## Where to Make Changes

- Core engine logic → `pegaflow-core/`
- gRPC/service/router changes → `pegaflow-server/`
- Python API surface → `python/src/lib.rs`
- vLLM connector logic → `python/pegaflow/connector/`

## Key Files

- `pegaflow-core/src/lib.rs`: Main PegaEngine implementation
- `pegaflow-server/src/lib.rs`: gRPC server wiring and tracing setup
- `python/src/lib.rs`: PyO3 bindings for PegaEngine + client
- `python/pegaflow/connector/`: vLLM v1 connector logic

## Code Style Guidelines

### Rust

- Run `cargo fmt` and keep formatting rustfmt-compatible.
- Keep `use` blocks ordered: std → external crates → local crate.
- Naming: `snake_case` for functions/modules, `CamelCase` for types/traits.
- Prefer explicit error enums + `Display` impls; use `Result<T, E>` and `?`.
- Avoid `unwrap`/`expect` outside tests; bubble errors instead.
- Logging via `tracing` macros (`info!`, `warn!`, `error!`).
- Unsafe code: prefer `NonNull` over raw `*mut` where possible.
- Keep `#[allow(...)]` scoped and justified (e.g., `too_many_arguments`).

### Python (3.10+)

- Use native generics (`list`, `dict`, `tuple`) instead of `typing.List`.
- Use PEP 604 unions (`X | None`) instead of `Optional`.
- Logging uses `%s` formatting (avoid f-strings in log calls).
- Keep imports grouped: standard lib → third-party → local modules.
- Type hint public APIs in `python/pegaflow/` when practical.

### PyO3 Bindings

- Use `#[pyclass]` and `#[pymethods]` for Python-exposed types.
- Convert Rust errors to `PyErr` cleanly; avoid panics crossing the FFI.
- Keep Python-facing APIs thin; delegate core logic to `pegaflow-core`.

### Git commit message format

we use commitizen commit message format.
