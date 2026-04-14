# PegaFlow Agent Guide

This file provides guidance for agents working in the PegaFlow repository.

## Project Overview

PegaFlow is a high-performance KV cache transfer system for LLM inference, designed for vLLM and SGLang.

- Single-node KV cache offloading between GPU and host memory
- Cross-node KV cache sharing via RDMA
- Prefix cache reuse for repeated requests
- Python bindings and connectors for inference frameworks

## Repository Layout

```text
pegaflow/
├── pegaflow-common/       # Shared utilities such as logging and NUMA helpers
├── pegaflow-core/         # Core KV cache engine, storage, transfer, backing store
├── pegaflow-proto/        # Protobuf and gRPC definitions
├── pegaflow-server/       # gRPC server, router, HTTP metrics/health endpoints
├── pegaflow-metaserver/   # Cross-node block metadata registry
├── pegaflow-transfer/     # RDMA transfer layer
├── python/                # PyO3 bindings and Python integrations
├── examples/              # Python examples and benchmarks
├── scripts/               # Project helper scripts
└── prek.toml              # Local check configuration
```

## Where To Change Code

| Target | Location |
|--------|----------|
| Shared Rust utilities | `pegaflow-common/` |
| Core engine and storage path | `pegaflow-core/` |
| gRPC protocol changes | `pegaflow-proto/` |
| Server and router logic | `pegaflow-server/` |
| Cross-node metadata service | `pegaflow-metaserver/` |
| RDMA transfer path | `pegaflow-transfer/` |
| PyO3 bindings | `python/src/lib.rs` |
| Python package and helpers | `python/pegaflow/` |
| vLLM connector | `python/pegaflow/connector/` |
| SGLang integration | `python/pegaflow/sglang/` |

## Key Entry Points

- `pegaflow-core/src/lib.rs`: main Rust engine entry
- `pegaflow-core/src/storage/mod.rs`: storage pipeline
- `pegaflow-core/src/backing/`: SSD and RDMA backing implementations
- `pegaflow-core/src/internode/`: cross-node coordination
- `pegaflow-server/src/service.rs`: gRPC service
- `pegaflow-server/src/http_server.rs`: HTTP health and metrics
- `pegaflow-metaserver/src/`: metaserver implementation
- `pegaflow-transfer/src/`: transfer engine implementation
- `python/src/lib.rs`: PyO3 bindings
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector
- `python/pegaflow/pegaflow.pyi`: Python type stubs

## Build, Check, Test

### Rust

```bash
cargo build
cargo build --release
cargo test
```

### Python Bindings

```bash
cd python
maturin develop
maturin develop --release
```

### Local Checks

```bash
prek run
```

Notes:
- `prek` is the local check entrypoint.
- `prek run` will fail on `master` or `main` because of the `no-commit-to-branch` hook in `prek.toml`.

### E2E Correctness Test

```bash
pytest python/tests/test_vllm_e2e_correctness.py -v -s
pytest python/tests/test_vllm_e2e_correctness.py -v -s --model /path/to/model --max-model-len 4096
```

pegaflow-server is auto-started via `cargo run -r` and stopped by the test. No manual setup needed.

### Benchmarks and Examples

```bash
uv run python examples/basic_vllm.py
uv run python examples/bench_kv_cache.py --model /path/to/model --num-prompts 10
```

## Run Services

### Server

```bash
cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --pool-size 30gb
```

### MetaServer

```bash
cargo run -r --bin pegaflow-metaserver
```
## Code Style

### General

- Use English in comments
- Use `.venv` for the Python virtual environment
- Keep changes scoped and aligned with the existing module structure
- Code should be self-documenting. If a comment seems necessary, first try refactoring so the code explains itself.

### Rust

- Prefer minimal visibility: `fn` > `pub(crate)` > `pub`
- Prefer explicit errors over `unwrap` and `expect`
- Keep `use` ordering consistent: std, external crates, local crate
- Prefer `NonNull` over raw pointers where practical in unsafe code

### Python

- Target Python 3.10+
- Use native generics such as `list` and `dict`
- Use PEP 604 unions such as `X | None`
- Use `%s` formatting in logging calls
- Keep imports grouped: standard library, third-party, local

### PyO3

- Keep Python-facing APIs thin and delegate core logic to Rust crates
- Convert Rust errors to `PyErr` cleanly
- When modifying `python/src/lib.rs`, update `python/pegaflow/pegaflow.pyi`

## Testing Principles

- Do not add tests just for the sake of adding tests
- Prefer integration tests over unit tests. Add unit tests only when integration tests cannot cover the behavior effectively.

## Git Workflow

- Do not commit directly to `master`
- Create a `feat/`, `fix/`, `chore/`, `refactor/`, `style/`, or `ci/` branch first
- We use Commitizen commit message format
- Use `cz c` when creating commits interactively
