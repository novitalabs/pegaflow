# PegaFlow Agent Guide

This file provides guidance for agents working in the PegaFlow repository.

## Project Overview

PegaFlow is a high-performance KV cache transfer system for LLM inference, designed for vLLM.

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

### Python Test Gates

```bash
cd python
uv run --extra test pytest
```

CI uses the source-only form below so the default Python gate does not compile the native extension or require torch, vLLM, CUDA, or a running server:

```bash
cd python
uv run --isolated --no-project --with pytest --with numpy --with requests pytest
```

Python heavy gates are explicit and resource-bound:

| Change area | Gate command |
|-------------|--------------|
| Server/native/client/session lifecycle | `cd python && uv run --extra test pytest -m integration` |
| vLLM connector correctness, cache semantics, save/load/hit behavior | `cd python && uv run --extra test pytest -m e2e tests/test_vllm_e2e_correctness.py --model /data/models/Qwen3-4B --max-model-len 4096` |
| Query probe, preemption, pending unpin, scheduler concurrency | `cd python && uv run --extra test pytest -m stress tests/test_vllm_query_probe_stress.py --model /data/models/Qwen3-4B --max-model-len 4096` |
| Published wheel/image, loader path, installed console script, CUDA runtime | release smoke from `python/tests/README.md` |

Do not default to running all of `python/tests`. The default gate is the dev-friendly PR feedback loop; heavy gates are targeted evidence for the change area.

Do not add an `xtask` or wrapper just to hide command complexity. Current project taste is `uv` + pytest markers for Python and Cargo/CI for Rust. Consider `xtask` only after the command contract is stable and repeated manual execution has become the actual bottleneck.

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

- Do not add tests just for the sake of adding tests.
- Before adding or keeping a test, answer: would skipping this test materially reduce confidence to merge a PR in its trigger area? If not, delete it or keep it out of routine gates.
- Merge tests that protect the same contract; prefer table-driven cases with clear ids over copy-pasted methods.
- Delete tests with no clear consumer. A test that no dev, reviewer, CI job, release gate, or scheduled job uses is noise.
- Default Python tests must be dev friendly: fast, stable, no GPU, no vLLM, no native extension build, and failure messages that point to a Python contract.
- Heavy tests must declare their trigger: integration, e2e, stress, or release smoke. A heavy test without a trigger should not live in the main pytest surface.
- Stub and mock tests are allowed only for local contracts and must not shadow real runtime modules during integration/e2e collection.
- Prefer integration tests when they prove a real boundary that units cannot. Prefer units when they give faster, clearer feedback for local connector state machines.

## Git Workflow

- Do not commit directly to `master`
- Create a `feat/`, `fix/`, `chore/`, `refactor/`, `style/`, or `ci/` branch first
- We use Commitizen commit message format
- Use `cz c` when creating commits interactively
