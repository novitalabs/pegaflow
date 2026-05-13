# Python Test Gates

`python/tests` is organized around developer workflow, not around how much code has accumulated. A test belongs here only when skipping it would materially reduce confidence to merge a PR for its trigger area. The default pytest invocation is intentionally small: it runs unit/helper contracts and excludes tests that start `pegaflow-server`, require CUDA, run vLLM, or create pressure workloads.

## What To Run

| Change area | Gate | Command | Failure boundary |
| --- | --- | --- | --- |
| Connector helper math, scheduler state, worker load failure handling, IPC wrapper compatibility | Default unit | `uv run --extra test pytest` | Python contract or local connector state-machine regression |
| Clean source-only Python changes, docs touching test layout, CI test dependency changes | Source-only default | `uv run --isolated --no-project --with pytest --with numpy --with requests pytest` | Default test accidentally depends on torch, vLLM, CUDA, or native extension |
| Server client, native extension, CUDA IPC registration, session lifecycle | Integration | `uv run --extra test pytest -m integration` | Server/native/GPU lifecycle regression |
| vLLM connector correctness, cache semantics, save/load/hit behavior, release candidate confidence | vLLM correctness E2E | `uv run --extra test pytest -m e2e tests/test_vllm_e2e_correctness.py --model /data/models/Qwen3-4B` | Real vLLM connector correctness regression |
| Query probe, preemption, pending unpin, scheduler concurrency or pressure behavior | Stress | `uv run --extra test pytest -m stress tests/test_vllm_query_probe_stress.py --model /data/models/Qwen3-4B` | Concurrent scheduler/query-probe regression |
| Wheel, loader path, installed console script, target CUDA runtime, published package | Release smoke | See Release Smoke | Packaging, loader, final artifact, or runtime contract regression |

A heavy test without a clear trigger should not be promoted into a routine gate. A generated fuzz workload used to live here, but it had no stable owner, cadence, data contract, or debugging path; it was removed from the main pytest surface instead of pretending to be a regular gate.

## Default Unit Gate

```bash
cd python
uv run --extra test pytest
```

CI uses the source-only variant below so the unit gate does not compile the native extension or require CUDA:

```bash
cd python
uv run --isolated --no-project --with pytest --with numpy --with requests pytest
```

Runs:
- connector arithmetic and scheduler state-machine contracts (`test_combine_hashes.py`)
- connector load fault-tolerance unit tests with fake transport (`test_connector_fault_tolerance.py`)
- CUDA IPC wrapper shape compatibility with fake torch objects (`test_ipc_wrapper.py`)
- import-stub safety checks for default unit tests (`test_unit_stubs.py`)

This gate must collect and run without torch, vLLM, CUDA, external models, or a running PegaFlow server. Stub modules are allowed only inside tests that explicitly mock the connector boundary, and they must not shadow a real runtime during integration or E2E collection.

## Server Integration Gate

```bash
cd python
uv run --extra test pytest -m integration
```

Runs tests that start or require a local `pegaflow-server` but do not run vLLM:
- `test_engine_client.py`
- `test_session_watcher.py`

Requirements:
- built Python extension, for example `uv run maturin develop -r`
- a discoverable `pegaflow-server` binary from the installed package or Cargo target
- CUDA-capable GPU for tests that register real CUDA IPC tensors
- loader paths for the active Python and CUDA runtime, when the environment does not provide them globally

## vLLM Correctness E2E Gate

```bash
cd python
uv run --extra test pytest -m e2e tests/test_vllm_e2e_correctness.py \
  --model /data/models/Qwen3-4B \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 4096
```

This is the main correctness E2E. It starts baseline vLLM and PegaFlow-enabled vLLM with vLLM's neutral generation defaults, compares deterministic completions across one execution plan, and checks that PegaFlow metrics show save/load/hit activity.

Requirements:
- vLLM installed in the active environment
- local model path, not an implicit network download
- GPU runtime compatible with the installed wheel variant
- enough free GPU memory for the model and configured context length

## Stress Gate

```bash
cd python
uv run --extra test pytest -m stress tests/test_vllm_query_probe_stress.py \
  --model /data/models/Qwen3-4B \
  --max-model-len 4096
```

This is a targeted vLLM scenario for scheduler query-probe reuse under pressure. Run it for query-probe, preemption, pending unpin, scheduler concurrency, or cache-lookup memoization changes. It is not a default PR gate.

## Release Smoke

Release smoke validates the final installed package, not the source checkout. It should use a clean non-editable environment and record Python libdir, `PYTHONHOME`, `PYTHONPATH`, CUDA runtime path, package name/version, GPU, model path, and metrics excerpt.

Minimum checks:
- `pegaflow-server --help`
- minimal installed `pegaflow-server` startup and `/health` 200
- vLLM + `PegaKVConnector` with `/v1/models`, one completion, one repeated long prompt, and non-zero save/load/hit/HLL metrics
