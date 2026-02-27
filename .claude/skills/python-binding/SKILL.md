---
name: python-binding
description: >
  Use when developing under python/ directory. Covers PyO3 bindings, maturin build,
  type stub sync, connector and SGLang code conventions.
---

# Python Development

## Build

```bash
cd python
maturin develop          # Dev build
maturin develop --release  # Release build
```

**Important:** When modifying `python/src/lib.rs` (PyO3 bindings), update the type stub file `python/pegaflow/pegaflow.pyi` to keep type hints in sync.

## Key Files

- `python/src/lib.rs`: PyO3 bindings exposing `PegaEngine` and gRPC client
- `python/pegaflow/pegaflow.pyi`: Type stubs — must stay in sync with `lib.rs`
- `python/pegaflow/connector/scheduler.py`: vLLM scheduler-side connector
- `python/pegaflow/connector/worker.py`: vLLM worker-side connector
- `python/pegaflow/sglang/pegaflow_radix_cache.py`: SGLang radix cache
- `python/pegaflow/ipc_wrapper.py`: CUDA IPC handle wrapper

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

Connector is split into scheduler-side (`scheduler.py`) and worker-side (`worker.py`).

## SGLang Integration

Drop-in replacement for SGLang's `RadixCache` using PegaEngine for distributed KV cache.

```python
from pegaflow.sglang.peagflow_radix_cache import PeagflowRadixCache

kv_cache = PeagflowRadixCache(
    params=cache_params,       # CacheInitParams
    model_config=model_config, # ModelConfig
    tp_size=tp_size,
    rank=tp_rank,
)
```

Behavioral differences vs default RadixCache:
- On prefix miss, queries PegaEngine for remote blocks and loads into local GPU buffers
- On request finish, saves changed blocks back to engine
- All KV operations batched per block and per layer
- Auto-registers CUDA IPC handles on construction, auto-unregisters on shutdown
