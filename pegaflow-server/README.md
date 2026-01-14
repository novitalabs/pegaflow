# PegaFlow Engine gRPC Server

This crate wraps the Rust `PegaEngine` and exposes the same functionality as `python/pegaflow/engine_server.py`, but over a tonic gRPC service.

## Building

The binary embeds CPython via PyO3 so it can reconstruct CUDA IPC tensors with Torch, just like the Python server. Before running cargo commands, point PyO3 to the exact interpreter you want (usually the repo's `.venv`) so linking works and the runtime can import `pegaflow.ipc_wrapper`:

```bash
# Explicitly set your Python interpreter path if needed:
export PYO3_PYTHON="$(pwd)/.venv/bin/python"

export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

cargo run -r --bin pegaflow-server -- --addr 0.0.0.0:50055 --device 0 --pool-size 30gb
```

Adjust the Python path if your venv uses a different minor version.

## Flags

- `--addr`: Bind address for the tonic server (`127.0.0.1:50055` by default).
- `--device`: Default CUDA device id. This matches the Python server's behavior
  and ensures Torch/CUDA are initialized on the correct GPU.
