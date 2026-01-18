# PegaFlow Python Package

High-performance key-value storage engine with Python bindings, built with Rust and PyO3.

## Features

- **PegaEngine**: Fast Rust-based key-value storage with Python bindings
- **PegaKVConnector**: vLLM KV connector for distributed inference with KV cache transfer

## Installation

### From Source

```bash
# Install maturin if you haven't already
pip install maturin

# Build and install in development mode
cd python
maturin develop

# Or build a wheel
maturin build --release
```

### From PyPI (coming soon)

```bash
pip install pegaflow
```

## Usage

### Basic KV Storage

```python
from pegaflow import PegaEngine

# Create a new engine
engine = PegaEngine()

# Store key-value pairs
engine.put("name", "PegaFlow")
engine.put("version", "0.1.0")

# Retrieve values
name = engine.get("name")  # Returns "PegaFlow"
missing = engine.get("nonexistent")  # Returns None

# Remove keys
removed = engine.remove("name")  # Returns "PegaFlow"
```

#### Sglang Examples:

example 1:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B --trust-remote-code --enable-cache-report --page-size 256 --host "0.0.0.0" --port 8000 --mem-fraction-static 0.8 --max-running-requests 32 --enable-pegaflow
```

example 2:

```bash
python3 -m sglang.launch_server --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 64}"  --model-path deepseek-ai/DeepSeek-V3.2 --served-model-name deepseek-ai/DeepSeek-V3.2 --trust-remote-code --page-size "64" --reasoning-parser deepseek-v3 --tool-call-parser deepseekv32 --enable-cache-report --host "0.0.0.0" --port 8031 --mem-fraction-static 0.83 --max-running-requests 64 --tp-size "8" --enable-pegaflow
```

### vLLM KV Connector

```python
from vllm import LLM
from vllm.distributed.kv_transfer.kv_transfer_agent import KVTransferConfig

# Configure vLLM to use PegaKVConnector
kv_transfer_config = KVTransferConfig(
    kv_connector="PegaKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="pegaflow.connector",
)

# Create LLM with KV transfer enabled
llm = LLM(
    model="gpt2",
    kv_transfer_config=kv_transfer_config,
)
```

## Development

See the [examples](../examples/) directory for more usage examples.

## Testing

### Running Unit Tests

The test suite includes integration tests that verify the `EngineRpcClient` can correctly communicate with a running `pegaflow-server` instance.

#### Prerequisites

1. **Build the Rust extension**:

   ```bash
   cd python
   maturin develop --release
   ```

2. **Build the server binary**:

   ```bash
   cd ..
   cargo build --release --bin pegaflow-server
   ```

3. **Ensure CUDA is available** (tests require GPU):
   ```bash
   python -c "import torch; assert torch.cuda.is_available()"
   ```

#### Running Tests

```bash
cd python

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_engine_client.py -v

# Run with coverage
pytest tests/ --cov=pegaflow --cov-report=html
```

#### Test Structure

- **`tests/conftest.py`**: Contains pytest fixtures for:

  - `pega_server`: Automatically starts/stops `pegaflow-server` for integration tests
  - `engine_client`: Creates an `EngineRpcClient` connected to the test server
  - `client_context`: Provides a `ClientContext` representing a vLLM instance with GPU KV cache tensors
  - `registered_instance`: Provides a registered instance ID for query tests

- **`tests/test_engine_client.py`**: Integration tests for:
  - Server connectivity
  - Query operations with various inputs

#### Test Fixtures

The `ClientContext` class abstracts a vLLM instance and provides:

- `register_kv_caches()`: Register GPU KV cache tensors with the server
- `query(block_hashes)`: Query available blocks
- `unregister_context()`: Unregister context from server

Example test usage:

```python
def test_query(client_context):
    """Test query operation."""
    result = client_context.query([])
    assert result is not None
```

## License

MIT
