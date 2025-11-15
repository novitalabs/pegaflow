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

## License

MIT

