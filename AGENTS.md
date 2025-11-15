# PegaFlow Agents Documentation

## Project Overview

PegaFlow is a Rust-based key-value storage engine with Python bindings. The project provides:
1. **PegaEngine**: High-performance KV storage implemented in Rust with Python bindings via PyO3
2. **PegaKVConnector**: vLLM KV connector for distributed inference with KV cache transfer

## Project Structure

```
pegaflow/
├── Cargo.toml              # Workspace configuration
├── AGENTS.md               # This file - project documentation
├── examples/               # Python usage examples
│   ├── basic_usage.py      # PegaEngine basic operations
│   └── basic_vllm.py       # vLLM integration example
├── pega-core/              # Core Rust implementation
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs          # PegaEngine implementation
└── python/                 # Python package (published as "pegaflow")
    ├── Cargo.toml          # Rust crate config for PyO3
    ├── pyproject.toml      # Python package config
    ├── README.md           # Python package documentation
    ├── src/
    │   └── lib.rs          # PyO3 bindings for PegaEngine
    └── pegaflow/           # Python source code
        ├── __init__.py     # Package entry point
        └── connector.py    # vLLM KV connector implementation
```

## Architecture

### pega-core

The core Rust library that implements the storage engine:

- **PegaEngine**: Main struct that wraps a HashMap for key-value storage
- **Methods**:
  - `new()`: Create a new engine instance
  - `get(key)`: Retrieve a value by key
  - `put(key, value)`: Insert or update a key-value pair
  - `remove(key)`: Delete a key-value pair

### python/

The Python package layer (published as `pegaflow` on PyPI):

#### Rust Extension (src/lib.rs)
- Uses PyO3 to expose `PegaEngine` as a Python class
- Wraps all core methods for Python consumption
- Built as `pegaflow.pegaflow` extension module using maturin

#### Python Package (pegaflow/)
- **`__init__.py`**: Package entry point, exports `PegaEngine` and `PegaKVConnector`
- **`connector.py`**: vLLM v1 KV connector implementation
  - `PegaKVConnector`: Subclass of `KVConnectorBase_V1`
  - Currently a skeleton implementation (methods raise `NotImplementedError`)
  - Used by vLLM for distributed inference with KV cache transfer

## Development Workflow

### Building the Project

1. **Build Rust crates**:
   ```bash
   cargo build
   ```

2. **Build Python bindings**:
   ```bash
   cd python
   maturin develop
   ```

3. **Run Python examples**:
   ```bash
   python examples/basic_usage.py
   python examples/basic_vllm.py
   ```

### Testing

The `examples/basic_usage.py` script demonstrates basic usage:
- Create a PegaEngine instance
- Put key-value pairs
- Get values by key
- Remove keys
- Verify operations

The `examples/basic_vllm.py` script demonstrates vLLM integration:
- Configure vLLM to use PegaKVConnector
- Load a model (GPT-2) with KV transfer enabled
- Generate text (currently fails at connector methods - expected for skeleton implementation)

## Agent Workflow

When working on this project, agents should:

1. **Understand the separation of concerns**:
   - `pega-core`: Pure Rust logic, no Python dependencies
   - `python/src/`: Python bindings only, delegates to pega-core
   - `python/pegaflow/`: Pure Python utilities (vLLM connector, etc.)

2. **Make changes in the correct layer**:
   - Business logic changes → `pega-core`
   - Python API changes → `python/src/lib.rs`
   - vLLM connector changes → `python/pegaflow/connector.py`

3. **Test changes**:
   - Run `cargo test` for Rust tests
   - Run `cd python && maturin develop` to rebuild Python bindings
   - Run `python examples/basic_usage.py` to verify PegaEngine
   - Run `python examples/basic_vllm.py` to verify vLLM integration

4. **Follow Rust best practices**:
   - Use proper error handling
   - Follow ownership and borrowing rules
   - Write idiomatic Rust code

5. **Follow PyO3 patterns**:
   - Use `#[pyclass]` for Python-exposed structs
   - Use `#[pymethods]` for Python-exposed methods
   - Handle Python exceptions properly

## Dependencies

### Rust Dependencies
- `pyo3`: Python bindings framework (python/ only)

### Python Dependencies
- `maturin`: Build tool for PyO3 projects
- `vllm`: Large language model inference framework (for connector integration)

### Build Tools
- `uv`: Fast Python package installer and environment manager

## Package Naming

The Python package is published as **`pegaflow`** on PyPI:
- Package name: `pegaflow` (all lowercase, no hyphens)
- Import name: `from pegaflow import PegaEngine, PegaKVConnector`
- Version: `0.0.1` (initial development release)

PyPI allows lowercase package names, and this follows Python naming conventions (PEP 8).

## Future Enhancements

Potential areas for expansion:
- Implement PegaKVConnector methods for actual KV cache transfer
- Persistence (save/load from disk)
- Advanced data structures (sorted sets, lists)
- Async operations
- Transaction support
- Compression
- Replication

