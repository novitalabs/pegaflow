"""PegaFlow - High-performance key-value storage engine with Python bindings.

This package provides:
1. PegaEngine: Rust-based high-performance KV storage (via PyO3)
2. PegaKVConnector: vLLM KV connector for distributed inference
"""

# Import Rust-based PegaEngine from the compiled extension
try:
    from .pegaflow import PegaEngine
except ImportError:
    # Fallback for development when the Rust extension is not built
    PegaEngine = None

# Import Python-based vLLM connector
from .connector import PegaKVConnector, KVConnectorRole

__version__ = "0.0.1"
__all__ = ["PegaEngine", "PegaKVConnector", "KVConnectorRole"]

