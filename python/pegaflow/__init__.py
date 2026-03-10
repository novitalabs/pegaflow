"""PegaFlow - High-performance key-value storage engine with Python bindings.

This package provides:
1. PegaEngine: Rust-based high-performance KV storage (via PyO3)
2. PegaKVConnector: vLLM KV connector for distributed inference
"""

# Import Rust-based PegaEngine from the compiled extension
try:
    from .pegaflow import (
        EngineRpcClient,
        PegaEngine,
        PegaFlowBusinessError,
        PegaFlowError,
        PegaFlowServiceError,
        PyLoadState,
        TransferEngine,
    )
    from .pegaflow import (
        __version__ as _rust_version,
    )
except ImportError:
    # Fallback for development when the Rust extension is not built
    EngineRpcClient = None
    PegaEngine = None
    PyLoadState = None
    TransferEngine = None
    raise ImportError(
        "pegaflow rust extension is not available, check pegaflow-xxx.so file exists"
    ) from None

__version__ = _rust_version

__all__ = [
    "__version__",
    "EngineRpcClient",
    "PegaEngine",
    "PegaFlowBusinessError",
    "PegaFlowError",
    "PegaFlowServiceError",
    "PyLoadState",
    "TransferEngine",
]
