"""PegaFlow - High-performance key-value storage engine with Python bindings.

This package provides:
1. EngineRpcClient: gRPC client for remote PegaFlow server communication
2. PegaKVConnector: vLLM KV connector for distributed inference
"""

try:
    from .pegaflow import (
        EngineRpcClient,
        PegaFlowBusinessError,
        PegaFlowError,
        PegaFlowServiceError,
        PyLoadState,
    )
    from .pegaflow import __version__ as _rust_version
except ImportError:
    raise ImportError(
        "pegaflow rust extension is not available, check pegaflow-xxx.so file exists"
    ) from None

__version__ = _rust_version

__all__ = [
    "__version__",
    "EngineRpcClient",
    "PegaFlowBusinessError",
    "PegaFlowError",
    "PegaFlowServiceError",
    "PyLoadState",
]
