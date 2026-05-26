"""PegaFlow - High-performance key-value storage engine with Python bindings.

This package provides:
1. EngineRpcClient: gRPC client for remote PegaFlow server communication
2. PegaKVConnector: vLLM KV connector for distributed inference
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

_NATIVE_EXPORTS = {
    "EngineRpcClient",
    "PegaFlowError",
    "PegaflowInternal",
    "PyLoadState",
    "QueryLoading",
    "QueryReady",
}

try:
    from . import pegaflow as _native
except ImportError:
    _native = None

try:
    __version__ = _native.__version__ if _native is not None else version("pegaflow-llm")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(name)
    if _native is None:
        raise ImportError(
            "pegaflow rust extension is not available, check pegaflow-xxx.so file exists"
        ) from None
    return getattr(_native, name)


__all__ = [
    "__version__",
    "EngineRpcClient",
    "PegaFlowError",
    "PegaflowInternal",
    "PyLoadState",
    "QueryLoading",
    "QueryReady",
]
