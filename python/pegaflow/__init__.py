"""PegaFlow - High-performance key-value storage engine with Python bindings.

This package provides:
1. PegaEngine: Rust-based high-performance KV storage (via PyO3)
2. PegaKVConnector: vLLM KV connector for distributed inference
"""

from importlib.metadata import PackageNotFoundError, version as package_version


def _detect_version() -> str:
    for dist_name in ("pegaflow-llm", "pegaflow"):
        try:
            return package_version(dist_name)
        except PackageNotFoundError:
            continue
    return "unknown"

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
except ImportError:
    # Fallback for development when the Rust extension is not built
    EngineRpcClient = None
    PegaEngine = None
    PyLoadState = None
    TransferEngine = None
    raise ImportError(
        "pegaflow rust extension is not available, check pegaflow-xxx.so file exists"
    ) from None

__version__ = _detect_version()
__all__ = [
    "EngineRpcClient",
    "PegaEngine",
    "PegaFlowBusinessError",
    "PegaFlowError",
    "PegaFlowServiceError",
    "PyLoadState",
    "TransferEngine",
]
