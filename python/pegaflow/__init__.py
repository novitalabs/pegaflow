"""PegaFlow Python bindings."""

try:
    from .pegaflow import (
        PegaFlowBusinessError,
        PegaFlowError,
        PegaFlowServiceError,
    )
    from .pegaflow import __version__ as _rust_version
except ImportError:
    raise ImportError(
        "pegaflow rust extension is not available, check pegaflow-xxx.so file exists"
    ) from None

from .client import (
    KvCacheLayer,
    KvCacheRegistration,
    LayerSave,
    LoadHandle,
    LoadItem,
    LoadPlan,
    LoadRequest,
    PegaClient,
    PrepareLoadHandle,
    PrepareLoadRequest,
    PrepareLoadResult,
    SaveRequest,
)

__version__ = _rust_version

__all__ = [
    "__version__",
    "KvCacheLayer",
    "KvCacheRegistration",
    "LayerSave",
    "LoadHandle",
    "LoadItem",
    "LoadPlan",
    "LoadRequest",
    "PegaClient",
    "PegaFlowBusinessError",
    "PegaFlowError",
    "PegaFlowServiceError",
    "PrepareLoadResult",
    "PrepareLoadHandle",
    "PrepareLoadRequest",
    "SaveRequest",
]
