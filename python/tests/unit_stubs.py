"""Lightweight import stubs for connector unit tests.

The default Python test gate should exercise scheduler/worker helper contracts
without requiring a full vLLM + torch runtime. These stubs are intentionally
small and are only installed by tests that mock the connector boundary.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from importlib import util as importlib_util
from unittest.mock import MagicMock


def install_connector_unit_stubs() -> None:
    """Install minimal torch/vLLM/native-extension modules for unit tests."""

    _install_torch_stub()
    _install_vllm_stubs()
    _install_native_extension_stub()


def _install_torch_stub() -> None:
    if _module_available("torch"):
        return

    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def get_device_properties(_device_index):
            raise RuntimeError("torch stub has no CUDA devices")

        @staticmethod
        def empty_cache() -> None:
            return None

    class _Random:
        @staticmethod
        def manual_seed(_seed: int) -> None:
            return None

    torch.cuda = _Cuda()  # type: ignore[attr-defined]
    torch.random = _Random()  # type: ignore[attr-defined]
    torch.Tensor = object  # type: ignore[attr-defined]
    torch.dtype = object  # type: ignore[attr-defined]
    torch.device = lambda value: value  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    sys.modules["torch"] = torch


def _install_vllm_stubs() -> None:
    if _module_available("vllm"):
        return

    _ensure_module("vllm")
    _ensure_module("vllm.distributed")
    _ensure_module("vllm.distributed.kv_transfer")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1")

    base = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorRole(Enum):
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorBase_V1:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

    class KVConnectorMetadata:
        return_none: bool = False

    base.KVConnectorRole = KVConnectorRole
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorMetadata = KVConnectorMetadata

    metrics = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

    @dataclass
    class KVConnectorStats:
        data: dict | None = field(default_factory=dict)

    class KVConnectorPromMetrics:
        per_engine_labelvalues: dict = {}

        def make_per_engine(self, metric):
            return {}

    class PromMetric:
        pass

    metrics.KVConnectorStats = KVConnectorStats
    metrics.KVConnectorPromMetrics = KVConnectorPromMetrics
    metrics.PromMetric = PromMetric
    metrics.PromMetricT = PromMetric

    parallel_state = _ensure_module("vllm.distributed.parallel_state")
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_decode_context_model_parallel_rank = lambda: 0

    class _PPGroup:
        rank_in_group = 0

    parallel_state.get_pp_group = lambda: _PPGroup()

    _ensure_module("vllm.config").VllmConfig = object
    _ensure_module("vllm.v1")
    _ensure_module("vllm.v1.metrics")
    _ensure_module("vllm.v1.metrics.utils").create_metric_per_engine = lambda *_args, **_kwargs: {}


def _install_native_extension_stub() -> None:
    module_name = "pegaflow.pegaflow"
    if _module_available(module_name):
        return

    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    class _FakeLoadState:
        def shm_name(self) -> str:
            return "test-shm"

        def is_ready(self) -> bool:
            return False

        def get_state(self) -> int:
            return 0

    module.EngineRpcClient = getattr(module, "EngineRpcClient", MagicMock)
    module.PegaFlowBusinessError = getattr(
        module, "PegaFlowBusinessError", type("PegaFlowBusinessError", (Exception,), {})
    )
    module.PegaFlowError = getattr(module, "PegaFlowError", type("PegaFlowError", (Exception,), {}))
    module.PegaFlowServiceError = getattr(
        module, "PegaFlowServiceError", type("PegaFlowServiceError", (Exception,), {})
    )
    module.PyLoadState = getattr(module, "PyLoadState", _FakeLoadState)
    module.__version__ = getattr(module, "__version__", "test")


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib_util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False
