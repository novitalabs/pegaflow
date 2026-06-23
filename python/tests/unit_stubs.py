"""Lightweight import stubs for connector unit tests.

The default Python test gate should exercise scheduler/worker helper contracts
without requiring a full vLLM + torch runtime. These stubs are intentionally
small and are only installed by tests that mock the connector boundary.
"""

from __future__ import annotations

import logging
import pickle
import re
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from importlib import util as importlib_util
from unittest.mock import MagicMock


def install_connector_unit_stubs() -> None:
    """Install minimal torch/vLLM/native-extension modules for unit tests."""

    _install_torch_stub()
    _install_regex_stub()
    _install_zmq_stub()
    _install_vllm_stubs()
    _install_msgspec_stub()
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


def _install_regex_stub() -> None:
    if _module_available("regex"):
        return

    regex = types.ModuleType("regex")
    regex.IGNORECASE = re.IGNORECASE
    regex.compile = re.compile
    sys.modules["regex"] = regex


def _install_zmq_stub() -> None:
    if _module_available("zmq"):
        return

    zmq = types.ModuleType("zmq")

    class Again(Exception):
        pass

    class Context:
        def destroy(self, *, linger: int = 0) -> None:
            return None

    zmq.Again = Again
    zmq.Context = Context
    zmq.REQ = 1
    zmq.ROUTER = 2
    zmq.RCVTIMEO = 3
    zmq.SNDTIMEO = 4
    zmq.Socket = MagicMock
    sys.modules["zmq"] = zmq


def _install_vllm_stubs() -> None:
    if _module_available("vllm"):
        return

    vllm = _ensure_module("vllm")
    envs = _ensure_module("vllm.envs")
    envs.VLLM_NIXL_SIDE_CHANNEL_HOST = "127.0.0.1"
    envs.VLLM_NIXL_SIDE_CHANNEL_PORT = 5555
    vllm.envs = envs
    _ensure_module("vllm.distributed")
    _ensure_module("vllm.distributed.kv_transfer")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1")
    utils = _ensure_module("vllm.distributed.kv_transfer.kv_connector.utils")
    utils.BlockIds = tuple
    utils.EngineId = str
    utils.yield_req_data = lambda _scheduler_output: iter(())

    @dataclass
    class EngineTransferInfo:
        remote_tp_size: int
        remote_block_size: int
        remote_block_len: int = 0
        remote_physical_blocks_per_logical: int = 1
        remote_pp_rank: int = 0

    class TransferTopology:
        def __init__(
            self,
            *,
            tp_rank: int,
            tp_size: int,
            block_size: int,
            engine_id: str,
            is_mla: bool,
            total_num_kv_heads: int,
            attn_backends,
            tensor_shape=None,
            is_mamba: bool = False,
        ) -> None:
            self.tp_rank = tp_rank
            self.tp_size = tp_size
            self.block_size = block_size
            self.engine_id = engine_id
            self.is_mla = is_mla
            self.total_num_kv_heads = total_num_kv_heads
            self.attn_backends = attn_backends
            self.tensor_shape = tensor_shape
            self.is_mamba = is_mamba
            self._engines = {}
            self._cross_layers_blocks = False
            self._is_kv_layout_blocks_first = False

        @property
        def cross_layers_blocks(self) -> bool:
            return self._cross_layers_blocks

        @property
        def virtually_split_kv_in_blocks(self) -> bool:
            return False

        @property
        def split_k_and_v(self) -> bool:
            return True

        def handshake_target_ranks(self, remote_tp_size: int):
            return (self.tp_rank * remote_tp_size // self.tp_size,)

        def register_remote_engine(self, remote_engine_id: str, info):
            self._engines[(remote_engine_id, 0)] = info
            return info

        def get_engine_info(self, remote_engine_id: str, remote_pp_rank: int = 0):
            return self._engines[(remote_engine_id, remote_pp_rank)]

        def unregister_remote_engine(self, remote_engine_id: str) -> None:
            for key in list(self._engines):
                if key[0] == remote_engine_id:
                    del self._engines[key]

        def tp_ratio(self, remote_tp_size: int) -> int:
            return self.tp_size // remote_tp_size if self.tp_size >= remote_tp_size else -(remote_tp_size // self.tp_size)

        def block_size_ratio(self, remote_block_size: int) -> int:
            return self.block_size // remote_block_size

        def is_kv_replicated(self, _remote_engine_id: str) -> bool:
            return False

        def describe(self, _remote_engine_id: str) -> str:
            return "stub-transfer-topology"

    utils.EngineTransferInfo = EngineTransferInfo
    utils.TransferTopology = TransferTopology
    utils.get_current_attn_backends = lambda _config: []
    utils.get_current_attn_backend = lambda _config: None
    utils.kv_postprocess_blksize_and_layout_on_receive = lambda *_args, **_kwargs: None
    utils.kv_postprocess_blksize_on_receive = lambda *_args, **_kwargs: None
    utils.kv_postprocess_layout_on_receive = lambda *_args, **_kwargs: None

    base = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorRole(Enum):
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorBase_V1:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

    class KVConnectorMetadata:
        return_none: bool = False

    class KVConnectorHandshakeMetadata:
        pass

    class KVConnectorWorkerMetadata:
        pass

    class SupportsHMA:
        pass

    class CopyBlocksOp:
        pass

    base.KVConnectorRole = KVConnectorRole
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorHandshakeMetadata = KVConnectorHandshakeMetadata
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata
    base.SupportsHMA = SupportsHMA
    base.CopyBlocksOp = CopyBlocksOp

    metrics = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

    @dataclass
    class KVConnectorStats:
        data: dict | None = field(default_factory=dict)

    class KVConnectorPromMetrics:
        def __init__(
            self,
            _vllm_config=None,
            _metric_types=None,
            _labelnames=None,
            per_engine_labelvalues=None,
        ) -> None:
            self.per_engine_labelvalues = per_engine_labelvalues or {0: []}

        def make_per_engine(self, metric):
            return {}

    class PromMetric:
        pass

    metrics.KVConnectorStats = KVConnectorStats
    metrics.KVConnectorPromMetrics = KVConnectorPromMetrics
    metrics.PromMetric = PromMetric
    metrics.PromMetricT = PromMetric

    parallel_state = _ensure_module("vllm.distributed.parallel_state")

    def _not_in_distributed_context():
        raise RuntimeError("unit test stub: not in distributed context")

    parallel_state.get_tensor_model_parallel_rank = _not_in_distributed_context
    parallel_state.get_tensor_model_parallel_world_size = _not_in_distributed_context
    parallel_state.get_decode_context_model_parallel_rank = lambda: 0

    class _PPGroup:
        rank_in_group = 0

    parallel_state.get_pp_group = lambda: _PPGroup()

    _ensure_module("vllm.config").VllmConfig = object
    _ensure_module("vllm.logger").init_logger = lambda _name: logging.getLogger(_name)
    nixl_utils = _ensure_module("vllm.distributed.nixl_utils")
    nixl_utils.NixlWrapper = None
    nixl_utils.nixl_agent_config = None
    platforms = _ensure_module("vllm.platforms")

    class _CurrentPlatform:
        device_type = "cuda"

        @staticmethod
        def set_device(_device_id: int) -> None:
            return None

        @staticmethod
        def get_nixl_memory_type() -> str:
            return "VRAM"

        @staticmethod
        def get_nixl_supported_devices() -> dict:
            return {}

    platforms.current_platform = _CurrentPlatform()
    _ensure_module("vllm.utils")
    _ensure_module("vllm.utils.math_utils").cdiv = lambda a, b: (a + b - 1) // b
    network_utils = _ensure_module("vllm.utils.network_utils")
    network_utils.make_zmq_path = lambda protocol, host, port: f"{protocol}://{host}:{port}"
    network_utils.make_zmq_socket = lambda *, ctx, path, socket_type, bind: MagicMock()
    _ensure_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils"
    ).derive_mamba_conv_split = lambda *_args, **_kwargs: None
    _ensure_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils"
    ).MambaConvSplitInfo = object
    models_utils = _ensure_module("vllm.model_executor.models.utils")

    def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
        int_vals: list[int] = []
        for subname in layer_name.split("."):
            try:
                int_vals.append(int(subname))
            except ValueError:
                continue
        if num_attn_module == 1 or "attn" not in layer_name:
            assert len(int_vals) == 1, f"layer name {layer_name} should only contain one integer"
            return int_vals[0]
        assert len(int_vals) <= 2, f"layer name {layer_name} should contain most two integers"
        return int_vals[0] * num_attn_module + int_vals[1] if len(int_vals) == 2 else int_vals[0]

    models_utils.extract_layer_index = extract_layer_index
    _ensure_module("vllm.v1")
    _ensure_module("vllm.v1.core")
    _ensure_module("vllm.v1.core.sched")
    _ensure_module("vllm.v1.core.sched.output").SchedulerOutput = object
    _ensure_module("vllm.v1.attention")
    _ensure_module("vllm.v1.attention.backends")
    _ensure_module("vllm.v1.attention.backends.utils").get_kv_cache_layout = lambda: "HND"
    kv_iface = _ensure_module("vllm.v1.kv_cache_interface")

    class KVCacheSpec:
        pass

    class AttentionSpec(KVCacheSpec):
        pass

    class FullAttentionSpec(AttentionSpec):
        pass

    class MLAAttentionSpec(AttentionSpec):
        pass

    class MambaSpec(KVCacheSpec):
        pass

    class SlidingWindowSpec(FullAttentionSpec):
        pass

    class UniformTypeKVCacheSpecs:
        def __init__(self, kv_cache_specs=None):
            self.kv_cache_specs = kv_cache_specs or {}

    kv_iface.KVCacheSpec = KVCacheSpec
    kv_iface.AttentionSpec = AttentionSpec
    kv_iface.FullAttentionSpec = FullAttentionSpec
    kv_iface.MLAAttentionSpec = MLAAttentionSpec
    kv_iface.MambaSpec = MambaSpec
    kv_iface.SlidingWindowSpec = SlidingWindowSpec
    kv_iface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    _ensure_module("vllm.v1.worker")
    _ensure_module("vllm.v1.worker.block_table").BlockTable = object
    _ensure_module("vllm.v1.worker.utils").select_common_block_size = (
        lambda block_size, _backends: block_size
    )
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

    class _QueryLoading:
        pass

    class _QueryReady:
        def __init__(self, num_hit_blocks: int = 0, lease: bytes = b"") -> None:
            self.num_hit_blocks = num_hit_blocks
            self.lease = lease

    module.EngineRpcClient = getattr(module, "EngineRpcClient", MagicMock)
    module.PegaFlowError = getattr(module, "PegaFlowError", type("PegaFlowError", (Exception,), {}))
    module.PegaflowInternal = getattr(
        module, "PegaflowInternal", type("PegaflowInternal", (Exception,), {})
    )
    module.PyLoadState = getattr(module, "PyLoadState", _FakeLoadState)
    module.QueryLoading = getattr(module, "QueryLoading", _QueryLoading)
    module.QueryReady = getattr(module, "QueryReady", _QueryReady)
    module.__version__ = getattr(module, "__version__", "test")


def _install_msgspec_stub() -> None:
    if _module_available("msgspec"):
        return

    msgspec = types.ModuleType("msgspec")

    class DecodeError(Exception):
        pass

    class ValidationError(Exception):
        pass

    class _Msgpack:
        @staticmethod
        def encode(value):
            return pickle.dumps(value)

        @staticmethod
        def decode(value, *, type=None):
            decoded = pickle.loads(value)
            if type is not None and isinstance(decoded, dict):
                return type(**decoded)
            return decoded

        class Encoder:
            def encode(self, value):
                return pickle.dumps(value)

        class Decoder:
            def __init__(self, type=None):
                self.type = type

            def decode(self, value):
                decoded = pickle.loads(value)
                if self.type is not None and isinstance(decoded, dict):
                    return self.type(**decoded)
                return decoded

    msgspec.DecodeError = DecodeError
    msgspec.ValidationError = ValidationError
    msgspec.msgpack = _Msgpack
    sys.modules["msgspec"] = msgspec


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
