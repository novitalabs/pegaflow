from __future__ import annotations

# ruff: noqa: E402,F401
import queue
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

native = types.ModuleType("pegaflow.pegaflow")
native.EngineRpcClient = MagicMock
native.PegaFlowError = type("PegaFlowError", (Exception,), {})
native.PegaflowInternal = type("PegaflowInternal", (Exception,), {})
native.PyLoadState = object
native.QueryLoading = object
native.QueryReady = object
native.__version__ = "test"
sys.modules["pegaflow.pegaflow"] = native

import pegaflow.pd_connector.prefill_worker as prefill_worker_mod  # noqa: E402
import pegaflow.pd_connector.prefill as prefill_mod  # noqa: E402
import pegaflow.pd_connector.decode_worker as decode_worker_mod  # noqa: E402
import pegaflow.pd_connector.worker as worker_mod  # noqa: E402
from pegaflow.pd_connector import PdConnector  # noqa: E402
from pegaflow.pd_connector.kv_params import parse_consumer  # noqa: E402
from pegaflow.pd_connector.layout import (  # noqa: E402
    BlockRegionSlice,
    FlashAttnHndLayout,
    LayerBlockSlices,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (  # noqa: E402
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdHandshake,
    PdWorkerMetadata,
    RELEASE_CONSUMER_ABORT,
    RELEASE_PRODUCER_ABORT,
    RELEASE_PRODUCER_FINISHED,
    RELEASE_PRODUCER_PREEMPTED,
    PushReqMeta,
    TransferRegionLayout,
    WaitReqMeta,
    handshake_from_dict,
    handshake_to_compact_dict,
    handshake_to_dict,
)
from pegaflow.pd_connector.prefill import (  # noqa: E402
    AsyncPrefillSender,
    PrefillHttpTask,
)
from pegaflow.pd_connector.proxy import (  # noqa: E402
    ProxyConfig,
    build_pd_proxy_request,
    iter_http_stream_chunks,
)
from pegaflow.pd_connector.rdma import (  # noqa: E402
    MockRdmaPort,
    RealRdmaPort,
    _layer_blocks_to_native,
)
from pegaflow.pd_connector.scheduler import PdSchedulerConnector  # noqa: E402
from pegaflow.pd_connector.worker import PdWorkerConnector  # noqa: E402


def teardown_module() -> None:
    sys.modules.pop("pegaflow.pegaflow", None)


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        ptr: int = 0x1000,
        element_size: int = 2,
        device_index: int | None = None,
    ) -> None:
        self.shape = shape
        self._stride = stride
        self._ptr = ptr
        self._element_size = element_size
        self.device = SimpleNamespace(index=device_index) if device_index is not None else None

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def data_ptr(self) -> int:
        return self._ptr

    def element_size(self) -> int:
        return self._element_size


class FakeSlotMapping(list[int]):
    def __init__(self, values: list[int]) -> None:
        super().__init__(values)
        self.cpu_calls = 0

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def tolist(self):
        return list(self)


class FakePrefillSender:
    def __init__(self) -> None:
        self.tasks = []
        self.cancelled = []

    def submit(self, task) -> None:
        self.tasks.append(task)

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class FakeNativeRdmaEngine:
    def __init__(self) -> None:
        self.local_layers = []
        self.remote_regs = []
        self.pushed_layers = []
        self.done_reqs = []
        self.waited_reqs = []
        self.polled_reqs = []
        self.marked_reqs = []
        self.finished_sending = ["sent-1"]
        self.finished_recving = ["recv-1"]

    def register_local_layers(self, layers):
        self.local_layers.append(layers)
        registered = []
        for layer in layers:
            assert "regions" in layer
            assert "k_block_addrs" not in layer
            assert "v_block_addrs" not in layer
            ptr = min(region["base_addr"] for region in layer["regions"])
            registered.append(
                {
                    **layer,
                    "mr_desc": {
                        "ptr": ptr,
                        "addr_rkey_list": [["10.0.0.1:1", 17]],
                    },
                }
            )
        return registered

    def register_remote(self, req_id, handshake):
        assert handshake is not None
        assert handshake["request_id"]
        assert handshake.get("imm_id") is None or isinstance(handshake["imm_id"], int)
        assert handshake.get("fail_imm_id") is None or isinstance(handshake["fail_imm_id"], int)
        assert handshake.get("abort_imm_id") is None or isinstance(handshake["abort_imm_id"], int)
        for layer in handshake["layers"]:
            assert layer["mr_desc"]["addr_rkey_list"]
            assert isinstance(layer["mr_desc"]["addr_rkey_list"][0], tuple)
            assert layer["block_ids"]
            assert layer["regions"]
            assert [region["region_idx"] for region in layer["regions"]] == list(
                range(len(layer["regions"]))
            )
            assert "k_block_addrs" not in layer
            assert "v_block_addrs" not in layer
        self.remote_regs.append((req_id, handshake))

    def push_layer(self, req_id, layer_idx, blocks):
        for block in blocks:
            assert set(block) == {"regions"}
            assert [region["region_idx"] for region in block["regions"]] == list(
                range(len(block["regions"]))
            )
        self.pushed_layers.append((req_id, layer_idx, blocks))

    def push_done(self, req_id):
        self.done_reqs.append(req_id)

    def fail_request(self, req_id):
        return None

    def abort_request(self, req_id):
        self.finished_recving.append(req_id)

    def wait_done(self, req_id):
        self.waited_reqs.append(req_id)

    def poll_done(self, req_id):
        self.polled_reqs.append(req_id)
        return req_id in self.finished_recving

    def mark_done(self, req_id):
        self.marked_reqs.append(req_id)

    def pop_finished_sending(self):
        finished = self.finished_sending
        self.finished_sending = []
        return finished

    def pop_finished_recving(self):
        finished = self.finished_recving
        self.finished_recving = []
        return finished


class FakeNativeRdmaEngineCtor(FakeNativeRdmaEngine):
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).last_kwargs = kwargs

    def num_domains(self):
        return 1

    def num_groups(self):
        return 1

    def aggregated_link_speed(self):
        return 400_000_000_000


def drain_pd_pushes(worker: PdWorkerConnector) -> None:
    worker._push_sender.wait_all()
    worker._push_finalizer.wait_all()


def pushed_layers_by_idx(
    rdma: MockRdmaPort,
    req_id: str,
) -> dict[int, list[LayerBlockSlices]]:
    return dict(rdma.pushed_layers[req_id])


DUMMY_HANDSHAKE = PdHandshake(
    request_id="",
    engine_id="",
    tp_rank=0,
    tp_size=1,
    block_size=16,
    layers=(),
)


def hnd_remote_layer(
    *,
    layer_name: str = "layer.0",
    layer_idx: int = 0,
    block_ids: tuple[int, ...] = (0,),
    k_base: int = 0x1000,
    v_base: int = 0x8000,
    block_len: int = 0x400,
) -> LayerRemoteLayout:
    return LayerRemoteLayout(
        layer_name=layer_name,
        layer_idx=layer_idx,
        block_ids=block_ids,
        regions=(
            TransferRegionLayout(region_idx=0, base_addr=k_base, block_len=block_len),
            TransferRegionLayout(region_idx=1, base_addr=v_base, block_len=block_len),
        ),
    )


def decode_handshakes(tp_size: int, *, block_size: int = 16) -> tuple[PdHandshake, ...]:
    return tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=tp_size,
            block_size=block_size,
            layers=(),
        )
        for rank in range(tp_size)
    )


def fake_mla_config(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    block_size: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="pd"),
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=tp_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def fake_mtp_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="pd"),
        model_config=SimpleNamespace(
            use_mla=False,
            hf_text_config=SimpleNamespace(num_nextn_predict_layers=1),
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def fake_kv_cache_config(
    *,
    num_blocks: int,
    specs: dict[str, SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=tuple(specs),
                kv_cache_spec=SimpleNamespace(kv_cache_specs=specs),
            )
        ],
    )


__all__ = [name for name in globals() if not name.startswith("__") and name != "teardown_module"]
