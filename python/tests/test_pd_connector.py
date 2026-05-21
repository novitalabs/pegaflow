from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

native = types.ModuleType("pegaflow.pegaflow")
native.EngineRpcClient = MagicMock
native.PegaFlowError = type("PegaFlowError", (Exception,), {})
native.PegaflowInternal = type("PegaflowInternal", (Exception,), {})
native.PyLoadState = object
native.QueryLoading = object
native.QueryReady = object
native.__version__ = "test"
sys.modules["pegaflow.pegaflow"] = native

from pegaflow.pd_connector.layout import (  # noqa: E402
    FlashAttnHndLayout,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (  # noqa: E402
    PdConnectorMetadata,
    PushReqMeta,
    RemoteEndpoint,
    WaitReqMeta,
)
from pegaflow.pd_connector.oob import InMemoryOobPort  # noqa: E402
from pegaflow.pd_connector.proxy import (  # noqa: E402
    ProxyConfig,
    build_pd_proxy_request,
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
    ) -> None:
        self.shape = shape
        self._stride = stride
        self._ptr = ptr
        self._element_size = element_size

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def data_ptr(self) -> int:
        return self._ptr

    def element_size(self) -> int:
        return self._element_size


class FakeSlotMapping(list[int]):
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def test_flash_attn_hnd_layout_offsets() -> None:
    # Logical shape [2, num_blocks, block_size, num_kv_heads, head_size].
    # HND physical order [2, num_blocks, num_kv_heads, block_size, head_size].
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )

    layout = FlashAttnHndLayout.from_tensor("layer.0", tensor)

    assert layout.block_bytes == 16 * 4 * 32 * 2
    assert layout.block_offset_bytes(0, 3) == 3 * 4 * 16 * 32 * 2
    assert layout.block_offset_bytes(1, 3) == (8 * 4 * 16 * 32 + 3 * 4 * 16 * 32) * 2


def test_pd_worker_pushes_flash_attn_hnd_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(SimpleNamespace())
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target=RemoteEndpoint(engine_id="decode"),
                target_request_id="req-1",
                num_prompt_tokens=32,
            )
        }
    )

    worker.start_load_kv(meta, None)
    attn_metadata = SimpleNamespace(slot_mapping=FakeSlotMapping([16, 17, 32, -1]))
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert unique_blocks_from_slot_mapping(attn_metadata.slot_mapping, 16) == {1, 2}
    assert worker.rdma.pushed_layers["req-1"][0][0] == 0
    assert worker.rdma.pushed_layers["req-1"][1][0] == 1
    finished_sending, finished_recving = worker.get_finished(set())
    assert finished_sending == {"req-1"}
    assert finished_recving is None


def test_pd_worker_publishes_wait_handshake_and_delays_done_until_all_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    oob = InMemoryOobPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        oob=oob,
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})

    wait_meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1, 2],),
                remote=RemoteEndpoint(engine_id="prefill"),
                remote_request_id="req-1",
                num_prompt_tokens=32,
                prompt_token_ids=(101, 102, 103),
            )
        }
    )
    worker.start_load_kv(wait_meta, None)

    handshake = oob.get_handshake("req-1")
    assert handshake is not None
    assert handshake.engine_id == "decode"
    assert handshake.block_size == 16
    assert handshake.layers[0].k_block_addrs == (
        tensor.data_ptr() + 1 * 4 * 16 * 32 * 2,
        tensor.data_ptr() + 2 * 4 * 16 * 32 * 2,
    )
    prefill_request = oob.get_prefill_request("req-1")
    assert prefill_request is not None
    assert prefill_request.prompt_token_ids == (101, 102, 103)
    assert prefill_request.producer_kv_transfer_params == {
        "do_remote_prefill_sender": True,
        "target_engine_id": "decode",
        "target_request_id": "req-1",
    }

    push_worker = PdWorkerConnector(SimpleNamespace(), oob=oob)
    push_worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    push_meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target=RemoteEndpoint(engine_id="decode"),
                target_request_id="req-1",
                num_prompt_tokens=32,
            )
        }
    )
    push_worker.start_load_kv(push_meta, None)
    assert push_worker.rdma.remote_handshakes["req-1"] is handshake

    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    assert push_worker.get_finished(set()) == (None, None)

    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    assert push_worker.get_finished(set()) == ({"req-1"}, None)


def test_scheduler_delays_producer_block_free_until_send_finishes() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        num_prompt_tokens=32,
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "req-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())
    assert set(meta.reqs_to_push) == {"req-1"}

    delay_free, params = scheduler.request_finished(request, ([1, 2],))
    assert delay_free is True
    assert params is None

    scheduler.update_connector_output(
        SimpleNamespace(finished_sending={"req-1"}, finished_recving=None)
    )
    release_meta = scheduler.build_connector_meta(SimpleNamespace())
    assert release_meta.reqs_to_release == {"req-1"}


def test_scheduler_carries_prompt_tokens_for_d_to_p_oob() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        num_prompt_tokens=3,
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={"do_remote_prefill": True, "remote_engine_id": "prefill"},
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].prompt_token_ids == (11, 12, 13)


def test_scheduler_carries_fake_rdma_done_endpoint() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        num_prompt_tokens=3,
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_engine_id": "prefill",
            "done_endpoint": "tcp://127.0.0.1:7200",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].remote.done_endpoint == "tcp://127.0.0.1:7200"


def test_pd_proxy_injects_p_and_d_transfer_params() -> None:
    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
        done_endpoint="tcp://127.0.0.1:7200",
        timeout_s=30,
        prefill_max_tokens=1,
    )

    req = build_pd_proxy_request(
        {
            "model": "/data/Qwen3-4B",
            "prompt": "hello",
            "max_tokens": 4,
        },
        config,
        request_id="pd-test",
    )

    assert req.prefill_body["request_id"] == "pd-test-p"
    assert req.prefill_body["max_tokens"] == 1
    assert req.prefill_body["kv_transfer_params"] == {
        "do_remote_prefill_sender": True,
        "target_engine_id": "decode",
        "target_request_id": "pd-test-d",
        "done_endpoint": "tcp://127.0.0.1:7200",
    }
    assert req.decode_body["request_id"] == "pd-test-d"
    assert req.decode_body["max_tokens"] == 4
    assert req.decode_body["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "remote_engine_id": "prefill",
        "remote_request_id": "pd-test-p",
        "done_endpoint": "tcp://127.0.0.1:7200",
    }
