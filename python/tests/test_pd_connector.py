from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdHandshake,
    PushReqMeta,
    RemoteEndpoint,
    WaitReqMeta,
)
from pegaflow.pd_connector.oob import InMemoryOobPort  # noqa: E402
from pegaflow.pd_connector.proxy import (  # noqa: E402
    ProxyConfig,
    build_pd_proxy_request,
)
from pegaflow.pd_connector.rdma import RealRdmaPort  # noqa: E402
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
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakePrefillSender:
    def __init__(self) -> None:
        self.tasks = []

    def submit(self, task) -> None:
        self.tasks.append(task)


class FakeNativeRdmaEngine:
    def __init__(self) -> None:
        self.local_layers = []
        self.remote_regs = []
        self.pushed_layers = []
        self.done_reqs = []
        self.waited_reqs = []
        self.marked_reqs = []
        self.finished_sending = ["sent-1"]
        self.finished_recving = ["recv-1"]

    def register_local_layers(self, layers):
        self.local_layers.append(layers)
        registered = []
        for layer in layers:
            registered.append(
                {
                    **layer,
                    "mr_desc": {
                        "ptr": layer["base_addr"],
                        "addr_rkey_list": [["10.0.0.1:1", 17]],
                    },
                }
            )
        return registered

    def register_remote(self, req_id, handshake):
        assert handshake is not None
        assert handshake["request_id"]
        for layer in handshake["layers"]:
            assert layer["mr_desc"]["addr_rkey_list"]
            assert (
                len(layer["block_ids"])
                == len(layer["k_block_addrs"])
                == len(layer["v_block_addrs"])
            )
        self.remote_regs.append((req_id, handshake))

    def push_layer(self, req_id, layer_idx, blocks):
        for block in blocks:
            assert set(block) == {"k", "v"}
            assert block["k"]["block_id"] == block["v"]["block_id"]
            assert block["k"]["bytes"] == block["v"]["bytes"]
        self.pushed_layers.append((req_id, layer_idx, blocks))

    def push_done(self, req_id):
        self.done_reqs.append(req_id)

    def wait_done(self, req_id):
        self.waited_reqs.append(req_id)

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


def test_real_rdma_port_preserves_native_contract_for_pd_push() -> None:
    native_engine = FakeNativeRdmaEngine()
    rdma = RealRdmaPort(native_engine)
    layer = LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        base_addr=0x1000,
        block_bytes=1024,
        block_ids=(0, 1),
        k_block_addrs=(0x1000, 0x1400),
        v_block_addrs=(0x1800, 0x1C00),
    )

    registered = rdma.register_local_layers((layer,))

    assert registered[0].mr_desc == {
        "ptr": 0x1000,
        "addr_rkey_list": [["10.0.0.1:1", 17]],
    }
    assert registered[0].block_ids == (0, 1)
    assert registered[0].k_block_addrs == (0x1000, 0x1400)
    assert registered[0].v_block_addrs == (0x1800, 0x1C00)

    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        kv_layout="flash_attn_hnd",
        layers=registered,
    )
    rdma.register_remote("req-1", handshake)

    assert native_engine.remote_regs[0][0] == "req-1"
    assert native_engine.remote_regs[0][1]["layers"][0]["mr_desc"]["ptr"] == 0x1000
    assert native_engine.remote_regs[0][1]["layers"][0]["block_ids"] == [0, 1]

    rdma.push_layer(
        "req-1",
        0,
        [
            FlashAttnHndLayout(
                "layer.0", (2, 8, 16, 4, 32), (16384, 2048, 32, 512, 1), 2, 0x1000
            ).block_slices(1)
        ],
    )
    rdma.push_done("req-1")
    rdma.wait_done("req-1")
    rdma.mark_done("req-1")

    _, _, pushed_blocks = native_engine.pushed_layers[0]
    assert pushed_blocks == [
        {
            "k": {"block_id": 1, "src_offset_bytes": 2048 * 2, "bytes": 4096},
            "v": {
                "block_id": 1,
                "src_offset_bytes": (16384 + 2048) * 2,
                "bytes": 4096,
            },
        }
    ]
    assert native_engine.done_reqs == ["req-1"]
    assert native_engine.waited_reqs == ["req-1"]
    assert native_engine.marked_reqs == ["req-1"]
    assert rdma.pop_finished_sending() == {"sent-1"}
    assert rdma.pop_finished_sending() == set()
    assert rdma.pop_finished_recving() == {"recv-1"}
    assert rdma.pop_finished_recving() == set()


def test_real_rdma_port_rejects_native_layout_without_block_mapping() -> None:
    native_engine = FakeNativeRdmaEngine()

    def broken_register_local_layers(layers):
        registered = [{**layers[0], "mr_desc": {"ptr": 0x1000, "addr_rkey_list": []}}]
        registered[0].pop("block_ids")
        return registered

    native_engine.register_local_layers = broken_register_local_layers
    rdma = RealRdmaPort(native_engine)
    layer = LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        base_addr=0x1000,
        block_bytes=1024,
        block_ids=(0,),
        k_block_addrs=(0x1000,),
        v_block_addrs=(0x1400,),
    )

    with pytest.raises(KeyError, match="block_ids"):
        rdma.register_local_layers((layer,))


def test_pd_worker_builds_native_rdma_by_default_when_extension_exists(monkeypatch) -> None:
    monkeypatch.setattr(native, "PdRdmaEngine", FakeNativeRdmaEngineCtor, raising=False)
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default: {
                "pegaflow.pd.rdma.pin_worker_cpu": 64,
                "pegaflow.pd.rdma.pin_uvm_cpu": 66,
            }.get(key, default),
        )
    )

    worker = PdWorkerConnector(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert isinstance(worker.rdma, RealRdmaPort)
    assert FakeNativeRdmaEngineCtor.last_kwargs == {
        "cuda_device": 2,
        "numa_node": None,
        "domains": None,
        "device": "cuda",
        "pin_worker_cpu": 64,
        "pin_uvm_cpu": 66,
    }


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
                done_request_id="req-1",
                num_prompt_tokens=32,
                prompt_token_ids=(101, 102, 103),
                model="/data/Qwen3-4B",
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
    producer_params = dict(prefill_request.producer_kv_transfer_params)
    producer_handshake = producer_params.pop("pd_handshake")
    assert producer_params == {
        "do_remote_prefill_sender": True,
        "target_engine_id": "decode",
        "target_request_id": "req-1",
    }
    assert producer_handshake["request_id"] == "req-1"
    assert producer_handshake["layers"][0]["block_ids"] == [1, 2]

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


def test_scheduler_carries_cross_process_rdma_handshake() -> None:
    handshake = {
        "request_id": "decode-1",
        "engine_id": "decode",
        "tp_rank": 0,
        "tp_size": 1,
        "block_size": 16,
        "kv_layout": "HND",
        "layers": [
            {
                "layer_name": "layer.0",
                "layer_idx": 0,
                "base_addr": 0x1000,
                "block_bytes": 1024,
                "block_ids": [1],
                "k_block_addrs": [0x1000],
                "v_block_addrs": [0x1400],
                "mr_desc": {"addr_rkey_list": [["10.0.0.1:1", 17]]},
            }
        ],
    }
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="prefill-1",
        num_prompt_tokens=3,
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "decode-1",
            "pd_handshake": handshake,
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    parsed = meta.reqs_to_push["prefill-1"].handshake
    assert parsed is not None
    assert parsed.request_id == "decode-1"
    assert parsed.layers[0].mr_desc == {"addr_rkey_list": [["10.0.0.1:1", 17]]}


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
    assert meta.reqs_to_wait["req-1"].done_request_id == "req-1"


def test_scheduler_registers_remote_wait_once_until_done() -> None:
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
            "prefill_url": "http://127.0.0.1:8001",
            "done_endpoint": "tcp://127.0.0.1:7200",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    first = scheduler.build_connector_meta(SimpleNamespace())
    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    second = scheduler.build_connector_meta(SimpleNamespace())

    assert set(first.reqs_to_wait) == {"req-1"}
    assert second.reqs_to_wait == {}

    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=None, finished_recving={"req-1"})
    )
    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    third = scheduler.build_connector_meta(SimpleNamespace())
    assert third.reqs_to_wait == {}

    scheduler.request_finished(request, ([1],))
    scheduler.build_connector_meta(SimpleNamespace())
    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    after_release = scheduler.build_connector_meta(SimpleNamespace())
    assert set(after_release.reqs_to_wait) == {"req-1"}


def test_d_worker_submits_prefill_request_after_alloc() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        prefill_sender=prefill_sender,
    )
    worker.register_kv_caches({"layer.0": tensor})
    meta = PdConnectorMetadata(
        reqs_to_wait={
            "internal-d": WaitReqMeta(
                local_block_ids=([1],),
                remote=RemoteEndpoint(
                    engine_id="prefill",
                    done_endpoint="tcp://127.0.0.1:7200",
                ),
                remote_request_id="external-p",
                done_request_id="external-d",
                num_prompt_tokens=3,
                prompt_token_ids=(11, 12, 13),
                model="/data/Qwen3-4B",
                prefill_url="http://127.0.0.1:8001",
                prefill_max_tokens=1,
            )
        }
    )

    worker.start_load_kv(meta, None)

    assert len(prefill_sender.tasks) == 1
    task = prefill_sender.tasks[0]
    assert task.request_id == "external-p"
    assert task.prefill_url == "http://127.0.0.1:8001"
    assert task.model == "/data/Qwen3-4B"
    assert task.prompt_token_ids == (11, 12, 13)
    task_params = dict(task.kv_transfer_params)
    task_handshake = task_params.pop("pd_handshake")
    assert task_params == {
        "do_remote_prefill_sender": True,
        "target_engine_id": "decode",
        "target_request_id": "external-d",
        "done_endpoint": "tcp://127.0.0.1:7200",
    }
    assert task_handshake["request_id"] == "internal-d"
    assert task_handshake["layers"][0]["block_ids"] == [1]


def test_pd_proxy_only_sends_decode_request_with_prefill_hint() -> None:
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

    assert req.decode_body["request_id"] == "pd-test-d"
    assert req.decode_body["max_tokens"] == 4
    assert req.decode_body["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "model": "/data/Qwen3-4B",
        "prefill_url": "http://127.0.0.1:8001",
        "prefill_max_tokens": 1,
        "remote_engine_id": "prefill",
        "remote_request_id": "pd-test-p",
        "done_request_id": "pd-test-d",
        "done_endpoint": "tcp://127.0.0.1:7200",
    }
