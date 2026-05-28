from __future__ import annotations

import sys
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

import pegaflow.pd_connector.worker as worker_mod  # noqa: E402
from pegaflow.pd_connector import PdConnector  # noqa: E402
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
    PushReqMeta,
    TransferRegionLayout,
    WaitReqMeta,
    handshake_from_dict,
    handshake_to_dict,
)
from pegaflow.pd_connector.proxy import (  # noqa: E402
    ProxyConfig,
    build_pd_proxy_request,
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

    def submit(self, task) -> None:
        self.tasks.append(task)


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


def test_pd_worker_registers_mla_and_indexer_layouts_from_layer_specs() -> None:
    main_tensor = FakeTensor(
        shape=(8, 64, 656),
        stride=(64 * 656, 656, 1),
        ptr=0x1000,
        element_size=1,
    )
    indexer_tensor = FakeTensor(
        shape=(8, 64, 128),
        stride=(64 * 128, 128, 1),
        ptr=0x200000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": SimpleNamespace(block_size=64, page_size_bytes=64 * 656),
            "indexer.0": SimpleNamespace(block_size=64, page_size_bytes=64 * 128),
        },
    )
    worker = PdWorkerConnector(
        fake_mla_config(),
        kv_cache_config=kv_cache_config,
        rdma=MockRdmaPort(),
    )

    worker.register_kv_caches({"layer.0": main_tensor, "indexer.0": indexer_tensor})

    assert worker.layouts["layer.0"].block_size == 64
    assert worker.layouts["layer.0"].remote_layout(0).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x1000, block_len=64 * 656),
    )
    assert worker.layouts["indexer.0"].remote_layout(1).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x200000, block_len=64 * 128),
    )
    assert (
        worker.rdma.local_layers[0].regions[0].block_len
        != worker.rdma.local_layers[1].regions[0].block_len
    )


def test_pd_connector_mla_returns_default_layout_and_allows_128_block_size() -> None:
    assert PdConnector.get_required_kvcache_layout(fake_mla_config(block_size=64)) is None
    PdConnector(fake_mla_config(block_size=128), KVConnectorRole.WORKER)


def test_pd_worker_rejects_mla_physical_logical_block_split() -> None:
    tensor = FakeTensor(
        shape=(8, 32, 128),
        stride=(32 * 128, 128, 1),
        ptr=0x1000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": SimpleNamespace(block_size=64, page_size_bytes=32 * 128),
        },
    )
    worker = PdWorkerConnector(
        fake_mla_config(block_size=64),
        kv_cache_config=kv_cache_config,
        rdma=MockRdmaPort(),
    )

    with pytest.raises(AssertionError, match="physical/logical block split"):
        worker.register_kv_caches({"layer.0": tensor})


def test_p_worker_maps_mla_prefill_tp_greater_than_decode_tp() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PdWorkerConnector(
        fake_mla_config(tp_rank=2, tp_size=8),
        rdma=MockRdmaPort(),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r2": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert worker.rdma.remote_handshakes["prefill-r2"] is handshakes[1]


def test_p_worker_skips_non_representative_mla_prefill_rank() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PdWorkerConnector(
        fake_mla_config(tp_rank=3, tp_size=8),
        rdma=MockRdmaPort(),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r3": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert "prefill-r3" not in worker.rdma.remote_handshakes
    assert worker.get_finished({"prefill-r3"}) == ({"prefill-r3"}, None)


def test_real_rdma_port_preserves_native_contract_for_pd_push() -> None:
    native_engine = FakeNativeRdmaEngine()
    rdma = RealRdmaPort(native_engine)
    layer = hnd_remote_layer(
        block_ids=(0, 1),
        k_base=0x1000,
        v_base=0x9000,
        block_len=4096,
    )

    registered = rdma.register_local_layers((layer,))

    assert registered[0].mr_desc == {
        "ptr": 0x1000,
        "addr_rkey_list": [["10.0.0.1:1", 17]],
    }
    assert registered[0].block_ids == (0, 1)
    assert registered[0].regions == layer.regions

    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=registered,
    )
    rdma.open_request("req-1", handshake)

    assert native_engine.remote_regs[0][0] == "req-1"
    assert native_engine.remote_regs[0][1]["layers"][0]["mr_desc"]["ptr"] == 0x1000
    assert native_engine.remote_regs[0][1]["layers"][0]["block_ids"] == [0, 1]
    assert native_engine.remote_regs[0][1]["layers"][0]["regions"] == [
        {"region_idx": 0, "base_addr": 0x1000, "block_len": 4096},
        {"region_idx": 1, "base_addr": 0x9000, "block_len": 4096},
    ]

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
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 1,
                    "src_offset_bytes": 2048 * 2,
                    "bytes": 4096,
                },
                {
                    "region_idx": 1,
                    "block_id": 1,
                    "src_offset_bytes": (16384 + 2048) * 2,
                    "bytes": 4096,
                },
            ],
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
    layer = hnd_remote_layer(block_ids=(0,), block_len=1024)

    with pytest.raises(KeyError, match="block_ids"):
        rdma.register_local_layers((layer,))


def test_pd_handshake_serializes_regions_layout() -> None:
    layer = hnd_remote_layer(
        block_ids=(8, 9, 10),
        k_base=0x10_000,
        v_base=0x20_000,
        block_len=0x400,
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )

    data = handshake_to_dict(handshake)
    assert data is not None
    layer_data = data["layers"][0]
    assert layer_data["block_ids"] == [8, 9, 10]
    assert layer_data["regions"] == [
        {"region_idx": 0, "base_addr": 0x10_000, "block_len": 0x400},
        {"region_idx": 1, "base_addr": 0x20_000, "block_len": 0x400},
    ]
    assert "k_block_addrs" not in layer_data
    assert "v_block_addrs" not in layer_data
    assert "linear" not in layer_data

    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].block_ids == layer.block_ids
    assert restored.layers[0].regions == layer.regions
    assert restored.layers[0].region_block_addrs(0) == (
        0x10_000 + 8 * 0x400,
        0x10_000 + 9 * 0x400,
        0x10_000 + 10 * 0x400,
    )


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
                "pegaflow.pd.rdma.rank_map": {"0": {"nic": "mlx5_2", "worker_cpu": 64}},
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = PdWorkerConnector(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert isinstance(worker.rdma, RealRdmaPort)
    assert FakeNativeRdmaEngineCtor.last_kwargs == {
        "cuda_device": 2,
        "numa_node": None,
        "domains": ["mlx5_2"],
        "device": "cuda",
        "pin_worker_cpu": 64,
    }


def test_pd_worker_uses_runtime_tp_rank_for_rdma_rank_map(monkeypatch) -> None:
    monkeypatch.setattr(native, "PdRdmaEngine", FakeNativeRdmaEngineCtor, raising=False)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_world_size", lambda: 8)
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default: {
                "pegaflow.pd.rdma.rank_map": {
                    "0": {"nic": "mlx5_1", "worker_cpu": 16},
                    "2": {"nic": "mlx5_2", "worker_cpu": 60},
                },
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=8),
    )

    worker = PdWorkerConnector(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert FakeNativeRdmaEngineCtor.last_kwargs["domains"] == ["mlx5_2"]
    assert FakeNativeRdmaEngineCtor.last_kwargs["pin_worker_cpu"] == 60


def test_pd_worker_rejects_legacy_global_rdma_config(monkeypatch) -> None:
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
                "pegaflow.pd.rdma.domains": ["mlx5_2"],
                "pegaflow.pd.rdma.rank_map": {"0": {"nic": "mlx5_2", "worker_cpu": 64}},
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = PdWorkerConnector(config)
    with pytest.raises(RuntimeError, match="legacy keys"):
        worker.register_kv_caches({"layer.0": tensor})


def test_rdma_native_blocks_coalesce_contiguous_ranges() -> None:
    blocks = [
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=10, src_offset_bytes=0x1000, bytes=0x400),
                BlockRegionSlice(block_id=10, src_offset_bytes=0x9000, bytes=0x400),
            ),
        ),
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=11, src_offset_bytes=0x1400, bytes=0x400),
                BlockRegionSlice(block_id=11, src_offset_bytes=0x9400, bytes=0x400),
            ),
        ),
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=13, src_offset_bytes=0x2000, bytes=0x400),
                BlockRegionSlice(block_id=13, src_offset_bytes=0xA000, bytes=0x400),
            ),
        ),
    ]

    native_blocks = _layer_blocks_to_native(blocks)

    assert native_blocks == [
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 10,
                    "src_offset_bytes": 0x1000,
                    "bytes": 0x800,
                },
                {
                    "region_idx": 1,
                    "block_id": 10,
                    "src_offset_bytes": 0x9000,
                    "bytes": 0x800,
                },
            ],
        },
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 13,
                    "src_offset_bytes": 0x2000,
                    "bytes": 0x400,
                },
                {
                    "region_idx": 1,
                    "block_id": 13,
                    "src_offset_bytes": 0xA000,
                    "bytes": 0x400,
                },
            ],
        },
    ]


def test_pd_worker_wait_handshake_uses_registered_native_mr_desc() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=RealRdmaPort(native_engine),
    )
    worker.register_kv_caches({"layer.0": tensor})
    meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1],),
                remote_request_id="req-1-p",
                done_request_id="req-1-d",
                prompt_token_ids=(11, 12, 13),
                prefill_url="http://p:8001",
            )
        }
    )

    worker.start_load_kv(meta, None)

    _, handshake = native_engine.remote_regs[-1]
    layer = handshake["layers"][0]
    assert layer["mr_desc"] == {
        "ptr": tensor.data_ptr(),
        "addr_rkey_list": [("10.0.0.1:1", 17)],
    }
    assert layer["block_ids"] == [1]


def test_pd_worker_pushes_flash_attn_hnd_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target_request_id="req-1",
                handshakes=(
                    PdHandshake(
                        request_id="req-1",
                        engine_id="decode",
                        tp_rank=0,
                        tp_size=1,
                        block_size=16,
                        layers=(),
                    ),
                ),
            )
        }
    )

    worker.start_load_kv(meta, None)
    attn_metadata = SimpleNamespace(slot_mapping=FakeSlotMapping([16, 17, 32, -1]))
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert unique_blocks_from_slot_mapping(attn_metadata.slot_mapping, 16) == {1, 2}
    worker.wait_for_save()
    drain_pd_pushes(worker)
    assert worker.rdma.pushed_layers["req-1"][0][0] == 0
    first_layer_blocks = worker.rdma.pushed_layers["req-1"][0][1]
    assert len(first_layer_blocks) == 1
    assert first_layer_blocks[0].regions[0].block_id == 1
    assert first_layer_blocks[0].regions[0].bytes == tensor.element_size() * 16 * 4 * 32 * 2
    assert worker.rdma.pushed_layers["req-1"][1][0] == 1
    finished_sending, finished_recving = worker.get_finished({"req-1"})
    assert finished_sending == {"req-1"}
    assert finished_recving is None
    assert "req-1" not in worker.rdma.registered


def test_pd_worker_get_finished_does_not_poll_wait_reqs() -> None:
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=rdma
    )
    worker._wait_reqs["req-1"] = WaitReqMeta(
        local_block_ids=([1],),
        remote_request_id="req-1-p",
        done_request_id="req-1-d",
        prompt_token_ids=(1,),
        prefill_url="http://p:8001",
    )

    def fail_poll_done(req_id: str) -> bool:
        raise AssertionError(f"poll_done should not run for {req_id}")

    rdma.poll_done = fail_poll_done  # type: ignore[method-assign]
    rdma.mark_done("req-1")

    assert worker.get_finished(set()) == (None, {"req-1"})


def test_p_worker_extracts_slot_mapping_blocks_once_per_forward() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )

    slot_mapping = FakeSlotMapping([16, 17])
    attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert slot_mapping.cpu_calls == 1
    worker.wait_for_save()


def test_p_worker_slot_mapping_cache_survives_wait_for_save_within_step() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )

    slot_mapping = FakeSlotMapping([16, 17])
    attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.wait_for_save()
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert slot_mapping.cpu_calls == 1


def test_pd_worker_publishes_wait_handshake_and_delays_done_until_all_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    d_rdma = MockRdmaPort()
    d_worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=d_rdma,
    )
    d_worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})

    wait_meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1, 2],),
                remote_request_id="req-1",
                done_request_id="req-1",
                prompt_token_ids=(101, 102, 103),
                prefill_url="http://p:8001",
            )
        }
    )
    d_worker.start_load_kv(wait_meta, None)

    handshake = d_rdma.remote_handshakes.get("req-1")
    assert handshake is not None
    assert handshake.engine_id == "decode"
    assert handshake.block_size == 16
    assert isinstance(handshake.imm_id, int)
    assert handshake.layers[0].block_ids == (1, 2)
    assert handshake.layers[0].regions[0] == TransferRegionLayout(
        region_idx=0,
        base_addr=tensor.data_ptr(),
        block_len=4 * 16 * 32 * 2,
    )
    assert handshake.layers[0].region_block_addrs(0)[0] == (tensor.data_ptr() + 1 * 4 * 16 * 32 * 2)

    push_worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    push_worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    push_meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target_request_id="req-1",
                handshakes=(handshake,),
            )
        }
    )
    push_worker.start_load_kv(push_meta, None)
    assert push_worker.rdma.remote_handshakes["req-1"] is handshake

    push_worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="req-1",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )
    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    push_worker.wait_for_save()
    drain_pd_pushes(push_worker)
    assert push_worker.get_finished(set()) == (None, None)

    push_worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([2],),
                    target_request_id="req-1",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )
    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    push_worker.wait_for_save()
    drain_pd_pushes(push_worker)
    assert push_worker.get_finished({"req-1"}) == ({"req-1"}, None)


def test_scheduler_delays_producer_block_free_until_send_finishes() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
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

    early_release_meta = scheduler.build_connector_meta(SimpleNamespace())
    assert early_release_meta.reqs_to_release == set()

    scheduler.update_connector_output(
        SimpleNamespace(finished_sending={"req-1"}, finished_recving=None)
    )
    release_meta = scheduler.build_connector_meta(SimpleNamespace())
    assert release_meta.reqs_to_release == {"req-1"}


def test_scheduler_emits_cached_producer_chunks() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "decode-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    first = scheduler.build_connector_meta(
        SimpleNamespace(scheduled_cached_reqs=None, total_num_scheduled_tokens=8192)
    )
    assert first.reqs_to_push["req-1"].local_block_ids == ([1, 2],)

    cached = SimpleNamespace(req_ids=["req-1"], new_block_ids=[([3],)])
    second = scheduler.build_connector_meta(
        SimpleNamespace(scheduled_cached_reqs=cached, total_num_scheduled_tokens=1808)
    )

    assert second.reqs_to_push["req-1"].local_block_ids == ([3],)
    assert second.reqs_to_push["req-1"].target_request_id == "decode-1"


def test_scheduler_carries_prompt_tokens_for_d_to_p_oob() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={"do_remote_prefill": True, "prefill_url": "http://p:8001"},
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
        "layers": [
            {
                "layer_name": "layer.0",
                "layer_idx": 0,
                "block_ids": [1],
                "regions": [
                    {"region_idx": 0, "base_addr": 0x1000, "block_len": 1024},
                    {"region_idx": 1, "base_addr": 0x1400, "block_len": 1024},
                ],
                "mr_desc": {"addr_rkey_list": [["10.0.0.1:1", 17]]},
            }
        ],
    }
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="prefill-1",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "decode-1",
            "pd_handshakes": [handshake],
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    push_req = meta.reqs_to_push["prefill-1"]
    assert len(push_req.handshakes) == 1
    parsed = push_req.handshakes[0]
    assert parsed.request_id == "decode-1"
    assert parsed.layers[0].mr_desc == {"addr_rkey_list": [["10.0.0.1:1", 17]]}


def test_scheduler_ignores_legacy_fake_rdma_done_endpoint() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://p:8001",
            "done_endpoint": "tcp://127.0.0.1:7200",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].done_request_id == "req-1"


def test_scheduler_registers_remote_wait_once_until_done() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://127.0.0.1:8001",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    first = scheduler.build_connector_meta(SimpleNamespace())
    assert set(first.reqs_to_wait) == {"req-1"}

    # After finished_recving, _completed_waits prevents re-registration (preemption safety)
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


def test_d_worker_rank0_dispatches_prefill_on_wait() -> None:
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
        rdma=MockRdmaPort(),
        prefill_sender=prefill_sender,
    )
    worker.register_kv_caches({"layer.0": tensor})
    meta = PdConnectorMetadata(
        reqs_to_wait={
            "internal-d": WaitReqMeta(
                local_block_ids=([1],),
                remote_request_id="external-p",
                done_request_id="external-d",
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
    assert task.kv_transfer_params["target_request_id"] == "external-d"
    handshakes = task.kv_transfer_params["pd_handshakes"]
    assert len(handshakes) == 1
    assert handshakes[0]["request_id"] == "external-d"
    assert handshakes[0]["layers"][0]["block_ids"] == [1]


def test_p_worker_selects_matching_tp_rank_handshake() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    handshakes = (
        PdHandshake(
            request_id="decode-r0",
            engine_id="decode",
            tp_rank=0,
            tp_size=2,
            block_size=16,
            layers=(),
        ),
        PdHandshake(
            request_id="decode-r1",
            engine_id="decode",
            tp_rank=1,
            tp_size=2,
            block_size=16,
            layers=(),
        ),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=1, tensor_parallel_size=2),
        ),
        rdma=MockRdmaPort(),
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert worker.rdma.remote_handshakes["prefill-r1"] is handshakes[1]


def test_p_worker_pushes_registered_blocks_from_save_kv_layer() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(
                        PdHandshake(
                            request_id="decode",
                            engine_id="decode",
                            tp_rank=0,
                            tp_size=1,
                            block_size=16,
                            layers=(),
                        ),
                    ),
                )
            }
        ),
        None,
    )

    attn_metadata = SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert [layer_idx for layer_idx, _ in rdma.pushed_layers["prefill-r0"]] == [0, 1]
    assert worker.get_finished({"prefill-r0"})[0] == {"prefill-r0"}


def test_p_worker_maps_local_blocks_to_remote_blocks_by_position() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([3, 4],),
                    target_request_id="decode",
                    handshakes=(
                        PdHandshake(
                            request_id="decode",
                            engine_id="decode",
                            tp_rank=0,
                            tp_size=1,
                            block_size=16,
                            layers=(
                                hnd_remote_layer(
                                    block_ids=(68, 69),
                                    block_len=4096,
                                ),
                            ),
                        ),
                    ),
                )
            }
        ),
        None,
    )

    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16, 4 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    _, pushed = rdma.pushed_layers["prefill-r0"][0]
    assert [block.regions[0].block_id for block in pushed] == [68]
    assert len(rdma.pushed_layers["prefill-r0"]) == 1
    assert [block.regions[0].src_offset_bytes for block in pushed] == [
        tensor.stride()[1] * 3 * tensor.element_size()
    ]
    assert pushed[0].regions[0].bytes == tensor.stride()[1] * tensor.element_size() * 2


def test_p_worker_advances_remote_blocks_across_chunk_prefill() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    handshake = PdHandshake(
        request_id="decode",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(
            hnd_remote_layer(
                block_ids=(68, 69, 70, 71),
                block_len=4096,
            ),
        ),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([3, 4],),
                    target_request_id="decode",
                    handshakes=(handshake,),
                )
            }
        ),
        None,
    )
    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16, 4 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert worker.get_finished(set())[0] is None
    assert [block.regions[0].block_id for block in rdma.pushed_layers["prefill-r0"][0][1]] == [68]
    assert len(rdma.pushed_layers["prefill-r0"]) == 1

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([5, 6],),
                    target_request_id="decode",
                    handshakes=(handshake,),
                )
            }
        ),
        None,
    )
    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([5 * 16, 6 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert [block.regions[0].block_id for block in rdma.pushed_layers["prefill-r0"][1][1]] == [70]
    assert worker.get_finished({"prefill-r0"})[0] == {"prefill-r0"}


def test_pd_proxy_only_sends_decode_request_with_prefill_hint() -> None:
    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
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
    assert "stream" not in req.decode_body
    assert req.decode_body["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "prefill_url": "http://127.0.0.1:8001",
        "remote_request_id": "pd-test-p",
        "done_request_id": "pd-test-d",
    }


def test_pd_proxy_preserves_streaming_decode_request() -> None:
    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
        timeout_s=30,
        prefill_max_tokens=1,
    )

    req = build_pd_proxy_request(
        {
            "model": "/data/Qwen3-4B",
            "prompt": "hello",
            "max_tokens": 4,
            "stream": True,
        },
        config,
        request_id="pd-test",
    )

    assert req.decode_body["stream"] is True


def test_pd_connector_requires_piecewise_cudagraph_for_layer_push() -> None:
    assert PdConnector.requires_piecewise_for_cudagraph({}) is True
