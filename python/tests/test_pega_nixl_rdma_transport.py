from __future__ import annotations

import queue
from types import SimpleNamespace

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.nixl_connector.metadata import NixlHandshakePayload
from pegaflow.nixl_connector.base_worker import (
    NixlBaseConnectorWorker,
    _unregister_remote_engine_if_supported,
    _virtually_split_kv_in_blocks,
)
from pegaflow.nixl_connector.pega_pull_worker import (
    PegaNixlPullConnectorWorker,
    encode_pega_rdma_handshake_payload,
)
from pegaflow.nixl_connector.rdma_transport import PegaNixlRdmaTransport
from pegaflow.pd_connector.metadata import LayerRemoteLayout
from pegaflow.pd_connector.rdma import MockRdmaPort


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2),
        kv_transfer_config=SimpleNamespace(engine_id="d0"),
    )


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, element_size: int = 2) -> None:
        self.shape = shape
        self._element_size = element_size
        self._strides = self._contiguous_strides(shape)
        self._base_addr = 0x100000

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def element_size(self) -> int:
        return self._element_size

    def data_ptr(self) -> int:
        return self._base_addr

    def get_device(self) -> int:
        return 0

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = 1
        result = []
        for dim in reversed(shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))


class RegisteringRdmaPort(MockRdmaPort):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        registered = tuple(
            LayerRemoteLayout(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                block_ids=layer.block_ids,
                regions=layer.regions,
                mr_desc={"test_mr": layer.layer_idx},
            )
            for layer in layers
        )
        self.local_layers = registered
        return registered


def test_transport_registers_layers_and_builds_handshake() -> None:
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="d0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    kv_cache = FakeTensor((2, 4, 2, 1, 8))

    transport.register_kv_caches({"layer.0": kv_cache})
    handshake = transport.build_local_handshake("decode-req", [1, 3])

    assert handshake.request_id == "decode-req"
    assert handshake.engine_id == "d0"
    assert handshake.block_size == 2
    assert [layer.layer_name for layer in handshake.layers] == ["layer.0"]
    assert handshake.layers[0].block_ids == (1, 3)
    assert handshake.layers[0].mr_desc is not None


def test_transport_pushes_matching_local_blocks_to_remote_blocks() -> None:
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="p0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    kv_cache = FakeTensor((2, 8, 2, 1, 8))
    transport.register_kv_caches({"layer.0": kv_cache})
    handshake = transport.build_local_handshake("decode-req", [5, 7])

    transport.push_blocks(
        request_id="prefill-req",
        remote_handshake=handshake,
        local_block_ids=([1, 3],),
        remote_block_ids=([5, 7],),
    )

    pushed = rdma.pushed_layers["prefill-req"]
    assert len(pushed) == 1
    layer_idx, blocks = pushed[0]
    assert layer_idx == 0
    assert [block.regions[0].block_id for block in blocks] == [5, 7]
    assert [block.regions[0].src_offset_bytes for block in blocks] == [
        kv_cache.stride()[1] * kv_cache.element_size(),
        3 * kv_cache.stride()[1] * kv_cache.element_size(),
    ]
    assert "prefill-req" in rdma.pop_finished_sending()


def test_transport_pulls_remote_blocks_into_local_blocks() -> None:
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="d0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    kv_cache = FakeTensor((2, 8, 2, 1, 8))
    transport.register_kv_caches({"layer.0": kv_cache})
    remote_handshake = transport.build_local_handshake("prefill-req", [2, 4])

    transport.pull_blocks(
        request_id="decode-req",
        remote_handshake=remote_handshake,
        local_block_ids=([6, 7],),
        remote_block_ids=([2, 4],),
    )

    pulled = rdma.pulled_layers["decode-req"]
    assert len(pulled) == 1
    layer_idx, blocks = pulled[0]
    assert layer_idx == 0
    assert [block.regions[0].block_id for block in blocks] == [6, 7]
    assert [block.regions[0].src_offset_bytes for block in blocks] == [
        2 * kv_cache.stride()[1] * kv_cache.element_size(),
        4 * kv_cache.stride()[1] * kv_cache.element_size(),
    ]
    assert "decode-req" in rdma.pop_finished_recving()


class RecordingPegaRdma:
    def __init__(self) -> None:
        self.pull_calls: list[dict[str, object]] = []

    def pull_blocks(self, **kwargs: object) -> None:
        self.pull_calls.append(kwargs)

    def pop_finished_recving(self) -> set[str]:
        return {"decode-req"}

    def pop_finished_sending(self) -> set[str]:
        return set()

    def close_request(self, request_id: str) -> None:
        return None


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        class DoneFuture:
            def __init__(self):
                self._result = fn(*args, **kwargs)

            def result(self):
                return self._result

            def add_done_callback(self, callback):
                callback(self)

        return DoneFuture()


def test_missing_virtual_split_topology_defaults_to_unsplit_layout() -> None:
    old_vllm_topology = SimpleNamespace()

    assert _virtually_split_kv_in_blocks(old_vllm_topology) is False


def test_missing_topology_unregister_method_is_ignored() -> None:
    old_vllm_topology = SimpleNamespace()

    _unregister_remote_engine_if_supported(old_vllm_topology, "p0")


def test_cleanup_remote_engine_without_last_active_is_idempotent() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    released_handles = []
    removed_agents = []
    worker.dst_xfer_side_handles = {"d0": {0: "handle-d0-r0"}}
    worker._remote_agents = {"d0": {0: "agent-d0-r0"}}
    worker.kv_caches_base_addr = {"d0": 0x1234}
    worker.dst_num_blocks = {"d0": 8}
    worker.tp_mappings = {"d0": object()}
    worker._engine_last_active = {}
    worker.transfer_topo = SimpleNamespace(unregister_remote_engine=lambda _engine_id: None)
    worker.nixl_wrapper = SimpleNamespace(
        release_dlist_handle=released_handles.append,
        remove_remote_agent=removed_agents.append,
    )

    worker._cleanup_remote_engine("d0", log_eviction=False)

    assert released_handles == ["handle-d0-r0"]
    assert removed_agents == ["agent-d0-r0"]
    assert worker._remote_agents == {}


def test_pega_pull_worker_uses_rdma_pull_and_reports_completion() -> None:
    worker = object.__new__(PegaNixlPullConnectorWorker)
    rdma = RecordingPegaRdma()
    worker.pega_rdma = rdma
    worker._recving_metadata = {}
    worker._recving_transfers = {}
    worker._failed_recv_reqs = SimpleNamespace(empty=lambda: True)
    worker._rdma_pull_executor = ImmediateExecutor()
    worker._pending_rdma_recvs = {}
    worker._completed_rdma_recvs = queue.Queue()
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._ready_requests = SimpleNamespace(empty=lambda: True)
    worker._engine_last_active = {}
    worker._remote_agents = {"p0": {0: "pega-rdma:p0:0"}}
    worker._remote_rdma_handshakes = {"p0": {0: _remote_handshake()}}
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda _engine_id: SimpleNamespace(
            remote_block_size=2,
            remote_tp_size=1,
            remote_physical_blocks_per_logical=1,
        ),
        block_size_ratio=lambda _remote_block_size: 1,
        tp_ratio=lambda _remote_tp_size: 1,
    )
    worker.block_size = 2
    worker.use_mla = False
    worker.world_size = 1
    worker.tp_rank = 0
    worker.tp_mappings = {
        "p0": SimpleNamespace(
            source_ranks_per_group=((0,),),
            all_source_ranks=(0,),
        )
    }
    worker._is_hma_required = False
    worker.use_host_buffer = False
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.xfer_stats = SimpleNamespace(record_kv_expired_req=lambda: None)
    worker._logical_to_kernel_block_ids = lambda block_ids: block_ids
    worker._logical_to_remote_kernel_block_ids = lambda block_ids, _ratio: block_ids
    worker._send_heartbeats = lambda _metadata: None
    worker._apply_prefix_caching = lambda local, remote, _physical: (local, remote)
    worker._handle_failed_transfer = lambda _req_id, _handle: None

    metadata = SimpleNamespace(
        reqs_to_recv={
            "decode-req": SimpleNamespace(
                local_block_ids=([6, 7],),
                local_physical_block_ids=(),
                tp_size=1,
                remote=SimpleNamespace(
                    engine_id="p0",
                    request_id="prefill-req",
                    block_ids=([2, 4],),
                    host="127.0.0.1",
                    port=5555,
                ),
            )
        },
        reqs_in_batch=set(),
        reqs_not_processed=set(),
        reqs_to_send={},
    )

    worker.start_load_kv(metadata)
    assert rdma.pull_calls == [
        {
            "request_id": "decode-req",
            "remote_handshake": _remote_handshake(),
            "local_block_ids": ([6, 7],),
            "remote_block_ids": ([2, 4],),
        }
    ]

    assert worker.get_finished() == (set(), {"decode-req"})


def test_pega_pull_handshake_payload_round_trips() -> None:
    payload = encode_pega_rdma_handshake_payload("compat", _remote_handshake())

    assert isinstance(payload, NixlHandshakePayload)
    assert payload.compatibility_hash == "compat"


def _remote_handshake():
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="p0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    transport.register_kv_caches({"layer.0": FakeTensor((2, 8, 2, 1, 8))})
    return transport.build_local_handshake("prefill-req", [2, 4])
