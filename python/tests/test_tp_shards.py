"""Contracts for splitting one vLLM TP replica across PegaFlow servers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

from pegaflow.connector import PegaKVConnector  # noqa: E402
from pegaflow.connector.common import (  # noqa: E402
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
    TpShardTopology,
)
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402
from pegaflow.connector.worker import WorkerConnector  # noqa: E402
from pegaflow.pegaflow import QueryLoading, QueryReady  # noqa: E402


def _topology() -> TpShardTopology:
    return TpShardTopology.from_config(
        default_endpoint="http://unused:50055",
        configured_endpoints=["http://node-a:50055", "http://node-b:50055"],
        global_tp_size=8,
        global_world_size=8,
    )


def _context(**kwargs) -> ConnectorContext:
    defaults = {
        "instance_id": "instance",
        "namespace": "namespace:tp-shard-0-of-2",
        "block_size": 16,
        "tp_size": 8,
        "world_size": 8,
        "tp_rank": 0,
        "device_id": 0,
        "engine_client": MagicMock(),
        "state_manager": MagicMock(),
        "tp_shards": _topology(),
    }
    defaults.update(kwargs)
    return ConnectorContext(**defaults)  # type: ignore[arg-type]


def _vllm_config(**parallel_overrides):
    extra_config = {
        "pegaflow.tp_shard_endpoints": [
            "http://node-a:50055",
            "http://node-b:50055",
        ]
    }
    kv_transfer_config = SimpleNamespace(
        engine_id="instance",
        get_from_extra_config=lambda key, default: extra_config.get(key, default),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="model",
            dtype="bfloat16",
            hf_text_config=SimpleNamespace(kv_lora_rank=None),
            get_total_num_kv_heads=lambda: 8,
            get_head_size=lambda: 128,
            get_total_num_hidden_layers=lambda: 32,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            **{
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "world_size": 8,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                **parallel_overrides,
            }
        ),
        kv_transfer_config=kv_transfer_config,
        additional_config={},
    )


def test_topology_maps_contiguous_global_tp_ranks_to_local_servers():
    topology = _topology()

    assert topology.local_tp_size == 4
    assert topology.local_world_size == 4
    assert topology.shard_index(3) == 0
    assert topology.shard_index(4) == 1
    assert topology.local_tp_rank(4) == 0
    assert topology.local_tp_rank(7) == 3
    assert topology.namespace("base", 1) == "base:tp-shard-1-of-2"


@pytest.mark.parametrize(
    ("endpoints", "tp_size", "world_size", "message"),
    [
        ([], 8, 8, "non-empty strings"),
        (["http://a", "http://a"], 8, 8, "duplicates"),
        (["http://a", "http://b", "http://c"], 8, 8, "tensor_parallel_size"),
        (["http://a", "http://b"], 8, 9, "world_size"),
    ],
)
def test_topology_rejects_ambiguous_shard_layouts(endpoints, tp_size, world_size, message):
    with pytest.raises(ValueError, match=message):
        TpShardTopology.from_config("http://unused", endpoints, tp_size, world_size)


def test_context_exposes_node_local_server_topology_for_hma():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=False)

    assert context.tp_shard_index == 1
    assert context.effective_tp_rank == 1
    assert context.effective_tp_size == 4
    assert context.effective_world_size == 4


def test_worker_connector_routes_global_tp_rank_to_its_local_server(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr("pegaflow.connector.get_tensor_model_parallel_rank", lambda: 5)
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr("pegaflow.connector.EngineRpcClient", client_factory)
    monkeypatch.setattr("pegaflow.connector.ServiceStateManager", MagicMock())

    connector = PegaKVConnector(_vllm_config(), KVConnectorRole.WORKER)
    try:
        assert connector._engine_endpoint == "http://node-b:50055"
        assert connector._ctx.namespace.endswith(":tp-shard-1-of-2")
        assert connector._ctx.effective_tp_rank == 1
        assert connector._ctx.effective_tp_size == 4
        assert connector._ctx.effective_world_size == 4
    finally:
        connector.shutdown()

    client_factory.assert_called_once_with("http://node-b:50055")


def test_scheduler_opens_a_local_topology_session_on_every_server(monkeypatch):
    first = MagicMock()
    second = MagicMock()
    client_factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr("pegaflow.connector.EngineRpcClient", client_factory)
    monkeypatch.setattr("pegaflow.connector.ServiceStateManager", MagicMock())

    connector = PegaKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    try:
        first.start_session_watcher.assert_called_once()
        first_session = first.start_session_watcher.call_args.args
        assert first_session[0] == "instance"
        assert first_session[1].endswith(":tp-shard-0-of-2")
        assert first_session[2:] == (4, 4)
        second.start_session_watcher.assert_called_once()
        second_session = second.start_session_watcher.call_args.args
        assert second_session[0] == "instance"
        assert second_session[1].endswith(":tp-shard-1-of-2")
        assert second_session[2:] == (4, 4)
    finally:
        connector.shutdown()


@pytest.mark.parametrize(
    "parallel_overrides",
    [
        {"pipeline_parallel_size": 2, "world_size": 16},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
    ],
)
def test_connector_rejects_non_tp_parallelism_across_server_shards(monkeypatch, parallel_overrides):
    monkeypatch.setattr("pegaflow.connector.EngineRpcClient", MagicMock())

    with pytest.raises(ValueError, match="TP-only parallelism"):
        PegaKVConnector(_vllm_config(**parallel_overrides), KVConnectorRole.SCHEDULER)


def test_pure_mla_collapses_storage_tp_but_stripes_saves_within_each_node():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=True)

    assert context.effective_tp_rank == 0
    assert context.effective_tp_size == 1
    assert context.effective_world_size == 4
    assert context.local_physical_tp_rank == 1
    assert context.local_physical_tp_size == 4


def test_scheduler_uses_common_prefix_and_exact_per_shard_leases():
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.side_effect = [
        QueryReady(3, b"first-long"),
        QueryReady(2, b"first-exact"),
    ]
    second.query_prefetch.return_value = QueryReady(2, b"second-exact")
    scheduler = SchedulerConnector(_context(), engine_clients=(first, second))
    hashes = [b"h0", b"h1", b"h2"]

    ready = scheduler._count_available_block_prefix(hashes, "request")

    assert ready is not None
    assert ready.num_hit_blocks == 2
    assert ready.leases == (b"first-exact", b"second-exact")
    assert first.query_prefetch.call_args_list == [
        call(
            "instance",
            hashes,
            req_id="request",
            wait_for_full_prefix=False,
        ),
        call(
            "instance",
            hashes[:2],
            req_id="request:tp-common-2",
            wait_for_full_prefix=False,
        ),
    ]
    first.release.assert_called_once_with(b"first-long")
    second.release.assert_not_called()


def test_scheduler_releases_ready_shards_when_another_shard_is_loading():
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.return_value = QueryReady(2, b"first")
    second.query_prefetch.return_value = QueryLoading()
    scheduler = SchedulerConnector(_context(), engine_clients=(first, second))

    assert scheduler._count_available_block_prefix([b"h0", b"h1"], "request") is None
    first.release.assert_called_once_with(b"first")


@pytest.mark.parametrize(
    "invalid_ready",
    [
        QueryReady(3, b"too-many"),
        QueryReady(1, b""),
    ],
)
def test_scheduler_rejects_invalid_shard_query_results_without_leaking_lease(invalid_ready):
    first = MagicMock()
    second = MagicMock()
    first.query_prefetch.return_value = QueryReady(2, b"first")
    second.query_prefetch.return_value = invalid_ready
    scheduler = SchedulerConnector(_context(), engine_clients=(first, second))

    with pytest.raises(RuntimeError, match="TP shard 1"):
        scheduler._count_available_block_prefix([b"h0", b"h1"], "request")

    first.release.assert_called_once_with(b"first")
    if invalid_ready.lease:
        second.release.assert_called_once_with(invalid_ready.lease)


def test_worker_selects_the_lease_for_its_local_server():
    engine_client = MagicMock()
    engine_client.load.return_value = (True, "")
    context = _context(
        tp_rank=5,
        device_id=1,
        namespace="namespace:tp-shard-1-of-2",
        engine_client=engine_client,
    )
    worker = WorkerConnector(context)
    worker._registered_layers = ["layer"]
    metadata = PegaConnectorMetadata(
        load_intents={
            "request": LoadIntent(
                block_ids_by_group=((7,),),
                leases=(b"node-a", b"node-b"),
                num_tokens=16,
            )
        }
    )

    try:
        worker.start_load_kv(metadata, SimpleNamespace(no_compile_layers={}))
    finally:
        worker._registered_layers = []
        worker.shutdown()

    loads = engine_client.load.call_args.args[5]
    assert loads == [(b"node-b", [[7]])]


def test_each_tp_shard_has_a_local_unregister_leader():
    for tp_rank in range(8):
        engine_client = MagicMock()
        engine_client.unregister_context.return_value = (True, "")
        worker = WorkerConnector(_context(tp_rank=tp_rank, engine_client=engine_client))
        worker._registered_layers = ["layer"]

        worker.unregister_context()
        worker.shutdown()

        if tp_rank in (0, 4):
            engine_client.unregister_context.assert_called_once_with("instance")
        else:
            engine_client.unregister_context.assert_not_called()
