"""Regression coverage for mixed logical block-size save/query mapping."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    CacheGroupLayout,
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
    SaveIntent,
)
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402
from pegaflow.connector.tp_shards import ShardedQueryReady  # noqa: E402
from pegaflow.connector.worker import WorkerConnector  # noqa: E402


def _layout() -> CacheGroupLayout:
    return CacheGroupLayout(
        layer_names=(("full",), ("sliding",)),
        hash_group_index=0,
        has_recurrent_state=False,
        recurrent_group_indices=frozenset(),
        recurrent_layer_names=frozenset(),
        sliding_window_group_indices=frozenset({1}),
        group_sliding_windows=(None, 32),
        storage_group_ids=(0, 1),
        group_block_sizes=(32, 16),
    )


def _scheduler() -> SchedulerConnector:
    context = ConnectorContext(
        instance_id="test",
        namespace="ns",
        block_size=32,
        hash_block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
    )
    scheduler = SchedulerConnector(context)
    scheduler._cache_groups = _layout()
    return scheduler


def test_save_maps_one_full_block_to_two_sliding_blocks():
    scheduler = _scheduler()
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(
        request_id="r1",
        num_tokens=128,
        num_prompt_tokens=128,
        block_hashes=list(hashes),
    )
    scheduler._requests["r1"] = request
    scheduler._block_hashes["r1"] = scheduler._request_block_hashes(request)
    scheduler._allocated_blocks["r1"] = [
        list(range(10, 14)),
        list(range(20, 28)),
    ]
    scheduler._scheduled_tokens["r1"] = 128
    scheduler._next_stored_block_idx["r1"] = 0

    intent = scheduler._consume_full_block_saves("r1")

    assert intent is not None
    assert intent.block_ids_by_group == (
        (10, 11, 12, 13),
        (20, 21, 22, 23, 24, 25, 26, 27),
    )
    assert intent.block_hashes_by_group == (
        (hashes[1], hashes[3], hashes[5], hashes[7]),
        hashes,
    )


@pytest.mark.parametrize("per_group_hashes", [False, True], ids=["legacy", "sliding"])
def test_worker_save_skips_vllm_null_blocks(per_group_hashes):
    worker = WorkerConnector(_scheduler()._ctx)
    worker._cache_groups = _layout()
    worker._registered_layers = ["full", "sliding"]
    worker._layer_to_group = worker._cache_groups.layer_to_group()
    save_intent = SaveIntent(
        block_ids_by_group=((6, 0, 7), (0, 0, 8)),
        block_hashes=(b"h0", b"h1", b"h2"),
        block_hashes_by_group=(
            ((b"h0", b"h1", b"h2"), (b"s0", b"s1", b"s2")) if per_group_hashes else None
        ),
    )
    rows = list(worker._layer_saves(save_intent))
    assert rows == [
        ("full", (6, 7), (b"h0", b"h2")),
        ("sliding", (8,), (b"s2" if per_group_hashes else b"h2",)),
    ]
    worker._registered_layers = []
    worker.shutdown()


def test_saved_sliding_suffix_hits_without_null_placeholders():
    scheduler = _scheduler()
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(request_id="r1", num_tokens=128, block_hashes=list(hashes))
    worker = WorkerConnector(scheduler._ctx)
    worker._cache_groups = _layout()
    worker._registered_layers = ["sliding"]
    worker._layer_to_group = {"sliding": 1}
    try:
        rows = list(
            worker._layer_saves(
                SaveIntent(
                    block_ids_by_group=((10, 11, 12, 13), (0, 0, 0, 0, 0, 0, 26, 27)),
                    block_hashes=hashes[1::2],
                    block_hashes_by_group=(hashes[1::2], hashes),
                )
            )
        )
        assert rows == [("sliding", (26, 27), hashes[6:8])]
        saved = set(rows[0][2])
        scheduler._tp_shard_client.query_group_membership = lambda _i, keys, _r, _g: [
            (tuple(index for index, key in enumerate(keys) if key in saved), b"sliding")
        ]
        ready = scheduler._attach_sliding_group_queries(
            request, 0, list(hashes[1::2]), ShardedQueryReady(4, (b"dense",)), "r1"
        )
        assert ready.num_hit_blocks == 4
        assert ready.block_ranges_by_group == ((0, 4), (6, 8))
    finally:
        worker._registered_layers = []
        worker.shutdown()


@pytest.mark.parametrize("window", [32, 96])
def test_sliding_query_uses_retained_window_suffix(window):
    scheduler = _scheduler()
    scheduler._cache_groups = replace(_layout(), group_sliding_windows=(None, window))
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(request_id="r1", num_tokens=128, block_hashes=list(hashes))
    results = []

    def query_group_membership(_instance, block_hashes, _req_id, _group_id):
        results.append(tuple(block_hashes))
        return [(tuple(range(len(block_hashes))), b"lease")]

    scheduler._tp_shard_client.query_group_membership = query_group_membership
    ready = ShardedQueryReady(num_hit_blocks=1, leases=(b"dense",))

    attached = scheduler._attach_sliding_group_queries(
        request, computed_blocks=2, full_hashes=list(hashes[2:]), ready=ready, req_id="r1"
    )

    assert results == [hashes[4:6]]
    assert attached.block_ranges_by_group == ((2, 3), (4, 6))
    assert attached.leases_by_group == ((b"dense",), (b"lease",))
    assert scheduler._load_block_ids_by_group(
        ((10, 11, 12), (20, 21, 22, 23, 24, 25)),
        2,
        1,
        block_ranges_by_group=attached.block_ranges_by_group,
    ) == ((12,), (24, 25))


def test_sliding_query_shrinks_to_latest_common_boundary_on_partial_hit():
    scheduler = _scheduler()
    hashes = tuple(bytes([index]) for index in range(8))
    request = SimpleNamespace(request_id="r1", num_tokens=128, block_hashes=list(hashes))
    membership_queries = []

    def query_group_membership(_instance, block_hashes, req_id, _group_id):
        query = tuple(block_hashes)
        membership_queries.append((req_id, query))
        if query == hashes[6:8]:
            # The latest 32-token window has a hole at its final block.
            return [((0,), b"initial")]
        if query == hashes[0:8]:
            # The preceding window [4:6] is complete, so the model can resume
            # at three dense blocks even though four were found by attention.
            return [((0, 1, 2, 3, 4, 5, 6), b"wide")]
        if query == hashes[4:6]:
            return [((0, 1), b"final")]
        raise AssertionError(f"unexpected membership query: {query!r}")

    scheduler._tp_shard_client.query_group_membership = query_group_membership

    def query(_instance, block_hashes, req_id, wait_for_full_prefix):
        assert req_id == "r1:dense-shrunk-3"
        assert wait_for_full_prefix is False
        assert tuple(block_hashes) == hashes[0:3]
        return ShardedQueryReady(num_hit_blocks=3, leases=(b"dense-shrunk",))

    scheduler._tp_shard_client.query = query
    scheduler._tp_shard_client.release = MagicMock(return_value=True)

    attached = scheduler._attach_sliding_group_queries(
        request,
        computed_blocks=0,
        full_hashes=list(hashes[0:4]),
        ready=ShardedQueryReady(num_hit_blocks=4, leases=(b"dense",)),
        req_id="r1",
    )

    assert attached.num_hit_blocks == 3
    assert attached.leases == (b"dense-shrunk",)
    assert attached.block_ranges_by_group == ((0, 3), (4, 6))
    assert attached.leases_by_group == ((b"dense-shrunk",), (b"final",))
    assert scheduler._load_block_ids_by_group(
        ((10, 11, 12), (0, 0, 0, 0, 24, 25)),
        0,
        3,
        block_ranges_by_group=attached.block_ranges_by_group,
    ) == ((10, 11, 12), (24, 25))
    with pytest.raises(RuntimeError, match="load block mismatch"):
        scheduler._load_block_ids_by_group(
            ((10, 11, 12), (0, 0, 0, 0, 24)),
            0,
            3,
            block_ranges_by_group=attached.block_ranges_by_group,
        )

    released = [args[0][0] for args in scheduler._tp_shard_client.release.call_args_list]
    assert sorted(released) == [(b"dense",), (b"initial",), (b"wide",)]


def test_sliding_groups_intersect_windows_after_boundary_shrinks():
    scheduler = _scheduler()
    scheduler._cache_groups = replace(
        _layout(),
        layer_names=(("full",), ("sliding",), ("sliding_other",)),
        sliding_window_group_indices=frozenset({1, 2}),
        group_sliding_windows=(None, 32, 32),
        storage_group_ids=(0, 1, 2),
        group_block_sizes=(32, 16, 16),
    )
    hashes = tuple(bytes([i]) for i in range(8))
    request = SimpleNamespace(request_id="r", num_tokens=128, block_hashes=list(hashes))

    def membership(_instance, keys, _req_id, group_id):
        available = {0, 1, 2, 3, 6, 7} if group_id == 1 else set(range(7))
        positions = tuple(i for i, key in enumerate(keys) if key[0] in available)
        return [(positions, bytes([group_id]) + b"".join(keys))]

    scheduler._tp_shard_client.query_group_membership = membership
    scheduler._tp_shard_client.query = MagicMock(
        return_value=ShardedQueryReady(2, (b"dense-shrunk",))
    )
    result = scheduler._attach_sliding_group_queries(
        request, 0, list(hashes[1::2]), ShardedQueryReady(4, (b"dense",)), "r"
    )
    # Group 1 serves boundaries 4 and 2, group 2 serves 3 and 2.
    assert result.num_hit_blocks == 2
    assert result.block_ranges_by_group == ((0, 2), (2, 4), (2, 4))


@pytest.mark.parametrize("extra_dense", [False, True], ids=["shared-sliding", "shared-dense"])
def test_shared_storage_uses_one_query_and_one_worker_load(extra_dense):
    scheduler = _scheduler()
    layout = replace(
        _layout(),
        layer_names=(("full",), ("sliding",), ("sliding_other",)),
        sliding_window_group_indices=frozenset({1}) if extra_dense else frozenset({1, 2}),
        group_sliding_windows=(None, 32, None if extra_dense else 32),
        storage_group_ids=(0, 1, 0 if extra_dense else 1),
        group_block_sizes=(32, 16, 32 if extra_dense else 16),
    )
    scheduler._cache_groups = layout
    scheduler._tp_shard_client.query_group_membership = MagicMock(
        return_value=[((0, 1), b"sliding")]
    )
    request = SimpleNamespace(num_tokens=128, block_hashes=[bytes([i]) for i in range(8)])
    result = scheduler._attach_sliding_group_queries(
        request, 0, [b"h"] * 4, ShardedQueryReady(4, (b"dense",)), "r"
    )
    scheduler._tp_shard_client.query_group_membership.assert_called_once()
    assert result.leases_by_group == (
        (b"dense",),
        (b"sliding",),
        (b"dense" if extra_dense else b"sliding",),
    )

    worker = WorkerConnector(scheduler._ctx)
    worker._cache_groups = layout
    worker._registered_layers = ["full", "sliding", "sliding_other"]
    worker._layer_to_group = layout.layer_to_group()
    scheduler._ctx.engine_client.load.return_value = (True, "")
    try:
        worker.start_load_kv(
            PegaConnectorMetadata(
                load_intents={
                    "r": LoadIntent(
                        block_ids_by_group=(
                            (10, 11, 12, 13),
                            (20, 21),
                            (30, 31, 32, 33) if extra_dense else (30, 31),
                        ),
                        leases=result.leases,
                        leases_by_group=result.leases_by_group,
                        num_tokens=128,
                    )
                }
            ),
            SimpleNamespace(),
        )
        assert scheduler._ctx.engine_client.load.call_args.args[5] == [
            (
                b"dense",
                [[10, 11, 12, 13], [None] * 4, [30, 31, 32, 33] if extra_dense else [None] * 4],
            ),
            (b"sliding", [[None, None], [20, 21], [None, None] if extra_dense else [30, 31]]),
        ]
    finally:
        worker._registered_layers = []
        worker.shutdown()

    scheduler._tp_shard_client.release = MagicMock(return_value=True)
    scheduler._release_query_probe(
        "r", SimpleNamespace(leases_by_group=result.leases_by_group, recurrent_hold=None)
    )
    assert scheduler._tp_shard_client.release.call_count == 2


@pytest.mark.parametrize("outcome", ["miss", "error"])
def test_sliding_query_cleans_all_leases_when_final_query_fails(outcome):
    scheduler = _scheduler()
    hashes = [bytes([i]) for i in range(8)]
    request = SimpleNamespace(num_tokens=128, block_hashes=hashes)
    final = [((0,), b"final-partial")] if outcome == "miss" else RuntimeError("query failed")
    scheduler._tp_shard_client.query_group_membership = MagicMock(
        side_effect=[
            [((0,), b"initial")],
            [(tuple(range(7)), b"wide")],
            final,
        ]
    )
    scheduler._tp_shard_client.query = MagicMock(
        return_value=ShardedQueryReady(3, (b"dense-shrunk",))
    )
    scheduler._tp_shard_client.release = MagicMock(return_value=True)
    if outcome == "error":
        with pytest.raises(RuntimeError, match="query failed"):
            scheduler._attach_sliding_group_queries(
                request, 0, hashes[1::2], ShardedQueryReady(4, (b"dense",)), "r"
            )
    else:
        result = scheduler._attach_sliding_group_queries(
            request, 0, hashes[1::2], ShardedQueryReady(4, (b"dense",)), "r"
        )
        assert result.num_hit_blocks == 0
    released = [args[0][0] for args in scheduler._tp_shard_client.release.call_args_list]
    expected = [(b"dense",), (b"dense-shrunk",), (b"initial",), (b"wide",)]
    if outcome == "miss":
        expected.append((b"final-partial",))
    assert sorted(released) == sorted(expected)


def test_dense_miss_skips_sliding_membership_queries():
    scheduler = _scheduler()
    scheduler._tp_shard_client.query_group_membership = MagicMock()
    result = scheduler._attach_sliding_group_queries(
        SimpleNamespace(), 0, [], ShardedQueryReady(0, (b"",)), "r"
    )
    assert result.num_hit_blocks == 0
    scheduler._tp_shard_client.query_group_membership.assert_not_called()
