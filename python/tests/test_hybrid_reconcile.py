"""Hybrid (HMA) prefix reconcile: attention prefix + recurrent membership.

Pure-function contract covering the scheduler-side hit derivation:

- a recurrent checkpoint at query position ``k`` resumes ``k + 1`` blocks
  (the state block ends with block ``k``'s tokens, vLLM convention);
- checkpoints outside the attention prefix are unusable;
- every recurrent group AND every TP shard must hold the boundary;
- ``usable`` positions drive re-derivation when the token budget shrinks
  the reconciled hit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    CacheGroupLayout,
    ConnectorContext,
    LocalSaveRef,
    PegaWorkerMetadata,
    reconcile_hybrid_hit,
)
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402
from pegaflow.connector.tp_shards import ShardedQueryReady  # noqa: E402

from .test_cache_group_layout import (  # noqa: E402
    _config,
    _full_attention,
    _group,
    _mamba,
)


def _hits(*per_shard_positions: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """One recurrent group's per-shard hit positions."""
    return tuple(tuple(p) for p in per_shard_positions)


class TestReconcileHybridHit:
    def test_rightmost_checkpoint_within_prefix(self):
        # attn=111, recur=001 -> hit = 2 + 1 = 3
        hit, checkpoint, usable = reconcile_hybrid_hit(3, (_hits((2,)),))
        assert hit == 3
        assert checkpoint == 2
        assert usable == frozenset({2})

    def test_checkpoint_beyond_attention_prefix_is_unusable(self):
        # The checkpoint at position 2 outruns the 2-block attention prefix.
        hit, checkpoint, usable = reconcile_hybrid_hit(2, (_hits((2,)),))
        assert hit == 0
        assert checkpoint is None
        assert usable == frozenset()

    def test_no_checkpoint_means_recompute(self):
        # Attention alone cannot resume a recurrent model.
        assert reconcile_hybrid_hit(3, (_hits(()),)) == (0, None, frozenset())

    def test_shards_must_agree_on_the_boundary(self):
        # shard0 has {1, 2}, shard1 has {2, 3}; prefix is 4 blocks.
        hit, checkpoint, usable = reconcile_hybrid_hit(4, (_hits((1, 2), (2, 3)),))
        assert hit == 3
        assert checkpoint == 2
        assert usable == frozenset({2})

    def test_all_recurrent_groups_must_hold_the_boundary(self):
        # Two recurrent groups (e.g. mamba + gated-deltanet stacks).
        hit, checkpoint, usable = reconcile_hybrid_hit(3, (_hits((0, 2)), _hits((1, 2))))
        assert hit == 3
        assert checkpoint == 2
        assert usable == frozenset({2})

        # Group 2 lost its position-2 checkpoint: fall back to nothing.
        hit, checkpoint, usable = reconcile_hybrid_hit(3, (_hits((0, 2)), _hits((1,))))
        assert hit == 0
        assert checkpoint is None

    def test_mid_prefix_gap_is_fine_below_rightmost(self):
        # recur=101 with attn prefix 3: the missing middle checkpoint does
        # not matter; rightmost hit at 2 still resumes 3 blocks.
        hit, checkpoint, usable = reconcile_hybrid_hit(3, (_hits((0, 2)),))
        assert hit == 3
        assert checkpoint == 2
        assert usable == frozenset({0, 2})

    def test_zero_attention_prefix_means_no_hit(self):
        assert reconcile_hybrid_hit(0, (_hits((0,)),)) == (0, None, frozenset())

    def test_earlier_boundary_survives_for_budget_rederivation(self):
        # usable carries every legal boundary so a token-budget clamp can
        # re-derive hit=2 from checkpoint 1 after 3 was trimmed away.
        _, checkpoint, usable = reconcile_hybrid_hit(4, (_hits((1, 3)),))
        assert checkpoint == 3
        assert usable == frozenset({1, 3})


class TestStorageGroupIds:
    def test_attention_first_layout(self):
        config = _config(
            _group("attn", _full_attention()),
            _group("mamba", _mamba()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0, 1)
        assert layout.storage_group_of(1) == 1

    def test_recurrent_group_can_come_first(self):
        # vLLM group order is not guaranteed; attention must always map to
        # storage group 0 regardless of connector position.
        config = _config(
            _group("mamba", _mamba()),
            _group("attn", _full_attention()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.hash_group_index == 1
        assert layout.storage_group_ids == (1, 0)

    def test_single_group_defaults_to_zero(self):
        config = _config(_group("attn", _full_attention()))
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0,)

    def test_multiple_recurrent_groups_get_dense_ids(self):
        config = _config(
            _group("attn", _full_attention()),
            _group("mamba_a", _mamba()),
            _group("mamba_b", _mamba()),
        )
        layout = CacheGroupLayout.from_config(config)
        assert layout.storage_group_ids == (0, 1, 2)
        assert layout.recurrent_group_indices == frozenset({1, 2})


def _local_scheduler(tp_size: int = 1) -> SchedulerConnector:
    config = _config(
        _group("attention", _full_attention()),
        _group("recurrent", _mamba(page_size_bytes=1024)),
    )
    context = ConnectorContext(
        instance_id="test",
        namespace="ns",
        block_size=16,
        tp_size=tp_size,
        world_size=tp_size,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
        num_linear_state_cache_slots=2,
    )
    return SchedulerConnector(context, kv_cache_config=config)


def _commit_hash(scheduler: SchedulerConnector, block_hash: bytes):
    assert scheduler._linear_state_cache is not None
    ref = scheduler._linear_state_cache.reserve(block_hash)
    assert ref is not None
    scheduler._linear_state_cache.commit(ref)
    return ref


def test_local_recurrent_reconcile_uses_rightmost_checkpoint_inside_remote_prefix():
    scheduler = _local_scheduler()
    hashes = [b"h0", b"h1", b"h2"]
    _commit_hash(scheduler, hashes[0])
    rightmost = _commit_hash(scheduler, hashes[2])
    scheduler._tp_shard_client = MagicMock()

    result = scheduler._reconcile_hybrid(
        hashes,
        ShardedQueryReady(3, (b"remote-lease",)),
        "req",
    )

    assert result.num_hit_blocks == 3
    assert result.local_recurrent == rightmost
    assert result.usable_positions == (0, 2)
    scheduler._tp_shard_client.query_group_membership.assert_not_called()


def test_remote_attention_hit_without_local_recurrent_checkpoint_is_a_miss():
    scheduler = _local_scheduler()
    scheduler._tp_shard_client = MagicMock()

    result = scheduler._reconcile_hybrid(
        [b"h0", b"h1"],
        ShardedQueryReady(2, (b"remote-lease",)),
        "req",
    )

    assert result.num_hit_blocks == 0
    scheduler._tp_shard_client.release.assert_called_once_with((b"remote-lease",), "req")


def test_tp2_local_save_commits_only_after_two_matching_acks():
    scheduler = _local_scheduler(tp_size=2)
    intent = scheduler._wrap_local_recurrent_save(
        "req",
        SimpleNamespace(
            block_ids_by_group=((11,), (21,)),
            block_hashes=(b"checkpoint",),
        ),
    )
    save = intent.local_recurrent_saves[0]
    ref = LocalSaveRef.from_save("req", save)
    assert scheduler._linear_state_cache is not None

    scheduler._apply_local_save_acks(PegaWorkerMetadata(succeeded={ref: 1}))
    assert scheduler._linear_state_cache.lookup(b"checkpoint") is None

    scheduler._apply_local_save_acks(PegaWorkerMetadata(succeeded={ref: 1}))
    assert scheduler._linear_state_cache.lookup(b"checkpoint") == save.target


def test_local_save_failure_ack_cancels_exact_reservation():
    scheduler = _local_scheduler()
    intent = scheduler._wrap_local_recurrent_save(
        "req",
        SimpleNamespace(
            block_ids_by_group=((11,), (21,)),
            block_hashes=(b"checkpoint",),
        ),
    )
    save = intent.local_recurrent_saves[0]
    ref = LocalSaveRef.from_save("req", save)

    scheduler._apply_local_save_acks(PegaWorkerMetadata(failed={ref: 1}))

    assert ref not in scheduler._pending_local_saves
    assert scheduler._local_saves_by_req == {}


def test_metadata_construction_failure_rolls_back_reservation_and_retries(monkeypatch):
    scheduler = _local_scheduler()
    raw = SimpleNamespace(
        block_ids_by_group=((11,), (21,)),
        block_hashes=(b"checkpoint",),
    )
    scheduler._deferred_save_intents["req"] = raw
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        preempted_req_ids=set(),
    )
    original = scheduler._wrap_local_recurrent_save

    def reserve_then_fail(req_id, intent):
        original(req_id, intent)
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(scheduler, "_wrap_local_recurrent_save", reserve_then_fail)
    with pytest.raises(RuntimeError, match="metadata failed"):
        scheduler.build_connector_meta(output)

    assert scheduler._pending_local_saves == {}
    assert scheduler._local_saves_by_req == {}
    assert scheduler._deferred_save_intents == {"req": raw}

    monkeypatch.setattr(scheduler, "_wrap_local_recurrent_save", original)
    metadata = scheduler.build_connector_meta(output)
    assert len(metadata.ready_save_intents["req"].local_recurrent_saves) == 1


def test_multiple_recurrent_groups_share_one_compound_reservation():
    scheduler = _local_scheduler()
    scheduler._cache_groups = SimpleNamespace(
        recurrent_group_indices=frozenset({1, 2}),
    )

    intent = scheduler._wrap_local_recurrent_save(
        "req",
        SimpleNamespace(
            block_ids_by_group=((11,), (21,), (31,)),
            block_hashes=(b"checkpoint",),
        ),
    )

    assert len(intent.local_recurrent_saves) == 1
    assert intent.local_recurrent_saves[0].source_block_ids == ((1, 21), (2, 31))
    assert intent.block_ids_by_group == ((11,), (0,), (0,))
    assert len(scheduler._pending_local_saves) == 1
