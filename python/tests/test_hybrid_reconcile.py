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

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    CacheGroupLayout,
    reconcile_hybrid_hit,
)

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
