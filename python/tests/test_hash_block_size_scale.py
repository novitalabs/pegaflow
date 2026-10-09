"""Block hashes finer than the scheduler block must be re-keyed per block.

vLLM computes ``Request.block_hashes`` every ``hash_block_size`` tokens: the
scheduler block for a single cache group, but the GCD of the group block sizes
or ``--prefix-match-unit`` for hybrid models (K3 production: 128-token hashes
over 1536-token blocks). Each hash chains over its whole prefix, so the hash
closing a block is that block's key (vLLM's ``BlockHashListWithBlockSize``).

Consuming the fine list 1-to-1 with block ids filed block ``i`` under the hash
of the first ``(i + 1) * 128`` tokens, so two sessions sharing only a system
prompt served each other's whole conversation.
"""

from __future__ import annotations

# ruff: noqa: E402
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (
    ConnectorContext,
    RecurrentLoadHold,
    SaveIntent,
    derive_namespace,
)
from pegaflow.connector.scheduler import (
    SchedulerConnector,
    _QueryProbe,
    block_hashes_per_block,
)
from pegaflow.connector.tp_shards import ShardedQueryReady

VBS = 1536
HASH_BLOCK = 128
SCALE = VBS // HASH_BLOCK


def _hash(i: int) -> bytes:
    return f"fine-{i}".encode()


def _key(block: int) -> bytes:
    return _hash(block * SCALE + SCALE - 1)


def _scheduler(hash_block_size: int | None = HASH_BLOCK) -> SchedulerConnector:
    ctx = ConnectorContext(
        instance_id="i",
        namespace="n",
        block_size=VBS,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
        hash_block_size=hash_block_size,
    )
    return SchedulerConnector(ctx)


def _request(req_id: str, num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        block_hashes=[_hash(i) for i in range(num_tokens // HASH_BLOCK)],
    )


def _output(req_id: str, block_ids: list[int], num_tokens: int, offloads=None) -> SimpleNamespace:
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id, block_ids=(block_ids,), num_computed_tokens=0)
        ]
        if block_ids
        else [],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(), new_block_ids=[], num_computed_tokens=[]
        ),
        num_scheduled_tokens={req_id: num_tokens} if block_ids else {},
        preempted_req_ids=set(),
    )
    if offloads is not None:
        output.kv_connector_block_state = SimpleNamespace(
            block_ids={}, boundary_state_offloads=offloads
        )
    return output


def test_block_hashes_per_block_keeps_the_hash_closing_each_block():
    fine = [_hash(i) for i in range(1245)]  # 159364 tokens // 128
    per_block = block_hashes_per_block(fine, SCALE)
    assert len(per_block) == 103  # 159364 // 1536
    assert per_block[0] == _hash(11) and per_block[102] == _hash(102 * SCALE + 11)
    assert block_hashes_per_block(fine, 1) == tuple(fine)


def test_query_and_save_key_blocks_by_their_closing_hash():
    scheduler = _scheduler()
    req = _request("r1", 4 * VBS + 300)

    # 6444 prompt tokens = 50 full hash units; the prompt tail key is the
    # fine hash closing unit 50, covering 256 of the 300 tail tokens (the
    # 44 sub-unit tokens are recomputed by the consumer).
    assert scheduler._build_query(req, 1) == (
        tuple(_key(i) for i in range(1, 4)) + (_hash(49),),
        256,
    )

    scheduler.update_state_after_alloc(req, None, 0)
    intent = scheduler.build_connector_meta(_output("r1", [10, 11, 12, 13, 14], req.num_tokens))
    assert intent.save_intents["r1"] == SaveIntent(
        block_ids_by_group=((10, 11, 12, 13, 14),),
        block_hashes=tuple(_key(i) for i in range(4)) + (_hash(49),),
    )


def test_boundary_offload_uses_the_hash_closing_the_boundary_block():
    scheduler = _scheduler()
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        recurrent_group_indices=frozenset({1}),
        scratch_group_indices=frozenset(),
    )
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_id=i) for i in range(64)],
        touch=lambda blocks: None,
        free_blocks=lambda blocks: None,
    )
    scheduler._requests["r1"] = _request("r1", 6 * VBS)

    metadata = scheduler.build_connector_meta(_output("r1", [], 0, {"r1": [(1, 21, 3 * VBS)]}))
    assert metadata.boundary_save_intents == {
        0: SaveIntent(block_ids_by_group=((0,), (21,)), block_hashes=(_key(2),))
    }


def test_more_keys_than_full_blocks_is_rejected():
    # Fine hashes consumed at scale 1 would alias requests; refuse instead.
    with pytest.raises(RuntimeError, match="finer"):
        _scheduler(hash_block_size=None)._build_query(_request("r1", 4 * VBS), 0)


def test_hash_past_a_popped_last_token_is_dropped():
    # NIXL/Mooncake pop the prefiller's last prompt token for hybrid models
    # after vLLM hashed the full prompt: a 30-block prompt keeps 360 hashes
    # over 29 full blocks. The hash closing the 30th block spans a token the
    # request no longer has and must not become a key (K3 production: group
    # restarts on every 1536-aligned prompt).
    req = _request("r1", 30 * VBS)
    req.num_tokens = req.num_prompt_tokens = 30 * VBS - 1
    keys, tail_tokens = _scheduler()._build_query(req, 0)
    # 45952 = 359 hash units = 29 full blocks + a 1408-token covered tail.
    assert keys == tuple(_key(i) for i in range(29)) + (_hash(358),)
    assert tail_tokens == 1408

    # An unaligned prompt leaves no stale hash and is unaffected.
    req = _request("r1", 30 * VBS + 5)
    req.num_tokens = req.num_prompt_tokens = 30 * VBS + 4
    assert _scheduler()._build_query(req, 0)[0] == tuple(_key(i) for i in range(30))


def test_fine_tail_excludes_sub_unit_remainder():
    # A 130-token tail with a 128-token hash unit: the key (closing hash of
    # unit 49) covers only the first 128 tail tokens; a same-prefix peer may
    # diverge inside the trailing sub-unit span, so it is never claimed.
    req = _request("r1", 4 * VBS + 130)
    keys, tail_tokens = _scheduler()._build_query(req, 0)
    assert keys == tuple(_key(i) for i in range(4)) + (_hash(48),)
    assert tail_tokens == 128


def test_fine_tail_needs_no_lora_salt_mm_exclusion():
    # The derived scheme bails on salted/LoRA/multimodal requests because its
    # key carries no extra_keys; the fine key IS the vLLM cache identity.
    req = _request("r1", 4 * VBS + 300)
    req.lora_request = object()
    req.cache_salt = "tenant-a"
    req.mm_features = [object()]
    keys, tail_tokens = _scheduler()._build_query(req, 0)
    assert keys[-1] == _hash(49)
    assert tail_tokens == 256


def test_fine_tail_defers_when_hashes_lag():
    # Fine hashes arrive per unit; if the prompt outruns them, skip the tail
    # this step instead of keying a stale or out-of-range hash.
    req = _request("r1", 4 * VBS + 300)
    req.block_hashes = req.block_hashes[:40]
    keys, tail_tokens = _scheduler()._build_query(req, 0)
    assert keys == tuple(_key(i) for i in range(3))
    assert tail_tokens == 0


def test_fine_tail_stays_enabled_with_hma(monkeypatch):
    # With fine hashing, vLLM materializes the recurrent state at the
    # prompt's last hash boundary and hands it off (boundary_state_offloads /
    # register_finished_partial_tail), so the fine tail works for HMA too:
    # the attention tail and the mamba state share the same fine key.
    import pegaflow.connector.scheduler as scheduler_mod

    monkeypatch.setattr(
        scheduler_mod.CacheGroupLayout,
        "from_config",
        staticmethod(
            lambda cfg: SimpleNamespace(
                group_count=2,
                hash_group_index=0,
                has_recurrent_state=True,
                recurrent_group_indices=frozenset({1}),
                scratch_group_indices=frozenset(),
            )
        ),
    )
    scheduler = _scheduler()
    assert scheduler._fine_tail is True
    assert scheduler._tail_save_enabled is True
    assert scheduler._tail_load_enabled is True


def test_hash_block_size_isolates_namespace():
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/data/models/Kimi-K3",
            dtype="bfloat16",
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 576,
            get_total_num_hidden_layers=lambda: 61,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        additional_config={},
    )
    assert derive_namespace(cfg, 8, hash_block_size=VBS) != derive_namespace(
        cfg, 8, hash_block_size=HASH_BLOCK
    )


def _hma_scheduler() -> SchedulerConnector:
    scheduler = _scheduler()
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        recurrent_group_indices=frozenset({1}),
        scratch_group_indices=frozenset(),
    )
    return scheduler


def test_hma_tail_save_zeroes_recurrent_rows():
    # The attention tail block saves positionally; the recurrent state at the
    # same boundary arrives through vLLM's boundary hand-off under the same
    # key, so the recurrent row must stay null (positional reads race the
    # in-place state updates). The short/unreliable recurrent mirror must not
    # stall the save either.
    scheduler = _hma_scheduler()
    req = _request("r1", 4 * VBS + 300)  # 50 units; tail key covers 256 tokens
    scheduler.update_state_after_alloc(req, None, 0)
    scheduler._allocated_blocks["r1"] = [[10, 11, 12, 13, 14], [21]]
    scheduler._scheduled_tokens["r1"] = req.num_tokens

    intent = scheduler._consume_save_intent("r1", req.num_tokens)

    assert intent == SaveIntent(
        block_ids_by_group=((10, 11, 12, 13, 14), (0, 0, 0, 0, 0)),
        block_hashes=tuple(_key(i) for i in range(4)) + (_hash(49),),
    )


def _hma_probe(
    query_hashes: tuple[bytes, ...],
    tail_tokens: int,
    hit_blocks: int,
    usable: tuple[int, ...],
    checkpoint: int,
) -> _QueryProbe:
    probe = _QueryProbe(computed_blocks=0, query_hashes=query_hashes, tail_tokens=tail_tokens)
    probe.mark_ready(
        ShardedQueryReady(
            hit_blocks,
            (b"L",),
            recurrent_hold=RecurrentLoadHold(
                leases=((b"M",),),
                hit_positions=(usable,),
                checkpoint=checkpoint,
            ),
            usable_positions=usable,
        )
    )
    return probe


def test_hma_lookup_reports_tail_boundary_hit():
    # Prompt 4472 = 2 full blocks + a 1400-token tail; the tail key (closing
    # unit 34) covers 1280 tokens and every recurrent group cached its state
    # there, so the hit extends to the tail boundary, not the block boundary.
    scheduler = _hma_scheduler()
    query = (_key(0), _key(1), _hash(33))
    probe = _hma_probe(query, tail_tokens=1280, hit_blocks=3, usable=(0, 1, 2), checkpoint=2)

    hit_tokens, load_async = scheduler._finish_cache_lookup(
        req_id="r1", num_tokens=4472, probe=probe, lookup_us=None, reused=False
    )

    assert (hit_tokens, load_async) == (2 * VBS + 1280, True)
    assert probe.hit_blocks == 3
    assert probe.recurrent_hold is not None and probe.recurrent_hold.checkpoint == 2
    assert scheduler._external_matched_tokens["r1"] == 2 * VBS + 1280
    assert scheduler._external_matched_blocks["r1"] == 3


def test_hma_lookup_falls_back_when_last_token_clamp_drops_the_tail():
    # Prompt ends exactly at the tail boundary (4352): vLLM must recompute
    # the final token, and the hash-unit floor then drops the whole tail
    # unit, so the hit falls back to the last full-block checkpoint.
    scheduler = _hma_scheduler()
    query = (_key(0), _key(1), _hash(33))
    probe = _hma_probe(query, tail_tokens=1280, hit_blocks=3, usable=(0, 1, 2), checkpoint=2)

    hit_tokens, _ = scheduler._finish_cache_lookup(
        req_id="r1", num_tokens=4352, probe=probe, lookup_us=None, reused=False
    )

    assert hit_tokens == 2 * VBS
    assert probe.hit_blocks == 2
    assert probe.recurrent_hold is not None and probe.recurrent_hold.checkpoint == 1
    assert scheduler._external_matched_tokens["r1"] == 2 * VBS
    assert scheduler._external_matched_blocks["r1"] == 2
