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

from pegaflow.connector.common import ConnectorContext, SaveIntent, derive_namespace
from pegaflow.connector.scheduler import SchedulerConnector, block_hashes_per_block

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

    assert scheduler._build_query(req, 1) == (tuple(_key(i) for i in range(1, 4)), 0)

    scheduler.update_state_after_alloc(req, None, 0)
    intent = scheduler.build_connector_meta(_output("r1", [10, 11, 12, 13, 14], req.num_tokens))
    assert intent.save_intents["r1"] == SaveIntent(
        block_ids_by_group=((10, 11, 12, 13),),
        block_hashes=tuple(_key(i) for i in range(4)),
    )


def test_boundary_offload_uses_the_hash_closing_the_boundary_block():
    scheduler = _scheduler()
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        recurrent_group_indices=frozenset({1}),
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
