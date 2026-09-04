"""Full-block save intents across a preempt/resume cycle.

vLLM frees a preempted request's blocks and resets its `num_computed_tokens`;
on resume the request re-enters the waiting queue, gets a fresh external-hit
lookup and a fresh block table that (under chunked prefill) covers only the
resume chunk. The connector's per-request bookkeeping from the first life is
stale by then. Before the fix, `_consume_full_block_saves` bounded the save
window by `base_block_idx + len(table)` — double-counting the external prefix
already inside the table — so a resumed request sliced block IDs past the
end of its table (silently short) while the prompt hashes, known up front,
were not: `save block/hash count mismatch ... blocks=1 hashes=10` on every
attention layer, which killed the save thread and leaked every later
request's blocks.
"""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext, PegaConnectorMode
from pegaflow.connector.scheduler import SchedulerConnector

VBS = 16


def _hash(i: int) -> bytes:
    return hashlib.sha256(f"block_{i}".encode()).digest()


def _make_scheduler(mode: PegaConnectorMode | None = None) -> SchedulerConnector:
    kwargs = {} if mode is None else {"mode": mode}
    ctx = ConnectorContext(
        instance_id="test",
        namespace="ns",
        block_size=VBS,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
        **kwargs,
    )  # type: ignore[arg-type]
    return SchedulerConnector(ctx)


def _make_request(req_id: str, num_full_blocks: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=num_full_blocks * VBS,
        block_hashes=[_hash(i) for i in range(num_full_blocks)],
    )


def _step(
    scheduler: SchedulerConnector,
    req_id: str,
    *,
    block_ids: list[int],
    num_tokens: int,
    num_computed_tokens: int,
    new: bool = False,
    resumed: bool = False,
):
    if new:
        scheduled_new_reqs = [
            SimpleNamespace(
                req_id=req_id,
                block_ids=(block_ids,),
                num_computed_tokens=num_computed_tokens,
            )
        ]
        cached = SimpleNamespace(
            req_ids=[], resumed_req_ids=set(), new_block_ids=[], num_computed_tokens=[]
        )
    else:
        scheduled_new_reqs = []
        cached = SimpleNamespace(
            req_ids=[req_id],
            resumed_req_ids={req_id} if resumed else set(),
            new_block_ids=[(block_ids,)],
            num_computed_tokens=[num_computed_tokens],
        )
    output = SimpleNamespace(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={req_id: num_tokens},
        preempted_req_ids=set(),
    )
    return scheduler.build_connector_meta(output).save_intents.get(req_id)


def _assert_consistent(intent) -> None:
    assert intent is not None
    for group in intent.block_ids_by_group:
        assert len(group) == len(intent.block_hashes)


def test_resumed_external_hit_request_keeps_block_and_hash_counts_aligned():
    """The production shape: 19-block prompt, 9 blocks loaded from PegaFlow,
    preempted after a sub-block first chunk, resumed with a chunk that the
    stale accumulator places well past the rebuilt table."""
    scheduler = _make_scheduler()
    req = _make_request("r1", 19)
    hashes = tuple(req.block_hashes)

    # Life 1: external hit of 9 blocks, first chunk shorter than one block.
    scheduler._external_matched_blocks["r1"] = 9
    scheduler.update_state_after_alloc(req, None, 0)
    assert (
        _step(
            scheduler,
            "r1",
            block_ids=list(range(100, 110)),
            num_tokens=VBS // 2,
            num_computed_tokens=9 * VBS,
            new=True,
        )
        is None
    )

    # Preempted, resumed: same external hit, table rebuilt for a 9.5-block chunk.
    scheduler._external_matched_blocks["r1"] = 9
    resumed_table = list(range(200, 210))
    intent = _step(
        scheduler,
        "r1",
        block_ids=resumed_table,
        num_tokens=9 * VBS + VBS // 2,
        num_computed_tokens=9 * VBS,
        resumed=True,
    )

    _assert_consistent(intent)
    # Global indices 9..18 are the rebuilt table's own entries; the block
    # holding the half-written chunk tail is not saved.
    assert intent.block_ids_by_group == (tuple(resumed_table[9:10]),)
    assert intent.block_hashes == hashes[9:10]
    assert scheduler._next_stored_block_idx["r1"] == 10


def test_resume_does_not_save_partially_recomputed_block():
    """Save-only mode carries `num_computed_tokens + scheduled` forward with
    max(); after preemption that watermark is ahead of what the resume chunk
    has actually recomputed."""
    scheduler = _make_scheduler(PegaConnectorMode.SAVE_ONLY)
    req = _make_request("r1", 4)
    hashes = tuple(req.block_hashes)
    scheduler.update_state_after_alloc(req, None, 0)

    # Life 1 wrote every block but nothing was consumed (say the save was
    # gated), then the request was preempted.
    scheduler._scheduled_tokens["r1"] = 4 * VBS

    # Resume recomputes from scratch, this chunk covers 2.5 blocks.
    intent = _step(
        scheduler,
        "r1",
        block_ids=[10, 11, 12],
        num_tokens=2 * VBS + VBS // 2,
        num_computed_tokens=0,
        resumed=True,
    )

    _assert_consistent(intent)
    assert intent.block_ids_by_group == ((10, 11),)
    assert intent.block_hashes == hashes[:2]


def test_resume_rebases_on_the_refreshed_external_hit():
    """Blocks stored in the first life count as external hits on resume; the
    save window restarts after them, never re-saving or skipping."""
    scheduler = _make_scheduler()
    req = _make_request("r1", 8)
    hashes = tuple(req.block_hashes)

    scheduler._external_matched_blocks["r1"] = 2
    scheduler.update_state_after_alloc(req, None, 0)
    first = _step(
        scheduler,
        "r1",
        block_ids=list(range(100, 106)),
        num_tokens=4 * VBS,
        num_computed_tokens=2 * VBS,
        new=True,
    )
    _assert_consistent(first)
    assert first.block_hashes == hashes[2:6]
    assert first.block_ids_by_group == ((102, 103, 104, 105),)

    # Resume finds blocks 0..5 in PegaFlow now and loads them all.
    scheduler._external_matched_blocks["r1"] = 6
    second = _step(
        scheduler,
        "r1",
        block_ids=list(range(200, 209)),
        num_tokens=2 * VBS + 1,
        num_computed_tokens=6 * VBS,
        resumed=True,
    )
    _assert_consistent(second)
    assert second.block_hashes == hashes[6:8]
    assert second.block_ids_by_group == ((206, 207),)
    assert scheduler._block_index_offsets["r1"] == 6


@pytest.mark.parametrize("scheduled_tokens", [3 * VBS, 100 * VBS])
def test_full_block_saves_never_exceed_the_mirrored_table(scheduled_tokens: int):
    """Direct check of the bound: whatever the accumulator says, the window
    ends at the table, and block IDs and hashes always pair up."""
    scheduler = _make_scheduler()
    hashes = tuple(_hash(i) for i in range(12))
    scheduler._block_hashes["r1"] = hashes
    scheduler._block_index_offsets["r1"] = 6
    scheduler._next_stored_block_idx["r1"] = 6
    scheduler._scheduled_tokens["r1"] = scheduled_tokens
    scheduler._allocated_blocks["r1"] = [[100, 101, 102, 103, 104, 105, 200, 201]]

    intent = scheduler._consume_full_block_saves("r1")

    _assert_consistent(intent)
    assert intent.block_ids_by_group == ((200, 201),)
    assert intent.block_hashes == hashes[6:8]
