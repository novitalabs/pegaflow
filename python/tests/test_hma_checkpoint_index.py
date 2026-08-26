"""Regression: the final HMA recurrent checkpoint must be a boundary state.

A hybrid request stores exactly one recurrent checkpoint, at request_finished.
Getting its *identity* wrong is silent: a later turn resumes from a state that
does not match the hash it was filed under, folding some tokens into the
recurrent state twice. Nothing crashes and the text stays plausible.

These tests build the recurrent block table with vLLM's own align-mode
formulas rather than a hand-written shape, and encode provenance in the block
ids so a failure says *which* state was saved:

    9000 + k -> block holding the running state after k tokens
    7000 + i -> speculative block i (draft state, never a boundary)

vLLM align mode, S = num_speculative_blocks
(single_type_kv_cache_manager.MambaManager.allocate_new_blocks +
worker/mamba_utils.preprocess_mamba):

    num_required_blocks = cdiv(written, vbs) + S
    null blocks         = indices [0, cdiv(written, vbs) - 1)
    running state block = index cdiv(written, vbs) - 1
    trailing S blocks   = speculative
    table length        = cdiv(written, vbs) + S

The connector's convention (RecurrentLoadHold): a state filed under hash index
j means "state after (j + 1) * vbs tokens".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext  # noqa: E402
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402

VBS = 16


def _hash(i: int) -> bytes:
    return f"hash-{i}".encode()


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _recurrent_table(written: int, num_speculative_blocks: int) -> list[int]:
    """The recurrent block table vLLM align mode would present."""
    state_idx = _cdiv(written, VBS) - 1
    table = [0] * state_idx
    table.append(9000 + written)
    table.extend(7000 + i for i in range(num_speculative_blocks))
    return table


def _make_scheduler(num_hashes: int, committed_boundary: int | None):
    """Scheduler whose prefix cache has committed `committed_boundary`.

    `committed_boundary` is a hash index; vLLM commits a recurrent block there
    only when a scheduler step ended at exactly (index + 1) * VBS tokens.
    """
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
    )
    scheduler = SchedulerConnector(ctx)
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        recurrent_group_indices=frozenset({1}),
    )
    scheduler._block_hashes["r1"] = tuple(_hash(i) for i in range(num_hashes))
    scheduler._block_index_offsets["r1"] = 0
    scheduler._next_stored_block_idx["r1"] = 0

    committed_hash = _hash(committed_boundary) if committed_boundary is not None else None

    def get_cached(block_hash, group_ids):
        blocks = []
        for group_id in group_ids:
            if group_id == 1:
                if committed_hash is None or block_hash != committed_hash:
                    return None
                # The committed boundary state: 555 marks "correctly keyed".
                blocks.append(SimpleNamespace(block_id=555))
            else:
                blocks.append(SimpleNamespace(block_id=100))
        return blocks

    scheduler._get_local_cached_blocks = get_cached
    return scheduler


@pytest.mark.parametrize(
    ("written", "num_speculative_blocks"),
    [
        pytest.param(645, 0, id="mid_block_no_spec"),
        pytest.param(645, 1, id="mid_block_spec1"),
        pytest.param(645, 2, id="mid_block_spec2"),
        pytest.param(640, 1, id="aligned_spec1"),
        pytest.param(640, 2, id="aligned_spec2"),
    ],
)
def test_checkpoint_is_committed_boundary_never_running_or_draft_state(
    written: int, num_speculative_blocks: int
):
    """The saved recurrent id must be the committed boundary block.

    Reading the live block table instead yields either the running state for a
    mid-block token count (keyed off by `written % VBS` tokens) or, when
    `written` is block-aligned, a speculative draft block.
    """
    num_hashes = _cdiv(written, VBS)
    saveable = written // VBS
    committed_boundary = saveable - 1

    scheduler = _make_scheduler(num_hashes, committed_boundary)
    attention_table = list(range(100, 100 + num_hashes + num_speculative_blocks))
    recurrent_table = _recurrent_table(written, num_speculative_blocks)

    intent = scheduler._consume_finished_hma_save("r1", (attention_table, recurrent_table), written)

    assert intent is not None
    recurrent_row = intent.block_ids_by_group[1]
    saved = [block_id for block_id in recurrent_row if block_id != 0]

    assert saved == [555], (
        f"expected the committed boundary state, got {saved} "
        f"(9000+k = running state after k tokens, 7000+i = speculative block)"
    )

    # And it must be filed under the hash naming that exact boundary.
    position = recurrent_row.index(555)
    assert intent.block_hashes[position] == _hash(committed_boundary)


def test_no_committed_boundary_saves_no_recurrent_state():
    """Better to store nothing than a state under the wrong hash."""
    written = 645
    num_hashes = _cdiv(written, VBS)
    scheduler = _make_scheduler(num_hashes, committed_boundary=None)

    intent = scheduler._consume_finished_hma_save(
        "r1",
        (list(range(100, 100 + num_hashes)), _recurrent_table(written, 1)),
        written,
    )

    assert intent is not None
    assert set(intent.block_ids_by_group[1]) == {0}


def test_aligned_written_without_spec_blocks_does_not_index_past_table():
    """`written` on a boundary with S=0 leaves no block past the state block.

    Indexing at `written // vbs` would run off the end of the table.
    """
    written = 640
    num_hashes = _cdiv(written, VBS)
    saveable = written // VBS
    scheduler = _make_scheduler(num_hashes, committed_boundary=saveable - 1)
    recurrent_table = _recurrent_table(written, 0)

    assert len(recurrent_table) == saveable  # no entry at index `saveable`

    intent = scheduler._consume_finished_hma_save(
        "r1", (list(range(100, 100 + num_hashes)), recurrent_table), written
    )

    assert intent is not None
    assert [b for b in intent.block_ids_by_group[1] if b != 0] == [555]


def test_finished_save_skips_when_prefix_already_stored():
    """A later turn that does not cross a new block boundary must not re-save."""
    written = 645
    num_hashes = _cdiv(written, VBS)
    saveable = written // VBS
    scheduler = _make_scheduler(num_hashes, committed_boundary=saveable - 1)
    scheduler._next_stored_block_idx["r1"] = saveable

    intent = scheduler._consume_finished_hma_save(
        "r1",
        (list(range(100, 100 + num_hashes)), _recurrent_table(written, 1)),
        written,
    )

    assert intent is None
    assert scheduler._consume_full_block_saves("r1") is None


def test_finished_save_after_prefix_only_files_the_new_boundary():
    """After an external hit of block 0, the final save files only hash[1]."""
    written = 32
    num_hashes = _cdiv(written, VBS)
    scheduler = _make_scheduler(num_hashes, committed_boundary=1)
    scheduler._next_stored_block_idx["r1"] = 1
    scheduler._block_index_offsets["r1"] = 1

    intent = scheduler._consume_finished_hma_save(
        "r1",
        (list(range(100, 100 + num_hashes + 2)), _recurrent_table(written, 2)),
        written,
    )

    assert intent is not None
    assert intent.block_hashes == (_hash(1),)
    assert intent.block_ids_by_group[1] == (555,)
