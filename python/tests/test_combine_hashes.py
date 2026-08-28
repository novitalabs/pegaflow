"""Unit tests for scheduler block size logic in ConnectorContext."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MambaSpec,
)

from pegaflow.connector.common import (  # noqa: E402
    ConnectorContext,
    PegaConnectorMetadata,
    PegaConnectorMode,
    SaveIntent,
)
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402
from pegaflow.pegaflow import QueryLoading, QueryReady  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash(i: int) -> bytes:
    """Deterministic 32-byte hash for integer *i*."""
    return hashlib.sha256(f"block_{i}".encode()).digest()


def _make_ctx(
    block_size: int = 16,
    dcp_world_size: int = 1,
    **kwargs,
) -> ConnectorContext:
    """Create a ConnectorContext with minimal required fields."""
    defaults = {
        "instance_id": "test",
        "namespace": "ns",
        "block_size": block_size,
        "tp_size": 1,
        "world_size": 1,
        "tp_rank": 0,
        "device_id": 0,
        "engine_client": MagicMock(),
        "state_manager": MagicMock(),
        "is_mla": False,
        "dcp_world_size": dcp_world_size,
        "dcp_rank": 0,
    }
    defaults.update(kwargs)
    return ConnectorContext(**defaults)  # type: ignore[arg-type]


def _make_recurrent_scheduler(committed_recurrent: frozenset[int] | None = None):
    """Two-group HMA scheduler over a 2-block request.

    ``committed_recurrent`` is the set of hash indices whose *recurrent* block
    vLLM has committed to its prefix cache. In align mode that is only the
    boundaries where a scheduler step ended, not every full block, so the
    lookup must be group-aware. Defaults to both blocks.
    """
    if committed_recurrent is None:
        committed_recurrent = frozenset({0, 1})
    scheduler = SchedulerConnector(_make_ctx())
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        recurrent_group_indices=frozenset({1}),
    )
    scheduler._block_hashes["r1"] = (_hash(0), _hash(1))
    scheduler._allocated_blocks["r1"] = [[11, 12], [21, 22]]
    scheduler._block_index_offsets["r1"] = 0
    scheduler._next_stored_block_idx["r1"] = 0
    attention_ids = {_hash(0): 11, _hash(1): 12}
    recurrent_ids = {_hash(0): 21, _hash(1): 22}
    hash_index = {_hash(0): 0, _hash(1): 1}

    def get_cached(block_hash, group_ids):
        if block_hash not in hash_index:
            return None
        blocks = []
        for group_id in group_ids:
            if group_id == 1:
                if hash_index[block_hash] not in committed_recurrent:
                    return None
                blocks.append(SimpleNamespace(block_id=recurrent_ids[block_hash]))
            else:
                blocks.append(SimpleNamespace(block_id=attention_ids[block_hash]))
        return blocks

    scheduler._get_local_cached_blocks = get_cached
    return scheduler


def test_recurrent_mid_request_saves_are_skipped():
    scheduler = _make_recurrent_scheduler()

    assert scheduler._consume_full_block_saves("r1") is None
    assert scheduler._next_stored_block_idx["r1"] == 0


def test_recurrent_final_step_resolves_checkpoint_from_committed_cache():
    scheduler = _make_recurrent_scheduler()

    assert scheduler._consume_full_block_saves("r1") is None
    request = SimpleNamespace(
        request_id="r1",
        num_computed_tokens=32,
        block_hashes=[_hash(0), _hash(1)],
    )

    delay_free, params = scheduler.request_finished(
        request,
        ([11, 12, 13], [0, 0, 22, 30, 31, 32]),
    )

    assert delay_free is True
    assert params is None

    scheduler.update_connector_output(SimpleNamespace(finished_sending={"r1"}))
    assert "r1" in scheduler._block_hashes

    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                resumed_req_ids=set(),
                new_block_ids=[],
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={},
            preempted_req_ids=set(),
        )
    )
    # Attention ids come from the finished block table; the recurrent id is the
    # boundary state vLLM committed under hash[1], NOT block_ids[1][2].
    assert metadata.ready_save_intents["r1"] == SaveIntent(
        block_ids_by_group=((11, 12), (0, 22)),
        block_hashes=(_hash(0), _hash(1)),
    )
    assert metadata.save_intents == {}

    scheduler.update_connector_output(SimpleNamespace(finished_sending={"r1"}))
    assert "r1" not in scheduler._block_hashes
    assert "r1" not in scheduler._held_requests


def test_hma_binding_disables_local_prefix_lookup_before_scheduling():
    scheduler = _make_recurrent_scheduler()
    block_pool = SimpleNamespace(get_cached_block=lambda *_args: object())
    scheduler.bind_gpu_block_pool(block_pool)

    assert block_pool.get_cached_block(b"hash", [0, 1]) is None


def test_hma_accepts_different_allocator_block_counts():
    scheduler = _make_recurrent_scheduler()

    assert scheduler._copy_block_ids_by_group([[11, 12], [21, 22, 23, 24, 25, 26, 27, 28]]) == (
        (11, 12),
        (21, 22, 23, 24, 25, 26, 27, 28),
    )


def test_hma_loads_only_final_recurrent_state():
    scheduler = _make_recurrent_scheduler()
    groups = (
        (11, 12, 13, 14),
        (21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
    )

    assert scheduler._load_block_ids_by_group(groups, 1, 2) == (
        (12, 13),
        (None, 23),
    )


# ---------------------------------------------------------------------------
# Tests — virtual_block_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "kwargs", "expected"),
    [
        pytest.param("no_dcp", {"block_size": 16, "dcp_world_size": 1}, 16, id="no_dcp"),
        pytest.param("dcp2", {"block_size": 16, "dcp_world_size": 2}, 32, id="dcp2"),
        pytest.param("dcp4", {"block_size": 16, "dcp_world_size": 4}, 64, id="dcp4"),
        pytest.param(
            "smaller_physical_block",
            {"block_size": 8, "dcp_world_size": 2},
            16,
            id="smaller_physical_block",
        ),
        pytest.param("pcp2", {"block_size": 16, "pcp_world_size": 2}, 16, id="pcp2"),
        pytest.param(
            "dcp2_pcp2",
            {"block_size": 16, "dcp_world_size": 2, "pcp_world_size": 2},
            32,
            id="dcp2_pcp2",
        ),
    ],
)
def test_virtual_block_size_cases(case: str, kwargs: dict, expected: int):
    ctx = _make_ctx(**kwargs)
    assert ctx.virtual_block_size == expected, case


@pytest.mark.parametrize(
    ("case", "kwargs", "expected_rank", "expected_size"),
    [
        pytest.param(
            "non_mla_keeps_tp",
            {"is_mla": False, "tp_rank": 3, "tp_size": 4, "dcp_world_size": 2, "dcp_rank": 1},
            3,
            4,
            id="non_mla_keeps_tp",
        ),
        pytest.param(
            "mla_without_dcp_collapses_tp",
            {"is_mla": True, "tp_rank": 3, "tp_size": 4},
            0,
            1,
            id="mla_without_dcp_collapses_tp",
        ),
        pytest.param(
            "mla_with_dcp_uses_dcp",
            {"is_mla": True, "tp_rank": 3, "tp_size": 4, "dcp_world_size": 2, "dcp_rank": 1},
            1,
            2,
            id="mla_with_dcp_uses_dcp",
        ),
        pytest.param(
            "hybrid_mla_keeps_tp",
            {"is_mla": True, "collapse_mla_tp": False, "tp_rank": 3, "tp_size": 4},
            3,
            4,
            id="hybrid_mla_keeps_tp",
        ),
    ],
)
def test_effective_tp_cases(case: str, kwargs: dict, expected_rank: int, expected_size: int):
    ctx = _make_ctx(**kwargs)
    assert ctx.effective_tp_rank == expected_rank, case
    assert ctx.effective_tp_size == expected_size, case


# ---------------------------------------------------------------------------
# Tests — page-first storage (all layers of a block in one page slot)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "kwargs", "additional_config", "expected"),
    [
        pytest.param(
            "mla_no_dcp_uses_page_first",
            {"is_mla": True, "tp_rank": 0, "tp_size": 2},
            {},
            True,
            id="mla_no_dcp",
        ),
        pytest.param(
            "mla_tp1_still_page_first",
            {"is_mla": True, "tp_rank": 0, "tp_size": 1},
            {},
            True,
            id="mla_tp1",
        ),
        pytest.param(
            # Layer-split: each rank holds a disjoint layer subset = one shard
            # and writes its own sub-page, so page-first applies (per-shard).
            "mla_layer_split_uses_page_first",
            {"is_mla": True, "tp_rank": 0, "tp_size": 2},
            {"mla_layer_split_kv_cache": True},
            True,
            id="layer_split",
        ),
        pytest.param(
            "dcp_mla_opts_out",
            {"is_mla": True, "tp_rank": 0, "tp_size": 2, "dcp_world_size": 2, "dcp_rank": 1},
            {},
            False,
            id="dcp",
        ),
        pytest.param(
            # PP stages each hold only their own layers, so no single worker can
            # write a block's whole page — page-first must opt out.
            "pp_mla_opts_out",
            {"is_mla": True, "tp_rank": 0, "tp_size": 2, "pp_size": 2, "pp_rank": 1},
            {},
            False,
            id="pp",
        ),
        pytest.param(
            "non_mla_opts_out",
            {"is_mla": False, "tp_rank": 0, "tp_size": 2},
            {},
            False,
            id="non_mla",
        ),
    ],
)
def test_use_page_first_detection(case: str, kwargs: dict, additional_config: dict, expected: bool):
    from pegaflow.connector.worker import WorkerConnector

    ctx = _make_ctx(**kwargs)
    worker = WorkerConnector(
        ctx,
        vllm_config=SimpleNamespace(additional_config=additional_config),
    )
    try:
        assert worker._use_page_first() is expected, case
    finally:
        worker.shutdown()


def test_hma_disables_page_first_registration():
    from pegaflow.connector.worker import WorkerConnector

    attention = FullAttentionSpec()
    attention.block_size = 16
    recurrent = MambaSpec()
    recurrent.block_size = 16
    recurrent.mamba_cache_mode = "align"
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(layer_names=("attention",), kv_cache_spec=attention),
            SimpleNamespace(layer_names=("recurrent",), kv_cache_spec=recurrent),
        )
    )
    worker = WorkerConnector(
        _make_ctx(is_mla=True),
        vllm_config=SimpleNamespace(additional_config={}),
        kv_cache_config=kv_cache_config,
    )
    try:
        assert not worker._use_page_first()
    finally:
        worker.shutdown()


def test_page_first_block_shard_is_a_partition():
    """Page-first distributes saves by block, not by layer. Across ranks the
    block stripes must be disjoint and cover every block — otherwise a block's
    page is dropped (never sealed) or saved twice."""
    from pegaflow.connector.worker import WorkerConnector

    block_ids = tuple(range(13))
    block_hashes = tuple(bytes([i]) for i in block_ids)
    intent = SaveIntent(block_ids_by_group=(block_ids,), block_hashes=block_hashes)
    tp_size = 4

    seen: list[int] = []
    for tp_rank in range(tp_size):
        ctx = _make_ctx(is_mla=True, tp_rank=tp_rank, tp_size=tp_size)
        worker = WorkerConnector(ctx, vllm_config=SimpleNamespace(additional_config={}))
        try:
            ids, hashes = worker._block_shard(intent.block_ids_by_group[0], intent.block_hashes)
        finally:
            worker.shutdown()
        # block_ids and block_hashes stay aligned after striping.
        assert [bytes([i]) for i in ids] == hashes, tp_rank
        seen.extend(ids)

    # Equal sorted multisets ⇒ complete coverage and no duplicates across ranks.
    assert sorted(seen) == sorted(block_ids)


def test_page_first_saves_all_layers_for_this_ranks_block_stripe():
    """A page needs every layer, so a page-first rank saves ALL layers but only
    its block stripe (block_id % tp_size == tp_rank)."""
    from pegaflow.connector.worker import SaveTask, WorkerConnector

    ctx = _make_ctx(is_mla=True, tp_rank=1, tp_size=2, device_id=1)
    worker = WorkerConnector(ctx, vllm_config=SimpleNamespace(additional_config={}))
    worker._registered_layers = ["a", "b", "c"]
    worker._page_first = True
    worker._torch_device = None
    ctx.engine_client.save.return_value = (True, "")

    meta = PegaConnectorMetadata(
        save_intents={
            "r1": SaveIntent(
                block_ids_by_group=((1, 2, 3, 4),),
                block_hashes=(b"h1", b"h2", b"h3", b"h4"),
            )
        }
    )
    try:
        with patch("torch.cuda.synchronize"):
            worker._process_save_batch([SaveTask(metadata=meta, request_ids=["r1"])])
    finally:
        worker._registered_layers = []  # skip mock unregister on shutdown
        worker.shutdown()

    ctx.engine_client.save.assert_called_once()
    saves_list = ctx.engine_client.save.call_args.args[4]
    # Every layer is saved (the whole page)...
    assert {name for name, _ids, _hashes in saves_list} == {"a", "b", "c"}
    # ...but only rank 1's block stripe (odd block ids), hashes kept aligned.
    for _name, ids, hashes in saves_list:
        assert list(ids) == [1, 3]
        assert list(hashes) == [b"h1", b"h3"]


def test_recurrent_save_omits_null_group_target():
    from pegaflow.connector.worker import SaveTask, WorkerConnector

    ctx = _make_ctx()
    worker = WorkerConnector(ctx, vllm_config=SimpleNamespace(additional_config={}))
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True)
    worker._registered_layers = ["attention", "recurrent"]
    worker._layer_to_group = {"attention": 0, "recurrent": 1}
    worker._torch_device = None
    ctx.engine_client.save.return_value = (True, "")
    metadata = PegaConnectorMetadata(
        save_intents={
            "r1": SaveIntent(
                block_ids_by_group=((11,), (0,)),
                block_hashes=(b"h0",),
            )
        }
    )

    try:
        with patch("torch.cuda.synchronize"):
            worker._process_save_batch([SaveTask(metadata=metadata, request_ids=["r1"])])
    finally:
        worker._registered_layers = []
        worker.shutdown()

    saves = ctx.engine_client.save.call_args.args[4]
    assert saves == [("attention", [11], [b"h0"])]


def test_page_first_layer_split_saves_own_layers_for_all_blocks():
    """Layer-split: each rank is the sole writer of its shard (its own layers),
    so it saves ALL blocks for its registered layers — no block striping."""
    from pegaflow.connector.worker import SaveTask, WorkerConnector

    ctx = _make_ctx(is_mla=True, tp_rank=1, tp_size=2, device_id=1)
    worker = WorkerConnector(
        ctx,
        vllm_config=SimpleNamespace(additional_config={"mla_layer_split_kv_cache": True}),
    )
    assert worker._use_mla_layer_split_registration
    # This rank's shard is layers {b, d}; the other rank holds the rest.
    worker._registered_layers = ["b", "d"]
    worker._page_first = True
    worker._torch_device = None
    ctx.engine_client.save.return_value = (True, "")

    meta = PegaConnectorMetadata(
        save_intents={
            "r1": SaveIntent(
                block_ids_by_group=((1, 2, 3, 4),),
                block_hashes=(b"h1", b"h2", b"h3", b"h4"),
            )
        }
    )
    try:
        with patch("torch.cuda.synchronize"):
            worker._process_save_batch([SaveTask(metadata=meta, request_ids=["r1"])])
    finally:
        worker._registered_layers = []  # skip mock unregister on shutdown
        worker.shutdown()

    ctx.engine_client.save.assert_called_once()
    saves_list = ctx.engine_client.save.call_args.args[4]
    # Only this rank's shard layers...
    assert {name for name, _ids, _hashes in saves_list} == {"b", "d"}
    # ...and every block (no striping), hashes kept aligned.
    for _name, ids, hashes in saves_list:
        assert list(ids) == [1, 2, 3, 4]
        assert list(hashes) == [b"h1", b"h2", b"h3", b"h4"]


# ---------------------------------------------------------------------------
# Tests — SchedulerConnector decode hash refresh
#
# Verify that _consume_save_intent picks up new hashes produced during
# decode, not just the initial prefill snapshot.
# ---------------------------------------------------------------------------


def _make_fake_request(req_id: str, block_hashes: list[bytes]):
    """Minimal object that quacks like vllm.v1.request.Request."""
    req = MagicMock()
    req.request_id = req_id
    req.num_tokens = len(block_hashes) * 32  # arbitrary
    req.block_hashes = block_hashes  # mutable list, like the real Request
    return req


def _make_fake_blocks(block_ids: list[int]):
    """Minimal object that quacks like KVCacheBlocks."""
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (block_ids,)
    return blocks


class TestDecodeHashRefresh:
    """Ensure SchedulerConnector refreshes block_hashes from the live Request
    so decode-phase blocks are also saved."""

    def _make_connector(self, dcp_world_size: int = 2) -> SchedulerConnector:
        ctx = _make_ctx(block_size=16, dcp_world_size=dcp_world_size)
        return SchedulerConnector(ctx)

    def test_prefill_only_saves(self):
        """Without hash refresh, only prefill blocks are saved."""
        sc = self._make_connector()
        hashes = [_hash(i) for i in range(4)]  # 4 virtual blocks
        req = _make_fake_request("r1", list(hashes))
        blocks = _make_fake_blocks([10, 11, 12, 13])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        sc._allocated_blocks["r1"] = [[10, 11, 12, 13]]
        sc._scheduled_tokens["r1"] = 128  # 4 * 32

        intent = sc._consume_save_intent("r1", 0)
        assert intent is not None
        assert len(intent.block_ids_by_group[0]) == 4
        assert len(intent.block_hashes) == 4

    def test_decode_blocks_saved_after_refresh(self):
        """After refreshing hashes, new decode blocks become saveable."""
        sc = self._make_connector()
        initial_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(initial_hashes))
        blocks = _make_fake_blocks([10, 11, 12, 13])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        sc._allocated_blocks["r1"] = [[10, 11, 12, 13]]
        sc._scheduled_tokens["r1"] = 128  # 4 * 32

        # Save initial 4 blocks
        intent = sc._consume_save_intent("r1", 0)
        assert intent is not None
        assert len(intent.block_ids_by_group[0]) == 4

        # Simulate decode: request grows by 2 blocks
        new_hashes = [_hash(i) for i in range(4, 6)]
        req.block_hashes.extend(new_hashes)  # live Request grows
        sc._allocated_blocks["r1"][0].extend([14, 15])  # new block_ids
        sc._scheduled_tokens["r1"] += 64  # 2 * 32 more tokens

        # Before refresh: _block_hashes is stale (4 entries) → no new saves
        stale_intent = sc._consume_save_intent("r1", 0)
        assert stale_intent is None  # still capped at 4

        # Refresh hashes (simulates what build_connector_meta does)
        sc._block_hashes["r1"] = tuple(req.block_hashes)

        # Now the 2 decode blocks become saveable
        intent2 = sc._consume_save_intent("r1", 0)
        assert intent2 is not None
        assert len(intent2.block_ids_by_group[0]) == 2
        assert intent2.block_ids_by_group == ((14, 15),)
        assert intent2.block_hashes == (new_hashes[0], new_hashes[1])

    def test_cleanup_removes_request_ref(self):
        """_cleanup_request removes the stored Request reference."""
        sc = self._make_connector()
        req = _make_fake_request("r1", [_hash(0)])
        blocks = _make_fake_blocks([10])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        assert "r1" in sc._requests

        sc._cleanup_request("r1")
        assert "r1" not in sc._requests
        assert "r1" not in sc._block_hashes
        assert "r1" not in sc._allocated_blocks

    def test_external_hit_save_uses_global_block_indices(self):
        """Save intents must skip prefix-loaded block IDs on external-hit requests."""
        sc = self._make_connector(dcp_world_size=1)
        block_hashes = tuple(_hash(i) for i in range(9))

        sc._block_hashes["r1"] = block_hashes
        sc._block_index_offsets["r1"] = 6
        sc._next_stored_block_idx["r1"] = 6
        sc._scheduled_tokens["r1"] = 48  # 3 virtual blocks beyond the external hit
        sc._allocated_blocks["r1"] = [[100, 101, 102, 103, 104, 105, 200, 201, 202]]

        intent = sc._consume_save_intent("r1", 0)

        assert intent is not None
        assert intent.block_hashes == block_hashes[6:9]
        assert intent.block_ids_by_group == ((200, 201, 202),)

    def test_save_only_mode_counts_precomputed_prefix_as_saveable(self):
        """NIXL-loaded prefix should be saveable in Pega save-only mode."""
        sc = SchedulerConnector(_make_ctx(mode=PegaConnectorMode.SAVE_ONLY))
        block_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(block_hashes))

        # MultiConnector passes empty blocks and zero external tokens to
        # non-owner children. In save-only mode, Pega must later rely on
        # scheduler output rather than this allocation callback.
        sc.update_state_after_alloc(req, _make_fake_blocks([]), num_external_tokens=0)

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="r1",
                    block_ids=([10, 11, 12, 13],),
                    num_computed_tokens=48,
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={"r1": 1},
            preempted_req_ids=set(),
        )

        metadata = sc.build_connector_meta(scheduler_output)

        intent = metadata.save_intents["r1"]
        assert intent.block_ids_by_group == ((10, 11, 12),)
        assert intent.block_hashes == tuple(block_hashes[:3])

    def test_save_only_mode_handles_full_prompt_hit_recompute_token(self):
        """vLLM backs full prompt hits up by one token before scheduling."""
        sc = SchedulerConnector(_make_ctx(mode=PegaConnectorMode.SAVE_ONLY))
        block_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(block_hashes))

        sc.update_state_after_alloc(req, _make_fake_blocks([]), num_external_tokens=0)

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="r1",
                    block_ids=([10, 11, 12, 13],),
                    num_computed_tokens=63,
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={"r1": 1},
            preempted_req_ids=set(),
        )

        metadata = sc.build_connector_meta(scheduler_output)

        intent = metadata.save_intents["r1"]
        assert intent.block_ids_by_group == ((10, 11, 12, 13),)
        assert intent.block_hashes == tuple(block_hashes)

    def test_read_write_mode_does_not_save_unowned_precomputed_prefix(self):
        """Default mode keeps old behavior for Pega-owned read/write paths."""
        sc = self._make_connector(dcp_world_size=1)
        block_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(block_hashes))

        sc.update_state_after_alloc(req, _make_fake_blocks([]), num_external_tokens=0)

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="r1",
                    block_ids=([10, 11, 12, 13],),
                    num_computed_tokens=48,
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={"r1": 1},
            preempted_req_ids=set(),
        )

        metadata = sc.build_connector_meta(scheduler_output)

        assert "r1" not in metadata.save_intents

    def test_resumed_cached_request_replaces_block_table(self):
        """vLLM resumed reqs send the full block table, not append-only blocks."""
        sc = SchedulerConnector(_make_ctx(mode=PegaConnectorMode.SAVE_ONLY))
        block_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(block_hashes))

        sc.update_state_after_alloc(req, _make_fake_blocks([]), num_external_tokens=0)
        sc._allocated_blocks["r1"] = [[1, 2]]
        sc._next_stored_block_idx["r1"] = 2
        sc._scheduled_tokens["r1"] = 32

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["r1"],
                resumed_req_ids={"r1"},
                new_block_ids=[([10, 11, 12, 13],)],
                num_computed_tokens=[32],
            ),
            num_scheduled_tokens={"r1": 16},
            preempted_req_ids=set(),
        )

        metadata = sc.build_connector_meta(scheduler_output)

        intent = metadata.save_intents["r1"]
        assert intent.block_hashes == tuple(block_hashes[2:3])
        assert intent.block_ids_by_group == ((12,),)


class TestSchedulerQueryProbeReuse:
    """Repeated scheduler probes should not repeat server-side query leases."""

    def _make_connector(self) -> tuple[SchedulerConnector, MagicMock]:
        engine_client = MagicMock()
        engine_client.query_prefetch.return_value = QueryReady(2, b"lease-1")
        engine_client.release.return_value = None
        state_manager = MagicMock()
        ctx = _make_ctx(
            engine_client=engine_client,
            state_manager=state_manager,
        )
        return SchedulerConnector(ctx), engine_client

    def test_repeated_same_probe_reuses_query_result(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        first = sc.get_num_new_matched_tokens(req, num_computed_tokens=0)
        second = sc.get_num_new_matched_tokens(req, num_computed_tokens=0)

        assert first == (32, True)
        assert second == (32, True)
        engine_client.query_prefetch.assert_called_once()
        engine_client.release.assert_not_called()

    def test_query_loading_returns_retry(self):
        sc, engine_client = self._make_connector()
        engine_client.query_prefetch.return_value = QueryLoading()
        hashes = [_hash(i) for i in range(4)]

        assert sc._count_available_block_prefix(hashes, "r1") is None
        engine_client.query_prefetch.assert_called_once_with(
            sc._ctx.instance_id,
            hashes,
            req_id="r1",
            wait_for_full_prefix=False,
        )

    def test_wait_for_full_prefix_is_forwarded(self):
        engine_client = MagicMock()
        engine_client.query_prefetch.return_value = QueryLoading()
        sc = SchedulerConnector(_make_ctx(engine_client=engine_client, wait_for_full_prefix=True))
        hashes = [_hash(i) for i in range(4)]

        assert sc._count_available_block_prefix(hashes, "r1") is None
        engine_client.query_prefetch.assert_called_once_with(
            sc._ctx.instance_id,
            hashes,
            req_id="r1",
            wait_for_full_prefix=True,
        )

    def test_prefetch_loading_cancelled_on_request_cleanup(self):
        sc, engine_client = self._make_connector()
        engine_client.query_prefetch.return_value = QueryLoading()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (None, False)
        assert sc._prefetch_tracker.pending_prefetches == 1

        sc._cleanup_request("r1")

        assert sc._prefetch_tracker.pending_prefetches == 0
        assert "r1" not in sc._prefetch_start_times
        assert "r1" not in sc._pending_query_probes

    def test_save_only_mode_skips_query(self):
        engine_client = MagicMock()
        sc = SchedulerConnector(
            _make_ctx(engine_client=engine_client, mode=PegaConnectorMode.SAVE_ONLY)
        )
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (0, False)
        engine_client.query_prefetch.assert_not_called()

    def test_query_prefetch_rejects_unknown_outcome(self):
        sc, engine_client = self._make_connector()
        engine_client.query_prefetch.return_value = object()

        with pytest.raises(TypeError, match="unexpected outcome"):
            sc._count_available_block_prefix([_hash(i) for i in range(4)], "r1")

    def test_committed_probe_is_not_released_on_cleanup(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(2)])
        blocks = _make_fake_blocks([10, 11])
        blocks.blocks = [[SimpleNamespace(block_hash=None), SimpleNamespace(block_hash=None)]]

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc.update_state_after_alloc(req, blocks, num_external_tokens=32)
        sc._cleanup_request("r1")

        engine_client.release.assert_not_called()

    def test_load_block_mismatch_releases_probe_and_raises(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(2)])
        blocks = _make_fake_blocks([10, 11])
        blocks.blocks = [[SimpleNamespace(block_hash=None), SimpleNamespace(block_hash=None)]]

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        with pytest.raises(RuntimeError, match="leased block mismatch"):
            sc.update_state_after_alloc(req, blocks, num_external_tokens=16)

        engine_client.release.assert_called_once_with(b"lease-1")
        assert "r1" not in sc._pending_query_probes
        assert "r1" not in sc._pending_load_intents

    def test_different_probe_releases_previous_uncommitted_probe(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        engine_client.query_prefetch.return_value = QueryReady(2, b"lease-2")
        req.block_hashes = [_hash(i) for i in range(10, 14)]
        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        assert engine_client.query_prefetch.call_count == 2
        engine_client.release.assert_called_once_with(b"lease-1")
        # Stale probe is released before the second server call.
        assert [call[0] for call in engine_client.method_calls] == [
            "query_prefetch",
            "release",
            "query_prefetch",
        ]

    def test_release_failure_does_not_abort_cleanup(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])
        engine_client.release.side_effect = RuntimeError("server gone")

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc._cleanup_request("r1")

        assert "r1" not in sc._pending_query_probes
        engine_client.release.assert_called_once_with(b"lease-1")

    def test_shutdown_releases_uncommitted_probe(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc.shutdown()

        assert "r1" not in sc._pending_query_probes
        engine_client.release.assert_called_once_with(b"lease-1")
