"""Unit tests for DCP-aware block size logic in ConnectorContext."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext, PegaConnectorMode  # noqa: E402
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
        pytest.param("pcp2", {"block_size": 16, "pcp_world_size": 2}, 32, id="pcp2"),
        pytest.param(
            "dcp2_pcp2",
            {"block_size": 16, "dcp_world_size": 2, "pcp_world_size": 2},
            64,
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
    ],
)
def test_effective_tp_cases(case: str, kwargs: dict, expected_rank: int, expected_size: int):
    ctx = _make_ctx(**kwargs)
    assert ctx.effective_tp_rank == expected_rank, case
    assert ctx.effective_tp_size == expected_size, case


@pytest.mark.parametrize(
    ("case", "kwargs", "additional_config", "expected"),
    [
        pytest.param(
            "ordinary_mla_replica_skips_nonzero_tp_save",
            {
                "is_mla": True,
                "tp_rank": 1,
                "tp_size": 2,
            },
            {},
            True,
            id="ordinary_mla_replica",
        ),
        pytest.param(
            "mla_layer_split_registration_does_not_skip_nonzero_tp_save",
            {
                "is_mla": True,
                "tp_rank": 1,
                "tp_size": 2,
            },
            {"mla_layer_split_kv_cache": True},
            False,
            id="mla_layer_split_registration",
        ),
        pytest.param(
            "dcp_mla_does_not_skip_save",
            {
                "is_mla": True,
                "tp_rank": 1,
                "tp_size": 2,
                "dcp_world_size": 2,
                "dcp_rank": 1,
            },
            {},
            False,
            id="dcp_mla",
        ),
    ],
)
def test_save_skip_keeps_ordinary_mla_optimization(
    case: str, kwargs: dict, additional_config: dict, expected: bool
):
    from pegaflow.connector.worker import WorkerConnector

    ctx = _make_ctx(**kwargs)
    worker = WorkerConnector(
        ctx,
        vllm_config=SimpleNamespace(additional_config=additional_config),
    )
    try:
        assert worker._should_skip_save_submission() is expected, case
    finally:
        worker.shutdown()


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
        sc._allocated_blocks["r1"] = [10, 11, 12, 13]
        sc._scheduled_tokens["r1"] = 128  # 4 * 32

        intent = sc._consume_save_intent("r1")
        assert intent is not None
        assert len(intent.block_ids) == 4
        assert len(intent.block_hashes) == 4

    def test_decode_blocks_saved_after_refresh(self):
        """After refreshing hashes, new decode blocks become saveable."""
        sc = self._make_connector()
        initial_hashes = [_hash(i) for i in range(4)]
        req = _make_fake_request("r1", list(initial_hashes))
        blocks = _make_fake_blocks([10, 11, 12, 13])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        sc._allocated_blocks["r1"] = [10, 11, 12, 13]
        sc._scheduled_tokens["r1"] = 128  # 4 * 32

        # Save initial 4 blocks
        intent = sc._consume_save_intent("r1")
        assert intent is not None
        assert len(intent.block_ids) == 4

        # Simulate decode: request grows by 2 blocks
        new_hashes = [_hash(i) for i in range(4, 6)]
        req.block_hashes.extend(new_hashes)  # live Request grows
        sc._allocated_blocks["r1"].extend([14, 15])  # new block_ids
        sc._scheduled_tokens["r1"] += 64  # 2 * 32 more tokens

        # Before refresh: _block_hashes is stale (4 entries) → no new saves
        stale_intent = sc._consume_save_intent("r1")
        assert stale_intent is None  # still capped at 4

        # Refresh hashes (simulates what build_connector_meta does)
        sc._block_hashes["r1"] = tuple(req.block_hashes)

        # Now the 2 decode blocks become saveable
        intent2 = sc._consume_save_intent("r1")
        assert intent2 is not None
        assert len(intent2.block_ids) == 2
        assert intent2.block_ids == (14, 15)
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
        sc._allocated_blocks["r1"] = [
            100,
            101,
            102,
            103,
            104,
            105,
            200,
            201,
            202,
        ]

        intent = sc._consume_save_intent("r1")

        assert intent is not None
        assert intent.block_hashes == block_hashes[6:9]
        assert intent.block_ids == (200, 201, 202)

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
        assert intent.block_ids == (10, 11, 12)
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
        assert intent.block_ids == (10, 11, 12, 13)
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
        sc._allocated_blocks["r1"] = [1, 2]
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
        assert intent.block_ids == (12,)


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

        assert sc._count_available_block_prefix([_hash(i) for i in range(4)], "r1") is None

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

        with pytest.raises(RuntimeError, match="load block mismatch"):
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
