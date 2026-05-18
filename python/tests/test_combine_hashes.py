"""Unit tests for DCP-aware block size logic in ConnectorContext."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext  # noqa: E402
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402

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
        "num_layers": 1,
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


class TestSchedulerQueryProbeReuse:
    """Repeated scheduler probes should not repeat server-side query pins."""

    def _make_connector(self, world_size: int = 1) -> tuple[SchedulerConnector, MagicMock]:
        engine_client = MagicMock()
        engine_client.query_prefetch.return_value = {
            "ok": True,
            "message": "ok",
            "hit_blocks": 2,
            "prefetch_state": "done",
            "loading_blocks": 0,
            "missing_blocks": 0,
        }
        engine_client.unpin.return_value = (True, "ok")
        state_manager = MagicMock()
        state_manager.is_available.return_value = True
        ctx = _make_ctx(
            engine_client=engine_client,
            state_manager=state_manager,
            world_size=world_size,
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
        engine_client.unpin.assert_not_called()

    def test_query_prefetch_dict_requires_contract_fields(self):
        sc, engine_client = self._make_connector()
        engine_client.query_prefetch.return_value = {
            "ok": True,
            "message": "ok",
            "hit_blocks": 2,
            "loading_blocks": 0,
            "missing_blocks": 0,
        }

        with pytest.raises(KeyError, match="prefetch_state"):
            sc._count_available_block_prefix([_hash(i) for i in range(4)], "r1")

    def test_query_prefetch_dict_rejects_unknown_prefetch_state(self):
        sc, engine_client = self._make_connector()
        engine_client.query_prefetch.return_value = {
            "ok": True,
            "message": "ok",
            "hit_blocks": 2,
            "prefetch_state": "weird",
            "loading_blocks": 0,
            "missing_blocks": 0,
        }

        with pytest.raises(RuntimeError, match="prefetch_state"):
            sc._count_available_block_prefix([_hash(i) for i in range(4)], "r1")

    def test_committed_probe_is_not_unpinned_on_cleanup(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(2)])
        blocks = _make_fake_blocks([10, 11])
        blocks.blocks = [[SimpleNamespace(block_hash=None), SimpleNamespace(block_hash=None)]]

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc.update_state_after_alloc(req, blocks, num_external_tokens=32)
        sc._cleanup_request("r1")

        engine_client.unpin.assert_not_called()

    def test_different_probe_releases_previous_uncommitted_probe(self):
        sc, engine_client = self._make_connector()
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        req.block_hashes = [_hash(i) for i in range(10, 14)]
        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        assert engine_client.query_prefetch.call_count == 2
        engine_client.unpin.assert_called_once_with("test", [_hash(0), _hash(1)], 1)

    def test_uncommitted_probe_release_matches_world_size_pin_count(self):
        sc, engine_client = self._make_connector(world_size=3)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc._cleanup_request("r1")

        engine_client.unpin.assert_called_once_with("test", [_hash(0), _hash(1)], 3)

    def test_failed_unpin_rpc_remains_tracked(self):
        sc, engine_client = self._make_connector(world_size=3)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])
        engine_client.unpin.side_effect = RuntimeError("temporary")

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc._cleanup_request("r1")

        engine_client.unpin.assert_called_once_with("test", [_hash(0), _hash(1)], 3)
        assert "r1" not in sc._pending_query_probes
        assert sc._pending_query_probe_releases["r1"].release_refs_per_hash == 3

    def test_different_probe_does_not_overwrite_failed_release(self):
        sc, engine_client = self._make_connector(world_size=2)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        engine_client.unpin.side_effect = RuntimeError("temporary")
        req.block_hashes = [_hash(i) for i in range(10, 14)]

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (None, False)
        assert engine_client.query_prefetch.call_count == 1
        assert "r1" not in sc._pending_query_probes
        assert sc._pending_query_probe_releases["r1"].remaining_hashes == tuple(
            _hash(i) for i in range(4)
        )
        assert sc._pending_query_probe_releases["r1"].release_refs_per_hash == 2

    def test_failed_release_must_clear_before_new_probe(self):
        sc, engine_client = self._make_connector(world_size=2)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        engine_client.unpin.side_effect = RuntimeError("temporary")
        req.block_hashes = [_hash(i) for i in range(10, 14)]
        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (None, False)

        engine_client.unpin.side_effect = None
        engine_client.unpin.return_value = (True, "ok")
        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        assert engine_client.query_prefetch.call_count == 2
        assert "r1" not in sc._pending_query_probe_releases
        assert sc._pending_query_probes["r1"].remaining_hashes == tuple(
            _hash(i) for i in range(10, 14)
        )

    def test_cleanup_release_retry_is_drained_by_stats(self):
        sc, engine_client = self._make_connector(world_size=1)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)

        engine_client.unpin.side_effect = [RuntimeError("temporary")]
        sc._cleanup_request("r1")
        assert sc._pending_query_probe_releases["r1"].release_refs_per_hash == 1

        engine_client.unpin.side_effect = None
        engine_client.unpin.return_value = (True, "ok")
        sc.get_stats()

        assert "r1" not in sc._pending_query_probe_releases

    def test_shutdown_releases_uncommitted_probe(self):
        sc, engine_client = self._make_connector(world_size=2)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])

        assert sc.get_num_new_matched_tokens(req, num_computed_tokens=0) == (32, True)
        sc.shutdown()

        assert "r1" not in sc._pending_query_probes
        assert "r1" not in sc._pending_query_probe_releases
        engine_client.unpin.assert_called_once_with("test", [_hash(0), _hash(1)], 2)
