"""Unit tests for DCP-aware block size logic in ConnectorContext.

The Rust extension (pegaflow.pegaflow) is not always available in CI/dev,
so we pre-load a stub into sys.modules before the real package __init__
tries to import it.
"""

from __future__ import annotations

import hashlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub the native extension *before* anything imports pegaflow.
# ---------------------------------------------------------------------------

_EXT_NAME = "pegaflow.pegaflow"
if _EXT_NAME not in sys.modules:
    _ext_stub = types.ModuleType(_EXT_NAME)
    _ext_stub.EngineRpcClient = MagicMock  # type: ignore[attr-defined]
    _ext_stub._NativeEngineClient = MagicMock  # type: ignore[attr-defined]
    _ext_stub._NativeLoadState = MagicMock  # type: ignore[attr-defined]
    _ext_stub.KvEgressRuntime = MagicMock  # type: ignore[attr-defined]
    _ext_stub.PegaFlowBusinessError = type("PegaFlowBusinessError", (Exception,), {})  # type: ignore[attr-defined]
    _ext_stub.PegaFlowError = type("PegaFlowError", (Exception,), {})  # type: ignore[attr-defined]
    _ext_stub.PegaFlowServiceError = type("PegaFlowServiceError", (Exception,), {})  # type: ignore[attr-defined]
    _ext_stub.PyLoadState = MagicMock  # type: ignore[attr-defined]
    _ext_stub.__version__ = "0.0.0-test"  # type: ignore[attr-defined]
    sys.modules[_EXT_NAME] = _ext_stub

from pegaflow.connector.common import ConnectorContext  # noqa: E402
from pegaflow.connector.scheduler import SchedulerConnector  # noqa: E402
from pegaflow import LoadPlan, LoadSourceKind, PrepareLoadResult  # noqa: E402

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
        "layer_names": ("ALL_LAYERS",),
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


class TestVirtualBlockSize:
    """Validate ConnectorContext.virtual_block_size property."""

    def test_no_dcp(self):
        ctx = _make_ctx(block_size=16, dcp_world_size=1)
        assert ctx.virtual_block_size == 16

    def test_dcp2(self):
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        assert ctx.virtual_block_size == 32

    def test_dcp4(self):
        ctx = _make_ctx(block_size=16, dcp_world_size=4)
        assert ctx.virtual_block_size == 64

    def test_different_physical_block_size(self):
        ctx = _make_ctx(block_size=8, dcp_world_size=2)
        assert ctx.virtual_block_size == 16

    def test_pcp2(self):
        ctx = _make_ctx(block_size=16, pcp_world_size=2)
        assert ctx.virtual_block_size == 32

    def test_dcp2_pcp2(self):
        ctx = _make_ctx(block_size=16, dcp_world_size=2, pcp_world_size=2)
        assert ctx.virtual_block_size == 64


# ---------------------------------------------------------------------------
# Tests — effective_tp_rank / effective_tp_size
# ---------------------------------------------------------------------------


class TestEffectiveTP:
    """Validate DCP-aware tp_rank / tp_size properties."""

    def test_non_mla_ignores_dcp(self):
        ctx = _make_ctx(is_mla=False, tp_rank=3, tp_size=4, dcp_world_size=2, dcp_rank=1)
        assert ctx.effective_tp_rank == 3
        assert ctx.effective_tp_size == 4

    def test_mla_no_dcp(self):
        ctx = _make_ctx(is_mla=True, tp_rank=3, tp_size=4, dcp_world_size=1, dcp_rank=0)
        assert ctx.effective_tp_rank == 0
        assert ctx.effective_tp_size == 1

    def test_mla_with_dcp(self):
        ctx = _make_ctx(is_mla=True, tp_rank=3, tp_size=4, dcp_world_size=2, dcp_rank=1)
        assert ctx.effective_tp_rank == 1  # dcp_rank
        assert ctx.effective_tp_size == 2  # dcp_world_size


# ---------------------------------------------------------------------------
# Tests — block index arithmetic
#
# These simulate the same arithmetic used in SchedulerConnector to make sure
# virtual_block_size produces correct results.
# ---------------------------------------------------------------------------


class TestBlockIndexArithmetic:
    """End-to-end sanity checks on the block arithmetic that the scheduler
    connector relies on (computed_blocks, num_load_blocks, saveable)."""

    def test_computed_blocks_no_dcp(self):
        """128 tokens / virtual_block_size(16) = 8 blocks."""
        ctx = _make_ctx(block_size=16, dcp_world_size=1)
        num_computed_tokens = 128
        computed_blocks = num_computed_tokens // ctx.virtual_block_size
        assert computed_blocks == 8

    def test_computed_blocks_dcp2(self):
        """128 tokens / virtual_block_size(32) = 4 blocks."""
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        num_computed_tokens = 128
        computed_blocks = num_computed_tokens // ctx.virtual_block_size
        assert computed_blocks == 4

    def test_remaining_hashes_index(self):
        """After skipping computed blocks, remaining hashes are correct."""
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        # 6 hashes, each covering 32 tokens → 192 tokens total
        all_hashes = [_hash(i) for i in range(6)]
        num_computed_tokens = 64  # 2 virtual blocks already computed
        computed_blocks = num_computed_tokens // ctx.virtual_block_size  # 2
        remaining = all_hashes[computed_blocks:]
        assert len(remaining) == 4
        assert remaining[0] == all_hashes[2]

    def test_num_load_blocks(self):
        """num_external_tokens / virtual_block_size gives load block count."""
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        num_external_tokens = 96  # 3 virtual blocks
        num_load_blocks = num_external_tokens // ctx.virtual_block_size
        assert num_load_blocks == 3

    def test_saveable_calculation(self):
        """saveable = min(hashes, allocated, scheduled // virtual_block_size)."""
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        block_hashes = [_hash(i) for i in range(5)]
        allocated = list(range(5))
        scheduled_tokens = 128  # 4 virtual blocks
        saveable = min(
            len(block_hashes), len(allocated), scheduled_tokens // ctx.virtual_block_size
        )
        assert saveable == 4  # limited by scheduled tokens

    def test_hit_tokens(self):
        """hit_blocks * virtual_block_size gives correct token count."""
        ctx = _make_ctx(block_size=16, dcp_world_size=2)
        hit_blocks = 3
        hit_tokens = hit_blocks * ctx.virtual_block_size
        assert hit_tokens == 96


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
    req.num_prompt_tokens = req.num_tokens
    req.prompt_token_ids = None
    req.block_hashes = block_hashes  # mutable list, like the real Request
    return req


def _make_fake_blocks(block_ids: list[int], computed: int = 0):
    """Minimal object that quacks like KVCacheBlocks."""
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (block_ids,)
    blocks.blocks = (
        tuple(
            SimpleNamespace(
                block_id=block_id,
                block_hash=_hash(i) if i < computed else None,
                is_null=False,
            )
            for i, block_id in enumerate(block_ids)
        ),
    )
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

    def test_source_pd_request_builds_egress_intent_from_prefill_tokens(self):
        """P/source requests push blocks selected by remote-prefill token count."""
        sc = self._make_connector(dcp_world_size=1)
        req = _make_fake_request("r1", [_hash(i) for i in range(2)])
        req.num_tokens = 33
        req.num_prompt_tokens = 33
        req.kv_transfer_params = {
            "pegaflow": {
                "type": "prefill_push",
                "request_id": "pd-r1",
                "decode_endpoint": "http://10.0.0.2:50055",
                "decode_instance_id": "decode-0",
            }
        }
        blocks = _make_fake_blocks([10, 11])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        sc._allocated_blocks["r1"] = [10, 11]
        sc._scheduled_tokens["r1"] = 32

        egress = sc._build_egress_intents()

        intent = egress["r1"]
        assert intent.request_id == "pd-r1"
        assert intent.remote_endpoint == "http://10.0.0.2:50055"
        assert intent.remote_instance_id == "decode-0"
        assert intent.block_ids == (10, 11)
        assert intent.block_hashes == (_hash(0), _hash(1))

    def test_source_pd_short_request_builds_egress_without_hashes(self):
        sc = self._make_connector(dcp_world_size=1)
        req = _make_fake_request("r1", [])
        req.num_tokens = 15
        req.num_prompt_tokens = 15
        req.kv_transfer_params = {
            "pegaflow": {
                "type": "prefill_push",
                "request_id": "pd-r1",
                "decode_endpoint": "http://10.0.0.2:50055",
                "decode_instance_id": "decode-0",
            }
        }
        blocks = _make_fake_blocks([10])

        sc.update_state_after_alloc(req, blocks, num_external_tokens=0)
        sc._allocated_blocks["r1"] = [10]
        sc._scheduled_tokens["r1"] = 14

        egress = sc._build_egress_intents()

        intent = egress["r1"]
        assert intent.block_ids == (10,)
        assert intent.block_hashes == ()

    def test_source_pd_request_does_not_prepare_d_receive(self):
        """The same pegaflow_pd_push flag is interpreted by role."""
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.state_manager.is_available.return_value = True
        ctx.engine_client.prepare_load.return_value = PrepareLoadResult.done(None)
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [_hash(i) for i in range(2)])
        req.num_tokens = 32
        req.kv_transfer_params = {
            "pegaflow": {
                "type": "prefill_push",
                "request_id": "pd-r1",
                "decode_endpoint": "http://10.0.0.2:50055",
                "decode_instance_id": "decode-0",
            }
        }

        assert sc.get_num_new_matched_tokens(req, 0) == (0, False)
        prepare_request = ctx.engine_client.prepare_load.call_args.args[0]
        assert prepare_request.decode_request_id is None


class TestPdReceivePrepare:
    """Validate D-side P/D prepare behavior in scheduler hook."""

    def test_pd_push_request_prepares_receive_and_blocks_gpu_alloc(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.engine_client.prepare_load.return_value = PrepareLoadResult.in_progress()
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])
        req.num_tokens = 64
        req.num_prompt_tokens = 64
        req.kv_transfer_params = {
            "pegaflow": {
                "type": "decode_load",
                "request_id": "pd1",
            }
        }

        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)

        prepare_request = ctx.engine_client.prepare_load.call_args.args[0]
        assert prepare_request.request_id == "r1"
        assert prepare_request.decode_request_id == "pd1"
        assert prepare_request.block_hashes == tuple(_hash(i) for i in range(4))
        assert prepare_request.virtual_block_size == 16

        # The scheduler only sees the high-level preparing state.
        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)
        assert ctx.engine_client.prepare_load.call_count == 2

    def test_pd_push_prepares_only_complete_hashed_blocks(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.engine_client.prepare_load.return_value = PrepareLoadResult.in_progress()
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [_hash(0)])
        req.num_tokens = 24
        req.num_prompt_tokens = 24
        req.kv_transfer_params = {"pegaflow": {"type": "decode_load"}}

        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)

        prepare_request = ctx.engine_client.prepare_load.call_args.args[0]
        assert prepare_request.block_hashes == (_hash(0),)
        assert prepare_request.num_prompt_tokens == 24
        assert prepare_request.virtual_block_size == 16

    def test_pd_push_without_complete_block_hashes_prepares_receive(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.engine_client.prepare_load.return_value = PrepareLoadResult.in_progress()
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [])
        req.num_tokens = 15
        req.num_prompt_tokens = 15
        req.kv_transfer_params = {"pegaflow": {"type": "decode_load"}}

        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)
        prepare_request = ctx.engine_client.prepare_load.call_args.args[0]
        assert prepare_request.block_hashes == ()
        assert prepare_request.decode_request_id == "r1"

    def test_pd_receive_partial_block_alloc_builds_hashless_load_intent(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [])
        req.num_tokens = 15
        req.num_prompt_tokens = 15
        req.kv_transfer_params = {"pegaflow": {"type": "decode_load"}}
        blocks = _make_fake_blocks([10])
        sc._load_plans["r1"] = LoadPlan(
            request_id="r1",
            source=LoadSourceKind.STAGED,
            num_tokens=14,
            num_blocks=1,
        )

        sc.update_state_after_alloc(req, blocks, num_external_tokens=14)

        intent = sc._pending_load_intents["r1"]
        assert intent.block_ids == (10,)
        assert intent.plan.block_hashes == ()
        assert intent.num_tokens == 14

    def test_pd_push_returns_external_tokens_after_imm_ready(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.engine_client.prepare_load.side_effect = [
            PrepareLoadResult.in_progress(),
            PrepareLoadResult.done(
                LoadPlan(
                    request_id="r1",
                    source=LoadSourceKind.STAGED,
                    num_tokens=63,
                    num_blocks=4,
                    block_hashes=tuple(_hash(i) for i in range(4)),
                    token="h1",
                )
            ),
        ]
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("r1", [_hash(i) for i in range(4)])
        req.num_tokens = 64
        req.num_prompt_tokens = 64
        req.kv_transfer_params = {"pegaflow": {"type": "decode_load"}}

        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)
        assert sc.get_num_new_matched_tokens(req, 0) == (63, True)

    def test_pd_prepare_cleanup_uses_vllm_request_id(self):
        ctx = _make_ctx(block_size=16, layer_names=("ALL_LAYERS",))
        ctx.engine_client.prepare_load.return_value = PrepareLoadResult.in_progress()
        sc = SchedulerConnector(ctx)
        req = _make_fake_request("vllm-r1", [_hash(i) for i in range(2)])
        req.num_tokens = 32
        req.num_prompt_tokens = 32
        req.kv_transfer_params = {
            "pegaflow": {
                "type": "decode_load",
                "request_id": "router-pd-r1",
            }
        }

        assert sc.get_num_new_matched_tokens(req, 0) == (None, False)
        prepare_request = ctx.engine_client.prepare_load.call_args.args[0]
        assert prepare_request.request_id == "vllm-r1"
        assert prepare_request.decode_request_id == "router-pd-r1"

        sc._cleanup_request("vllm-r1")

        assert sc._load_plans == {}
