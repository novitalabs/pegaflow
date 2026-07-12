"""P/D tail-block save trigger (`pegaflow.pd_tail_save`).

The trigger condition is the part that has bitten twice: the tail block must
be saved exactly on the step whose schedule finishes the prompt, using the
scheduler-authoritative `num_computed_tokens + num_scheduled_tokens` — any
connector-side accumulator lies across preempt/resume cycles and can save a
torn tail page into a content-addressed tier that dedups late corrections.
"""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext
from pegaflow.connector.scheduler import SchedulerConnector

VBS = 16  # virtual block size for these tests


def _hash(i: int) -> bytes:
    return hashlib.sha256(f"block_{i}".encode()).digest()


def _make_ctx() -> ConnectorContext:
    return ConnectorContext(
        **{
            "instance_id": "test",
            "namespace": "ns",
            "block_size": VBS,
            "tp_size": 1,
            "world_size": 1,
            "tp_rank": 0,
            "device_id": 0,
            "engine_client": MagicMock(),
            "state_manager": MagicMock(),
            "is_mla": False,
            "dcp_world_size": 1,
            "dcp_rank": 0,
        }
    )  # type: ignore[arg-type]


def _make_request(req_id: str, prompt_len: int, full_hashes: int):
    """Quacks like vllm.v1.request.Request, with the tail-relevant fields
    explicit (no MagicMock auto-attributes: the lora/salt/mm guard reads
    them)."""
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        block_hashes=[_hash(i) for i in range(full_hashes)],
        lora_request=None,
        cache_salt=None,
        mm_features=None,
    )


def _make_connector(req, allocated: list[int]) -> SchedulerConnector:
    sc = SchedulerConnector(_make_ctx())
    # Inject the tail machinery directly: these tests pin the TRIGGER, not
    # vLLM's hash function (covered by the cross-engine e2e gates).
    sc._tail_hash_fn = object()
    sc._kv_cache_utils = SimpleNamespace(NONE_HASH=b"\x00" * 32)
    sc._hash_block_tokens = lambda fn, parent, tokens, extra: b"tail:%d" % len(tokens)
    sc._requests[req.request_id] = req
    sc._block_hashes[req.request_id] = tuple(req.block_hashes)
    sc._allocated_blocks[req.request_id] = allocated
    sc._scheduled_tokens[req.request_id] = 0
    sc._block_index_offsets[req.request_id] = 0
    sc._next_stored_block_idx[req.request_id] = 0
    return sc


class TestTailSaveTrigger:
    def test_fires_exactly_when_written_covers_the_prompt(self):
        # prompt 50 = 3 full blocks (48) + 2-token tail in block idx 3
        req = _make_request("r1", prompt_len=50, full_hashes=3)
        sc = _make_connector(req, allocated=[10, 11, 12, 13])

        assert sc._consume_tail_save("r1", written=49) is None
        intent = sc._consume_tail_save("r1", written=50)
        assert intent is not None
        assert intent.block_ids == (13,)
        # Once saved, never again (content-addressed dedup upstream would
        # reject a correction, so a re-save is pure waste).
        assert sc._consume_tail_save("r1", written=51) is None

    def test_block_aligned_prompt_has_no_tail(self):
        req = _make_request("r1", prompt_len=48, full_hashes=3)
        sc = _make_connector(req, allocated=[10, 11, 12])
        assert sc._consume_tail_save("r1", written=48) is None

    def test_sub_block_prompt_uses_none_hash_parent(self):
        req = _make_request("r1", prompt_len=10, full_hashes=0)
        sc = _make_connector(req, allocated=[10])
        intent = sc._consume_tail_save("r1", written=10)
        assert intent is not None
        assert intent.block_ids == (10,)
        assert intent.block_hashes == (b"tail:10",)

    def test_salted_or_lora_requests_never_tail_save(self):
        # The tail key carries no extra_keys; saving would alias
        # differently-salted prompts onto one content-addressed key.
        for field, value in (("cache_salt", "tenant-a"), ("lora_request", object())):
            req = _make_request("r1", prompt_len=50, full_hashes=3)
            setattr(req, field, value)
            sc = _make_connector(req, allocated=[10, 11, 12, 13])
            assert sc._consume_tail_save("r1", written=50) is None

    def test_tail_block_not_yet_allocated_defers(self):
        req = _make_request("r1", prompt_len=50, full_hashes=3)
        sc = _make_connector(req, allocated=[10, 11, 12])  # tail block missing
        assert sc._consume_tail_save("r1", written=50) is None


class TestTailSaveThroughBuildConnectorMeta:
    """`written` must come from the scheduler's own per-step counters."""

    def _step(self, sc, *, new=None, cached=None):
        new = new or []
        cached = cached or []
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=r["id"],
                    block_ids=(r["blocks"],),
                    num_computed_tokens=r["computed"],
                )
                for r in new
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[r["id"] for r in cached],
                new_block_ids=[r.get("new_blocks") for r in cached],
                num_computed_tokens=[r["computed"] for r in cached],
                resumed_req_ids=set(
                    r["id"] for r in cached if r.get("resumed", False)
                ),
            ),
            num_scheduled_tokens={
                r["id"]: r["scheduled"] for r in (list(new) + list(cached))
            },
            preempted_req_ids=set(),
        )
        return sc.build_connector_meta(scheduler_output)

    def test_prefix_hit_new_request_fires_on_admission_step(self):
        # Multi-turn turn>=2: the whole previous context is a local prefix
        # hit (num_computed_tokens=48), only the 2-token tail is scheduled.
        req = _make_request("r1", prompt_len=50, full_hashes=3)
        sc = _make_connector(req, allocated=[10, 11, 12, 13])
        meta = self._step(
            sc,
            new=[{"id": "r1", "blocks": [10, 11, 12, 13], "computed": 48, "scheduled": 2}],
        )
        intent = meta.save_intents.get("r1")
        assert intent is not None
        assert 13 in intent.block_ids

    def test_preempt_resume_does_not_fire_early(self):
        # Chunked prefill: 32 of 50 scheduled, then preempted (all progress
        # discarded), then resumed 32, then the final 18. A cumulative
        # accumulator would see 32+32=64 >= 50 mid-resume and save a tail
        # page whose rows are not written yet.
        req = _make_request("r1", prompt_len=50, full_hashes=3)
        sc = _make_connector(req, allocated=[10, 11, 12, 13])

        meta = self._step(
            sc,
            new=[{"id": "r1", "blocks": [10, 11, 12, 13], "computed": 0, "scheduled": 32}],
        )
        assert meta.save_intents.get("r1") is None or 13 not in meta.save_intents["r1"].block_ids

        # Preempted; resume re-schedules from scratch.
        meta = self._step(
            sc,
            cached=[
                {
                    "id": "r1",
                    "new_blocks": ([10, 11, 12, 13],),
                    "computed": 0,
                    "scheduled": 32,
                    "resumed": True,
                }
            ],
        )
        assert meta.save_intents.get("r1") is None or 13 not in meta.save_intents["r1"].block_ids

        meta = self._step(
            sc,
            cached=[{"id": "r1", "new_blocks": None, "computed": 32, "scheduled": 18}],
        )
        intent = meta.save_intents.get("r1")
        assert intent is not None
        assert 13 in intent.block_ids
