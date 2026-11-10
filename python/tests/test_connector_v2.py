"""Connector v2 local contracts: resume rules, query planning, save pins, spans.

No vLLM, torch, native extension or server: the server is a fake.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import pytest

from pegaflow.connector_v2 import semantics as sem
from pegaflow.connector_v2.layout import Layout, NsSpec, Parallel, Unsupported, build_layout
from pegaflow.connector_v2.scheduler import LoadPlan, Scheduler, StepPlan
from pegaflow.connector_v2.worker import Region, Worker

FULL4 = sem.Group(sem.FULL, 4)
CKPT4 = sem.Group(sem.CHECKPOINT, 4)
WIN = sem.Group(sem.WINDOW, 4, 7)  # vLLM sliding_window = 8


@pytest.mark.parametrize(
    "groups,present,prompt_len,expected",
    [
        pytest.param([FULL4], [{0, 1, 2}], 100, 12, id="full-prefix"),
        pytest.param([FULL4], [{0, 1, 2}], 12, 8, id="last-token-clamp"),
        pytest.param([FULL4, CKPT4], [{0, 1, 2, 3}, {1, 2}], 21, 12, id="checkpoint-at-12"),
        pytest.param([FULL4, CKPT4], [{0, 1, 2, 3}, {0, 3}], 21, 16, id="checkpoint-at-16"),
        # Window needs blocks [(16-7)//4, 4) = [2, 4) at p=16: block 1 is a hole.
        pytest.param([FULL4, WIN], [{0, 1, 2, 3}, {2, 3}], 100, 16, id="window-hole-ok"),
        pytest.param([FULL4, WIN], [{0, 1, 2, 3}, {0, 1, 3}], 100, 8, id="window-gap"),
        pytest.param([FULL4], [set()], 100, 0, id="miss"),
    ],
)
def test_resolve(groups, present, prompt_len, expected):
    assert sem.resolve(groups, [frozenset(p) for p in present], prompt_len) == expected


def test_saveable():
    assert sem.saveable(FULL4, 12, 20) == range(3, 5)
    assert sem.saveable(CKPT4, 12, 20) == range(4, 5)
    assert sem.saveable(CKPT4, 12, 19) == range(0)
    assert sem.saveable(WIN, 0, 9) == range(0, 2)


# ------------------------------------------------------------------ layout
@dataclass
class _Spec:
    block_size: int
    page_size_bytes: int
    max_tp_shards: int | None = None


class FullAttentionSpec(_Spec):
    pass


@dataclass
class SlidingWindowSpec(_Spec):
    sliding_window: int = 8


@dataclass
class _Group:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class _Config:
    kv_cache_groups: list[_Group]
    hash_block_size: int = 4
    kv_cache_tensors: list = field(default_factory=list)


def test_layout_orders_full_first_and_splits_heads():
    cfg = _Config(
        [
            _Group(["w0"], SlidingWindowSpec(4, 512, max_tp_shards=2)),
            _Group(["f0", "f1"], FullAttentionSpec(4, 1024, max_tp_shards=2)),
        ]
    )
    layout = build_layout(cfg, Parallel(tp_size=4), {"model": "m"})
    full, win = layout.namespaces
    assert (full.gid, full.group, win.group) == (1, FULL4, WIN)
    # Two head shards over TP4: ranks 0,1 hold shard 0; 2,3 shard 1.
    assert full.savers == (0, 2)
    meta = full.meta(world_size=4, rank=3, device_id=7)
    assert meta["bytes_per_block"] == 2048 and meta["local_hidden"] * meta["block_size"] == 1024
    assert len(full.namespace) == 32 and full.namespace != win.namespace
    assert layout.step == 4


@pytest.mark.parametrize(
    "sockets,savers",
    [
        pytest.param((0, 0, 1, 1), (0, 2), id="replicas-on-one-socket"),
        pytest.param((0, 1, 0, 1), (0, 1, 2, 3), id="replicas-across-sockets"),
    ],
)
def test_layout_saves_each_shard_once_per_socket(sockets, savers):
    cfg = _Config([_Group(["f0"], FullAttentionSpec(4, 1024, max_tp_shards=2))])
    (ns,) = build_layout(cfg, Parallel(4, sockets), {}).namespaces
    assert ns.savers == savers
    (one,) = build_layout(cfg, Parallel(4), {}).namespaces
    assert ns.namespace == one.namespace


def test_layout_needs_a_full_group():
    cfg = _Config([_Group(["w0"], SlidingWindowSpec(4, 512))])
    with pytest.raises(Unsupported):
        build_layout(cfg, Parallel(), {})


# --------------------------------------------------------------- scheduler
@dataclass
class Ready:
    num_hit_blocks: int
    lease: bytes
    hit_positions: list[int] = field(default_factory=list)


class Loading:
    pass


class FakeRpc:
    """Holds a set of (namespace, prefix) keys; leases are counters."""

    def __init__(self, stored: set[tuple[bytes, bytes]]):
        self.stored = stored
        self.loading = 0  # answer Loading this many times first
        self.members_loading = 0  # same, for membership queries only
        self.leases: set[bytes] = set()
        self.queries: list[tuple[bool, int]] = []
        self.query_ids: list[str] = []

    def query_object(self, instance_id, keys, prefix, query_id):
        self.queries.append((prefix, len(keys)))
        self.query_ids.append(query_id)
        query_id = query_id or f"q{len(self.query_ids)}"
        if self.loading:
            self.loading -= 1
            return query_id, Loading()
        if not prefix and self.members_loading:
            self.members_loading -= 1
            return query_id, Loading()
        hits = [i for i, k in enumerate(keys) if k in self.stored]
        if prefix:
            n = 0
            while n < len(keys) and keys[n] in self.stored:
                n += 1
            hits = list(range(n))
        lease = f"lease{len(self.leases)}".encode() if hits else b""
        if lease:
            self.leases.add(lease)
        return query_id, Ready(len(hits), lease, [] if prefix else hits)

    def release(self, lease):
        self.leases.discard(lease)


@dataclass
class Req:
    request_id: str
    num_tokens: int
    block_hashes: list[bytes]
    shared_prefix_boundary: int = 0


def _ns(gid: int, group: sem.Group, tag: bytes) -> NsSpec:
    return NsSpec(gid, group, 1, ("l",), tag * 32, 64, 1, (0,))


def _hashes(n: int) -> list[bytes]:
    return [bytes([i]) * 32 for i in range(n)]


def _layout(*groups: sem.Group) -> Layout:
    nss = tuple(_ns(i, g, bytes([65 + i])) for i, g in enumerate(groups))
    return Layout(nss, sem.step_of(groups))


def test_hit_loads_only_past_local_and_releases_unused_leases():
    layout = _layout(FULL4, WIN)
    hashes = _hashes(6)
    full, win = layout.namespaces
    stored = {(full.namespace, h) for h in hashes[:5]} | {(win.namespace, h) for h in hashes[2:5]}
    rpc = FakeRpc(stored)
    s = Scheduler(layout, rpc, "inst")
    req = Req("r", 25, hashes)
    # Local hit 8 tokens; store backs blocks up to 5 -> p = 20.
    assert s.get_num_new_matched_tokens(req, 8) == (12, True)
    # vLLM asks again while the request waits for blocks: same hit, no new query.
    n = len(rpc.queries)
    assert s.get_num_new_matched_tokens(req, 8) == (12, True)
    assert len(rpc.queries) == n and len(rpc.leases) == 2
    assert s.get_loaded_group_ids("r") == (0, 1)
    block_ids = ([10, 11, 12, 13, 14, 15], [0, 0, 22, 23, 24, 25])
    s.update_state_after_alloc(req, block_ids, 12)
    (lp,) = s.take_plan().loads
    assert not lp.fail and len(lp.leases) == 2
    # Full loads blocks 2..4; the window needs [(20-7)//4, 5) = [3, 5).
    loaded = sorted((i, gpu) for i, _, gpu in lp.blocks)
    assert loaded == [(0, 12), (0, 13), (0, 14), (1, 23), (1, 24)]


def test_window_falls_back_to_every_point_when_the_top_misses():
    layout = _layout(FULL4, WIN)
    full, win = layout.namespaces
    hashes = _hashes(6)
    # Full blocks back 20 tokens; the window holds only blocks 1 and 2, the
    # window of p = 12 ([(12-7)//4, 3) = {1, 2}), far below the top point.
    stored = {(full.namespace, h) for h in hashes[:5]} | {(win.namespace, h) for h in hashes[1:3]}
    rpc = FakeRpc(stored)
    s = Scheduler(layout, rpc, "inst")
    assert s.get_num_new_matched_tokens(Req("r", 25, hashes), 0) == (12, True)
    assert rpc.queries == [(True, 6), (False, 2), (False, 5)] and len(rpc.leases) == 2


@pytest.mark.parametrize("ckpt_block,expected", [(3, 16), (0, 4)], ids=["near-top", "deep"])
def test_hybrid_hit_asks_vllm_to_checkpoint_where_the_full_prefix_ends(ckpt_block, expected):
    layout = _layout(FULL4, CKPT4)
    full, ckpt = layout.namespaces
    hashes = _hashes(6)
    stored = {(full.namespace, h) for h in hashes[:5]} | {(ckpt.namespace, hashes[ckpt_block])}
    s = Scheduler(layout, FakeRpc(stored), "inst")
    req = Req("r", 25, hashes)
    # Full blocks back 20 tokens; resume at the only checkpoint, however far
    # below, and have vLLM commit a checkpoint at 20 for the next request.
    assert s.get_num_new_matched_tokens(req, 0) == (expected, True)
    assert req.shared_prefix_boundary == 20


def test_loading_polls_only_what_is_still_loading():
    layout = _layout(FULL4, CKPT4)
    full, ckpt = layout.namespaces
    hashes = _hashes(6)
    stored = {(full.namespace, h) for h in hashes[:5]} | {(ckpt.namespace, hashes[3])}
    rpc = FakeRpc(stored)
    rpc.loading, rpc.members_loading = 1, 2
    s = Scheduler(layout, rpc, "inst")
    req = Req("r", 25, hashes)
    for _ in range(3):
        assert s.get_num_new_matched_tokens(req, 0) == (None, False)
    assert s.get_num_new_matched_tokens(req, 0) == (16, True)
    # The prefix answer is kept while the checkpoint query loads, and each
    # namespace polls under one query id, so the server's fetch can finish.
    assert [prefix for prefix, _ in rpc.queries] == [True, True, False, False, False]
    assert rpc.query_ids == ["", "q1", "", "q3", "q3"] and len(rpc.leases) == 2
    # One prefetch, as v1 counts them: the 4 full blocks and 1 checkpoint load.
    stats = s.take_stats()
    assert stats["prefetch_blocks"] == [5] and "pending_prefetches" not in stats


def test_drifted_loading_query_is_answered_then_dropped():
    layout = _layout(FULL4)
    rpc = FakeRpc({(layout.namespaces[0].namespace, h) for h in _hashes(4)})
    rpc.loading = 1
    s = Scheduler(layout, rpc, "inst")
    req = Req("r", 17, _hashes(4))
    assert s.get_num_new_matched_tokens(req, 0) == (None, False)
    # vLLM's own hit grew meanwhile: the old query is polled with its keys,
    # its lease released, and the new keys asked.
    assert s.get_num_new_matched_tokens(req, 4) == (12, True)
    assert rpc.queries == [(True, 4), (True, 4), (True, 3)] and len(rpc.leases) == 1


def test_failed_load_recomputes_and_never_loads_again():
    layout = _layout(FULL4)
    rpc = FakeRpc({(layout.namespaces[0].namespace, h) for h in _hashes(4)})
    s = Scheduler(layout, rpc, "inst")
    req = Req("r", 17, _hashes(4))
    assert s.get_num_new_matched_tokens(req, 0) == (16, True)
    # vLLM handed a null block where a full block must load: plan fails.
    s.update_state_after_alloc(req, ([1, 2, 0, 4],), 16)
    (lp,) = s.take_plan().loads
    assert lp.fail and not rpc.leases
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)


def test_saves_pin_blocks_until_every_worker_saved():
    layout = _layout(FULL4, CKPT4)
    rpc = FakeRpc(set())
    s = Scheduler(layout, rpc, "inst")
    pinned: list[list[int]] = []
    s.bind_pins(lambda ids: pinned.append(ids) or ids, lambda tok: pinned.remove(tok))
    req = Req("r", 9, _hashes(2))
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)
    s.plan_saves([("r", 0, 9)], lambda _: ([5, 6, 7], [0, 0, 0]), {"r": [(1, 31, 8)]})
    plan = s.take_plan()
    assert sorted(gpu for _, _, gpu in plan.saves) == [5, 6, 31]
    assert s.has_pending_saves() and pinned == [[5, 6, 31]]
    s.on_saved_through(plan.step)
    assert not s.has_pending_saves() and not pinned


# ------------------------------------------------------------------ worker
class _Rpc:
    def __init__(self):
        self.registered = None
        self.saved = []
        self.loaded = []

    def register_context_v2(self, instance_id, metas, regions, backend):
        self.registered = (metas, regions, backend)

    def save_objects(self, instance_id, rank, writes):
        self.saved.append(writes)

    def load_objects(self, instance_id, rank, device_id, shm, objects, leases):
        self.loaded.append((len(objects), sorted(leases)))


class _Ready:
    def synchronize(self):
        pass


def test_worker_registers_one_region_and_packs_spans():
    layout = Layout((NsSpec(0, FULL4, 1, ("a", "b"), b"N" * 32, 64, 1, (0,)),), 4)
    rpc = _Rpc()
    w = Worker(layout, rpc, "inst", 0, 1, 3, object, _Ready)
    regions = {"a": Region(b"h" * 64, 10_000, 100), "b": Region(b"h" * 64, 10_000, 5_000)}
    w.register({"a": "A", "b": "B"}, {"a": 64, "b": 128}, lambda t: regions[t.lower()])
    metas, regs, backend = rpc.registered
    assert len(regs) == 1 and backend == "kernel" and metas[0]["rank"] == 0
    # Two requests of one step computed the same block: one write.
    w.save(type("P", (), {"saves": [(0, b"k" * 32, 2), (0, b"k" * 32, 2)], "step": 0})())
    w.shutdown()
    ((key, region, spans),) = rpc.saved[0]
    assert key == (b"N" * 32, b"k" * 32) and region == 0
    assert struct.unpack("<4Q", spans) == (100 + 2 * 64, 64, 5_000 + 2 * 128, 64)
    assert w.saved_through() == 0


class _State:
    def __init__(self, state: int):
        self.state = state

    def shm_name(self):
        return "shm"

    def is_ready(self):
        return True

    def get_state(self):
        return self.state


def test_worker_loads_a_step_in_one_rpc_and_fails_it_as_a_whole():
    layout = Layout((NsSpec(0, FULL4, 1, ("a",), b"N" * 32, 64, 1, (0,)),), 4)
    rpc = _Rpc()
    states = [_State(1), _State(-1)]
    w = Worker(layout, rpc, "inst", 0, 1, 0, lambda: states.pop(0), _Ready)
    w.register({"a": "A"}, {"a": 64}, lambda t: Region(b"h" * 64, 10_000, 0))
    step = StepPlan(
        loads=[
            LoadPlan("r1", [b"l1"], [(0, b"x" * 32, 1)]),
            LoadPlan("r2", [b"l2"], [(0, b"y" * 32, 2), (0, b"z" * 32, 3)]),
            LoadPlan("r3", fail=True),
        ]
    )
    w.start(step)
    assert rpc.loaded == [(3, [b"l1", b"l2"])]
    assert w.poll() == ({"r1", "r2", "r3"}, {"r3"})
    w.start(StepPlan(loads=[LoadPlan("r4", [b"l4"], [(0, b"w" * 32, 4)])]))
    assert w.poll() == ({"r4"}, {"r4"})
    stats = w.take_stats()
    assert stats["load_blocks"] == [3, 1]
    assert (stats["load_success_count"], stats["load_failure_count"]) == (1, 1)
    w.shutdown()
