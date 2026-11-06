"""Contract of ``pegaflow.connector_v2.core`` (default unit gate, source-only).

Drives the scheduler/worker state machines through a stub adapter, a fake
server client and ``StubSemantics``, a small pure-Python model of
``pegaflow.semantics`` injected at the boundary, so no vLLM, torch, native
extension or server is needed. Cases marked ``native`` re-run against the real
``pegaflow.semantics`` and skip without the extension;
``test_stub_matches_native`` keeps the stub honest.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import sys
import types
from dataclasses import dataclass, field

import pytest

from pegaflow.connector_v2 import adapters
from pegaflow.connector_v2.adapters import LayerDecl, LayerView, Parallel, Unsupported
from pegaflow.connector_v2.core import (
    Scheduler,
    StepPlan,
    Worker,
    build_layout,
    may_zero_or_copy_blocks,
)
from pegaflow.connector_v2.core.layout import StorageRanks
from pegaflow.connector_v2.core.scheduler import GroupSave, LoadPlan
from pegaflow.connector_v2.core.worker import _wire_bytes
from pegaflow.connector_v2.policy import Policy, SaveItem, load_policy

BT = 4


# --------------------------------------------------------------- semantics stub
@dataclass(frozen=True)
class _Group:
    kind: str
    block_tokens: int
    window_tokens: int | None = None

    @staticmethod
    def full(bt: int) -> _Group:
        return _Group("full", bt)

    @staticmethod
    def window(bt: int, tokens: int) -> _Group:
        return _Group("window", bt, tokens)

    @staticmethod
    def checkpoint(bt: int) -> _Group:
        return _Group("checkpoint", bt)

    @staticmethod
    def scratch(bt: int) -> _Group:
        return _Group("scratch", bt)


@dataclass(frozen=True)
class _Shard:
    index: int = 0
    count: int = 1

    @staticmethod
    def replicated() -> _Shard:
        return _Shard()

    @staticmethod
    def heads(index: int, count: int) -> _Shard:
        return _Shard(index, count)


@dataclass(frozen=True)
class _Region:
    layer: str
    group: int
    segments: list[int]
    shard: _Shard


def _range(lo: int, hi: int) -> tuple[int, int]:
    return (lo, hi) if lo < hi else (0, 0)


def _needed(g: _Group, p: int) -> tuple[int, int]:
    n = p // g.block_tokens
    if g.kind == "full":
        return _range(0, n)
    if g.kind == "window":
        return _range(max(p - g.window_tokens, 0) // g.block_tokens, n)
    if g.kind == "checkpoint":
        return _range(max(n - 1, 0), n)
    return (0, 0)


def _resolve(groups, present, prompt_len: int) -> int:
    steps = [g.block_tokens for g in groups if g.kind != "scratch"]
    if not steps:
        return 0
    have = [set(s) for s in present]
    best = 0
    for p in range(math.lcm(*steps), prompt_len, math.lcm(*steps)):
        if all(set(range(*_needed(g, p))) <= h for g, h in zip(groups, have, strict=True)):
            best = p
    return best


def _saveable(g: _Group, before: int, after: int) -> tuple[int, int]:
    bt = g.block_tokens
    if after <= before:
        return (0, 0)
    if g.kind in ("full", "window"):
        return _range(before // bt, after // bt)
    if g.kind == "checkpoint" and after % bt == 0:
        return _range(after // bt - 1, after // bt)
    return (0, 0)


@dataclass(frozen=True)
class _Fragment:
    layer: str
    part: int
    holders: list[int]

    @property
    def writer(self) -> int:
        return self.holders[0]


@dataclass(frozen=True)
class _SealPlan:
    plan: dict[int, list[_Fragment]]

    @property
    def groups(self) -> list[int]:
        return sorted(self.plan)

    def fragments(self, group: int) -> list[_Fragment]:
        return self.plan[group]

    def slot_of(self, rank: int, layer: str) -> tuple[int, int] | None:
        for g, frags in self.plan.items():
            for slot, f in enumerate(frags):
                if f.layer == layer and rank in f.holders:
                    return g, slot
        return None


def _seal_plan(ranks):
    """The checks the core relies on (equal segments, every part held) and
    the fragment order: layer name, then part; holders ascending."""
    layers: dict[str, tuple[int, list[int], int, dict[int, set[int]]]] = {}
    for rank, regions in ranks:
        for r in regions:
            g, segs, count, parts = layers.setdefault(
                r.layer, (r.group, r.segments, r.shard.count, {})
            )
            if segs != r.segments or count != r.shard.count:
                raise ValueError(f"layer {r.layer}: segment sizes differ")
            parts.setdefault(r.shard.index, set()).add(rank)
    plan: dict[int, list[_Fragment]] = {}
    for layer, (g, _, count, parts) in sorted(layers.items()):
        if set(parts) != set(range(count)):
            raise ValueError(f"layer {layer}: missing part")
        plan.setdefault(g, []).extend(_Fragment(layer, p, sorted(parts[p])) for p in range(count))
    return _SealPlan(plan)


StubSemantics = types.SimpleNamespace(
    Group=_Group,
    Shard=_Shard,
    Region=_Region,
    needed=_needed,
    resolve=_resolve,
    saveable=_saveable,
    seal_plan=_seal_plan,
)


def _native():
    ext = pytest.importorskip("pegaflow.pegaflow", exc_type=ImportError)
    if getattr(ext, "__file__", None) is None:  # file-less stub from tests/unit_stubs.py
        pytest.skip("native extension not built (unit stub installed)")
    from pegaflow import semantics

    return semantics


@pytest.fixture
def sem(request):
    return _native() if getattr(request, "param", "stub") == "native" else StubSemantics


BOTH = pytest.mark.parametrize("sem", ["stub", "native"], indirect=True)


# ------------------------------------------------------------- adapter + fakes
@dataclass(frozen=True)
class StubFullSpec:
    block_size: int = BT
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class StubWindowSpec:
    block_size: int = BT
    window: int = 7
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class StubCheckpointSpec:
    block_size: int = BT
    segments: tuple[int, ...] = (64, 64)
    backend: str = "direct"


@dataclass(frozen=True)
class MysterySpec:
    block_size: int = BT


class _StubAdapter:
    def __init__(self, semantics):
        self.sem = semantics

    def group(self, spec):
        if isinstance(spec, StubWindowSpec):
            return self.sem.Group.window(spec.block_size, spec.window)
        if isinstance(spec, StubCheckpointSpec):
            return self.sem.Group.checkpoint(spec.block_size)
        return self.sem.Group.full(spec.block_size)

    def layer(self, spec, layer_name, par):
        return LayerDecl(
            spec.segments, self.sem.Shard.heads(par.tp_rank, par.tp_size), spec.backend
        )

    def view(self, spec, layer_name, kv):
        return LayerView(kv, num_blocks=16, bytes_per_block=64, kv_stride_bytes=1024)


@pytest.fixture(autouse=True)
def stub_adapter(monkeypatch, sem):
    module = types.ModuleType(f"{adapters.__name__}._test_stub")
    module.ADAPTER = _StubAdapter(sem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("StubFullSpec", "StubWindowSpec", "StubCheckpointSpec"):
        monkeypatch.setitem(adapters._REGISTRY, name, "_test_stub")


def _kv_cache_config(*specs):
    groups = [
        types.SimpleNamespace(kv_cache_spec=s, layer_names=[f"layers.{i}.attn"])
        for i, s in enumerate(specs)
    ]
    return types.SimpleNamespace(
        transfer_group_ids=tuple(range(len(specs))),
        kv_cache_groups=groups,
        hash_block_size=BT,
    )


def _layout(sem, *specs, par=None):
    specs = specs or (StubWindowSpec(), StubFullSpec())
    par = par or Parallel()
    return build_layout(_kv_cache_config(*specs), par, {"model": "m"}, semantics=sem)


@dataclass
class Ready:  # QueryReady
    num_hit_blocks: int
    lease: bytes
    hit_positions: list[int] = field(default_factory=list)


class Loading:  # QueryLoading
    pass


class FakeRpc:
    """``answers[sid]`` is a Ready/Loading or an exception to raise."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.released: list[bytes] = []
        self.queried: list[int] = []  # group_id of every query_prefetch call
        self.loads: list[list] = []
        self.load_ok: dict[bytes, bool] = {}
        self.registered: dict | None = None

    def query_prefetch(self, instance_id, hashes, req_id, wait_for_full_prefix, group_id):
        self.queried.append(group_id)
        ans = self.answers[group_id]
        if isinstance(ans, Exception):
            raise ans
        return ans

    def release(self, lease):
        self.released.append(lease)

    def load(self, instance_id, tp_rank, device_id, shm, layers, loads):
        self.loads.append(loads)
        if not all(self.load_ok.get(lease, True) for lease, _ in loads):
            return False, "query lease is unknown or expired"
        return True, ""

    def register_context_batch(self, *args):
        names = ["instance_id", "namespace", "tp_rank", "pp_rank", "tp_size", "world_size"]
        names += ["device_id", "names", "wrappers", "num_blocks", "bytes_per_block"]
        names += ["kv_strides", "segments", "backend", "page_first", "group_ids"]
        self.registered = dict(zip(names, args, strict=True))
        return True, ""

    def save(self, *args):
        return True, ""


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _req(num_tokens=13, n_hashes=3, req_id="r"):
    hashes = [f"h{i}".encode() for i in range(n_hashes)]
    return types.SimpleNamespace(request_id=req_id, num_tokens=num_tokens, block_hashes=hashes)


def _sched(sem, answers, policy=None, layout=None, **kw):
    rpc = FakeRpc(answers)
    layout = layout or _layout(sem)
    sched = Scheduler(layout, policy or Policy(), rpc, "inst", clock=Clock(), semantics=sem, **kw)
    return sched, rpc


# Full group (sid 0, prefix query) + window(4, 7) group (sid 1, membership).
# 13-token prompt: 3 hashed blocks; resume at 12 needs full {0,1,2}, window {1,2}.
HIT = {0: Ready(3, b"L0"), 1: Ready(3, b"L1", [0, 1, 2])}


class _Policy(Policy):
    def __init__(self, accept=None, choose=None):
        super().__init__(loading_wait_ms=0)
        self._accept, self._choose = accept, choose

    def accept(self, p, present):
        return self._accept(p, present) if self._accept else p

    def choose_saves(self, items):
        return self._choose(items) if self._choose else items


def _mutate_present(p, present):
    present[1].add(0)  # "assume the block is there"
    return p


def _raise(*_):
    raise RuntimeError("policy bug")


# ----------------------------------------------------------------------- hits
@BOTH
@pytest.mark.parametrize(
    "answers,policy,num_computed,expected,released",
    [
        pytest.param(HIT, Policy(), 4, (8, True), [], id="hit"),
        pytest.param(
            {0: Loading(), 1: Ready(3, b"L1", [0, 1, 2])},
            Policy(loading_wait_ms=100),
            0,
            (None, False),
            [b"L1"],
            id="loading-retries",
        ),
        pytest.param(
            {0: Ready(3, b"L0"), 1: RuntimeError("server down")},
            Policy(),
            0,
            (0, False),
            [b"L0"],
            id="server-error-is-miss",
        ),
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(1, b"L1", [2])},
            Policy(),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="window-gap-is-miss",
        ),
        # Window only has block 0: resolve lowers p to 4, both leases kept.
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(1, b"L1", [0])},
            Policy(),
            0,
            (4, True),
            [],
            id="window-short-lowers-p",
        ),
        # M1: a policy cannot widen the hit by mutating what it is shown.
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(2, b"L1", [1, 2])},
            _Policy(accept=_mutate_present),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="policy-mutates-present",
        ),
        # Returning more than semantics allows is capped at the legal p.
        pytest.param(
            HIT, _Policy(accept=lambda p, _: 10**6), 0, (12, True), [], id="policy-widens"
        ),
        # Lowering p is re-checked: at p=8 the window needs block 0 (absent).
        pytest.param(
            {0: Ready(3, b"L0"), 1: Ready(2, b"L1", [1, 2])},
            _Policy(accept=lambda p, _: 8),
            0,
            (0, False),
            [b"L0", b"L1"],
            id="policy-lowers-into-gap",
        ),
        # Lowering to a legal p keeps only the leases that p still needs.
        pytest.param(HIT, _Policy(accept=lambda p, _: 8), 0, (8, True), [], id="policy-lowers"),
        pytest.param(
            HIT, _Policy(accept=lambda p, _: 6), 0, (0, False), [b"L0", b"L1"], id="policy-off-step"
        ),
        # S6: a raising policy is a miss with every lease released, not a crash.
        pytest.param(
            HIT, _Policy(accept=_raise), 0, (0, False), [b"L0", b"L1"], id="policy-raises"
        ),
    ],
)
def test_matched_tokens(sem, answers, policy, num_computed, expected, released):
    sched, rpc = _sched(sem, answers, policy)
    assert sched.get_num_new_matched_tokens(_req(), num_computed) == expected
    assert sorted(rpc.released) == released


def test_semantics_error_releases_leases():
    def boom(*_):
        raise OverflowError("bad group")

    sem = types.SimpleNamespace(**{**vars(StubSemantics), "resolve": boom})
    sched, rpc = _sched(sem, HIT, layout=_layout(StubSemantics))
    assert sched.get_num_new_matched_tokens(_req(), 0) == (0, False)
    assert sorted(rpc.released) == [b"L0", b"L1"]


def test_hit_lifecycle(sem):
    """Lease needed blocks only -> load plan; an unallocated hit is released on finish."""
    # vLLM group 0 is the window group: the full group must still become sid 0.
    layout = _layout(sem)
    assert [g.gid for g in layout.groups] == [1, 0]
    sched, rpc = _sched(sem, HIT, layout=layout)
    req = _req()

    assert sched.get_num_new_matched_tokens(req, 4) == (8, True)
    assert sched.loaded_group_ids("r") == (0, 1)
    window_ids, full_ids = [0, 21, 22, 23], [10, 11, 12, 13]
    sched.update_state_after_alloc(req, (window_ids, full_ids), 8)
    (plan,) = sched.take_plan().loads
    assert not plan.fail
    # Block 0 of both groups is already on the GPU (num_computed=4): not loaded.
    assert plan.loads == [
        (b"L0", [[None, 11, 12], [None, None, None]]),
        (b"L1", [[None, None, None], [None, 21, 22]]),
    ]
    assert rpc.released == []  # leases are consumed by the workers' load

    # Hit that never gets allocated (e.g. aborted while waiting).
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    sched.request_finished("r")
    assert sorted(rpc.released) == [b"L0", b"L1"]

    # Loading wait expiry falls back to a miss.
    rpc.answers = {0: Loading(), 1: HIT[1]}
    rpc.released.clear()
    sched = Scheduler(
        layout, Policy(loading_wait_ms=100), rpc, "inst", clock=Clock(), semantics=sem
    )
    assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
    sched._clock.now += 0.2
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


def _other(num_tokens=13, n_hashes=3, req_id="b", diverge_at=None, skip_reading=False):
    req = _req(num_tokens, n_hashes, req_id)
    if diverge_at is not None:
        req.block_hashes[diverge_at:] = [b"x%d" % i for i in range(diverge_at, n_hashes)]
    req.skip_reading_prefix_cache = skip_reading
    return req


@BOTH
@pytest.mark.parametrize(
    "specs,kw,other,num_computed,event,expected",
    [
        # vLLM caches a's blocks only once its async load lands: b waits
        # (no query, no lease) instead of loading its own copy.
        pytest.param((StubFullSpec(),), {}, {}, 0, None, None, id="waits"),
        pytest.param((StubFullSpec(),), {}, {}, 0, "landed", (12, True), id="landed"),
        pytest.param((StubFullSpec(),), {}, {}, 0, "aborted", (12, True), id="loader-aborted"),
        pytest.param((StubFullSpec(),), {}, {}, 0, "expired", (12, True), id="wait-expired"),
        # Refcounted: another load of the same blocks is still in flight.
        pytest.param((StubFullSpec(),), {}, {}, 0, "one-of-two-landed", None, id="two-loaders"),
        # The wait clock restarts once the request was allocated: after an
        # expired wait, a load and a preemption it waits again.
        pytest.param((StubFullSpec(),), {}, {}, 0, "requeued", None, id="requeued-waits"),
        # The multi-turn shape: a loads blocks 1-2 past its local hit, b has
        # the same local hit, so a's load gives b a resume point at 12.
        pytest.param(
            (StubFullSpec(),), {"a_local": 4}, {}, 4, None, None, id="local-plus-inflight"
        ),
        # vLLM ignores the hit (and a None) of a request skipping its prefix cache.
        pytest.param(
            (StubFullSpec(),), {}, {"skip_reading": True}, 0, None, (12, True), id="skip-reading"
        ),
        # vLLM's local hit already covers the in-flight blocks: nothing to wait for.
        pytest.param((StubFullSpec(),), {}, {}, 12, None, (0, False), id="local-covers"),
        # Sharing only block 0 still waits: vLLM then serves 4 tokens locally.
        pytest.param((StubFullSpec(),), {}, {"diverge_at": 1}, 0, None, None, id="shares-a-prefix"),
        pytest.param(
            (StubFullSpec(),), {}, {"diverge_at": 0}, 0, None, (12, True), id="no-overlap"
        ),
        # A sync load's blocks are cached in its own step: never waited for.
        pytest.param(
            (StubFullSpec(),), {"sync_loads": True}, {}, 0, None, (12, False), id="sync-load"
        ),
        pytest.param(
            (StubFullSpec(),), {"prefix_caching": False}, {}, 0, None, (12, True), id="no-caching"
        ),
        pytest.param(
            (StubFullSpec(),),
            {"policy": Policy(loading_wait_ms=0)},
            {},
            0,
            None,
            (12, True),
            id="wait-disabled",
        ),
        # Checkpoint: a loads the state at 12 only. b shares blocks 0-1 but
        # not 2, so a's load gives b no resume point (no state at 4 or 8).
        pytest.param(
            (StubCheckpointSpec(), StubFullSpec()), {}, {}, 0, None, None, id="checkpoint-waits"
        ),
        pytest.param(
            (StubCheckpointSpec(), StubFullSpec()),
            {},
            {"diverge_at": 2},
            0,
            None,
            (0, False),
            id="checkpoint-no-gain",
        ),
    ],
)
def test_waits_for_inflight_loads_of_its_blocks(
    sem, specs, kw, other, num_computed, event, expected
):
    """Mirrors vLLM's OffloadingConnector ``_chunks_being_loaded``."""
    layout = _layout(sem, *specs)
    checkpoint = len(specs) == 2
    answers = {0: Ready(3, b"F")}
    if checkpoint:
        answers[1] = Ready(1, b"C", [2])
    rpc = FakeRpc(answers)
    kw = dict(kw)
    a_local = kw.pop("a_local", 0)
    policy = kw.pop("policy", Policy(sync_load_tokens=0 if not kw.get("sync_loads") else 256))
    sched = Scheduler(layout, policy, rpc, "inst", clock=Clock(), semantics=sem, **kw)
    ids = ([0, 0, 33], FULL_IDS) if checkpoint else (FULL_IDS,)

    def load(*req_ids):  # look all up first: none waits for another's load
        reqs = [_req(req_id=r) for r in req_ids]
        matched = [sched.get_num_new_matched_tokens(r, a_local) for r in reqs]
        for r, m in zip(reqs, matched, strict=True):
            sched.update_state_after_alloc(r, ids, m[0])
        assert not any(lp.fail for lp in sched.take_plan().loads)

    load(*(("a", "a2") if event == "one-of-two-landed" else ("a",)))
    b = _other(**other)
    if event in ("landed", "one-of-two-landed"):
        sched.on_loads_done({"a"})
    elif event == "aborted":
        sched.request_finished("a")
    elif event in ("expired", "requeued"):
        assert sched.get_num_new_matched_tokens(b, num_computed) == (None, False)
        sched._clock.now += 1.0
    if event == "requeued":
        assert sched.get_num_new_matched_tokens(b, num_computed) == (12, True)
        sched.update_state_after_alloc(b, ids, 12)
        sched.take_plan()
        sched.on_loads_done({"a", "b"})
        load("c")  # b, preempted, is looked up again while c loads its blocks
    if checkpoint and other.get("diverge_at") == 2:
        rpc.answers = {0: Ready(2, b"F2"), 1: Ready(0, b"")}
    rpc.queried.clear()
    got = sched.get_num_new_matched_tokens(b, num_computed)
    deferred = {"expired": 1, "requeued": 2}.get(event, 1 if expected is None else None)
    if expected is None:
        assert (got, rpc.queried) == ((None, False), [])
    else:
        assert got == expected
    assert sched.hit_stats.get("deferred") == deferred


def test_inflight_wait_restarts_prefetch_wait(sem):
    """The in-flight wait and the prefetch wait each get ``loading_wait_ms``:
    a prefetch clock left from before a deferral does not expire at once."""
    layout = _layout(sem, StubFullSpec())
    rpc = FakeRpc({0: Loading()})
    sched = Scheduler(layout, Policy(sync_load_tokens=0), rpc, "inst", clock=Clock(), semantics=sem)
    b = _other()
    assert sched.get_num_new_matched_tokens(b, 0) == (None, False)  # prefetching
    rpc.answers = {0: Ready(3, b"F")}
    a = _req(req_id="a")
    sched.update_state_after_alloc(a, (FULL_IDS,), sched.get_num_new_matched_tokens(a, 0)[0])
    sched._clock.now += 0.9
    assert sched.get_num_new_matched_tokens(b, 0) == (None, False)  # waits for a
    sched._clock.now += 0.2
    sched.on_loads_done({"a"})  # vLLM evicted a's blocks meanwhile: no local hit
    rpc.answers = {0: Loading()}
    assert sched.get_num_new_matched_tokens(b, 0) == (None, False)


FULL_IDS, WINDOW_IDS = [10, 11, 12, 13], [20, 21, 22, 23]


@pytest.mark.parametrize(
    "ext,ids,fail,loads,released",
    [
        pytest.param(0, (WINDOW_IDS, FULL_IDS), None, None, [b"L0", b"L1"], id="no-external"),
        pytest.param(16, (WINDOW_IDS, FULL_IDS), True, [], [b"L0", b"L1"], id="more-than-planned"),
        # vLLM trimmed the hit to 8 tokens: re-plan from the held leases (S1).
        pytest.param(
            8,
            (WINDOW_IDS, FULL_IDS),
            False,
            [
                (b"L0", [[10, 11, None], [None, None, None]]),
                (b"L1", [[None, None, None], [20, 21, None]]),
            ],
            [],
            id="trimmed-replans",
        ),
        pytest.param(6, (WINDOW_IDS, FULL_IDS), True, [], [b"L0", b"L1"], id="trimmed-off-step"),
        pytest.param(12, ([20, 0, 22, 23], FULL_IDS), True, [], [b"L0", b"L1"], id="null-block"),
        pytest.param(12, (WINDOW_IDS, [10, 11]), True, [], [b"L0", b"L1"], id="short-block-table"),
    ],
)
def test_alloc(ext, ids, fail, loads, released):
    sched, rpc = _sched(StubSemantics, HIT)
    req = _req()
    assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
    sched.update_state_after_alloc(req, ids, ext)
    plans = sched.take_plan().loads
    if fail is None:
        assert plans == []
    else:
        assert [(p.fail, p.loads) for p in plans] == [(fail, loads)]
    assert sorted(rpc.released) == released
    # A second call (allowed by vLLM) finds no hit and does nothing.
    sched.update_state_after_alloc(req, ids, ext)
    assert sched.take_plan().loads == []


@pytest.mark.parametrize(
    "sync_loads,threshold,num_computed,load_state,expected",
    [
        # Short hit: loaded before the forward, nothing reported on success.
        pytest.param(True, 256, 0, 1, ((12, False), set(), set()), id="sync-ok"),
        # S2: no deadline. Until the server signals, its copy may still land,
        # so a slow load is neither failed (vLLM would recompute into the
        # blocks) nor done; the worker clock jumps 60 s per read here.
        pytest.param(True, 256, 0, (5, 1), ((12, False), set(), set()), id="sync-slow-ok"),
        # A failed sync load cannot be a failed receive (vLLM already counts
        # the tokens): every GPU block holding hit tokens is reported invalid.
        pytest.param(True, 256, 4, -1, ((8, False), set(), {11, 12}), id="sync-failed"),
        pytest.param(True, 8, 0, 1, ((12, True), {"r"}, set()), id="long-hit-async"),
        # Layouts vLLM cannot recover a failed sync load for stay async.
        pytest.param(False, 256, 0, 1, ((12, True), {"r"}, set()), id="layout-async"),
    ],
)
def test_sync_loads(sync_loads, threshold, num_computed, load_state, expected):
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"L0")})
    sched = Scheduler(
        layout,
        Policy(sync_load_tokens=threshold),
        rpc,
        "inst",
        clock=Clock(),
        semantics=StubSemantics,
        sync_loads=sync_loads,
    )
    req = _req()
    matched = sched.get_num_new_matched_tokens(req, num_computed)
    sched.update_state_after_alloc(req, ([10, 11, 12, 13],), matched[0])
    polls, load_state = load_state if isinstance(load_state, tuple) else (0, load_state)
    state = FakeLoadState(state=load_state, polls=polls)
    worker = _worker(StubSemantics, rpc, [state], layout=layout, clock=JumpClock(60.0))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.start(sched.take_plan())
    _, recving, _ = worker.poll()
    assert (matched, recving, worker.take_invalid()) == expected
    assert sched.hit_stats.get("sync_planned") == (None if matched[1] else 1)
    worker.shutdown()


@pytest.mark.parametrize(
    "loads,order,stats,invalid",
    [
        # A sync load sent after an async one waits for its copy as well.
        pytest.param(
            ["long", "short", "long2"],
            [b"L1", b"L0", b"L2"],
            (1, 1, 1.0, 1, 3.0),
            set(),
            id="reordered",
        ),
        pytest.param(
            ["short", "long"], [b"L1", b"L0"], (1, 0, 1.0, 1, 3.0), set(), id="already-first"
        ),
        # A sync load whose RPC failed is invalid, never waited for.
        pytest.param(
            ["long", "bad"], [b"bad", b"L0"], (0, 1, 0.0, 1, 1.0), {2}, id="sync-rpc-failed"
        ),
        pytest.param([], [], (0, 0, 0.0, 0, 0.0), set(), id="no-loads"),
    ],
)
def test_worker_sends_sync_loads_first(loads, order, stats, invalid):
    rpc = FakeRpc()
    rpc.load_ok[b"bad"] = False
    plans = {
        "long": LoadPlan("long", [(b"L0", [[1]])]),
        "short": LoadPlan("short", [(b"L1", [[2]])], sync=True, covered=[2]),
        "bad": LoadPlan("bad", [(b"bad", [[2]])], sync=True, covered=[2]),
        "long2": LoadPlan("long2", [(b"L2", [[3]])]),
    }
    # The clock advances 1 s per read: sync_wait = 1, load_issue = 1 + 2 per sync wait.
    states = [FakeLoadState() for _ in loads]
    worker = _worker(StubSemantics, rpc, states, clock=JumpClock(1.0))
    worker.start(StepPlan(loads=[plans[k] for k in loads]))
    s = worker.save_stats
    got = (s.sync_loads, s.sync_first_steps, s.sync_wait_s, s.load_steps, s.load_issue_s)
    assert ([lo[0][0] for lo in rpc.loads], got, worker.take_invalid()) == (order, stats, invalid)
    worker.shutdown()


# Step 2 onwards, after vLLM processed step 0's outcome. "X" is shared's next
# turn: a new request whose prefix hit covers shared's blocks 10, 11, 20, 21.
OK_NEXT = [[("r", 17, 1), ("shared", 17, 1)], [("X", 16, 5)]]
FAIL_NEXT = [[("r", 4, 13), ("shared", 17, 1)], [("X", 16, 5)]]
# Finish fences after a failed sync load: only "other" saved (step 0).
ONLY_OTHER = dict.fromkeys([30, 31, 32, 33], 0)


@pytest.mark.parametrize(
    "load_state,steps,preempted,expected,fences",
    [
        pytest.param(
            1,
            OK_NEXT,
            (),
            [[13, 20, 21], [22]],
            {13: 2, 20: 2, 21: 2, **ONLY_OTHER},
            id="sync-ok",
        ),
        # vLLM (recompute policy) evicts nothing: r is rewound, but shared
        # (not rewound) built 20-21 on the bad block 11 and they stay cached
        # under correct hashes, so its next turn X would extend bad KV.
        pytest.param(-1, FAIL_NEXT, (), [[], []], ONLY_OTHER, id="sync-failed-sharer-next-turn"),
        # The rewound loader is preempted before recomputing past its failed
        # blocks: they stay hashed and freed; a later hit would reuse them.
        # The preemption's flush covers other's save: no fence left.
        pytest.param(
            -1,
            [[], [("X", 16, 5)]],
            ("r",),
            [[], []],
            {},
            id="sync-failed-loader-preempted",
        ),
    ],
)
def test_sync_load_outcome_gates_saves(load_state, steps, preempted, expected, fences):
    """M1: the server keeps the first copy of a key, so KV computed on a sync
    load is not saved before vLLM processed the load's outcome
    (``confirm_steps``: vLLM's max_concurrent_batches), and after a failed one
    nothing is saved at all (vLLM may serve KV built on it from its prefix
    cache to any later request).

    Step 0: r sync-loads blocks 1-2 and prefills to 17; other is unrelated.
    Step 1 (scheduled before step 0's output is processed): r decodes, shared
    prefills on a local prefix hit on r's blocks 0-1.
    """
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"L0")})
    sched = Scheduler(
        layout,
        Policy(),
        rpc,
        "inst",
        clock=Clock(),
        semantics=StubSemantics,
        sync_loads=True,
        confirm_steps=2,
    )
    tables = {
        "r": [10, 11, 12, 13],
        "shared": [10, 11, 20, 21],
        "other": [30, 31, 32, 33],
        "X": [10, 11, 20, 21, 22, 23],
    }
    for rid in ("r", "shared", "other"):
        sched.on_new_request(_req(17, 4, rid))
    req = sched._requests["r"]
    n, load_async = sched.get_num_new_matched_tokens(req, 4)
    assert (n, load_async) == (8, False)
    sched.update_state_after_alloc(req, (tables["r"],), n)

    def step(scheduled, preempted=()):
        sched.plan_saves(scheduled, lambda rid: (tables[rid],), preempted)
        return [b for s in sched.take_plan().saves for b in s.block_ids]

    # The load plan and step 0's saves ship in one StepPlan.
    sched.plan_saves([("r", 12, 5), ("other", 0, 17)], lambda rid: (tables[rid],))
    load_step = sched.take_plan()
    saved0 = [b for s in load_step.saves for b in s.block_ids]
    worker = _worker(StubSemantics, rpc, [FakeLoadState(state=load_state)], layout=layout)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.start(load_step)
    saved1 = step([("r", 17, 1), ("shared", 8, 9)])
    sched.on_load_errors(worker.take_invalid())  # vLLM processes step 0's output
    later = []
    for i, scheduled in enumerate(steps):
        if i == 1:
            sched.on_new_request(_req(21, 6, "X"))
        later.append(step(scheduled, preempted if i == 0 else ()))
    assert (saved0, saved1, later) == ([30, 31, 32, 33], [], expected)
    # A save held for an unconfirmed load is no fence: only blocks of chosen
    # saves are fenced at finish (on the plan step of their save).
    assert not any(sched.request_finished(rid) for rid in ("r", "shared", "other"))
    assert sched._fenced == fences
    worker.shutdown()


@pytest.mark.parametrize(
    "event,expected",
    [
        # "other" saved before the failure: its blocks may still be read by an
        # in-flight job, so a preemption must still flush and a finish must
        # still fence its blocks on that save.
        pytest.param("preempt", {"flush": True}, id="preempted-still-flushes"),
        pytest.param(
            "finish", {"fenced": dict.fromkeys([30, 31, 32, 33], 0)}, id="finished-still-fenced"
        ),
        # S1: later short hits load async (a failed async load is never cached).
        pytest.param("new-hit", {"hit": (8, True)}, id="later-loads-async"),
    ],
)
def test_lifecycle_after_saves_disabled(event, expected):
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"L0")})
    sched = Scheduler(
        layout,
        Policy(),
        rpc,
        "inst",
        clock=Clock(),
        semantics=StubSemantics,
        sync_loads=True,
        confirm_steps=2,
    )
    tables = {"r": [10, 11, 12, 13], "other": [30, 31, 32, 33]}
    for rid in ("r", "other"):
        sched.on_new_request(_req(17, 4, rid))
    n, _ = sched.get_num_new_matched_tokens(sched._requests["r"], 4)
    sched.update_state_after_alloc(sched._requests["r"], (tables["r"],), n)
    sched.plan_saves([("r", 12, 5), ("other", 0, 17)], lambda rid: (tables[rid],))
    assert [b for s in sched.take_plan().saves for b in s.block_ids] == [30, 31, 32, 33]
    sched.on_load_errors({11, 12})  # r's sync load failed

    got = {}
    if event == "preempt":
        sched.plan_saves([], lambda rid: (tables[rid],), preempted=("other",))
    elif event == "finish":
        assert sched.request_finished("other") is False
        got["fenced"] = sched._fenced
    else:
        got["hit"] = sched.get_num_new_matched_tokens(_req(17, 4, "late"), 4)
    plan = sched.take_plan()
    if event == "preempt":
        got["flush"] = plan.flush
    assert got == expected
    assert plan.saves == []


def test_shutdown_releases_unsent_plans():
    sched, rpc = _sched(StubSemantics, HIT)
    req = _req()
    sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, (WINDOW_IDS, FULL_IDS), 12)
    sched.shutdown()
    assert sorted(rpc.released) == [b"L0", b"L1"]


# ---------------------------------------------------------------------- saves
def _step_saves(sched, steps, block_ids):
    out = []
    for step in steps:
        sched.plan_saves(step, lambda rid: block_ids.get(rid))
        out.append([(s.sid, s.block_ids, s.block_hashes) for s in sched.take_plan().saves])
    return out


H = [f"h{i}".encode() for i in range(4)]


@BOTH
@pytest.mark.parametrize(
    "num_tokens,steps,block_ids,expected",
    [
        # Prefill 13 tokens in one step: full {0,1,2} and window {0,1,2}.
        pytest.param(
            13,
            [[("r", 0, 13)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [[(0, [10, 11, 12], H[:3]), (1, [20, 21, 22], H[:3])]],
            id="prefill",
        ),
        # Chunked prefill 6 + 7: block 1 becomes saveable only in the 2nd chunk.
        pytest.param(
            13,
            [[("r", 0, 6)], [("r", 6, 7)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [
                [(0, [10], H[:1]), (1, [20], H[:1])],
                [(0, [11, 12], H[1:3]), (1, [21, 22], H[1:3])],
            ],
            id="chunked",
        ),
        # Window block 0 already freed (null id): skipped, full still saved.
        pytest.param(
            13,
            [[("r", 0, 13)]],
            {"r": ([0, 21, 22, 23], FULL_IDS)},
            [[(0, [10, 11, 12], H[:3]), (1, [21, 22], H[1:3])]],
            id="null-id",
        ),
        # Async scheduling: 8 known tokens, 4 scheduled + spec tokens beyond:
        # `after` is capped at num_tokens, block 2 waits for a later step.
        pytest.param(
            10,
            [[("r", 0, 12)]],
            {"r": (WINDOW_IDS, FULL_IDS)},
            [[(0, [10, 11], H[:2]), (1, [20, 21], H[:2])]],
            id="async-capped",
        ),
        # No block ids this step: the range is offered again next step.
        pytest.param(
            13,
            [[("r", 0, 8)], [("r", 8, 5)]],
            {},
            [[], []],
            id="no-block-ids",
        ),
        pytest.param(13, [[("gone", 0, 13)]], {}, [[]], id="unknown-request"),
    ],
)
def test_saves(sem, num_tokens, steps, block_ids, expected):
    sched, _ = _sched(sem, {})
    sched.on_new_request(_req(num_tokens=num_tokens, n_hashes=num_tokens // BT))
    assert _step_saves(sched, steps, block_ids) == expected


# Checkpoint group (vLLM gid 0) + full group (gid 1, sid 0). Like vLLM's Mamba
# align table, the checkpoint row is null except the running state block.
CKPT_IDS = [0, 0, 0, 33]


@BOTH
@pytest.mark.parametrize(
    "steps,offers,expected",
    [
        # Prefill to 12 in chunks of 4: each chunk end's state is offered with
        # its block; the checkpoint saves exactly the offered block, never the
        # positional one (CKPT_IDS[b]).
        pytest.param(
            [("r", 0, 4), ("r", 4, 4), ("r", 8, 4)],
            [{"r": [(0, 31, 4)]}, {"r": [(0, 32, 8)]}, {"r": [(0, 33, 12)]}],
            [
                [(0, [10], H[:1]), (1, [31], H[:1])],
                [(0, [11], H[1:2]), (1, [32], H[1:2])],
                [(0, [12], H[2:3]), (1, [33], H[2:3])],
            ],
            id="offer-per-boundary",
        ),
        # vLLM did not retain (offer) the boundary at 4: no checkpoint save.
        pytest.param(
            [("r", 0, 4), ("r", 4, 4)],
            [{}, {"r": [(0, 32, 8)]}],
            [[(0, [10], H[:1])], [(0, [11], H[1:2]), (1, [32], H[1:2])]],
            id="unoffered-skipped",
        ),
        # Offers for another group, another request, an older boundary or an
        # unaligned one never stand in for this step's state.
        pytest.param(
            [("r", 0, 4), ("r", 4, 4)],
            [{"r": [(1, 31, 4)], "x": [(0, 39, 4)]}, {"r": [(0, 31, 4), (0, 35, 6)]}],
            [[(0, [10], H[:1])], [(0, [11], H[1:2])]],
            id="foreign-offers-ignored",
        ),
        # A step that ends off a boundary computes no checkpoint state.
        pytest.param([("r", 0, 6)], [{"r": [(0, 31, 4)]}], [[(0, [10], H[:1])]], id="off-boundary"),
    ],
)
def test_checkpoint_saves_from_offers(sem, steps, offers, expected):
    layout = _layout(sem, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(sem, {}, layout=layout)
    sched.on_new_request(_req(num_tokens=13, n_hashes=3))
    out = []
    for step, offer in zip(steps, offers, strict=True):
        sched.plan_saves([step], lambda _: (CKPT_IDS, FULL_IDS), offers=offer)
        out.append([(s.sid, s.block_ids, s.block_hashes) for s in sched.take_plan().saves])
    assert out == expected


@BOTH
@pytest.mark.parametrize(
    "num_computed,ckpt_ans,expected,loads",
    [
        # Cold: full blocks 0-2 and the checkpoint state at 12 (block 2).
        pytest.param(
            0,
            Ready(2, b"C", [0, 2]),
            (12, True),
            [(b"F", [[10, 11, 12], [None] * 3]), (b"C", [[None, None], [None, 33]])],
            id="cold",
        ),
        # Local hit covers 8 tokens: full loads only block 2; the state at 12
        # is loaded (vLLM drops its own local Mamba state for an external hit).
        pytest.param(
            8,
            Ready(2, b"C", [0, 2]),
            (4, True),
            [(b"F", [[None, None, 12], [None] * 3]), (b"C", [[None, None], [None, 33]])],
            id="after-local-hit",
        ),
        # Only the state at 4 is stored: resolve falls back to p=4.
        pytest.param(
            0,
            Ready(1, b"C", [0]),
            (4, True),
            [(b"F", [[10, None, None], [None] * 3]), (b"C", [[None], [33]])],
            id="older-checkpoint",
        ),
    ],
)
def test_checkpoint_loads_state_at_p(sem, num_computed, ckpt_ans, expected, loads):
    layout = _layout(sem, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(sem, {0: Ready(3, b"F"), 1: ckpt_ans}, layout=layout)
    req = _req()
    matched = sched.get_num_new_matched_tokens(req, num_computed)
    assert matched == expected
    p = num_computed + matched[0]
    # vLLM allocates the external hit's state block at (p - 1) // bt.
    ckpt_ids = [0] * (p // BT - 1) + [33]
    sched.update_state_after_alloc(req, (ckpt_ids, FULL_IDS), matched[0])
    (plan,) = sched.take_plan().loads
    assert (plan.fail, plan.loads) == (False, loads)


@BOTH
def test_miss_attribution_names_the_missing_group(sem):
    """Full blocks back p=12, the checkpoint only p=4: counted as missing sid 1."""
    layout = _layout(sem, StubCheckpointSpec(), StubFullSpec())
    sched, _ = _sched(sem, {0: Ready(3, b"F"), 1: Ready(1, b"C", [0])}, layout=layout)
    assert sched.get_num_new_matched_tokens(_req(), 0) == (4, True)
    sched._rpc.answers = {0: Ready(3, b"F"), 1: Ready(1, b"C", [2])}
    assert sched.get_num_new_matched_tokens(_req(req_id="s"), 0) == (12, True)
    # vLLM re-queries a request it could not allocate: counted once per
    # (request, resume point); a finished request id counts afresh.
    sched._rpc.answers = {0: Ready(3, b"F"), 1: Ready(1, b"C", [0])}
    sched._clock.now += Scheduler.HIT_REUSE_MAX_S  # too old to reuse: queried again
    sched.get_num_new_matched_tokens(_req(), 0)
    assert sched.hit_stats == {"lookups": 2, "missing=1": 1, "lost_tokens": 8}
    sched.request_finished("r")
    sched.get_num_new_matched_tokens(_req(), 0)
    assert sched.hit_stats == {"lookups": 3, "missing=1": 2, "lost_tokens": 16}


@BOTH
@pytest.mark.parametrize(
    ("specs", "num_computed", "full_ans", "queried", "released", "stats"),
    [
        # Full only (Qwen3-4B / DSV2 shape), 13 tokens: the prompt's last
        # loadable block ends at 12 = the local hit, so nothing is queried.
        pytest.param(
            (StubFullSpec(),), 12, Ready(3, b"F"), [], [], {}, id="full-only-local-hit-is-max"
        ),
        # Full only, one block past the local hit: queried as before; with a
        # single group nothing else is skipped.
        pytest.param(
            (StubFullSpec(),), 8, Ready(2, b"F"), [0], [b"F"], {}, id="full-only-one-block-past"
        ),
        # Hybrid, same block size: group 0 backs only 8 = the local hit, so
        # the checkpoint group is not queried (no lease is created for it).
        pytest.param(
            (StubCheckpointSpec(), StubFullSpec()),
            8,
            Ready(2, b"F"),
            [0],
            [b"F"],
            {},
            id="hybrid-full-bound-at-local-hit",
        ),
        # Hybrid, group 0 backs 12 > 8: the checkpoint group is queried as
        # before (its state at 4 does not help, so 12 is counted as its miss).
        pytest.param(
            (StubCheckpointSpec(), StubFullSpec()),
            8,
            Ready(3, b"F"),
            [0, 1],
            [b"F", b"C"],
            {"missing=1": 1, "lost_tokens": 4},
            id="hybrid-full-bound-past-local-hit",
        ),
        # Hybrid, checkpoint block 2 x the full block: the resume step is 8,
        # the next one (16) lies past the 13-token prompt. The full group
        # alone could reach 12, but no legal resume point beats 8: no query.
        pytest.param(
            (StubCheckpointSpec(block_size=2 * BT), StubFullSpec()),
            8,
            Ready(3, b"F"),
            [],
            [],
            {},
            id="hybrid-next-resume-step-past-prompt",
        ),
    ],
)
def test_lookup_skips_queries_that_cannot_beat_the_local_hit(
    sem, specs, num_computed, full_ans, queried, released, stats
):
    """vLLM re-queries every waiting request each scheduling pass: groups
    are not queried once no resume point past vLLM's local hit is possible
    (``semantics.resolve`` over what the unqueried groups could hold)."""
    layout = _layout(sem, *specs)
    sched, rpc = _sched(sem, {0: full_ans, 1: Ready(1, b"C", [0])}, layout=layout)
    assert sched.get_num_new_matched_tokens(_req(), num_computed) == (0, False)
    assert (rpc.queried, rpc.released) == (queried, released)
    assert sched.hit_stats == {"lookups": 1, **stats}


@pytest.mark.parametrize(
    "computed,age,reused",
    [
        pytest.param(0, 0.0, True, id="same-local-hit"),
        pytest.param(4, 0.0, False, id="local-hit-moved"),
        pytest.param(0, Scheduler.HIT_REUSE_MAX_S, False, id="too-old"),
    ],
)
def test_unallocated_hit_is_reused(computed, age, reused):
    """vLLM re-asks for a waiting head it could not allocate every scheduling
    pass. Within HIT_REUSE_MAX_S and for the same local hit, the held hit and
    its leases are offered again without a server query; otherwise they are
    released and the request is looked up afresh."""
    sched, rpc = _sched(StubSemantics, {0: Ready(3, b"F"), 1: Ready(3, b"W")})
    assert sched.get_num_new_matched_tokens(_req(), 0) == (12, True)
    queried = len(rpc.queried)
    sched._clock.now += age
    n = 12 - computed
    assert sched.get_num_new_matched_tokens(_req(), computed) == (n, True)
    assert (len(rpc.queried) == queried, sched.lookup_cost["reused"]) == (reused, int(reused))
    assert sorted(rpc.released) == ([] if reused else [b"F", b"W"])
    sched.update_state_after_alloc(_req(), ([20, 21, 22], [10, 11, 12]), n)
    assert [lp.req_id for lp in sched.take_plan().loads] == ["r"]


def test_held_hit_released_when_deferred():
    """A held hit whose request starts waiting for another request's async
    load of its blocks is released, not reused (the wait holds no lease)."""
    layout = _layout(StubSemantics, StubFullSpec())
    rpc = FakeRpc({0: Ready(3, b"B")})
    policy = Policy(sync_load_tokens=0)
    sched = Scheduler(layout, policy, rpc, "inst", clock=Clock(), semantics=StubSemantics)
    b = _other()
    assert sched.get_num_new_matched_tokens(b, 0) == (12, True)  # held: not allocated
    rpc.answers = {0: Ready(3, b"A")}
    a = _req(req_id="a")
    sched.update_state_after_alloc(a, (FULL_IDS,), sched.get_num_new_matched_tokens(a, 0)[0])
    sched.take_plan()
    rpc.queried.clear()
    assert sched.get_num_new_matched_tokens(b, 0) == (None, False)
    assert (rpc.queried, rpc.released, sched.lookup_cost["reused"]) == ([], [b"B"], 0)


def test_hit_stats_logged_when_idle_and_on_cadence():
    """The gate force-kills the engine core (no ``shutdown``): the counters
    must reach the log when the last request finishes and every HIT_LOG_S."""
    sched, _ = _sched(StubSemantics, {0: Ready(0, b""), 1: Ready(0, b"")})
    lines: list[str] = []
    # Own handler: caplog would also see propagated copies of each record.
    handler = logging.Handler(logging.INFO)
    handler.emit = lambda record: lines.append(record.getMessage())
    log = logging.getLogger("pegaflow.connector_v2.core.scheduler")
    level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        sched.get_num_new_matched_tokens(_req(), 0)
        sched.get_num_new_matched_tokens(_req(req_id="s"), 0)
        sched.request_finished("r")
        assert lines == []  # "s" still live, cadence not due
        sched._clock.now += Scheduler.HIT_LOG_S
        sched.get_num_new_matched_tokens(_req(req_id="t"), 0)
        cost = "{'calls': 3, 'rpcs': 3, 'us': 0, 'max_us': 0, 'reused': 0}"
        assert lines == [f"[PegaKVConnectorV2] hit stats {{'lookups': 3}} lookup_cost {cost}"]
        sched.request_finished("s")
        sched.request_finished("t")  # idle, but nothing new to report
        sched.shutdown()
        assert len(lines) == 1
        sched.get_num_new_matched_tokens(_req(req_id="u"), 0)
        sched.request_finished("u")  # idle with a new count
    finally:
        log.removeHandler(handler)
        log.setLevel(level)
    cost = "{'calls': 4, 'rpcs': 4, 'us': 0, 'max_us': 0, 'reused': 0}"
    assert lines[1:] == [f"[PegaKVConnectorV2] hit stats {{'lookups': 4}} lookup_cost {cost}"]


@BOTH
def test_saves_keep_one_entry_per_request(sem):
    # The server pins one region per entry: two requests saving in one step
    # must not share an entry. Groups in sid order, requests in scheduled order.
    sched, _ = _sched(sem, {})
    sched.on_new_request(_req(req_id="b", num_tokens=8, n_hashes=2))
    sched.on_new_request(_req(req_id="a"))
    ids = {"a": (WINDOW_IDS, FULL_IDS), "b": ([40, 41, 42], [30, 31, 32])}
    assert _step_saves(sched, [[("a", 0, 13), ("b", 0, 8)]], ids) == [
        [
            (0, [10, 11, 12], H[:3]),
            (0, [30, 31], H[:2]),
            (1, [20, 21, 22], H[:3]),
            (1, [40, 41], H[:2]),
        ]
    ]


def test_saves_reoffer_after_missing_block_ids():
    sched, _ = _sched(StubSemantics, {})
    sched.on_new_request(_req())
    ids: dict = {}
    assert _step_saves(sched, [[("r", 0, 8)]], ids) == [[]]
    ids["r"] = (WINDOW_IDS, FULL_IDS)
    assert _step_saves(sched, [[("r", 8, 5)]], ids) == [
        [(0, [10, 11, 12], H[:3]), (1, [20, 21, 22], H[:3])]
    ]


def _forge(items):
    a, b = items[0], items[1]
    return [SaveItem(a.req_id, a.group, a.kind, a.block, b.block_id, a.block_hash)]


def _append(items):
    items.append(items[0])
    return items


@pytest.mark.parametrize(
    "choose,expected",
    [
        pytest.param(lambda items: [], [], id="save-none"),
        pytest.param(
            lambda items: [i for i in items if i.kind == "full"], [(0, [10, 11, 12])], id="subset"
        ),
        pytest.param(lambda items: [items[0], items[0]], [(0, [10])], id="duplicate-once"),
        # M2: a forged (block, hash) pair or an in-place edit fails loudly.
        pytest.param(_forge, ValueError, id="forged-item"),
        pytest.param(_append, AttributeError, id="mutates-items"),
    ],
)
def test_choose_saves_is_validated(choose, expected):
    sched, _ = _sched(StubSemantics, {}, _Policy(choose=choose))
    sched.on_new_request(_req())
    plan = lambda: sched.plan_saves([("r", 0, 13)], lambda _: (WINDOW_IDS, FULL_IDS))  # noqa: E731
    if isinstance(expected, type):
        with pytest.raises(expected):
            plan()
        return
    plan()
    assert [(s.sid, s.block_ids) for s in sched.take_plan().saves] == expected


# --------------------------------------------------------------------- worker
class FakeLoadState:
    """``polls``: is_ready() calls answering False before it turns ready."""

    def __init__(self, ready=True, state=1, polls=0):
        self.ready, self.state, self.polls = ready, state, polls

    def shm_name(self):
        return "shm"

    def is_ready(self):
        if self.polls:
            self.polls -= 1
            return False
        return self.ready

    def get_state(self):
        return self.state


class FakeMark:
    """Compute-stream mark: ``synchronize`` blocks until the test releases it."""

    def __init__(self, gate=None):
        self.gate = gate

    def synchronize(self):
        if self.gate is not None:
            assert self.gate.wait(5), "test never released the compute mark"


class JumpClock:
    """Advances ``step`` seconds per read: any deadline in the code expires."""

    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _worker(sem, rpc, states, layout=None, mark=FakeMark, clock=None, par=None):
    it = iter(states)
    kwargs = {"clock": clock} if clock is not None else {}
    return Worker(
        layout or _layout(sem),
        rpc,
        "inst",
        par or Parallel(),
        0,
        lambda t: b"ipc",
        lambda: next(it),
        mark,
        **kwargs,
    )


def test_worker_loads_and_poll():
    rpc = FakeRpc()
    rpc.load_ok[b"bad"] = False
    pending = FakeLoadState(ready=False)
    # One state per load RPC: ok, bad-lease (unused), gpu-error, later.
    states = [FakeLoadState(), FakeLoadState(), FakeLoadState(state=-1), pending]
    clock = StepClock()
    worker = _worker(StubSemantics, rpc, states, clock=clock)
    plan = StepPlan(
        loads=[
            LoadPlan("fail", fail=True),
            LoadPlan("empty"),
            LoadPlan("ok", [(b"L0", [[1]])]),
            LoadPlan("bad-lease", [(b"bad", [[2]])]),  # S1: does not fail "ok"
            LoadPlan("gpu-error", [(b"L2", [[3]])]),
            LoadPlan("later", [(b"L3", [[4]])]),
        ]
    )
    worker.start(plan)
    assert len(rpc.loads) == 4  # one RPC per request with something to load
    clock.now = 1.0
    _, finished, failed = worker.poll()
    assert failed == {"fail", "bad-lease", "gpu-error"}
    assert finished == {"fail", "empty", "ok", "bad-lease", "gpu-error"}
    pending.ready = True
    clock.now = 3.0
    assert worker.poll() == (set(), {"later"}, set())
    assert worker.poll() == (set(), set(), set())
    # Polls until each load RPC was ready: 1 for ok and gpu-error, 2 for later;
    # seconds from the RPC to the poll that saw it ready: 1, 1 and 3.
    s = worker.save_stats
    assert (s.async_done, s.async_first_poll, s.async_polls, s.max_async_polls) == (3, 2, 4, 2)
    assert (s.async_ready_s, s.max_async_ready_s) == (5.0, 3.0)


@pytest.mark.parametrize(
    "spec,error",
    [
        pytest.param(StubFullSpec(), None, id="ok"),
        # S2: a declaration must equal what the server copies per segment.
        pytest.param(StubFullSpec(segments=(64, 32)), "!= server copy size", id="short-segment"),
        pytest.param(StubFullSpec(segments=(128,)), "!= server copy size", id="whole-block"),
        pytest.param(StubFullSpec(backend="kernel"), None, id="kernel-backend"),
    ],
)
def test_worker_register(spec, error):
    rpc = FakeRpc()
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, spec))
    kv = {"layers.0.attn": "tensor"}
    if error:
        with pytest.raises(Unsupported, match=error):
            worker.register(kv, {"layers.0.attn": spec})
        return
    worker.register(kv, {"layers.0.attn": spec})
    reg = rpc.registered
    assert (reg["names"], reg["segments"], reg["group_ids"]) == (["layers.0.attn"], [2], [0])
    assert (reg["backend"], reg["page_first"]) == (spec.backend, False)


def test_worker_save_by_storage_group():
    saved = []
    rpc = FakeRpc()
    rpc.save = lambda *args: saved.append(args[-1]) or (True, "")
    worker = _worker(StubSemantics, rpc, [])
    worker.register(
        {"layers.0.attn": "w", "layers.1.attn": "f"},
        {"layers.0.attn": StubWindowSpec(), "layers.1.attn": StubFullSpec()},
    )
    worker.save(StepPlan(saves=[GroupSave(1, [21], [b"h1"]), GroupSave(0, [11], [b"h1"])]))
    worker.save(StepPlan())
    assert worker.flush(5)
    assert saved == [[("layers.0.attn", [21], [b"h1"]), ("layers.1.attn", [11], [b"h1"])]]
    # S2: the gate log attributes G4 to save cost; 2 x (64 + 64) B segments.
    stats = worker.save_stats
    assert (stats.steps, stats.save_steps, stats.blocks, stats.bytes) == (2, 1, 2, 256)
    worker.report_save_stats()
    assert worker.save_stats.steps == 0
    worker.shutdown()


class _SaverExit(BaseException):
    """Escapes ``except Exception``: ends the saver thread."""


def _save_outcome(outcome):
    def save(calls, saves):
        calls.append(saves)
        if outcome == "rejected":  # InvalidArgument: the binding raises ValueError
            raise ValueError("save RPC failed: code: 'Client specified an invalid argument'")
        if outcome == "raises":  # any other status: a copy may still run
            raise RuntimeError("save RPC failed: code: 'Internal error'")
        if outcome == "saver-dies":
            raise _SaverExit
        return outcome == "saved", ""

    return save


def _blocked(fn, *args):
    """Run ``fn`` on a thread; return it if it is still blocked after 0.2 s."""
    import threading

    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    t.join(0.2)
    return t


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize("outcome", ["saved", "save-failed", "rejected", "raises", "saver-dies"])
def test_freed_block_is_not_written_while_a_save_reads_it(outcome):
    """vLLM gets a finished request's blocks back at once (request_finished is
    always False, no finished_sending hold), but a step that writes one of
    them while a save still reads it is fenced: the worker waits in
    handle_preemptions until that save's GPU->CPU copy is done.

    The step thread never waits on the GPU for a save: the save RPC runs
    after the compute mark. The fence clears once the save is done, stored
    or not (a reported failure, or a rejected request, is a future miss).
    S4: any other raised save RPC is not a finished copy (the server may
    still copy): the fence holds for the quiesce window, saves submitted
    meanwhile are dropped, then the job counts done and saving resumes.
    A dying saver thread stops saving for good: the fence is released
    instead of hanging, and later saves are dropped.
    """
    import threading

    gate, window = threading.Event(), threading.Event()
    rpc = FakeRpc()
    calls = []
    save = _save_outcome(outcome)
    rpc.save = lambda *args: save(calls, args[-1])
    layout = _layout(StubSemantics, StubFullSpec())
    worker = _worker(StubSemantics, rpc, [], layout=layout, mark=lambda: FakeMark(gate))
    worker._quiesce_wait = lambda s: window.wait(5)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})

    sched, _ = _sched(StubSemantics, {}, layout=layout)
    for rid in ("saver", "idle", "next"):
        sched.on_new_request(_req(req_id=rid))
    tables = {"saver": [10, 11, 12, 13], "idle": [20], "next": [30, 11, 31, 32]}
    sched.plan_saves([("saver", 0, 13), ("idle", 0, 3)], lambda rid: (tables[rid],))
    step0 = sched.take_plan()
    worker.start(step0)
    worker.save(step0)  # returns although the compute mark is not reached
    assert calls == [] and worker.save_stats.save_steps == 1
    assert sched.has_pending_saves()  # vLLM keeps stepping until it is reported

    # Both finish: neither is held, only the saved blocks are fenced.
    assert sched.request_finished("saver") is False
    assert sched.request_finished("idle") is False
    assert sched._fenced == {10: 0, 11: 0, 12: 0}

    # vLLM hands 11 to the next request, which prefills into it this step.
    sched.fence_writes([((tables["next"],), 0)])
    sched.plan_saves([], lambda rid: None)
    step1 = sched.take_plan()
    assert (step1.fence_step, step1.flush, sched._fenced) == (0, False, {})
    waiter = _blocked(worker.wait_saved, step1.fence_step)  # handle_preemptions
    assert waiter.is_alive() and calls == []  # the copy has not run: no KV write yet

    gate.set()
    raises = outcome == "raises"
    if raises:
        with worker._cv:
            assert worker._cv.wait_for(lambda: worker._quiescing, 5)
        assert waiter.join(0.2) is None and waiter.is_alive()  # the window holds the fence
        worker.save(step0)  # dropped: no RPC while the raised copy may run
        window.set()
    waiter.join(5)
    assert not waiter.is_alive() and len(calls) == 1
    s = worker.save_stats
    assert s.fences == 1 and s.max_fence_s > 0
    assert (s.failed, s.raised) == ({"saved": 0, "raises": 2}.get(outcome, 1), int(raises))
    worker.save(step1)
    sched.on_saved_through(worker.saved_through())
    assert not sched.has_pending_saves()
    dies = outcome == "saver-dies"
    assert worker._completed == (0 if dies else 1)
    if raises:  # saving resumed: the next save sends its RPC
        worker.save(step0)
        assert worker.flush(5) and len(calls) == 2
    if dies:
        worker._saver.join(5)
        assert not worker._saver.is_alive()
        worker.save(step0)
        assert worker.flush(5) and len(calls) == 1
        assert worker.save_stats.failed == 2
    worker.shutdown()


def _quiesce_rig(outcomes, monkeypatch=None):
    """Worker whose save RPCs return or raise ``outcomes`` in order. Each
    quiesce window records its length in ``windows`` and lasts until the
    test releases ``ends`` once."""
    import threading

    rpc, calls, outs = FakeRpc(), [], iter(outcomes)
    rpc.save = lambda *args: _save_outcome(next(outs))(calls, args[-1])
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    windows, ends = [], threading.Semaphore(0)

    def wait(seconds):
        windows.append(seconds)
        assert ends.acquire(timeout=5), "test never ended the quiesce window"

    worker._quiesce_wait = wait
    return worker, calls, windows, ends


def _save_plan(step, send=False):
    return StepPlan(saves=[GroupSave(0, [step], [b"h%d" % step])], step=step, send_saves=send)


def _wait_quiescing(worker):
    with worker._cv:
        assert worker._cv.wait_for(lambda: worker._quiescing, 5)


def test_quiesce_window_holds_only_the_raised_and_later_saves(monkeypatch):
    """During a window: a fence on a step before the raised job's clears at
    once; a fence on its step and a flush wait for the window; a new save
    and a held batch released meanwhile are dropped without an RPC. Then
    saving resumes."""
    one_step = _wire_bytes([("layers.0.attn", [0], [b"h0"])])
    monkeypatch.setattr(Worker, "SAVE_RPC_MAX_BYTES", one_step)  # each step sends the last
    worker, calls, windows, ends = _quiesce_rig(["saved", "raises", "saved"])
    worker._batch_steps = 100
    worker.save(_save_plan(0))  # held
    worker.save(_save_plan(1))  # sends step 0 (stored), holds step 1
    worker.save(_save_plan(2))  # sends step 1 (raises), holds step 2
    _wait_quiescing(worker)
    assert not _blocked(worker.wait_saved, 0).is_alive()
    fence = _blocked(worker.wait_saved, 1)
    flush = _blocked(worker.flush)  # releases held step 2 into the window
    assert fence.is_alive() and flush.is_alive()
    worker.save(_save_plan(3))  # dropped
    ends.release()
    fence.join(5)
    flush.join(5)
    assert not fence.is_alive() and not flush.is_alive()
    s = worker.save_stats
    assert (len(calls), windows, s.raised, s.rpcs, s.failed) == (2, [10.0], 1, 2, 3)
    worker.save(_save_plan(4))
    assert worker.flush(5) and len(calls) == 3
    worker.shutdown()


def test_quiesce_window_backs_off():
    """Each consecutive raise doubles the window up to SAVE_QUIESCE_MAX_S; a
    save RPC that stores its blocks resets it (a failed reply does not)."""
    outcomes = ["raises"] * 4 + ["save-failed", "raises", "saved", "raises"]
    worker, calls, windows, ends = _quiesce_rig(outcomes)
    for step, outcome in enumerate(outcomes):
        worker.save(_save_plan(step))
        if outcome == "raises":
            ends.release()
        assert worker.flush(5)
    assert windows == [10.0, 20.0, 40.0, 60.0, 60.0, 10.0]
    assert len(calls) == len(outcomes)
    worker.shutdown()


def test_shutdown_ends_a_quiesce_window():
    """Shutdown does not wait out a window: the saver resumes at once and
    exits, and no further save RPC is sent."""
    import time

    rpc, calls = FakeRpc(), []
    save = _save_outcome("raises")
    rpc.save = lambda *args: save(calls, args[-1])
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.save(_save_plan(0))
    _wait_quiescing(worker)
    worker.save(_save_plan(1))  # dropped
    t0 = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - t0 < Worker.SAVE_QUIESCE_S / 2
    assert not worker._saver.is_alive() and len(calls) == 1


@pytest.mark.parametrize(
    "computed,blocks,fence",
    [
        # A new request's blocks wholly below its computed tokens are prefix
        # hits on a fenced block (vLLM only reads them): no fence.
        pytest.param(8, [10, 11, 30], -1, id="prefix-hit-read-only"),
        pytest.param(4, [10, 11, 30], 0, id="past-the-hit"),
        pytest.param(0, [30, 21], 1, id="newer-save"),
        pytest.param(0, [30, 11, 21], 1, id="latest-of-two"),
        pytest.param(0, [30, 31], -1, id="unfenced"),
    ],
)
def test_fence_writes(computed, blocks, fence):
    """``fence_writes`` sets the plan's fence to the latest save step among
    the fenced blocks this step writes; the fence covers every fence up to
    that step, so later writes of those blocks are free."""
    sched, _ = _sched(StubSemantics, {}, layout=_layout(StubSemantics, StubFullSpec()))
    tables = {"a": [10, 11, 12, 13], "b": [20, 21, 22, 23]}
    for rid in tables:
        sched.on_new_request(_req(req_id=rid))
    sched.plan_saves([("a", 0, 13)], lambda rid: (tables[rid],))  # step 0
    sched.plan_saves([("b", 0, 13)], lambda rid: (tables[rid],))  # step 1
    sched.take_plan()
    sched.request_finished("a")
    sched.request_finished("b")
    fenced = {10: 0, 11: 0, 12: 0, 20: 1, 21: 1, 22: 1}
    assert sched._fenced == fenced
    sched.fence_writes([((blocks,), computed)])
    assert sched.take_plan().fence_step == fence
    assert sched._fenced == {b: s for b, s in fenced.items() if s > fence}
    sched.fence_writes([((blocks,), computed)])
    assert sched.take_plan().fence_step == -1


@pytest.mark.parametrize(
    "event,fence",
    [
        pytest.param("zeroed", 0, id="zeroed-or-copy-target"),
        pytest.param("load-target", 0, id="load-target"),
        # vLLM recomputes into a failed sync load's blocks (recompute policy).
        pytest.param("sync-load-plan-fail", 0, id="sync-load-plan-fail"),
        # A block shared through the prefix cache stays fenced whoever frees it.
        pytest.param("still-shared", 0, id="still-shared"),
        pytest.param("saved-through", -1, id="reported-saved"),
        pytest.param("flushed", -1, id="flush-covers-it"),
        pytest.param("finish-after-saved", -1, id="finish-after-saved"),
    ],
)
def test_fence_sources(event, fence):
    sync = event == "sync-load-plan-fail"
    sched, _ = _sched(
        StubSemantics,
        {0: Ready(3, b"L0")},
        policy=Policy(sync_load_tokens=256 if sync else 0),
        layout=_layout(StubSemantics, StubFullSpec()),
        sync_loads=sync,
    )
    tables = {"a": [10, 11, 12, 13], "b": [10, 11, 40, 41]}
    for rid in ("a", "b"):
        sched.on_new_request(_req(req_id=rid))
    sched.plan_saves([("a", 0, 13)], lambda rid: (tables[rid],))  # step 0
    sched.take_plan()
    if event == "finish-after-saved":
        sched.on_saved_through(0)
    elif event == "flushed":
        sched.plan_saves([], lambda rid: None, preempted=("b",))
        assert sched.take_plan().flush
    sched.request_finished("a")
    if event == "saved-through":
        sched.on_saved_through(0)
    if event == "zeroed":
        sched.fence_writes([], [30, 12])
    elif event == "load-target":
        req = _req(req_id="c")
        sched.on_new_request(req)
        assert sched.get_num_new_matched_tokens(req, 0) == (12, True)
        sched.update_state_after_alloc(req, ([30, 11, 31, 32],), 12)
    elif sync:
        req = _req(req_id="c")
        sched.on_new_request(req)
        assert sched.get_num_new_matched_tokens(req, 0) == (12, False)
        # A null target block fails the plan; 10 is still in the covered range.
        sched.update_state_after_alloc(req, ([0, 10, 31, 32],), 12)
        assert sched._plan.loads[-1].fail
    else:
        # b keeps 10-11 (shared with a, read only); then 10 is reused.
        sched.fence_writes([(([42],), 0), (([10, 11, 40, 41],), 8)])
        assert sched._plan.fence_step == -1
        sched.fence_writes([(([10, 50],), 0)])
    assert sched.take_plan().fence_step == fence


def _fence_rig(outcomes=None, policy=None):
    """Scheduler + worker on one full group; each save job waits for its own
    compute mark (``gates[i]`` for the i-th job) and returns ``outcomes[i]``."""
    import threading

    gates = [threading.Event() for _ in range(4)]
    marks = iter(gates)
    outcomes = iter(outcomes or ["saved"] * 4)
    rpc = FakeRpc()
    rpc.save = lambda *args: _save_outcome(next(outcomes))([], args[-1])
    layout = _layout(StubSemantics, StubFullSpec())
    worker = _worker(StubSemantics, rpc, [], layout=layout, mark=lambda: FakeMark(next(marks)))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker._quiesce_wait = lambda s: None
    sched, _ = _sched(StubSemantics, {}, policy=policy, layout=layout)
    tables = {"a": [10, 11, 12, 13], "b": [20, 21, 22, 23], "c": [30, 31, 32, 33]}
    for rid in tables:
        sched.on_new_request(_req(req_id=rid))

    def step(scheduled, preempted=()):
        sched.plan_saves(scheduled, lambda rid: (tables[rid],), preempted)
        plan = sched.take_plan()
        worker.start(plan)
        if plan.flush:  # handle_preemptions
            worker.flush(5)
        worker.save(plan)
        return plan

    def done(n):  # wait until the saver completed n jobs (or stopped)
        with worker._cv:
            assert worker._cv.wait_for(lambda: worker._settled(n), 5)

    return sched, worker, gates, step, done


A_FENCED, B_FENCED = {10: 0, 11: 0, 12: 0}, {20: 1, 21: 1, 22: 1}


@pytest.mark.parametrize(
    "outcomes,release,expected",
    [
        # a's job is done, b's (submitted later) is not: a's fences clear now.
        pytest.param(None, [[0], [1]], [B_FENCED, {}], id="own-job-done"),
        # One saver thread completes jobs in submit order: b's mark firing
        # first completes nothing, so "job k done" means "all <= k".
        pytest.param(None, [[1], [0]], [{**A_FENCED, **B_FENCED}, {}], id="fifo"),
        # A failed save still clears the fence (future miss).
        pytest.param(["save-failed", "saved"], [[0], [1]], [B_FENCED, {}], id="own-job-failed"),
        # A raising save RPC quiesces (its window is cut to 0 here): then a's
        # job counts done, and b's, queued behind it, is dropped without a
        # copy, so every fence clears.
        pytest.param(["raises", "saved"], [[0], [1]], [{}, {}], id="raise-quiesces"),
        # A rejected request copied nothing: a future miss, the saver goes on.
        pytest.param(["rejected", "saved"], [[0], [1]], [B_FENCED, {}], id="rejected"),
    ],
)
def test_finish_fences_on_the_requests_last_save(outcomes, release, expected):
    """A finished request's blocks are fenced on the plan step of their own
    last save, and cleared once every worker reports that step saved
    (``saved_through``), not when every later job is. A request with no
    chosen save (c: the policy dropped its blocks) fences nothing."""
    policy = _Policy(choose=lambda items: [i for i in items if i.req_id != "c"])
    sched, worker, gates, step, done = _fence_rig(outcomes, policy)
    step([("a", 0, 13)])  # step 0: job 1 saves a
    step([("a", 13, 1), ("b", 0, 13), ("c", 0, 13)])  # step 1: job 2 saves b only
    assert not any(sched.request_finished(rid) for rid in "abc")
    assert sched._fenced == {**A_FENCED, **B_FENCED}
    step([])
    got = []
    for jobs in release:
        for j in jobs:
            gates[j].set()
        # The saver can finish only jobs whose marks, and all earlier ones, fired.
        done(next(i for i, g in enumerate(gates) if not g.is_set()))
        sched.on_saved_through(worker.saved_through())
        got.append(dict(sched._fenced))
    assert got == expected
    assert worker.poll()[0] == set()  # never held: nothing is ever sending
    worker.shutdown()


def test_preempted_request_fences_on_its_last_save():
    """The preemption flush finishes a's older saves; after resuming, a fences
    on its new save only, and a write of its block waits for that save, not
    for c's later job."""
    sched, worker, gates, step, done = _fence_rig()
    step([("a", 0, 13)])  # step 0: job 1
    gates[0].set()  # lets the preemption flush finish
    plan = step([("b", 0, 13)], preempted=("a",))  # step 1: flush, job 2 (b)
    assert plan.flush and worker._completed == 1
    step([("a", 0, 13)])  # step 2: recompute, job 3
    step([("c", 0, 13)])  # step 3: job 4 (c)
    sched.request_finished("a")
    assert sched._fenced == {10: 2, 11: 2, 12: 2}
    sched.fence_writes([(([10],), 0)])
    plan = step([])
    assert plan.fence_step == 2
    gates[1].set()
    gates[2].set()
    done(3)
    waiter = _blocked(worker.wait_saved, plan.fence_step)
    assert not waiter.is_alive()  # c's job 4 is still pending
    gates[3].set()
    worker.shutdown()


def test_preempted_request_finishing_while_reloading_is_not_held():
    """A preempted request re-admitted with an async load and then aborted must
    not be reported in finished_sending: vLLM would free the blocks the load
    still copies into, then fail on the load's finished_recving."""
    layout = _layout(StubSemantics, StubFullSpec())
    sched, _ = _sched(StubSemantics, {0: Ready(3, b"L0")}, layout=layout)
    pending = FakeLoadState(ready=False)
    worker = _worker(StubSemantics, FakeRpc(), [pending], layout=layout)
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    req = _req()
    sched.on_new_request(req)
    polls = []

    def step(scheduled, tables=lambda rid: None, preempted=()):
        sched.plan_saves(scheduled, tables, preempted)
        plan = sched.take_plan()
        worker.start(plan)
        if plan.flush:
            worker.flush(5)
        worker.save(plan)
        polls.append(worker.poll())

    step([("r", 0, 13)], lambda rid: ([10, 11, 12, 13],))  # r saves
    step([], preempted=("r",))
    n, is_async = sched.get_num_new_matched_tokens(req, 0)
    assert n and is_async
    sched.update_state_after_alloc(req, ([20, 21, 22, 23],), n)
    step([])  # the load is issued
    assert sched.request_finished("r") is False  # aborted while loading
    step([])
    pending.ready = True
    step([])
    assert [sending for sending, _, _ in polls] == [set()] * 5
    assert polls[-1][1] == {"r"}
    worker.shutdown()


class StepClock:
    """Reads the time the test set; ``tick`` advances it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# Steps: block id saved that step (None = a step without saves), then the
# step's clock advance in ms. Expected: block ids per save RPC, in order.
# The staleness cap (Worker.SAVE_HOLD_MAX_S) is 10 ms here.
@pytest.mark.parametrize(
    "batch_steps,steps,expected",
    [
        pytest.param(1, [(1, 0), (2, 0), (3, 0)], [[1], [2], [3]], id="off"),
        pytest.param(3, [(1, 0), (2, 0), (3, 0), (4, 0)], [[1, 2, 3]], id="steps"),
        # Steps without saves still age the held jobs (vLLM calls
        # wait_for_save every engine step).
        pytest.param(3, [(1, 0), (None, 0), (None, 0)], [[1]], id="idle-steps-count"),
        # The cap: the step that finds the batch due joins it (sent at 12 ms).
        pytest.param(100, [(1, 4), (2, 4), (3, 4), (4, 0), (5, 0)], [[1, 2, 3, 4]], id="cap"),
        # Whichever bound comes first.
        pytest.param(8, [(1, 6), (2, 6), (3, 0)], [[1, 2, 3]], id="cap-before-steps"),
        pytest.param(2, [(1, 0), (2, 0), (3, 0)], [[1, 2]], id="steps-before-cap"),
    ],
)
def test_save_batching(batch_steps, steps, expected, monkeypatch):
    """Policy save_batch_steps: consecutive steps' saves go out as one save RPC,
    released by the step bound or the staleness cap; nothing is lost. Each
    step keeps its own layer entry in the RPC (the server pins one region per
    entry: one entry per layer for many steps strands evicted bytes)."""
    monkeypatch.setattr(Worker, "SAVE_HOLD_MAX_S", 10e-3)
    calls = []
    rpc = FakeRpc()
    rpc.save = lambda *args: calls.append(args[-1]) or (True, "")
    clock = StepClock()
    worker = Worker(
        _layout(StubSemantics, StubFullSpec()),
        rpc,
        "inst",
        Parallel(),
        0,
        lambda t: b"ipc",
        lambda: None,
        FakeMark,
        clock=clock,
        batch_steps=batch_steps,
    )
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    for block, advance_ms in steps:
        saves = [] if block is None else [GroupSave(0, [block], [b"h%d" % block])]
        worker.save(StepPlan(saves=saves))
        clock.now += advance_ms / 1e3
    worker._jobs.put(None)  # stop the saver after what was released so far
    worker._saver.join(5)
    assert [[ids for _, ids, _ in c] for c in calls] == [[[b] for b in e] for e in expected]
    assert [[h for _, _, h in c] for c in calls] == [[[b"h%d" % b] for b in e] for e in expected]
    assert all(name == "layers.0.attn" for c in calls for name, _, _ in c)
    assert worker.save_stats.rpcs == len(expected)


@pytest.mark.parametrize(
    "kv_usage,threshold,send",
    [
        pytest.param(0.3, 0.9, False, id="low"),
        pytest.param(0.9, 0.9, True, id="at-threshold"),
        pytest.param(1.0, 1.01, False, id="disabled"),
        # vLLM did not bind its cache manager: fail loud and stop batching.
        pytest.param(None, 0.9, True, id="unknown"),
    ],
)
def test_kv_pressure_sends_held_saves(kv_usage, threshold, send, caplog):
    """Under GPU KV pressure, held saves go out in the step's RPC instead of
    waiting for the batch bound: freed blocks are reused soon, and a fence on
    a held save would wait for the whole batch. The step does not wait.
    Unknown usage counts as pressure, warned once and counted in save stats."""
    layout = _layout(StubSemantics, StubFullSpec())
    sched, _ = _sched(
        StubSemantics, {}, policy=Policy(save_batch_max_kv_usage=threshold), layout=layout
    )
    with caplog.at_level("WARNING", logger="pegaflow.connector_v2.core.scheduler"):
        sched.plan_saves([], lambda rid: None, kv_usage=kv_usage)
        pressured = sched.take_plan()
        sched.plan_saves([], lambda rid: None, kv_usage=kv_usage)
    warned = [r for r in caplog.records if "usage is unavailable" in r.getMessage()]
    assert len(warned) == (1 if kv_usage is None else 0)
    assert pressured.send_saves is send and pressured.flush is False
    assert pressured.kv_usage_unknown is (kv_usage is None)

    calls = []
    rpc = FakeRpc()
    rpc.save = lambda *args: calls.append(args[-1]) or (True, "")
    worker = _worker(StubSemantics, rpc, [], layout=layout)
    worker._batch_steps = 32
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.save(StepPlan(saves=[GroupSave(0, [11], [b"h1"])]))
    pressured.saves = [GroupSave(0, [12], [b"h2"])]
    worker.save(pressured)
    assert len(worker._held) == (0 if send else 2)
    assert worker.save_stats.pressure_rpcs == (1 if send else 0)
    assert worker.save_stats.kv_usage_unknown_steps == (1 if kv_usage is None else 0)
    assert worker.flush(5)
    assert [[ids for _, ids, _ in c] for c in calls] == [[[11], [12]]]
    worker.shutdown()


def test_merged_save_rpc_stays_under_size_budget(monkeypatch):
    """A merged save RPC must stay under the gRPC message limit: a step that
    would push the held batch past SAVE_RPC_MAX_BYTES sends the batch first.
    Nothing is lost, and the order of blocks is kept."""
    one_step = _wire_bytes([("layers.0.attn", [1], [b"h1"])])
    monkeypatch.setattr(Worker, "SAVE_RPC_MAX_BYTES", 2 * one_step)
    calls = []
    rpc = FakeRpc()
    rpc.save = lambda *args: calls.append(args[-1]) or (True, "")
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker._batch_steps = 100
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    for b in (1, 2, 3):
        worker.save(StepPlan(saves=[GroupSave(0, [b], [b"h%d" % b])]))
    assert worker.flush(5)
    assert [[ids for _, ids, _ in c] for c in calls] == [[[1], [2]], [[3]]]
    assert worker.save_stats.rpcs == 2
    worker.shutdown()


def test_save_batching_flush_and_fence():
    """A held job counts as submitted: a fence on its step and flush
    (preemption, shutdown) send held jobs at once instead of waiting for the
    batch bound."""
    calls = []
    rpc = FakeRpc()
    rpc.save = lambda *args: calls.append(args[-1]) or (True, "")
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker._batch_steps = 100  # never released by the bound in this test
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    worker.save(StepPlan(saves=[GroupSave(0, [11], [b"h1"])], step=0))
    worker.save(StepPlan(step=1))
    assert calls == [] and worker.saved_through() == -1
    worker.wait_saved(0)
    assert len(calls) == 1 and worker.saved_through() == 1
    worker.save(StepPlan(saves=[GroupSave(0, [12], [b"h2"])], step=2))
    worker.wait_saved(1)  # step 2's held job is no part of the fence
    assert len(calls) == 1
    assert worker.flush(5)
    assert len(calls) == 2 and worker.saved_through() == 2
    worker.shutdown()


@pytest.mark.parametrize(
    "fields,pin",
    [
        pytest.param({"needs_kv_cache_zeroing": False, "hash_block_size": 4}, False, id="plain"),
        pytest.param({"needs_kv_cache_zeroing": True, "hash_block_size": 4}, True, id="zeroing"),
        # Prefix hits inside a block: vLLM copies the shared tail (CoW).
        pytest.param({"needs_kv_cache_zeroing": False, "hash_block_size": 2}, True, id="cow"),
        pytest.param(
            {"needs_kv_cache_zeroing": False, "hash_block_size": None}, True, id="unknown"
        ),
    ],
)
def test_layouts_that_zero_or_copy_blocks(fields, pin):
    """S1: vLLM's V2 model runner zeroes and copies blocks before
    handle_preemptions, too late for a flush there: such layouts pin their
    saved blocks (connector ``_pin_saves``)."""
    cfg = _kv_cache_config(StubFullSpec())
    for name, value in fields.items():
        setattr(cfg, name, value)
    assert may_zero_or_copy_blocks(cfg) is pin


@pytest.mark.parametrize(
    "specs,preempted,flush",
    [
        pytest.param((StubFullSpec(),), (), False, id="full-steady"),
        # vLLM freed a preempted request's blocks; this step may reuse them.
        pytest.param((StubFullSpec(),), ("r",), True, id="full-preempted"),
        # Window blocks are freed as they slide out of the window mid-request.
        pytest.param((StubWindowSpec(), StubFullSpec()), (), True, id="window-always"),
    ],
)
def test_save_flush_before_block_reuse(specs, preempted, flush):
    sched, _ = _sched(StubSemantics, {}, layout=_layout(StubSemantics, *specs))
    sched.on_new_request(_req())
    sched.plan_saves([("r", 0, 13)], lambda rid: tuple([10, 11, 12, 13] for _ in specs))
    sched.take_plan()
    sched.plan_saves([], lambda rid: None, preempted)
    assert sched.take_plan().flush is flush
    # Saves a flush covered (a preempted request's, every one before a
    # window flush) need no fence at finish.
    assert sched.request_finished("r") is False
    assert sched._fenced == ({} if flush else {10: 0, 11: 0, 12: 0})


class FakePool:
    """vLLM BlockPool references by block id: pinned blocks are never reused."""

    def __init__(self):
        self.refs: dict[int, int] = {}

    def pin(self, block_ids):
        for b in block_ids:
            self.refs[b] = self.refs.get(b, 0) + 1
        return list(block_ids)

    def unpin(self, token):
        for b in token:
            self.refs[b] -= 1
            if not self.refs[b]:
                del self.refs[b]


def _pinned_sched(layout, pool=None):
    sched = Scheduler(
        layout, Policy(), FakeRpc(), "inst", clock=Clock(), semantics=StubSemantics, pin_saves=True
    )
    if pool is not None:
        sched.bind_pins(pool.pin, pool.unpin)
    return sched


def test_pinned_saves_hold_blocks_until_every_worker_saved():
    """Layouts where vLLM zeroes, COW-copies or frees state blocks before a
    pending save could run pin every saved block in vLLM's pool from the plan
    until the workers report that step saved: no flush, no fence, and a
    preempted request's blocks stay pinned (never reallocated)."""
    pool = FakePool()
    sched = _pinned_sched(_layout(StubSemantics, StubCheckpointSpec(), StubFullSpec()), pool)
    sched.on_new_request(_req())
    ids = lambda _: (CKPT_IDS, FULL_IDS)  # noqa: E731
    sched.plan_saves([("r", 0, 4)], ids, offers={"r": [(0, 31, 4)]})
    p0 = sched.take_plan()
    assert (p0.step, pool.refs) == (0, {10: 1, 31: 1})
    sched.plan_saves([("r", 4, 4)], ids, ("r",), offers={"r": [(0, 32, 8)]})
    p1 = sched.take_plan()
    assert (p1.step, p1.flush, pool.refs) == (1, False, {10: 1, 31: 1, 11: 1, 32: 1})
    assert sched.request_finished("r") is False and sched._fenced == {}
    sched.on_saved_through(0)
    assert pool.refs == {11: 1, 32: 1} and sched.has_pending_saves()
    sched.on_saved_through(5)
    assert pool.refs == {} and not sched.has_pending_saves()


@pytest.mark.parametrize("outcome", ["rejected", "raises"])
def test_pinned_saves_released_when_a_save_raises(outcome):
    """A raised save releases its pins: at once if the request was rejected,
    after the quiesce window otherwise (its copy may still run). Saving goes
    on either way."""
    import threading

    pool = FakePool()
    sched = _pinned_sched(_layout(StubSemantics, StubFullSpec()), pool)
    sched.on_new_request(_req())
    sched.plan_saves([("r", 0, 13)], lambda _: (FULL_IDS,))
    plan = sched.take_plan()
    assert pool.refs == {10: 1, 11: 1, 12: 1}
    rpc, calls = FakeRpc(), []
    save = _save_outcome(outcome)
    rpc.save = lambda *args: save(calls, args[-1])
    worker = _worker(StubSemantics, rpc, [], layout=_layout(StubSemantics, StubFullSpec()))
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    window = threading.Event()
    worker._quiesce_wait = lambda s: window.wait(5)
    worker.save(plan)
    if outcome == "raises":
        with worker._cv:
            assert worker._cv.wait_for(lambda: worker._quiescing, 5)
        sched.on_saved_through(worker.saved_through())
        assert pool.refs == {10: 1, 11: 1, 12: 1}  # held for the window
        window.set()
    with worker._cv:
        assert worker._cv.wait_for(lambda: worker._settled(1), 5)
    sched.on_saved_through(worker.saved_through())
    assert pool.refs == {} and not sched.has_pending_saves()
    window.set()
    worker.save(StepPlan(saves=plan.saves, step=1))
    assert worker.flush(5) and len(calls) == 2
    worker.shutdown()


def test_pinned_saves_fail_closed_without_a_pool():
    sched = _pinned_sched(_layout(StubSemantics, StubFullSpec()))
    sched.on_new_request(_req())
    sched.plan_saves([("r", 0, 13)], lambda _: (FULL_IDS,))
    assert sched.take_plan().saves == [] and not sched.has_pending_saves()


def test_worker_reports_saved_through():
    """saved_through (pins and fences) is the last plan step with every save
    before it done; a step without saves counts as done once the earlier
    ones are."""
    import threading

    gates = [threading.Event(), threading.Event()]
    marks = iter(gates)
    rpc = FakeRpc()
    layout = _layout(StubSemantics, StubFullSpec())
    worker = Worker(
        layout,
        rpc,
        "inst",
        Parallel(),
        0,
        lambda t: b"ipc",
        lambda: None,
        lambda: FakeMark(next(marks)),
    )
    worker.register({"layers.0.attn": "f"}, {"layers.0.attn": StubFullSpec()})
    assert worker.saved_through() == -1
    worker.save(StepPlan(saves=[GroupSave(0, [10], [b"h0"])], step=0))
    worker.save(StepPlan(step=1))
    worker.save(StepPlan(saves=[GroupSave(0, [11], [b"h1"])], step=2))
    assert worker.saved_through() == -1  # nothing copied yet: wait_for_save never blocked
    gates[0].set()
    with worker._cv:
        assert worker._cv.wait_for(lambda: worker._completed >= 1, 5)
    assert worker.saved_through() == 1
    gates[1].set()
    assert worker.flush(5)
    assert worker.saved_through() == 2
    worker.save(StepPlan(step=3))
    assert worker.saved_through() == 3
    worker.shutdown()


# --------------------------------------------------------------------- layout
@BOTH
@pytest.mark.parametrize(
    "specs,par,match",
    [
        pytest.param(
            (StubFullSpec(), MysterySpec()),
            Parallel(),
            r"no adapter for MysterySpec \(group 1\)",
            id="unknown-spec",
        ),
        pytest.param((StubWindowSpec(),), Parallel(), "no full-attention group", id="window-only"),
        # S2 / V0: TP2 ranks declaring different segment sizes fail seal_plan.
        pytest.param(
            (StubFullSpec(segments=(64, 64)),),
            Parallel(tp_size=2),
            None,
            id="tp2-ok",
        ),
        pytest.param(
            (StubFullSpec(), StubWindowSpec(backend="kernel")),
            Parallel(),
            r"disagree on \(transfer_backend, page_first\)",
            id="mixed-backend",
        ),
        pytest.param(
            (StubFullSpec(backend="dma"),), Parallel(), "unknown transfer_backend", id="bad-backend"
        ),
    ],
)
def test_layout_rejects(sem, specs, par, match):
    if match is None:
        _layout(sem, *specs, par=par)
        return
    with pytest.raises(Unsupported, match=match):
        _layout(sem, *specs, par=par)


class _RankSkewedSpec(StubFullSpec):
    pass


class _RankSkewedAdapter(_StubAdapter):
    def layer(self, spec, layer_name, par):
        segs = (64, 64) if par.tp_rank == 0 else (64, 32)
        return LayerDecl(segs, self.sem.Shard.heads(par.tp_rank, par.tp_size))


@BOTH
def test_layout_runs_seal_plan(sem, monkeypatch):
    module = types.ModuleType(f"{adapters.__name__}._test_skewed")
    module.ADAPTER = _RankSkewedAdapter(sem)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(adapters._REGISTRY, "_RankSkewedSpec", "_test_skewed")
    with pytest.raises(Unsupported, match="fail seal_plan"):
        _layout(sem, _RankSkewedSpec(), par=Parallel(tp_size=2))


def test_namespace_covers_transfer_mode():
    a = _layout(StubSemantics, StubFullSpec())
    b = _layout(StubSemantics, StubFullSpec(backend="kernel"))
    assert (a.transfer_backend, b.transfer_backend) == ("direct", "kernel")
    assert a.namespace != b.namespace
    assert a.namespace == _layout(StubSemantics, StubFullSpec()).namespace


# --------------------------------------------------------------------- policy
def test_policy_defaults_match_shipped_toml():
    assert load_policy(None) == Policy()


@pytest.mark.parametrize("profile", ["../x", "a/b", "x y"])
def test_policy_profile_name_is_checked(profile):
    with pytest.raises(ValueError, match="model_profile"):
        load_policy(profile)


# ----------------------------------------------------------- stub vs native
def test_stub_matches_native():
    native = _native()
    rng = random.Random(0)
    kinds = [("full", ()), ("window", (7,)), ("window", (12,)), ("checkpoint", ()), ("scratch", ())]
    for _ in range(300):
        picks = [rng.choice(kinds) + (rng.choice([2, 4, 8]),) for _ in range(rng.randint(1, 3))]
        stub_g = [getattr(_Group, k)(bt, *a) for k, a, bt in picks]
        nat_g = [getattr(native.Group, k)(bt, *a) for k, a, bt in picks]
        present = [{b for b in range(16) if rng.random() < 0.75} for _ in picks]
        prompt_len = rng.randrange(0, 64)
        assert _resolve(stub_g, present, prompt_len) == native.resolve(nat_g, present, prompt_len)
        for s, n in zip(stub_g, nat_g, strict=True):
            assert (s.kind, s.block_tokens, s.window_tokens) == (
                n.kind,
                n.block_tokens,
                n.window_tokens,
            )
            before = rng.randrange(0, 40)
            after = before + rng.randrange(0, 20)
            assert _needed(s, after) == tuple(native.needed(n, after))
            assert _saveable(s, before, after) == tuple(native.saveable(n, before, after))


# ------------------------------------------------- adapters/full_attention.py
@dataclass(frozen=True)
class FullAttentionSpec:  # duck-typed vLLM 0.32 spec; the registry keys on the class name
    """Defaults match Qwen3-4B's full-attention spec: bf16, block 16, 8 KV heads x 128."""

    block_size: int = 16
    num_kv_heads: int = 8
    head_size: int = 128
    page_size_bytes: int = 65536
    max_tp_shards: int | None = None
    dcp_sharded: bool = True
    has_layer_views: bool = True
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    page_size_padded: int | None = None
    num_head_slots: int | None = None
    state_content_bytes: int | None = None
    tokens_per_state: int = 1
    kv_quant_mode: int = 0
    non_causal: bool = False


@dataclass
class FakeKV:
    """The bits of a torch.Tensor the adapter reads; strides in elements."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    elem: int = 2

    def stride(self):
        return self.strides

    def element_size(self):
        return self.elem


QWEN3_BLOCKS = 4745
# vLLM 0.32 LBNHC layer view [B, H, N, C/2] in bf16 (probe spec dump, 4090 run).
QWEN3_KV = FakeKV((QWEN3_BLOCKS, 8, 16, 256), (32768, 256, 2048, 1))


@pytest.fixture
def full_attention(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import full_attention as mod

    adapter = mod.FullAttentionAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


@BOTH
@pytest.mark.parametrize(
    "kv,num_blocks",
    [
        pytest.param(QWEN3_KV, QWEN3_BLOCKS, id="qwen3-4b-LBNHC"),
        pytest.param(FakeKV((10, 8, 16, 256), (32768, 4096, 256, 1)), 10, id="LBHNC"),
        # Kernel blocks of 8 tokens split each 16-token manager block in two.
        pytest.param(FakeKV((20, 8, 8, 256), (16384, 256, 2048, 1)), 10, id="kernel-split"),
        # M1: a split [H, N, C] page is stored as 2 x [H, N/2, C], which the
        # namespace (kv_cache_layout) does not tell apart from the unsplit page.
        pytest.param(FakeKV((20, 8, 8, 256), (16384, 2048, 256, 1)), None, id="LBHNC-split"),
        # Block-outer layouts put the other layers' pages between our blocks.
        pytest.param(FakeKV((10, 8, 16, 256), (36 * 32768, 256, 2048, 1)), None, id="BLNHC"),
        pytest.param(FakeKV((10, 8, 16, 256), (32768, 256, 4096, 1)), None, id="holes"),
        pytest.param(FakeKV((10, 8, 16, 128), (16384, 128, 1024, 1)), None, id="wrong-page"),
    ],
)
def test_full_attention_view(sem, full_attention, kv, num_blocks):
    spec = FullAttentionSpec()
    if num_blocks is None:
        with pytest.raises(Unsupported):
            full_attention.view(spec, "l0", kv)
        return
    view = full_attention.view(spec, "l0", kv)
    # One segment = the whole page: LBNHC interleaves K and V per token/head.
    assert (view.num_blocks, view.bytes_per_block, view.kv_stride_bytes) == (
        num_blocks,
        65536,
        65536,
    )
    assert view.tensor is kv


def _attn_cfg(spec, n_layers, kv_cache_layout="LBNHC"):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(n_layers)]
    group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=names)
    return names, types.SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[group],
        hash_block_size=16,
        kv_cache_layout=kv_cache_layout,
    )


# Per-rank page for 4 of the 8 kv heads (TP2, or TP4 over 2 head shards).
HALF_SPEC = FullAttentionSpec(num_kv_heads=4, page_size_bytes=32768)
HALF_KV = FakeKV((QWEN3_BLOCKS, 4, 16, 256), (16384, 256, 1024, 1))


@BOTH
@pytest.mark.parametrize(
    "spec,par,kv,shards",
    [
        # Qwen3-4B as run by the gate: 36 layers, TP1, LBNHC.
        pytest.param(FullAttentionSpec(), Parallel(), QWEN3_KV, [(0, 1)], id="tp1"),
        pytest.param(HALF_SPEC, Parallel(tp_size=2), HALF_KV, [(0, 2), (1, 2)], id="tp2"),
        # 4 ranks over 2 head shards: consecutive ranks hold replicas.
        pytest.param(
            dataclasses.replace(HALF_SPEC, max_tp_shards=2),
            Parallel(tp_size=4, kv_tp_replicas=2),
            HALF_KV,
            [(0, 2), (0, 2), (1, 2), (1, 2)],
            id="tp4-replicas",
        ),
    ],
)
def test_full_attention_layout(sem, full_attention, spec, par, kv, shards):
    names, cfg = _attn_cfg(spec, 36)
    layout = build_layout(cfg, par, {"model": "Qwen3-4B"}, semantics=sem)  # runs seal_plan
    assert [(g.sem.kind, g.sem.block_tokens, g.hash_scale) for g in layout.groups] == [
        ("full", 16, 1)
    ]
    ranks = [dataclasses.replace(par, tp_rank=r) for r in range(par.tp_size)]
    decls = [full_attention.layer(spec, names[0], r) for r in ranks]
    assert [d.shard for d in decls] == [sem.Shard.heads(i, n) for i, n in shards]
    # One storage rank per head shard; its first holder writes it.
    first = {i: r for r, (i, _) in reversed(list(enumerate(shards)))}
    assert layout.storage == StorageRanks(
        shards[0][1],
        tuple(i for i, _ in shards),
        tuple(first[i] == r for r, (i, _) in enumerate(shards)),
    )
    # Rank 0 registration as the server sees it: one page-sized segment per layer.
    rpc = FakeRpc()
    _worker(sem, rpc, [], layout=layout).register(
        dict.fromkeys(names, kv), dict.fromkeys(names, spec)
    )
    reg = rpc.registered
    page = spec.page_size_bytes
    assert reg["names"] == names
    assert set(reg["segments"]) == {1}
    assert set(reg["bytes_per_block"]) == set(reg["kv_strides"]) == {page}
    assert set(reg["num_blocks"]) == {QWEN3_BLOCKS}
    assert (reg["backend"], reg["page_first"], set(reg["group_ids"])) == ("direct", False, {0})


@BOTH
@pytest.mark.parametrize(
    "field,value,par",
    [
        ("sliding_window", 4096, Parallel()),
        ("page_size_padded", 69632, Parallel()),
        ("kv_quant_mode", 1, Parallel()),
        ("num_head_slots", 4, Parallel()),
        ("non_causal", True, Parallel()),
        ("has_layer_views", False, Parallel()),
        # vLLM says ranks replicate (kv_tp_replicas) but the spec declares none.
        ("max_tp_shards", None, Parallel(tp_size=4, kv_tp_replicas=2)),
    ],
)
def test_full_attention_rejects_unvalidated_specs(sem, full_attention, field, value, par):
    spec = dataclasses.replace(FullAttentionSpec(), **{field: value})
    with pytest.raises(Unsupported):
        _layout_of(sem, spec, par)


def _layout_of(sem, spec, par=None, kv_cache_layout="LBNHC"):
    _, cfg = _attn_cfg(spec, 2, kv_cache_layout)
    return build_layout(cfg, par or Parallel(), {"model": "Qwen3-4B"}, semantics=sem)


def test_namespace_covers_page_byte_order(sem, full_attention):
    """M1: [N, H, C] and [H, N, C] pages hold different bytes for one declaration."""
    nhc = _layout_of(sem, FullAttentionSpec(), kv_cache_layout="LBNHC")
    hnc = _layout_of(sem, FullAttentionSpec(), kv_cache_layout="LBHNC")
    assert nhc.namespace != hnc.namespace
    assert nhc.namespace == _layout_of(sem, FullAttentionSpec()).namespace


# ---------------------------------------------------------------- adapters/mamba.py
@dataclass(frozen=True)
class MambaSpec:  # duck-typed vLLM 0.32 spec; the registry keys on the class name
    """Defaults match Qwen3.5-4B's GDN spec: align mode, page padded."""

    block_size: int = 528
    shapes: tuple[tuple[int, ...], ...] = ((3, 8192), (32, 128, 128))
    dtypes: tuple[str, ...] = ("bfloat16", "float32")
    page_size_bytes: int = 2162688
    mamba_cache_mode: str = "align"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None
    num_heads: int = 1
    tokens_per_state: int = -1
    tp_replicated: bool = False
    has_layer_views: bool = True


QWEN35_BLOCKS = 481
QWEN35_REAL = 2146304
# vLLM 0.32 raw Mamba layer view [B, 1, 1, real] int8, pages padded to 2162688.
QWEN35_KV = FakeKV((QWEN35_BLOCKS, 1, 1, QWEN35_REAL), (2162688, QWEN35_REAL, QWEN35_REAL, 1), 1)


@pytest.fixture
def mamba(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import mamba as mod

    adapter = mod.MambaAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


@BOTH
def test_mamba_declares_one_padded_page_checkpoint(sem, mamba):
    spec = MambaSpec()
    g = mamba.group(spec)
    assert (g.kind, g.block_tokens) == ("checkpoint", 528)
    decl = mamba.layer(spec, "layers.0.linear_attn", Parallel())
    # The copy unit is the block stride: the server addresses data_ptr + b * bytes_per_block.
    assert decl.segments == (2162688,)
    assert decl.layout == (((3, 8192), "bfloat16"), ((32, 128, 128), "float32"))
    view = mamba.view(spec, "layers.0.linear_attn", QWEN35_KV)
    assert (view.num_blocks, view.bytes_per_block, view.kv_stride_bytes) == (
        QWEN35_BLOCKS,
        2162688,
        2162688,
    )
    tp2 = [mamba.layer(spec, "l", Parallel(tp_rank=r, tp_size=2)).shard for r in (0, 1)]
    assert tp2 == [sem.Shard.heads(0, 2), sem.Shard.heads(1, 2)]
    replicated = dataclasses.replace(spec, tp_replicated=True)
    assert mamba.layer(replicated, "l", Parallel(tp_size=2)).shard == sem.Shard.replicated()


@BOTH
@pytest.mark.parametrize(
    "field,value",
    [
        ("mamba_cache_mode", "none"),
        ("num_speculative_blocks", 1),
        ("num_prefill_checkpoint_blocks", 2),
        ("tokens_per_state", 16),
        ("has_layer_views", False),
        ("page_size_bytes", 2146303),  # states do not fit the page
    ],
)
def test_mamba_rejects_unvalidated_specs(sem, mamba, field, value):
    spec = dataclasses.replace(MambaSpec(), **{field: value})
    with pytest.raises(Unsupported):
        mamba.group(spec)


@BOTH
@pytest.mark.parametrize(
    "kv",
    [
        # Block-outer layouts put other layers' pages between our blocks.
        pytest.param(
            FakeKV((10, 1, 1, QWEN35_REAL), (8 * 2162688, QWEN35_REAL, QWEN35_REAL, 1), 1),
            id="block-outer",
        ),
        pytest.param(FakeKV((10, 1, 1, 2097152), (2162688, 2097152, 2097152, 1), 1), id="ssm-only"),
        pytest.param(FakeKV((10, 3, 8192), (2162688, 8192, 1), 1), id="not-4d"),
    ],
)
def test_mamba_view_rejects_foreign_geometry(sem, mamba, kv):
    with pytest.raises(Unsupported):
        mamba.view(MambaSpec(), "l0", kv)


def _qwen35_cfg():
    """Qwen3.5-4B layout: 3 GDN groups + 1 attention group, block 528."""
    attn = FullAttentionSpec(block_size=528, num_kv_heads=4, head_size=256, page_size_bytes=2162688)
    groups = [
        types.SimpleNamespace(
            kv_cache_spec=MambaSpec(),
            layer_names=[f"layers.{i}.linear_attn" for i in range(g, 32, 4)],
        )
        for g in range(3)
    ]
    groups.append(
        types.SimpleNamespace(
            kv_cache_spec=attn, layer_names=[f"layers.{i}.self_attn.attn" for i in range(3, 32, 4)]
        )
    )
    return types.SimpleNamespace(
        transfer_group_ids=(0, 1, 2, 3),
        kv_cache_groups=groups,
        hash_block_size=528,
        kv_cache_layout="LBNHC",
    )


@BOTH
def test_qwen35_layout(sem, mamba, full_attention):
    """Full group first (prefix query), then the 3 GDN checkpoint groups; step 528.

    The Mamba state layout enters the namespace: a different state dtype, or
    vLLM's ``VLLM_SSM_CONV_STATE_LAYOUT=DS`` conv orientation (same page size,
    other byte order), is a different namespace.
    """
    layout = build_layout(_qwen35_cfg(), Parallel(), {"model": "Qwen3.5-4B"}, semantics=sem)
    assert [(g.gid, g.sem.kind) for g in layout.groups] == [
        (3, "full"),
        (0, "checkpoint"),
        (1, "checkpoint"),
        (2, "checkpoint"),
    ]
    assert layout.step == 528
    for change in (
        {"dtypes": ("bfloat16", "bfloat16")},
        {"shapes": ((8192, 3), (32, 128, 128))},  # conv layout DS
    ):
        cfg = _qwen35_cfg()
        for g in cfg.kv_cache_groups[:3]:
            g.kv_cache_spec = dataclasses.replace(g.kv_cache_spec, **change)
        other = build_layout(cfg, Parallel(), {"model": "Qwen3.5-4B"}, semantics=sem)
        assert other.namespace != layout.namespace, change


# Qwen3-4B test layout namespaces computed by the P4 core (563a693, with
# storage_tp, before LayerDecl.layout); stub and native differ only in
# repr(Shard). Was cf6b5725 / cd8acc0c before storage_tp entered the namespace.
QWEN3_NAMESPACE_V1 = {"stub": "a25ea150", "native": "bdda8997"}


@BOTH
def test_namespace_vector_unchanged_for_full_attention(sem, full_attention):
    """V1: LayerDecl.layout is hashed only when declared, so full-attention
    layouts keep the namespace (and the stored blocks) they had before it."""
    _, cfg = _attn_cfg(FullAttentionSpec(), 36)
    layout = build_layout(cfg, Parallel(), {"model": "Qwen3-4B"}, semantics=sem)
    which = "stub" if sem is StubSemantics else "native"
    assert layout.namespace == QWEN3_NAMESPACE_V1[which]


@BOTH
@pytest.mark.parametrize("tp", [1, 2])
def test_qwen35_declarations_seal(sem, mamba, full_attention, tp):
    """V0 for the hybrid layout: every rank's GDN + attention regions seal."""
    cfg = _qwen35_cfg()
    ranks = []
    for r in range(tp):
        par = Parallel(tp_rank=r, tp_size=tp)
        regions = []
        for sid, g in enumerate(cfg.kv_cache_groups):
            spec = g.kv_cache_spec
            if tp > 1 and isinstance(spec, FullAttentionSpec):
                spec = dataclasses.replace(spec, num_kv_heads=2, page_size_bytes=1081344)
            adapter = mamba if isinstance(spec, MambaSpec) else full_attention
            for name in g.layer_names:
                d = adapter.layer(spec, name, par)
                regions.append(sem.Region(name, sid, [int(s) for s in d.segments], d.shard))
        ranks.append((r, regions))
    sem.seal_plan(ranks)


# -------------------------------------------------- adapters/mla_attention.py
@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):  # registry keys on the class name
    """Defaults match DeepSeek-V2-Lite-Chat's MLA spec (bf16 latent
    kv_lora_rank 512 + rope 64 = 576, one head, replicated over TP)."""

    num_kv_heads: int = 1
    head_size: int = 576
    head_size_v: int = 0
    page_size_bytes: int = 18432
    max_tp_shards: int | None = 1
    cache_dtype_str: str | None = "auto"
    alignment: int | None = None
    model_version: str | None = None
    cache_role: str = "sparse"
    is_index_group_leader: bool = False
    non_causal_multi_token_decode: bool = False


DSV2_BLOCKS = 6222
# vLLM 0.32 LBNHC layer view [B, H=1, N, C=576] in bf16 (probe dump, both TP2 ranks).
DSV2_KV = FakeKV((DSV2_BLOCKS, 1, 16, 576), (9216, 576, 576, 1))


@pytest.fixture
def mla_attention(monkeypatch, sem):
    from pegaflow.connector_v2.adapters import mla_attention as mod

    adapter = mod.MLAAttentionAdapter(sem)
    monkeypatch.setattr(mod, "ADAPTER", adapter)
    return adapter


@BOTH
@pytest.mark.parametrize(
    "par",
    [
        pytest.param(Parallel(), id="tp1"),
        # DeepSeek-V2-Lite as run by the gate: TP2, vLLM kv_tp_replicas=2.
        pytest.param(Parallel(tp_size=2, kv_tp_replicas=2), id="tp2"),
        pytest.param(Parallel(tp_size=4, kv_tp_replicas=4), id="tp4"),
    ],
)
def test_mla_attention_layout(sem, mla_attention, par):
    spec = MLAAttentionSpec()
    names, cfg = _attn_cfg(spec, 27)
    layout = build_layout(cfg, par, {"model": "DeepSeek-V2-Lite-Chat"}, semantics=sem)
    assert [(g.sem.kind, g.sem.block_tokens) for g in layout.groups] == [("full", 16)]
    ranks = [dataclasses.replace(par, tp_rank=r) for r in range(par.tp_size)]
    # Every rank holds the same latent page: one fragment per layer after seal_plan.
    assert {mla_attention.layer(spec, names[0], r).shard for r in ranks} == {sem.Shard.replicated()}
    # Replicas share one storage rank: one slot per layer, only rank 0 saves.
    n = par.tp_size
    assert layout.storage == StorageRanks(1, (0,) * n, (True,) + (False,) * (n - 1))
    for r in ranks:
        rpc = FakeRpc()
        saved = []
        rpc.save = lambda *args, saved=saved: saved.append(args) or (True, "")
        loaded = []
        rpc.load = lambda *args, loaded=loaded: loaded.append(args[1]) or (True, "")
        worker = _worker(sem, rpc, [FakeLoadState()], layout=layout, par=r)
        worker.register(dict.fromkeys(names, DSV2_KV), dict.fromkeys(names, spec))
        reg = rpc.registered
        assert (reg["tp_rank"], reg["tp_size"]) == (0, 1)
        assert set(reg["segments"]) == {1}
        assert set(reg["bytes_per_block"]) == set(reg["kv_strides"]) == {18432}
        assert set(reg["num_blocks"]) == {DSV2_BLOCKS}
        assert (reg["backend"], reg["page_first"]) == ("direct", False)
        worker.save(StepPlan(saves=[GroupSave(0, [11], [b"h1"])]))
        assert worker.flush(5)
        assert [(a[1], len(a[-1])) for a in saved] == ([(0, 27)] if r.tp_rank == 0 else [])
        worker.start(StepPlan(loads=[LoadPlan("q", [(b"L0", [[3]])], sync=False)]))
        assert loaded == [0]  # every holder loads its own GPU from storage rank 0
        worker.shutdown()


def test_mla_seal_plan_one_writer():
    """Replicated MLA pages seal as one fragment per layer, written by rank 0."""
    sem = _native()
    spec = MLAAttentionSpec()
    from pegaflow.connector_v2.adapters import mla_attention as mod

    adapter = mod.MLAAttentionAdapter(sem)
    ranks = []
    for r in range(2):
        d = adapter.layer(spec, "l0", Parallel(tp_rank=r, tp_size=2, kv_tp_replicas=2))
        ranks.append((r, [sem.Region("l0", 0, list(d.segments), d.shard)]))
    plan = sem.seal_plan(ranks)
    (frag,) = plan.fragments(0)
    assert (frag.holders, frag.writer, frag.bytes) == ([0, 1], 0, 18432)
    assert plan.is_writer(0, 0, 0) and not plan.is_writer(1, 0, 0)


@BOTH
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tp_shards", None),
        ("max_tp_shards", 2),
        ("alignment", 576),
        ("model_version", "deepseek_v4"),
        ("cache_role", "indexer"),
        ("is_index_group_leader", True),
        ("cache_dtype_str", "fp8_ds_mla"),
        ("state_content_bytes", 656),
        ("kv_quant_mode", 1),
        ("page_size_padded", 18944),
        ("non_causal_multi_token_decode", True),
        ("sliding_window", 4096),
        ("head_size_v", 128),
    ],
)
def test_mla_attention_rejects_unvalidated_specs(sem, mla_attention, field, value):
    spec = dataclasses.replace(MLAAttentionSpec(), **{field: value})
    _, cfg = _attn_cfg(spec, 2)
    with pytest.raises(Unsupported):
        build_layout(cfg, Parallel(tp_size=2, kv_tp_replicas=2), {"model": "m"}, semantics=sem)


def test_mla_and_full_attention_never_share_a_namespace(sem, full_attention, mla_attention):
    """Equal page sizes, different bytes: the spec class is part of the namespace."""
    mla = MLAAttentionSpec()
    full = FullAttentionSpec(num_kv_heads=1, head_size=288, page_size_bytes=18432)
    assert _layout_of(sem, mla).namespace != _layout_of(sem, full).namespace
