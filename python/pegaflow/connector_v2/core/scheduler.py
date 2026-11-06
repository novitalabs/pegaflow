"""Scheduler-side state machine: query -> resolve -> policy -> lease -> load plan; saves.

vLLM-free: the connector passes vLLM objects in, the server client is duck-typed
(``query_prefetch`` / ``release``) and ``semantics`` is injectable, so unit
tests drive it with fakes and a stub semantics module.

Policy safety: the policy only ever sees immutable copies, and every answer it
gives is validated against what the core offered (``accept`` can only lower
``p``; ``choose_saves`` can only drop items). A misbehaving policy costs hit
rate or fails loudly; it cannot widen a hit or forge a save.
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..policy import Policy, SaveItem
from .layout import GroupInfo, Layout, native_semantics

logger = logging.getLogger(__name__)


@dataclass
class LoadPlan:
    """One request's load, consumed by every worker.

    ``loads[i]`` is ``(lease, targets_by_group)``; ``targets_by_group[g][j]`` is the
    GPU block id for the lease's j-th block in storage group g, or None to skip.
    ``fail`` asks workers to report the request as a failed receive without
    loading (the scheduler found the plan unusable after allocation).

    ``sync`` loads finish before this step's forward. vLLM already counts
    their tokens as computed, so a failure is reported as invalid blocks
    (``covered``: the GPU blocks holding the hit's tokens) instead of a
    failed receive, and a success is not reported at all.
    """

    req_id: str
    loads: list[tuple[bytes, list[list[int | None]]]] = field(default_factory=list)
    fail: bool = False
    sync: bool = False
    covered: list[int] = field(default_factory=list)


@dataclass
class GroupSave:
    sid: int  # storage group (index into Layout.groups)
    block_ids: list[int]
    block_hashes: list[bytes]


@dataclass
class StepPlan:
    """One step's work for every worker.

    Saves are asynchronous on the worker, so the plan also carries the block
    lifecycle the worker must respect, before this step writes any KV:
    ``fence_step`` (>= 0) asks it to wait until the saves of plans up to that
    step are done (this step writes a block vLLM freed while one of them may
    still read it: ``Scheduler.fence_writes``), and ``flush`` to finish every
    pending save (vLLM freed blocks without telling which, e.g. on
    preemption). With pinned saves (``Scheduler(pin_saves=True)``) neither is
    used: the scheduler keeps every block it plans to save allocated in vLLM's
    block pool until all workers report the step's saves done (``step``,
    ``Scheduler.on_saved_through``). ``send_saves`` asks the worker to send
    its held (batched) saves with this step's, without waiting for them:
    vLLM's GPU KV cache is under pressure, so freed blocks are reused soon and
    a fence on a held save would wait for the whole batch.
    """

    loads: list[LoadPlan] = field(default_factory=list)
    saves: list[GroupSave] = field(default_factory=list)
    fence_step: int = -1  # wait for the saves of plans <= this before any KV write
    flush: bool = False
    send_saves: bool = False
    kv_usage_unknown: bool = False  # vLLM's KV usage was unavailable (send_saves set)
    step: int = -1  # plan_saves call that built this plan (pinned saves report it back)


@dataclass(frozen=True)
class _Lease:
    lease: bytes
    sid: int
    positions: tuple[int, ...]  # block index of each lease block, in lease order
    wanted: frozenset[int] = frozenset()  # positions to load


@dataclass
class _Hit:
    p: int
    num_computed: int
    present: tuple[frozenset[int], ...]  # server answers per sid (for re-planning)
    leases: list[_Lease]  # held leases, each with a non-empty ``wanted``
    sync: bool = False  # loaded before the forward of the step that schedules it
    at: float = 0.0  # scheduler clock when the server answered


@dataclass
class _Lookup:
    leases: list[_Lease]
    present: tuple[frozenset[int], ...]  # rebuilt from the server answers only
    loading: bool


def _offered_ids(g: GroupInfo, blocks: range, offers: Sequence[tuple[int, int, int]]) -> list[int]:
    """Positional GPU ids for a checkpoint group's ``blocks``, from vLLM's
    offers (0 where vLLM offered no state for that boundary)."""
    bt = g.sem.block_tokens
    ids = [0] * blocks.stop
    for gid, block_id, boundary in offers:
        b = boundary // bt - 1
        if gid == g.gid and boundary % bt == 0 and b in blocks:
            ids[b] = block_id
    return ids


class Scheduler:
    HIT_LOG_S = 30.0  # min seconds between "hit stats" lines while lookups run
    # Age cap for reusing an unallocated hit; a bound, not a tuned policy.
    # vLLM re-asks for a head it did not allocate every scheduling pass. The
    # cap bounds how long a newer save of the request's blocks goes unseen,
    # and stays far below the server's lease TTL (600 s). In a TP1 replay
    # gate diag (2026-10-26, ebe69db) the longest wait was 1.5 s, so
    # 2 s answers each wait with one query; 0.5 s would add about 11% more
    # queries, still about 1 ms each. A reused hit skips `Policy.accept`:
    # its inputs (`p`, the server answers) are unchanged.
    HIT_REUSE_MAX_S = 2.0

    def __init__(
        self,
        layout: Layout,
        policy: Policy,
        rpc: Any,
        instance_id: str,
        clock: Callable[[], float] = time.monotonic,
        semantics: Any | None = None,
        sync_loads: bool = False,
        confirm_steps: int = 2,
        pin_saves: bool = False,
        prefix_caching: bool = True,
    ):
        """``sync_loads``: vLLM can recover from a failed synchronous load of this
        layout (it maps ``invalid_block_ids`` back to requests only with one
        KV cache group, and the worker's block zeroing is not ordered against
        the server's copy).

        ``confirm_steps``: steps vLLM schedules before it processes a step's
        output (vLLM's ``max_concurrent_batches``). A sync load's outcome, and
        the rewind of a failed one, is visible only that many steps later.

        ``pin_saves``: blocks to save are pinned in vLLM's block pool
        (``bind_pins``) from the step that plans them until every worker
        reports that step's saves done (``on_saved_through``). A pinned block
        is never handed out again, so vLLM cannot zero, copy into (COW) or
        reuse it while the worker copies it: saves stay asynchronous without
        preemption flushes or fences. For layouts where vLLM writes
        reallocated blocks before ``handle_preemptions`` (block zeroing, COW)
        or frees checkpoint state blocks while they are still being read.
        Saves are disabled (fail closed) if no pool was bound.

        ``prefix_caching``: vLLM prefix-caches computed and loaded blocks, so
        a request may wait for another request's load of its blocks
        (``_defer_for_inflight``)."""
        self._layout = layout
        self._policy = policy
        self._rpc = rpc
        self._instance_id = instance_id
        self._clock = clock
        self._sem = semantics if semantics is not None else native_semantics()
        self._sync_loads = sync_loads
        self._requests: dict[str, Any] = {}  # live vLLM Request objects
        self._hits: dict[str, _Hit] = {}  # leased, waiting for allocation
        self._loading_since: dict[str, float] = {}
        # Async loads in flight: req -> the (sid, block hash) keys it writes,
        # and how many in-flight loads write each key.
        self._inflight: dict[str, list[tuple[int, bytes]]] = {}
        self._inflight_refs: dict[tuple[int, bytes], int] = {}
        # req -> when it was first deferred for another request's load.
        self._deferred_since: dict[str, float] = {}
        self._defer_inflight = prefix_caching
        self._saved_upto: dict[str, int] = {}  # token count already offered for saving
        # Fences (unpinned saves). req -> {GPU block: plan step of its last
        # chosen save}. A finished request's entries move to ``_fenced``:
        # vLLM gets the blocks back at once, and a step that writes a fenced
        # block first waits for that step's saves (``fence_writes``).
        self._saved_blocks: dict[str, dict[int, int]] = {}
        self._fenced: dict[int, int] = {}  # GPU block -> plan step of its save
        self._fence_steps: dict[int, list[int]] = {}  # plan step -> blocks fenced on it
        self._last_save_step = -1  # last plan step with saves
        self._saved_through = -1  # last plan step every worker reported saved
        # Saves of plans up to this step are done before any later plan
        # writes KV (reported saved, or a planned flush or fence waits for them).
        self._fence_floor = -1
        # Block size of the groups whose table is positional (a block below
        # the computed tokens is a prefix hit, read only); any other group's
        # new blocks all count as written.
        self._positional_bt = {
            g.gid: g.sem.block_tokens for g in layout.groups if g.sem.kind in ("full", "window")
        }
        # vLLM frees window blocks that slide out while the request runs, which a
        # pending save may still read: such layouts flush before every step.
        # Pinned saves cover that too: a pinned block is never reallocated.
        self._flush_always = not pin_saves and any(g.sem.kind == "window" for g in layout.groups)
        # GPU block -> first step that may save on top of it. A sync load lands
        # before its step's forward; if it failed, that step and the ones vLLM
        # scheduled before seeing the failure computed on its bytes. The server
        # keeps the first copy of a key, so such saves would persist wrong hits.
        self._confirm_steps = max(1, confirm_steps)
        self._steps = 0  # plan_saves calls so far
        # (gid, block index, GPU block id) -> step from which saves may resume.
        # Prefix-cache sharing keeps a block at the same index in every table.
        self._unconfirmed: dict[tuple[int, int, int], int] = {}
        # Fail closed after a failed sync load: vLLM (recompute policy) keeps
        # the invalid blocks and every block computed on them in its prefix
        # cache under their correct hashes, so any later request may extend
        # bad KV. No save is planned once vLLM reported one (``on_load_errors``),
        # and later loads are async so no further bad block enters that cache.
        self._saves_disabled = False
        self._kv_usage_warned = False
        self._pin_saves = pin_saves
        self._pin: Callable[[list[int]], Any] | None = None
        self._unpin: Callable[[Any], None] | None = None
        self._pins: dict[int, Any] = {}  # plan step -> pin token, in step order
        self._plan = StepPlan()
        # Lookups, and per missing-group set how often (and how many tokens)
        # the full groups backed a longer hit than the one taken. Counted once
        # per (request, resume point): vLLM re-queries a waiting request every
        # scheduling pass until it can be allocated.
        self.hit_stats: dict[str, int] = {}
        # Wall time of every _lookup call (vLLM's scheduler thread waits for
        # it, re-queries of a waiting request included) and its query RPCs.
        self.lookup_cost = {"calls": 0, "rpcs": 0, "us": 0, "max_us": 0, "reused": 0}
        self._counted: dict[str, tuple[int, int]] = {}
        self._hit_dirty = False  # counters changed since the last line
        self._hit_logged_at = clock()

    # ------------------------------------------------------------------ hits
    def on_new_request(self, request: Any) -> None:
        self._requests[request.request_id] = request

    def get_num_new_matched_tokens(
        self, request: Any, num_computed: int
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        self._requests.setdefault(req_id, request)
        if self._defer_for_inflight(request, num_computed):
            self._drop_hit(req_id)
            # Its own clock bounds this wait: a prefetch wait before it must not
            # expire the moment the request is looked up again.
            self._loading_since.pop(req_id, None)
            return None, False
        # A previous hit vLLM did not allocate: reused if nothing it rests on
        # changed, else stale.
        reuse = self._reusable_hit(req_id, num_computed)
        if reuse is not None:
            self.lookup_cost["reused"] += 1
            self._hit_dirty = True
            return self._offer(req_id, reuse, num_computed)

        t0 = self._clock()
        try:
            lookup = self._lookup(request, num_computed)
        except Exception:
            logger.warning("[PegaKVConnectorV2] query failed req=%s", req_id, exc_info=True)
            self._loading_since.pop(req_id, None)
            return 0, False
        finally:
            self._time_lookup(self._clock() - t0)

        if lookup.loading:
            self._release(lookup.leases)
            start = self._loading_since.setdefault(req_id, self._clock())
            if (self._clock() - start) * 1000 < self._policy.loading_wait_ms:
                return None, False
            logger.info("[PegaKVConnectorV2] prefetch wait expired req=%s", req_id)
            self._loading_since.pop(req_id, None)
            return 0, False
        self._loading_since.pop(req_id, None)

        try:
            hit = self._plan_hit(request, num_computed, lookup)
        except Exception:
            # semantics or policy raised: never leak the leases or crash vLLM.
            logger.warning("[PegaKVConnectorV2] hit planning failed req=%s", req_id, exc_info=True)
            hit = None
        if hit is None:
            self._release(lookup.leases)
            return 0, False
        self._release_unkept(lookup.leases, hit.leases)
        hit.at = t0
        return self._offer(req_id, hit, num_computed)

    def _offer(self, req_id: str, hit: _Hit, num_computed: int) -> tuple[int, bool]:
        n = hit.p - num_computed
        # A short load costs about a millisecond; waiting for it before the
        # forward beats the two scheduler round trips of an async load. After
        # a failed sync load every load is async: vLLM keeps a failed sync
        # load's blocks prefix-cached, a failed async load's never.
        hit.sync = (
            self._sync_loads and not self._saves_disabled and n <= self._policy.sync_load_tokens
        )
        self._hits[req_id] = hit
        return n, not hit.sync

    def _reusable_hit(self, req_id: str, num_computed: int) -> _Hit | None:
        """Pop the request's unallocated hit; return it if it can be offered
        again without a server query, else release it.

        Its leases keep the blocks it loads on the server, so the answer still
        holds; it is reused only for the same local hit and within
        ``HIT_REUSE_MAX_S``. A longer hit (another request saved more of the
        prefix meanwhile) is picked up at the next fresh query.
        """
        hit = self._hits.pop(req_id, None)
        if hit is None:
            return None
        if hit.num_computed == num_computed and self._clock() - hit.at < self.HIT_REUSE_MAX_S:
            return hit
        self._release(hit.leases)
        return None

    def _defer_for_inflight(self, request: Any, num_computed: int) -> bool:
        """Wait for another request's async load of this request's blocks.

        vLLM prefix-caches an async load's blocks only once the load lands,
        so a request looked up meanwhile misses them locally and would load
        its own copy into its own blocks. Waiting turns them into vLLM's
        local hit; only the tail past them is loaded. Deferred only if the
        in-flight blocks, with the local hit, give a resume point past it
        (``semantics.resolve``: e.g. no checkpoint state at a shared
        boundary, no gain). No server query, so no lease is held while
        deferred. A failed or aborted load ends the wait (``on_loads_done``,
        ``request_finished``); ``loading_wait_ms`` bounds it per request,
        separately from the prefetch wait. Not for a request that skips
        vLLM's prefix cache: vLLM ignores its hit (and a None) anyway.
        """
        req_id = request.request_id
        if not self._defer_inflight or self._policy.loading_wait_ms <= 0:
            return False
        if getattr(request, "skip_reading_prefix_cache", False):
            return False
        if not self._inflight_refs or req_id in self._inflight:
            self._deferred_since.pop(req_id, None)
            return False
        now = self._clock()
        since = self._deferred_since.get(req_id)
        if since is not None and (now - since) * 1000 >= self._policy.loading_wait_ms:
            return False
        layout = self._layout
        limit_tokens = max(request.num_tokens - 1, 0)
        known: list[frozenset[int]] = []
        for sid, g in enumerate(layout.groups):
            bt = g.sem.block_tokens
            hashes = layout.group_hashes(g, request.block_hashes)[: limit_tokens // bt]
            local = num_computed // bt
            known.append(
                frozenset(
                    b for b, h in enumerate(hashes) if b < local or (sid, h) in self._inflight_refs
                )
            )
        try:
            bound = self._sem.resolve([g.sem for g in layout.groups], known, request.num_tokens)
        except Exception:
            return False  # only a shortcut: look up as usual
        if bound <= num_computed:
            self._deferred_since.pop(req_id, None)
            return False
        if since is None:
            self._deferred_since[req_id] = now
            self._count("deferred")
        return True

    def _count(self, key: str) -> None:
        self.hit_stats[key] = self.hit_stats.get(key, 0) + 1
        self._hit_dirty = True

    def _time_lookup(self, seconds: float) -> None:
        """Every ``_lookup`` call (vLLM's scheduler thread waits for it),
        re-queries of a waiting request included."""
        us = int(seconds * 1e6)
        cost = self.lookup_cost
        cost["calls"] += 1
        cost["us"] += us
        cost["max_us"] = max(cost["max_us"], us)
        self._hit_dirty = True

    def on_loads_done(self, req_ids: Iterable[str]) -> None:
        """Async loads vLLM saw finish or fail (``finished_recving`` /
        ``failed_recving``), before it caches their blocks this pass."""
        for req_id in req_ids:
            self._untrack_load(req_id)

    def _track_load(self, request: Any, leases: list[_Lease]) -> None:
        keys: list[tuple[int, bytes]] = []
        for lease in leases:
            hashes = self._layout.group_hashes(self._layout.groups[lease.sid], request.block_hashes)
            keys.extend((lease.sid, hashes[b]) for b in sorted(lease.wanted) if b < len(hashes))
        self._untrack_load(request.request_id)
        if not keys:
            return
        self._inflight[request.request_id] = keys
        for key in keys:
            self._inflight_refs[key] = self._inflight_refs.get(key, 0) + 1

    def _untrack_load(self, req_id: str) -> None:
        for key in self._inflight.pop(req_id, ()):
            n = self._inflight_refs[key] - 1
            if n:
                self._inflight_refs[key] = n
            else:
                del self._inflight_refs[key]

    def _lookup(self, request: Any, num_computed: int) -> _Lookup:
        """Query the server, storage group 0 first.

        vLLM re-queries every waiting request on every scheduling pass, so
        queries that cannot beat vLLM's local hit are skipped. The bound is
        ``semantics.resolve`` over optimistic answers (every block of a group
        not yet queried present): before any query, then after group 0. If
        it is not past ``num_computed``, no answer from the remaining groups
        can make a hit. For a full-only layout that is "no full block past
        the local hit"; with a checkpoint group it is the next resume step.
        A skipped group creates no lease; group 0's lease is released.
        """
        layout = self._layout
        groups = [g.sem for g in layout.groups]
        # resolve keeps the last token for vLLM, so blocks past it are never needed.
        limit_tokens = max(request.num_tokens - 1, 0)
        hashes = [
            layout.group_hashes(g, request.block_hashes)[: limit_tokens // g.sem.block_tokens]
            for g in layout.groups
        ]
        everything = [frozenset(range(len(h))) for h in hashes]

        def cannot_beat_local(known: list[frozenset[int]]) -> bool:
            try:
                bound = self._sem.resolve(
                    groups, known + everything[len(known) :], request.num_tokens
                )
            except Exception:
                return False  # only a shortcut: query as usual, _plan_hit reports it
            return bound <= num_computed

        none = tuple(frozenset() for _ in layout.groups)
        if cannot_beat_local([]):
            return _Lookup([], none, False)
        leases: list[_Lease] = []
        present: list[frozenset[int]] = []
        loading = False
        try:
            for sid, g in enumerate(layout.groups):
                if not hashes[sid] or g.sem.kind == "scratch":
                    present.append(frozenset())
                    continue
                self.lookup_cost["rpcs"] += 1
                ans = self._rpc.query_prefetch(
                    self._instance_id, hashes[sid], request.request_id, False, sid
                )
                if not hasattr(ans, "num_hit_blocks"):  # QueryLoading
                    loading = True
                    present.append(frozenset())
                    continue
                positions = tuple(ans.hit_positions or range(ans.num_hit_blocks))
                if ans.lease:
                    leases.append(_Lease(ans.lease, sid, positions))
                elif positions:
                    raise RuntimeError(f"group {sid}: {len(positions)} hits without a lease")
                present.append(frozenset(positions))
                if sid == 0 and len(layout.groups) > 1 and cannot_beat_local(present):
                    self._release(leases)
                    return _Lookup([], (present[0], *none[1:]), False)
        except BaseException:
            self._release(leases)
            raise
        return _Lookup(leases, tuple(present), loading)

    def _plan_hit(self, request: Any, num_computed: int, lookup: _Lookup) -> _Hit | None:
        """Pick ``p`` and the blocks to load; releases nothing (the caller does)."""
        groups = [g.sem for g in self._layout.groups]
        p = self._sem.resolve(groups, lookup.present, request.num_tokens)
        # The policy gets frozensets in a tuple: it cannot add blocks the server
        # never reported, and its answer can only lower p.
        accepted = operator.index(self._policy.accept(p, lookup.present))
        p = max(0, min(accepted, p))
        self._count_miss(request, num_computed, lookup.present, p)
        if p <= num_computed or p % self._layout.step:
            return None
        kept = self._assign(p, num_computed, lookup.present, lookup.leases)
        if kept is None:
            return None
        return _Hit(p, num_computed, lookup.present, kept)

    def _assign(
        self,
        p: int,
        num_computed: int,
        present: tuple[frozenset[int], ...],
        leases: list[_Lease],
    ) -> list[_Lease] | None:
        """Leases to keep for a hit at ``p``, each with its blocks to load.

        None if ``p`` is not backed: a needed block the server did not report,
        a needed block straddling the local hit, or a wanted block no lease
        covers. Never returns an empty list (a hit must load something).
        """
        by_sid = {lease.sid: lease for lease in leases}
        kept: list[_Lease] = []
        for sid, g in enumerate(self._layout.groups):
            lo, hi = self._sem.needed(g.sem, p)
            bt = g.sem.block_tokens
            need = range(lo, hi)
            if not present[sid].issuperset(need):
                return None
            # Blocks fully inside the local hit are already on the GPU. A block
            # straddling it cannot be loaded without overwriting shared blocks;
            # unreachable in vLLM 0.32 (the local hit is aligned to the lcm of
            # the group block sizes), so it is only a fail-safe.
            if any(b * bt < num_computed < (b + 1) * bt for b in need):
                return None
            # A checkpoint's needed block [p - bt, p) always lies past the local
            # hit (p - num_computed >= step >= bt), so the state at p is always
            # loaded: vLLM drops its own local state for an external hit.
            wanted = frozenset(b for b in need if b * bt >= num_computed)
            if not wanted:
                continue
            lease = by_sid.get(sid)
            if lease is None or not wanted.issubset(lease.positions):
                return None
            kept.append(_Lease(lease.lease, sid, lease.positions, wanted))
        return kept or None

    def loaded_group_ids(self, req_id: str) -> tuple[int, ...]:
        """vLLM groups the pending hit writes into (vLLM skips zeroing them)."""
        hit = self._hits.get(req_id)
        if hit is None:
            return ()
        return tuple(sorted({self._layout.groups[lease.sid].gid for lease in hit.leases}))

    def update_state_after_alloc(
        self, request: Any, block_ids: tuple[list[int], ...], num_external_tokens: int
    ) -> None:
        req_id = request.request_id
        self._deferred_since.pop(req_id, None)  # scheduled: a later wait starts afresh
        hit = self._hits.pop(req_id, None)
        if hit is None:
            return
        if num_external_tokens == 0:
            self._release(hit.leases)
            return
        plan = LoadPlan(req_id, sync=hit.sync)
        if plan.sync:
            covered = self._covered_blocks(hit.num_computed, num_external_tokens, block_ids)
            plan.covered = [block_id for _, _, block_id in covered]
            # This step's plan_saves is the next call: hold saves for it and
            # the steps vLLM schedules before it sees this step's output.
            until = self._steps + self._confirm_steps
            for key in covered:
                self._unconfirmed[key] = until
        try:
            leases = self._recut(hit, num_external_tokens)
            if leases is None:
                plan.fail = True
            else:
                plan.fail = not self._fill_targets(plan, leases, block_ids)
        except Exception:
            # vLLM is now waiting for this load: report it failed, never crash.
            logger.warning("[PegaKVConnectorV2] load planning failed req=%s", req_id, exc_info=True)
            plan.fail = True
            leases = None
        if plan.fail:
            self._release(hit.leases)
            plan.loads = []
        else:
            self._release_unkept(hit.leases, leases or [])
            if plan.sync:
                self._count("sync_planned")
            else:  # a sync load's blocks are cached at once
                self._track_load(request, leases or [])
        # The load writes its target blocks this step. A failed sync load,
        # planned or not, leaves its covered blocks to be recomputed by the
        # running request (kv_load_failure_policy=recompute): no later table
        # shows those writes, so they are fenced now too.
        self._fence_blocks(
            [
                *plan.covered,
                *(b for _, targets in plan.loads for row in targets for b in row if b is not None),
            ]
        )
        self._plan.loads.append(plan)

    def _recut(self, hit: _Hit, num_external_tokens: int) -> list[_Lease] | None:
        """Leases for the hit vLLM actually allocated.

        vLLM may trim the hit (e.g. block-aligning a replayed prefix); a shorter
        ``p`` that is still a legal resume point is re-planned from the held
        leases instead of failing the request.
        """
        planned = hit.p - hit.num_computed
        if num_external_tokens == planned:
            return hit.leases
        p = hit.num_computed + num_external_tokens
        leases = None
        if num_external_tokens < planned and p % self._layout.step == 0:
            leases = self._assign(p, hit.num_computed, hit.present, hit.leases)
        if leases is None:
            logger.warning(
                "[PegaKVConnectorV2] external tokens %s != planned %s and not re-plannable",
                num_external_tokens,
                planned,
            )
        return leases

    def _covered_blocks(
        self, num_computed: int, num_external_tokens: int, block_ids: tuple[list[int], ...]
    ) -> list[tuple[int, int, int]]:
        """(gid, block index, GPU block id) of the blocks holding the external
        tokens (what a failed sync load leaves invalid)."""
        out: list[tuple[int, int, int]] = []
        end = num_computed + num_external_tokens
        for g in self._layout.groups:
            bt = g.sem.block_tokens
            ids = block_ids[g.gid]
            for b in range(num_computed // bt, min(-(-end // bt), len(ids))):
                if ids[b] != 0:
                    out.append((g.gid, b, ids[b]))
        return out

    def _fill_targets(
        self, plan: LoadPlan, leases: list[_Lease], block_ids: tuple[list[int], ...]
    ) -> bool:
        n_groups = len(self._layout.groups)
        for lease in leases:
            ids = block_ids[self._layout.groups[lease.sid].gid]
            row: list[int | None] = []
            for b in lease.positions:
                if b not in lease.wanted:
                    row.append(None)
                elif b < len(ids) and ids[b] != 0:  # 0 is vLLM's null block
                    row.append(ids[b])
                else:
                    logger.warning(
                        "[PegaKVConnectorV2] req=%s group %s block %s has no GPU block",
                        plan.req_id,
                        lease.sid,
                        b,
                    )
                    return False
            targets: list[list[int | None]] = [[None] * len(row) for _ in range(n_groups)]
            targets[lease.sid] = row
            plan.loads.append((lease.lease, targets))
        return True

    # ----------------------------------------------------------------- saves
    def on_load_errors(self, block_ids: Iterable[int]) -> None:
        """vLLM processed failed sync loads of these GPU blocks (before it
        schedules the next step): stop saving for the rest of the process."""
        block_ids = set(block_ids)
        if block_ids and not self._saves_disabled:
            self._saves_disabled = True
            logger.error(
                "[PegaKVConnectorV2] sync load failed (%d GPU blocks): vLLM may reuse KV "
                "computed on them, saves and sync loads disabled for this process",
                len(block_ids),
            )

    def plan_saves(
        self,
        scheduled: list[tuple[str, int, int]],
        block_ids_of: Callable[[str], tuple[list[int], ...] | None],
        preempted: Iterable[str] = (),
        kv_usage: float | None = 0.0,
        offers: Mapping[str, Sequence[tuple[int, int, int]]] | None = None,
    ) -> None:
        """``scheduled``: (req_id, num_computed_before_step, num_scheduled_tokens).

        ``preempted``: requests vLLM preempted (and whose blocks it freed) since
        the last plan. Their blocks may be reallocated in this very step, so
        the workers must finish pending saves before the step runs.

        ``kv_usage``: vLLM's GPU KV cache usage (0..1) after this step's
        allocations. At or above the policy's ``save_batch_max_kv_usage`` the
        workers send held saves this step (``StepPlan.send_saves``). None
        (unknown) counts as pressure, so batching is off: a silent 0.0 would
        hold blocks that waiting requests need.

        ``offers``: req_id -> [(gid, GPU block id, boundary tokens)], the
        blocks vLLM hands out holding a checkpoint state (vLLM 0.32
        ``boundary_state_offloads``). Checkpoint groups save only from these:
        their block table does not say which block holds which state. An
        offer is used only for the boundary this step's forward computes
        (``semantics.saveable``); the worker saves it before the next step
        can free or reuse the block (checkpoint layouts pin it, ``pin_saves``).

        A request whose blocks include an unconfirmed sync-load block saves
        nothing yet (its range is offered again once the outcome is visible).
        After a failed sync load nothing is saved at all.

        Raises ValueError if the policy returns an item it was not offered: a
        forged (block, hash) pair would poison the store for every reader.
        """
        layout = self._layout
        step = self._steps
        self._steps += 1
        if self._unconfirmed:
            self._unconfirmed = {b: s for b, s in self._unconfirmed.items() if s > step}
        self._plan.step = step
        if self._pin_saves:
            if self._pin is None and not self._saves_disabled:
                self._saves_disabled = True
                logger.error(
                    "[PegaKVConnectorV2] vLLM bound no block pool: saves need pinned "
                    "blocks on this layout, saves disabled for this process"
                )
        else:
            for req_id in preempted:
                self._plan.flush = True
                # The flush finishes the saves of its freed blocks.
                self._saved_blocks.pop(req_id, None)
        self._plan.flush |= self._flush_always
        if self._plan.flush:
            # Every save planned before this step is done before it writes.
            self._clear_fences(step - 1)
        if kv_usage is None:
            if not self._kv_usage_warned:
                self._kv_usage_warned = True
                logger.warning(
                    "[PegaKVConnectorV2] vLLM's KV cache usage is unavailable (cache "
                    "manager not bound): saves are sent every step, batching is off"
                )
            self._plan.kv_usage_unknown = True
            self._plan.send_saves = True
        elif kv_usage >= self._policy.save_batch_max_kv_usage:
            self._plan.send_saves = True
        if self._saves_disabled:
            return
        items: list[SaveItem] = []
        for req_id, computed, n_sched in scheduled:
            request = self._requests.get(req_id)
            if request is None or n_sched <= 0:
                continue
            # Only tokens vLLM already knows have hashes; draft (spec-decode) or
            # not-yet-sampled (async scheduling) positions are offered by a
            # later step, which starts from where this one stopped.
            before = min(computed, self._saved_upto.get(req_id, computed))
            after = min(computed + n_sched, request.num_tokens)
            if after <= before:
                continue
            ranges = []
            for g in layout.groups:
                lo, hi = self._sem.saveable(g.sem, before, after)
                if lo < hi:
                    ranges.append((g, range(lo, hi)))
            if ranges:
                block_ids = block_ids_of(req_id)
                if block_ids is None:
                    # No block ids this step: keep `before` so a later step
                    # offers this range again.
                    self._saved_upto[req_id] = before
                    continue
                if self._unconfirmed and self._touches_unconfirmed(block_ids):
                    # Computed on top of a sync load whose outcome vLLM has
                    # not processed yet (or sharing its blocks): hold.
                    self._saved_upto[req_id] = before
                    continue
                for g, blocks in ranges:
                    if g.sem.kind == "checkpoint":
                        ids = _offered_ids(g, blocks, (offers or {}).get(req_id, ()))
                    else:
                        ids = block_ids[g.gid]
                    items.extend(self._save_items(request, g, blocks, ids))
            # Blocks _save_items skips (null id, no hash yet) are not offered
            # again. In vLLM 0.32 neither happens for full or window groups:
            # `after` is capped at num_tokens (every full block has a hash) and
            # blocks written this step are allocated, not null. A checkpoint
            # vLLM did not offer (it did not retain that boundary) is skipped.
            self._saved_upto[req_id] = after
        if not items:
            return
        offered = {item: item for item in items}
        chosen = self._policy.choose_saves(tuple(items))
        # One GroupSave per (storage group, request): the server pins one region
        # per layer entry, freed only once every block in it is evicted, so an
        # entry mixing requests keeps the whole region pinned for the
        # longest-lived of them.
        by_key: dict[tuple[int, str], GroupSave] = {}
        sid_of = {g.gid: sid for sid, g in enumerate(layout.groups)}
        seen: set[SaveItem] = set()
        for picked in chosen:
            item = offered.get(picked)
            if item is None:
                raise ValueError(
                    f"policy.choose_saves returned an item it was not offered: {picked!r}"
                )
            if item in seen:
                continue
            seen.add(item)
            if not self._pin_saves:
                self._saved_blocks.setdefault(item.req_id, {})[item.block_id] = step
            sid = sid_of[item.group]
            save = by_key.setdefault((sid, item.req_id), GroupSave(sid, [], []))
            save.block_ids.append(item.block_id)
            save.block_hashes.append(item.block_hash)
        # Stable sort: within a group, requests keep their scheduled order.
        saves = sorted(by_key.values(), key=lambda s: s.sid)
        if self._pin_saves and saves:
            assert self._pin is not None
            self._pins[step] = self._pin([b for gs in saves for b in gs.block_ids])
        if saves:
            self._last_save_step = step
        self._plan.saves.extend(saves)

    # ----------------------------------------------------------- pinned saves
    def bind_pins(self, pin: Callable[[list[int]], Any], unpin: Callable[[Any], None]) -> None:
        """``pin(block_ids)`` takes a reference on each GPU block (a block may
        repeat) and returns a token; ``unpin(token)`` drops them."""
        self._pin, self._unpin = pin, unpin

    def on_saved_through(self, step: int) -> None:
        """Every worker finished the saves of plans up to ``step``: unpin
        them, and drop the fences on them."""
        self._clear_fences(step)
        self._saved_through = max(self._saved_through, step)
        if not self._pins:
            return
        assert self._unpin is not None
        for s in [s for s in self._pins if s <= step]:
            self._unpin(self._pins.pop(s))

    # ----------------------------------------------------------------- fences
    def fence_writes(
        self,
        tables: Iterable[tuple[Sequence[Sequence[int] | None], int]],
        blocks: Iterable[int] = (),
    ) -> None:
        """This step's KV writes, before ``plan_saves``: a write to a fenced
        block first waits for the saves that read it (``StepPlan.fence_step``).

        ``tables``: per request vLLM allocated blocks for this step, (block ids
        per KV cache group, tokens computed before the step). In a positional
        group a block wholly below those tokens is a prefix hit, only read;
        the rest is written by this step or a later one. Pass all-new blocks
        (appended to a running request's table) with 0 tokens. ``blocks``:
        other blocks written before the forward (zeroed, copy targets). Load
        targets are fenced when the load is planned.
        """
        if not self._fenced:
            return

        def written() -> Iterable[int]:
            yield from blocks
            for groups, computed in tables:
                for gid, ids in enumerate(groups):
                    if ids:
                        bt = self._positional_bt.get(gid)
                        yield from ids[computed // bt :] if bt else ids

        self._fence_blocks(written())

    def _fence_blocks(self, blocks: Iterable[int]) -> None:
        fenced = self._fenced
        if not fenced:
            return
        step = max((fenced.get(b, -1) for b in blocks), default=-1)
        if step < 0:
            return
        self._plan.fence_step = max(self._plan.fence_step, step)
        # The worker waits for every save of plans up to ``step`` before
        # this plan's writes: those fences are done for every later plan.
        self._clear_fences(step)

    def _clear_fences(self, step: int) -> None:
        self._fence_floor = max(self._fence_floor, step)
        if not self._fence_steps:
            return
        for s in [s for s in self._fence_steps if s <= step]:
            for b in self._fence_steps.pop(s):
                if self._fenced.get(b) == s:
                    del self._fenced[b]

    def has_pending_saves(self) -> bool:
        """Saves were planned that the workers have not reported done: vLLM
        must keep stepping. Blocks stay pinned until then, and held (batched)
        saves age only as steps run, so an idle engine would never send them
        (the next request's lookup would miss them)."""
        return bool(self._pins) or self._last_save_step > self._saved_through

    def _count_miss(
        self, request: Any, num_computed: int, present: tuple[frozenset[int], ...], p: int
    ) -> None:
        """Group coherence: at the longest resume point the full groups back,
        which other groups are missing (``hit stats`` log line)."""
        groups = self._layout.groups
        step = self._layout.step
        # Storage group 0 is full: its contiguous prefix bounds every hit.
        contiguous = 0
        while contiguous in present[0]:
            contiguous += 1
        cap = max(request.num_tokens - 1, 0)
        p_full = min(contiguous * groups[0].sem.block_tokens, cap) // step * step
        key = (num_computed, p_full)
        if self._counted.get(request.request_id) == key:
            return
        self._counted[request.request_id] = key
        stats = self.hit_stats
        stats["lookups"] = stats.get("lookups", 0) + 1
        if p_full > max(p, num_computed):
            # Storage groups (sids) lacking their needed blocks at p_full.
            missing = [
                sid
                for sid, g in enumerate(groups)
                if not present[sid].issuperset(range(*self._sem.needed(g.sem, p_full)))
            ]
            key = "missing=" + ",".join(map(str, missing))  # "missing=" alone: policy cut
            stats[key] = stats.get(key, 0) + 1
            stats["lost_tokens"] = stats.get("lost_tokens", 0) + p_full - max(p, num_computed)
        self._hit_dirty = True
        if self._clock() - self._hit_logged_at >= self.HIT_LOG_S:
            self.report_hit_stats()

    def report_hit_stats(self) -> None:
        """Log the cumulative counters. Called at most every ``HIT_LOG_S`` while
        lookups run, when the last live request finishes, and at shutdown:
        vLLM may kill the engine core without running ``shutdown``, so a run's
        final counters must not depend on it."""
        self._hit_dirty = False
        self._hit_logged_at = self._clock()
        logger.info(
            "[PegaKVConnectorV2] hit stats %s lookup_cost %s",
            dict(sorted(self.hit_stats.items())),
            self.lookup_cost,
        )

    def _touches_unconfirmed(self, block_ids: tuple[list[int], ...]) -> bool:
        return any(
            b < len(block_ids[gid]) and block_ids[gid][b] == block_id
            for gid, b, block_id in self._unconfirmed
        )

    def _save_items(
        self, request: Any, g: GroupInfo, blocks: range, ids: list[int]
    ) -> list[SaveItem]:
        hashes = self._layout.group_hashes(g, request.block_hashes)
        out = []
        for b in blocks:
            if b >= len(hashes) or b >= len(ids) or ids[b] == 0:
                continue
            out.append(SaveItem(request.request_id, g.gid, g.sem.kind, b, ids[b], hashes[b]))
        return out

    def take_plan(self) -> StepPlan:
        plan, self._plan = self._plan, StepPlan()
        return plan

    # ------------------------------------------------------------- lifecycle
    def request_finished(self, req_id: str) -> bool:
        """Always False: vLLM frees the request's blocks at once.

        A block one of the request's saves may still read is fenced on the
        plan step of that save instead: vLLM may hand it out again, but a step
        that writes it first waits until the save is done (``fence_writes``),
        and the worker reports a save done only once its GPU->CPU copy has
        completed. A block the request shares through the prefix cache may be
        freed only later, by its last holder: the fence stays until the save
        is done, whoever frees it. A preempted request's saves are covered by
        the step's flush."""
        self._drop_hit(req_id)
        self._loading_since.pop(req_id, None)
        self._untrack_load(req_id)  # aborted while loading: never cached
        self._deferred_since.pop(req_id, None)
        self._saved_upto.pop(req_id, None)
        self._requests.pop(req_id, None)
        self._counted.pop(req_id, None)
        if self._hit_dirty and not self._requests:
            self.report_hit_stats()  # idle: the counters are final for now
        saved = self._saved_blocks.pop(req_id, None)
        for block, step in (saved or {}).items():
            if step > self._fence_floor and self._fenced.get(block, -1) < step:
                self._fenced[block] = step
                self._fence_steps.setdefault(step, []).append(block)
        return False

    def shutdown(self) -> None:
        if self._hit_dirty:
            self.report_hit_stats()
        for req_id in list(self._hits):
            self._drop_hit(req_id)
        # Load plans never handed to the workers still hold their leases.
        self._release_tokens(lease for lp in self._plan.loads for lease, _ in lp.loads)
        self._plan.loads.clear()

    def _drop_hit(self, req_id: str) -> None:
        hit = self._hits.pop(req_id, None)
        if hit is not None:
            self._release(hit.leases)

    def _release_unkept(self, held: list[_Lease], kept: list[_Lease]) -> None:
        keep = {lease.lease for lease in kept}
        self._release([lease for lease in held if lease.lease not in keep])

    def _release(self, leases: list[_Lease]) -> None:
        self._release_tokens(lease.lease for lease in leases)

    def _release_tokens(self, tokens: Iterable[bytes]) -> None:
        for token in tokens:
            try:
                self._rpc.release(token)
            except Exception:
                # Unreleased leases expire on the server; nothing else to undo.
                logger.warning("[PegaKVConnectorV2] lease release failed", exc_info=True)
