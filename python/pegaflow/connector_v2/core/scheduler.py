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
from collections.abc import Callable, Iterable
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
    lifecycle the worker must respect: ``finished`` lists requests whose blocks
    vLLM holds for our saves (report them in ``finished_sending`` once every
    save that may read them is done), and ``flush`` asks the worker to finish
    every pending save before this step touches any KV (vLLM freed blocks a
    pending save may still read, e.g. on preemption).
    """

    loads: list[LoadPlan] = field(default_factory=list)
    saves: list[GroupSave] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    flush: bool = False


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


@dataclass
class _Lookup:
    leases: list[_Lease]
    present: tuple[frozenset[int], ...]  # rebuilt from the server answers only
    loading: bool


class Scheduler:
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
    ):
        """``sync_loads``: vLLM can recover from a failed synchronous load of this
        layout (it maps ``invalid_block_ids`` back to requests only with one
        KV cache group, and the worker's block zeroing is not ordered against
        the server's copy).

        ``confirm_steps``: steps vLLM schedules before it processes a step's
        output (vLLM's ``max_concurrent_batches``). A sync load's outcome, and
        the rewind of a failed one, is visible only that many steps later."""
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
        self._saved_upto: dict[str, int] = {}  # token count already offered for saving
        self._saving: set[str] = set()  # requests with saves since their blocks were last freed
        # vLLM frees window blocks that slide out while the request runs, which a
        # pending save may still read: such layouts flush before every step.
        self._flush_always = any(g.sem.kind == "window" for g in layout.groups)
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
        self._plan = StepPlan()

    # ------------------------------------------------------------------ hits
    def on_new_request(self, request: Any) -> None:
        self._requests[request.request_id] = request

    def get_num_new_matched_tokens(
        self, request: Any, num_computed: int
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        self._requests.setdefault(req_id, request)
        # A previous hit that was never allocated (allocation failed) is stale.
        self._drop_hit(req_id)

        try:
            lookup = self._lookup(request)
        except Exception:
            logger.warning("[PegaKVConnectorV2] query failed req=%s", req_id, exc_info=True)
            self._loading_since.pop(req_id, None)
            return 0, False

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

    def _lookup(self, request: Any) -> _Lookup:
        layout = self._layout
        # resolve keeps the last token for vLLM, so blocks past it are never needed.
        limit_tokens = max(request.num_tokens - 1, 0)
        leases: list[_Lease] = []
        present: list[frozenset[int]] = []
        loading = False
        try:
            for sid, g in enumerate(layout.groups):
                hashes = layout.group_hashes(g, request.block_hashes)
                hashes = hashes[: limit_tokens // g.sem.block_tokens]
                if not hashes or g.sem.kind == "scratch":
                    present.append(frozenset())
                    continue
                ans = self._rpc.query_prefetch(
                    self._instance_id, hashes, request.request_id, False, sid
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
    ) -> None:
        """``scheduled``: (req_id, num_computed_before_step, num_scheduled_tokens).

        ``preempted``: requests vLLM preempted (and whose blocks it freed) since
        the last plan. Their blocks may be reallocated in this very step, so
        the workers must finish pending saves before the step runs.

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
        for req_id in preempted:
            self._plan.flush = True
            self._saving.discard(req_id)  # the flush covers its old blocks
        self._plan.flush |= self._flush_always
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
                    items.extend(self._save_items(request, g, blocks, block_ids[g.gid]))
            # Blocks _save_items skips (null id, no hash yet) are not offered
            # again. In vLLM 0.32 neither happens for full or window groups:
            # `after` is capped at num_tokens (every full block has a hash) and
            # blocks written this step are allocated, not null.
            self._saved_upto[req_id] = after
        if not items:
            return
        offered = {item: item for item in items}
        chosen = self._policy.choose_saves(tuple(items))
        by_sid: dict[int, GroupSave] = {}
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
            self._saving.add(item.req_id)
            sid = sid_of[item.group]
            save = by_sid.setdefault(sid, GroupSave(sid, [], []))
            save.block_ids.append(item.block_id)
            save.block_hashes.append(item.block_hash)
        self._plan.saves.extend(by_sid[sid] for sid in sorted(by_sid))

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
        """True if vLLM must hold the request's blocks until the workers report
        it in ``finished_sending`` (a save of its blocks may still be pending)."""
        self._drop_hit(req_id)
        self._loading_since.pop(req_id, None)
        self._saved_upto.pop(req_id, None)
        self._requests.pop(req_id, None)
        if req_id not in self._saving:
            return False
        self._saving.discard(req_id)
        self._plan.finished.append(req_id)
        return True

    def shutdown(self) -> None:
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
