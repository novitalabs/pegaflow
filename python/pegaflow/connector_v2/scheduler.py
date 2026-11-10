"""Scheduler side: query -> resolve -> lease -> load plan, and save plans.

vLLM-free: the connector passes plain values in and the server client is
duck-typed (``query_object`` / ``release``), so unit tests drive it with fakes.

Lookup, per request, with ``local`` = vLLM's own hit:

1. Every full namespace gets a prefix query from block ``local / bt``. The
   shortest answer bounds every hit (``top``).
2. Window and checkpoint namespaces get a membership query over the blocks
   the resume points ``<= top`` need. Checkpoints ask every point, one block
   each, as v1 asks membership over the whole prefix (checkpoints sit where
   past prefill chunks happened to end). Windows ask the top point first; if
   no hit reaches it, they ask every point back to the local hit, as v1 does
   because window hits are not monotone. Blocks below the local hit are on
   the GPU already and count as present.
3. ``semantics.resolve`` picks ``p``; only blocks past the local hit load.

While the server fetches from SSD or peers a query answers Loading and vLLM
asks again next step. As v1 keeps a probe per request, each namespace's
answer is kept until the lookup ends, so a poll only re-asks what is still
loading, under the same query id and keys. There is no wait budget: as in v1
the request waits until the server answers.

Loads are always asynchronous. A plan that cannot load (vLLM trimmed the hit
off a resume point or handed a null block) is a failed receive like a failed
transfer; vLLM applies its ``kv_load_failure_policy``. The request never loads
again if it is recomputed.

Saves: every block a step completes is pinned in vLLM's block pool from the
step that plans it until every worker reports that step saved
(``on_saved_through``), so vLLM never reuses a block a save still reads. That
also covers preemption, where v1 waits for the request's saves in
``handle_preemptions``.

Stats: a lookup that waited on Loading is one prefetch, as in v1
(``take_stats``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import semantics as sem
from .layout import Layout

logger = logging.getLogger(__name__)


@dataclass
class Policy:
    """Choices inside what semantics allows; can only narrow a hit."""

    def accept(self, p: int) -> int:
        return p


@dataclass
class LoadPlan:
    """One request's load, run by every worker.

    ``blocks``: (namespace index, object prefix, GPU block id).
    ``fail``: report the request as a failed receive without loading.
    """

    req_id: str
    leases: list[bytes] = field(default_factory=list)
    blocks: list[tuple[int, bytes, int]] = field(default_factory=list)
    fail: bool = False


@dataclass
class StepPlan:
    loads: list[LoadPlan] = field(default_factory=list)
    # (namespace index, object prefix, GPU block id)
    saves: list[tuple[int, bytes, int]] = field(default_factory=list)
    step: int = -1


@dataclass
class _Hit:
    p: int
    local: int
    present: list[frozenset[int]]  # per namespace, local blocks included
    keys: list[list[bytes]]  # per namespace
    leases: dict[int, bytes]  # namespace index -> lease
    n_tokens: int


@dataclass
class _Ask:
    """One namespace query of a lookup; ``hits`` is None while Loading."""

    keys: list[bytes]
    prefix: bool
    hits: list[int] | None = None
    lease: bytes | None = None


@dataclass
class _Req:
    request: Any
    # Per namespace index, the server's query id, kept for the request's life
    # as v1 keys prefetches by request id: SSD/RDMA fetches and the server's
    # failed-peer list follow the request across scheduler calls.
    query_ids: dict[int, str] = field(default_factory=dict)
    # Per (namespace index, phase), the answer of the lookup in progress.
    asks: dict[tuple[int, int], _Ask] = field(default_factory=dict)
    hit: _Hit | None = None
    saved_upto: int | None = None
    junction: int = 0  # where the stored full-attention prefix ends
    no_load: bool = False  # a load failed: if vLLM recomputes, never load again


class _Loading(Exception):
    """A namespace query is still fetching from SSD or peers."""


class Scheduler:
    def __init__(
        self,
        layout: Layout,
        rpc: Any,
        instance_id: str,
        policy: Policy | None = None,
    ):
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._policy = policy or Policy()
        self._reqs: dict[str, _Req] = {}
        self._plan = StepPlan()
        self._steps = 0
        self._pin: Callable[[list[int]], Any] | None = None
        self._unpin: Callable[[Any], None] | None = None
        self._pins: dict[int, Any] = {}  # plan step -> pin token
        self._has_checkpoint = any(ns.group.kind == sem.CHECKPOINT for ns in layout.namespaces)
        self._prefetching: dict[str, float] = {}  # req id -> first Loading answer
        self._stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ hits
    def _rec(self, request: Any) -> _Req:
        rec = self._reqs.get(request.request_id)
        if rec is None:
            rec = self._reqs[request.request_id] = _Req(request)
        return rec

    def get_num_new_matched_tokens(self, request: Any, local: int) -> tuple[int | None, bool]:
        rec = self._rec(request)
        hit = rec.hit
        # vLLM asks again while the request waits for blocks: reuse the leased
        # hit as long as the request has not moved, instead of querying again.
        if hit is not None and hit.local == local and hit.n_tokens == request.num_tokens:
            return hit.p - local, True
        self._drop_hit(rec)
        if rec.no_load or getattr(request, "skip_reading_prefix_cache", False):
            return 0, False
        try:
            hit = self._lookup(rec, local)
        except _Loading:
            self._prefetching.setdefault(request.request_id, time.perf_counter())
            return None, False
        except Exception:
            logger.warning(
                "[PegaKVConnectorV2] lookup failed req=%s", request.request_id, exc_info=True
            )
            self._prefetching.pop(request.request_id, None)  # cancelled, as v1
            hit = None
        # The lookup is over: release every answer the hit does not keep.
        kept = set(hit.leases.values()) if hit is not None else set()
        self._drop_asks(rec, kept)
        self._hint_junction(request, hit.p if hit is not None else local, rec.junction)
        since = self._prefetching.pop(request.request_id, None)
        if since is not None:
            blocks = sum(map(len, (self._wanted(hit, hit.p) or {}).values())) if hit else 0
            self._stats.setdefault("prefetch_duration", []).append(
                (time.perf_counter() - since) * 1000
            )
            self._stats.setdefault("prefetch_blocks", []).append(blocks)
        if hit is None:
            return 0, False
        rec.hit = hit
        return hit.p - local, True

    def _query(
        self, rec: _Req, i: int, keys: list[bytes], prefix: bool, phase: int = 0
    ) -> tuple[list[int], bytes | None]:
        """Hit positions into ``keys`` and the lease; raises ``_Loading``.

        A kept answer for the same keys is reused. A query still loading for
        other keys is polled with its own keys first (the server answers
        positions of the keys it was asked), then discarded, as v1 drops a
        drifted probe.
        """
        ask = rec.asks.get((i, phase))
        if ask is not None and ask.hits is not None and (ask.keys, ask.prefix) != (keys, prefix):
            self._release([ask.lease] if ask.lease else [])
            ask = None
        if ask is None:
            ask = rec.asks[(i, phase)] = _Ask(list(keys), prefix)
        if ask.hits is None:
            ns = self._layout.namespaces[i].namespace
            query_id, ans = self._rpc.query_object(
                self._instance_id,
                [(ns, k) for k in ask.keys],
                ask.prefix,
                rec.query_ids.get(i, ""),
            )
            if query_id:
                rec.query_ids[i] = query_id
            if not hasattr(ans, "num_hit_blocks"):  # QueryLoading
                raise _Loading
            hits = list(ans.hit_positions) if not ask.prefix else list(range(ans.num_hit_blocks))
            ask.hits, ask.lease = hits, (ans.lease or None)
            if hits and not ask.lease:
                raise RuntimeError(f"namespace {i}: {len(hits)} hits without a lease")
        if (ask.keys, ask.prefix) != (keys, prefix):
            return self._query(rec, i, keys, prefix, phase)
        return ask.hits, ask.lease

    def _members(
        self,
        rec: _Req,
        i: int,
        keys: list[bytes],
        local: int,
        points: Sequence[int],
        leases: dict[int, bytes],
        phase: int,
    ) -> frozenset[int]:
        """Blocks of namespace ``i`` present for ``points``, GPU ones included."""
        g = self._layout.namespaces[i].group
        on_gpu = frozenset(range(min(local // g.block_tokens, len(keys))))
        want = sorted({b for p in points for b in sem.needed(g, p)} - on_gpu)
        want = [b for b in want if b < len(keys)]
        if not want:
            return on_gpu
        hits, lease = self._query(rec, i, [keys[b] for b in want], False, phase)
        if lease:
            leases[i] = lease
        return on_gpu | {want[h] for h in hits}

    def _lookup(self, rec: _Req, local: int) -> _Hit | None:
        request = rec.request
        layout = self._layout
        nss = layout.namespaces
        groups = layout.groups
        n_tokens = request.num_tokens
        rec.junction = 0
        # vLLM computes at least the last prompt token: blocks past it are never
        # needed. Without a checkpoint the block holding it loads too (as v1)
        # and vLLM recomputes the token over it.
        cap = max(n_tokens - 1, 0)
        keys = []
        for ns in nss:
            bt = ns.group.block_tokens
            n = cap // bt if self._has_checkpoint else -(-cap // bt)
            keys.append(ns.keys(request.block_hashes)[:n])
        if sem.resolve(groups, [frozenset(range(len(k))) for k in keys], n_tokens) <= local:
            return None  # even a store holding everything cannot beat vLLM

        leases: dict[int, bytes] = {}
        present: list[frozenset[int]] = [frozenset()] * len(nss)
        top = cap
        for i, ns in enumerate(nss):
            if ns.group.kind != sem.FULL:
                continue
            start = local // ns.group.block_tokens
            hits, lease = [], None
            if start < len(keys[i]):
                hits, lease = self._query(rec, i, keys[i][start:], True)
            if lease:
                leases[i] = lease
            present[i] = frozenset(range(start + len(hits)))
            top = min(top, (start + len(hits)) * ns.group.block_tokens)
        if self._has_checkpoint:
            rec.junction = top
        points = [p for p in sem.candidates(groups, top, n_tokens) if p > local]
        if not points:
            return None
        windows = [i for i, ns in enumerate(nss) if ns.group.kind == sem.WINDOW]
        for i, ns in enumerate(nss):
            if ns.group.kind == sem.FULL:
                continue
            probe = points if ns.group.kind == sem.CHECKPOINT else points[:1]
            present[i] = self._members(rec, i, keys[i], local, probe, leases, 0)
        if windows and sem.resolve(groups, present, n_tokens) < points[0]:
            # As v1: an earlier point needs another window, so ask every
            # point back to the local hit. This answer replaces the first,
            # whose lease the lookup's end releases.
            for i in windows:
                leases.pop(i, None)
                present[i] = self._members(rec, i, keys[i], local, points, leases, 1)

        p = sem.resolve(groups, present, n_tokens)
        p = max(0, min(int(self._policy.accept(p)), p))
        hit = _Hit(p, local, present, keys, leases, n_tokens)
        if p <= local or not sem.resumable_at(groups, p) or not self._wanted(hit, p):
            return None
        return hit

    def _hint_junction(self, request: Any, resumable: int, junction: int) -> None:
        """As v1: when the stored full-attention prefix runs past the resume
        point, ask vLLM to end a prefill chunk at its end
        (``Request.shared_prefix_boundary``). vLLM then commits the checkpoint
        state there and offers it for saving, so the next request sharing the
        prefix resumes from its end instead of an earlier checkpoint."""
        if not hasattr(request, "shared_prefix_boundary"):
            return
        if resumable < junction < request.num_tokens and junction > request.shared_prefix_boundary:
            request.shared_prefix_boundary = junction

    def _wanted(self, hit: _Hit, p: int) -> dict[int, list[int]] | None:
        """Blocks to load per namespace for a hit at ``p``; None if not backed."""
        out: dict[int, list[int]] = {}
        for i, ns in enumerate(self._layout.namespaces):
            need = sem.needed(ns.group, p)
            if not hit.present[i].issuperset(need):
                return None
            blocks = [b for b in need if b * ns.group.block_tokens >= hit.local]
            if blocks:
                if i not in hit.leases:
                    return None
                out[i] = blocks
        return out or None

    def get_loaded_group_ids(self, req_id: str) -> tuple[int, ...]:
        """vLLM groups the pending hit writes into (vLLM skips zeroing them)."""
        rec = self._reqs.get(req_id)
        if rec is None or rec.hit is None:
            return ()
        wanted = self._wanted(rec.hit, rec.hit.p) or {}
        return tuple(sorted(self._layout.namespaces[i].gid for i in wanted))

    def update_state_after_alloc(
        self, request: Any, block_ids: Sequence[Sequence[int]], num_external: int
    ) -> None:
        rec = self._reqs.get(request.request_id)
        if rec is None or rec.hit is None:
            return
        hit, rec.hit = rec.hit, None
        if num_external == 0:
            self._release(hit.leases.values())
            return
        plan = LoadPlan(request.request_id)
        p = hit.local + num_external  # vLLM may trim the hit
        wanted = self._wanted(hit, p) if sem.resumable_at(self._layout.groups, p) else None
        if wanted is not None:
            for i, blocks in wanted.items():
                gpu = block_ids[self._layout.namespaces[i].gid]
                for b in blocks:
                    if b >= len(gpu) or gpu[b] == 0:  # 0 is vLLM's null block
                        wanted = None
                        break
                    plan.blocks.append((i, hit.keys[i][b], gpu[b]))
                if wanted is None:
                    break
        if wanted is None:
            logger.warning(
                "[PegaKVConnectorV2] req=%s cannot load %d external tokens: load failed",
                request.request_id,
                num_external,
            )
            plan.fail, plan.blocks = True, []
            rec.no_load = True
            self._release(hit.leases.values())
        else:
            keep = {hit.leases[i] for i in wanted}
            self._release(lease for lease in hit.leases.values() if lease not in keep)
            plan.leases = sorted(keep)
            rec.saved_upto = p  # the loaded prefix is in the store already
        self._plan.loads.append(plan)

    def on_loads_failed(self, req_ids: Iterable[str]) -> None:
        """If vLLM recomputes these requests, they never load again."""
        for req_id in req_ids:
            rec = self._reqs.get(req_id)
            if rec is not None:
                rec.no_load = True

    # ----------------------------------------------------------------- saves
    def bind_pins(self, pin: Callable[[list[int]], Any], unpin: Callable[[Any], None]) -> None:
        self._pin, self._unpin = pin, unpin

    def plan_saves(
        self,
        scheduled: Iterable[tuple[str, int, int]],
        block_ids_of: Callable[[str], Sequence[Sequence[int]] | None],
        offers: Mapping[str, Sequence[tuple[int, int, int]]] | None = None,
    ) -> None:
        """``scheduled``: (req_id, num_computed_before, num_scheduled_tokens).
        ``offers``: req_id -> [(gid, GPU block id, boundary tokens)] of
        checkpoint states vLLM hands out (``boundary_state_offloads``)."""
        step = self._steps
        self._steps += 1
        self._plan.step = step
        if self._pin is None:
            return  # no block pool bound: saves would be unsafe
        nss = self._layout.namespaces
        saves: list[tuple[int, bytes, int]] = []
        for req_id, computed, n_sched in scheduled:
            rec = self._reqs.get(req_id)
            if rec is None or n_sched <= 0:
                continue
            request = rec.request
            before = computed if rec.saved_upto is None else min(computed, rec.saved_upto)
            # Only tokens vLLM knows have hashes (no drafts, no unsampled ones).
            after = min(computed + n_sched, request.num_tokens)
            if after <= before:
                continue
            ranges = [
                (i, sem.saveable(ns.group, before, after))
                for i, ns in enumerate(nss)
                if ns.group.kind in (sem.FULL, sem.WINDOW)
            ]
            if any(r for _, r in ranges):
                block_ids = block_ids_of(req_id)
                if block_ids is None:
                    rec.saved_upto = before  # offered again next step
                    continue
                for i, blocks in ranges:
                    keys = nss[i].keys(request.block_hashes)
                    gpu = block_ids[nss[i].gid]
                    saves.extend(
                        (i, keys[b], gpu[b])
                        for b in blocks
                        if b < len(keys) and b < len(gpu) and gpu[b] != 0
                    )
            rec.saved_upto = after
        by_gid = {ns.gid: i for i, ns in enumerate(nss) if ns.group.kind == sem.CHECKPOINT}
        for req_id, states in (offers or {}).items():
            rec = self._reqs.get(req_id)
            if rec is None or not by_gid:
                continue
            for gid, block_id, boundary in states:
                i = by_gid.get(gid)
                if i is None or block_id == 0:
                    continue
                bt = nss[i].group.block_tokens
                keys = nss[i].keys(rec.request.block_hashes)
                b = boundary // bt - 1
                if boundary % bt == 0 and 0 <= b < len(keys):
                    saves.append((i, keys[b], block_id))
        if saves:
            self._pins[step] = self._pin([block_id for _, _, block_id in saves])
            self._plan.saves = saves

    def on_saved_through(self, step: int) -> None:
        """Every worker finished the saves of plans up to ``step``: unpin them."""
        for s in [s for s in self._pins if s <= step]:
            assert self._unpin is not None
            self._unpin(self._pins.pop(s))

    def has_pending_saves(self) -> bool:
        return bool(self._pins)

    def take_plan(self) -> StepPlan:
        plan, self._plan = self._plan, StepPlan()
        return plan

    # ------------------------------------------------------------- lifecycle
    def request_finished(self, req_id: str) -> None:
        rec = self._reqs.pop(req_id, None)
        self._prefetching.pop(req_id, None)
        if rec is not None:
            self._drop_hit(rec)
            self._drop_asks(rec)

    def take_stats(self) -> dict[str, Any]:
        """Stats since the last call, as ``PegaKVConnectorStats`` data."""
        stats, self._stats = self._stats, {}
        if self._prefetching:
            stats["pending_prefetches"] = len(self._prefetching)
        return stats

    def shutdown(self) -> None:
        for rec in self._reqs.values():
            self._drop_hit(rec)
            self._drop_asks(rec)
        self._release(lease for lp in self._plan.loads for lease in lp.leases)
        self._plan.loads.clear()

    def _drop_hit(self, rec: _Req) -> None:
        if rec.hit is not None:
            hit, rec.hit = rec.hit, None
            self._release(hit.leases.values())

    def _drop_asks(self, rec: _Req, kept: Container[bytes] = ()) -> None:
        asks, rec.asks = rec.asks, {}
        self._release(a.lease for a in asks.values() if a.lease and a.lease not in kept)

    def _release(self, leases: Iterable[bytes]) -> None:
        for lease in leases:
            try:
                self._rpc.release(lease)
            except Exception:
                # Unreleased leases expire on the server.
                logger.warning("[PegaKVConnectorV2] lease release failed", exc_info=True)
