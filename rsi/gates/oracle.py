"""Offline hit-rate oracle (G3) over an RsiProbe ``requests.jsonl``.

Replays the trace with an unlimited store. A block of request ``j`` ending at
token ``e`` (prompt or generated) is stored at the first scheduler step whose
``progress`` entry has ``computed_after >= e`` and is visible to requests
first scheduled *after* that step (a connector can only save it once the
step's forward ran). Checkpoint groups (Mamba / linear-attention state) only
store a block when a step ended exactly on its boundary: vLLM materializes
state at step ends, never in the middle of a chunk.

The oracle hit of a request is ``semantics.resolve`` over the blocks present
for its prompt at its first schedule.

Group Kinds come from the spec dump, not from the connector under test, so a
connector cannot lower the bar by declaring a stricter Kind:

- attention without a window (Full / MLA / sink): Full
- sliding window ``sw``: Window{sw - 1}; ``sw == 1`` needs no past tokens: Scratch
- Mamba / linear-attention state: Checkpoint
- ``UniformTypeKVCacheSpecs``: the Kind of its inner specs (all must agree)
- anything else: rejected, the gate owner has to map it first.

Per request ``i`` (first episode): ``local_i`` = vLLM's local prefix-cache hit,
``actual_i`` = tokens computed at first schedule (local + external),
``oracle_i`` = oracle hit. The connector is scored on what only it can serve:

- ``ext_i  = max(0, min(actual_i, oracle_i) - local_i)``
- ``ext*_i = max(0, oracle_i - local_i)``
- ``E_ext  = sum(ext) / sum(ext*)``
- ``excess = sum(max(0, actual_i - max(oracle_i, local_i)))``: tokens the
  connector claimed beyond an unlimited store. An honest connector has none
  beyond a partial last block (e.g. reporting ``prompt_len - 1``).
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pegaflow import semantics as sem

_FULL = {"FullAttentionSpec", "MLAAttentionSpec", "SinkFullAttentionSpec"}
_WINDOW = {"SlidingWindowSpec", "SlidingWindowMLASpec"}
_STATE = {"MambaSpec"}


def _kind(s: dict[str, Any], where: str) -> str:
    """'full' | 'window<N>' | 'scratch' | 'checkpoint' for one spec dict."""
    mro = set(s["mro"])
    if "UniformTypeKVCacheSpecs" in mro:
        inner = {_kind(v, f"{where}/{n}") for n, v in s["kv_cache_specs"].items()}
        if len(inner) != 1:
            raise ValueError(f"oracle: {where} UniformTypeKVCacheSpecs mixes Kinds {inner}")
        return inner.pop()
    sw = s.get("sliding_window")
    if mro & _WINDOW or (mro & _FULL and sw):
        return "scratch" if sw == 1 else f"window{sw - 1}"
    if mro & _FULL:
        return "full"
    if mro & _STATE:
        return "checkpoint"
    raise ValueError(f"oracle has no Kind mapping for {where}: {s['class']}")


def group_kinds(spec: dict[str, Any]) -> list[tuple[str, Any, int]]:
    """(label, semantics.Group | None, block_size) per vLLM group; None = skipped."""
    out = []
    for g in spec["kv_cache_config"]["groups"]:
        s = g["spec"]
        bt = s["block_size"]
        if not g.get("enable_kv_transfer", True) or g.get("is_eagle_group"):
            out.append(("skip", None, bt))
            continue
        label = _kind(s, f"group {g['id']}")
        if label == "full":
            group = sem.Group.full(bt)
        elif label == "scratch":
            group = sem.Group.scratch(bt)
        elif label == "checkpoint":
            group = sem.Group.checkpoint(bt)
        else:
            group = sem.Group.window(bt, int(label.removeprefix("window")))
        out.append((label, group, bt))
    return out


def warmup_problems(reqs: list[dict[str, Any]], warmup: str = "_warmup") -> list[str]:
    """The warmup request must share no prefix block with the workload:
    keys chain the whole prefix, so disjoint keys = no common prefix block."""
    warm = [r for r in reqs if r.get("pass") == warmup]
    keys = {k for r in warm for k in r["keys"]}
    shared = sum(1 for r in reqs if r.get("pass") != warmup and any(k in keys for k in r["keys"]))
    out = [f"{len(warm)} warmup requests (expected at most 1)"] if len(warm) > 1 else []
    if shared:
        out.append(f"{shared} workload requests share a prefix block with the warmup")
    return out


def load_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _store_steps(r: dict[str, Any], bt: int, exact: bool) -> list[tuple[int, int]]:
    """(step, block_end) at which each complete block of ``r`` was first computed.

    ``progress`` is in step order; after a preemption it restarts lower, so
    the earliest entry reaching the block end wins. ``exact`` (checkpoint):
    only an entry ending exactly on the block end counts.
    """
    out = []
    for e in range(bt, r["computed"] + 1, bt):
        hit = (p[0] for p in r["progress"] if (p[1] == e if exact else p[1] >= e))
        step = next(hit, None)
        if step is not None:
            out.append((step, e))
    return out


def _active(spec: dict[str, Any]) -> list[tuple[str, Any, int]]:
    """Non-skipped groups; empty when nothing is resumable (all scratch)."""
    kinds = [(label, g, bt) for label, g, bt in group_kinds(spec) if g is not None]
    if not kinds or all(label == "scratch" for label, _, _ in kinds):
        return []
    return kinds


def _store_events(
    kinds: list[tuple[str, Any, int]], reqs: list[dict[str, Any]]
) -> list[tuple[int, int, str, int]]:
    """Store events ``(visible_after_step, group, key, request_index)`` in save
    order: by step, then request, group, block end (a connector saves a
    request's blocks front to back; LRU insertion order depends on it)."""
    events = []
    for i, r in enumerate(reqs):
        kt = r["key_tokens"]
        for gi, (label, _, bt) in enumerate(kinds):
            if label == "scratch":
                continue
            for step, e in _store_steps(r, bt, exact=label == "checkpoint"):
                events.append((step, i, gi, e, r["keys"][e // kt - 1]))
    events.sort()
    return [(step, gi, key, i) for step, i, gi, _, key in events]


def _schedule_order(reqs: list[dict[str, Any]]) -> list[int]:
    return sorted(range(len(reqs)), key=lambda i: (reqs[i]["first_step"] or 0, i))


def replay(spec: dict[str, Any], reqs: list[dict[str, Any]]) -> list[int]:
    """Oracle resumable tokens for each request (first episode), in input order."""
    kinds = _active(spec)
    if not kinds:
        return [0] * len(reqs)

    # Store events: (visible_after_step, group, key, _).
    events = _store_events(kinds, reqs)
    order = _schedule_order(reqs)
    stored: list[set[str]] = [set() for _ in kinds]
    ei = 0
    result = [0] * len(reqs)
    for i in order:
        r = reqs[i]
        first = r["first_step"] or 0
        while ei < len(events) and events[ei][0] < first:
            _, gi, key, _ = events[ei]
            stored[gi].add(key)
            ei += 1
        kt = r["key_tokens"]
        present = []
        for gi, (_, _, bt) in enumerate(kinds):
            n = min(r["prompt_len"], r["computed"]) // bt
            present.append([b for b in range(n) if r["keys"][(b + 1) * bt // kt - 1] in stored[gi]])
        result[i] = sem.resolve([g for _, g, _ in kinds], present, r["prompt_len"])
    return result


def summarize(
    spec: dict[str, Any],
    reqs: list[dict[str, Any]],
    segments: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Hit numbers and ``E_ext`` per segment (name -> set of ``rsi_pass`` labels).

    The oracle always replays the whole trace; segments only select which
    requests are scored. ``None`` scores everything as ``all``.
    """
    oracle = replay(spec, reqs)
    kinds = group_kinds(spec)
    block = max((bt for _, g, bt in kinds if g is not None), default=0)
    out: dict[str, Any] = {"kinds": [label for label, _, _ in kinds]}
    for name, labels in (segments or {"all": None}).items():
        part = [i for i, r in enumerate(reqs) if labels is None or r.get("pass") in labels]
        prompt = actual = local = best = ext = ext_star = excess = 0
        for i in part:
            r = reqs[i]
            a, lo, o = r["first_computed"] or 0, r["local_hit"] or 0, oracle[i]
            prompt += r["prompt_len"]
            actual += a
            local += lo
            best += o
            ext += max(0, min(a, o) - lo)
            ext_star += max(0, o - lo)
            excess += max(0, a - max(o, lo))
        out[name] = {
            "requests": len(part),
            "prompt_tokens": prompt,
            "hit_tokens": actual,
            "local_hit_tokens": local,
            "external_hit_tokens": actual - local,
            "external_hit_rate": (actual - local) / (prompt - local) if prompt > local else 0.0,
            "oracle_tokens": best,
            "oracle_hit_rate": best / prompt if prompt else 0.0,
            "hit_rate": actual / prompt if prompt else 0.0,
            "ext_tokens": ext,
            "ext_oracle_tokens": ext_star,
            "efficiency": ext / ext_star if ext_star else None,
            "excess_tokens": excess,
            "excess_limit": block * len(part),
        }
    return out


# ------------------------------------------------------------ G5 capacity


def tp_copies(spec: dict[str, Any]) -> int:
    """Distinct KV copies over the TP ranks: ``tp / kv_tp_replicas``."""
    tp = int(spec.get("parallel", {}).get("tp", 1) or 1)
    replicas = int(spec["kv_cache_config"].get("kv_tp_replicas", 1) or 1)
    if tp % replicas:
        raise ValueError(f"kv_tp_replicas {replicas} does not divide tp {tp}")
    return tp // replicas


def block_bytes(spec: dict[str, Any]) -> list[int]:
    """Bytes of one (group, key) block per non-skipped group, all ranks.

    Unpadded page bytes (``real_page_size_bytes``, as digested) x layers in
    the group x distinct copies over the TP ranks: ``tp / kv_tp_replicas``
    (vLLM's ``KVCacheConfig.kv_tp_replicas``: ranks holding the same bytes,
    e.g. MLA's single latent head at TP2 = 2). A store keeps one copy of
    replicated bytes: PegaFlow's save bytes on a TP2 MLA run equal
    ``insertions x page x layers`` (DeepSeek-V2-Lite, 2026-10-23), and vLLM's
    offloading spec stores one copy under ``replicated_layout``.
    ``UniformTypeKVCacheSpecs`` already sums its per-layer pages, so it is
    not multiplied by the layer count.
    """
    copies = tp_copies(spec)
    out = []
    for g in spec["kv_cache_config"]["groups"]:
        if not g.get("enable_kv_transfer", True) or g.get("is_eagle_group"):
            continue
        s = g["spec"]
        page = s.get("real_page_size_bytes")
        if not isinstance(page, int):
            page = s["page_size_bytes"]
        layers = 1 if "UniformTypeKVCacheSpecs" in s["mro"] else g["num_layers"]
        out.append(page * layers * copies)
    return out


def unique_bytes(
    spec: dict[str, Any], reqs: list[dict[str, Any]], passes: set[str] | None = None
) -> int:
    """``U``: bytes of the distinct (group, key) blocks the unlimited oracle
    stores for the requests tagged with one of ``passes`` (None = all)."""
    kinds = _active(spec)
    if not kinds:
        return 0
    sizes = block_bytes(spec)
    part = [r for r in reqs if passes is None or r.get("pass") in passes]
    items = {(gi, key) for _, gi, key, _ in _store_events(kinds, part)}
    return sum(sizes[gi] for gi, _ in items)


_NEVER = 1 << 62  # next-use position of an item no later request needs


def _chain(r: dict[str, Any], bt: int, b: int) -> str:
    return r["keys"][(b + 1) * bt // r["key_tokens"] - 1]


def _needed_keys(
    kinds: list[tuple[str, Any, int]], r: dict[str, Any], p: int
) -> list[tuple[int, str]]:
    """(group, key) blocks request ``r`` loads to resume at ``p`` tokens."""
    out = []
    if p <= 0:
        return out
    for gi, (label, g, bt) in enumerate(kinds):
        if label == "scratch":
            continue
        lo, hi = sem.needed(g, p)
        out += [(gi, _chain(r, bt, b)) for b in range(lo, hi)]
    return out


def _external_keys(
    kinds: list[tuple[str, Any, int]], r: dict[str, Any], p: int, local: int
) -> list[tuple[int, str]]:
    """Blocks needed at ``p`` that vLLM's local hit (``needed`` at ``local``)
    does not already hold, without duplicates."""
    have = set(_needed_keys(kinds, r, local))
    return [x for x in dict.fromkeys(_needed_keys(kinds, r, p)) if x not in have]


def _resolve(
    kinds: list[tuple[str, Any, int]],
    r: dict[str, Any],
    resident: dict[tuple[int, str], int],
    local: int = 0,
) -> int:
    """``resolve`` over resident blocks plus what vLLM's local hit at
    ``local`` tokens holds (every group's ``needed`` range at ``local``)."""
    present = []
    for gi, (label, g, bt) in enumerate(kinds):
        n = min(r["prompt_len"], r["computed"]) // bt
        lo, hi = sem.needed(g, local) if local > 0 and label != "scratch" else (0, 0)
        present.append([b for b in range(n) if lo <= b < hi or (gi, _chain(r, bt, b)) in resident])
    return sem.resolve([g for _, g, _ in kinds], present, r["prompt_len"])


def simulate(
    spec: dict[str, Any],
    reqs: list[dict[str, Any]],
    capacity: int,
    policy: str,
    unlimited: list[int] | None = None,
    local: list[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Hit tokens per request (input order) for a store of ``capacity`` bytes.

    ``local[i]``: tokens of request ``i`` vLLM holds itself (default: its
    recorded ``local_hit``); G5 passes the baseline arm's local hit so an arm
    cannot raise its bar by lowering its own.

    Same store events and visibility as :func:`replay`, same ``resolve``
    scoring; only residency differs.

    - ``policy="belady"`` (``H*_C``): an item's next use is the next request
      (by first schedule) whose unlimited-oracle resolved prefix needs it
      (``semantics.needed`` at that request's unlimited ``P``). An arriving
      item with no next use is not stored; otherwise items with the furthest
      next use are evicted first, and the arriving item is not stored if
      it is itself the furthest. Clairvoyant and Kind-aware through the use
      sets, but greedy under variable item sizes and blind to coherence of
      partial prefixes: a strong reference, not a proven optimum.
    - ``policy="lru"`` (``H_lru``): group-oblivious LRU, every item stored, a
      re-store of a resident item does not refresh it (as the server's
      ``insert`` on an existing key), a hit refreshes the blocks loaded.
    - ``policy="freeze"``: store until full, never evict (a degenerate policy
      the gate must reject; reported so the bar's separation is visible).

    Diagnostics per group: stored / not admitted / evicted counts and bytes,
    and at the end of each pass (schedule order, ``rsi_pass`` tag) the
    resident and *stranded* bytes: resident blocks that are not on the
    resolved prefix of any request in the trace, resolving over the blocks
    resident then plus that request's recorded local hit (e.g. a checkpoint
    whose full-attention prefix was evicted).
    """
    if policy not in ("belady", "lru", "freeze"):
        raise ValueError(f"unknown policy {policy!r}")
    kinds = _active(spec)
    if not kinds:
        return [0] * len(reqs), {"groups": [], "passes": {}}
    sizes = block_bytes(spec)
    events = _store_events(kinds, reqs)
    order = _schedule_order(reqs)
    if unlimited is None:
        unlimited = replay(spec, reqs)
    if local is None:
        local = [r["local_hit"] or 0 for r in reqs]

    # Use positions per item (positions in schedule order, increasing).
    uses: dict[tuple[int, str], list[int]] = {}
    for t, i in enumerate(order):
        r = reqs[i]
        for item in _external_keys(kinds, r, unlimited[i], local[i]):
            uses.setdefault(item, []).append(t)

    def next_use(item: tuple[int, str], t: int) -> int:
        u = uses.get(item)
        if not u:
            return _NEVER
        k = bisect.bisect_left(u, t)
        return u[k] if k < len(u) else _NEVER

    # Stranded-diagnostic probes: one per distinct (prompt prefix, local hit).
    probes = list(
        {(r["prompt_len"], local[i],
          tuple(r["keys"][: r["prompt_len"] // r["key_tokens"]])): (r, local[i])
         for i, r in enumerate(reqs)}.values()
    )  # fmt: skip

    groups = [
        {"kind": label, "block_bytes": sizes[gi], "stored": 0, "not_admitted": 0, "evicted": 0}
        for gi, (label, _, _) in enumerate(kinds)
    ]
    passes: dict[str, Any] = {}
    # resident: item -> next use (belady) or insertion counter (lru, ordered).
    resident: dict[tuple[int, str], int] | OrderedDict[tuple[int, str], int]
    resident = {} if policy == "belady" else OrderedDict()
    heap: list[tuple[int, int, str]] = []  # (-next_use, group, key), lazy
    used = 0

    def evict(item: tuple[int, str]) -> None:
        nonlocal used
        del resident[item]
        used -= sizes[item[0]]
        groups[item[0]]["evicted"] += 1

    def admit(item: tuple[int, str], t: int) -> None:
        nonlocal used
        gi = item[0]
        if item in resident:
            return
        size = sizes[gi]
        if size > capacity:
            groups[gi]["not_admitted"] += 1
            return
        if policy == "freeze":
            if used + size > capacity:
                groups[gi]["not_admitted"] += 1
                return
            resident[item] = 0
        elif policy == "lru":
            while used + size > capacity:
                evict(next(iter(resident)))
            resident[item] = 0
        else:
            nu = next_use(item, t)
            if nu == _NEVER:
                groups[gi]["not_admitted"] += 1
                return
            victims: list[tuple[int, int, str]] = []
            chosen: set[tuple[int, str]] = set()
            freed = capacity - used
            while freed < size and heap:
                neg, vg, vk = heap[0]
                if resident.get((vg, vk)) != -neg or (vg, vk) in chosen:
                    heapq.heappop(heap)  # stale or duplicate entry
                    continue
                if -neg <= nu:
                    break  # every remaining resident item is needed sooner
                heapq.heappop(heap)
                victims.append((neg, vg, vk))
                chosen.add((vg, vk))
                freed += sizes[vg]
            if freed < size:
                for v in victims:
                    heapq.heappush(heap, v)
                groups[gi]["not_admitted"] += 1
                return
            for _, vg, vk in victims:
                evict((vg, vk))
            resident[item] = nu
            heapq.heappush(heap, (-nu, gi, item[1]))
        used += size
        groups[gi]["stored"] += 1

    def snapshot() -> dict[str, Any]:
        live: set[tuple[int, str]] = set()
        for r, lo in probes:
            live.update(_needed_keys(kinds, r, _resolve(kinds, r, resident, lo)))
        res = [0] * len(kinds)
        stranded = [0] * len(kinds)
        for item in resident:
            res[item[0]] += sizes[item[0]]
            if item not in live:
                stranded[item[0]] += sizes[item[0]]
        return {"resident_bytes": res, "stranded_bytes": stranded}

    ei = 0
    hits = [0] * len(reqs)
    current = None
    for t, i in enumerate(order):
        r = reqs[i]
        first = r["first_step"] or 0
        while ei < len(events) and events[ei][0] < first:
            _, gi, key, _ = events[ei]
            admit((gi, key), t)
            ei += 1
        label = r.get("pass") or ""
        if current is not None and label != current and current not in passes:
            passes[current] = snapshot()
        current = label
        lo = local[i]
        # Local blocks count as present; never credit more than an unlimited store.
        hits[i] = min(_resolve(kinds, r, resident, lo), max(unlimited[i], lo))
        if policy == "lru":
            for item in _external_keys(kinds, r, hits[i], lo):
                if item in resident:
                    resident.move_to_end(item)
        elif policy == "belady":
            for item in _external_keys(kinds, r, unlimited[i], lo):
                if item in resident:
                    nu = next_use(item, t + 1)
                    resident[item] = nu
                    heapq.heappush(heap, (-nu, item[0], item[1]))
    while ei < len(events):
        _, gi, key, _ = events[ei]
        admit((gi, key), len(order))
        ei += 1
    if current is not None and current not in passes:
        passes[current] = snapshot()
    for gi, g in enumerate(groups):
        for k in ("stored", "not_admitted", "evicted"):
            g[f"{k}_bytes"] = g[k] * sizes[gi]
    return hits, {"groups": groups, "passes": passes}


def external_load_bytes(
    spec: dict[str, Any], reqs: list[dict[str, Any]], passes: set[str] | None = None
) -> int:
    """Bytes of the blocks the requests in ``passes`` loaded through the
    connector: per request the blocks ``needed`` at its first-schedule
    computed tokens that its own local hit did not hold."""
    kinds = _active(spec)
    if not kinds:
        return 0
    sizes = block_bytes(spec)
    total = 0
    for r in reqs:
        if passes is not None and r.get("pass") not in passes:
            continue
        a, lo = r["first_computed"] or 0, r["local_hit"] or 0
        if a > lo:
            total += sum(sizes[gi] for gi, _ in _external_keys(kinds, r, a, lo))
    return total


def summarize_capacity(
    spec: dict[str, Any],
    reqs: list[dict[str, Any]],
    capacity: int,
    segments: dict[str, set[str]] | None = None,
    ref_local: list[int | None] | None = None,
) -> dict[str, Any]:
    """G5 numbers per segment for a store of ``capacity`` bytes.

    Per request: ``a`` = actual first-schedule computed tokens, ``ref`` =
    reference local hit (``ref_local[i]``: the baseline arm's local hit for
    the same workload position; the arm's own local hit where ``None`` or
    when ``ref_local`` is not given, counted as ``unmatched``), ``o`` =
    unlimited oracle, ``oc`` / ``lr`` / ``fz`` = Belady / LRU / freeze at
    capacity (all simulated with ``ref`` as the local hit).

    - ``ext = max(0, min(a, o) - max(ref, own))`` (capped by the unlimited
      oracle, so a connector cannot score above what an unlimited store could
      serve; a lower own local hit only raises what it has to load, a higher
      one is vLLM's cache, not the store's, and earns nothing)
    - ``ext*_C = max(0, oc - ref)``; ``efficiency_c`` (``E_C``) = Σext / Σext*_C
    - ``lru_vs_oracle_c`` / ``freeze_vs_oracle_c`` = Σmax(0, lr|fz - ref) / Σext*_C
    - ``efficiency_c_own_local``: the same ratio with the arm's own local hit
      for both ext and ext*_C (diagnostic only).
    - ``*_hit_rate`` = tokens / prompt tokens; ``excess`` as in G3 (unlimited).
    """
    unlimited = replay(spec, reqs)
    own = [r["local_hit"] or 0 for r in reqs]
    ref = list(own)
    matched = [ref_local is not None] * len(reqs)
    if ref_local is not None:
        for i, x in enumerate(ref_local):
            if x is None:
                matched[i] = False
            else:
                ref[i] = x
    belady, bdiag = simulate(spec, reqs, capacity, "belady", unlimited, ref)
    lru, ldiag = simulate(spec, reqs, capacity, "lru", unlimited, ref)
    freeze, _ = simulate(spec, reqs, capacity, "freeze", unlimited, ref)
    belady_own = (
        belady if ref == own else simulate(spec, reqs, capacity, "belady", unlimited, own)[0]
    )
    kinds = group_kinds(spec)
    block = max((bt for _, g, bt in kinds if g is not None), default=0)
    out: dict[str, Any] = {
        "capacity_bytes": capacity,
        "kinds": [label for label, _, _ in kinds],
        "diagnostics": {"oracle_c": bdiag, "lru": ldiag},
    }
    names = ("prompt", "actual", "local", "ref", "o", "oc", "lr", "fz", "ext", "ext_c",
             "ext_lru", "ext_fz", "ext_own", "ext_c_own", "excess", "unmatched")  # fmt: skip
    for name, labels in (segments or {"all": None}).items():
        part = [i for i, r in enumerate(reqs) if labels is None or r.get("pass") in labels]
        t = dict.fromkeys(names, 0)
        for i in part:
            r = reqs[i]
            a, lo, rl = r["first_computed"] or 0, own[i], ref[i]
            o = unlimited[i]
            t["prompt"] += r["prompt_len"]
            t["actual"] += a
            t["local"] += lo
            t["ref"] += rl
            t["o"] += o
            t["oc"] += belady[i]
            t["lr"] += lru[i]
            t["fz"] += freeze[i]
            t["ext"] += max(0, min(a, o) - max(rl, lo))
            t["ext_c"] += max(0, belady[i] - rl)
            t["ext_lru"] += max(0, lru[i] - rl)
            t["ext_fz"] += max(0, freeze[i] - rl)
            t["ext_own"] += max(0, min(a, o) - lo)
            t["ext_c_own"] += max(0, belady_own[i] - lo)
            t["excess"] += max(0, a - max(o, lo))
            t["unmatched"] += not matched[i]
        p, ec = t["prompt"], t["ext_c"]
        out[name] = {
            "requests": len(part),
            "unmatched_requests": t["unmatched"],
            "prompt_tokens": p,
            "hit_tokens": t["actual"],
            "local_hit_tokens": t["local"],
            "reference_local_tokens": t["ref"],
            "hit_rate": t["actual"] / p if p else 0.0,
            "oracle_tokens": t["o"],
            "oracle_hit_rate": t["o"] / p if p else 0.0,
            "oracle_c_tokens": t["oc"],
            "oracle_c_hit_rate": t["oc"] / p if p else 0.0,
            "lru_tokens": t["lr"],
            "lru_hit_rate": t["lr"] / p if p else 0.0,
            "freeze_hit_rate": t["fz"] / p if p else 0.0,
            "ext_tokens": t["ext"],
            "ext_oracle_c_tokens": ec,
            "ext_oracle_c_share": ec / p if p else 0.0,
            "ext_lru_tokens": t["ext_lru"],
            "efficiency_c": t["ext"] / ec if ec else None,
            "lru_vs_oracle_c": t["ext_lru"] / ec if ec else None,
            "freeze_vs_oracle_c": t["ext_fz"] / ec if ec else None,
            "efficiency_c_own_local": t["ext_own"] / t["ext_c_own"] if t["ext_c_own"] else None,
            "excess_tokens": t["excess"],
            "excess_limit": block * len(part),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--by-pass", action="store_true", help="split by the rsi_pass tag")
    ap.add_argument(
        "--capacity", type=int, default=0, help="G5: store bytes (0 = G3 unlimited only)"
    )
    args = ap.parse_args(argv)
    spec = json.loads((args.probe_dir / "spec.json").read_text())
    reqs = load_requests(args.probe_dir / "requests.jsonl")
    passes = {r.get("pass") or "" for r in reqs}
    segments = {p: {p} for p in sorted(passes)} if args.by_pass else None
    out = summarize(spec, reqs, segments)
    out["unique_bytes"] = unique_bytes(spec, reqs)
    if args.capacity:
        out["capacity"] = summarize_capacity(spec, reqs, args.capacity, segments)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
