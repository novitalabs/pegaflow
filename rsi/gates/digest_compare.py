"""Compare RsiProbe digest directories (G2 V3).

Each digested block (all layers of one group on one rank) is tagged from the
request's *last* schedule episode in ``requests.jsonl`` (the bytes held at
finish come from it):

- ``local``: ``covered <= local_hit`` (vLLM's own prefix cache);
- ``external``: ``local_hit < covered <= start`` (loaded by the connector);
- ``computed``: the request computed it.

Reference = the *computed* occurrences of the same (rank, group, prefix key)
in REF. A local or external block is

- ``matched`` when every layer equals some reference copy (any copy: a
  connector may legitimately have saved any of them);
- ``mismatched`` when it equals none and the reference is *exhaustive*: the
  runs are batch-invariant (all computed copies are byte-equal), or every
  (request, episode) of REF that computed the key was digested, so the copy
  the connector saved is among the references;
- ``uncertain`` otherwise: no invariance and some computation of the key in
  REF was not digested (an untagged request or a preempted episode);
  ``uncertain_preempted`` counts those whose only undigested computations
  are preempted episodes of digested requests (no untagged computation);
- ``unreferenced`` when REF never computed the key.

Computed blocks are compared against the other computed copies: ``exact``
when equal to one of them; under batch invariance every copy must be exact.
``max_rel_l1`` reports the closest per-layer float L1 for the inexact ones.
Without split invariance (``split_invariant=False``: the computed KV depends
on where the computation started, e.g. vLLM's MLA prefill attends to the
cached context separately and merges), a computed block is judged only
against copies computed with the same *split*: the start of the computing
episode and the chunk ends before the block (``progress`` values below its
end). Blocks with copies at other splits only are ``computed_unmatched_split``
(not judged); with REF != CAND a computed block REF never computed is
``computed_unreferenced``. ``computed_by_pass`` breaks the computed counts
down by the candidate's ``rsi_pass`` (the gate requires a minimum coverage per
pass on split-variant models). The split key misses a prefill chunk that ends
without crossing a ``key_tokens`` boundary; such a pair of computations gets
the same key with different MLA splits, which can only make an honest block
inexact (fail-closed), never a wrong one exact.

``REF`` and ``CAND`` may be the same directory (self-consistency of a
cold-then-warm run in one server); a computed block never references itself.

Usage: python -m rsi.gates.digest_compare REF_DIR CAND_DIR [--invariant]
       [--split-variant] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

Key = tuple[str, int, str]  # rank, group, prefix key
_LOADED = ("local", "external")


def _requests(probe_dir: Path) -> Iterator[dict[str, Any]]:
    for line in (probe_dir / "requests.jsonl").read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def computed_ranges(r: dict[str, Any]) -> list[tuple[int, int]]:
    """``(start, end]`` token ranges computed by each episode of a request."""
    eps = r["episodes"]
    out = []
    for k, (step, _, start) in enumerate(eps):
        if k + 1 == len(eps):
            end = r["computed"]
        else:
            nxt = eps[k + 1][0]
            end = max((p[1] for p in r["progress"] if step <= p[0] < nxt), default=start)
        out.append((start, end))
    return out


Split = tuple[int, tuple[int, ...]]  # episode start, chunk ends before the block


def records(
    probe_dir: Path,
) -> Iterator[tuple[Key, str, tuple[str, ...], list[float], str, Split]]:
    """(key, tag, per-layer exact digests, per-layer l1, req_id, split) per
    digested block; ``split`` = (start of the last episode, its ``progress``
    chunk ends below the block's end)."""
    last: dict[str, tuple[int, int, list[int]]] = {}
    for r in _requests(probe_dir):
        if r["episodes"]:
            step, local, start = r["episodes"][-1]
            ends = [p[1] for p in r["progress"] if p[0] >= step]
            last[r["req_id"]] = (local or 0, start, ends)
    layers: dict[tuple[str, int], list[str]] = {}
    for f in sorted(probe_dir.glob("digest-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["req_id"] not in last:
                raise ValueError(f"{f}: digested request {rec['req_id']} not in requests.jsonl")
            if layers.setdefault((rec["rank"], rec["group"]), rec["layers"]) != rec["layers"]:
                raise ValueError(f"{f}: layer list of group {rec['group']} changed")
            local, start, ends = last[rec["req_id"]]
            for blk in rec["blocks"]:
                c = blk["covered"]
                tag = "local" if c <= local else "external" if c <= start else "computed"
                key = (rec["rank"], rec["group"], blk["key"])
                split = (start, tuple(e for e in ends if e < c))
                yield key, tag, tuple(blk["exact"]), blk["l1"], rec["req_id"], split


class Reference:
    """Computed copies of one probe dir, and how often each key was computed."""

    def __init__(self, probe_dir: Path):
        # key -> per-layer digest tuple -> [occurrences, per-layer l1]
        self.copies: dict[Key, dict[tuple[str, ...], list[Any]]] = defaultdict(dict)
        # (key, split) -> per-layer digest tuple -> [occurrences, per-layer l1]
        self.split_copies: dict[tuple[Key, Split], dict[tuple[str, ...], list[Any]]] = defaultdict(
            dict
        )
        for key, tag, exact, l1, _, split in records(probe_dir):
            if tag == "computed":
                self.copies[key].setdefault(exact, [0, l1])[0] += 1
                self.split_copies[(key, split)].setdefault(exact, [0, l1])[0] += 1
        # prefix key -> number of (request, episode) that computed its block end
        self.computers: Counter[str] = Counter()
        # Of those, computations whose bytes were never digested: a preempted
        # (non-last) episode of a digested request, or an untagged request.
        self.preempted: Counter[str] = Counter()
        self.untagged: Counter[str] = Counter()
        for r in _requests(probe_dir):
            kt = r["key_tokens"]
            ranges = computed_ranges(r)
            for k, (start, end) in enumerate(ranges):
                for e in range((start // kt + 1) * kt, end + 1, kt):
                    key = r["keys"][e // kt - 1]
                    self.computers[key] += 1
                    if r.get("digest") is None:
                        self.untagged[key] += 1
                    elif k + 1 < len(ranges):
                        self.preempted[key] += 1

    def occurrences(self, key: Key) -> int:
        return sum(c[0] for c in self.copies.get(key, {}).values())


def _group() -> dict[str, Any]:
    g: dict[str, Any] = {}
    for t in _LOADED:
        for k in (
            "compared",
            "matched",
            "mismatched",
            "uncertain",
            "uncertain_preempted",
            "unreferenced",
        ):
            g[f"{t}_{k}"] = 0
    g |= {
        "computed_compared": 0,
        "computed_exact": 0,
        "computed_unmatched_split": 0,
        "computed_unreferenced": 0,
        "max_rel_l1": 0.0,
        "first_bad": None,
    }
    return g


def _rel_l1(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / max(abs(x), 1e-9) for x, y in zip(a, b, strict=True))


def _uncertain(g: dict[str, Any], tag: str, ref: Reference, key: Key) -> None:
    g[f"{tag}_uncertain"] += 1
    k = key[2]
    if not ref.untagged[k] and ref.occurrences(key) + ref.preempted[k] >= ref.computers[k]:
        g[f"{tag}_uncertain_preempted"] += 1


def compare(
    ref_dir: Path,
    cand_dir: Path,
    invariant: bool,
    split_invariant: bool = True,
    align: int | None = None,
) -> dict[str, Any]:
    """``align``: vLLM's hit alignment; with it, ``computed_by_pass`` also
    counts unjudged computed blocks of requests whose prompt length is not a
    multiple of it (``unjudged_nonaligned``; a missing length counts)."""
    ref = Reference(ref_dir)
    own = 1 if ref_dir.resolve() == cand_dir.resolve() else 0
    groups: dict[int, dict[str, Any]] = defaultdict(_group)
    reqs = list(_requests(cand_dir))
    pass_of = {r["req_id"]: r.get("pass") for r in reqs}
    nonaligned = {
        r["req_id"] for r in reqs if align and (not r.get("prompt_len") or r["prompt_len"] % align)
    }
    by_pass: dict[str, Counter[str]] = defaultdict(Counter)
    for key, tag, exact, l1, rid, split in records(cand_dir):
        g = groups[key[1]]
        copies = ref.copies.get(key, {})
        if tag == "computed":
            bp = by_pass[str(pass_of.get(rid))]
            if ref.occurrences(key) - own <= 0:
                if not own:
                    g["computed_unreferenced"] += 1
                    bp["unreferenced"] += 1
                    bp["unjudged_nonaligned"] += rid in nonaligned
                continue
            pool = copies
            if not split_invariant:
                # Only a copy computed with the same split is a reference.
                pool = ref.split_copies.get((key, split), {})
                if sum(c[0] for c in pool.values()) - own <= 0:
                    g["computed_unmatched_split"] += 1
                    bp["unmatched_split"] += 1
                    bp["unjudged_nonaligned"] += rid in nonaligned
                    continue
            g["computed_compared"] += 1
            bp["compared"] += 1
            if pool.get(exact, [0])[0] - own > 0:
                g["computed_exact"] += 1
                bp["exact"] += 1
            else:
                rel = min(_rel_l1(c[1], l1) for d, c in pool.items() if d != exact)
                g["max_rel_l1"] = max(g["max_rel_l1"], rel)
            continue
        if not copies:
            if ref.computers[key[2]]:
                _uncertain(g, tag, ref, key)
            else:
                g[f"{tag}_unreferenced"] += 1
            continue
        g[f"{tag}_compared"] += 1
        if exact in copies:
            g[f"{tag}_matched"] += 1
        elif invariant or ref.occurrences(key) >= ref.computers[key[2]]:
            g[f"{tag}_mismatched"] += 1
            if g["first_bad"] is None:
                best = min(copies, key=lambda d: sum(x != y for x, y in zip(d, exact, strict=True)))
                bad = [i for i, (x, y) in enumerate(zip(best, exact, strict=True)) if x != y]
                g["first_bad"] = {
                    "tag": tag,
                    "rank": key[0],
                    "key": key[2],
                    "req_id": rid,
                    "bad_layers": bad,
                }
        else:
            _uncertain(g, tag, ref, key)
    counters = [k for k, v in _group().items() if isinstance(v, int)]
    total: dict[str, Any] = {k: sum(g[k] for g in groups.values()) for k in counters}
    total["max_rel_l1"] = max((g["max_rel_l1"] for g in groups.values()), default=0.0)
    total["invariant"] = invariant
    total["split_invariant"] = split_invariant
    # With ``align`` every pass reports ``unjudged_nonaligned`` (0 included).
    extra = {"unjudged_nonaligned": 0} if align else {}
    total["computed_by_pass"] = {p: extra | dict(c) for p, c in sorted(by_pass.items())}
    total["groups"] = {str(k): v for k, v in sorted(groups.items())}
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--invariant", action="store_true", help="runs were batch-invariant")
    ap.add_argument(
        "--split-variant",
        action="store_true",
        help="computed KV depends on the split point: judge computed blocks per split",
    )
    ap.add_argument("--json", type=Path, help="write the report here")
    args = ap.parse_args(argv)
    report = compare(args.ref, args.cand, args.invariant, not args.split_variant)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 1 if report["local_mismatched"] or report["external_mismatched"] else 0


if __name__ == "__main__":
    sys.exit(main())
