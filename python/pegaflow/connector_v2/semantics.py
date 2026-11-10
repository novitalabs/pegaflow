"""What resuming a request at token ``p`` needs from each KV cache group.

Pure functions over block indices, no vLLM, torch or server types. A group
of ``block_tokens``-token blocks has one of four kinds:

- ``full``: every past token is attended; resuming at ``p`` needs blocks
  ``[0, ceil(p / bt))``.
- ``window``: only the last ``window_tokens`` tokens are attended; it needs
  ``[(p - window_tokens) / bt, ceil(p / bt))``. For vLLM ``sliding_window = W``
  the exact value is ``W - 1`` (plus ``extra_retained_tokens``).
- ``checkpoint``: block ``k - 1`` holds the state after ``k * bt`` tokens
  (Mamba ``align``); it needs ``[p / bt - 1, p / bt)``.
- ``scratch``: nothing survives; it needs nothing and saves nothing.

Attention resumes at any ``p``: a partly used last block loads whole and vLLM
recomputes the tokens from ``p`` on over it, as v1 does when it recomputes the
last prompt token. A checkpoint only exists at a block boundary, so a layout
with one resumes at multiples of ``step_of``.

``resolve`` picks the largest ``p`` that every group can back, ``saveable``
lists the blocks a step completes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

FULL = "full"
WINDOW = "window"
CHECKPOINT = "checkpoint"
SCRATCH = "scratch"
KINDS = (FULL, WINDOW, CHECKPOINT, SCRATCH)


@dataclass(frozen=True)
class Group:
    kind: str
    block_tokens: int
    window_tokens: int = 0  # window only

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown group kind {self.kind!r}")
        if self.block_tokens <= 0:
            raise ValueError(f"block_tokens must be positive, got {self.block_tokens}")
        if self.kind == WINDOW and self.window_tokens <= 0:
            raise ValueError(f"window_tokens must be positive, got {self.window_tokens}")


def needed(g: Group, p: int) -> range:
    """Blocks of ``g`` a request resuming at ``p`` must hold."""
    bt = g.block_tokens
    if g.kind == FULL:
        return range(0, -(-p // bt))
    if g.kind == WINDOW:
        return range(max(p - g.window_tokens, 0) // bt, -(-p // bt))
    if g.kind == CHECKPOINT:
        n = p // bt
        return range(max(n - 1, 0), n)
    return range(0)


def step_of(groups: Sequence[Group]) -> int:
    """Resume granularity: lcm of the non-scratch block sizes (0 if none)."""
    sizes = [g.block_tokens for g in groups if g.kind != SCRATCH]
    return math.lcm(*sizes) if sizes else 0


def resumable_at(groups: Sequence[Group], p: int) -> bool:
    """Whether a request can resume at ``p`` (``p > 0``)."""
    if any(g.kind == CHECKPOINT for g in groups):
        return p % step_of(groups) == 0
    return True


def candidates(groups: Sequence[Group], top: int, prompt_len: int) -> list[int]:
    """Resume points ``<= top`` worth trying, largest first.

    vLLM must compute at least the last prompt token, so ``p <= prompt_len - 1``.
    Between two multiples of ``step_of`` an attention-only layout needs the
    blocks of the upper one, so only the cap itself is added.
    """
    step = step_of(groups)
    cap = min(top, prompt_len - 1)
    if step == 0 or cap <= 0:
        return []
    points = list(range(cap // step * step, 0, -step))
    if cap % step and resumable_at(groups, cap):
        points.insert(0, cap)
    return points


def resolve(
    groups: Sequence[Group], present: Sequence[frozenset[int] | set[int]], prompt_len: int
) -> int:
    """Largest legal ``p`` at which every group holds ``needed(g, p)``; 0 if none."""
    runs = [_Runs(blocks) for blocks in present]
    for p in candidates(groups, prompt_len, prompt_len):
        if all(r.covers(needed(g, p)) for g, r in zip(groups, runs, strict=True)):
            return p
    return 0


def saveable(g: Group, before: int, after: int) -> range:
    """Blocks of ``g`` a step taking the request from ``before`` to ``after``
    computed tokens completes."""
    if after <= before:
        return range(0)
    bt = g.block_tokens
    if g.kind in (FULL, WINDOW):
        return range(before // bt, after // bt)
    if g.kind == CHECKPOINT and after % bt == 0:
        return range(after // bt - 1, after // bt)
    return range(0)


class _Runs:
    """Contiguous-run lookup over a set of block indices: ``covers`` is O(1)."""

    def __init__(self, blocks: frozenset[int] | set[int]):
        # run_end[b] = first index >= b missing from the set, for b in the set.
        self._end: dict[int, int] = {}
        for b in sorted(blocks, reverse=True):
            self._end[b] = self._end.get(b + 1, b + 1)

    def covers(self, r: range) -> bool:
        if not r:
            return True
        return self._end.get(r.start, r.start) >= r.stop
