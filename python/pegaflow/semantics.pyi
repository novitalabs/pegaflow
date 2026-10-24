"""Type stubs for pegaflow.semantics (Rust `pegaflow-semantics`)."""

from collections.abc import Iterable

class Group:
    """One cache group: dependency kind plus tokens per block."""

    kind: str  # "full" | "window" | "checkpoint" | "scratch"
    block_tokens: int
    window_tokens: int | None
    @staticmethod
    def full(block_tokens: int) -> Group: ...
    @staticmethod
    def window(block_tokens: int, tokens: int) -> Group:
        """`tokens` = trailing cached tokens attended (vLLM sliding_window - 1 exact)."""
    @staticmethod
    def checkpoint(block_tokens: int) -> Group: ...
    @staticmethod
    def scratch(block_tokens: int) -> Group: ...

class Shard:
    @staticmethod
    def replicated() -> Shard: ...
    @staticmethod
    def heads(index: int, count: int) -> Shard: ...
    @staticmethod
    def tokens(index: int, count: int, interleave: int) -> Shard: ...

class Region:
    layer: str
    group: int
    segments: list[int]
    shard: Shard
    def __init__(self, layer: str, group: int, segments: list[int], shard: Shard) -> None: ...

class Fragment:
    layer: str
    part: int
    segments: list[int]
    holders: list[int]
    writer: int
    bytes: int

class SealPlan:
    groups: list[int]
    def fragments(self, group: int) -> list[Fragment]: ...
    def slot_of(self, rank: int, layer: str) -> tuple[int, int] | None: ...
    def is_writer(self, rank: int, group: int, slot: int) -> bool: ...

def needed(group: Group, p: int) -> tuple[int, int]:
    """Half-open block range `group` needs to resume at `p` tokens ((0, 0) if none)."""

def resolve(groups: list[Group], present: list[Iterable[int]], prompt_len: int) -> int:
    """Longest token prefix every group can resume from (< prompt_len)."""

def saveable(group: Group, before: int, after: int) -> tuple[int, int]:
    """Half-open block range that became storable from `before` to `after` ((0, 0) if none)."""

def key(block_hash: bytes, group: int) -> bytes:
    """Store key of `block_hash` in cache group `group`."""

def seal_plan(ranks: list[tuple[int, list[Region]]]) -> SealPlan:
    """Merge per-rank declarations; raises ValueError on gaps/overlaps."""
