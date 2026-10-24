"""Shared resume/save/sharding rules (Rust `pegaflow-semantics`).

Re-exports the native `semantics` submodule so callers can write
`from pegaflow.semantics import Group, resolve`.
"""

from collections.abc import Iterable

from .pegaflow import semantics as _native

Group = _native.Group
Shard = _native.Shard
Region = _native.Region
Fragment = _native.Fragment
SealPlan = _native.SealPlan
needed = _native.needed
saveable = _native.saveable
key = _native.key
seal_plan = _native.seal_plan


def resolve(groups: list[Group], present: list[Iterable[int]], prompt_len: int) -> int:
    """Longest resumable prefix; `present[i]` may be any iterable (set, list)."""
    return _native.resolve(groups, [list(p) for p in present], prompt_len)


__all__ = [
    "Fragment",
    "Group",
    "Region",
    "SealPlan",
    "Shard",
    "key",
    "needed",
    "resolve",
    "saveable",
    "seal_plan",
]
