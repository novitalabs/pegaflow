"""vLLM ``KVCacheConfig`` -> RPC V2 namespaces, one per transferable cache group.

The scheduler and every worker build the same ``Layout`` from the same config,
so they agree on namespaces, kinds and block sizes without talking.

Duck-typed over vLLM objects (spec classes are matched by name) so the default
unit gate builds layouts from stubs without importing vLLM.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import semantics as sem


class Unsupported(RuntimeError):
    """The model's KV cache layout is outside what connector v2 handles."""


@dataclass(frozen=True)
class Parallel:
    """vLLM's TP x PCP grid of one instance, and DCP over it.

    Rank ``(t, p)`` is world rank ``p * tp_size + t`` (vLLM puts TP
    innermost). PCP replicates the KV cache; DCP shards it by token over
    groups that span PCP first, then TP.
    """

    tp_size: int = 1
    pcp_size: int = 1
    dcp_size: int = 1
    # CPU socket next to each world rank's GPU, given when the policy keeps a
    # copy per socket; empty: one copy per slice.
    sockets: tuple[int, ...] = ()

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pcp_size

    def world_rank(self, tp_rank: int, pcp_rank: int = 0) -> int:
        return pcp_rank * self.tp_size + tp_rank

    def dcp_rank(self, world_rank: int) -> int:
        """vLLM's DCP rank of a world rank: its DCP groups are
        ``arange(world).view(pcp, tp).T.reshape(-1, dcp)``."""
        t, p = world_rank % self.tp_size, world_rank // self.tp_size
        return (t * self.pcp_size + p) % self.dcp_size


@dataclass(frozen=True)
class NsSpec:
    """One cache group stored as one RPC V2 namespace."""

    gid: int  # vLLM kv_cache_group index
    group: sem.Group
    hash_scale: int  # group block_size // vLLM hash_block_size
    layers: tuple[str, ...]
    namespace: bytes  # 32-byte digest
    page_bytes: int  # one rank's bytes of one layer of one block
    slices: int  # distinct parts one layer is split into across ranks
    savers: tuple[int, ...]  # server ranks that save this namespace

    def keys(self, block_hashes: Sequence[bytes]) -> list[bytes]:
        """32-byte object prefixes of this group's blocks."""
        k = self.hash_scale
        return [_digest(h) for h in block_hashes[k - 1 :: k]]

    def meta(self, world_size: int, rank: int, device_id: int) -> dict[str, Any]:
        """``NamespaceMeta`` for ``register_context_v2``."""
        bt = self.group.block_tokens
        # The server only uses local_hidden x block_size (one rank's slice);
        # a padded page that does not split per token is sent as one "token".
        block_size = bt if self.page_bytes % bt == 0 else 1
        return {
            "namespace": self.namespace,
            "world_size": world_size,
            "device_id": device_id,
            "rank": rank,
            "layer_names": list(self.layers),
            "bytes_per_block": self.page_bytes * self.slices,
            "block_size": block_size,
            "local_hidden": self.page_bytes // block_size,
            "saver_rank": list(self.savers),
        }


@dataclass(frozen=True)
class Layout:
    namespaces: tuple[NsSpec, ...]  # full groups first
    # Server rank of each world rank: ranks holding one slice are
    # consecutive, as the server's ``slice = rank * S / world_size`` needs.
    ranks: tuple[int, ...] = (0,)

    @property
    def groups(self) -> list[sem.Group]:
        return [ns.group for ns in self.namespaces]

    @property
    def world_size(self) -> int:
        return len(self.ranks)


def _content(spec: Any, par: Parallel, gid: int) -> tuple[int, ...]:
    """Per world rank, which part of a layer it holds: its head shard over TP
    and, for a DCP-sharded spec, its DCP rank. Ranks with equal values hold
    the same bytes."""
    name = type(spec).__name__
    tp = par.tp_size
    if name == "MLAAttentionSpec":
        heads = 1
    elif name == "MambaSpec":
        heads = 1 if getattr(spec, "tp_replicated", False) else tp
    else:
        heads = min(tp, getattr(spec, "max_tp_shards", None) or tp)
    if tp % heads:
        raise Unsupported(f"group {gid}: tp_size {tp} is not a multiple of {heads} shards")
    # vLLM's default: attention specs are DCP-sharded, Mamba states are not.
    dcp = par.dcp_size if getattr(spec, "dcp_sharded", name != "MambaSpec") else 1
    per_head = tp // heads
    return tuple(
        (r % tp) // per_head * dcp + (par.dcp_rank(r) if dcp > 1 else 0)
        for r in range(par.world_size)
    )


def _ranks(contents: Sequence[tuple[int, ...]], world: int) -> tuple[int, ...]:
    """Server rank of each world rank. Sorting by content, coarsest namespace
    first, makes every namespace's holders of one part consecutive whenever
    their splits nest; otherwise raises ``Unsupported``."""
    order = sorted(contents, key=lambda c: len(set(c)))
    by_label = sorted(range(world), key=lambda r: (tuple(c[r] for c in order), r))
    ranks = [0] * world
    for label, r in enumerate(by_label):
        ranks[r] = label
    for c in contents:
        n = len(set(c))
        part_of: dict[int, int] = {}
        for r in range(world):
            # The server's slice of this label must name one part only.
            if part_of.setdefault(ranks[r] * n // world, c[r]) != c[r]:
                raise Unsupported(f"no rank order keeps every namespace's slices together: {c}")
    return tuple(ranks)


def _savers(
    content: Sequence[int], ranks: Sequence[int], sockets: Sequence[int]
) -> tuple[int, ...]:
    """Per part, the first server rank holding it on each CPU socket, so a
    load reads a copy on its own socket."""
    out: dict[tuple[int, int], int] = {}
    for r in sorted(range(len(ranks)), key=lambda r: ranks[r]):
        out.setdefault((content[r], sockets[r]), ranks[r])
    return tuple(sorted(out.values()))


def _digest(h: bytes) -> bytes:
    return h if len(h) == 32 else hashlib.sha256(h).digest()


def _layer_specs(spec: Any, layer_names: Sequence[str]) -> list[Any]:
    per_layer = getattr(spec, "kv_cache_specs", None)  # UniformTypeKVCacheSpecs
    if per_layer is None:
        return [spec] * len(layer_names)
    return [per_layer[name] for name in layer_names]


def _kind(spec: Any, gid: int, dcp_size: int) -> sem.Group:
    name = type(spec).__name__
    # One vLLM block spans block_size tokens per DCP rank (resolve_dcp_kv_block_size).
    sharded = getattr(spec, "dcp_sharded", name != "MambaSpec")
    bt = int(spec.block_size) * (dcp_size if sharded else 1)
    if name in ("FullAttentionSpec", "MLAAttentionSpec"):
        if getattr(spec, "sliding_window", None) or getattr(spec, "attention_chunk_size", None):
            raise Unsupported(f"group {gid}: {name} with a window or chunk is not supported")
        if getattr(spec, "non_causal", False):
            raise Unsupported(f"group {gid}: non-causal attention is not prefix-reusable")
        return sem.Group(sem.FULL, bt)
    if name == "SlidingWindowSpec":
        extra = int(getattr(spec, "extra_retained_tokens", 0) or 0)
        return sem.Group(sem.WINDOW, bt, int(spec.sliding_window) - 1 + extra)
    if name == "MambaSpec":
        mode = getattr(spec, "mamba_cache_mode", None)
        if mode != "align":
            raise Unsupported(f"group {gid}: MambaSpec mamba_cache_mode={mode!r}, need 'align'")
        if getattr(spec, "num_speculative_blocks", 0):
            raise Unsupported(f"group {gid}: MambaSpec with speculative blocks")
        return sem.Group(sem.CHECKPOINT, bt)
    raise Unsupported(f"group {gid}: no connector v2 support for {name}")


def build_layout(kv_cache_config: Any, par: Parallel, identity: dict[str, Any]) -> Layout:
    """Raises ``Unsupported`` for layouts connector v2 cannot store."""
    hash_bs = getattr(kv_cache_config, "hash_block_size", None)
    gids = getattr(kv_cache_config, "prefix_cacheable_group_ids", None)
    if gids is None:
        gids = range(len(kv_cache_config.kv_cache_groups))
    found = []
    for gid in gids:
        kv_group = kv_cache_config.kv_cache_groups[gid]
        layers = tuple(kv_group.layer_names)
        specs = _layer_specs(kv_group.kv_cache_spec, layers)
        if not specs:
            continue
        groups = {_kind(s, gid, par.dcp_size) for s in specs}
        pages = {int(s.page_size_bytes) for s in specs}
        contents = {_content(s, par, gid) for s in specs}
        if len(groups) != 1 or len(pages) != 1 or len(contents) != 1:
            raise Unsupported(f"group {gid}: layers differ in kind, page size or rank split")
        (group,), (page,), (content,) = groups, pages, contents
        bt = group.block_tokens
        hbs = hash_bs or bt
        if bt % hbs:
            raise Unsupported(f"group {gid}: block_size {bt} is not a multiple of hash {hbs}")
        found.append((gid, kv_group, layers, group, page, content, bt // hbs))
    ranks = _ranks([f[5] for f in found], par.world_size)
    sockets = par.sockets or (0,) * par.world_size
    out: list[NsSpec] = []
    for gid, kv_group, layers, group, page, content, hash_scale in found:
        n_slices = len(set(content))
        doc = {
            **identity,
            "gid": gid,
            "spec": repr(kv_group.kv_cache_spec),
            "kind": [group.kind, group.block_tokens, group.window_tokens],
            "slices": n_slices,
            "kv_cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
        }
        blob = json.dumps(doc, sort_keys=True, default=str).encode()
        out.append(
            NsSpec(
                gid=gid,
                group=group,
                hash_scale=hash_scale,
                layers=layers,
                namespace=hashlib.sha256(blob).digest(),
                page_bytes=page,
                slices=n_slices,
                savers=_savers(content, ranks, sockets),
            )
        )
    out.sort(key=lambda ns: ns.group.kind != sem.FULL)
    if not out or out[0].group.kind != sem.FULL:
        raise Unsupported("connector v2 needs a full-attention group to anchor prefix queries")
    return Layout(tuple(out), ranks)
