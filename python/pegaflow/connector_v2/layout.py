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
    tp_size: int = 1
    # CPU socket next to each TP rank's GPU, given when the policy keeps a
    # copy per socket; empty: one copy per slice.
    sockets: tuple[int, ...] = ()


@dataclass(frozen=True)
class NsSpec:
    """One cache group stored as one RPC V2 namespace."""

    gid: int  # vLLM kv_cache_group index
    group: sem.Group
    hash_scale: int  # group block_size // vLLM hash_block_size
    layers: tuple[str, ...]
    namespace: bytes  # 32-byte digest
    page_bytes: int  # one rank's bytes of one layer of one block
    slices: int  # slices one layer is split into across TP
    savers: tuple[int, ...]  # ranks that save this namespace

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

    @property
    def groups(self) -> list[sem.Group]:
        return [ns.group for ns in self.namespaces]


def _savers(slices: int, sockets: Sequence[int]) -> tuple[int, ...]:
    """Per slice, the first rank holding it on each CPU socket, so a load
    reads a copy on its own socket. Rank r holds slice r // replicas."""
    replicas = len(sockets) // slices
    out: dict[tuple[int, int], int] = {}
    for r, socket in enumerate(sockets):
        out.setdefault((r // replicas, socket), r)
    return tuple(sorted(out.values()))


def _digest(h: bytes) -> bytes:
    return h if len(h) == 32 else hashlib.sha256(h).digest()


def _layer_specs(spec: Any, layer_names: Sequence[str]) -> list[Any]:
    per_layer = getattr(spec, "kv_cache_specs", None)  # UniformTypeKVCacheSpecs
    if per_layer is None:
        return [spec] * len(layer_names)
    return [per_layer[name] for name in layer_names]


def _kind(spec: Any, gid: int) -> sem.Group:
    name = type(spec).__name__
    bt = int(spec.block_size)
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


def _slices(spec: Any, tp_size: int, gid: int) -> int:
    """Distinct shards a layer splits into across TP (1: every rank holds it all)."""
    name = type(spec).__name__
    if name == "MLAAttentionSpec":
        return 1
    if name == "MambaSpec":
        return 1 if getattr(spec, "tp_replicated", False) else tp_size
    shards = min(tp_size, getattr(spec, "max_tp_shards", None) or tp_size)
    if tp_size % shards:
        raise Unsupported(f"group {gid}: tp_size {tp_size} is not a multiple of {shards} shards")
    return shards


def build_layout(kv_cache_config: Any, par: Parallel, identity: dict[str, Any]) -> Layout:
    """Raises ``Unsupported`` for layouts connector v2 cannot store."""
    hash_bs = getattr(kv_cache_config, "hash_block_size", None)
    gids = getattr(kv_cache_config, "prefix_cacheable_group_ids", None)
    if gids is None:
        gids = range(len(kv_cache_config.kv_cache_groups))
    out: list[NsSpec] = []
    for gid in gids:
        kv_group = kv_cache_config.kv_cache_groups[gid]
        layers = tuple(kv_group.layer_names)
        specs = _layer_specs(kv_group.kv_cache_spec, layers)
        if not specs:
            continue
        groups = {_kind(s, gid) for s in specs}
        pages = {int(s.page_size_bytes) for s in specs}
        slices = {_slices(s, par.tp_size, gid) for s in specs}
        if len(groups) != 1 or len(pages) != 1 or len(slices) != 1:
            raise Unsupported(f"group {gid}: layers differ in kind, page size or TP split")
        (group,), (page,), (n_slices,) = groups, pages, slices
        bt = group.block_tokens
        hbs = hash_bs or bt
        if bt % hbs:
            raise Unsupported(f"group {gid}: block_size {bt} is not a multiple of hash {hbs}")
        savers = _savers(n_slices, par.sockets or (0,) * par.tp_size)
        doc = {
            **identity,
            "gid": gid,
            "spec": repr(kv_group.kv_cache_spec),
            "kind": [group.kind, bt, group.window_tokens],
            "slices": n_slices,
            "kv_cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
        }
        blob = json.dumps(doc, sort_keys=True, default=str).encode()
        out.append(
            NsSpec(
                gid=gid,
                group=group,
                hash_scale=bt // hbs,
                layers=layers,
                namespace=hashlib.sha256(blob).digest(),
                page_bytes=page,
                slices=n_slices,
                savers=savers,
            )
        )
    out.sort(key=lambda ns: ns.group.kind != sem.FULL)
    if not out or out[0].group.kind != sem.FULL:
        raise Unsupported("connector v2 needs a full-attention group to anchor prefix queries")
    return Layout(tuple(out))
