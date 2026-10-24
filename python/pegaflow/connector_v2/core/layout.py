"""Model layout from adapter declarations: semantic groups and the namespace.

Duck-typed over vLLM's ``KVCacheConfig`` so the default unit gate can build a
layout without importing vLLM; ``semantics`` is injectable so it also builds
without the native extension.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from ..adapters import TRANSFER_BACKENDS, LayerDecl, Parallel, Unsupported, adapter_for


def native_semantics() -> Any:
    """``pegaflow.semantics``; imported lazily so core imports without the extension."""
    from pegaflow import semantics

    return semantics


@dataclass(frozen=True)
class GroupInfo:
    gid: int  # vLLM kv_cache_group index
    sem: Any  # semantics.Group
    hash_scale: int  # group block_tokens // vLLM hash_block_size
    layers: tuple[str, ...]  # layer names in this config (this rank on workers)


@dataclass(frozen=True)
class Layout:
    # Transferable groups; the index is the server storage group id ("sid").
    groups: tuple[GroupInfo, ...]
    step: int  # resolve granularity: lcm of non-scratch block_tokens
    namespace: str
    transfer_backend: str = "direct"  # RegisterContext transfer_mode, from LayerDecl
    page_first: bool = False  # RegisterContext page_first, from LayerDecl

    def group_hashes(self, g: GroupInfo, block_hashes: list[bytes]) -> list[bytes]:
        """Hashes of the group's blocks: the last hash_block_size hash of each."""
        k = g.hash_scale
        return list(block_hashes[k - 1 :: k])


def layer_specs(spec: Any, layer_names: list[str]) -> dict[str, Any]:
    """Per-layer spec; UniformTypeKVCacheSpecs carries one spec per layer."""
    per_layer = getattr(spec, "kv_cache_specs", None)
    if per_layer is not None:
        return {name: per_layer[name] for name in layer_names}
    return dict.fromkeys(layer_names, spec)


def build_layout(
    kv_cache_config: Any,
    par: Parallel,
    identity: dict[str, Any],
    semantics: Any | None = None,
) -> Layout:
    """Translate every transferable group through its adapter.

    Raises ``Unsupported`` for spec classes without an adapter, for layouts the
    core cannot handle yet, and for declarations ``seal_plan`` rejects (the
    design's V0 check), so both roles fail at connector init with the same
    message.
    """
    sem_mod = semantics if semantics is not None else native_semantics()
    hash_bs = kv_cache_config.hash_block_size
    groups: list[GroupInfo] = []
    group_specs: dict[int, dict[str, Any]] = {}
    fingerprint: list[dict[str, Any]] = []
    for gid in kv_cache_config.transfer_group_ids:
        kv_group = kv_cache_config.kv_cache_groups[gid]
        specs = layer_specs(kv_group.kv_cache_spec, list(kv_group.layer_names))
        group_spec = next(iter(specs.values()), kv_group.kv_cache_spec)
        adapter = adapter_for(group_spec, gid)
        sem = adapter.group(group_spec)
        if sem.kind == "checkpoint":
            # vLLM 0.32 hands Mamba "align" boundary states off with exact block
            # ids (KVConnectorBlockState.boundary_state_offloads); positional
            # save/load would store wrong state under a correct key (P5 core task).
            raise Unsupported(
                f"group {gid} ({type(group_spec).__name__}): checkpoint groups need core "
                f"support for vLLM boundary_state_offloads"
            )
        bt = sem.block_tokens
        if bt != group_spec.block_size:
            raise Unsupported(
                f"adapter for {type(group_spec).__name__} (group {gid}) declares "
                f"block_tokens={bt}, vLLM block_size={group_spec.block_size}"
            )
        hbs = hash_bs or bt
        if bt % hbs:
            raise Unsupported(f"group {gid}: block_size {bt} not a multiple of hash {hbs}")
        groups.append(GroupInfo(gid, sem, bt // hbs, tuple(kv_group.layer_names)))
        group_specs[gid] = specs
        fingerprint.append(
            {
                "gid": gid,
                "spec": type(group_spec).__name__,
                "kind": sem.kind,
                "block_tokens": bt,
                "window_tokens": sem.window_tokens,
                "layers": _layer_fingerprint(gid, specs, par),
            }
        )
    if not groups:
        raise Unsupported("model has no transferable KV cache group")
    # Storage group 0 gets the server's prefix query (contiguous hit + SSD/RDMA
    # prefetch); that is only exact for a full group, so put one first.
    groups.sort(key=lambda g: g.sem.kind != "full")
    if groups[0].sem.kind != "full":
        raise Unsupported(
            "no full-attention group: storage group 0 uses the server's prefix query, "
            "which needs a membership-only mode for window-only models"
        )

    decls = _rank_decls(groups, group_specs, par)
    backend, page_first = _transfer_mode(decls, len(groups))
    _check_seal_plan(sem_mod, decls)

    step = 1
    for g in groups:
        if g.sem.kind != "scratch":
            step = math.lcm(step, g.sem.block_tokens)
    doc = {
        "groups": fingerprint,
        # Page byte order (e.g. LBNHC pages are [N, H, C], LBHNC [H, N, C]):
        # adapters declare sizes only, so two orders would otherwise share a
        # namespace and load transposed bytes. Set on both roles in vLLM 0.32.
        "kv_cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
        "parallel": [par.tp_size, par.pp_size, par.dcp_size, par.dcp_interleave],
        "transfer": [backend, page_first],
        **identity,
    }
    blob = json.dumps(doc, sort_keys=True, default=str).encode()
    namespace = hashlib.sha256(blob).hexdigest()[:8]
    return Layout(tuple(groups), step, namespace, backend, page_first)


def _tp_ranks(par: Parallel) -> list[Parallel]:
    """This PP stage's TP ranks; DCP ranks are taken as tp_rank % dcp_size."""
    return [
        dataclasses.replace(par, tp_rank=r, dcp_rank=r % par.dcp_size) for r in range(par.tp_size)
    ]


def _layer_fingerprint(gid: int, specs: dict[str, Any], par: Parallel) -> list[Any]:
    """Distinct per-layer declarations across all TP ranks.

    Rank- and layer-name-independent, so every TP/PP rank and the scheduler
    derive the same namespace.
    """
    seen: set[str] = set()
    for name, spec in specs.items():
        adapter = adapter_for(spec, gid)
        decls = []
        for rank in _tp_ranks(par):
            decl = adapter.layer(spec, name, rank)
            decls.append([list(decl.segments), repr(decl.shard)])
        seen.add(json.dumps([type(spec).__name__, decls]))
    return sorted(seen)


# (tp_rank, sid, layer name, declaration)
_Decl = tuple[int, int, str, LayerDecl]


def _rank_decls(
    groups: list[GroupInfo], group_specs: dict[int, dict[str, Any]], par: Parallel
) -> list[_Decl]:
    out: list[_Decl] = []
    for rank in _tp_ranks(par):
        for sid, g in enumerate(groups):
            for name, spec in group_specs[g.gid].items():
                out.append(
                    (rank.tp_rank, sid, name, adapter_for(spec, g.gid).layer(spec, name, rank))
                )
    return out


def _transfer_mode(decls: list[_Decl], n_groups: int) -> tuple[str, bool]:
    modes = {(d.transfer_backend, d.page_first) for _, _, _, d in decls}
    if len(modes) != 1:
        raise Unsupported(f"layers disagree on (transfer_backend, page_first): {sorted(modes)}")
    ((backend, page_first),) = modes
    if backend not in TRANSFER_BACKENDS:
        raise Unsupported(f"unknown transfer_backend {backend!r}; known: {TRANSFER_BACKENDS}")
    if page_first and n_groups > 1:
        raise Unsupported("page_first storage supports a single storage group only")
    return backend, page_first


def _check_seal_plan(semantics: Any, decls: list[_Decl]) -> None:
    """V0: every TP rank's declarations must seal without gaps or overlaps."""
    by_rank: dict[int, list[Any]] = {}
    for rank, sid, name, d in decls:
        region = semantics.Region(name, sid, [int(s) for s in d.segments], d.shard)
        by_rank.setdefault(rank, []).append(region)
    try:
        semantics.seal_plan(sorted(by_rank.items()))
    except ValueError as e:
        raise Unsupported(f"adapter declarations fail seal_plan: {e}") from e


def may_zero_or_copy_blocks(kv_cache_config: Any) -> bool:
    """vLLM may zero new blocks or copy partial prefix hits (copy-on-write)
    into them at the start of a step.

    vLLM's V2 model runner does both before ``handle_preemptions``, so a
    pending save of a freed block could read the overwritten bytes. Zeroing:
    ``needs_kv_cache_zeroing`` (Mamba or mixed-precision KV). Copy-on-write:
    prefix hits finer than a group's block (``hash_block_size`` smaller than
    its ``block_size``). Unknown fields count as "may".
    """
    if getattr(kv_cache_config, "needs_kv_cache_zeroing", True):
        return True
    hash_block_size = getattr(kv_cache_config, "hash_block_size", None)
    if hash_block_size is None:
        return True
    return any(
        g.kv_cache_spec.block_size != hash_block_size for g in kv_cache_config.kv_cache_groups
    )
