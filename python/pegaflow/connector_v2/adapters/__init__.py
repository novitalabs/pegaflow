"""Adapter registry: one module per vLLM KVCacheSpec class.

An adapter translates one spec class into
- a ``semantics.Group`` (what resuming at token ``p`` needs), and
- per-layer declarations: stored byte layout + ``semantics.Shard`` (``layer``)
  and the GPU addressing the server registers (``view``).

Lookup is by exact class name: subclasses (e.g. ``MLAAttentionSpec`` of
``FullAttentionSpec``) have different layouts and need their own entry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol


class Unsupported(RuntimeError):
    """The model's KV layout cannot be expressed by the registered adapters."""


@dataclass(frozen=True)
class Parallel:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    dcp_rank: int = 0
    dcp_size: int = 1
    dcp_interleave: int = 1
    world_size: int = 1
    # vLLM KVCacheConfig.kv_tp_replicas: consecutive TP ranks holding identical KV.
    kv_tp_replicas: int = 1


# RegisterContext transfer backends (``python/src/lib.rs`` register_context_batch).
TRANSFER_BACKENDS = ("direct", "kernel")


@dataclass(frozen=True)
class LayerDecl:
    """What one rank stores for one layer per block.

    ``transfer_backend`` / ``page_first`` are instance-wide RegisterContext
    fields: every layer must declare the same values (checked by the core) and
    they are part of the namespace.
    """

    segments: tuple[int, ...]  # byte size of each stored segment (e.g. K, V)
    shard: Any  # semantics.Shard
    transfer_backend: str = "direct"  # "direct" | "kernel"
    page_first: bool = False  # all layers of a block in one page per rank


@dataclass(frozen=True)
class LayerView:
    """How the server addresses one layer in GPU memory (RegisterContext fields).

    ``bytes_per_block`` is the size of ONE segment of one block, which is what
    the server copies per segment (for split K/V that is K alone, not K+V).
    Every declared segment must have exactly this size; the core checks it at
    registration. ``kv_stride_bytes`` is the distance between a block's
    segments (equal to ``bytes_per_block`` when they are adjacent).
    """

    tensor: Any  # torch.Tensor shared with the server over CUDA IPC
    num_blocks: int
    bytes_per_block: int
    kv_stride_bytes: int


class Adapter(Protocol):
    def group(self, spec: Any) -> Any: ...

    def layer(self, spec: Any, layer_name: str, par: Parallel) -> LayerDecl: ...

    def view(self, spec: Any, layer_name: str, kv: Any) -> LayerView: ...


# vLLM KVCacheSpec class name -> module in this package that exposes ``ADAPTER``.
_REGISTRY: dict[str, str] = {
    "FullAttentionSpec": "full_attention",
}


def adapter_for(spec: Any, group_index: int) -> Adapter:
    name = type(spec).__name__
    module = _REGISTRY.get(name)
    if module is None:
        raise Unsupported(
            f"no adapter for {name} (group {group_index}); add "
            f"python/pegaflow/connector_v2/adapters/<name>.py exposing ADAPTER and "
            f"register {name!r} in adapters/__init__.py _REGISTRY"
        )
    return importlib.import_module(f"{__name__}.{module}").ADAPTER
