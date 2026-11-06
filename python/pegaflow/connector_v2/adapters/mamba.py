"""vLLM ``MambaSpec``: recurrent state (Mamba, GDN, ...) in ``align`` mode.

Dependency: ``Checkpoint``. In ``mamba_cache_mode="align"`` vLLM keeps one
state page per manager block, and the page at block index ``k - 1`` holds the
state after exactly ``k * block_size`` tokens once a step ends there. To
resume at ``p`` a request needs only that one page (``needed = [p/bt - 1]``).

Stored unit: one manager block of one layer = one state page, copied as ONE
segment. A page packs the layer's states back to back (e.g. GDN conv bf16
``[3, 8192]`` then ssm fp32 ``[32, 128, 128]``) and may be padded to the
attention page (``page_size_padded``). The segment is the whole padded page:
vLLM's per-layer view is ``[B, 1, 1, real]`` int8 with blocks ``page`` bytes
apart, and the server's registration addresses block ``b`` at
``data_ptr + b * bytes_per_block``, so the copy unit must equal the block
stride. The padding is never read by vLLM; storing it costs
``page / real - 1`` (0.76% on Qwen3.5-4B) and needs no server change.

Byte order: the declaration carries the page size only, so the state shapes
and dtypes go into ``LayerDecl.layout`` (namespace). That covers
``mamba_cache_dtype`` / ``mamba_ssm_cache_dtype`` changing the page while the
model path stays the same.

The core saves checkpoint groups from vLLM's ``boundary_state_offloads``
(``core/scheduler.py`` ``plan_saves``); this adapter only declares facts.

Not visible in ``MambaSpec``, so not rejected here: ``cache_config.use_replayssm``
(Mamba2 decode writes the state back only on flush, so a block-end page may be
stale when offered). GDN (Qwen3.5) does not use it; a Mamba2 model must check
it before it relies on this adapter.
"""

from __future__ import annotations

from math import prod
from typing import Any

from . import LayerDecl, LayerView, Parallel, Unsupported


class MambaAdapter:
    def __init__(self, semantics: Any | None = None):
        self._semantics = semantics

    @property
    def sem(self) -> Any:
        if self._semantics is None:
            from pegaflow import semantics

            self._semantics = semantics
        return self._semantics

    # ------------------------------------------------------------------ group
    def group(self, spec: Any) -> Any:
        _check_spec(spec)
        return self.sem.Group.checkpoint(spec.block_size)

    # ------------------------------------------------------------------ layer
    def layer(self, spec: Any, layer_name: str, par: Parallel) -> LayerDecl:
        _check_spec(spec)
        if par.dcp_size > 1:
            raise Unsupported("MambaSpec: DCP (state replicated across DCP ranks) not validated")
        # GDN/Mamba split their heads and conv channels across TP ranks
        # (vLLM ``tp_replicated=False``); a replicated state is one copy.
        if getattr(spec, "tp_replicated", False):
            shard = self.sem.Shard.replicated()
        else:
            shard = self.sem.Shard.heads(par.tp_rank, par.tp_size)
        return LayerDecl((spec.page_size_bytes,), shard, layout=_state_layout(spec))

    # ------------------------------------------------------------------- view
    def view(self, spec: Any, layer_name: str, kv: Any) -> LayerView:
        """``kv``: vLLM's raw per-layer view ``[B, 1, 1, real]`` (int8)."""
        page = spec.page_size_bytes
        real = _real_bytes(spec)
        shape, stride, elem = tuple(kv.shape), tuple(kv.stride()), kv.element_size()
        if len(shape) != 4 or shape[1:] != (1, 1, real // elem) or real % elem:
            raise Unsupported(
                f"{layer_name}: expected a [B, 1, 1, {real} B] state view, got shape {shape}"
            )
        if stride[3] != 1 or stride[0] * elem != page:
            raise Unsupported(
                f"{layer_name}: state pages are {stride[0] * elem} B apart (stride {stride}), "
                f"need dense {page} B pages (layer-compact KV layout)"
            )
        return LayerView(kv, shape[0], page, page)


def _real_bytes(spec: Any) -> int:
    return sum(
        prod(shape) * _dtype_bytes(dtype)
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
    )


_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_bytes(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    name = _dtype_name(dtype)
    if name not in _DTYPE_BYTES:
        raise Unsupported(f"MambaSpec: unknown state dtype {dtype}")
    return _DTYPE_BYTES[name]


def _state_layout(spec: Any) -> tuple[Any, ...]:
    """Shapes and dtypes of the packed states, in page order."""
    return tuple(
        (tuple(int(d) for d in shape), _dtype_name(dtype))
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
    )


# Fields that change which positions hold a state or how many pages a
# request owns; each needs its own semantics when a model brings it.
_DEFAULT_FIELDS = {
    # Speculative decoding adds state pages per request and shifts
    # checkpoints (EAGLE block-drop).
    "num_speculative_blocks": 0,
    "num_prefill_checkpoint_blocks": 0,
    "prefill_checkpoint_alignment": None,
    # One state per block (not per token / per chunk).
    "num_heads": 1,
    "tokens_per_state": -1,
}


def _check_spec(spec: Any) -> None:
    mode = getattr(spec, "mamba_cache_mode", None)
    if mode != "align":
        # "none" keeps one running state per request: no position holds a
        # resumable checkpoint, and vLLM disables prefix caching anyway.
        raise Unsupported(f"MambaSpec with mamba_cache_mode={mode!r}: only 'align' is supported")
    odd = {
        name: getattr(spec, name)
        for name, default in _DEFAULT_FIELDS.items()
        if getattr(spec, name, default) != default
    }
    if odd:
        raise Unsupported(f"MambaSpec with {odd} is not supported yet")
    if not getattr(spec, "has_layer_views", True):
        raise Unsupported("MambaSpec without per-layer views is not supported")
    real = _real_bytes(spec)
    if not 0 < real <= spec.page_size_bytes:
        raise Unsupported(
            f"MambaSpec: states take {real} B, page_size_bytes={spec.page_size_bytes}"
        )


ADAPTER = MambaAdapter()
