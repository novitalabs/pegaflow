"""vLLM ``MLAAttentionSpec``: multi-head latent attention (DeepSeek-V2/V3 style).

Stored unit: one manager block of one layer = one page of per-token latent
vectors (``kv_lora_rank`` compressed KV + ``qk_rope_head_dim`` rope key, one
"head", no separate V), copied as ONE segment. Every past token is attended,
so the group is Full.

Shard: vLLM builds the latent from the full hidden state on every TP rank, so
all ranks hold identical pages (``max_tp_shards == 1``, vLLM
``kv_tp_replicas == tp_size``). The declaration is ``Shard.replicated()``;
``seal_plan`` collapses the ranks into one fragment per layer.

Geometry is the full-attention page check (dense ``[H=1, N, C]`` pages
``page_size_bytes`` apart), inherited from ``full_attention``.

Fail closed on the MLA variants this adapter has not been validated on:
DeepSeek-V4 fields (``alignment``, ``model_version``), sparse indexer caches
(``cache_role``, ``is_index_group_leader``), packed / quantized latents
(``cache_dtype_str`` other than auto, ``kv_quant_mode``, ``state_content_bytes``)
and non-causal decode blocks.
"""

from __future__ import annotations

from typing import Any

from . import LayerDecl, Parallel, Unsupported
from .full_attention import FullAttentionAdapter, _check_spec

# Values of the MLA-only fields this adapter has been validated on (vLLM 0.32
# defaults for dense MLA; DeepSeek-V2-Lite-Chat spec dump).
_MLA_FIELDS = {
    "alignment": None,
    "model_version": None,
    "is_index_group_leader": False,
    "non_causal_multi_token_decode": False,
}


class MLAAttentionAdapter(FullAttentionAdapter):
    def group(self, spec: Any) -> Any:
        _check_mla(spec)
        return super().group(spec)

    def layer(self, spec: Any, layer_name: str, par: Parallel) -> LayerDecl:
        _check_mla(spec)
        return LayerDecl((spec.page_size_bytes,), self._shard(spec, par))

    def _shard(self, spec: Any, par: Parallel) -> Any:
        if par.dcp_size > 1 and getattr(spec, "dcp_sharded", True):
            raise Unsupported("MLAAttentionSpec: DCP token sharding is not declared yet")
        if spec.max_tp_shards != 1:
            raise Unsupported(
                f"MLAAttentionSpec with max_tp_shards={spec.max_tp_shards}: the latent "
                f"is expected to be replicated on every TP rank"
            )
        # kv_tp_replicas is the gcd over every layer; a replicated layer is
        # consistent with any divisor of tp_size.
        if par.tp_size % par.kv_tp_replicas:
            raise Unsupported(
                f"MLAAttentionSpec: vLLM kv_tp_replicas={par.kv_tp_replicas} does not "
                f"divide tp_size {par.tp_size}"
            )
        return self.sem.Shard.replicated()


def _check_mla(spec: Any) -> None:
    _check_spec(spec)
    odd = {
        name: getattr(spec, name)
        for name, default in _MLA_FIELDS.items()
        if getattr(spec, name, default) != default
    }
    # SparseCacheRole is a str enum: compare by value.
    role = getattr(spec, "cache_role", "sparse")
    if getattr(role, "value", role) != "sparse":
        odd["cache_role"] = role
    if getattr(spec, "cache_dtype_str", None) not in (None, "auto"):
        odd["cache_dtype_str"] = spec.cache_dtype_str
    if getattr(spec, "head_size_v", 0) != 0:
        odd["head_size_v"] = spec.head_size_v
    if odd:
        raise Unsupported(f"MLAAttentionSpec with {odd} is not supported yet")


ADAPTER = MLAAttentionAdapter()
