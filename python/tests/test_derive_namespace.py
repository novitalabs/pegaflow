"""Unit tests for namespace derivation factors.

The namespace isolates storage by KV layout: any config that changes the
on-storage block layout must change the namespace, or two incompatible layouts
collide under one namespace and loads fail the server-side slot-count guard.
"""

from __future__ import annotations

from types import SimpleNamespace

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import derive_namespace  # noqa: E402


def _make_vllm_config(
    *,
    pp_size: int = 1,
    mla_layer_split: bool = False,
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        model="/data/models/GLM-5.2-FP8",
        dtype="bfloat16",
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: 576,
        get_total_num_hidden_layers=lambda: 78,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        additional_config={"mla_layer_split_kv_cache": mla_layer_split},
    )


def _ns(**kwargs) -> str:
    return derive_namespace(_make_vllm_config(**kwargs), tp_size=8)


def test_pp_size_isolates_namespace():
    # Same model, different pipeline-parallel degree -> different layer split
    # per server -> must not share storage.
    assert _ns(pp_size=1) != _ns(pp_size=8)


def test_mla_layer_split_isolates_namespace():
    # Layer-split registration shards each block's slots differently from the
    # default full-slot layout (the haitao GLM-5.2 99-vs-156 collision).
    assert _ns(mla_layer_split=False) != _ns(mla_layer_split=True)


def test_namespace_is_stable_for_same_config():
    assert _ns(pp_size=4, mla_layer_split=True) == _ns(pp_size=4, mla_layer_split=True)


def test_missing_additional_config_defaults_to_no_split():
    cfg = _make_vllm_config(pp_size=4)
    cfg.additional_config = None
    assert derive_namespace(cfg, tp_size=8) == _ns(pp_size=4, mla_layer_split=False)
