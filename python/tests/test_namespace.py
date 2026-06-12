"""derive_namespace contract: split whenever stored blocks would be incompatible.

Two deployments sharing a namespace must produce byte-compatible blocks.
The dangerous direction is a missed split: same namespace, different block
geometry or registered KV layer set (e.g. MTP drafter on vs off) leads to
phantom hits or cross-contaminated loads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import derive_namespace  # noqa: E402


def _vllm_config(block_size: int = 16, speculative_config=None):
    model_config = SimpleNamespace(
        model="Qwen3-4B",
        dtype="bfloat16",
        get_total_num_kv_heads=lambda: 8,
        get_head_size=lambda: 128,
        get_total_num_hidden_layers=lambda: 36,
    )
    cache_config = SimpleNamespace(cache_dtype="auto", block_size=block_size)
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        speculative_config=speculative_config,
    )


def _spec(method: str, model: str | None = None):
    return SimpleNamespace(method=method, model=model)


@pytest.mark.parametrize(
    ("other", "expect_same"),
    [
        # vLLM normalizes all MTP variants ("deepseek_mtp", ...) to "mtp"
        # before the connector sees them.
        pytest.param(_vllm_config(speculative_config=_spec("mtp")), False, id="mtp-splits"),
        pytest.param(
            _vllm_config(speculative_config=_spec("eagle", "drafter-path")),
            False,
            id="eagle-splits",
        ),
        pytest.param(
            _vllm_config(speculative_config=_spec("future_method")),
            False,
            id="unknown-method-splits",
        ),
        pytest.param(_vllm_config(speculative_config=_spec("ngram")), True, id="ngram-shares"),
        pytest.param(_vllm_config(block_size=32), False, id="block-size-splits"),
        pytest.param(_vllm_config(), True, id="identical-shares"),
    ],
)
def test_namespace_splits_on_kv_layout_factors(other, expect_same):
    base = derive_namespace(_vllm_config(), tp_size=1)
    derived = derive_namespace(other, tp_size=1)
    assert (derived == base) is expect_same


def test_namespace_golden_value():
    """The namespace is a cross-process persistent key (read cache, SSD index,
    metaserver). Any change to the factor set or its serialization invalidates
    every warm cache in a cluster on upgrade — this pin makes that an explicit
    decision instead of a refactoring side effect. If you changed the factors
    on purpose, update the value and say so in the PR.
    """
    assert derive_namespace(_vllm_config(), tp_size=1) == "d84eb2db"
