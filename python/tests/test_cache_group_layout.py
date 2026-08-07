"""Capability tests for vLLM cache-group layouts supported by PegaFlow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

from pegaflow.connector.common import CacheGroupLayout  # noqa: E402


def _group(name, spec):
    return SimpleNamespace(layer_names=(name,), kv_cache_spec=spec)


def _config(*groups):
    return SimpleNamespace(kv_cache_groups=groups)


def _full_attention(block_size=16):
    spec = FullAttentionSpec()
    spec.block_size = block_size
    return spec


def _mamba(block_size=16, mode="align"):
    spec = MambaSpec()
    spec.block_size = block_size
    spec.mamba_cache_mode = mode
    return spec


def test_accepts_full_attention_with_aligned_mamba():
    config = _config(
        _group("attention", _full_attention(block_size=528)),
        _group("recurrent", _mamba(block_size=528)),
    )

    layout = CacheGroupLayout.from_config(config)

    assert layout.layer_names == (("attention",), ("recurrent",))
    assert layout.hash_group_index == 0
    assert layout.has_recurrent_state
    assert layout.recurrent_group_indices == frozenset({1})
    assert layout.recurrent_layer_names == frozenset({"recurrent"})


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, MLAAttentionSpec])
def test_accepts_single_attention_group(spec_type):
    attention = spec_type()
    attention.block_size = 16

    layout = CacheGroupLayout.from_config(_config(_group("attention", attention)))

    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state


@pytest.mark.parametrize("mode", ["align", "all"])
def test_rejects_single_mamba_group(mode):
    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("recurrent", _mamba(mode=mode))))


def test_rejects_single_sliding_window_group():
    sliding_window = SlidingWindowSpec()
    sliding_window.block_size = 16

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("sliding_window", sliding_window)))


def test_rejects_misaligned_logical_block_sizes():
    config = _config(
        _group("attention", _full_attention(block_size=16)),
        _group("recurrent", _mamba(block_size=32)),
    )

    with pytest.raises(RuntimeError, match="identical logical block sizes"):
        CacheGroupLayout.from_config(config)


def test_rejects_multiple_full_attention_groups_without_mamba():
    config = _config(
        _group("first", _full_attention()),
        _group("second", _full_attention()),
    )

    with pytest.raises(RuntimeError, match="both FullAttention and Mamba"):
        CacheGroupLayout.from_config(config)


def test_rejects_mamba_groups_without_full_attention():
    config = _config(
        _group("recurrent.0", _mamba(block_size=528)),
        _group("recurrent.1", _mamba(block_size=528)),
    )

    with pytest.raises(RuntimeError, match="dense FullAttention"):
        CacheGroupLayout.from_config(config)


def test_rejects_full_attention_with_sliding_window():
    sliding_window = SlidingWindowSpec()
    sliding_window.block_size = 16
    config = _config(
        _group("attention", _full_attention()),
        _group("sliding_window", sliding_window),
    )

    with pytest.raises(RuntimeError, match="only FullAttention and Mamba"):
        CacheGroupLayout.from_config(config)


def test_rejects_mla_with_mamba():
    mla = MLAAttentionSpec()
    mla.block_size = 16
    config = _config(
        _group("attention", mla),
        _group("recurrent", _mamba()),
    )

    with pytest.raises(RuntimeError, match="only FullAttention and Mamba"):
        CacheGroupLayout.from_config(config)


def test_rejects_non_align_mamba_mode():
    config = _config(
        _group("attention", _full_attention()),
        _group("recurrent", _mamba(mode="all")),
    )

    with pytest.raises(RuntimeError, match="mamba_cache_mode='align'"):
        CacheGroupLayout.from_config(config)
