"""Capability tests for vLLM cache-group layouts supported by PegaFlow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from pegaflow.connector.common import CacheGroupLayout  # noqa: E402


def _group(name, spec):
    return SimpleNamespace(layer_names=(name,), kv_cache_spec=spec)


def _config(*groups):
    return SimpleNamespace(kv_cache_groups=groups)


def _full_attention(block_size=16, spec_type=FullAttentionSpec):
    try:
        return spec_type(block_size=block_size, num_kv_heads=1, head_size=1, dtype=None)
    except TypeError:
        spec = spec_type()
        spec.block_size = block_size
        return spec


def _mamba(block_size=16, mode="align"):
    try:
        return MambaSpec(
            block_size=block_size, shapes=((1,),), dtypes=(None,), mamba_cache_mode=mode
        )
    except TypeError:
        spec = MambaSpec()
        spec.block_size = block_size
        spec.mamba_cache_mode = mode
        return spec


def _mla(block_size=16, head_size=128):
    try:
        return MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=None
        )
    except TypeError:
        spec = MLAAttentionSpec()
        spec.block_size = block_size
        spec.head_size = head_size
        return spec


def _sliding_window(block_size=16, window=1024, extra_retained_tokens=0):
    try:
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=None,
            sliding_window=window,
            extra_retained_tokens=extra_retained_tokens,
        )
    except TypeError:
        spec = SlidingWindowSpec()
        spec.block_size = block_size
        spec.sliding_window = window
        spec.extra_retained_tokens = extra_retained_tokens
        return spec


def _spec_of(spec_type, block_size=16):
    if spec_type is FullAttentionSpec:
        return _full_attention(block_size)
    if spec_type is MLAAttentionSpec:
        return _mla(block_size)
    if spec_type is SlidingWindowSpec:
        return _sliding_window(block_size)
    if issubclass(spec_type, FullAttentionSpec):
        return _full_attention(block_size, spec_type)
    return spec_type()


class SpecializedFullAttentionSpec(FullAttentionSpec):
    pass


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, SpecializedFullAttentionSpec])
def test_accepts_full_attention_with_aligned_mamba(spec_type):
    attention = _spec_of(spec_type, 528)
    config = _config(
        _group("attention", attention),
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
    attention = _spec_of(spec_type)

    layout = CacheGroupLayout.from_config(_config(_group("attention", attention)))

    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state


@pytest.mark.parametrize("spec_type", [FullAttentionSpec, MLAAttentionSpec])
def test_single_group_dcp_hash_spans_multiple_physical_blocks(spec_type):
    config = _config(_group("attention", _spec_of(spec_type, block_size=64)))

    layout = CacheGroupLayout.from_config(config, hash_block_size=128)

    assert layout.group_block_sizes == (64,)
    assert layout.storage_group_ids == (0,)


def test_accepts_single_uniform_mla_group():
    group = SimpleNamespace(
        layer_names=("model.layers.0.self_attn.attn", "model.layers.0.self_attn.indexer.k_cache"),
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "model.layers.0.self_attn.attn": _mla(head_size=576),
                "model.layers.0.self_attn.indexer.k_cache": _mla(head_size=128),
            },
        ),
    )

    layout = CacheGroupLayout.from_config(_config(group))

    assert layout.layer_names == (group.layer_names,)
    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state


def test_rejects_per_layer_cadences_that_cannot_share_a_group_hash():
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attention": _mla(16), "other": _mla(32)},
    )
    group = SimpleNamespace(layer_names=("attention", "other"), kv_cache_spec=spec)

    with pytest.raises(RuntimeError, match="within each cache group"):
        CacheGroupLayout.from_config(_config(group))


def test_rejects_dense_groups_with_different_hash_cadences():
    config = _config(
        _group("attention", _mla(16)),
        _group("other_attention", _mla(32)),
        _group("recurrent", _mamba(16)),
    )

    with pytest.raises(RuntimeError, match="dense attention groups"):
        CacheGroupLayout.from_config(config)


@pytest.mark.parametrize("other_spec_type", [FullAttentionSpec, SlidingWindowSpec])
def test_rejects_uniform_group_with_non_mla_layer(other_spec_type):
    other_spec = _spec_of(other_spec_type)
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attention": _mla(), "other": other_spec},
    )

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("attention", spec)))


def test_rejects_empty_uniform_mla_group():
    spec = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={})

    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("attention", spec)))


@pytest.mark.parametrize("mode", ["align", "all"])
def test_rejects_single_mamba_group(mode):
    with pytest.raises(RuntimeError, match="single cache group"):
        CacheGroupLayout.from_config(_config(_group("recurrent", _mamba(mode=mode))))


def test_rejects_single_sliding_window_group():
    sliding_window = _sliding_window()

    with pytest.raises(RuntimeError, match="dense FullAttention"):
        CacheGroupLayout.from_config(_config(_group("sliding_window", sliding_window)))


def test_rejects_sliding_window_when_hybrid_manager_disabled():
    sliding_window = _sliding_window()
    with pytest.raises(RuntimeError, match="hybrid KV cache manager"):
        CacheGroupLayout.from_config(
            _config(_group("sliding_window", sliding_window)),
            allow_sliding_window=False,
        )


def test_rejects_different_recurrent_block_sizes():
    config = _config(
        _group("attention", _full_attention(block_size=16)),
        _group("recurrent", _mamba(block_size=32)),
    )

    with pytest.raises(RuntimeError, match="recurrent cache groups"):
        CacheGroupLayout.from_config(config, hash_block_size=16)


def test_rejects_non_integral_logical_block_size_ratio():
    config = _config(
        _group("attention", _full_attention(block_size=16)),
        _group("sliding_window", _sliding_window(block_size=24)),
    )
    with pytest.raises(RuntimeError, match="integer multiples"):
        CacheGroupLayout.from_config(config, hash_block_size=16)


def test_rejects_multiple_full_attention_groups_without_mamba():
    config = _config(
        _group("first", _full_attention()),
        _group("second", _full_attention()),
    )

    with pytest.raises(RuntimeError, match="SlidingWindow or Mamba"):
        CacheGroupLayout.from_config(config)


def test_rejects_mamba_groups_without_full_attention():
    config = _config(
        _group("recurrent.0", _mamba(block_size=528)),
        _group("recurrent.1", _mamba(block_size=528)),
    )

    with pytest.raises(RuntimeError, match="dense FullAttention"):
        CacheGroupLayout.from_config(config)


def test_accepts_full_attention_with_sliding_window():
    sliding_window = _sliding_window()
    config = _config(
        _group("attention", _full_attention()),
        _group("sliding_window", sliding_window),
    )

    layout = CacheGroupLayout.from_config(config, hash_block_size=16)
    assert layout.hash_group_index == 0
    assert not layout.has_recurrent_state
    assert not layout.requires_group_specific_block_mapping


def test_rejects_sliding_window_with_mamba():
    config = _config(
        _group("attention", _full_attention()),
        _group("sliding_window", _sliding_window()),
        _group("recurrent", _mamba()),
    )

    with pytest.raises(RuntimeError, match="SlidingWindowSpec with Mamba"):
        CacheGroupLayout.from_config(config)


def test_accepts_heterogeneous_sliding_window_mapping():
    config = _config(
        _group("attention", _full_attention(block_size=32)),
        _group("sliding_window", _sliding_window(block_size=16)),
    )

    layout = CacheGroupLayout.from_config(config, hash_block_size=16)

    assert layout.sliding_window_group_indices == frozenset({1})
    assert layout.requires_group_specific_block_mapping


@pytest.mark.parametrize(
    ("sliding_sizes", "storage_groups"),
    [
        ((32, 16), (0, 1, 2)),
        ((16, 16), (0, 1, 1)),
    ],
)
def test_sliding_groups_share_storage_only_at_the_same_block_cadence(
    sliding_sizes, storage_groups
):
    config = _config(
        _group("attention", _full_attention(block_size=32)),
        *(
            _group(f"sliding_{index}", _sliding_window(block_size=size))
            for index, size in enumerate(sliding_sizes)
        ),
    )

    layout = CacheGroupLayout.from_config(config, hash_block_size=16)

    assert layout.storage_group_ids == storage_groups


def test_sliding_window_layout_preserves_extra_retained_tokens():
    config = _config(
        _group("attention", _full_attention()),
        _group("sliding_window", _sliding_window(extra_retained_tokens=15)),
    )

    layout = CacheGroupLayout.from_config(config, hash_block_size=16)

    assert layout.group_extra_retained_tokens == (None, 15)
    # vLLM retains sliding_window - 1 tokens for attention plus the extra
    # trailing speculative tokens below that window.
    assert layout.sliding_retained_tokens_of(1) == 1024 - 1 + 15


def test_rejects_negative_extra_retained_tokens():
    config = _config(
        _group("attention", _full_attention()),
        _group("sliding_window", _sliding_window(extra_retained_tokens=-1)),
    )

    with pytest.raises(RuntimeError, match="extra_retained_tokens"):
        CacheGroupLayout.from_config(config, hash_block_size=16)


@pytest.mark.parametrize("dense_size", [16, 48], ids=["smaller-dense", "non-divisible"])
def test_rejects_dense_cadence_smaller_than_scheduler_alignment(dense_size):
    config = _config(
        _group("attention", _full_attention(block_size=dense_size)),
        _group("sliding_window", _sliding_window(block_size=32)),
    )

    with pytest.raises(RuntimeError, match="dense KV blocks match the scheduler alignment"):
        CacheGroupLayout.from_config(config, hash_block_size=16)


def test_rejects_uniform_attention_group_with_sliding_window_layers():
    full = _full_attention(block_size=16)
    sliding = _sliding_window(32, 128)
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"full": full, "sliding": sliding},
    )
    with pytest.raises(RuntimeError, match="separate cache group"):
        CacheGroupLayout.from_config(
            _config(SimpleNamespace(layer_names=("full", "sliding"), kv_cache_spec=spec)),
            hash_block_size=16,
        )


def test_accepts_glm53_flash_layout_with_kpool_tail_scratch():
    """GLM-5.3-Flash: uniform MLA + uniform KpoolTail scratch + mamba groups.

    The kpool tail group has its own tiny block size and holds no prefix
    state, so it is classified scratch: excluded from hashing, from the
    block-size uniformity check, and from save/load.
    """
    tail_spec = UniformTypeKVCacheSpecs(
        block_size=4,
        kv_cache_specs={"tail.0": KpoolTailSpec(), "tail.1": KpoolTailSpec()},
    )
    config = _config(
        SimpleNamespace(
            layer_names=("attn.0", "idx.0"),
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=8960,
                kv_cache_specs={"attn.0": _mla(block_size=8960), "idx.0": _mla(block_size=8960)},
            ),
        ),
        SimpleNamespace(layer_names=("tail.0", "tail.1"), kv_cache_spec=tail_spec),
        _group("recurrent.0", _mamba(block_size=8960)),
        _group("recurrent.1", _mamba(block_size=8960)),
    )

    layout = CacheGroupLayout.from_config(config, hash_block_size=8960)

    assert layout.hash_group_index == 0
    assert layout.has_recurrent_state
    assert not layout.requires_group_specific_block_mapping
    assert layout.group_block_sizes == (8960, 4, 8960, 8960)
    assert layout.recurrent_group_indices == frozenset({2, 3})
    assert layout.scratch_group_indices == frozenset({1})
    assert layout.scratch_layer_names == frozenset({"tail.0", "tail.1"})
    assert layout.storage_group_ids == (0, 0, 1, 2)


def test_accepts_mla_with_mamba():
    mla = _mla()
    config = _config(
        _group("attention", mla),
        _group("recurrent", _mamba()),
    )

    layout = CacheGroupLayout.from_config(config)

    assert layout.hash_group_index == 0
    assert layout.has_recurrent_state


def test_rejects_non_align_mamba_mode():
    config = _config(
        _group("attention", _full_attention()),
        _group("recurrent", _mamba(mode="all")),
    )

    with pytest.raises(RuntimeError, match="mamba_cache_mode='align'"):
        CacheGroupLayout.from_config(config)
