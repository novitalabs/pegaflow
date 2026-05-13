"""Unit tests for worker-side KV cache layout registration."""

from __future__ import annotations

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.worker import _infer_kv_cache_registration  # noqa: E402


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], stride: tuple[int, ...], element_size: int = 2):
        self.shape = shape
        self._stride = stride
        self._element_size = element_size

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def element_size(self) -> int:
        return self._element_size


def test_blocks_first_flashmla_physical_rows_are_grouped_into_logical_blocks():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(6, 64, 576),
            stride=(64 * 576, 576, 1),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 3
    assert info.bytes_per_block == 2 * 64 * 576 * 2
    assert info.kv_stride_bytes == 0
    assert info.segments == 1
    assert info.physical_blocks_per_logical_block == 2


def test_kv_first_physical_rows_are_grouped_per_kv_segment():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(2, 6, 64, 4, 8),
            stride=(6 * 64 * 4 * 8, 64 * 4 * 8, 4 * 8, 8, 1),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.layout == "KV-first"
    assert info.num_blocks == 3
    assert info.bytes_per_block == 2 * 64 * 4 * 8 * 2
    assert info.kv_stride_bytes == 6 * 64 * 4 * 8 * 2
    assert info.segments == 2
    assert info.physical_blocks_per_logical_block == 2


def test_mla_prefers_blocks_first_when_first_dimension_is_two():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(2, 64, 576),
            stride=(64 * 576, 576, 1),
            element_size=2,
        ),
        logical_block_size=128,
        is_mla=True,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 1
    assert info.bytes_per_block == 2 * 64 * 576 * 2
    assert info.kv_stride_bytes == 0
    assert info.segments == 1
    assert info.physical_blocks_per_logical_block == 2


def test_equal_physical_and_logical_block_size_is_unchanged():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(3, 128, 576),
            stride=(128 * 576, 576, 1),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.num_blocks == 3
    assert info.bytes_per_block == 128 * 576 * 2
    assert info.physical_blocks_per_logical_block == 1


def test_blocks_first_standard_attention_uses_third_dim_as_physical_block_size():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(6, 2, 64, 4, 8),
            stride=(2 * 64 * 4 * 8, 64 * 4 * 8, 4 * 8, 8, 1),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 3
    assert info.bytes_per_block == 2 * 2 * 64 * 4 * 8 * 2
    assert info.physical_blocks_per_logical_block == 2


def test_logical_block_size_must_be_multiple_of_physical_block_size():
    with pytest.raises(ValueError, match="logical block size"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(3, 96, 576),
                stride=(96 * 576, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
        )


def test_logical_block_size_must_be_positive():
    with pytest.raises(ValueError, match="logical block size must be > 0"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(3, 128, 576),
                stride=(128 * 576, 576, 1),
                element_size=2,
            ),
            logical_block_size=0,
        )


def test_physical_block_count_must_be_positive():
    with pytest.raises(ValueError, match="physical block count must be > 0"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(0, 128, 576),
                stride=(128 * 576, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
        )


def test_physical_block_size_must_be_positive():
    with pytest.raises(ValueError, match="physical block size must be > 0"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(3, 0, 576),
                stride=(0, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
        )


def test_physical_block_count_must_be_divisible_by_split_ratio():
    with pytest.raises(ValueError, match="physical block count"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(5, 64, 576),
                stride=(64 * 576, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
        )


def test_bytes_per_block_must_be_nonzero():
    with pytest.raises(ValueError, match="Invalid bytes_per_block"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(3, 128, 576),
                stride=(0, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
        )
