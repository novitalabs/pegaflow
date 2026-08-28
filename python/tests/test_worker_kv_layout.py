"""Unit tests for worker-side KV cache layout registration."""

from __future__ import annotations

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import LocalRecurrentLoad, LocalRecurrentSave  # noqa: E402
from pegaflow.connector.linear_state_cache import LinearStateSlot  # noqa: E402
from pegaflow.connector.worker import (  # noqa: E402
    WorkerConnector,
    _infer_kv_cache_registration,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], stride: tuple[int, ...], element_size: int = 2):
        self.shape = shape
        self._stride = stride
        self._element_size = element_size

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def element_size(self) -> int:
        return self._element_size


def test_mla_blocks_first_physical_rows_are_grouped_into_logical_blocks():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(6, 64, 576),
            stride=(64 * 576, 576, 1),
            element_size=2,
        ),
        logical_block_size=128,
        is_mla=True,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 3
    assert info.bytes_per_block == 2 * 64 * 576 * 2
    assert info.kv_stride_bytes == 0
    assert info.segments == 1
    assert info.physical_blocks_per_logical_block == 2


def test_non_mla_kv_first_uses_legacy_block_stride():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(2, 6, 64, 4, 8),
            stride=(6 * 64 * 4 * 8, 64 * 4 * 8, 4 * 8, 8, 1),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.layout == "KV-first"
    assert info.num_blocks == 6
    assert info.bytes_per_block == 64 * 4 * 8 * 2
    assert info.kv_stride_bytes == 6 * 64 * 4 * 8 * 2
    assert info.segments == 2
    assert info.physical_blocks_per_logical_block == 1


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


def test_mla_equal_physical_and_logical_block_size_is_unchanged():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(3, 128, 576),
            stride=(128 * 576, 576, 1),
            element_size=2,
        ),
        logical_block_size=128,
        is_mla=True,
    )

    assert info.num_blocks == 3
    assert info.bytes_per_block == 128 * 576 * 2
    assert info.physical_blocks_per_logical_block == 1


def test_recurrent_state_uses_one_page_per_logical_block():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(2, 1, 4096),
            stride=(4096, 4096, 1),
            element_size=2,
        ),
        logical_block_size=1536,
        is_mla=True,
        is_recurrent_state=True,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 2
    assert info.bytes_per_block == 4096 * 2
    assert info.kv_stride_bytes == 0
    assert info.segments == 1
    assert info.physical_blocks_per_logical_block == 1


def test_non_mla_cross_layer_layout_uses_legacy_block_stride():
    info = _infer_kv_cache_registration(
        FakeTensor(
            shape=(6, 93, 2, 64, 1, 128),
            stride=(
                93 * 2 * 64 * 1 * 128,
                2 * 64 * 1 * 128,
                64 * 1 * 128,
                1 * 128,
                128,
                1,
            ),
            element_size=2,
        ),
        logical_block_size=128,
    )

    assert info.layout == "blocks-first"
    assert info.num_blocks == 6
    assert info.bytes_per_block == 93 * 2 * 64 * 1 * 128 * 2
    assert info.physical_blocks_per_logical_block == 1


def test_logical_block_size_must_be_multiple_of_physical_block_size():
    with pytest.raises(ValueError, match="logical block size"):
        _infer_kv_cache_registration(
            FakeTensor(
                shape=(3, 96, 576),
                stride=(96 * 576, 576, 1),
                element_size=2,
            ),
            logical_block_size=128,
            is_mla=True,
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
            is_mla=True,
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
            is_mla=True,
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
            is_mla=True,
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
            is_mla=True,
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
            is_mla=True,
        )


class _Page:
    def __init__(self, value: bytes):
        self.value = value

    def copy_(self, other: _Page, non_blocking: bool = False):
        assert non_blocking
        self.value = other.value


class _Pages:
    def __init__(self, *values: bytes):
        self.rows = [_Page(value) for value in values]
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, index: int) -> _Page:
        return self.rows[index]


def _local_worker_for_copy():
    worker = WorkerConnector.__new__(WorkerConnector)
    worker._local_recurrent_layers = {
        "recurrent": type(
            "Layer",
            (),
            {
                "group_index": 1,
                "pages": _Pages(b"src0", b"src1", b"empty"),
                "pool": _Pages(b"pool0", b"pool1"),
                "generations": [1, 0],
            },
        )()
    }
    worker._pending_local_load_events = {}
    return worker


def test_local_recurrent_save_copies_full_page_and_advances_generation():
    worker = _local_worker_for_copy()
    ref = LinearStateSlot(slot=1, generation=2, block_hash=b"hash")

    worker._save_local_recurrent(LocalRecurrentSave(target=ref, source_block_ids=((1, 1),)))

    layer = worker._local_recurrent_layers["recurrent"]
    assert layer.pool[0].value == b"pool0"
    assert layer.pool[1].value == b"src1"
    assert layer.generations == [1, 2]


def test_local_recurrent_load_rejects_stale_generation(monkeypatch):
    worker = _local_worker_for_copy()
    monkeypatch.setattr(
        "pegaflow.connector.worker.torch.cuda.Event",
        lambda: type("Event", (), {"record": lambda self: None})(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="stale local recurrent slot"):
        worker._start_local_recurrent_load(
            "req",
            LocalRecurrentLoad(
                source=LinearStateSlot(slot=0, generation=2, block_hash=b"hash"),
                destination_block_ids=((1, 2),),
            ),
        )

    assert worker._local_recurrent_layers["recurrent"].pages[2].value == b"empty"
