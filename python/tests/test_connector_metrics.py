from __future__ import annotations

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.connector_metrics import (  # noqa: E402
    KV_TRANSFER_DURATION_BUCKETS,
    build_buckets,
)


def test_build_buckets_keeps_fractional_seconds() -> None:
    assert build_buckets([1, 2, 4, 8], 1, -2) == [
        0.01,
        0.02,
        0.04,
        0.08,
        0.1,
        0.2,
        0.4,
        0.8,
        1,
    ]


def test_kv_transfer_duration_buckets_cover_slow_loads_and_saves() -> None:
    assert 0.2 in KV_TRANSFER_DURATION_BUCKETS
    assert 100 in KV_TRANSFER_DURATION_BUCKETS
    assert KV_TRANSFER_DURATION_BUCKETS[-1] == 1000
