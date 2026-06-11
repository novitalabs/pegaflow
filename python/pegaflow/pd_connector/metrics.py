"""Metrics collection for the experimental P/D connector."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

from pegaflow.connector.connector_metrics import build_buckets

if TYPE_CHECKING:
    from vllm.config import VllmConfig

try:
    from vllm.v1.metrics.utils import create_metric_per_engine
except ImportError:
    create_metric_per_engine = None


PD_STATS_KEYS = {
    "pd_decode_active_waits": 0,
    "pd_prefill_active_pushes": 0,
    "pd_prefill_inflight_push_tasks": 0,
    "pd_prefill_inflight_finalize_tasks": 0,
    "pd_load_success_count": 0,
    "pd_load_failure_count": 0,
    "pd_prefill_push_success_count": 0,
    "pd_prefill_push_failure_count": 0,
    "pd_decode_abort_count": 0,
    "pd_prefill_release_count": 0,
    "pd_prefill_skipped_push_count": 0,
}

PD_LIST_KEYS = (
    "pd_decode_wait_duration",
    "pd_decode_rdma_wait_duration",
    "pd_decode_prefill_http_submit_duration",
    "pd_load_blocks",
    "pd_prefill_push_duration",
    "pd_prefill_first_save_to_done_duration",
    "pd_prefill_wait_for_pushes_duration",
    "pd_prefill_push_blocks",
    "pd_prefill_push_bytes",
    "pd_prefill_push_gbps",
)


def _bind_metric_per_engine(
    prom_metrics: KVConnectorPromMetrics,
    metric: PromMetric,
) -> dict[int, PromMetric]:
    bind_method = getattr(prom_metrics, "make_per_engine", None)
    if callable(bind_method):
        return bind_method(metric)
    if create_metric_per_engine is None:
        raise RuntimeError(
            "Incompatible vLLM metrics API: missing both "
            "KVConnectorPromMetrics.make_per_engine and "
            "vllm.v1.metrics.utils.create_metric_per_engine"
        )
    return create_metric_per_engine(metric, prom_metrics.per_engine_labelvalues)


@dataclass
class PdKVConnectorStats(KVConnectorStats):
    """Stats payload consumed by vLLM KV connector metrics."""

    data: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        self.data = {**PD_STATS_KEYS, **{key: [] for key in PD_LIST_KEYS}}

    def aggregate(self, other: PdKVConnectorStats) -> PdKVConnectorStats:
        assert self.data is not None
        assert other.data is not None
        self.data["pd_decode_active_waits"] = other.data.get("pd_decode_active_waits", 0)
        self.data["pd_prefill_active_pushes"] = other.data.get("pd_prefill_active_pushes", 0)
        self.data["pd_prefill_inflight_push_tasks"] = other.data.get(
            "pd_prefill_inflight_push_tasks", 0
        )
        self.data["pd_prefill_inflight_finalize_tasks"] = other.data.get(
            "pd_prefill_inflight_finalize_tasks", 0
        )
        for key in PD_STATS_KEYS:
            if key.endswith("_count"):
                self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)
        for key in PD_LIST_KEYS:
            current = self.data.get(key, [])
            incoming = other.data.get(key, [])
            if isinstance(current, list) and isinstance(incoming, list):
                current.extend(incoming)
                self.data[key] = current
        return self

    def is_empty(self) -> bool:
        assert self.data is not None
        return all(self.data.get(key, 0) == 0 for key in PD_STATS_KEYS) and all(
            len(self.data.get(key, [])) == 0 for key in PD_LIST_KEYS
        )

    def clone_and_reset(self) -> PdKVConnectorStats:
        old = copy.deepcopy(self)
        self.reset()
        return old


class PdMetricsTracker:
    def __init__(self) -> None:
        self._stats = PdKVConnectorStats()
        self._lock = threading.Lock()

    def set_decode_active_waits(self, count: int) -> None:
        self._set_gauge("pd_decode_active_waits", count)

    def set_prefill_active_pushes(self, count: int) -> None:
        self._set_gauge("pd_prefill_active_pushes", count)

    def set_prefill_inflight_push_tasks(self, count: int) -> None:
        self._set_gauge("pd_prefill_inflight_push_tasks", count)

    def set_prefill_inflight_finalize_tasks(self, count: int) -> None:
        self._set_gauge("pd_prefill_inflight_finalize_tasks", count)

    def record_decode_wait(
        self,
        *,
        duration_s: float,
        rdma_wait_s: float | None,
        blocks: int,
        success: bool,
    ) -> None:
        with self._lock:
            data = self._stats.data
            assert data is not None
            data["pd_decode_wait_duration"].append(max(0.0, duration_s))
            data["pd_load_blocks"].append(max(0, blocks))
            if rdma_wait_s is not None:
                data["pd_decode_rdma_wait_duration"].append(max(0.0, rdma_wait_s))
            if success:
                data["pd_load_success_count"] += 1
            else:
                data["pd_load_failure_count"] += 1

    def record_prefill_http_submit(self, duration_s: float) -> None:
        self._append("pd_decode_prefill_http_submit_duration", max(0.0, duration_s))

    def record_decode_rdma_wait(self, duration_s: float) -> None:
        self._append("pd_decode_rdma_wait_duration", max(0.0, duration_s))

    def record_decode_abort(self) -> None:
        self._increment("pd_decode_abort_count")

    def record_prefill_push(
        self,
        *,
        duration_s: float,
        first_save_to_done_s: float | None,
        wait_for_pushes_s: float | None,
        blocks: int,
        bytes_total: int,
        success: bool,
        gbps: float | None = None,
    ) -> None:
        with self._lock:
            data = self._stats.data
            assert data is not None
            data["pd_prefill_push_duration"].append(max(0.0, duration_s))
            data["pd_prefill_push_blocks"].append(max(0, blocks))
            data["pd_prefill_push_bytes"].append(max(0, bytes_total))
            if gbps is not None:
                data["pd_prefill_push_gbps"].append(max(0.0, gbps))
            if first_save_to_done_s is not None:
                data["pd_prefill_first_save_to_done_duration"].append(
                    max(0.0, first_save_to_done_s)
                )
            if wait_for_pushes_s is not None:
                data["pd_prefill_wait_for_pushes_duration"].append(max(0.0, wait_for_pushes_s))
            if success:
                data["pd_prefill_push_success_count"] += 1
            else:
                data["pd_prefill_push_failure_count"] += 1

    def record_prefill_release(self) -> None:
        self._increment("pd_prefill_release_count")

    def record_prefill_skipped_push(self) -> None:
        self._increment("pd_prefill_skipped_push_count")

    def get_stats(self) -> PdKVConnectorStats:
        with self._lock:
            stats = self._stats.clone_and_reset()
            self._stats.data["pd_decode_active_waits"] = stats.data.get("pd_decode_active_waits", 0)
            self._stats.data["pd_prefill_active_pushes"] = stats.data.get(
                "pd_prefill_active_pushes", 0
            )
            self._stats.data["pd_prefill_inflight_push_tasks"] = stats.data.get(
                "pd_prefill_inflight_push_tasks", 0
            )
            self._stats.data["pd_prefill_inflight_finalize_tasks"] = stats.data.get(
                "pd_prefill_inflight_finalize_tasks", 0
            )
            return stats

    def _set_gauge(self, key: str, value: int) -> None:
        with self._lock:
            data = self._stats.data
            assert data is not None
            data[key] = max(0, int(value))

    def _increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            data = self._stats.data
            assert data is not None
            data[key] += amount

    def _append(self, key: str, value: float) -> None:
        with self._lock:
            data = self._stats.data
            assert data is not None
            data[key].append(value)


class PdPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        self.gauge_decode_active_waits = _bind_metric_per_engine(
            self,
            self._gauge_cls(
                name="vllm:pega_pd_decode_active_waits",
                documentation="Number of active P/D decode-side KV waits.",
                labelnames=labelnames,
            ),
        )
        self.gauge_prefill_active_pushes = _bind_metric_per_engine(
            self,
            self._gauge_cls(
                name="vllm:pega_pd_prefill_active_pushes",
                documentation="Number of active P/D prefill-side KV pushes.",
                labelnames=labelnames,
            ),
        )
        self.gauge_prefill_inflight_push_tasks = _bind_metric_per_engine(
            self,
            self._gauge_cls(
                name="vllm:pega_pd_prefill_inflight_push_tasks",
                documentation="Number of in-flight P/D RDMA push tasks.",
                labelnames=labelnames,
            ),
        )
        self.gauge_prefill_inflight_finalize_tasks = _bind_metric_per_engine(
            self,
            self._gauge_cls(
                name="vllm:pega_pd_prefill_inflight_finalize_tasks",
                documentation="Number of in-flight P/D RDMA push finalizer tasks.",
                labelnames=labelnames,
            ),
        )

        duration_buckets = build_buckets([1, 2, 4, 8], 100, -3)
        block_buckets = build_buckets([1, 2, 4, 8], 4096, 0)
        byte_buckets = build_buckets([1, 2, 4, 8], 1 << 34, 10)
        gbps_buckets = build_buckets([1, 2, 4, 8], 1024, -1)

        self.hist_decode_wait_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_decode_wait_duration_seconds",
                documentation="Duration from decode-side KV wait scheduling to completion.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_decode_rdma_wait_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_decode_rdma_wait_duration_seconds",
                documentation="Duration spent waiting for decode-side RDMA done IMM.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_decode_prefill_http_submit_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_decode_prefill_http_submit_duration_seconds",
                documentation="Duration to submit the decode-to-prefill HTTP trigger.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_load_blocks = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_load_blocks",
                documentation="Blocks in each decode-side P/D KV load.",
                buckets=block_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_push_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_push_duration_seconds",
                documentation="Duration from prefill-side push scheduling to done IMM.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_first_save_to_done_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_first_save_to_done_duration_seconds",
                documentation="Duration from first prefill KV save layer to done IMM.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_wait_for_pushes_duration = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_wait_for_pushes_duration_seconds",
                documentation="Duration spent waiting for prefill-side RDMA writes.",
                buckets=duration_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_push_blocks = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_push_blocks",
                documentation="Blocks in each prefill-side P/D KV push.",
                buckets=block_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_push_bytes = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_push_bytes",
                documentation="Bytes in each prefill-side P/D KV push.",
                buckets=byte_buckets,
                labelnames=labelnames,
            ),
        )
        self.hist_prefill_push_gbps = _bind_metric_per_engine(
            self,
            self._histogram_cls(
                name="vllm:pega_pd_prefill_push_gbps",
                documentation="Effective prefill-side P/D KV push throughput in GB/s.",
                buckets=gbps_buckets,
                labelnames=labelnames,
            ),
        )

        self.counter_load_success = self._counter("vllm:pega_pd_load_success_total", labelnames)
        self.counter_load_failure = self._counter("vllm:pega_pd_load_failure_total", labelnames)
        self.counter_prefill_push_success = self._counter(
            "vllm:pega_pd_prefill_push_success_total", labelnames
        )
        self.counter_prefill_push_failure = self._counter(
            "vllm:pega_pd_prefill_push_failure_total", labelnames
        )
        self.counter_decode_abort = self._counter("vllm:pega_pd_decode_abort_total", labelnames)
        self.counter_prefill_release = self._counter(
            "vllm:pega_pd_prefill_release_total", labelnames
        )
        self.counter_prefill_skipped_push = self._counter(
            "vllm:pega_pd_prefill_skipped_push_total", labelnames
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        self.gauge_decode_active_waits[engine_idx].set(
            transfer_stats_data.get("pd_decode_active_waits", 0)
        )
        self.gauge_prefill_active_pushes[engine_idx].set(
            transfer_stats_data.get("pd_prefill_active_pushes", 0)
        )
        self.gauge_prefill_inflight_push_tasks[engine_idx].set(
            transfer_stats_data.get("pd_prefill_inflight_push_tasks", 0)
        )
        self.gauge_prefill_inflight_finalize_tasks[engine_idx].set(
            transfer_stats_data.get("pd_prefill_inflight_finalize_tasks", 0)
        )

        histograms = (
            (self.hist_decode_wait_duration, "pd_decode_wait_duration"),
            (self.hist_decode_rdma_wait_duration, "pd_decode_rdma_wait_duration"),
            (
                self.hist_decode_prefill_http_submit_duration,
                "pd_decode_prefill_http_submit_duration",
            ),
            (self.hist_load_blocks, "pd_load_blocks"),
            (self.hist_prefill_push_duration, "pd_prefill_push_duration"),
            (
                self.hist_prefill_first_save_to_done_duration,
                "pd_prefill_first_save_to_done_duration",
            ),
            (
                self.hist_prefill_wait_for_pushes_duration,
                "pd_prefill_wait_for_pushes_duration",
            ),
            (self.hist_prefill_push_blocks, "pd_prefill_push_blocks"),
            (self.hist_prefill_push_bytes, "pd_prefill_push_bytes"),
            (self.hist_prefill_push_gbps, "pd_prefill_push_gbps"),
        )
        for metric, key in histograms:
            self._observe_hist(engine_idx, metric, transfer_stats_data, key)

        counters = (
            (self.counter_load_success, "pd_load_success_count"),
            (self.counter_load_failure, "pd_load_failure_count"),
            (self.counter_prefill_push_success, "pd_prefill_push_success_count"),
            (self.counter_prefill_push_failure, "pd_prefill_push_failure_count"),
            (self.counter_decode_abort, "pd_decode_abort_count"),
            (self.counter_prefill_release, "pd_prefill_release_count"),
            (self.counter_prefill_skipped_push, "pd_prefill_skipped_push_count"),
        )
        for metric, key in counters:
            self._inc_counter(engine_idx, metric, transfer_stats_data, key)

    def _counter(self, name: str, labelnames: list[str]) -> dict[int, PromMetric]:
        return _bind_metric_per_engine(
            self,
            self._counter_cls(
                name=name,
                documentation=name.replace("vllm:", "").replace("_", " "),
                labelnames=labelnames,
            ),
        )

    @staticmethod
    def _observe_hist(
        engine_idx: int,
        metric: dict[int, Any],
        data: dict[str, Any],
        key: str,
    ) -> None:
        for value in data.get(key, []):
            metric[engine_idx].observe(value)

    @staticmethod
    def _inc_counter(
        engine_idx: int,
        metric: dict[int, Any],
        data: dict[str, Any],
        key: str,
    ) -> None:
        value = data.get(key, 0)
        if value > 0:
            metric[engine_idx].inc(value)


__all__ = [
    "PdKVConnectorStats",
    "PdMetricsTracker",
    "PdPromMetrics",
]
