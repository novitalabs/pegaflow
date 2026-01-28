"""
Metrics collection and Prometheus integration for PegaFlow KV connector.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class PegaKVConnectorStats(KVConnectorStats):
    """Stats for PegaFlow KV connector.

    Metrics collected:
    - Scheduler-side (gauge):
        - pending_load_requests: number of requests waiting for remote KV
        - pending_load_blocks: total blocks waiting to be loaded
    - Scheduler-side (counter):
        - fallback_count: requests that bypassed cache lookup
    - Worker-side (list for histogram):
        - save_duration: save operation duration in seconds
        - save_blocks: number of blocks per save operation
    - Worker-side (counter):
        - save_success_count: successful save operations
        - save_failure_count: failed save operations
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict = {
            # Scheduler-side gauges
            "pending_load_requests": 0,
            "pending_load_blocks": 0,
            # Scheduler-side counters
            "fallback_count": 0,
            # Worker-side lists (for histogram)
            "load_duration": [],
            "load_blocks": [],
            "save_duration": [],
            "save_blocks": [],
            # Worker-side counters
            "load_success_count": 0,
            "load_failure_count": 0,
            "save_success_count": 0,
            "save_failure_count": 0,
        }

    def record_load(self, duration_seconds: float, num_blocks: int, success: bool):
        """Record a load operation."""
        self.data["load_duration"].append(duration_seconds)
        self.data["load_blocks"].append(num_blocks)
        if success:
            self.data["load_success_count"] += 1
        else:
            self.data["load_failure_count"] += 1

    def record_save(self, duration_seconds: float, num_blocks: int, success: bool):
        """Record a save operation."""
        self.data["save_duration"].append(duration_seconds)
        self.data["save_blocks"].append(num_blocks)
        if success:
            self.data["save_success_count"] += 1
        else:
            self.data["save_failure_count"] += 1

    def aggregate(self, other: "PegaKVConnectorStats") -> "PegaKVConnectorStats":
        # Gauge-like metrics: take the latest value
        self.data["pending_load_requests"] = other.data.get("pending_load_requests", 0)
        self.data["pending_load_blocks"] = other.data.get("pending_load_blocks", 0)

        # Counter-like metrics: sum them
        for key in [
            "fallback_count",
            "load_success_count",
            "load_failure_count",
            "save_success_count",
            "save_failure_count",
        ]:
            self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)

        # List metrics: extend
        for key in ["load_duration", "load_blocks", "save_duration", "save_blocks"]:
            self_list = self.data.get(key, [])
            other_list = other.data.get(key, [])
            if isinstance(self_list, list) and isinstance(other_list, list):
                self_list.extend(other_list)
                self.data[key] = self_list

        return self

    def reduce(self) -> dict[str, int | float]:
        """Reduce stats to summary values for logging."""
        result: dict[str, int | float] = {
            "pending_load_requests": self.data.get("pending_load_requests", 0),
            "pending_load_blocks": self.data.get("pending_load_blocks", 0),
            "fallback_count": self.data.get("fallback_count", 0),
        }

        # Load stats
        load_durations = self.data.get("load_duration", [])
        load_blocks = self.data.get("load_blocks", [])
        num_loads = len(load_durations)
        if num_loads > 0:
            result["num_loads"] = num_loads
            result["avg_load_duration_ms"] = round(sum(load_durations) / num_loads * 1000, 3)
            result["total_load_blocks"] = sum(load_blocks)
            result["avg_load_blocks"] = round(sum(load_blocks) / num_loads, 1)
        result["load_success_count"] = self.data.get("load_success_count", 0)
        result["load_failure_count"] = self.data.get("load_failure_count", 0)

        # Save stats
        save_durations = self.data.get("save_duration", [])
        save_blocks = self.data.get("save_blocks", [])
        num_saves = len(save_durations)
        if num_saves > 0:
            result["num_saves"] = num_saves
            result["avg_save_duration_ms"] = round(sum(save_durations) / num_saves * 1000, 3)
            result["total_save_blocks"] = sum(save_blocks)
            result["avg_save_blocks"] = round(sum(save_blocks) / num_saves, 1)
        result["save_success_count"] = self.data.get("save_success_count", 0)
        result["save_failure_count"] = self.data.get("save_failure_count", 0)

        return result

    def is_empty(self) -> bool:
        return (
            self.data.get("pending_load_requests", 0) == 0
            and self.data.get("pending_load_blocks", 0) == 0
            and self.data.get("fallback_count", 0) == 0
            and len(self.data.get("load_duration", [])) == 0
            and self.data.get("load_success_count", 0) == 0
            and self.data.get("load_failure_count", 0) == 0
            and len(self.data.get("save_duration", [])) == 0
            and self.data.get("save_success_count", 0) == 0
            and self.data.get("save_failure_count", 0) == 0
        )

    def clone_and_reset(self) -> "PegaKVConnectorStats":
        """Clone current stats and reset for next interval."""
        import copy

        old = copy.deepcopy(self)
        self.reset()
        return old


class PegaPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for PegaFlow KV connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        # Gauge metrics for scheduler-side state
        gauge_pending_load_requests = self._gauge_cls(
            name="vllm:pega_pending_load_requests",
            documentation="Number of requests waiting for remote KV cache load.",
            labelnames=labelnames,
        )
        self.gauge_pending_load_requests = self.make_per_engine(gauge_pending_load_requests)

        gauge_pending_load_blocks = self._gauge_cls(
            name="vllm:pega_pending_load_blocks",
            documentation="Total blocks waiting to be loaded from remote KV cache.",
            labelnames=labelnames,
        )
        self.gauge_pending_load_blocks = self.make_per_engine(gauge_pending_load_blocks)

        # Counter for fallback events
        counter_fallback = self._counter_cls(
            name="vllm:pega_fallback_total",
            documentation="Number of requests that bypassed cache lookup due to high pending loads.",
            labelnames=labelnames,
        )
        self.counter_fallback = self.make_per_engine(counter_fallback)

        # Histogram for load operations
        duration_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        histogram_load_duration = self._histogram_cls(
            name="vllm:pega_load_duration_seconds",
            documentation="Histogram of KV cache load duration in seconds.",
            buckets=duration_buckets,
            labelnames=labelnames,
        )
        self.histogram_load_duration = self.make_per_engine(histogram_load_duration)

        blocks_buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        histogram_load_blocks = self._histogram_cls(
            name="vllm:pega_load_blocks",
            documentation="Histogram of blocks per KV cache load operation.",
            buckets=blocks_buckets,
            labelnames=labelnames,
        )
        self.histogram_load_blocks = self.make_per_engine(histogram_load_blocks)

        counter_load_success = self._counter_cls(
            name="vllm:pega_load_success_total",
            documentation="Number of successful KV cache load operations.",
            labelnames=labelnames,
        )
        self.counter_load_success = self.make_per_engine(counter_load_success)

        counter_load_failure = self._counter_cls(
            name="vllm:pega_load_failure_total",
            documentation="Number of failed KV cache load operations.",
            labelnames=labelnames,
        )
        self.counter_load_failure = self.make_per_engine(counter_load_failure)

        # Histogram for save operations
        histogram_save_duration = self._histogram_cls(
            name="vllm:pega_save_duration_seconds",
            documentation="Histogram of KV cache save duration in seconds.",
            buckets=duration_buckets,
            labelnames=labelnames,
        )
        self.histogram_save_duration = self.make_per_engine(histogram_save_duration)

        histogram_save_blocks = self._histogram_cls(
            name="vllm:pega_save_blocks",
            documentation="Histogram of blocks per KV cache save operation.",
            buckets=blocks_buckets,
            labelnames=labelnames,
        )
        self.histogram_save_blocks = self.make_per_engine(histogram_save_blocks)

        counter_save_success = self._counter_cls(
            name="vllm:pega_save_success_total",
            documentation="Number of successful KV cache save operations.",
            labelnames=labelnames,
        )
        self.counter_save_success = self.make_per_engine(counter_save_success)

        counter_save_failure = self._counter_cls(
            name="vllm:pega_save_failure_total",
            documentation="Number of failed KV cache save operations.",
            labelnames=labelnames,
        )
        self.counter_save_failure = self.make_per_engine(counter_save_failure)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record stats to Prometheus metrics."""
        # Gauge metrics
        self.gauge_pending_load_requests[engine_idx].set(
            transfer_stats_data.get("pending_load_requests", 0)
        )
        self.gauge_pending_load_blocks[engine_idx].set(
            transfer_stats_data.get("pending_load_blocks", 0)
        )

        # Counter: fallback
        fallback_count = transfer_stats_data.get("fallback_count", 0)
        if fallback_count > 0:
            self.counter_fallback[engine_idx].inc(fallback_count)

        # Histogram: load duration and blocks
        for duration in transfer_stats_data.get("load_duration", []):
            self.histogram_load_duration[engine_idx].observe(duration)
        for blocks in transfer_stats_data.get("load_blocks", []):
            self.histogram_load_blocks[engine_idx].observe(blocks)

        # Counter: load success/failure
        load_success = transfer_stats_data.get("load_success_count", 0)
        if load_success > 0:
            self.counter_load_success[engine_idx].inc(load_success)
        load_failure = transfer_stats_data.get("load_failure_count", 0)
        if load_failure > 0:
            self.counter_load_failure[engine_idx].inc(load_failure)

        # Histogram: save duration and blocks
        for duration in transfer_stats_data.get("save_duration", []):
            self.histogram_save_duration[engine_idx].observe(duration)
        for blocks in transfer_stats_data.get("save_blocks", []):
            self.histogram_save_blocks[engine_idx].observe(blocks)

        # Counter: save success/failure
        save_success = transfer_stats_data.get("save_success_count", 0)
        if save_success > 0:
            self.counter_save_success[engine_idx].inc(save_success)
        save_failure = transfer_stats_data.get("save_failure_count", 0)
        if save_failure > 0:
            self.counter_save_failure[engine_idx].inc(save_failure)


__all__ = [
    "PegaKVConnectorStats",
    "PegaPromMetrics",
]
