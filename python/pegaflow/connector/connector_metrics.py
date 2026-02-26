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


class PrefetchTracker:
    """Track prefetch queue depth, duration and blocks for metrics.

    This class tracks:
    - pending prefetch count (for gauge metric)
    - prefetch duration history (for histogram metric)
    - prefetch blocks history (for histogram metric)
    """

    def __init__(self):
        self._pending_prefetches: int = 0
        self._prefetch_durations: list[float] = []  # in milliseconds
        self._prefetch_blocks: list[int] = []

    def on_prefetch_start(self) -> None:
        """Called when a request enters prefetch loading state."""
        self._pending_prefetches += 1

    def on_prefetch_complete(self, duration_ms: float, hit_blocks: int) -> None:
        """Called when a request's prefetch completes.

        Args:
            duration_ms: Time spent waiting for prefetch in milliseconds.
            hit_blocks: Number of blocks that were prefetched.
        """
        self._pending_prefetches = max(0, self._pending_prefetches - 1)
        self._prefetch_durations.append(duration_ms)
        self._prefetch_blocks.append(hit_blocks)

    @property
    def pending_prefetches(self) -> int:
        """Current number of requests waiting for prefetch."""
        return self._pending_prefetches

    def get_stats(self) -> dict:
        """Get stats for Prometheus metrics and reset history.

        Returns:
            Dictionary with pending_prefetches (gauge), prefetch_duration and
            prefetch_blocks (histogram data).
        """
        durations = self._prefetch_durations
        blocks = self._prefetch_blocks
        self._prefetch_durations = []  # Reset for next interval
        self._prefetch_blocks = []
        return {
            "pending_prefetches": self._pending_prefetches,
            "prefetch_duration": durations,
            "prefetch_blocks": blocks,
        }


@dataclass
class PegaKVConnectorStats(KVConnectorStats):
    """Stats for PegaFlow KV connector.

    Metrics collected:
    - Scheduler-side (gauge):
        - pending_prefetches: number of requests waiting for SSD prefetch
    - Scheduler-side (counter):
        - bypass_count: requests that bypassed cache lookup
    - Scheduler-side (histogram):
        - prefetch_duration: prefetch operation duration in milliseconds
        - prefetch_blocks: number of blocks per prefetch operation
    - Worker-side (histogram):
        - load_duration: load operation duration in seconds
        - load_blocks: number of blocks per load operation
        - save_duration: save operation duration in seconds
        - save_blocks: number of blocks per save operation
    - Worker-side (counter):
        - load_success_count: successful load operations
        - load_failure_count: failed load operations
        - save_success_count: successful save operations
        - save_failure_count: failed save operations
        - save_dropped_count: save operations dropped due to queue limit
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict = {
            # Scheduler-side gauges
            "pending_prefetches": 0,
            # Scheduler-side counters
            "bypass_count": 0,
            # Scheduler-side prefetch histogram data
            "prefetch_duration": [],  # list[float] in ms
            "prefetch_blocks": [],  # list[int]
            # Worker-side gauges
            "pending_save_requests": 0,
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
            "save_dropped_count": 0,
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
        self.data["pending_prefetches"] = other.data.get("pending_prefetches", 0)
        self.data["pending_save_requests"] = other.data.get("pending_save_requests", 0)

        # Counter-like metrics: sum them
        for key in [
            "bypass_count",
            "load_success_count",
            "load_failure_count",
            "save_success_count",
            "save_failure_count",
            "save_dropped_count",
        ]:
            self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)

        # List metrics: extend
        for key in [
            "prefetch_duration",
            "prefetch_blocks",
            "load_duration",
            "load_blocks",
            "save_duration",
            "save_blocks",
        ]:
            self_list = self.data.get(key, [])
            other_list = other.data.get(key, [])
            if isinstance(self_list, list) and isinstance(other_list, list):
                self_list.extend(other_list)
                self.data[key] = self_list

        return self

    def reduce(self) -> dict[str, int | float | str]:
        """Reduce stats to summary values for logging."""
        result: dict[str, int | float | str] = {
            "pending_prefetches": self.data.get("pending_prefetches", 0),
            "pending_save_requests": self.data.get("pending_save_requests", 0),
            "bypass_count": self.data.get("bypass_count", 0),
        }

        # Prefetch stats (scheduler-side)
        prefetch_durations = self.data.get("prefetch_duration", [])
        prefetch_blocks = self.data.get("prefetch_blocks", [])
        num_prefetches = len(prefetch_durations)
        if num_prefetches > 0:
            result["num_prefetches"] = num_prefetches
            # prefetch_duration is in ms, keep as ms for consistency
            result["avg_prefetch_duration_ms"] = round(sum(prefetch_durations) / num_prefetches, 3)
            result["total_prefetch_blocks"] = sum(prefetch_blocks)
            result["avg_prefetch_blocks"] = round(sum(prefetch_blocks) / num_prefetches, 1)

        # Load stats (worker-side)
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

        # Save stats (worker-side)
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
        result["save_dropped_count"] = self.data.get("save_dropped_count", 0)

        return result

    def is_empty(self) -> bool:
        return (
            self.data.get("pending_prefetches", 0) == 0
            and self.data.get("bypass_count", 0) == 0
            and len(self.data.get("prefetch_duration", [])) == 0
            and len(self.data.get("load_duration", [])) == 0
            and self.data.get("load_success_count", 0) == 0
            and self.data.get("load_failure_count", 0) == 0
            and len(self.data.get("save_duration", [])) == 0
            and self.data.get("save_success_count", 0) == 0
            and self.data.get("save_failure_count", 0) == 0
            and self.data.get("save_dropped_count", 0) == 0
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
        gauge_pending_prefetches = self._gauge_cls(
            name="vllm:pega_pending_prefetches",
            documentation="Number of requests waiting for SSD prefetch to complete.",
            labelnames=labelnames,
        )
        self.gauge_pending_prefetches = self.make_per_engine(gauge_pending_prefetches)

        # Gauge metrics for worker-side state
        gauge_pending_save_requests = self._gauge_cls(
            name="vllm:pega_pending_save_requests",
            documentation="Number of requests with pending save operations.",
            labelnames=labelnames,
        )
        self.gauge_pending_save_requests = self.make_per_engine(gauge_pending_save_requests)

        # Counter for bypass events (scheduler-side)
        counter_bypass = self._counter_cls(
            name="vllm:pega_bypass_total",
            documentation="Number of requests that bypassed cache lookup due to short request.",
            labelnames=labelnames,
        )
        self.counter_bypass = self.make_per_engine(counter_bypass)

        # Histogram for prefetch operations (scheduler-side)
        # Optimized for fast SSD: typical range 10-500ms
        # Buckets: 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000 ms
        prefetch_buckets = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        histogram_prefetch_duration = self._histogram_cls(
            name="vllm:pega_prefetch_duration_seconds",
            documentation="Histogram of prefetch duration in seconds.",
            buckets=prefetch_buckets,
            labelnames=labelnames,
        )
        self.histogram_prefetch_duration = self.make_per_engine(histogram_prefetch_duration)

        blocks_buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        histogram_prefetch_blocks = self._histogram_cls(
            name="vllm:pega_prefetch_blocks",
            documentation="Histogram of blocks per prefetch operation.",
            buckets=blocks_buckets,
            labelnames=labelnames,
        )
        self.histogram_prefetch_blocks = self.make_per_engine(histogram_prefetch_blocks)

        # Histogram for load operations (worker-side)
        # Optimized for fast SSD: typical range 1-50ms
        # Buckets: 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100 ms
        duration_buckets = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]
        histogram_load_duration = self._histogram_cls(
            name="vllm:pega_load_duration_seconds",
            documentation="Histogram of KV cache load duration in seconds.",
            buckets=duration_buckets,
            labelnames=labelnames,
        )
        self.histogram_load_duration = self.make_per_engine(histogram_load_duration)

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

        counter_save_dropped = self._counter_cls(
            name="vllm:pega_save_dropped_total",
            documentation="Number of save operations dropped due to queue limit.",
            labelnames=labelnames,
        )
        self.counter_save_dropped = self.make_per_engine(counter_save_dropped)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record stats to Prometheus metrics."""
        # Gauge metrics (scheduler-side)
        self.gauge_pending_prefetches[engine_idx].set(
            transfer_stats_data.get("pending_prefetches", 0)
        )

        # Gauge metrics (worker-side)
        self.gauge_pending_save_requests[engine_idx].set(
            transfer_stats_data.get("pending_save_requests", 0)
        )

        # Counter: bypass (scheduler-side)
        bypass_count = transfer_stats_data.get("bypass_count", 0)
        if bypass_count > 0:
            self.counter_bypass[engine_idx].inc(bypass_count)

        # Histogram: prefetch duration (scheduler-side)
        # prefetch_duration is in ms, convert to seconds for Prometheus
        for duration_ms in transfer_stats_data.get("prefetch_duration", []):
            self.histogram_prefetch_duration[engine_idx].observe(duration_ms / 1000.0)
        for blocks in transfer_stats_data.get("prefetch_blocks", []):
            self.histogram_prefetch_blocks[engine_idx].observe(blocks)

        # Histogram: load duration (worker-side)
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

        # Histogram: save duration
        for duration in transfer_stats_data.get("save_duration", []):
            self.histogram_save_duration[engine_idx].observe(duration)
        for blocks in transfer_stats_data.get("save_blocks", []):
            self.histogram_save_blocks[engine_idx].observe(blocks)

        # Counter: save success/failure/dropped
        save_success = transfer_stats_data.get("save_success_count", 0)
        if save_success > 0:
            self.counter_save_success[engine_idx].inc(save_success)
        save_failure = transfer_stats_data.get("save_failure_count", 0)
        if save_failure > 0:
            self.counter_save_failure[engine_idx].inc(save_failure)
        save_dropped = transfer_stats_data.get("save_dropped_count", 0)
        if save_dropped > 0:
            self.counter_save_dropped[engine_idx].inc(save_dropped)


__all__ = [
    "PrefetchTracker",
    "PegaKVConnectorStats",
    "PegaPromMetrics",
]
