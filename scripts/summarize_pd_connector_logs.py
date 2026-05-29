#!/usr/bin/env python3
"""Summarize PdConnector and PdProxy latency logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

KEY_VALUE_RE = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<value>\S+)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
BASE_REQUEST_RE = re.compile(r"(pd-[0-9a-fA-F]{32})")


@dataclass(frozen=True)
class EventSpec:
    name: str
    markers: tuple[str, ...]
    metrics: tuple[str, ...]

    def matches(self, line: str) -> bool:
        return all(marker in line for marker in self.markers)


@dataclass(frozen=True)
class Record:
    event: str
    source: Path
    line_no: int
    values: dict[str, str]
    numbers: dict[str, float]
    base_request: str | None


@dataclass(frozen=True)
class MetricSpec:
    column: str
    event: str
    metric: str
    reducer: Callable[[list[float]], float]


EVENTS = (
    EventSpec(
        name="proxy_first_chunk",
        markers=("[PdProxy]", "first stream chunk"),
        metrics=("ttft_ms", "bytes"),
    ),
    EventSpec(
        name="p_all_chunks_done",
        markers=("[PdConnector] P all chunks done",),
        metrics=(
            "schedule_to_save_ms",
            "forward_ms",
            "gbps",
            "link_gbps",
            "ready_link_util_pct",
        ),
    ),
    EventSpec(
        name="p_rdma_done",
        markers=("[PdConnector] P RDMA done",),
        metrics=(
            "save_to_imm_ms",
            "schedule_to_imm_ms",
            "wait_sender_ms",
            "wait_writes_ms",
            "imm_ms",
            "save_gbps",
            "tail_gbps",
            "ready_window_gbps",
            "link_gbps",
            "ready_link_util_pct",
            "push_queue_p95_ms",
            "push_event_p95_ms",
            "push_native_avg_ms",
            "push_native_p95_ms",
        ),
    ),
    EventSpec(
        name="d_wait_queued",
        markers=("[PdConnector] D RDMA wait queued",),
        metrics=("queue_depth", "workers", "blocks"),
    ),
    EventSpec(
        name="d_rdma_done",
        markers=("[PdConnector] D received RDMA done",),
        metrics=("queue_wait_ms", "wait_ms", "blocks"),
    ),
    EventSpec(
        name="d_worker_finished",
        markers=("[PdConnector] D worker finished_recving",),
        metrics=("count", "remaining_wait_before"),
    ),
    EventSpec(
        name="d_scheduler_finished",
        markers=("[PdConnector] scheduler finished recving",),
        metrics=(
            "proxy_to_finished_ms",
            "matched_to_finished_ms",
            "wait_to_finished_ms",
            "wait_ms",
        ),
    ),
)

TIMELINE_METRICS = (
    MetricSpec("proxy_ttft_ms", "proxy_first_chunk", "ttft_ms", max),
    MetricSpec("p_forward_ms", "p_all_chunks_done", "forward_ms", max),
    MetricSpec("p_save_to_imm_ms", "p_rdma_done", "save_to_imm_ms", max),
    MetricSpec("p_wait_sender_ms", "p_rdma_done", "wait_sender_ms", max),
    MetricSpec("p_wait_writes_ms", "p_rdma_done", "wait_writes_ms", max),
    MetricSpec("p_ready_window_gbps", "p_rdma_done", "ready_window_gbps", mean),
    MetricSpec("p_ready_link_util_pct", "p_rdma_done", "ready_link_util_pct", mean),
    MetricSpec("d_queue_wait_ms", "d_rdma_done", "queue_wait_ms", max),
    MetricSpec("d_wait_ms", "d_rdma_done", "wait_ms", max),
    MetricSpec(
        "sched_proxy_to_finished_ms",
        "d_scheduler_finished",
        "proxy_to_finished_ms",
        max,
    ),
    MetricSpec(
        "sched_wait_to_finished_ms",
        "d_scheduler_finished",
        "wait_to_finished_ms",
        max,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize PdConnector/PdProxy timing fields from logs."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--top-requests", type=int, default=20)
    args = parser.parse_args()

    records = list(load_records(args.logs))
    print("# PD Connector Log Summary")
    print()
    print("## Inputs")
    for path in args.logs:
        print(f"- {path}")
    print()

    print_event_counts(records)
    print_metric_summary(records)
    print_latency_ranking(records)
    print_request_timeline(records, args.top_requests)


def load_records(paths: Iterable[Path]) -> Iterable[Record]:
    for path in paths:
        with path.open(errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                spec = classify(line)
                if spec is None:
                    continue
                values = parse_key_values(line)
                numbers = {
                    key: float(value)
                    for key, value in values.items()
                    if NUMBER_RE.match(value)
                }
                yield Record(
                    event=spec.name,
                    source=path,
                    line_no=line_no,
                    values=values,
                    numbers=numbers,
                    base_request=base_request(values, line),
                )


def classify(line: str) -> EventSpec | None:
    for spec in EVENTS:
        if spec.matches(line):
            return spec
    return None


def parse_key_values(line: str) -> dict[str, str]:
    values = {}
    for match in KEY_VALUE_RE.finditer(line):
        values[match.group("key")] = match.group("value").rstrip(",")
    return values


def base_request(values: dict[str, str], line: str) -> str | None:
    for key in ("request", "target_req", "remote_req", "done_req", "req"):
        req = values.get(key)
        if req is None:
            continue
        match = BASE_REQUEST_RE.search(req)
        if match is not None:
            return match.group(1)
    match = BASE_REQUEST_RE.search(line)
    if match is None:
        return None
    return match.group(1)


def print_event_counts(records: list[Record]) -> None:
    counts = defaultdict(int)
    for record in records:
        counts[record.event] += 1

    print("## Event Counts")
    print("| event | count |")
    print("|-------|------:|")
    for spec in EVENTS:
        print(f"| {spec.name} | {counts[spec.name]} |")
    print()


def print_metric_summary(records: list[Record]) -> None:
    by_event = defaultdict(list)
    for record in records:
        by_event[record.event].append(record)

    print("## Metric Summary")
    for spec in EVENTS:
        event_records = by_event[spec.name]
        if not event_records:
            continue
        rows = []
        for metric in spec.metrics:
            values = [
                record.numbers[metric]
                for record in event_records
                if metric in record.numbers
            ]
            if values:
                rows.append((metric, describe(values)))
        if not rows:
            continue
        print(f"### {spec.name}")
        print("| metric | count | mean | p50 | p95 | min | max |")
        print("|--------|------:|-----:|----:|----:|----:|----:|")
        for metric, stats in rows:
            print(
                f"| {metric} | {stats['count']} | {fmt(stats['mean'])} | "
                f"{fmt(stats['p50'])} | {fmt(stats['p95'])} | "
                f"{fmt(stats['min'])} | {fmt(stats['max'])} |"
            )
        print()


def print_latency_ranking(records: list[Record]) -> None:
    rows = []
    for spec in EVENTS:
        event_records = [record for record in records if record.event == spec.name]
        if not event_records:
            continue
        for metric in spec.metrics:
            if not metric.endswith("_ms"):
                continue
            values = [
                record.numbers[metric]
                for record in event_records
                if metric in record.numbers
            ]
            if values:
                stats = describe(values)
                rows.append((stats["mean"], spec.name, metric, stats))

    print("## Largest Mean Latency Metrics")
    print("| event | metric | count | mean | p95 | max |")
    print("|-------|--------|------:|-----:|----:|----:|")
    for _mean, event, metric, stats in sorted(rows, reverse=True)[:12]:
        print(
            f"| {event} | {metric} | {stats['count']} | {fmt(stats['mean'])} | "
            f"{fmt(stats['p95'])} | {fmt(stats['max'])} |"
        )
    print()


def print_request_timeline(records: list[Record], limit: int) -> None:
    by_request = defaultdict(list)
    for record in records:
        if record.base_request is not None:
            by_request[record.base_request].append(record)

    rows = []
    for req, req_records in by_request.items():
        row = {"request": req}
        for metric in TIMELINE_METRICS:
            values = [
                record.numbers[metric.metric]
                for record in req_records
                if record.event == metric.event and metric.metric in record.numbers
            ]
            if values:
                row[metric.column] = metric.reducer(values)
        rows.append(row)

    rows.sort(key=timeline_sort_key, reverse=True)
    if limit >= 0:
        rows = rows[:limit]

    print("## Request Timeline")
    print(
        "| request | proxy_ttft_ms | p_forward_ms | p_save_to_imm_ms | "
        "p_wait_sender_ms | p_wait_writes_ms | p_ready_window_gbps | "
        "p_ready_link_util_pct | d_queue_wait_ms | d_wait_ms | "
        "sched_proxy_to_finished_ms | sched_wait_to_finished_ms |"
    )
    print(
        "|---------|--------------:|-------------:|-----------------:|"
        "-----------------:|----------------:|--------------------:|"
        "----------------------:|----------------:|----------:|"
        "---------------------------:|--------------------------:|"
    )
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["request"]),
                    fmt(row.get("proxy_ttft_ms")),
                    fmt(row.get("p_forward_ms")),
                    fmt(row.get("p_save_to_imm_ms")),
                    fmt(row.get("p_wait_sender_ms")),
                    fmt(row.get("p_wait_writes_ms")),
                    fmt(row.get("p_ready_window_gbps")),
                    fmt(row.get("p_ready_link_util_pct")),
                    fmt(row.get("d_queue_wait_ms")),
                    fmt(row.get("d_wait_ms")),
                    fmt(row.get("sched_proxy_to_finished_ms")),
                    fmt(row.get("sched_wait_to_finished_ms")),
                ]
            )
            + " |"
        )
    print()


def timeline_sort_key(row: dict[str, float | str]) -> tuple[float, float, float]:
    return (
        float(row.get("proxy_ttft_ms", -1.0)),
        float(row.get("d_wait_ms", -1.0)),
        float(row.get("p_save_to_imm_ms", -1.0)),
    )


def describe(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def percentile(ordered_values: list[float], quantile: float) -> float:
    idx = round((len(ordered_values) - 1) * quantile)
    return ordered_values[idx]


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
