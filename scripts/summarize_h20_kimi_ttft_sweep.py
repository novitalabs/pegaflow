#!/usr/bin/env python3
"""Build the H20 Kimi fixed-shape TTFT sweep table."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

INPUT_RE = re.compile(r"-in(?P<input_len>\d+)-")
NIC_RE = re.compile(
    r"^(?P<nic>\S+) .*avg_xmit_gbps=(?P<avg_xmit>[0-9.]+) "
    r"peak_xmit_gbps=(?P<peak_xmit>[0-9.]+) "
    r"avg_rcv_gbps=(?P<avg_rcv>[0-9.]+) "
    r"peak_rcv_gbps=(?P<peak_rcv>[0-9.]+)"
)


@dataclass(frozen=True)
class BenchResult:
    input_len: int
    mean_ttft_ms: float
    p99_ttft_ms: float
    completed: int
    failed: int
    request_throughput: float


@dataclass(frozen=True)
class NicResult:
    avg_gbps_per_nic: float | None
    peak_gbps_per_nic: float | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline-prefix", default="kimi-baseline-fixed32k")
    parser.add_argument("--proxy-prefix", default="kimi-proxy-fixed32k")
    args = parser.parse_args()

    baseline = load_results(args.result_dir, args.baseline_prefix)
    proxy = load_results(args.result_dir, args.proxy_prefix)

    print(
        "| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | "
        "delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | "
        "baseline_success | proxy_success | baseline_req_s | proxy_req_s | "
        "proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |"
    )
    print(
        "|-----------|-----------------------|-----------------------|----------|"
        "-----------|-----------------------|-------------------|"
        "------------------|---------------|----------------|-------------|"
        "-----------------------------|------------------------------|-------|"
    )
    for input_len in sorted(set(baseline) | set(proxy)):
        base = baseline.get(input_len)
        pd = proxy.get(input_len)
        nic = load_nic_result(args.result_dir, args.proxy_prefix, input_len)
        notes = []
        if base is None:
            notes.append("missing baseline")
        if pd is None:
            notes.append("missing proxy")
        if base is not None and base.failed:
            notes.append(f"baseline failed={base.failed}")
        if pd is not None and pd.failed:
            notes.append(f"proxy failed={pd.failed}")
        print(
            "| "
            + " | ".join(
                [
                    str(input_len),
                    fmt(base.mean_ttft_ms if base else None),
                    fmt(pd.mean_ttft_ms if pd else None),
                    fmt(delta_ms(base, pd)),
                    fmt(delta_pct(base, pd)),
                    fmt(base.p99_ttft_ms if base else None),
                    fmt(pd.p99_ttft_ms if pd else None),
                    success(base),
                    success(pd),
                    fmt(base.request_throughput if base else None),
                    fmt(pd.request_throughput if pd else None),
                    fmt(nic.avg_gbps_per_nic),
                    fmt(nic.peak_gbps_per_nic),
                    ", ".join(notes) if notes else "fixed 32k, c1",
                ]
            )
            + " |"
        )


def load_results(result_dir: Path, prefix: str) -> dict[int, BenchResult]:
    results: dict[int, BenchResult] = {}
    for path in result_dir.glob(f"{prefix}-in*.json"):
        input_len = input_len_from_name(path)
        if input_len is None:
            continue
        with path.open() as f:
            data = json.load(f)
        results[input_len] = BenchResult(
            input_len=input_len,
            mean_ttft_ms=float(data["mean_ttft_ms"]),
            p99_ttft_ms=float(data["p99_ttft_ms"]),
            completed=int(data["completed"]),
            failed=int(data["failed"]),
            request_throughput=float(data["request_throughput"]),
        )
    return results


def load_nic_result(result_dir: Path, proxy_prefix: str, input_len: int) -> NicResult:
    summaries = list(result_dir.glob(f"{proxy_prefix}-in{input_len}-*-nic-summary.txt"))
    by_nic: dict[str, tuple[float, float]] = {}
    for path in summaries:
        for line in path.read_text().splitlines():
            match = NIC_RE.match(line)
            if match is None:
                continue
            nic = match.group("nic")
            active_avg = max(
                float(match.group("avg_xmit")), float(match.group("avg_rcv"))
            )
            active_peak = max(
                float(match.group("peak_xmit")), float(match.group("peak_rcv"))
            )
            old_avg, old_peak = by_nic.get(nic, (0.0, 0.0))
            by_nic[nic] = (max(old_avg, active_avg), max(old_peak, active_peak))
    if not by_nic:
        return NicResult(None, None)
    return NicResult(
        avg_gbps_per_nic=sum(avg for avg, _peak in by_nic.values()) / len(by_nic),
        peak_gbps_per_nic=max(peak for _avg, peak in by_nic.values()),
    )


def input_len_from_name(path: Path) -> int | None:
    match = INPUT_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("input_len"))


def delta_ms(base: BenchResult | None, pd: BenchResult | None) -> float | None:
    if base is None or pd is None:
        return None
    return pd.mean_ttft_ms - base.mean_ttft_ms


def delta_pct(base: BenchResult | None, pd: BenchResult | None) -> float | None:
    if base is None or pd is None or base.mean_ttft_ms == 0:
        return None
    return (pd.mean_ttft_ms - base.mean_ttft_ms) / base.mean_ttft_ms * 100


def fmt(value: float | None) -> str:
    if value is None:
        return "TBD"
    return f"{value:.2f}"


def success(result: BenchResult | None) -> str:
    if result is None:
        return "TBD"
    return f"{result.completed}/{result.completed + result.failed}"


if __name__ == "__main__":
    main()
