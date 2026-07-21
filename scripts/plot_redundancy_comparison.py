#!/usr/bin/env python3

"""Render baseline/feature MetaServer redundancy comparison charts.

Usage:
    PLOT_ROOT=/path/to/artifacts \
    PLOT_TITLE="2P2D redundancy: baseline vs feature" \
    PLOT_OUTPUT_STEM=redundancy_comparison \
    python3 scripts/plot_redundancy_comparison.py

The root directory must contain ``baseline`` and ``feature`` directories with
``redundancy_timeseries.json``, ``metrics.json``, and ``bench-tail.log``.
The owner distribution assumes a four-server topology, so ``owners >= 4`` is
equivalent to exactly four owners.
"""

import json
import os
import re
from pathlib import Path

if "PLOT_ROOT" not in os.environ:
    raise SystemExit("PLOT_ROOT must point to a benchmark artifact directory")

ROOT = Path(os.environ["PLOT_ROOT"])
COLORS = {"baseline": "#555b66", "feature": "#16856b"}
LABELS = {"baseline": "Baseline", "feature": "Feature: retained vs reclaimable"}
TITLE = os.environ.get(
    "PLOT_TITLE",
    "KV cache redundancy comparison: baseline vs feature",
)
SUBTITLE = os.environ.get(
    "PLOT_SUBTITLE",
    "8192 input tokens/turn, 128 output tokens, 32 clients, seed 0, "
    "5-second raw sampling",
)
OUTPUT_STEM = os.environ.get("PLOT_OUTPUT_STEM", "redundancy_baseline_vs_feature")


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def render_chart(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, tuple[list[float], list[float]]],
    y_floor: float,
) -> None:
    all_x = [value for values, _ in series.values() for value in values]
    all_y = [value for _, values in series.values() for value in values]
    x_min = 0.0
    x_max = max(all_x)
    y_min = min(y_floor, min(all_y))
    y_max = max(all_y) * 1.04
    if y_max <= y_min:
        y_max = y_min + 1

    plot_x = x + 70
    plot_y = y + 42
    plot_w = width - 92
    plot_h = height - 96

    parts.append(
        f'<text x="{x + width / 2:.1f}" y="{y + 20:.1f}" class="panel-title" '
        f'text-anchor="middle">{title}</text>'
    )
    for index in range(6):
        ratio = index / 5
        gx = plot_x + plot_w * ratio
        gy = plot_y + plot_h * ratio
        x_value = x_min + (x_max - x_min) * ratio
        y_value = y_max - (y_max - y_min) * ratio
        parts.append(
            f'<line x1="{gx:.1f}" y1="{plot_y:.1f}" x2="{gx:.1f}" '
            f'y2="{plot_y + plot_h:.1f}" class="grid"/>'
        )
        parts.append(
            f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x + plot_w:.1f}" '
            f'y2="{gy:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{plot_y + plot_h + 20:.1f}" class="tick" '
            f'text-anchor="middle">{compact_number(x_value)}</text>'
        )
        parts.append(
            f'<text x="{plot_x - 10:.1f}" y="{gy + 4:.1f}" class="tick" '
            f'text-anchor="end">{compact_number(y_value)}</text>'
        )

    parts.append(
        f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" '
        f'x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" '
        f'y2="{plot_y + plot_h:.1f}" class="axis"/>'
    )
    parts.append(
        f'<text x="{plot_x + plot_w / 2:.1f}" y="{y + height - 8:.1f}" '
        f'class="axis-label" text-anchor="middle">{x_label}</text>'
    )
    parts.append(
        f'<text x="{x + 16:.1f}" y="{plot_y + plot_h / 2:.1f}" '
        f'class="axis-label" text-anchor="middle" '
        f'transform="rotate(-90 {x + 16:.1f} {plot_y + plot_h / 2:.1f})">'
        f"{y_label}</text>"
    )

    for mode, (x_values, y_values) in series.items():
        points = []
        for x_value, y_value in zip(x_values, y_values):
            px = plot_x + (x_value - x_min) / (x_max - x_min) * plot_w
            py = plot_y + (y_max - y_value) / (y_max - y_min) * plot_h
            points.append(f"{px:.1f},{py:.1f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{COLORS[mode]}" stroke-width="2.2"/>'
        )


def exact_owner_count(sample: dict, owner_count: int) -> float:
    if sample["block_redundancy_avg"] <= 0:
        return 0.0
    unique_blocks = round(sample["block_owners"] / sample["block_redundancy_avg"])
    owners_ge_2 = sample["owners_ge_2"]
    owners_ge_3 = sample["owners_ge_3"]
    owners_ge_4 = sample["owners_ge_4"]
    if owner_count == 1:
        return max(0.0, unique_blocks - owners_ge_2)
    if owner_count == 2:
        return max(0.0, owners_ge_2 - owners_ge_3)
    if owner_count == 3:
        return max(0.0, owners_ge_3 - owners_ge_4)
    if owner_count == 4:
        return owners_ge_4
    raise ValueError(f"unsupported owner count: {owner_count}")


def write_svg(data: dict[str, dict], path: Path) -> None:
    width, height = 1400, 1080
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:24px;"
        "font-weight:700}.subtitle{font-size:14px;fill:#555b66}.panel-title{font-size:16px;"
        "font-weight:700}.axis-label{font-size:12px;fill:#454b54}.tick{font-size:11px;"
        "fill:#6a717c}.grid{stroke:#d8dce2;stroke-width:1}.axis{stroke:#78808b;"
        "stroke-width:1.2}.legend{font-size:13px}</style>",
        '<text x="700" y="38" class="title" text-anchor="middle">' f"{TITLE}</text>",
        '<text x="700" y="62" class="subtitle" text-anchor="middle">'
        f"{SUBTITLE}</text>",
    ]

    elapsed = {
        mode: [sample["elapsed_s"] / 60 for sample in result["timeseries"]]
        for mode, result in data.items()
    }
    redundancy = {
        mode: [sample["block_redundancy_avg"] for sample in result["timeseries"]]
        for mode, result in data.items()
    }
    owners_by_count = {
        owner_count: {
            mode: [
                exact_owner_count(sample, owner_count)
                for sample in result["timeseries"]
            ]
            for mode, result in data.items()
        }
        for owner_count in range(1, 5)
    }

    panels = [
        (
            385,
            105,
            "Average redundancy",
            "Elapsed time (minutes)",
            "Average owner count",
            {m: (elapsed[m], redundancy[m]) for m in data},
            1.0,
            630,
            280,
        ),
        (
            55,
            410,
            "Blocks with exactly 1 owner",
            "Elapsed time (minutes)",
            "Block count",
            {m: (elapsed[m], owners_by_count[1][m]) for m in data},
            0.0,
            630,
            290,
        ),
        (
            715,
            410,
            "Blocks with exactly 2 owners",
            "Elapsed time (minutes)",
            "Block count",
            {m: (elapsed[m], owners_by_count[2][m]) for m in data},
            0.0,
            630,
            290,
        ),
        (
            55,
            725,
            "Blocks with exactly 3 owners",
            "Elapsed time (minutes)",
            "Block count",
            {m: (elapsed[m], owners_by_count[3][m]) for m in data},
            0.0,
            630,
            290,
        ),
        (
            715,
            725,
            "Blocks with exactly 4 owners",
            "Elapsed time (minutes)",
            "Block count",
            {m: (elapsed[m], owners_by_count[4][m]) for m in data},
            0.0,
            630,
            290,
        ),
    ]
    for x, y, title, x_label, y_label, series, y_floor, width, height in panels:
        render_chart(
            parts, x, y, width, height, title, x_label, y_label, series, y_floor
        )

    legend_y = 86
    for index, mode in enumerate(("baseline", "feature")):
        legend_x = 535 + index * 185
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 34}" '
            f'y2="{legend_y}" stroke="{COLORS[mode]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 43}" y="{legend_y + 5}" class="legend">'
            f"{LABELS[mode]}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def metric_total(snapshot: dict, prefix: str) -> float:
    return sum(
        value
        for node in snapshot["pega"].values()
        for key, value in node.items()
        if key.startswith(prefix)
    )


def parse_summary(log: Path) -> dict[str, float]:
    text = log.read_text()

    def number(pattern: str) -> float:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing summary field: {pattern}")
        return float(match.group(1))

    return {
        "runtime_sec": number(r"runtime_sec = ([0-9.]+)"),
        "requests_per_sec": number(r"requests_per_sec = ([0-9.]+)"),
        "ttft_mean_ms": number(r"ttft_ms\s+[0-9.]+\s+([0-9.]+)"),
        "input_tokens_mean": number(r"input_num_tokens\s+[0-9.]+\s+([0-9.]+)"),
    }


def load_mode(mode: str) -> dict:
    directory = ROOT / mode
    timeseries = json.loads((directory / "redundancy_timeseries.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    samples = [
        sample
        for sample in timeseries["samples"]
        if sample["block_owners"] and sample["block_redundancy_avg"] > 0
    ]
    final = samples[-1]
    redundancy = [sample["block_redundancy_avg"] for sample in samples]
    tail = redundancy[int(len(redundancy) * 0.8) :]
    summary = parse_summary(directory / "bench-tail.log")

    total_evictions = metric_total(
        metrics["snapshot"], "pegaflow_cache_block_evictions_total{"
    )
    result = {
        **summary,
        "sample_count": len(samples),
        "duration_sec": samples[-1]["elapsed_s"],
        "final_redundancy": final["block_redundancy_avg"],
        "mean_redundancy": sum(redundancy) / len(redundancy),
        "tail_20pct_mean_redundancy": sum(tail) / len(tail),
        "final_owners": final["block_owners"],
        "final_owners_ge_2": final["owners_ge_2"],
        "final_owners_ge_3": final["owners_ge_3"],
        "final_owners_ge_4": final["owners_ge_4"],
        "peak_redundancy": max(redundancy),
        "evictions": total_evictions,
        "rdma_fetch_bytes": metric_total(
            metrics["snapshot"], 'pegaflow_rdma_fetch_bytes_total{status="ok"'
        ),
        "rdma_fetch_count": metric_total(
            metrics["snapshot"], 'pegaflow_rdma_fetch_total_total{status="ok"'
        ),
        "cache_miss_blocks": metric_total(
            metrics["snapshot"], 'pegaflow_cache_tier_block_requests_total{tier="miss"'
        ),
        "cache_rdma_blocks": metric_total(
            metrics["snapshot"], 'pegaflow_cache_tier_block_requests_total{tier="rdma"'
        ),
        "resident_bytes": metric_total(
            metrics["snapshot"], "pegaflow_cache_resident_bytes"
        ),
        "timeseries": samples,
    }
    if mode == "feature":
        result.update(
            {
                "demotions": metric_total(
                    metrics["snapshot"], "pegaflow_cache_block_demotions_total{"
                ),
                "reclaimable_evictions": metric_total(
                    metrics["snapshot"],
                    'pegaflow_cache_block_evictions_by_class_total{class="reclaimable"',
                ),
                "retained_evictions": metric_total(
                    metrics["snapshot"],
                    'pegaflow_cache_block_evictions_by_class_total{class="retained"',
                ),
            }
        )
    else:
        result.update(
            {
                "demotions": 0.0,
                "reclaimable_evictions": 0.0,
                "retained_evictions": 0.0,
            }
        )
    return result


def main() -> None:
    data = {mode: load_mode(mode) for mode in ("baseline", "feature")}
    svg_path = ROOT / f"{OUTPUT_STEM}.svg"
    write_svg(data, svg_path)

    for mode in data:
        data[mode].pop("timeseries")
    baseline = data["baseline"]
    feature = data["feature"]
    data["delta_percent"] = {
        field: (feature[field] / baseline[field] - 1) * 100
        for field in (
            "final_redundancy",
            "mean_redundancy",
            "tail_20pct_mean_redundancy",
            "final_owners",
            "final_owners_ge_2",
            "final_owners_ge_4",
            "evictions",
            "rdma_fetch_bytes",
            "cache_miss_blocks",
        )
    }
    summary_path = ROOT / "summary.json"
    summary_path.write_text(json.dumps(data, indent=2) + "\n")
    print(svg_path)
    print(summary_path)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
