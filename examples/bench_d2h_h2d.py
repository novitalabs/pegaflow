#!/usr/bin/env python3
"""Isolated D2H / H2D microbenchmark for P/D disaggregation baseline.

Measures raw GPU↔CPU pinned-memory throughput and latency without the
inference engine.  This is the foundation for deciding whether local vs
remote prefill is cheaper in a Conditional P/D policy.

Usage::

    python examples/bench_d2h_h2d.py --dtype float16 --iterations 100
    python examples/bench_d2h_h2d.py --model-style qwen3-4b --iterations 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BenchmarkConfig:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    pinned: bool
    async_copy: bool
    iterations: int
    warmup: int


@dataclass(frozen=True)
class BenchmarkResult:
    direction: str
    total_bytes: int
    times_ms: list[float]
    throughput_gbps: list[float]
    p50_ms: float
    p99_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    avg_gbps: float
    max_gbps: float


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name.lower() not in mapping:
        raise ValueError(f"unsupported dtype {name}; choose from {list(mapping)}")
    return mapping[name.lower()]


def _model_preset(name: str) -> dict[str, Any]:
    presets = {
        "qwen3-4b": {
            "num_layers": 36,
            "block_size": 16,
            "num_kv_heads": 4,
            "head_size": 128,
            "dtype": torch.float16,
        },
        "qwen3-8b": {
            "num_layers": 36,
            "block_size": 16,
            "num_kv_heads": 4,
            "head_size": 128,
            "dtype": torch.bfloat16,
        },
        "deepseek-v3": {
            "num_layers": 61,
            "block_size": 64,
            "num_kv_heads": 1,  # MLA collapses to latent vector
            "head_size": 512,
            "dtype": torch.bfloat16,
        },
    }
    if name.lower() not in presets:
        raise ValueError(f"unknown model preset {name}")
    return presets[name.lower()]


def _build_tensor_shape(cfg: BenchmarkConfig) -> tuple[int, ...]:
    """Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]

    The leading ``num_layers`` dimension lets us benchmark multi-layer
    transfers as a single contiguous copy, which is what the P/D connector
    does when it coalesces layers.
    """
    return (cfg.num_layers, 2, cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_size)


def _alloc_cpu_buffer(shape: tuple[int, ...], dtype: torch.dtype, pinned: bool) -> torch.Tensor:
    t = torch.empty(shape, dtype=dtype, pin_memory=pinned)
    if pinned:
        # Touch pages to avoid page-fault overhead during the timed region
        t.fill_(0)
    return t


def _run_direction(
    cfg: BenchmarkConfig,
    gpu_buf: torch.Tensor,
    cpu_buf: torch.Tensor,
    direction: str,
) -> BenchmarkResult:
    """Direction is ``D2H`` (gpu→cpu) or ``H2D`` (cpu→gpu)."""
    total_bytes = gpu_buf.numel() * gpu_buf.element_size()
    times_ms: list[float] = []

    for _ in range(cfg.warmup):
        if direction == "D2H":
            cpu_buf.copy_(gpu_buf, non_blocking=cfg.async_copy)
        else:
            gpu_buf.copy_(cpu_buf, non_blocking=cfg.async_copy)
        if cfg.async_copy:
            torch.cuda.synchronize()

    for _ in range(cfg.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Synchronise before timing so the GPU is idle
        torch.cuda.synchronize()
        start.record()
        if direction == "D2H":
            cpu_buf.copy_(gpu_buf, non_blocking=cfg.async_copy)
        else:
            gpu_buf.copy_(cpu_buf, non_blocking=cfg.async_copy)
        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)  # milliseconds
        times_ms.append(elapsed)

    throughput = [total_bytes / (t / 1e3) / 1e9 for t in times_ms]  # GB/s
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    p50 = sorted_times[n // 2] if n % 2 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
    p99 = sorted_times[int(n * 0.99)]

    return BenchmarkResult(
        direction=direction,
        total_bytes=total_bytes,
        times_ms=times_ms,
        throughput_gbps=throughput,
        p50_ms=p50,
        p99_ms=p99,
        avg_ms=statistics.mean(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        avg_gbps=statistics.mean(throughput),
        max_gbps=max(throughput),
    )


def run_benchmark(cfg: BenchmarkConfig) -> tuple[BenchmarkResult, BenchmarkResult]:
    shape = _build_tensor_shape(cfg)
    gpu_buf = torch.empty(shape, dtype=cfg.dtype, device="cuda")
    cpu_buf = _alloc_cpu_buffer(shape, cfg.dtype, cfg.pinned)

    # Fill GPU buffer with non-zero data so the copy is not elided
    gpu_buf.normal_(mean=0.0, std=1.0)
    torch.cuda.synchronize()

    d2h = _run_direction(cfg, gpu_buf, cpu_buf, "D2H")
    h2d = _run_direction(cfg, gpu_buf, cpu_buf, "H2D")
    return d2h, h2d


def _sweep_parameter(
    base_cfg: BenchmarkConfig,
    param_name: str,
    values: list[Any],
) -> list[dict[str, Any]]:
    """Run a parameter sweep and return a list of result dicts."""
    rows: list[dict[str, Any]] = []
    for value in values:
        kwargs = {**asdict(base_cfg), param_name: value}
        cfg = BenchmarkConfig(**kwargs)
        d2h, h2d = run_benchmark(cfg)
        rows.append(
            {
                param_name: value,
                "total_mib": d2h.total_bytes / (1024 * 1024),
                "d2h_p50_ms": d2h.p50_ms,
                "d2h_p99_ms": d2h.p99_ms,
                "d2h_avg_gbps": d2h.avg_gbps,
                "d2h_max_gbps": d2h.max_gbps,
                "h2d_p50_ms": h2d.p50_ms,
                "h2d_p99_ms": h2d.p99_ms,
                "h2d_avg_gbps": h2d.avg_gbps,
                "h2d_max_gbps": h2d.max_gbps,
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    # Header
    header = " | ".join(f"{k:>18}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                vals.append(f"{v:>18.2f}")
            else:
                vals.append(f"{v:>18}")
        print(" | ".join(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="D2H/H2D microbenchmark")
    parser.add_argument("--model-style", choices=["qwen3-4b", "qwen3-8b", "deepseek-v3"])
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=_parse_dtype, default="float16")
    parser.add_argument("--pinned", action="store_true", default=True)
    parser.add_argument("--pageable", action="store_true", help="Use pageable CPU memory (default is pinned)")
    parser.add_argument("--async-copy", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sweep", choices=["block_size", "num_blocks", "num_layers", "pinned"])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    pinned = not args.pageable if args.pageable else args.pinned

    if args.model_style:
        preset = _model_preset(args.model_style)
        cfg = BenchmarkConfig(
            num_layers=preset["num_layers"],
            num_blocks=args.num_blocks,
            block_size=preset["block_size"],
            num_kv_heads=preset["num_kv_heads"],
            head_size=preset["head_size"],
            dtype=preset["dtype"],
            pinned=pinned,
            async_copy=args.async_copy,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    else:
        cfg = BenchmarkConfig(
            num_layers=args.num_layers,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            dtype=args.dtype,
            pinned=pinned,
            async_copy=args.async_copy,
            iterations=args.iterations,
            warmup=args.warmup,
        )

    if args.sweep:
        print(f"Sweeping {args.sweep} ...")
        if args.sweep == "block_size":
            rows = _sweep_parameter(cfg, "block_size", [8, 16, 32, 64, 128])
        elif args.sweep == "num_blocks":
            rows = _sweep_parameter(cfg, "num_blocks", [32, 64, 128, 256, 512])
        elif args.sweep == "num_layers":
            rows = _sweep_parameter(cfg, "num_layers", [1, 10, 20, 36, 61])
        elif args.sweep == "pinned":
            rows = _sweep_parameter(cfg, "pinned", [False, True])
        else:
            rows = []
        _print_table(rows)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"\nSaved sweep results to {args.json_out}")
        return 0

    # Single run
    d2h, h2d = run_benchmark(cfg)
    print("Configuration:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print()
    print(f"Tensor shape: {_build_tensor_shape(cfg)}")
    print(f"Total bytes per transfer: {d2h.total_bytes / (1024 * 1024):.2f} MiB")
    print()

    def _print_result(r: BenchmarkResult) -> None:
        print(f"{r.direction}:")
        print(f"  p50 latency : {r.p50_ms:.3f} ms")
        print(f"  p99 latency : {r.p99_ms:.3f} ms")
        print(f"  avg latency : {r.avg_ms:.3f} ms")
        print(f"  avg throughput: {r.avg_gbps:.2f} GB/s")
        print(f"  max throughput: {r.max_gbps:.2f} GB/s")

    _print_result(d2h)
    print()
    _print_result(h2d)

    if args.json_out:
        payload = {
            "device": device_name,
            "config": asdict(cfg),
            "d2h": asdict(d2h),
            "h2d": asdict(h2d),
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
