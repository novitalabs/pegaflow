#!/usr/bin/env python3
"""Find H20 hosts with enough idle GPUs for Kimi P/D runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from dataclasses import dataclass


DEFAULT_HOSTS = tuple(f"h20-{idx}" for idx in range(90, 113))


@dataclass(frozen=True)
class GpuMemory:
    index: int
    used_mib: int
    total_mib: int


@dataclass(frozen=True)
class HostResult:
    host: str
    gpus: tuple[GpuMemory, ...] = ()
    error: str | None = None

    def free_gpu_count(self, busy_threshold_mib: int) -> int:
        return sum(gpu.used_mib <= busy_threshold_mib for gpu in self.gpus)

    def busy_gpu_count(self, busy_threshold_mib: int) -> int:
        return sum(gpu.used_mib > busy_threshold_mib for gpu in self.gpus)

    def has_enough_free_gpus(
        self,
        min_free_gpus: int,
        busy_threshold_mib: int,
    ) -> bool:
        return (
            self.error is None
            and self.free_gpu_count(busy_threshold_mib) >= min_free_gpus
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hosts", nargs="*", default=DEFAULT_HOSTS)
    parser.add_argument("--min-free-gpus", type=int, default=8)
    parser.add_argument("--busy-threshold-mib", type=int, default=1024)
    parser.add_argument("--connect-timeout-s", type=int, default=3)
    parser.add_argument("--ssh-timeout-s", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--require-free",
        action="store_true",
        help="exit 1 when no host has --min-free-gpus idle GPUs",
    )
    args = parser.parse_args()

    if args.min_free_gpus <= 0:
        raise ValueError("--min-free-gpus must be positive")
    if args.busy_threshold_mib < 0:
        raise ValueError("--busy-threshold-mib must be nonnegative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    results = scan_hosts(
        tuple(args.hosts),
        connect_timeout_s=args.connect_timeout_s,
        ssh_timeout_s=args.ssh_timeout_s,
        workers=args.workers,
    )
    print_results(
        results,
        min_free_gpus=args.min_free_gpus,
        busy_threshold_mib=args.busy_threshold_mib,
    )
    if args.require_free and not any(
        result.has_enough_free_gpus(args.min_free_gpus, args.busy_threshold_mib)
        for result in results
    ):
        return 1
    return 0


def scan_hosts(
    hosts: tuple[str, ...],
    *,
    connect_timeout_s: int,
    ssh_timeout_s: int,
    workers: int,
) -> list[HostResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_host,
                host,
                connect_timeout_s=connect_timeout_s,
                ssh_timeout_s=ssh_timeout_s,
            ): host
            for host in hosts
        }
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    return sorted(results, key=lambda result: result.host)


def probe_host(
    host: str,
    *,
    connect_timeout_s: int,
    ssh_timeout_s: int,
) -> HostResult:
    command = (
        "nvidia-smi --query-gpu=index,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    try:
        output = subprocess.check_output(
            [
                "ssh",
                "-o",
                f"ConnectTimeout={connect_timeout_s}",
                "-o",
                "BatchMode=yes",
                host,
                command,
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=ssh_timeout_s,
        )
    except Exception as exc:
        return HostResult(host=host, error=first_line(str(exc)) or type(exc).__name__)

    gpus = tuple(parse_gpu_memory(output))
    if not gpus:
        return HostResult(host=host, error="no GPU memory rows returned")
    return HostResult(host=host, gpus=gpus)


def parse_gpu_memory(output: str) -> list[GpuMemory]:
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                GpuMemory(
                    index=int(parts[0]),
                    used_mib=int(parts[1]),
                    total_mib=int(parts[2]),
                )
            )
        except ValueError:
            continue
    return rows


def print_results(
    results: list[HostResult],
    *,
    min_free_gpus: int,
    busy_threshold_mib: int,
) -> None:
    print(
        "| host | status | free_gpus | busy_gpus | gpu_memory_mib | error |\n"
        "|------|--------|----------:|----------:|----------------|-------|"
    )
    for result in results:
        if result.error is not None:
            print(f"| {result.host} | ERR | 0 | 0 | - | {result.error} |")
            continue
        free = result.free_gpu_count(busy_threshold_mib)
        busy = result.busy_gpu_count(busy_threshold_mib)
        status = "FREE" if free >= min_free_gpus else "BUSY"
        memory = " ".join(
            f"{gpu.index}:{gpu.used_mib}/{gpu.total_mib}" for gpu in result.gpus
        )
        print(f"| {result.host} | {status} | {free} | {busy} | {memory} | - |")


def first_line(value: str) -> str:
    return value.splitlines()[0] if value.splitlines() else ""


if __name__ == "__main__":
    raise SystemExit(main())
