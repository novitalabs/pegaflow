#!/usr/bin/env python3
"""Find H20 hosts with enough idle GPUs for Kimi P/D runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import shlex
import subprocess
from dataclasses import dataclass


DEFAULT_HOSTS = tuple(f"h20-{idx}" for idx in range(90, 113))
DEFAULT_NICS = ("mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4")


@dataclass(frozen=True)
class GpuMemory:
    index: int
    used_mib: int
    total_mib: int


@dataclass(frozen=True)
class NicTraffic:
    name: str
    sample_s: float
    xmit_gbps: float
    rcv_gbps: float

    @property
    def active_gbps(self) -> float:
        return max(self.xmit_gbps, self.rcv_gbps)


@dataclass(frozen=True)
class HostResult:
    host: str
    gpus: tuple[GpuMemory, ...] = ()
    nics: tuple[NicTraffic, ...] = ()
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

    def has_idle_nics(self, expected_nics: int, max_nic_gbps: float) -> bool:
        if expected_nics == 0:
            return True
        return len(self.nics) >= expected_nics and all(
            nic.active_gbps <= max_nic_gbps for nic in self.nics
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hosts", nargs="*", default=DEFAULT_HOSTS)
    parser.add_argument("--min-free-hosts", type=int, default=1)
    parser.add_argument("--min-free-gpus", type=int, default=8)
    parser.add_argument("--busy-threshold-mib", type=int, default=1024)
    parser.add_argument("--connect-timeout-s", type=int, default=3)
    parser.add_argument("--ssh-timeout-s", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--nics", nargs="*", default=list(DEFAULT_NICS))
    parser.add_argument("--skip-nics", action="store_true")
    parser.add_argument("--nic-sample-s", type=float, default=1.0)
    parser.add_argument("--max-nic-gbps", type=float, default=1.0)
    parser.add_argument(
        "--require-free",
        action="store_true",
        help="exit 1 unless at least --min-free-hosts hosts have enough idle GPUs and idle requested NICs",
    )
    args = parser.parse_args()

    if args.min_free_hosts <= 0:
        raise ValueError("--min-free-hosts must be positive")
    if args.min_free_gpus <= 0:
        raise ValueError("--min-free-gpus must be positive")
    if args.busy_threshold_mib < 0:
        raise ValueError("--busy-threshold-mib must be nonnegative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.nic_sample_s <= 0:
        raise ValueError("--nic-sample-s must be positive")
    if args.max_nic_gbps < 0:
        raise ValueError("--max-nic-gbps must be nonnegative")

    nics = () if args.skip_nics else tuple(args.nics)
    results = scan_hosts(
        tuple(args.hosts),
        nics=nics,
        nic_sample_s=args.nic_sample_s,
        connect_timeout_s=args.connect_timeout_s,
        ssh_timeout_s=args.ssh_timeout_s,
        workers=args.workers,
    )
    print_results(
        results,
        min_free_gpus=args.min_free_gpus,
        busy_threshold_mib=args.busy_threshold_mib,
        expected_nics=len(nics),
        max_nic_gbps=args.max_nic_gbps,
    )
    free_hosts = count_free_hosts(
        results,
        min_free_gpus=args.min_free_gpus,
        busy_threshold_mib=args.busy_threshold_mib,
        expected_nics=len(nics),
        max_nic_gbps=args.max_nic_gbps,
    )
    if args.require_free and free_hosts < args.min_free_hosts:
        return 1
    return 0


def scan_hosts(
    hosts: tuple[str, ...],
    *,
    nics: tuple[str, ...],
    nic_sample_s: float,
    connect_timeout_s: int,
    ssh_timeout_s: int,
    workers: int,
) -> list[HostResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_host,
                host,
                nics=nics,
                nic_sample_s=nic_sample_s,
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
    nics: tuple[str, ...],
    nic_sample_s: float,
    connect_timeout_s: int,
    ssh_timeout_s: int,
) -> HostResult:
    command = remote_probe_command(nics, nic_sample_s)
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
            timeout=ssh_timeout_s + nic_sample_s + 2,
        )
    except Exception as exc:
        return HostResult(host=host, error=first_line(str(exc)) or type(exc).__name__)

    gpus = tuple(parse_gpu_memory(output))
    if not gpus:
        return HostResult(host=host, error="no GPU memory rows returned")
    return HostResult(host=host, gpus=gpus, nics=tuple(parse_nic_traffic(output)))


def remote_probe_command(nics: tuple[str, ...], nic_sample_s: float) -> str:
    command = [
        "nvidia-smi --query-gpu=index,memory.used,memory.total "
        "--format=csv,noheader,nounits | sed 's/^/GPU,/'"
    ]
    if nics:
        quoted_nics = " ".join(shlex.quote(nic) for nic in nics)
        command.append(
            "\n".join(
                [
                    "tmp=$(mktemp)",
                    "t0=$(date +%s.%N)",
                    f"for nic in {quoted_nics}; do",
                    "  counters=/sys/class/infiniband/$nic/ports/1/counters",
                    '  if [ -r "$counters/port_xmit_data" ] && [ -r "$counters/port_rcv_data" ]; then',
                    '    printf \'%s,%s,%s\\n\' "$nic" "$(cat $counters/port_xmit_data)" "$(cat $counters/port_rcv_data)" >> "$tmp"',
                    "  fi",
                    "done",
                    f"sleep {shlex.quote(str(nic_sample_s))}",
                    "t1=$(date +%s.%N)",
                    "while IFS=, read -r nic x0 r0; do",
                    "  counters=/sys/class/infiniband/$nic/ports/1/counters",
                    '  if [ -r "$counters/port_xmit_data" ] && [ -r "$counters/port_rcv_data" ]; then',
                    '    printf \'NIC,%s,%s,%s,%s,%s,%s,%s\\n\' "$nic" "$t0" "$t1" "$x0" "$r0" "$(cat $counters/port_xmit_data)" "$(cat $counters/port_rcv_data)"',
                    "  fi",
                    'done < "$tmp"',
                    'rm -f "$tmp"',
                ]
            )
        )
    return "set -e; " + "; ".join(command)


def parse_gpu_memory(output: str) -> list[GpuMemory]:
    rows = []
    for line in output.splitlines():
        if not line.startswith("GPU,"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                GpuMemory(
                    index=int(parts[1]),
                    used_mib=int(parts[2]),
                    total_mib=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return rows


def parse_nic_traffic(output: str) -> list[NicTraffic]:
    rows = []
    for line in output.splitlines():
        if not line.startswith("NIC,"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            name = parts[1]
            start_s = float(parts[2])
            end_s = float(parts[3])
            xmit0 = int(parts[4])
            rcv0 = int(parts[5])
            xmit1 = int(parts[6])
            rcv1 = int(parts[7])
        except ValueError:
            continue
        sample_s = end_s - start_s
        if sample_s <= 0:
            continue
        rows.append(
            NicTraffic(
                name=name,
                sample_s=sample_s,
                xmit_gbps=counter_gbps(xmit1 - xmit0, sample_s),
                rcv_gbps=counter_gbps(rcv1 - rcv0, sample_s),
            )
        )
    return rows


def counter_gbps(counter_delta: int, sample_s: float) -> float:
    # IB port_xmit_data/port_rcv_data counters are 4-byte words.
    return counter_delta * 32 / sample_s / 1e9


def count_free_hosts(
    results: list[HostResult],
    *,
    min_free_gpus: int,
    busy_threshold_mib: int,
    expected_nics: int,
    max_nic_gbps: float,
) -> int:
    return sum(
        result.has_enough_free_gpus(min_free_gpus, busy_threshold_mib)
        and result.has_idle_nics(expected_nics, max_nic_gbps)
        for result in results
    )


def print_results(
    results: list[HostResult],
    *,
    min_free_gpus: int,
    busy_threshold_mib: int,
    expected_nics: int,
    max_nic_gbps: float,
) -> None:
    print(
        "| host | status | free_gpus | busy_gpus | gpu_memory_mib | nic_gbps | error |\n"
        "|------|--------|----------:|----------:|----------------|----------|-------|"
    )
    for result in results:
        if result.error is not None:
            print(f"| {result.host} | ERR | 0 | 0 | - | - | {result.error} |")
            continue
        free = result.free_gpu_count(busy_threshold_mib)
        busy = result.busy_gpu_count(busy_threshold_mib)
        nic_idle = result.has_idle_nics(expected_nics, max_nic_gbps)
        status = "FREE" if free >= min_free_gpus and nic_idle else "BUSY"
        memory = " ".join(
            f"{gpu.index}:{gpu.used_mib}/{gpu.total_mib}" for gpu in result.gpus
        )
        nics = " ".join(
            f"{nic.name}:x{nic.xmit_gbps:.2f}/r{nic.rcv_gbps:.2f}"
            for nic in result.nics
        )
        print(
            f"| {result.host} | {status} | {free} | {busy} | {memory} | {nics or '-'} | - |"
        )


def first_line(value: str) -> str:
    return value.splitlines()[0] if value.splitlines() else ""


if __name__ == "__main__":
    raise SystemExit(main())
