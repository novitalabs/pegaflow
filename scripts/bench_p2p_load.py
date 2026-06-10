#!/usr/bin/env python
"""End-to-end P2P load benchmark.

Starts one metaserver, two PegaFlow servers (both seeing all GPUs), and two
vLLM instances (one per server). For each trial it sends the same long prompt
three times and compares TTFT:

  1. cold  -> vllm1: full prefill, KV saved to server1
  2. p2p   -> vllm2: server2 fetches the KV from server1 over RDMA
  3. warm  -> vllm1: server1 serves the KV from local pinned memory

Run on an 8-GPU RDMA node:

    PEGAFLOW_NICS=mlx5_gpu0,... PEGAFLOW_HOST_IP=10.0.0.1 \
      python/.venv/bin/python scripts/bench_p2p_load.py \
      --model /data/models/Qwen3-8B --prompt-tokens 4000 --trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
from contextlib import ExitStack, suppress
from pathlib import Path

import requests

MODEL_ALIAS = "pegaflow-p2p-bench"
STARTUP_TIMEOUT_SECONDS = 900

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu amber basalt cobalt dune ember fjord glacier harbor "
    "island jasper karst lagoon meadow nectar onyx prairie quartz reef summit "
    "tundra umbra vortex willow zenith"
).split()


def project_root() -> Path:
    return Path(__file__).parent.parent


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def server_cmd(binary_name: str) -> list[str]:
    release_binary = project_root() / "target" / "release" / binary_name
    if release_binary.exists():
        return [str(release_binary)]
    if path := shutil.which(binary_name):
        return [path]
    raise FileNotFoundError(
        f"{binary_name} not found; run `cargo build --release` first"
    )


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYO3_PYTHON"] = sys.executable
    env["PYTHONHOME"] = sys.base_prefix
    if libdir := sysconfig.get_config_var("LIBDIR"):
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    python_dir = str(project_root() / "python")
    site_packages = next((p for p in sys.path if "site-packages" in p), None)
    env["PYTHONPATH"] = python_dir + (f":{site_packages}" if site_packages else "")
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    return env


class ManagedProcess:
    def __init__(
        self,
        label: str,
        cmd: list[str],
        log_file: Path,
        ready_url: str,
        env: dict[str, str],
    ) -> None:
        self.label = label
        self.cmd = cmd
        self.log_file = log_file
        self.ready_url = ready_url
        self.env = env
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def __enter__(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_file.open("wb")
        print(f"[{self.label}] starting, log: {self.log_file}")
        self.process = subprocess.Popen(
            self.cmd,
            cwd=project_root(),
            env=self.env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._wait_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process is not None:
            with suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=30)
        if self._log_handle is not None:
            self._log_handle.close()

    def _wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.time() + STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} exited with code {self.process.returncode}\n"
                    f"log tail:\n{self.tail()}"
                )
            try:
                if requests.get(self.ready_url, timeout=1).status_code == 200:
                    print(f"[{self.label}] ready")
                    time.sleep(2)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.label} not ready\nlog tail:\n{self.tail()}")

    def tail(self, limit: int = 4000) -> str:
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(errors="replace")[-limit:]


class MetaServer(ManagedProcess):
    def __init__(self, log_dir: Path, host: str) -> None:
        self.grpc_port = find_available_port()
        http_port = find_available_port()
        cmd = server_cmd("pegaflow-metaserver") + [
            "--addr",
            f"{host}:{self.grpc_port}",
            "--http-addr",
            f"{host}:{http_port}",
            "--log-level",
            "info",
        ]
        super().__init__(
            "metaserver",
            cmd,
            log_dir / "metaserver.log",
            f"http://{host}:{http_port}/health",
            base_env(),
        )


class PegaServer(ManagedProcess):
    def __init__(
        self,
        label: str,
        devices: str,
        log_dir: Path,
        pool_size: str,
        metaserver: MetaServer,
        nics: str,
        advertise_host: str,
    ) -> None:
        self.grpc_port = find_available_port()
        self.http_port = find_available_port()
        self.grpc_host = advertise_host
        cmd = server_cmd("pegaflow-server") + [
            "--addr",
            f"{advertise_host}:{self.grpc_port}",
            "--http-addr",
            f"127.0.0.1:{self.http_port}",
            "--devices",
            devices,
            "--pool-size",
            pool_size,
            "--transfer-backend",
            "kernel",
            "--log-level",
            "info",
            "--disable-numa-affinity",
            "--enable-prometheus",
            "--metaserver-addr",
            f"http://{advertise_host}:{metaserver.grpc_port}",
            "--nics",
            nics,
        ]
        super().__init__(
            f"pega-{label}",
            cmd,
            log_dir / f"pega-{label}.log",
            f"http://127.0.0.1:{self.http_port}/health",
            base_env(),
        )


class VllmReplica(ManagedProcess):
    def __init__(
        self,
        label: str,
        model: str,
        port: int,
        devices: str,
        tp_size: int,
        gpu_memory_utilization: float,
        pega_server: PegaServer,
        log_dir: Path,
        max_model_len: int,
    ) -> None:
        self.port = port
        kv_config = {
            "kv_connector": "PegaKVConnector",
            "kv_connector_module_path": "pegaflow.connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "pegaflow.host": f"http://{pega_server.grpc_host}",
                "pegaflow.port": pega_server.grpc_port,
                "pegaflow.mode": "read_write",
            },
        }
        env = base_env()
        env["CUDA_VISIBLE_DEVICES"] = devices
        cmd = [
            "vllm",
            "serve",
            model,
            "--served-model-name",
            MODEL_ALIAS,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp_size),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--disable-hybrid-kv-cache-manager",
            "--kv-transfer-config",
            json.dumps(kv_config, separators=(",", ":")),
        ]
        super().__init__(
            f"vllm-{label}",
            cmd,
            log_dir / f"vllm-{label}.log",
            f"http://127.0.0.1:{port}/v1/models",
            env,
        )


def fetch_metric(http_port: int, prefixes: tuple[str, ...]) -> float:
    response = requests.get(f"http://localhost:{http_port}/metrics", timeout=5)
    response.raise_for_status()
    total = 0.0
    pattern = re.compile(
        r"^(" + "|".join(re.escape(p) for p in prefixes) + r")(?:_total)?"
        r"(?:\{[^}]*\})?\s+([\d.eE+-]+)$"
    )
    for line in response.text.splitlines():
        if match := pattern.match(line):
            total += float(match.group(2))
    return total


SAVE_METRICS = ("pegaflow_save_bytes",)
FETCH_BYTES_METRICS = ("pegaflow_rdma_fetch_bytes",)
LOAD_BYTES_METRICS = ("pegaflow_load_bytes",)


def wait_metric_settled(
    http_port: int, prefixes: tuple[str, ...], baseline: float
) -> float:
    """Wait until the metric rises above baseline and stops increasing."""
    deadline = time.time() + 120
    last = baseline
    while time.time() < deadline:
        time.sleep(2)
        current = fetch_metric(http_port, prefixes)
        if current > baseline and current == last:
            return current
        last = current
    raise TimeoutError(
        f"metric {prefixes} did not settle (baseline={baseline}, last={last})"
    )


def timed_completion(port: int, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": MODEL_ALIAS,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    ttft = None
    prompt_tokens = None
    for raw in response.iter_lines():
        if not raw.startswith(b"data: "):
            continue
        payload = raw[len(b"data: ") :]
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if usage := chunk.get("usage"):
            prompt_tokens = usage.get("prompt_tokens")
        if ttft is None and any(c.get("text") for c in chunk.get("choices", [])):
            ttft = time.perf_counter() - start
    return {
        "ttft_ms": (ttft or 0.0) * 1e3,
        "total_ms": (time.perf_counter() - start) * 1e3,
        "prompt_tokens": prompt_tokens,
    }


def make_prompt(trial: int, target_tokens: int) -> str:
    rng = random.Random(20260610 + trial)
    body = " ".join(rng.choice(WORDS) for _ in range(target_tokens))
    return f"Log {trial}: {body}\nSummarize the pattern above:"


def fmt_mib(value: float) -> str:
    return f"{value / (1 << 20):.0f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pool-size", default="40gb")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--log-dir", default="/tmp/pegaflow_p2p_bench")
    args = parser.parse_args()

    nics = os.environ.get("PEGAFLOW_NICS", "")
    if not nics:
        raise SystemExit("PEGAFLOW_NICS is required (comma-separated RDMA device list)")
    host = os.environ.get("PEGAFLOW_HOST_IP", "127.0.0.1")

    log_dir = Path(args.log_dir)
    results = []

    with ExitStack() as stack:
        metaserver = stack.enter_context(MetaServer(log_dir, host))
        pega1 = stack.enter_context(
            PegaServer(
                "1", args.devices, log_dir, args.pool_size, metaserver, nics, host
            )
        )
        pega2 = stack.enter_context(
            PegaServer(
                "2", args.devices, log_dir, args.pool_size, metaserver, nics, host
            )
        )
        vllm1 = stack.enter_context(
            VllmReplica(
                "1",
                args.model,
                find_available_port(),
                args.devices,
                args.tp,
                args.gpu_memory_utilization,
                pega1,
                log_dir,
                args.max_model_len,
            )
        )
        vllm2 = stack.enter_context(
            VllmReplica(
                "2",
                args.model,
                find_available_port(),
                args.devices,
                args.tp,
                args.gpu_memory_utilization,
                pega2,
                log_dir,
                args.max_model_len,
            )
        )

        # Warm up both replicas so CUDA graphs/allocator state do not skew trial 1.
        timed_completion(vllm1.port, make_prompt(-1, 64), args.max_tokens)
        timed_completion(vllm2.port, make_prompt(-2, 64), args.max_tokens)

        for trial in range(args.trials):
            prompt = make_prompt(trial, args.prompt_tokens)

            save_before = fetch_metric(pega1.http_port, SAVE_METRICS)
            cold = timed_completion(vllm1.port, prompt, args.max_tokens)
            # Wait for the async save to land on server1 and propagate to the
            # metaserver before asking server2 for the same blocks.
            wait_metric_settled(pega1.http_port, SAVE_METRICS, save_before)
            time.sleep(3)

            fetch_before = fetch_metric(pega2.http_port, FETCH_BYTES_METRICS)
            p2p = timed_completion(vllm2.port, prompt, args.max_tokens)
            fetched = fetch_metric(pega2.http_port, FETCH_BYTES_METRICS) - fetch_before

            load_before = fetch_metric(pega1.http_port, LOAD_BYTES_METRICS)
            warm = timed_completion(vllm1.port, prompt, args.max_tokens)
            loaded = fetch_metric(pega1.http_port, LOAD_BYTES_METRICS) - load_before

            results.append(
                {
                    "trial": trial,
                    "tokens": cold["prompt_tokens"],
                    "cold": cold,
                    "p2p": p2p,
                    "warm": warm,
                    "fetched": fetched,
                    "loaded": loaded,
                }
            )
            print(
                f"trial {trial} ({cold['prompt_tokens']} prompt tokens): "
                f"cold={cold['ttft_ms']:.0f}ms "
                f"p2p={p2p['ttft_ms']:.0f}ms (rdma {fmt_mib(fetched)}) "
                f"warm={warm['ttft_ms']:.0f}ms (local {fmt_mib(loaded)})"
            )

    print("\n=== P2P load benchmark ===")
    print(f"model={args.model} tp={args.tp} prompt_tokens~{args.prompt_tokens}")
    print(
        f"{'trial':>5} {'tokens':>7} {'cold TTFT':>10} {'p2p TTFT':>10} {'warm TTFT':>10} {'rdma fetched':>13}"
    )
    for r in results:
        print(
            f"{r['trial']:>5} {r['tokens'] or '?':>7} "
            f"{r['cold']['ttft_ms']:>8.0f}ms {r['p2p']['ttft_ms']:>8.0f}ms "
            f"{r['warm']['ttft_ms']:>8.0f}ms {fmt_mib(r['fetched']):>13}"
        )
    for r in results:
        if r["fetched"] <= 0:
            print(
                f"WARNING: trial {r['trial']} fetched 0 RDMA bytes — p2p path not exercised"
            )


if __name__ == "__main__":
    main()
