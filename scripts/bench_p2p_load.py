#!/usr/bin/env python
"""End-to-end P2P load benchmark.

Starts one metaserver, two PegaFlow servers (both seeing all GPUs), and two
vLLM instances (one per server). For each prompt length in the sweep it sends
the same prompt three times and compares TTFT:

  1. cold  -> vllm1: full prefill, KV saved to server1
  2. p2p   -> vllm2: server2 fetches the KV from server1 over RDMA
  3. warm  -> vllm1: server1 serves the KV from local pinned memory

Run on an 8-GPU RDMA node:

    PEGAFLOW_NICS=mlx5_gpu0,... PEGAFLOW_HOST_IP=10.0.0.1 \
      python/.venv/bin/python scripts/bench_p2p_load.py \
      --model /data/models/Qwen3-8B --prompt-tokens 2000,8000,16000,31000
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
import statistics
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
    # Identical block hashes across vLLM instances; without a fixed hash seed
    # the two replicas compute different hashes and never hit each other's KV.
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
    deadline = time.time() + 300
    last = baseline
    while time.time() < deadline:
        time.sleep(1)
        current = fetch_metric(http_port, prefixes)
        if current > baseline and current == last:
            return current
        last = current
    raise TimeoutError(
        f"metric {prefixes} did not settle (baseline={baseline}, last={last})"
    )


def gpu_mem_snapshot() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "n/a"
    used = [int(line) for line in out.split() if line]
    return f"{min(used)}-{max(used)} MiB/GPU" if used else "n/a"


def count_tokens(port: int, text: str) -> int:
    response = requests.post(
        f"http://127.0.0.1:{port}/tokenize",
        json={"model": MODEL_ALIAS, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["count"]


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


def make_words(seed: int, num_words: int) -> str:
    rng = random.Random(20260610 + seed)
    return " ".join(rng.choice(WORDS) for _ in range(num_words))


def make_prompt(port: int, seed: int, target_tokens: int) -> str:
    """Build a unique prompt of ~target_tokens, verified via /tokenize."""
    probe = make_words(seed, 512)
    tokens_per_word = count_tokens(port, probe) / 512
    num_words = int(target_tokens / tokens_per_word)
    for _ in range(8):
        prompt = f"Log {seed}: {make_words(seed, num_words)}\nSummarize briefly:"
        actual = count_tokens(port, prompt)
        if abs(actual - target_tokens) <= max(32, target_tokens // 100):
            return prompt
        num_words = int(num_words * target_tokens / actual)
    return prompt


def fmt_mib(value: float) -> str:
    return f"{value / (1 << 20):.0f}MiB"


def run_trial(
    vllm1: VllmReplica,
    vllm2: VllmReplica,
    pega1: PegaServer,
    pega2: PegaServer,
    prompt: str,
    max_tokens: int,
) -> dict:
    save_before = fetch_metric(pega1.http_port, SAVE_METRICS)
    cold = timed_completion(vllm1.port, prompt, max_tokens)
    # Wait for the async save to land on server1 and propagate to the
    # metaserver before asking server2 for the same blocks.
    wait_metric_settled(pega1.http_port, SAVE_METRICS, save_before)
    time.sleep(3)

    fetch_before = fetch_metric(pega2.http_port, FETCH_BYTES_METRICS)
    p2p = timed_completion(vllm2.port, prompt, max_tokens)
    fetched = fetch_metric(pega2.http_port, FETCH_BYTES_METRICS) - fetch_before

    load_before = fetch_metric(pega1.http_port, LOAD_BYTES_METRICS)
    warm = timed_completion(vllm1.port, prompt, max_tokens)
    loaded = fetch_metric(pega1.http_port, LOAD_BYTES_METRICS) - load_before

    return {
        "tokens": cold["prompt_tokens"],
        "cold": cold,
        "p2p": p2p,
        "warm": warm,
        "fetched": fetched,
        "loaded": loaded,
    }


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-tokens",
        default="2000,4000,8000,16000,31000",
        help="comma-separated prompt lengths to sweep",
    )
    parser.add_argument("--trials", type=int, default=2, help="trials per length")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="0 = max sweep length + 256",
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pool-size", default="60gb")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--log-dir", default="/tmp/pegaflow_p2p_bench")
    args = parser.parse_args()

    sweep = [int(x) for x in args.prompt_tokens.split(",")]
    max_model_len = args.max_model_len or max(sweep) + 256

    nics = os.environ.get("PEGAFLOW_NICS", "")
    if not nics:
        raise SystemExit("PEGAFLOW_NICS is required (comma-separated RDMA devices)")
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
                max_model_len,
            )
        )
        print(f"[gpu] after vllm-1: {gpu_mem_snapshot()}")
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
                max_model_len,
            )
        )
        print(f"[gpu] after vllm-2: {gpu_mem_snapshot()}")

        # Warm up both replicas so allocator/runtime state does not skew trial 1.
        timed_completion(vllm1.port, make_prompt(vllm1.port, -1, 256), args.max_tokens)
        timed_completion(vllm2.port, make_prompt(vllm1.port, -2, 256), args.max_tokens)

        seed = 0
        for target in sweep:
            for trial in range(args.trials):
                seed += 1
                prompt = make_prompt(vllm1.port, seed, target)
                row = run_trial(vllm1, vllm2, pega1, pega2, prompt, args.max_tokens)
                row.update(target=target, trial=trial)
                results.append(row)
                print(
                    f"tokens={row['tokens']} trial={trial}: "
                    f"cold={row['cold']['ttft_ms']:.0f}ms "
                    f"p2p={row['p2p']['ttft_ms']:.0f}ms "
                    f"(rdma {fmt_mib(row['fetched'])}) "
                    f"warm={row['warm']['ttft_ms']:.0f}ms "
                    f"(local {fmt_mib(row['loaded'])}) "
                    f"gpu={gpu_mem_snapshot()}"
                )

    (log_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "rows": results}, indent=2)
    )

    print("\n=== P2P load benchmark (median TTFT over trials) ===")
    print(f"model={args.model} tp={args.tp} trials={args.trials}")
    header = f"{'tokens':>7} {'cold':>9} {'p2p':>9} {'warm':>9} {'cold/p2p':>9} {'rdma fetched':>13}"
    print(header)
    for target in sweep:
        rows = [r for r in results if r["target"] == target]
        cold = statistics.median(r["cold"]["ttft_ms"] for r in rows)
        p2p = statistics.median(r["p2p"]["ttft_ms"] for r in rows)
        warm = statistics.median(r["warm"]["ttft_ms"] for r in rows)
        fetched = statistics.median(r["fetched"] for r in rows)
        print(
            f"{rows[0]['tokens']:>7} {cold:>7.0f}ms {p2p:>7.0f}ms {warm:>7.0f}ms "
            f"{cold / p2p if p2p else 0:>8.1f}x {fmt_mib(fetched):>13}"
        )
    for r in results:
        if r["fetched"] <= 0:
            print(
                f"WARNING: tokens={r['target']} trial={r['trial']} fetched 0 RDMA "
                "bytes — p2p path not exercised"
            )
        if r["loaded"] <= 0:
            print(
                f"WARNING: tokens={r['target']} trial={r['trial']} loaded 0 local "
                "bytes — warm path not exercised"
            )


if __name__ == "__main__":
    main()
