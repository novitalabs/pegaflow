"""Real vLLM PP4 + PegaFlow P2P correctness probe.

This test is intentionally hardware-gated. It starts:

1. one source PegaFlow server with RDMA enabled and a PP4 producer vLLM,
2. one local PP4 consumer on the source server,
3. one remote PegaFlow server plus a PP4 consumer that fetches the same KV over RDMA.

The local consumer and the RDMA consumer both load KV produced by the same
producer run. That avoids false failures from independent PP replicas choosing
slightly different greedy outputs on numerically close logits.

Run on an 8-GPU RDMA node:

    PEGAFLOW_NICS=mlx5_... PEGAFLOW_HOST_IP=172.31.13.39 \
      uv run python -m pytest -m e2e python/tests/test_vllm_p2p_pp4_e2e.py \
      --model /data/models/Qwen3-8B --pegaflow-pool-size 60gb -s -v
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

import pytest
import requests

from .vllm_helpers import fetch_pegaflow_metrics, find_available_port

PP_SIZE = 4
TP_SIZE = 1
MODEL_ALIAS = "pegaflow-p2p-pp4"
DEFAULT_MAX_MODEL_LEN = 512
DEFAULT_GPU_MEMORY_UTILIZATION = 0.55
DEFAULT_POOL_SIZE = "30gb"
STARTUP_TIMEOUT_SECONDS = 900
RDMA_FETCH_COUNTER_NAMES = (
    "pegaflow_rdma_fetch_total",
    "pegaflow_rdma_fetch_total_total",
    "pegaflow_rdma_fetch",
)
RDMA_FETCH_BYTES_COUNTER_NAMES = (
    "pegaflow_rdma_fetch_bytes",
    "pegaflow_rdma_fetch_bytes_total",
    "pegaflow_rdma_fetch_bytes_total_total",
)

PROMPTS = [
    (
        "A warehouse is auditing a shipment with three product categories and "
        "two stages of discounts. Category A has 18 boxes with 24 units per "
        "box. Category B has 27 boxes with 16 units per box. Category C has 9 "
        "boxes with 48 units per box. The auditor removes 37 damaged units, "
        "then applies a promotion that gives away 3 units for every complete "
        "batch of 50 remaining units. Shipping costs are computed after the "
        "giveaway: the first 600 shipped units cost 4 dollars each, every "
        "additional shipped unit costs 7 dollars each, and the whole invoice "
        "gets a final 12 percent discount. Check the arithmetic carefully, "
        "keep the batch giveaway as an integer floor division, and report the "
        "final invoice in dollars. Step by step, the total shipped units and "
        "final invoice are"
    ),
    (
        "A serving cluster has four pipeline stages. Stage P0 handles layers "
        "0 through 8, P1 handles layers 9 through 17, P2 handles layers 18 "
        "through 26, and P3 handles layers 27 through 35. A request contains "
        "224 prompt tokens and needs 32 generated tokens. Each pipeline stage "
        "can process 16 tokens per microbatch, so the prompt is split into "
        "ceil(224 / 16) microbatches. The scheduler starts one microbatch on "
        "P0 every 5 milliseconds, each stage takes 11 milliseconds per "
        "microbatch, and a stage cannot start a microbatch until the previous "
        "stage has finished it. Separately, a cache verifier receives slot "
        "ids in this order: 2, 3, 0, 1 on the producer and 1, 0, 3, 2 on the "
        "consumer. Explain whether the numeric slot order is safe, compute "
        "the earliest finish time for the last prompt microbatch on P3, and "
        "state the exact reason a wrong cache slot would corrupt the answer. "
        "The correct conclusion is"
    ),
]


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _server_cmd(binary_name: str) -> list[str]:
    if path := shutil.which(binary_name):
        return [path]

    release_binary = _project_root() / "target" / "release" / binary_name
    if release_binary.exists():
        return [str(release_binary)]

    return ["cargo", "run", "-r", "--bin", binary_name, "--"]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["PYO3_PYTHON"] = sys.executable
    env["PYTHONHOME"] = sys.base_prefix
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    if libdir := sysconfig.get_config_var("LIBDIR"):
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"

    python_dir = str(Path(__file__).parent.parent)
    site_packages = next((p for p in sys.path if "site-packages" in p), None)
    env["PYTHONPATH"] = f"{python_dir}" + (f":{site_packages}" if site_packages else "")
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    return env


class ManagedProcess:
    def __init__(
        self,
        label: str,
        cmd: list[str],
        log_file: Path,
        ready_url: str,
        cwd: Path,
        env: dict[str, str],
        timeout: int = STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.label = label
        self.cmd = cmd
        self.log_file = log_file
        self.ready_url = ready_url
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def __enter__(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_file.open("wb")
        print(f"[{self.label}] log: {self.log_file}")
        self.process = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
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
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} exited with code {self.process.returncode}\n"
                    f"log tail:\n{self.tail()}"
                )
            try:
                if requests.get(self.ready_url, timeout=1).status_code == 200:
                    print(f"[{self.label}] ready: {self.ready_url}")
                    time.sleep(2)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.label} did not become ready\nlog tail:\n{self.tail()}")

    def tail(self, limit: int = 4000) -> str:
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(errors="replace")[-limit:]


class MetaServer(ManagedProcess):
    def __init__(self, log_dir: Path, host: str) -> None:
        self.grpc_port = find_available_port()
        self.http_port = find_available_port()

        cmd = _server_cmd("pegaflow-metaserver")
        cmd.extend(
            [
                "--addr",
                f"{host}:{self.grpc_port}",
                "--http-addr",
                f"{host}:{self.http_port}",
                "--log-level",
                os.environ.get("PEGAFLOW_LOG_LEVEL", "debug"),
            ]
        )

        super().__init__(
            "metaserver",
            cmd,
            log_dir / "metaserver.log",
            f"http://{host}:{self.http_port}/health",
            _project_root(),
            _env(),
        )


class PegaServer(ManagedProcess):
    def __init__(
        self,
        label: str,
        devices: str,
        log_dir: Path,
        pool_size: str,
        transfer_backend: str,
        metaserver: MetaServer,
        nics: str,
        advertise_host: str,
    ) -> None:
        self.grpc_port = find_available_port()
        self.http_port = find_available_port()
        self.grpc_host = advertise_host

        cmd = _server_cmd("pegaflow-server")
        cmd.extend(
            [
                "--addr",
                f"{advertise_host}:{self.grpc_port}",
                "--http-addr",
                f"127.0.0.1:{self.http_port}",
                "--devices",
                devices,
                "--pool-size",
                pool_size,
                "--transfer-backend",
                transfer_backend,
                "--log-level",
                os.environ.get("PEGAFLOW_LOG_LEVEL", "debug"),
                "--disable-numa-affinity",
                "--enable-prometheus",
                "--metaserver-addr",
                f"http://{advertise_host}:{metaserver.grpc_port}",
                "--nics",
                nics,
            ]
        )
        if os.environ.get("PEGAFLOW_USE_HUGEPAGES", "0") == "1":
            cmd.append("--use-hugepages")

        super().__init__(
            f"pega-{label}",
            cmd,
            log_dir / f"pega-{label}.log",
            f"http://127.0.0.1:{self.http_port}/health",
            _project_root(),
            _env(),
        )

    @property
    def metrics_port(self) -> int:
        return self.http_port


class VllmReplica(ManagedProcess):
    def __init__(
        self,
        label: str,
        model: str,
        port: int,
        devices: str,
        pega_server: PegaServer,
        log_dir: Path,
        max_model_len: int,
    ) -> None:
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
        env = _env()
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["PEGAFLOW_PORT"] = str(pega_server.grpc_port)

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
            "--pipeline-parallel-size",
            str(PP_SIZE),
            "--tensor-parallel-size",
            str(TP_SIZE),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            os.environ.get(
                "PEGAFLOW_GPU_MEMORY_UTILIZATION",
                str(DEFAULT_GPU_MEMORY_UTILIZATION),
            ),
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--disable-hybrid-kv-cache-manager",
            "--kv-transfer-config",
            json.dumps(kv_config, separators=(",", ":")),
        ]
        cmd.extend(shlex.split(os.environ.get("PEGAFLOW_VLLM_EXTRA_ARGS", "")))

        super().__init__(
            f"vllm-{label}",
            cmd,
            log_dir / f"vllm-{label}.log",
            f"http://127.0.0.1:{port}/v1/models",
            _project_root(),
            env,
        )


def _require_gpus(count: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} visible GPUs")


def _metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def _metric_sum(
    before: dict[str, float],
    after: dict[str, float],
    names: tuple[str, ...],
) -> float:
    return sum(_metric_delta(before, after, name) for name in names)


def _metric_value(
    metrics_port: int,
    names: tuple[str, ...],
    *,
    labels: dict[str, str] | None = None,
) -> float:
    response = requests.get(f"http://localhost:{metrics_port}/metrics", timeout=5)
    response.raise_for_status()

    total = 0.0
    name_pattern = "|".join(re.escape(name) for name in names)
    pattern = re.compile(rf"^({name_pattern})(?:\{{([^}}]*)\}})?\s+([\d.eE+-]+)$")
    for line in response.text.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        metric_labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2) or ""))
        if labels is not None and any(
            metric_labels.get(key) != value for key, value in labels.items()
        ):
            continue
        total += float(match.group(3))
    return total


def _metric_value_delta(
    metrics_port: int,
    before: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], float],
    names: tuple[str, ...],
    *,
    labels: dict[str, str] | None = None,
) -> float:
    label_items = tuple(sorted((labels or {}).items()))
    key = (names, label_items)
    return _metric_value(metrics_port, names, labels=labels) - before.get(key, 0.0)


def _snapshot_metrics(
    metrics_port: int,
    series: tuple[tuple[tuple[str, ...], dict[str, str] | None], ...],
) -> dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], float]:
    return {
        (names, tuple(sorted((labels or {}).items()))): _metric_value(
            metrics_port,
            names,
            labels=labels,
        )
        for names, labels in series
    }


def _fetch_rpc_failures(
    metrics_port: int,
    *,
    ignored_methods: frozenset[str] = frozenset(),
) -> dict[str, float]:
    response = requests.get(f"http://localhost:{metrics_port}/metrics", timeout=5)
    response.raise_for_status()

    failures: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line.startswith("pegaflow_rpc_requests"):
            continue
        match = re.match(r"^pegaflow_rpc_requests(?:_total)?\{([^}]*)\}\s+([\d.eE+-]+)$", line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        if labels.get("status") == "ok":
            continue
        method = labels.get("method", "")
        if method in ignored_methods:
            continue
        key = f"{method}/{labels.get('status', '')}"
        failures[key] = failures.get(key, 0.0) + float(match.group(2))
    return failures


def _wait_for_metrics(
    metrics_port: int,
    before: dict[str, float],
    names: tuple[str, ...],
    timeout: int = 90,
) -> dict[str, float]:
    deadline = time.time() + timeout
    current = fetch_pegaflow_metrics(metrics_port)
    while time.time() < deadline:
        current = fetch_pegaflow_metrics(metrics_port)
        if any(_metric_delta(before, current, name) > 0 for name in names):
            return current
        time.sleep(2)
    return current


def _call_batch(port: int, prompts: list[str], max_tokens: int) -> list[dict[str, Any]]:
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": MODEL_ALIAS,
            "prompt": prompts,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 42,
            "logprobs": 5,
        },
        timeout=180,
    )
    response.raise_for_status()
    choices = sorted(response.json()["choices"], key=lambda choice: choice["index"])
    return [
        {
            "text": choice.get("text", ""),
            "token_logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
            "finish_reason": choice.get("finish_reason"),
        }
        for choice in choices
    ]


def _assert_outputs_match(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    mismatches = []
    for idx, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left != right:
            mismatches.append(
                f"{label} choice {idx} mismatch\n"
                f"expected={json.dumps(left, ensure_ascii=True, sort_keys=True)}\n"
                f"actual={json.dumps(right, ensure_ascii=True, sort_keys=True)}"
            )
    assert not mismatches, "\n\n".join(mismatches)


@pytest.mark.e2e
@pytest.mark.gpu
def test_pp4_p2p_matches_local_cache_load(
    model: str,
    max_model_len: int | None,
    pegaflow_transfer_backend: str,
    pegaflow_pool_size: str,
    tmp_path: Path,
) -> None:
    _require_gpus(8)

    nics = os.environ.get("PEGAFLOW_NICS", "")
    if not nics:
        pytest.skip("PEGAFLOW_NICS is required for PP4 P2P E2E")

    server_host = os.environ.get("PEGAFLOW_HOST_IP", "127.0.0.1")
    producer_devices = os.environ.get("PEGAFLOW_PRODUCER_GPUS", "0,1,2,3")
    consumer_devices = os.environ.get("PEGAFLOW_CONSUMER_GPUS", "4,5,6,7")
    source_devices = os.environ.get(
        "PEGAFLOW_SOURCE_SERVER_GPUS",
        f"{producer_devices},{consumer_devices}",
    )
    pool_size = os.environ.get("PEGAFLOW_POOL_SIZE", pegaflow_pool_size or DEFAULT_POOL_SIZE)
    request_max_tokens = int(os.environ.get("PEGAFLOW_REQUEST_MAX_TOKENS", "8"))
    server_max_model_len = int(
        os.environ.get("PEGAFLOW_MAX_MODEL_LEN", max_model_len or DEFAULT_MAX_MODEL_LEN)
    )

    log_dir = tmp_path / "p2p_pp4_logs"
    producer_port = find_available_port()
    local_consumer_port = find_available_port()
    rdma_consumer_port = find_available_port()

    with ExitStack() as stack:
        metaserver = stack.enter_context(MetaServer(log_dir, server_host))
        source_pega = stack.enter_context(
            PegaServer(
                "source",
                source_devices,
                log_dir,
                pool_size,
                pegaflow_transfer_backend,
                metaserver,
                nics,
                server_host,
            )
        )
        stack.enter_context(
            VllmReplica(
                "producer",
                model,
                producer_port,
                producer_devices,
                source_pega,
                log_dir,
                server_max_model_len,
            )
        )

        producer_before = fetch_pegaflow_metrics(source_pega.metrics_port)
        _call_batch(producer_port, PROMPTS, request_max_tokens)
        producer_after = _wait_for_metrics(
            source_pega.metrics_port,
            producer_before,
            (
                "pegaflow_save_bytes_total",
                "pegaflow_save_bytes",
                "pegaflow_cache_block_insertions_total",
                "pegaflow_cache_block_insertions",
            ),
        )

        with VllmReplica(
            "local-consumer",
            model,
            local_consumer_port,
            consumer_devices,
            source_pega,
            log_dir,
            server_max_model_len,
        ):
            local_before = fetch_pegaflow_metrics(source_pega.metrics_port)
            local_outputs = _call_batch(local_consumer_port, PROMPTS, request_max_tokens)
            local_after = _wait_for_metrics(
                source_pega.metrics_port,
                local_before,
                (
                    "pegaflow_load_bytes_total",
                    "pegaflow_load_bytes",
                    "pegaflow_cache_block_hits_total",
                    "pegaflow_cache_block_hits",
                ),
            )

        remote_pega = stack.enter_context(
            PegaServer(
                "remote",
                consumer_devices,
                log_dir,
                pool_size,
                pegaflow_transfer_backend,
                metaserver,
                nics,
                server_host,
            )
        )
        with VllmReplica(
            "rdma-consumer",
            model,
            rdma_consumer_port,
            consumer_devices,
            remote_pega,
            log_dir,
            server_max_model_len,
        ):
            rdma_before = fetch_pegaflow_metrics(remote_pega.metrics_port)
            rdma_series_before = _snapshot_metrics(
                remote_pega.metrics_port,
                (
                    (RDMA_FETCH_COUNTER_NAMES, {"status": "ok"}),
                    (RDMA_FETCH_COUNTER_NAMES, {"status": "error"}),
                    (RDMA_FETCH_BYTES_COUNTER_NAMES, {"status": "ok"}),
                ),
            )
            rdma_outputs = _call_batch(rdma_consumer_port, PROMPTS, request_max_tokens)
            rdma_after = _wait_for_metrics(
                remote_pega.metrics_port,
                rdma_before,
                (
                    "pegaflow_rdma_fetch_total",
                    "pegaflow_rdma_fetch_bytes_total",
                    "pegaflow_rdma_fetch_bytes",
                    "pegaflow_load_bytes_total",
                    "pegaflow_load_bytes",
                ),
            )

        # The producer computes the full prompt, while both consumers take the
        # external-prefix path and only recompute the tail. With PP this can
        # legitimately expose tiny numerical differences on close logits. The
        # P2P invariant is stricter and narrower: RDMA-loaded KV must produce
        # the same outputs as a same-server load from the same saved KV.
        _assert_outputs_match(local_outputs, rdma_outputs, label="rdma load")

        producer_saved = _metric_sum(
            producer_before,
            producer_after,
            (
                "pegaflow_save_bytes_total",
                "pegaflow_save_bytes",
                "pegaflow_cache_block_insertions_total",
                "pegaflow_cache_block_insertions",
            ),
        )
        local_loaded = _metric_sum(
            local_before,
            local_after,
            (
                "pegaflow_load_bytes_total",
                "pegaflow_load_bytes",
                "pegaflow_cache_block_hits_total",
                "pegaflow_cache_block_hits",
            ),
        )
        rdma_loaded = _metric_sum(
            rdma_before,
            rdma_after,
            (
                "pegaflow_rdma_fetch_total",
                "pegaflow_rdma_fetch_bytes_total",
                "pegaflow_rdma_fetch_bytes",
                "pegaflow_load_bytes_total",
                "pegaflow_load_bytes",
            ),
        )

        assert producer_saved > 0, f"producer produced no save evidence; logs={log_dir}"
        assert local_loaded > 0, f"local consumer produced no load evidence; logs={log_dir}"
        assert rdma_loaded > 0, f"RDMA consumer produced no fetch/load evidence; logs={log_dir}"
        assert (
            _metric_value_delta(
                remote_pega.metrics_port,
                rdma_series_before,
                RDMA_FETCH_COUNTER_NAMES,
                labels={"status": "ok"},
            )
            > 0
        ), f"RDMA consumer produced no successful RDMA fetch; logs={log_dir}"
        assert (
            _metric_value_delta(
                remote_pega.metrics_port,
                rdma_series_before,
                RDMA_FETCH_COUNTER_NAMES,
                labels={"status": "error"},
            )
            == 0
        ), f"RDMA consumer recorded RDMA fetch errors; logs={log_dir}"
        assert (
            _metric_value_delta(
                remote_pega.metrics_port,
                rdma_series_before,
                RDMA_FETCH_BYTES_COUNTER_NAMES,
                labels={"status": "ok"},
            )
            > 0
        ), f"RDMA consumer fetched no RDMA bytes; logs={log_dir}"

        source_failures = _fetch_rpc_failures(source_pega.metrics_port)
        # The RDMA consumer validates the read path. PegaFlow currently has no
        # read-only connector mode, so vLLM may submit a best-effort save for
        # the generated tail after the load has already completed.
        remote_failures = _fetch_rpc_failures(
            remote_pega.metrics_port,
            ignored_methods=frozenset({"save"}),
        )
        assert not source_failures, f"source RPC failures: {source_failures}; logs={log_dir}"
        assert not remote_failures, f"remote RPC failures: {remote_failures}; logs={log_dir}"
