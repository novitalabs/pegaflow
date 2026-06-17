"""Shared utilities for vLLM E2E tests.

Extracted from test_vllm_e2e_correctness.py to enable reuse across test modules.
"""

import json
import os
import re
import signal
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import requests

DEFAULT_VLLM_SEED = 42


def _detect_pegaflow_cargo_features() -> list[str]:
    try:
        import torch
    except ImportError:
        return []

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return []

    # Keep parity with the default feature set (cuda-12 + rdma) so the e2e
    # server build shares the cargo cache with maturin dev builds instead of
    # invalidating it on every alternation.
    major = cuda_version.split(".", maxsplit=1)[0]
    return ["cuda-13", "rdma"] if major == "13" else []


class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    def __init__(
        self,
        model: str,
        port: int,
        use_pegaflow: bool = False,
        pegaflow_port: int | None = None,
        log_file: Path | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        extra_args: list[str] | None = None,
        startup_timeout: int = 600,
        use_noop_connector: bool = False,
        kv_transfer_config: dict[str, object] | None = None,
        server_label: str | None = None,
        env_overrides: dict[str, str] | None = None,
        transfer_backend: str | None = None,
    ):
        self.model = model
        self.port = port
        self.use_pegaflow = use_pegaflow
        self.pegaflow_port = pegaflow_port
        self.transfer_backend = transfer_backend
        self.log_file = log_file
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.use_noop_connector = use_noop_connector
        self.kv_transfer_config = kv_transfer_config
        self.server_label = server_label
        self.env_overrides = env_overrides or {}
        self.health_endpoints = ["/health", "/v1/models"]
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def __enter__(self):
        """Start the vLLM server."""
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("tensor/pipeline parallel sizes must be >= 1")
        world_size = self.tensor_parallel_size * self.pipeline_parallel_size
        if world_size > 4:
            raise ValueError(
                f"tensor_parallel_size * pipeline_parallel_size must be <= 4 (got {world_size})"
            )

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["VLLM_BATCH_INVARIANT"] = "1"

        if self.pegaflow_port is not None:
            env["PEGAFLOW_PORT"] = str(self.pegaflow_port)
        env.update(self.env_overrides)

        # Resolve vllm binary from the current Python environment's bin dir
        import sys

        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--attention-backend",
            # FLASH_ATTN keeps outputs batch-invariant for the baseline/warm
            # comparison, but it cannot serve MLA models on every GPU
            # generation (e.g. sm_120 has no FlashMLA); allow overriding with
            # an MLA-capable backend like TRITON_MLA there.
            os.environ.get("VLLM_TEST_ATTN_BACKEND", "FLASH_ATTN"),
            "--generation-config",
            "vllm",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
        ]

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.kv_transfer_config is not None:
            cmd.extend(["--kv-transfer-config", json.dumps(self.kv_transfer_config)])
        elif self.use_pegaflow or self.use_noop_connector:
            connector_name = "PegaKVConnector" if self.use_pegaflow else "NoopKVConnector"
            kv_config: dict[str, object] = {
                "kv_connector": connector_name,
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
            # The server no longer selects a transfer backend; the connector
            # does. Force it here so --pegaflow-transfer-backend still exercises
            # the chosen path end-to-end, regardless of the model's MLA default.
            if self.use_pegaflow and self.transfer_backend is not None:
                kv_config["kv_connector_extra_config"] = {
                    "pegaflow.transfer_backend": self.transfer_backend,
                }
            cmd.extend(["--kv-transfer-config", json.dumps(kv_config)])

        cmd.extend(self.extra_args)

        server_label = self.server_label or (
            "PegaFlow" if self.use_pegaflow or self.kv_transfer_config is not None else "Baseline"
        )
        print(f"\n[{server_label}] Starting vLLM server on port {self.port}")

        if self.log_file:
            print(f"[{server_label}] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                preexec_fn=os.setsid,
            )

        self._wait_for_ready(timeout=self.startup_timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the vLLM server and all child processes."""
        if self.process:
            server_label = self.server_label or (
                "PegaFlow"
                if self.use_pegaflow or self.kv_transfer_config is not None
                else "Baseline"
            )
            print(f"\n[{server_label}] Stopping vLLM server...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                if self.process:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=5)
            print("Server stopped.\n")

        if self.log_handle:
            self.log_handle.close()

    def _wait_for_ready(self, timeout: int = 180):
        """Wait for the server to be ready to accept requests."""
        start_time = time.time()
        print("Waiting for server to be ready...")
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                hint = f", see {self.log_file}" if self.log_file else ""
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}{hint}")
            for endpoint in self.health_endpoints:
                url = f"http://localhost:{self.port}{endpoint}"
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        print(f"Server is ready! (checked {endpoint})\n")
                        time.sleep(2)  # Extra buffer
                        return
                except requests.exceptions.RequestException:
                    continue
            time.sleep(2)

        raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def fetch_pegaflow_metrics(metrics_port: int) -> dict[str, float]:
    """Fetch and parse Prometheus metrics from PegaFlow server.

    Args:
        metrics_port: Port where PegaFlow exposes /metrics endpoint.

    Returns:
        Dict mapping metric name to value (for counters/gauges).
    """
    url = f"http://localhost:{metrics_port}/metrics"
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    metrics = {}
    for line in response.text.splitlines():
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue
        # Parse: metric_name{labels} value or metric_name value
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\{[^}]*\})?\s+([\d.eE+-]+)$", line)
        if match:
            name, value = match.groups()
            # Accumulate values for metrics with labels (e.g., sum across all labels)
            metrics[name] = metrics.get(name, 0) + float(value)
    return metrics


def fetch_pegaflow_rpc_failures(metrics_port: int, method: str | None = None) -> dict[str, float]:
    """Return non-ok RPC counts keyed by ``"method/status"``.

    ``fetch_pegaflow_metrics`` collapses label sets, so it cannot tell a failed
    RPC from a successful one. This keeps the ``method``/``status`` labels so a
    test can assert that no connector<->server RPC failed. Pass ``method`` to
    restrict to one RPC; ``None`` (default) reports failures across all methods.
    """
    url = f"http://localhost:{metrics_port}/metrics"
    response = requests.get(url, timeout=5)
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
        rpc = labels.get("method", "")
        if method is not None and rpc != method:
            continue
        key = f"{rpc}/{labels.get('status', '')}"
        failures[key] = failures.get(key, 0.0) + float(match.group(2))
    return failures


def check_pegaflow_server(metrics_port: int) -> bool:
    """Check if PegaFlow server is running and metrics endpoint is available."""
    try:
        fetch_pegaflow_metrics(metrics_port)
        return True
    except requests.exceptions.RequestException:
        return False


def find_available_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class PegaFlowServer:
    """Context manager for pegaflow-server lifecycle.

    Auto-starts pegaflow-server with prometheus metrics enabled.
    Picks random available ports for gRPC and HTTP endpoints.
    """

    def __init__(
        self,
        log_file: Path | None = None,
        pool_size: str = "30gb",
        log_level: str | None = None,
        cargo_features: list[str] | None = None,
        use_hugepages: bool = False,
    ):
        self.grpc_port = find_available_port()
        self.http_port = find_available_port()
        self.pool_size = pool_size
        self.log_level = log_level
        self.use_hugepages = use_hugepages
        self.cargo_features = (
            cargo_features if cargo_features is not None else _detect_pegaflow_cargo_features()
        )
        self.log_file = log_file
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    @property
    def metrics_port(self) -> int:
        return self.http_port

    def __enter__(self):
        project_root = Path(__file__).parent.parent.parent

        cmd = [
            "cargo",
            "run",
            "-r",
        ]
        if self.cargo_features:
            cmd.append("--no-default-features")
            cmd.extend(["--features", ",".join(self.cargo_features)])
        cmd.extend(
            [
                "--bin",
                "pegaflow-server",
                "--",
                "--addr",
                f"127.0.0.1:{self.grpc_port}",
                "--http-addr",
                f"0.0.0.0:{self.http_port}",
                "--pool-size",
                self.pool_size,
                "--enable-prometheus",
            ]
        )
        if self.use_hugepages:
            cmd.append("--use-hugepages")
        if self.log_level is not None:
            cmd.extend(["--log-level", self.log_level])

        # pegaflow-server embeds Python via PyO3 (for CUDA device detection via torch)
        import sys
        import sysconfig

        env = os.environ.copy()
        # The connector reports global device ids (it un-maps
        # CUDA_VISIBLE_DEVICES before registering); the server must see every
        # GPU so those ids and the IPC tensor devices line up. Without this, a
        # masked pytest run (e.g. CUDA_VISIBLE_DEVICES=1,2 on a shared box)
        # fails registration with "pinned to device N but got M".
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONHASHSEED"] = "0"
        env["PYO3_PYTHON"] = sys.executable
        env["PYTHONHOME"] = sys.base_prefix
        if libdir := sysconfig.get_config_var("LIBDIR"):
            env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
        python_dir = str(Path(__file__).parent.parent)
        site_packages = next((p for p in sys.path if "site-packages" in p), None)
        env["PYTHONPATH"] = f"{python_dir}" + (f":{site_packages}" if site_packages else "")

        feature_label = ",".join(self.cargo_features) or "default"
        print(
            f"\n[PegaFlow Server] cargo run -r features={feature_label} "
            f"on gRPC={self.grpc_port}, HTTP={self.http_port}"
        )

        if self.log_file:
            print(f"[PegaFlow Server] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

        self._wait_for_ready()
        return self

    def _wait_for_ready(self, timeout: int = 300):
        """Wait for pegaflow-server HTTP health endpoint."""
        start = time.time()
        print("Waiting for pegaflow-server to be ready...")
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"pegaflow-server exited with code {self.process.returncode}")
            try:
                resp = requests.get(f"http://localhost:{self.http_port}/health", timeout=1)
                if resp.status_code == 200:
                    print("pegaflow-server is ready!\n")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        raise TimeoutError(f"pegaflow-server did not become ready within {timeout}s")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            print("\n[PegaFlow Server] Stopping...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                if self.process:
                    with suppress(ProcessLookupError, OSError):
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    try:
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print("pegaflow-server did not exit after SIGKILL")
            print("pegaflow-server stopped.\n")
        if self.log_handle:
            self.log_handle.close()


def call_openai_api(
    port: int,
    model: str,
    prompt: str | list[int],
    max_tokens: int = 50,
    temperature: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Call vLLM's OpenAI-compatible API.

    Returns dict with 'text' and 'logprobs' (if available).
    """
    url = f"http://localhost:{port}/v1/completions"

    if seed is None:
        seed = DEFAULT_VLLM_SEED

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": 5,  # Request logprobs for detailed comparison
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "logprobs": choice.get("logprobs"),
        "finish_reason": choice["finish_reason"],
        "usage": data.get("usage"),
    }
