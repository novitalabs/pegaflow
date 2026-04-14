"""Shared utilities for vLLM E2E and fuzz tests.

Extracted from test_vllm_e2e_correctness.py to enable reuse across test modules.
"""

import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

DEFAULT_VLLM_SEED = 42


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
    ):
        self.model = model
        self.port = port
        self.use_pegaflow = use_pegaflow
        self.pegaflow_port = pegaflow_port
        self.log_file = log_file
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
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
        env["VLLM_BATCH_INVARIANT"] = "1"

        if self.use_pegaflow and self.pegaflow_port is not None:
            env["PEGAFLOW_PORT"] = str(self.pegaflow_port)

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
            "0.8",
            "--attention-backend",
            "FLASH_ATTN",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
        ]

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.use_pegaflow:
            kv_config = {
                "kv_connector": "PegaKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
            cmd.extend(["--kv-transfer-config", json.dumps(kv_config)])

        server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
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

        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the vLLM server and all child processes."""
        if self.process:
            server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
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

    def __init__(self, log_file: Path | None = None, pool_size: str = "30gb"):
        self.grpc_port = find_available_port()
        self.http_port = find_available_port()
        self.pool_size = pool_size
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

        # pegaflow-server embeds Python via PyO3 (for CUDA device detection via torch)
        import sys
        import sysconfig

        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable
        env["PYTHONHOME"] = sys.base_prefix
        if libdir := sysconfig.get_config_var("LIBDIR"):
            env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
        python_dir = str(Path(__file__).parent.parent)
        site_packages = next((p for p in sys.path if "site-packages" in p), None)
        env["PYTHONPATH"] = f"{python_dir}" + (f":{site_packages}" if site_packages else "")

        print(f"\n[PegaFlow Server] cargo run -r on gRPC={self.grpc_port}, HTTP={self.http_port}")

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
                self.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                if self.process:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=5)
            print("pegaflow-server stopped.\n")
        if self.log_handle:
            self.log_handle.close()


def call_openai_api(
    port: int,
    model: str,
    prompt: str,
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
