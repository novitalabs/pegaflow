#!/usr/bin/env python3
"""Run a direct TP4 P/D CPU-staging push smoke test.

This launcher intentionally does not require a router binary. It starts P/D
PegaFlow + vLLM instances, then sends paired OpenAI completion requests:

- D request prepares the receive lease and waits for IMM.
- P request computes the same prompt and pushes KV to D via RDMA WRITE_WITH_IMM.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _default_host() -> str:
    env_host = os.environ.get("PEGAFLOW_PD_HOST")
    if env_host:
        return env_host
    candidates: list[str] = []
    try:
        candidates.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        candidates.extend(output.split())
    except Exception:
        pass
    for candidate in candidates:
        if candidate.startswith("10."):
            return candidate
    for candidate in candidates:
        if not candidate.startswith("127."):
            return candidate
    return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models/Qwen/Qwen3-8b")
    parser.add_argument("--served-model-name", default="pd-test")
    parser.add_argument("--host", default=_default_host())
    parser.add_argument("--p-gpus", default="0,2,4,6")
    parser.add_argument("--d-gpus", default="1,3,5,7")
    parser.add_argument("--nics", default="mlx5_1,mlx5_2,mlx5_3,mlx5_4")
    parser.add_argument("--pool-size", default="32gb")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--metaserver-port", type=int, default=51056)
    parser.add_argument("--p-engine-port", type=int, default=51055)
    parser.add_argument("--d-engine-port", type=int, default=51155)
    parser.add_argument("--p-vllm-port", type=int, default=18100)
    parser.add_argument("--d-vllm-port", type=int, default=18200)
    parser.add_argument("--log-dir", default="examples/pd_push_logs")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-running", action="store_true")
    return parser.parse_args()


class Stack:
    def __init__(self, repo: Path, log_dir: Path):
        self.repo = repo
        self.log_dir = log_dir
        self.procs: list[tuple[str, subprocess.Popen, Path]] = []
        self.handles = []

    def start(self, name: str, cmd: list[str], env: dict[str, str]) -> None:
        log = self.log_dir / f"{name}.log"
        print(f"START {name}: {' '.join(cmd)}", flush=True)
        handle = log.open("w")
        self.handles.append(handle)
        proc = subprocess.Popen(
            cmd,
            cwd=self.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.procs.append((name, proc, log))

    def assert_alive(self) -> None:
        for name, proc, log in self.procs:
            if proc.poll() is None:
                continue
            tail = ""
            try:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(f"process {name} exited rc={proc.returncode}; log={log}\n{tail}")

    def cleanup(self) -> None:
        for _, proc, _ in reversed(self.procs):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 30
        for _, proc, _ in reversed(self.procs):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    proc.kill()
        for handle in self.handles:
            handle.close()


def wait_http(stack: Stack, url: str, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        stack.assert_alive()
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"timeout waiting for {url}: {last_error}")


def post_json(url: str, payload: dict, timeout_s: float = 900.0) -> tuple[float, dict]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return time.perf_counter() - start, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def completion_payload(
    served_model: str,
    prompt: str,
    kv_transfer_params: dict | None = None,
) -> dict:
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    if kv_transfer_params is not None:
        payload["kv_transfer_params"] = kv_transfer_params
    return payload


def pd_pair(args: argparse.Namespace, prompt: str, label: str) -> dict:
    pd_request_id = f"pd-{label}-{uuid.uuid4().hex[:12]}"
    d_payload = completion_payload(
        args.served_model_name,
        prompt,
        {
            "pegaflow_pd_push": True,
            "pd_request_id": pd_request_id,
            "dst_instance_id": "pd-d0",
        },
    )
    p_payload = completion_payload(
        args.served_model_name,
        prompt,
        {
            "role": "source",
            "pegaflow_pd_push": True,
            "pd_request_id": pd_request_id,
            "d_pegaflow_addr": f"{args.host}:{args.d_engine_port}",
            "dst_instance_id": "pd-d0",
        },
    )
    p_url = f"http://{args.host}:{args.p_vllm_port}/v1/completions"
    d_url = f"http://{args.host}:{args.d_vllm_port}/v1/completions"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        d_future = executor.submit(post_json, d_url, d_payload)
        time.sleep(0.25)
        p_future = executor.submit(post_json, p_url, p_payload)
        p_elapsed, p_result = p_future.result()
        d_elapsed, d_result = d_future.result()
    return {
        "label": label,
        "pd_request_id": pd_request_id,
        "p_elapsed_s": round(p_elapsed, 4),
        "d_elapsed_s": round(d_elapsed, 4),
        "usage": d_result.get("usage"),
        "p_text": p_result.get("choices", [{}])[0].get("text", "")[:80],
        "d_text": d_result.get("choices", [{}])[0].get("text", "")[:80],
    }


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.log_dir) / time.strftime("direct_pd_smoke_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"LOG_DIR {run_dir}", flush=True)

    if not args.skip_build:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "pegaflow-metaserver",
                "-p",
                "pegaflow-server",
                "--bins",
            ],
            cwd=repo,
            check=True,
        )

    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo / "python") + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env.setdefault("RUST_LOG", "info,pegaflow_core=info,pegaflow_server=info")
    nics = [nic.strip() for nic in args.nics.split(",") if nic.strip()]

    stack = Stack(repo, run_dir)
    try:
        stack.start(
            "metaserver",
            [
                str(repo / "target/release/pegaflow-metaserver"),
                "--addr",
                f"{args.host}:{args.metaserver_port}",
                "--http-addr",
                f"{args.host}:19192",
                "--log-level",
                "info",
            ],
            env.copy(),
        )
        time.sleep(2)

        for name, port, http_port, devices in (
            ("p-engine", args.p_engine_port, 19191, args.p_gpus),
            ("d-engine", args.d_engine_port, 19193, args.d_gpus),
        ):
            stack.start(
                name,
                [
                    str(repo / "target/release/pegaflow-server"),
                    "--addr",
                    f"{args.host}:{port}",
                    "--http-addr",
                    f"{args.host}:{http_port}",
                    "--pool-size",
                    args.pool_size,
                    "--devices",
                    devices,
                    "--nics",
                    *nics,
                    "--metaserver-addr",
                    f"http://{args.host}:{args.metaserver_port}",
                    "--log-level",
                    "info",
                ],
                env.copy(),
            )
            time.sleep(4)

        kv_config = json.dumps(
            {
                "kv_connector": "PegaKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
        )

        def vllm_cmd(port: int) -> list[str]:
            return [
                "vllm",
                "serve",
                args.model,
                "--served-model-name",
                args.served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--tensor-parallel-size",
                "4",
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                str(args.max_model_len),
                "--trust-remote-code",
                "--no-enable-prefix-caching",
                "--kv-transfer-config",
                kv_config,
            ]

        p_env = env.copy()
        p_env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.p_gpus,
                "PEGAFLOW_INSTANCE_ID": "pd-p0",
                "PEGAFLOW_HOST": f"http://{args.host}",
                "PEGAFLOW_PORT": str(args.p_engine_port),
                "PEGAFLOW_KV_EGRESS": "1",
                "PEGAFLOW_KV_EGRESS_NICS": args.nics,
                "PEGAFLOW_RDMA_NICS": args.nics,
            }
        )
        d_env = env.copy()
        d_env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.d_gpus,
                "PEGAFLOW_INSTANCE_ID": "pd-d0",
                "PEGAFLOW_HOST": f"http://{args.host}",
                "PEGAFLOW_PORT": str(args.d_engine_port),
                "PEGAFLOW_RDMA_NICS": args.nics,
            }
        )
        stack.start("p-vllm", vllm_cmd(args.p_vllm_port), p_env)
        time.sleep(8)
        stack.start("d-vllm", vllm_cmd(args.d_vllm_port), d_env)

        wait_http(stack, f"http://{args.host}:{args.p_vllm_port}/v1/models")
        wait_http(stack, f"http://{args.host}:{args.d_vllm_port}/v1/models")
        print("READY", flush=True)

        for side, port in (("p", args.p_vllm_port), ("d", args.d_vllm_port)):
            elapsed, _ = post_json(
                f"http://{args.host}:{port}/v1/completions",
                completion_payload(args.served_model_name, "warmup"),
                timeout_s=300,
            )
            print(f"BASELINE_WARMUP {side} {elapsed:.4f}s", flush=True)

        results = []
        cases = [
            ("hello", "hello"),
            ("short_block", "hello " * 16),
            ("medium", "PegaFlow validates CPU staging push. " * 180),
        ]
        for label, prompt in cases:
            result = pd_pair(args, prompt, label)
            results.append(result)
            print("PD_RESULT", json.dumps(result, ensure_ascii=False), flush=True)

        concurrent_cases = [
            (f"multi{i}", (f"Concurrent P/D validation {i}. " * 160))
            for i in range(4)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(pd_pair, args, prompt, label)
                for label, prompt in concurrent_cases
            ]
            for future in concurrent.futures.as_completed(futures, timeout=1200):
                result = future.result()
                results.append(result)
                print("PD_RESULT", json.dumps(result, ensure_ascii=False), flush=True)

        direct_prompt = "PegaFlow validates CPU staging push. " * 180
        elapsed, direct_result = post_json(
            f"http://{args.host}:{args.d_vllm_port}/v1/completions",
            completion_payload(args.served_model_name, direct_prompt),
            timeout_s=300,
        )
        print(
            "DIRECT_D_RESULT",
            json.dumps(
                {
                    "elapsed_s": round(elapsed, 4),
                    "usage": direct_result.get("usage"),
                    "text": direct_result.get("choices", [{}])[0].get("text", "")[:80],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        print("SUMMARY", json.dumps(results, ensure_ascii=False, indent=2), flush=True)

        if args.keep_running:
            while True:
                stack.assert_alive()
                time.sleep(5)
    finally:
        stack.cleanup()
        print(f"CLEANED {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
