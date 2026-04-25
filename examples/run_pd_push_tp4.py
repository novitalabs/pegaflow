#!/usr/bin/env python3
"""Run a local TP4 P/D CPU-staging push stack for validation.

This is intentionally a toy launcher for one host with eight visible GPUs:

- PegaFlow metaserver
- P-side PegaFlow server
- D-side PegaFlow server
- TP4 P vLLM
- TP4 D vLLM
- pegaflow-router with P/D push metadata injection

The router sends P and D requests concurrently. It only injects rendezvous
identity; it does not pass CUDA device IDs, TP ranks, token counts, or layer
names.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_SMOKE_PROMPT = " ".join(["PegaFlow validation prompt with repeated cache blocks."] * 180)


def _default_host() -> str:
    env_host = os.environ.get("PEGAFLOW_PD_HOST")
    if env_host:
        return env_host

    candidates: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        candidates.extend(info[4][0] for info in infos)
    except OSError:
        pass

    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        candidates.extend(output.split())
    except Exception:
        pass

    seen: set[str] = set()
    filtered = []
    for candidate in candidates:
        if candidate in seen or candidate.startswith("127."):
            continue
        seen.add(candidate)
        filtered.append(candidate)

    for candidate in filtered:
        if candidate.startswith("10."):
            return candidate
    if filtered:
        return filtered[0]
    return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models/Qwen/Qwen3-8b")
    parser.add_argument("--served-model-name", default="pd-test")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--p-gpus", default="0,2,4,6")
    parser.add_argument("--d-gpus", default="1,3,5,7")
    parser.add_argument("--nics", default="mlx5_1,mlx5_2,mlx5_3,mlx5_4")
    parser.add_argument("--host", default=_default_host())
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--p-vllm-port", type=int, default=8100)
    parser.add_argument("--d-vllm-port", type=int, default=8200)
    parser.add_argument("--metaserver-port", type=int, default=50056)
    parser.add_argument("--metaserver-http-port", type=int, default=19092)
    parser.add_argument("--p-engine-port", type=int, default=50055)
    parser.add_argument("--d-engine-port", type=int, default=50155)
    parser.add_argument("--p-engine-http-port", type=int, default=19091)
    parser.add_argument("--d-engine-http-port", type=int, default=19093)
    parser.add_argument("--pool-size", default="16gb")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--log-dir", default="examples/pd_push_logs")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--exit-after-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_SMOKE_PROMPT)
    parser.add_argument("--smoke-count", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def start(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    processes: list[subprocess.Popen],
    log_handles: list,
) -> None:
    log_file = log_dir / f"{name}.log"
    print(f"[{name}] {' '.join(cmd)}")
    print(f"[{name}] log: {log_file}")
    handle = log_file.open("w")
    log_handles.append(handle)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    processes.append(proc)


def wait_http_json(url: str, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def post_json(url: str, payload: dict, timeout_s: float = 300.0) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir) / time.strftime("pd_push_tp4_%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.tp != 4:
        raise SystemExit("this validation launcher is intentionally TP4-only; use --tp 4")
    if len(args.p_gpus.split(",")) != 4 or len(args.d_gpus.split(",")) != 4:
        raise SystemExit("--p-gpus and --d-gpus must each contain four visible GPU ids")

    if not args.skip_build:
        run(
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
        )

    bin_dir = repo / "target" / "release"
    metaserver_bin = bin_dir / "pegaflow-metaserver"
    engine_bin = bin_dir / "pegaflow-server"
    router_bin = bin_dir / "pegaflow-router"
    vllm_bin = shutil.which("vllm")
    if vllm_bin is None:
        raise SystemExit("vllm executable not found in PATH; activate the vLLM venv first")

    host = args.host
    metaserver_addr = f"{host}:{args.metaserver_port}"
    metaserver_url = f"http://{metaserver_addr}"
    p_engine_addr = f"{host}:{args.p_engine_port}"
    d_engine_addr = f"{host}:{args.d_engine_port}"
    p_vllm_url = f"http://{host}:{args.p_vllm_port}"
    d_vllm_url = f"http://{host}:{args.d_vllm_port}"
    router_url = f"http://{host}:{args.router_port}"

    common_env = os.environ.copy()
    common_env.setdefault("VLLM_USE_V1", "1")
    common_env.setdefault("PYTHONUNBUFFERED", "1")
    python_path = str(repo / "python")
    common_env["PYTHONPATH"] = (
        python_path
        if not common_env.get("PYTHONPATH")
        else f"{python_path}:{common_env['PYTHONPATH']}"
    )

    processes: list[subprocess.Popen] = []
    log_handles: list = []

    def cleanup(*_: object) -> None:
        print("\nshutting down...")
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(processes):
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        for handle in log_handles:
            handle.close()
        print(f"logs: {log_dir}")

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    commands: list[tuple[str, list[str], dict[str, str]]] = []
    commands.append(
        (
            "metaserver",
            [
                str(metaserver_bin),
                "--addr",
                metaserver_addr,
                "--http-addr",
                f"{host}:{args.metaserver_http_port}",
            ],
            common_env.copy(),
        )
    )

    for name, port, http_port, gpus in (
        ("p-engine", args.p_engine_port, args.p_engine_http_port, args.p_gpus),
        ("d-engine", args.d_engine_port, args.d_engine_http_port, args.d_gpus),
    ):
        env = common_env.copy()
        commands.append(
            (
                name,
                [
                    str(engine_bin),
                    "--addr",
                    f"{host}:{port}",
                    "--http-addr",
                    f"{host}:{http_port}",
                    "--pool-size",
                    args.pool_size,
                    "--devices",
                    gpus,
                    "--nics",
                    *args.nics.split(","),
                    "--metaserver-addr",
                    metaserver_url,
                ],
                env,
            )
        )

    kv_config = {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
    }

    def vllm_cmd(port: int) -> list[str]:
        return [
            vllm_bin,
            "serve",
            args.model,
            "--served-model-name",
            args.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(args.tp),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps(kv_config),
        ]

    p_env = common_env.copy()
    p_env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.p_gpus,
            "PEGAFLOW_INSTANCE_ID": "pd-p0",
            "PEGAFLOW_HOST": f"http://{host}",
            "PEGAFLOW_PORT": str(args.p_engine_port),
            "PEGAFLOW_KV_EGRESS": "1",
            "PEGAFLOW_KV_EGRESS_NICS": args.nics,
            "PEGAFLOW_RDMA_NICS": args.nics,
        }
    )
    commands.append(("p-vllm", vllm_cmd(args.p_vllm_port), p_env))

    d_env = common_env.copy()
    d_env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.d_gpus,
            "PEGAFLOW_INSTANCE_ID": "pd-d0",
            "PEGAFLOW_HOST": f"http://{host}",
            "PEGAFLOW_PORT": str(args.d_engine_port),
            "PEGAFLOW_RDMA_NICS": args.nics,
        }
    )
    commands.append(("d-vllm", vllm_cmd(args.d_vllm_port), d_env))

    commands.append(
        (
            "router",
            [
                str(router_bin),
                "--host",
                "0.0.0.0",
                "--port",
                str(args.router_port),
                "--prefill",
                p_vllm_url,
                "--decode",
                d_vllm_url,
                "--pd-push",
                "--decode-pegaflow",
                d_engine_addr,
                "--decode-instance",
                "pd-d0",
            ],
            common_env.copy(),
        )
    )

    print(f"logs: {log_dir}")
    for name, cmd, env in commands:
        print(f"{name}: {' '.join(cmd)}")
        if name.endswith("vllm") or name.endswith("engine"):
            print(f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")

    if args.dry_run:
        return 0

    try:
        for name, cmd, env in commands:
            start(
                name,
                cmd,
                cwd=repo,
                env=env,
                log_dir=log_dir,
                processes=processes,
                log_handles=log_handles,
            )
            if name == "metaserver":
                time.sleep(2)
            elif name.endswith("engine"):
                time.sleep(4)
            elif name.endswith("vllm"):
                time.sleep(8)

        wait_http_json(f"{p_vllm_url}/v1/models", timeout_s=600)
        wait_http_json(f"{d_vllm_url}/v1/models", timeout_s=600)

        if not args.no_smoke:
            payload = {
                "model": args.served_model_name,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
            }
            for idx in range(args.smoke_count):
                print(f"smoke request {idx + 1}/{args.smoke_count} -> {router_url}/v1/completions")
                result = post_json(f"{router_url}/v1/completions", payload)
                print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
            if args.exit_after_smoke:
                cleanup()
                return 0

        print(f"running; logs: {log_dir}")
        while True:
            for idx, proc in enumerate(processes):
                ret = proc.poll()
                if ret is not None:
                    print(f"process {idx} exited with code {ret}")
                    cleanup()
                    return ret
            time.sleep(5)
    except Exception:
        cleanup()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
