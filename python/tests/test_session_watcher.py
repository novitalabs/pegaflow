"""Contract test for the Session-stream liveness feature.

When a vllm process dies while holding registered CUDA IPC mappings,
pegaflow-server must release them. Without this, the IPC handles pin
the GPU memory on the server side and a DaemonSet-restarted vllm will
OOM at `torch.cuda.init()`.

Approach: spawn a worker subprocess that registers a real GPU tensor
as an IPC context and opens a session, then SIGKILL it. The server
logs how many CUDA tensors it dropped on session close; we assert that
log line appears within a short deadline.

Full GPU-memory-released-to-driver verification (nvidia-smi delta) is
intentionally not automated — the server's drop path already calls
`torch.cuda.empty_cache()` after dropping IPC handles, so observing
"dropped N CUDA tensors" implies release.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest

HELPER = Path(__file__).parent / "session_crash_helper.py"
READY_TIMEOUT_SEC = 30.0
CLEANUP_LOG_DEADLINE_SEC = 5.0


def _worker_env() -> dict[str, str]:
    """Match pega_server fixture's environment so the helper finds the
    same libpython / pegaflow module the server sees."""
    env = os.environ.copy()
    if libdir := sysconfig.get_config_var("LIBDIR"):
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    python_dir = Path(__file__).parent.parent
    site_packages = next((p for p in sys.path if "site-packages" in p), None)
    env["PYTHONPATH"] = str(python_dir) + (f":{site_packages}" if site_packages else "")
    return env


def test_crashed_client_releases_ipc(pega_server, tmp_path):
    instance_id = f"inst-crash-{os.getpid()}"
    ready_file = tmp_path / "ready"

    worker = subprocess.Popen(
        [sys.executable, str(HELPER), pega_server.endpoint, instance_id, str(ready_file)],
        env=_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if ready_file.exists():
                break
            if worker.poll() is not None:
                out, err = worker.communicate()
                pytest.fail(
                    f"worker exited early with code {worker.returncode}\n"
                    f"stdout={out.decode(errors='replace')}\n"
                    f"stderr={err.decode(errors='replace')}"
                )
            time.sleep(0.05)
        else:
            pytest.fail(f"worker did not become ready within {READY_TIMEOUT_SEC}s")

        worker.send_signal(signal.SIGKILL)
        worker.wait(timeout=5)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)

    target = f"dropped 1 CUDA tensors for instance {instance_id}"
    deadline = time.monotonic() + CLEANUP_LOG_DEADLINE_SEC
    while time.monotonic() < deadline:
        if target in pega_server.read_logs():
            return
        time.sleep(0.05)

    tail = pega_server.read_logs()[-2000:]
    pytest.fail(f"expected cleanup log not seen within {CLEANUP_LOG_DEADLINE_SEC}s\n---\n{tail}")
