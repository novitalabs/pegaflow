"""Contract test: a SIGKILLed client must never take pegaflow-server down.

When a vllm process dies, session cleanup drops its CUDA IPC tensors
(unmapping the IPC VA ranges). Save tasks already queued on the per-device
save worker still hold raw device pointers into those mappings. If the
unmap wins the race against a queued task, the task's cuMemcpyDtoHAsync
either SIGSEGVs the server (VA unmapped — the common case, since a
restarted vllm takes minutes to re-register) or silently copies another
mapping's bytes and seals them under the dead client's hashes (VA reused).

Driver-level evidence for both outcomes: scripts/cuda_ipc_kill_probe.py
(verified on RTX 5070 Ti and H20-3e).

This test drives the real server: a worker subprocess keeps a deep,
slow-draining save backlog in flight (stride-2 block ids defeat copy-range
merging, so each task holds the save worker for tens of milliseconds),
gets SIGKILLed mid-backlog, and the server must stay alive and healthy
while session cleanup races the queued saves. The race is probabilistic
but hits reliably with this sizing; the observed pre-fix signature is
SIGABRT via `c10::AcceleratorError: CUDA error: unspecified launch
failure` thrown in a tensor destructor after the stale copy poisons the
CUDA context.
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest

from .conftest import PegaServerProcess, find_available_port

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

HELPER = Path(__file__).parent / "save_backlog_kill_helper.py"
READY_TIMEOUT_SEC = 60.0
# Enough completed saves to know the submit threads outpace the save worker,
# i.e. the server-side queue has a real backlog when we kill.
MIN_SAVES_BEFORE_KILL = 8
CLEANUP_LOG_DEADLINE_SEC = 10.0
SURVIVAL_WATCH_SEC = 12.0


def _worker_env() -> dict[str, str]:
    """Match the server fixture's environment so the helper finds the same
    libpython / pegaflow module the server sees."""
    env = os.environ.copy()
    if libdir := sysconfig.get_config_var("LIBDIR"):
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    python_dir = Path(__file__).parent.parent
    site_packages = next((p for p in sys.path if "site-packages" in p), None)
    env["PYTHONPATH"] = str(python_dir) + (f":{site_packages}" if site_packages else "")
    return env


@pytest.fixture
def backlog_server(monkeypatch):
    """Dedicated server with a pool large enough for a multi-save backlog."""
    monkeypatch.setenv("RUST_LOG", "info,pegaflow_core=debug,pegaflow_server=debug")
    server = PegaServerProcess(port=find_available_port(), pool_size="3gb")
    if not server.start() or not server._binary_path:
        pytest.skip("PegaServer binary not found or failed to start")
    yield server
    server.stop()


def _count_progress_lines(progress_file: Path) -> int:
    if not progress_file.exists():
        return 0
    return len(progress_file.read_text(errors="replace").splitlines())


def test_server_survives_sigkill_with_save_backlog(backlog_server, tmp_path):
    instance_id = f"inst-backlog-{os.getpid()}"
    ready_file = tmp_path / "ready"
    progress_file = tmp_path / "progress"

    worker = subprocess.Popen(
        [
            sys.executable,
            str(HELPER),
            backlog_server.endpoint,
            instance_id,
            "test-ns",
            str(ready_file),
            str(progress_file),
        ],
        env=_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if (
                ready_file.exists()
                and _count_progress_lines(progress_file) >= MIN_SAVES_BEFORE_KILL
            ):
                break
            if worker.poll() is not None:
                out, err = worker.communicate()
                pytest.fail(
                    f"worker exited early with code {worker.returncode}\n"
                    f"stdout={out.decode(errors='replace')}\n"
                    f"stderr={err.decode(errors='replace')}"
                )
            time.sleep(0.02)
        else:
            pytest.fail(
                f"worker did not reach {MIN_SAVES_BEFORE_KILL} saves within "
                f"{READY_TIMEOUT_SEC}s (got {_count_progress_lines(progress_file)})"
            )

        # Kill while save RPCs are in flight and queued server-side.
        worker.send_signal(signal.SIGKILL)
        worker.wait(timeout=5)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)

    # Session cleanup must fire (drops the client's CUDA IPC tensors) —
    # otherwise this test is not exercising the race at all.
    cleanup_marker = f"Teardown (stream closed): instance {instance_id} removed"
    cleanup_seen = False

    deadline = time.monotonic() + SURVIVAL_WATCH_SEC
    cleanup_deadline = time.monotonic() + CLEANUP_LOG_DEADLINE_SEC
    while time.monotonic() < deadline:
        if not backlog_server.is_running():
            rc = backlog_server.process.returncode if backlog_server.process else None
            tail = backlog_server.read_logs()[-4000:]
            pytest.fail(
                f"pegaflow-server died after client SIGKILL (rc={rc}, "
                f"SIGSEGV={rc == -signal.SIGSEGV}) — queued save tasks used the "
                f"client's unmapped CUDA IPC pointers\n--- server log tail ---\n{tail}"
            )
        if not cleanup_seen and cleanup_marker in backlog_server.read_logs():
            cleanup_seen = True
        if not cleanup_seen and time.monotonic() > cleanup_deadline:
            pytest.fail(
                f"session cleanup did not fire within {CLEANUP_LOG_DEADLINE_SEC}s — "
                "test precondition broken (no unmap happened)"
            )
        time.sleep(0.1)

    assert cleanup_seen, "session cleanup never observed in server logs"

    # Drain-before-unmap ordering: both GPU worker threads must have exited
    # (queue fully drained) before the teardown dropped the IPC tensors.
    logs = backlog_server.read_logs()
    save_exit = logs.find("Save worker shutting down")
    load_exit = logs.find("Load worker shutting down")
    unmap = logs.find(cleanup_marker)
    assert save_exit != -1 and load_exit != -1, (
        "GPU worker threads did not exit during teardown — queue was not drained"
    )
    assert max(save_exit, load_exit) < unmap, (
        "CUDA IPC tensors were dropped before the GPU workers exited — unmap raced the queued tasks"
    )

    # The server must still answer RPCs.
    pegaflow_module = importlib.import_module("pegaflow.pegaflow")
    ok, message = pegaflow_module.EngineRpcClient(backlog_server.endpoint).health()
    assert ok, f"server unhealthy after client SIGKILL: {message}"
