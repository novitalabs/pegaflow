"""Victim worker for test_ipc_mapping_lifetime.

Simulates a vLLM worker SIGKILLed while save RPCs are queued server-side.
Registers a 2-layer KV cache via CUDA IPC, opens the liveness session
stream, then runs several threads that loop synchronous save RPCs with
fresh content hashes — the server's hash filter never dedups them, so
every RPC queues real GPU->CPU copy work on the per-device save worker.

Appends one line to the progress file per completed save so the parent
test can SIGKILL this process at a moment when the server's save queue
has a backlog.

Run as:
python save_backlog_kill_helper.py <endpoint> <instance_id> <namespace> <ready_file> <progress_file>
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
import threading
import uuid

import torch

from pegaflow.ipc_wrapper import CudaIPCWrapper

NUM_LAYERS = 2
NUM_BLOCKS = 4096
SEG_BYTES = 16 << 10  # bytes per K (or V) segment per block
SAVE_THREADS = 12
# Save every other block: the gaps defeat the server's copy-range merging, so
# each task enqueues thousands of small cuMemcpy calls and stays on the save
# worker for tens of milliseconds. A deep, slow-draining backlog is the point:
# the queue must outlive the session-cleanup latency after SIGKILL.
SAVE_BLOCK_IDS = list(range(1, NUM_BLOCKS, 2))


def main() -> int:
    endpoint, instance_id, namespace, ready_file, progress_file = sys.argv[1:6]

    pegaflow_module = importlib.import_module("pegaflow.pegaflow")
    client = pegaflow_module.EngineRpcClient(endpoint)

    device = torch.device("cuda:0")
    kv_caches = [
        torch.zeros((2, NUM_BLOCKS, SEG_BYTES), dtype=torch.uint8, device=device)
        for _ in range(NUM_LAYERS)
    ]
    layer_names = [f"layer_{i}" for i in range(NUM_LAYERS)]

    ok, msg = client.register_context_batch(
        instance_id,
        namespace,
        0,  # tp_rank
        0,  # pp_rank
        1,  # tp_size
        1,  # world_size
        0,  # device_id
        NUM_LAYERS,
        layer_names,
        [pickle.dumps(CudaIPCWrapper(kv)) for kv in kv_caches],
        [NUM_BLOCKS] * NUM_LAYERS,
        [SEG_BYTES] * NUM_LAYERS,
        [NUM_BLOCKS * SEG_BYTES] * NUM_LAYERS,  # kv_stride_bytes
        [2] * NUM_LAYERS,  # segments
    )
    if not ok:
        print(f"register_context_batch failed: {msg}", file=sys.stderr)
        return 1

    client.start_session_watcher(instance_id, namespace, 1, 1)

    run_id = uuid.uuid4().hex
    # Never closed deliberately: this process only exits via SIGKILL.
    progress = open(progress_file, "w", buffering=1)  # noqa: SIM115
    progress_lock = threading.Lock()

    def save_loop(tid: int) -> None:
        thread_client = pegaflow_module.EngineRpcClient(endpoint)
        i = 0
        while True:
            hashes = [
                hashlib.sha256(f"{run_id}:{tid}:{i}:{b}".encode()).digest() for b in SAVE_BLOCK_IDS
            ]
            saves = [(name, SAVE_BLOCK_IDS, hashes) for name in layer_names]
            ok, msg = thread_client.save(instance_id, 0, 0, 0, saves)
            with progress_lock:
                progress.write(f"{'ok' if ok else 'failed'} {tid} {i}\n")
                progress.flush()
            i += 1

    threads = [
        threading.Thread(target=save_loop, args=(tid,), daemon=True, name=f"save-{tid}")
        for tid in range(SAVE_THREADS)
    ]
    for t in threads:
        t.start()

    with open(ready_file, "w") as f:
        f.write("ready")

    # Block until SIGKILLed; saves keep flowing on the worker threads.
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
