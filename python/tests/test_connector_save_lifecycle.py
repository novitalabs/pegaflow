from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    ConnectorContext,
    PegaConnectorMetadata,
    SaveIntent,
)
from pegaflow.connector.worker import WorkerConnector  # noqa: E402


def make_worker() -> WorkerConnector:
    context = ConnectorContext(
        instance_id="test",
        namespace="test",
        block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
    )
    with patch("pegaflow.connector.worker.threading.Thread.start"):
        return WorkerConnector(
            context,
            vllm_config=SimpleNamespace(additional_config={}),
        )


def enqueue_save(
    worker: WorkerConnector,
    block_id: int = 1,
    block_hash: bytes = b"hash",
) -> threading.Event:
    worker._current_metadata = PegaConnectorMetadata(
        save_intents={
            "request": SaveIntent(
                block_ids=(block_id,),
                block_hashes=(block_hash,),
            )
        }
    )
    worker.wait_for_save()
    return worker._save_completion_events["request"]


def complete_next_save(worker: WorkerConnector) -> None:
    task = worker._save_queue.get_nowait()
    worker._complete_save_requests(task.request_ids)


def process_next_save(worker: WorkerConnector) -> None:
    task = worker._save_queue.get_nowait()
    with patch("torch.cuda.synchronize"):
        worker._process_save_batch([task])


def test_blocks_are_not_reused_until_every_save_task_completes():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    gpu_blocks = {1: b"first request block", 2: b"second request block"}
    stored_blocks: dict[bytes, bytes] = {}

    def save(_instance, _tp_rank, _pp_rank, _device_id, saves):
        for _layer, block_ids, block_hashes in saves:
            for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
                stored_blocks[block_hash] = gpu_blocks[block_id]
        return True, ""

    worker._ctx.engine_client.save.side_effect = save
    enqueue_save(worker, 1, b"first hash")
    enqueue_save(worker, 2, b"second hash")
    process_next_save(worker)

    finished_sending, _ = worker.get_finished({"request"})
    if finished_sending:
        gpu_blocks[2] = b"reused by another request"

    process_next_save(worker)

    assert stored_blocks[b"second hash"] == b"second request block"


def test_later_save_reopens_completed_request():
    worker = make_worker()
    first_completion = enqueue_save(worker)
    complete_next_save(worker)

    assert first_completion.is_set()

    second_completion = enqueue_save(worker)

    assert not second_completion.is_set()
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending is None

    complete_next_save(worker)

    assert second_completion.is_set()
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending == {"request"}
