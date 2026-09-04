from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
                block_ids_by_group=((block_id,),),
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


def test_boundary_state_saves_run_async_and_report_job_completion():
    """HMA boundary-state jobs ride the async save queue.

    Their blocks are pinned by the scheduler, so nothing needs to finish
    before the next step; completion (success or not) is reported back as
    PegaWorkerMetadata so the scheduler can release the pins.
    """
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True, group_count=2)
    worker._registered_layers = ["attention", "recurrent"]
    worker._layer_to_group = {"attention": 0, "recurrent": 1}
    saved: list[tuple[str, list[int], list[bytes]]] = []

    def save(_instance, _tp_rank, _pp_rank, _device_id, saves):
        saved.extend(saves)
        return False, "store unavailable"

    worker._ctx.engine_client.save.side_effect = save
    worker._current_metadata = PegaConnectorMetadata(
        boundary_save_intents={
            7: SaveIntent(
                block_ids_by_group=((0, 0), (21, 22)),
                block_hashes=(b"h0", b"h2"),
            )
        }
    )

    worker.wait_for_save()
    worker._ctx.engine_client.save.assert_not_called()
    assert worker.build_connector_worker_meta() is None

    process_next_save(worker)

    # Only the recurrent layer sees the handed-off blocks.
    assert saved == [("recurrent", [21, 22], [b"h0", b"h2"])]
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_boundary_jobs == {7: 1}
    assert worker.build_connector_worker_meta() is None
    # Boundary jobs are not requests: no finished_sending entry.
    assert worker.get_finished(set()) == (None, None)


def test_boundary_state_saves_require_hma():
    worker = make_worker()
    worker._current_metadata = PegaConnectorMetadata(
        boundary_save_intents={0: SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"hash",))}
    )

    with pytest.raises(RuntimeError, match="only valid for HMA"):
        worker.wait_for_save()


def test_hma_request_saves_run_async():
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True, group_count=1)
    worker._registered_layers = ["layer"]
    worker._ctx.engine_client.save.return_value = (True, "")
    worker._current_metadata = PegaConnectorMetadata(
        save_intents={
            "request": SaveIntent(
                block_ids_by_group=((1,),),
                block_hashes=(b"hash",),
            )
        }
    )

    worker.wait_for_save()
    worker._ctx.engine_client.save.assert_not_called()

    process_next_save(worker)
    worker._ctx.engine_client.save.assert_called_once()
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending == {"request"}


def test_preemption_waits_for_every_save_task():
    worker = make_worker()
    completion = enqueue_save(worker)
    enqueue_save(worker)
    wait_started = threading.Event()
    preemption_returned = threading.Event()
    original_wait = completion.wait

    def wait_for_completion(timeout: float | None = None) -> bool:
        wait_started.set()
        return original_wait(timeout)

    def handle_preemption() -> None:
        worker.handle_preemptions({"request"})
        preemption_returned.set()

    completed_tasks = 0
    with patch.object(completion, "wait", side_effect=wait_for_completion):
        preemption_thread = threading.Thread(target=handle_preemption)
        preemption_thread.start()
        try:
            assert wait_started.wait(timeout=1)

            complete_next_save(worker)
            completed_tasks += 1
            assert not preemption_returned.wait(timeout=0.1)

            complete_next_save(worker)
            completed_tasks += 1
            assert preemption_returned.wait(timeout=1)
        finally:
            for _ in range(2 - completed_tasks):
                complete_next_save(worker)
            preemption_thread.join(timeout=1)

    assert not preemption_thread.is_alive()


def test_malformed_save_intent_is_skipped_and_still_completes():
    """A scheduler bug must not take the save thread down with it: the
    request's blocks are released (unsaved) and other intents in the batch
    still reach the store."""
    worker = make_worker()
    worker._registered_layers = ["layer"]
    worker._ctx.engine_client.save.return_value = (True, "")
    worker._current_metadata = PegaConnectorMetadata(
        save_intents={
            "torn": SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"h0", b"h1")),
            "good": SaveIntent(block_ids_by_group=((2,),), block_hashes=(b"h2",)),
        }
    )
    worker.wait_for_save()

    process_next_save(worker)

    (_, _, _, _, saves), _ = worker._ctx.engine_client.save.call_args
    assert saves == [("layer", [2], [b"h2"])]
    finished_sending, _ = worker.get_finished({"torn", "good"})
    assert finished_sending == {"torn", "good"}


def test_save_worker_survives_a_failing_batch():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    completion = enqueue_save(worker)
    worker._save_queue.put(None)

    with patch("torch.cuda.synchronize", side_effect=RuntimeError("device lost")):
        worker._save_worker()

    assert completion.is_set()
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending == {"request"}
