from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    ConnectorContext,
    LocalRecurrentSave,
    LocalSaveRef,
    PegaConnectorMetadata,
    SaveIntent,
)
from pegaflow.connector.linear_state_cache import LinearStateSlot  # noqa: E402
from pegaflow.connector.worker import WorkerConnector  # noqa: E402


def make_worker(num_linear_state_cache_slots: int = 0) -> WorkerConnector:
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
        num_linear_state_cache_slots=num_linear_state_cache_slots,
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


def test_committed_save_runs_on_a_no_forward_step():
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True)
    worker._registered_layers = ["layer"]
    worker._ctx.engine_client.save.return_value = (True, "")
    metadata = PegaConnectorMetadata(
        ready_save_intents={
            "request": SaveIntent(
                block_ids_by_group=((1,),),
                block_hashes=(b"hash",),
            )
        }
    )

    with patch("torch.cuda.synchronize"):
        worker.start_load_kv(metadata, None)

    worker._ctx.engine_client.save.assert_called_once()
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending == {"request"}


def test_current_step_save_waits_for_forward_completion():
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True)
    worker._registered_layers = ["layer"]
    worker._ctx.engine_client.save.return_value = (True, "")
    metadata = PegaConnectorMetadata(
        save_intents={
            "request": SaveIntent(
                block_ids_by_group=((1,),),
                block_hashes=(b"hash",),
            )
        }
    )

    worker.start_load_kv(metadata, None)
    worker._ctx.engine_client.save.assert_not_called()

    with patch("torch.cuda.synchronize"):
        worker.wait_for_save()
    worker._ctx.engine_client.save.assert_called_once()


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


def _local_save_intent(req_id: str = "request"):
    ref = LocalSaveRef(req_id, 0, 1, b"checkpoint")
    save = LocalRecurrentSave(
        target=LinearStateSlot(ref.slot, ref.generation, ref.block_hash),
        source_block_ids=((1, 2),),
    )
    return ref, SaveIntent(
        block_ids_by_group=((), ()),
        block_hashes=(ref.block_hash,),
        local_recurrent_saves=(save,),
    )


def test_running_local_save_ack_does_not_enter_finished_sending():
    worker = make_worker(num_linear_state_cache_slots=1)
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True)
    ref, intent = _local_save_intent()
    worker._current_metadata = PegaConnectorMetadata(save_intents={"request": intent})

    with (
        patch("torch.cuda.synchronize"),
        patch.object(worker, "_save_local_recurrent"),
    ):
        worker.wait_for_save()

    finished_sending, _ = worker.get_finished(set())
    metadata = worker.build_connector_worker_meta()
    assert finished_sending is None
    assert metadata is not None
    assert metadata.succeeded == {ref: 1}
    assert metadata.failed == {}


def test_local_d2d_failure_acks_failure_and_reaches_request_terminal_state():
    worker = make_worker(num_linear_state_cache_slots=1)
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True)
    ref, intent = _local_save_intent()
    worker._current_metadata = PegaConnectorMetadata(save_intents={"request": intent})

    with (
        patch("torch.cuda.synchronize"),
        patch.object(worker, "_save_local_recurrent", side_effect=RuntimeError("copy failed")),
    ):
        worker.wait_for_save()

    metadata = worker.build_connector_worker_meta()
    assert metadata is not None
    assert metadata.failed == {ref: 1}
    assert "request" not in worker._req_pending_save_tasks
    finished_sending, _ = worker.get_finished({"request"})
    assert finished_sending == {"request"}


def test_remote_and_local_load_must_both_complete_before_finished_recving():
    worker = make_worker(num_linear_state_cache_slots=1)
    event = MagicMock()
    event.query.side_effect = [False, True]
    worker._pending_local_load_events["request"] = event
    worker._failed_load_reqs = {"request"}

    _, finished_recving = worker.get_finished(set())
    assert finished_recving is None
    assert "request" in worker._remote_load_completed

    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {"request"}
    assert worker._pending_local_load_events == {}
    assert worker._remote_load_completed == set()
