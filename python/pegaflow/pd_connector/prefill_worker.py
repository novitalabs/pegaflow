"""P-side (prefill) worker logic — pushes KV via RDMA."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.layout import (
    LayerBlockSlices,
    block_ranges_for_remote_write,
    block_slices_bytes,
    layout_from_tensor,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (
    PdHandshake,
    PushReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.rdma import RdmaPort

if TYPE_CHECKING:
    from pegaflow.pd_connector.worker import PdWorkerConnector

logger = get_connector_logger()
PD_RDMA_PUSH_SENDER_THREADS = 4


class PrefillHandler:
    """Handles P-side (prefill) requests: KV push via RDMA."""

    def __init__(self, worker: PdWorkerConnector) -> None:
        self._w = worker
        self._push_reqs: dict[str, PushReqMeta] = {}
        self._pending_push_chunks: set[str] = set()
        self._push_chunk_maps: dict[str, tuple[dict[int, int], bool]] = {}
        self._tracker = ChunkTracker()
        self._completed_pushes: set[str] = set()
        self._producer_finished_req_ids: set[str] = set()
        self._remote_block_offsets: dict[str, int] = {}
        self._push_traces: dict[str, _PushTrace] = {}
        self._slot_mapping_cache: tuple[int, set[int]] | None = None
        self._skipped_pushes = 0
        self._push_sender = _AsyncLayerPushSender()
        self._push_finalizer = _AsyncPushFinalizer(self._push_sender)

    def process_push_reqs(self, reqs_to_push: dict[str, PushReqMeta]) -> None:
        for req_id, req in reqs_to_push.items():
            self._tracker.add_request(req_id)
            try:
                handshake = self._select_push_handshake(req)
            except _SkipPushRank:
                self._skipped_pushes += 1
                self._completed_pushes.add(req_id)
                logger.info(
                    "[PdConnector] P skipped MLA push req=%s target_req=%s rank=%d skipped_total=%d",
                    req_id,
                    req.target_request_id,
                    self._w.tp_rank,
                    self._skipped_pushes,
                )
                continue
            self._push_reqs[req_id] = req
            self._push_traces.setdefault(req_id, _PushTrace(queued_ts_ns=time.time_ns()))
            self._pending_push_chunks.add(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self._w.rdma.open_request(req_id, handshake)
            logger.info(
                "[PdConnector] P queued push req=%s target_req=%s rank=%d blocks=%d",
                req_id,
                req.target_request_id,
                self._w.tp_rank,
                len(flatten_block_ids(req.local_block_ids)),
            )

    def release(self, req_id: str) -> None:
        self._push_reqs.pop(req_id, None)
        self._pending_push_chunks.discard(req_id)
        self._push_chunk_maps.pop(req_id, None)
        self._push_traces.pop(req_id, None)
        self._push_sender.discard_req_stats(req_id)
        self._tracker.remove(req_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        layout = self._w.layouts.get(layer_name)
        assert layout is not None, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self._w.layouts)}"
        )
        # Re-assert the runtime tensor. CUDA graph or backend changes must not
        # silently swap in a different layout.
        runtime_layout = layout_from_tensor(
            layer_name,
            kv_layer,
            layer_spec=self._w._layer_spec(layer_name),
            logical_block_size=self._w.logical_block_size,
            expected_num_blocks=layout.num_blocks,
        )
        assert type(runtime_layout) is type(layout), (
            f"PdConnector KV layout type changed for {layer_name}: "
            f"registered={type(layout).__name__} runtime={type(runtime_layout).__name__}"
        )
        assert runtime_layout.shape == layout.shape, (
            f"PdConnector KV shape changed for {layer_name}: "
            f"registered={layout.shape} runtime={runtime_layout.shape}"
        )
        if not self._push_reqs:
            return
        slot_mapping = attn_metadata.slot_mapping
        touched_blocks = self._cached_unique_blocks_from_slot_mapping(
            slot_mapping,
            layout.block_size,
        )
        if not touched_blocks:
            return
        import torch

        if torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            event = None
        self._push_touched_blocks(layer_name, touched_blocks, event)

    def wait_for_save(self) -> None:
        logger.debug(
            "[PdConnector] worker wait_for_save push_reqs=%s pending_chunks=%s chunk_maps=%s tracker=%s",
            sorted(self._push_reqs),
            sorted(self._pending_push_chunks),
            sorted(self._push_chunk_maps),
            sorted(self._tracker._requests),
        )

    def get_finished_sending(self, finished_req_ids: set[str]) -> set[str]:
        """Return req_ids that are done sending and also finished by the producer."""
        self._producer_finished_req_ids.update(finished_req_ids)
        finished_sending = self._w.rdma.pop_finished_sending()
        self._completed_pushes.update(finished_sending)
        releasable_sending = self._completed_pushes & self._producer_finished_req_ids
        for req_id in releasable_sending:
            self._completed_pushes.discard(req_id)
            self._producer_finished_req_ids.discard(req_id)
            self._push_reqs.pop(req_id, None)
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self._push_traces.pop(req_id, None)
            self._tracker.remove(req_id)
            self._remote_block_offsets = {
                key: value for key, value in self._remote_block_offsets.items() if key != req_id
            }
        return releasable_sending

    def shutdown(self) -> None:
        self._push_reqs.clear()
        self._pending_push_chunks.clear()
        self._push_chunk_maps.clear()
        self._slot_mapping_cache = None
        self._completed_pushes.clear()
        self._producer_finished_req_ids.clear()
        self._remote_block_offsets.clear()
        self._push_traces.clear()
        self._push_finalizer.close()
        self._push_sender.close()

    @property
    def push_reqs(self) -> dict[str, PushReqMeta]:
        return self._push_reqs

    def _cached_unique_blocks_from_slot_mapping(
        self,
        slot_mapping: Any,
        block_size: int,
    ) -> set[int]:
        step_id = self._w._forward_step_id
        if self._slot_mapping_cache is not None and self._slot_mapping_cache[0] == step_id:
            return self._slot_mapping_cache[1]
        start = time.perf_counter()
        blocks = unique_blocks_from_slot_mapping(slot_mapping, block_size)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._slot_mapping_cache = (step_id, blocks)
        logger.info(
            "[PdConnector] P extracted slot_mapping blocks blocks=%d latency_ms=%.3f",
            len(blocks),
            elapsed_ms,
        )
        return blocks

    def _push_touched_blocks(self, layer_name: str, touched_blocks: set[int], event: Any) -> None:
        layer_idx = self._w._layer_idx(layer_name)
        layout = self._w.layouts[layer_name]
        for req_id, req in list(self._push_reqs.items()):
            req = self._push_reqs.get(req_id)
            if req is None:
                continue
            if self._tracker.is_done(req_id):
                continue
            req_blocks = flatten_block_ids(req.local_block_ids)
            if not req_blocks:
                continue
            selected_blocks = req_blocks & touched_blocks
            if not selected_blocks:
                continue
            trace = self._push_traces.setdefault(
                req_id,
                _PushTrace(queued_ts_ns=time.time_ns()),
            )
            if trace.first_save_ts_ns is None:
                trace.first_save_ts_ns = time.time_ns()
            remote_block_ids, all_chunks_seen = self._push_chunk_maps.get(req_id, ({}, False))
            if not remote_block_ids:
                remote_block_ids, all_chunks_seen = self._remote_block_id_map(
                    req_id,
                    req,
                    selected_blocks,
                )
                self._push_chunk_maps[req_id] = (remote_block_ids, all_chunks_seen)
            assert selected_blocks.issubset(remote_block_ids), (
                "PdConnector selected blocks must match the current registered push chunk; "
                f"req={req_id} selected={sorted(selected_blocks)} "
                f"mapped={sorted(remote_block_ids)}"
            )
            block_slices = block_ranges_for_remote_write(
                layout,
                selected_blocks,
                remote_block_ids,
            )
            rdma_bytes = block_slices_bytes(block_slices)
            assert self._w.rdma is not None
            chunk_idx = trace.chunk_count + 1
            self._push_sender.submit(
                _LayerPushTask(
                    rdma=self._w.rdma,
                    req_id=req_id,
                    target_request_id=req.target_request_id,
                    chunk_idx=chunk_idx,
                    layer_idx=layer_idx,
                    block_count=len(selected_blocks),
                    rdma_bytes=rdma_bytes,
                    block_slices=block_slices,
                    enqueued_ts_ns=time.time_ns(),
                    event=event,
                )
            )
            trace.rdma_bytes += rdma_bytes
            self._tracker.mark_blocks_pushed(req_id, layer_idx, selected_blocks)
            if layer_idx != len(self._w.layer_names) - 1:
                continue
            trace.chunk_count += 1
            trace.last_save_ts_ns = time.time_ns()
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            chunk_complete = self._tracker.has_pushed_all_blocks(
                req_id,
                selected_blocks,
                num_layers=len(self._w.layer_names),
            )
            if not (chunk_complete and all_chunks_seen):
                logger.info(
                    "[PdConnector] P chunk req=%s target_req=%s chunk=%d blocks=%d forward_ms=%.3f",
                    req_id,
                    req.target_request_id,
                    trace.chunk_count,
                    len(selected_blocks),
                    _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                )
                continue
            self._tracker.mark_done(req_id)
            finalize_ts_ns = time.time_ns()
            logger.info(
                "[PdConnector] P all chunks done req=%s target_req=%s chunks=%d blocks=%d rdma_bytes=%d schedule_to_save_ms=%.3f forward_ms=%.3f gbps=%.2f",
                req_id,
                req.target_request_id,
                trace.chunk_count,
                len(selected_blocks),
                trace.rdma_bytes,
                (finalize_ts_ns - trace.queued_ts_ns) / 1_000_000,
                _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                _gbps(trace.rdma_bytes, trace.first_save_ts_ns, trace.last_save_ts_ns),
            )
            self._push_finalizer.submit(
                _PushFinalizeTask(
                    rdma=self._w.rdma,
                    req_id=req_id,
                    target_request_id=req.target_request_id,
                    num_blocks=len(selected_blocks),
                    chunk_count=trace.chunk_count,
                    first_save_ts_ns=trace.first_save_ts_ns,
                    finalize_queued_ts_ns=finalize_ts_ns,
                    rdma_bytes=trace.rdma_bytes,
                )
            )

    def _select_push_handshake(self, req: PushReqMeta) -> PdHandshake:
        assert req.handshakes, (
            f"PdConnector push request has no handshakes; target_req={req.target_request_id}"
        )
        _assert_handshake_tp_consistency(req.handshakes)
        decode_tp_size = req.handshakes[0].tp_size
        if self._w.use_mla:
            assert self._w.tp_size >= decode_tp_size, (
                "PdConnector MLA heterogeneous TP requires prefill TP >= decode TP; "
                f"prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
            )
            assert self._w.tp_size % decode_tp_size == 0, (
                "PdConnector MLA heterogeneous TP requires prefill TP to be a "
                f"multiple of decode TP; prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
            )
            ratio = self._w.tp_size // decode_tp_size
            if self._w.tp_rank % ratio != 0:
                raise _SkipPushRank
            target_rank = self._w.tp_rank // ratio
            for handshake in req.handshakes:
                if handshake.tp_rank == target_rank:
                    return handshake
            raise AssertionError(
                f"PdConnector missing MLA target handshake for prefill_tp_rank={self._w.tp_rank} "
                f"decode_tp_rank={target_rank}; "
                f"available={[handshake.tp_rank for handshake in req.handshakes]}"
            )

        assert self._w.tp_size == decode_tp_size, (
            "PdConnector non-MLA requires equal P/D TP sizes; "
            f"prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
        )
        for handshake in req.handshakes:
            if handshake.tp_rank == self._w.tp_rank:
                return handshake
        raise AssertionError(
            f"PdConnector missing handshake for tp_rank={self._w.tp_rank}; "
            f"available={[handshake.tp_rank for handshake in req.handshakes]}"
        )

    def _remote_block_id_map(
        self,
        req_id: str,
        req: PushReqMeta,
        local_block_ids: set[int],
    ) -> tuple[dict[int, int], bool]:
        handshake = self._select_push_handshake(req)
        if not handshake.layers:
            return {block_id: block_id for block_id in local_block_ids}, True
        remote_block_ids = handshake.layers[0].block_ids
        for layer in handshake.layers[1:]:
            assert layer.block_ids == remote_block_ids, (
                "PdConnector expects one decode block-id layout shared by all layers; "
                f"layer=0 blocks={list(remote_block_ids)} layer={layer.layer_idx} "
                f"blocks={list(layer.block_ids)}"
            )
        ordered_local = sorted(local_block_ids)
        if len(ordered_local) == len(remote_block_ids):
            self._remote_block_offsets[req_id] = len(remote_block_ids)
            return dict(zip(ordered_local, remote_block_ids, strict=True)), True

        offset = self._remote_block_offsets.get(req_id, 0)
        next_offset = offset + len(ordered_local)
        assert next_offset <= len(remote_block_ids), (
            "PdConnector P/D block count mismatch "
            f"offset={offset} local_blocks={ordered_local} remote_blocks={list(remote_block_ids)}"
        )
        remote_chunk = remote_block_ids[offset:next_offset]
        self._remote_block_offsets[req_id] = next_offset
        return dict(zip(ordered_local, remote_chunk, strict=True)), next_offset == len(
            remote_block_ids
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _elapsed_ms(start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if start_ts_ns is None or end_ts_ns is None:
        return 0.0
    return (end_ts_ns - start_ts_ns) / 1_000_000


def _gbps(bytes_total: int, start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if bytes_total <= 0 or start_ts_ns is None or end_ts_ns is None or end_ts_ns <= start_ts_ns:
        return 0.0
    return bytes_total * 8 / ((end_ts_ns - start_ts_ns) / 1_000_000_000) / 1e9


def _assert_handshake_tp_consistency(handshakes: tuple[PdHandshake, ...]) -> None:
    tp_size = handshakes[0].tp_size
    assert all(handshake.tp_size == tp_size for handshake in handshakes), (
        f"PdConnector handshakes disagree on decode TP size: "
        f"{[handshake.tp_size for handshake in handshakes]}"
    )


class _SkipPushRank(Exception):
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LayerPushTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    chunk_idx: int
    layer_idx: int
    block_count: int
    rdma_bytes: int
    block_slices: list[LayerBlockSlices]
    enqueued_ts_ns: int
    event: Any = None


@dataclass
class _PushTrace:
    queued_ts_ns: int
    first_save_ts_ns: int | None = None
    last_save_ts_ns: int | None = None
    rdma_bytes: int = 0
    chunk_count: int = 0


@dataclass(frozen=True)
class _PushFinalizeTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    num_blocks: int
    chunk_count: int
    first_save_ts_ns: int | None
    finalize_queued_ts_ns: int
    rdma_bytes: int


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------


class _AsyncLayerPushSender:
    def __init__(self) -> None:
        self._queue: queue.Queue[_LayerPushTask | None] = queue.Queue()
        self._condition = threading.Condition()
        self._worker_count = PD_RDMA_PUSH_SENDER_THREADS
        self._inflight = 0
        self._inflight_by_req: dict[str, int] = {}
        self._stats_by_req: dict[str, _ReqPushStats] = {}
        self._discarded_stats_reqs: set[str] = set()
        self._error: BaseException | None = None
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"pd-rdma-push-sender-{idx}",
                daemon=True,
            )
            for idx in range(self._worker_count)
        ]
        for thread in self._threads:
            thread.start()
        logger.info(
            "[PdConnector] P layer push sender started workers=%d",
            self._worker_count,
        )

    def submit(self, task: _LayerPushTask) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            self._inflight += 1
            self._inflight_by_req[task.req_id] = self._inflight_by_req.get(task.req_id, 0) + 1
            inflight = self._inflight
            inflight_req = self._inflight_by_req[task.req_id]
        logger.debug(
            "[PdConnector] P layer push queued req=%s target_req=%s chunk=%d layer=%d blocks=%d bytes=%d inflight=%d inflight_req=%d queue_depth=%d",
            task.req_id,
            task.target_request_id,
            task.chunk_idx,
            task.layer_idx,
            task.block_count,
            task.rdma_bytes,
            inflight,
            inflight_req,
            self._queue.qsize(),
        )
        self._queue.put(task)

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def wait_req(self, req_id: str) -> None:
        with self._condition:
            while self._inflight_by_req.get(req_id, 0) > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        for _ in self._threads:
            self._queue.put(None)

    def pop_req_stats(self, req_id: str) -> _ReqPushStats:
        with self._condition:
            self._discarded_stats_reqs.discard(req_id)
            return self._stats_by_req.pop(req_id, _ReqPushStats())

    def discard_req_stats(self, req_id: str) -> None:
        with self._condition:
            self._stats_by_req.pop(req_id, None)
            if self._inflight_by_req.get(req_id, 0) > 0:
                self._discarded_stats_reqs.add(req_id)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                result = _run_layer_push(task)
                with self._condition:
                    if task.req_id not in self._discarded_stats_reqs:
                        self._stats_by_req.setdefault(task.req_id, _ReqPushStats()).add(result)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            finally:
                if task is not None:
                    with self._condition:
                        self._inflight -= 1
                        remaining = self._inflight_by_req.get(task.req_id, 0) - 1
                        if remaining > 0:
                            self._inflight_by_req[task.req_id] = remaining
                        else:
                            self._inflight_by_req.pop(task.req_id, None)
                            if task.req_id in self._discarded_stats_reqs:
                                self._discarded_stats_reqs.discard(task.req_id)
                                self._stats_by_req.pop(task.req_id, None)
                        self._condition.notify_all()
                self._queue.task_done()


@dataclass(frozen=True)
class _LayerPushResult:
    bytes: int
    queue_wait_ms: float
    event_ms: float
    native_ms: float


@dataclass
class _ReqPushStats:
    tasks: int = 0
    bytes: int = 0
    queue_wait_ms: float = 0.0
    event_ms: float = 0.0
    native_ms: float = 0.0
    max_queue_wait_ms: float = 0.0
    max_event_ms: float = 0.0
    max_native_ms: float = 0.0
    queue_wait_samples_ms: list[float] = field(default_factory=list)
    event_samples_ms: list[float] = field(default_factory=list)
    native_samples_ms: list[float] = field(default_factory=list)

    def add(self, result: _LayerPushResult) -> None:
        self.tasks += 1
        self.bytes += result.bytes
        self.queue_wait_ms += result.queue_wait_ms
        self.event_ms += result.event_ms
        self.native_ms += result.native_ms
        self.max_queue_wait_ms = max(self.max_queue_wait_ms, result.queue_wait_ms)
        self.max_event_ms = max(self.max_event_ms, result.event_ms)
        self.max_native_ms = max(self.max_native_ms, result.native_ms)
        self.queue_wait_samples_ms.append(result.queue_wait_ms)
        self.event_samples_ms.append(result.event_ms)
        self.native_samples_ms.append(result.native_ms)

    def avg_queue_wait_ms(self) -> float:
        return _avg_ms(self.queue_wait_samples_ms)

    def p50_queue_wait_ms(self) -> float:
        return _percentile_ms(self.queue_wait_samples_ms, 0.50)

    def p95_queue_wait_ms(self) -> float:
        return _percentile_ms(self.queue_wait_samples_ms, 0.95)

    def avg_event_ms(self) -> float:
        return _avg_ms(self.event_samples_ms)

    def p50_event_ms(self) -> float:
        return _percentile_ms(self.event_samples_ms, 0.50)

    def p95_event_ms(self) -> float:
        return _percentile_ms(self.event_samples_ms, 0.95)

    def avg_native_ms(self) -> float:
        return _avg_ms(self.native_samples_ms)

    def p50_native_ms(self) -> float:
        return _percentile_ms(self.native_samples_ms, 0.50)

    def p95_native_ms(self) -> float:
        return _percentile_ms(self.native_samples_ms, 0.95)


def _avg_ms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def _percentile_ms(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = round((len(ordered) - 1) * percentile)
    return ordered[idx]


def _run_layer_push(task: _LayerPushTask) -> _LayerPushResult:
    start_ts_ns = time.time_ns()
    event_done_ts_ns = start_ts_ns
    native_done_ts_ns = start_ts_ns
    failed = False
    if task.event is not None:
        task.event.synchronize()
    event_done_ts_ns = time.time_ns()
    try:
        task.rdma.push_layer(task.req_id, task.layer_idx, task.block_slices)
        native_done_ts_ns = time.time_ns()
    except BaseException:
        failed = True
        native_done_ts_ns = time.time_ns()
        raise
    finally:
        queue_wait_ms = (start_ts_ns - task.enqueued_ts_ns) / 1_000_000
        event_ms = (event_done_ts_ns - start_ts_ns) / 1_000_000
        native_ms = (native_done_ts_ns - event_done_ts_ns) / 1_000_000
        logger.debug(
            "[PdConnector] P layer push %s req=%s target_req=%s chunk=%d layer=%d blocks=%d bytes=%d queue_wait_ms=%.3f event_ms=%.3f native_ms=%.3f total_ms=%.3f",
            "failed" if failed else "done",
            task.req_id,
            task.target_request_id,
            task.chunk_idx,
            task.layer_idx,
            task.block_count,
            task.rdma_bytes,
            queue_wait_ms,
            event_ms,
            native_ms,
            (native_done_ts_ns - task.enqueued_ts_ns) / 1_000_000,
        )
    return _LayerPushResult(
        bytes=task.rdma_bytes,
        queue_wait_ms=queue_wait_ms,
        event_ms=event_ms,
        native_ms=native_ms,
    )


class _AsyncPushFinalizer:
    def __init__(self, push_sender: _AsyncLayerPushSender) -> None:
        self._push_sender = push_sender
        self._queue: queue.Queue[_PushFinalizeTask | None] = queue.Queue()
        self._condition = threading.Condition()
        self._submitted: set[str] = set()
        self._inflight = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="pd-rdma-push-finalizer", daemon=True
        )
        self._thread.start()

    def submit(self, task: _PushFinalizeTask) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            if task.req_id in self._submitted:
                return
            self._submitted.add(task.req_id)
            self._inflight += 1
            inflight = self._inflight
        logger.info(
            "[PdConnector] P finalize queued req=%s target_req=%s chunks=%d blocks=%d bytes=%d inflight=%d queue_depth=%d",
            task.req_id,
            task.target_request_id,
            task.chunk_count,
            task.num_blocks,
            task.rdma_bytes,
            inflight,
            self._queue.qsize(),
        )
        self._queue.put(task)

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                start_ts_ns = time.time_ns()
                self._push_sender.wait_req(task.req_id)
                sender_done_ts_ns = time.time_ns()
                push_stats = self._push_sender.pop_req_stats(task.req_id)
                task.rdma.wait_for_pushes(task.req_id)
                writes_done_ts_ns = time.time_ns()
                task.rdma.push_done(task.req_id)
                done_ts_ns = time.time_ns()
                logger.info(
                    "[PdConnector] P RDMA done req=%s target_req=%s chunks=%d blocks=%d rdma_bytes=%d save_to_imm_ms=%.3f schedule_to_imm_ms=%.3f wait_sender_ms=%.3f wait_writes_ms=%.3f imm_ms=%.3f save_gbps=%.2f tail_gbps=%.2f sender_workers=%d push_tasks=%d push_bytes=%d push_queue_sum_ms=%.3f push_queue_avg_ms=%.3f push_queue_p50_ms=%.3f push_queue_p95_ms=%.3f push_queue_max_ms=%.3f push_event_sum_ms=%.3f push_event_avg_ms=%.3f push_event_p50_ms=%.3f push_event_p95_ms=%.3f push_event_max_ms=%.3f push_native_sum_ms=%.3f push_native_avg_ms=%.3f push_native_p50_ms=%.3f push_native_p95_ms=%.3f push_native_max_ms=%.3f",
                    task.req_id,
                    task.target_request_id,
                    task.chunk_count,
                    task.num_blocks,
                    task.rdma_bytes,
                    _elapsed_ms(task.first_save_ts_ns, done_ts_ns),
                    _elapsed_ms(task.finalize_queued_ts_ns, done_ts_ns),
                    _elapsed_ms(start_ts_ns, sender_done_ts_ns),
                    _elapsed_ms(sender_done_ts_ns, writes_done_ts_ns),
                    _elapsed_ms(writes_done_ts_ns, done_ts_ns),
                    _gbps(task.rdma_bytes, task.first_save_ts_ns, done_ts_ns),
                    _gbps(task.rdma_bytes, task.finalize_queued_ts_ns, done_ts_ns),
                    self._push_sender._worker_count,
                    push_stats.tasks,
                    push_stats.bytes,
                    push_stats.queue_wait_ms,
                    push_stats.avg_queue_wait_ms(),
                    push_stats.p50_queue_wait_ms(),
                    push_stats.p95_queue_wait_ms(),
                    push_stats.max_queue_wait_ms,
                    push_stats.event_ms,
                    push_stats.avg_event_ms(),
                    push_stats.p50_event_ms(),
                    push_stats.p95_event_ms(),
                    push_stats.max_event_ms,
                    push_stats.native_ms,
                    push_stats.avg_native_ms(),
                    push_stats.p50_native_ms(),
                    push_stats.p95_native_ms(),
                    push_stats.max_native_ms,
                )
            except BaseException as exc:
                if task is not None:
                    self._push_sender.discard_req_stats(task.req_id)
                    logger.exception(
                        "[PdConnector] P finalize failed req=%s target_req=%s chunks=%d blocks=%d bytes=%d",
                        task.req_id,
                        task.target_request_id,
                        task.chunk_count,
                        task.num_blocks,
                        task.rdma_bytes,
                    )
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            finally:
                if task is not None:
                    with self._condition:
                        self._submitted.discard(task.req_id)
                        self._inflight -= 1
                        self._condition.notify_all()
                self._queue.task_done()
