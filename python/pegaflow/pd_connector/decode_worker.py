"""D-side (decode) worker logic — receives KV via RDMA."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.layout import KvCacheLayout
from pegaflow.pd_connector.layout_mapping import (
    decode_rank_source_counts,
)
from pegaflow.pd_connector.metadata import (
    BlockIds,
    LayerRemoteLayout,
    PdHandshake,
    WaitReqMeta,
    flatten_block_ids,
    layer_layout_to_compact_dict,
)
from pegaflow.pd_connector.prefill import AsyncPrefillSender, PrefillHttpTask
from pegaflow.pd_connector.rdma import RdmaPort

if TYPE_CHECKING:
    from pegaflow.pd_connector.worker import PdWorkerBase

from pegaflow.pd_connector.config import extra_config_value

logger = get_connector_logger()


class DecodeHandler:
    """Handles D-side (decode) requests: RDMA receive, handshake, prefill dispatch."""

    def __init__(
        self,
        worker: PdWorkerBase,
        prefill_sender: Any | None = None,
    ) -> None:
        self._w = worker
        self._lock = threading.Lock()
        self._wait_reqs: dict[str, WaitReqMeta] = {}
        self._peer_layouts: dict[int, dict[str, KvCacheLayout]] = {}
        self._peer_mr_descs: dict[int, dict[str, Any]] = {}
        self._peer_layer_templates: dict[int, tuple[dict[str, Any], ...]] = {}
        self._expected_imm_counts: dict[int, int] = {}
        self._peer_block_size: int | None = None
        self._failed_recving: set[str] = set()
        self._failed_recving_for_meta: set[str] = set()
        self._failed_block_ids: set[int] = set()
        self._aborted_waits: set[str] = set()
        self._finished_aborted_recving: set[str] = set()
        self._next_imm_id = 1
        self._rdma_waiter: _AsyncRdmaDoneWaiter | None = (
            _AsyncRdmaDoneWaiter(
                worker.rdma,
                failure_callback=self._mark_wait_failed,
                success_callback=self._record_rdma_wait_done,
            )
            if worker.rdma is not None
            else None
        )
        prefill_sender_worker_count = int(
            extra_config_value(
                worker.vllm_config,
                "pegaflow.pd.prefill_sender_worker_count",
                3,
            )
        )
        self._prefill_sender = prefill_sender or AsyncPrefillSender(
            worker_count=prefill_sender_worker_count,
            failure_callback=self._mark_prefill_failed,
        )

    def init_rdma_waiter(self) -> None:
        """Called after RDMA port is built (during register_kv_caches)."""
        if self._rdma_waiter is None and self._w.rdma is not None:
            self._rdma_waiter = _AsyncRdmaDoneWaiter(
                self._w.rdma,
                failure_callback=self._mark_wait_failed,
                success_callback=self._record_rdma_wait_done,
            )

    def gather_peer_info(self) -> None:
        mr_descs = {name: layer.mr_desc for name, layer in self._w._registered_layers.items()}
        if self._w.tp_size <= 1:
            self._peer_layouts = {0: self._w.layouts}
            self._peer_mr_descs = {0: mr_descs}
            self._refresh_peer_layer_templates()
            return
        try:
            gathered = _all_gather_peer_info(self._w.layouts, mr_descs, self._w.tp_size)
        except Exception:
            logger.warning(
                "[PdConnector] all_gather_object unavailable, rank 0 dispatch limited to local rank",
            )
            self._peer_layouts = {self._w.tp_rank: self._w.layouts}
            self._peer_mr_descs = {self._w.tp_rank: mr_descs}
            self._refresh_peer_layer_templates()
            return
        for rank, (layouts, descs) in enumerate(gathered):
            self._peer_layouts[rank] = layouts
            self._peer_mr_descs[rank] = descs
        self._refresh_peer_layer_templates()

    def process_wait_reqs(self, reqs_to_wait: dict[str, WaitReqMeta]) -> None:
        assert self._w.rdma is not None, "PdConnector RDMA port is not initialized"
        assert self._rdma_waiter is not None, "PdConnector RDMA waiter is not initialized"
        for req_id, req in reqs_to_wait.items():
            if req_id in self._wait_reqs:
                logger.info("[PdConnector] D wait req=%s already registered", req_id)
                continue
            process_ts_ns = time.time_ns()
            with self._lock:
                self._wait_reqs[req_id] = req
                self._w.metrics.set_decode_active_waits(len(self._wait_reqs))
            block_ids = flatten_block_ids(req.local_block_ids)
            imm_id = self._alloc_imm_id()
            wait_handshake = self._build_wait_handshake(
                req.done_request_id,
                req.local_block_ids,
                imm_id,
            )
            self._w.rdma.open_request(req_id, wait_handshake)
            local_block_count = len(block_ids)
            waiter_queued_ts_ns = time.time_ns()
            self._rdma_waiter.submit(
                _RdmaWaitTask(
                    req_id=req_id,
                    generation=0,
                    remote_request_id=req.remote_request_id,
                    done_request_id=req.done_request_id,
                    prefill_url=req.prefill_url,
                    rank=self._w.tp_rank,
                    block_count=local_block_count,
                    queued_ts_ns=waiter_queued_ts_ns,
                )
            )
            queued_ts_ns = time.time_ns()
            logger.info(
                "[PdConnector] D queued async wait req=%s remote_req=%s prefill_url=%s rank=%d blocks=%d "
                "proxy_to_worker_ms=%.3f matched_to_worker_ms=%.3f scheduler_wait_to_worker_ms=%.3f "
                "open_request_ms=%.3f ts_ns=%d",
                req_id,
                req.remote_request_id,
                req.prefill_url or "<oob>",
                self._w.tp_rank,
                local_block_count,
                _elapsed_ms(req.proxy_start_ts_ns, queued_ts_ns),
                _elapsed_ms(req.matched_ts_ns, queued_ts_ns),
                _elapsed_ms(req.scheduler_wait_ts_ns, queued_ts_ns),
                _elapsed_ms(process_ts_ns, queued_ts_ns),
                queued_ts_ns,
            )
            if req.prefill_url and self._w.tp_rank == 0:
                self._dispatch_prefill(req, req.local_block_ids, imm_id)

    def release(self, req_id: str) -> None:
        with self._lock:
            req = self._wait_reqs.get(req_id)
            if req is not None:
                self._aborted_waits.add(req_id)
                self._w.metrics.record_decode_abort()
        cancel_prefill = getattr(self._prefill_sender, "cancel", None)
        if req is not None and cancel_prefill is not None:
            cancel_prefill(req.remote_request_id)

    def finish_recving(self, finished_recving: set[str]) -> None:
        for req_id in finished_recving:
            with self._lock:
                req = self._wait_reqs.pop(req_id, None)
                was_aborted = req_id in self._aborted_waits
                self._aborted_waits.discard(req_id)
                self._w.metrics.set_decode_active_waits(len(self._wait_reqs))
            if req is not None and not was_aborted:
                self._w.metrics.record_decode_wait(
                    duration_s=_elapsed_seconds(req.scheduler_wait_ts_ns, time.time_ns()),
                    rdma_wait_s=None,
                    blocks=len(flatten_block_ids(req.local_block_ids)),
                    success=req_id not in self._failed_recving,
                )
            self._w.rdma.close_request(req_id)

    def pop_finished_aborted_recving(self) -> set[str]:
        with self._lock:
            finished = self._finished_aborted_recving
            self._finished_aborted_recving = set()
            return finished

    def pop_failed_recving(self) -> set[str]:
        with self._lock:
            failed = self._failed_recving
            self._failed_recving = set()
            return failed

    def pop_failed_recving_for_meta(self) -> set[str]:
        with self._lock:
            failed = self._failed_recving_for_meta
            self._failed_recving_for_meta = set()
            return failed

    def pop_failed_block_ids(self) -> set[int]:
        with self._lock:
            failed = self._failed_block_ids
            self._failed_block_ids = set()
            return failed

    def shutdown(self) -> None:
        with self._lock:
            self._wait_reqs.clear()
            self._failed_recving.clear()
            self._failed_recving_for_meta.clear()
            self._failed_block_ids.clear()
            self._aborted_waits.clear()
            self._finished_aborted_recving.clear()
        if self._rdma_waiter is not None:
            self._rdma_waiter.close()
        close = getattr(self._prefill_sender, "close", None)
        if close is not None:
            close()

    @property
    def wait_reqs(self) -> dict[str, WaitReqMeta]:
        return self._wait_reqs

    def is_idle(self) -> bool:
        with self._lock:
            return (
                not self._wait_reqs
                and not self._failed_recving
                and not self._failed_block_ids
                and not self._finished_aborted_recving
            )

    def _mark_prefill_failed(self, remote_request_id: str, exc: BaseException) -> None:
        with self._lock:
            for req_id, req in list(self._wait_reqs.items()):
                if req.remote_request_id != remote_request_id:
                    continue
                self._mark_failed_locked(req_id, req, exc)
                return
        logger.warning(
            "[PdConnector] D prefill failed for unknown/finished remote_req=%s: %s",
            remote_request_id,
            exc,
        )

    def _mark_wait_failed(self, req_id: str, exc: BaseException) -> None:
        with self._lock:
            req = self._wait_reqs.get(req_id)
            if req is None:
                logger.warning(
                    "[PdConnector] D wait failed for unknown/finished req=%s: %s",
                    req_id,
                    exc,
                )
                return
            if req_id in self._aborted_waits:
                self._wait_reqs.pop(req_id, None)
                self._aborted_waits.discard(req_id)
                self._finished_aborted_recving.add(req_id)
                self._w.metrics.set_decode_active_waits(len(self._wait_reqs))
                logger.info(
                    "[PdConnector] D treating aborted wait as finished req=%s remote_req=%s error=%s",
                    req_id,
                    req.remote_request_id,
                    exc,
                )
                return
            self._mark_failed_locked(req_id, req, exc)

    def _record_rdma_wait_done(self, req_id: str, wait_s: float) -> None:
        with self._lock:
            if req_id not in self._wait_reqs:
                return
        self._w.metrics.record_decode_rdma_wait(wait_s)

    def _mark_failed_locked(self, req_id: str, req: WaitReqMeta, exc: BaseException) -> None:
        failed_blocks = flatten_block_ids(req.local_block_ids)
        self._failed_recving.add(req_id)
        self._failed_recving_for_meta.add(req_id)
        self._failed_block_ids.update(failed_blocks)
        self._aborted_waits.discard(req_id)
        self._wait_reqs.pop(req_id, None)
        self._w.metrics.set_decode_active_waits(len(self._wait_reqs))
        self._w.metrics.record_decode_wait(
            duration_s=_elapsed_seconds(req.scheduler_wait_ts_ns, time.time_ns()),
            rdma_wait_s=None,
            blocks=len(failed_blocks),
            success=False,
        )
        logger.error(
            "[PdConnector] D marked recv failed req=%s remote_req=%s blocks=%d error=%s",
            req_id,
            req.remote_request_id,
            len(failed_blocks),
            exc,
        )
        if self._rdma_waiter is not None:
            self._rdma_waiter.cancel(req_id)

    def _build_wait_handshake(
        self,
        req_id: str,
        block_ids: BlockIds,
        imm_id: int,
    ) -> PdHandshake:
        layers = []
        for layer_idx, layer_name in enumerate(self._w.layer_names):
            layer_block_ids = self._w.block_ids_for_layer(block_ids, layer_name)
            if not layer_block_ids:
                continue
            layers.append(
                self._remote_layout_with_mr_desc(
                    layer_name,
                    layer_idx,
                    (min(layer_block_ids),),
                )
            )
        assert layers, f"PdConnector D wait req={req_id} has no local KV blocks"
        return PdHandshake(
            request_id=req_id,
            engine_id=self._w.engine_id,
            tp_rank=self._w.tp_rank,
            tp_size=self._w.tp_size,
            block_size=next(iter(self._w.layouts.values())).block_size,
            layers=tuple(layers),
            imm_id=imm_id,
            fail_imm_id=_fail_imm_id(imm_id),
            abort_imm_id=_abort_imm_id(imm_id),
            expected_imm_count=self._expected_imm_count_for_local_rank(),
        )

    def _dispatch_prefill(
        self,
        req: WaitReqMeta,
        block_ids: BlockIds,
        imm_id: int,
    ) -> None:
        started_ts_ns = time.time_ns()
        all_handshakes = self._build_all_rank_handshake_dicts(
            req.done_request_id,
            block_ids,
            imm_id,
        )
        kv_transfer_params: dict[str, Any] = {
            "do_remote_prefill_sender": True,
            "target_engine_id": self._w.engine_id,
            "target_request_id": req.done_request_id,
            "pd_handshakes": all_handshakes,
            "pd_consumer_abort_returns_ack": True,
        }
        task = PrefillHttpTask(
            request_id=req.remote_request_id,
            prefill_url=req.prefill_url,
            model=req.model,
            prompt_token_ids=req.prompt_token_ids,
            max_tokens=req.prefill_max_tokens,
            kv_transfer_params=kv_transfer_params,
        )
        self._prefill_sender.submit(task)
        submitted_ts_ns = time.time_ns()
        self._w.metrics.record_prefill_http_submit(
            (submitted_ts_ns - started_ts_ns) / 1_000_000_000
        )
        logger.info(
            "[PdConnector] D rank0 dispatched prefill req=%s remote_req=%s ranks=%d blocks=%d "
            "build_ms=%.3f proxy_to_dispatch_ms=%.3f matched_to_dispatch_ms=%.3f "
            "scheduler_wait_to_dispatch_ms=%.3f ts_ns=%d",
            req.remote_request_id,
            req.done_request_id,
            len(all_handshakes),
            len(flatten_block_ids(block_ids)),
            (submitted_ts_ns - started_ts_ns) / 1_000_000,
            _elapsed_ms(req.proxy_start_ts_ns, submitted_ts_ns),
            _elapsed_ms(req.matched_ts_ns, submitted_ts_ns),
            _elapsed_ms(req.scheduler_wait_ts_ns, submitted_ts_ns),
            submitted_ts_ns,
        )

    def _build_all_rank_handshake_dicts(
        self,
        req_id: str,
        block_ids: BlockIds,
        imm_id: int,
    ) -> list[dict[str, Any]]:
        result = []
        block_size = self._peer_block_size
        assert block_size is not None, "peer layer templates are not initialized"
        for rank in range(self._w.tp_size):
            layers = []
            for layer in self._peer_layer_templates[rank]:
                layer_name = str(layer["layer_name"])
                layer_block_ids = self._w.block_ids_for_layer(block_ids, layer_name)
                if not layer_block_ids:
                    continue
                layer_dict = dict(layer)
                layer_dict["block_ids"] = sorted(layer_block_ids)
                layers.append(layer_dict)
            assert layers, f"PdConnector D wait req={req_id} has no local KV blocks"
            block_ids_by_layer = [tuple(layer["block_ids"]) for layer in layers]
            shared_block_ids = block_ids_by_layer[0]
            compact = all(ids == shared_block_ids for ids in block_ids_by_layer)
            if compact:
                payload_layers = [
                    {key: value for key, value in layer.items() if key != "block_ids"}
                    for layer in layers
                ]
            else:
                payload_layers = layers
            payload: dict[str, Any] = {
                "request_id": req_id,
                "engine_id": self._w.engine_id,
                "tp_rank": rank,
                "tp_size": self._w.tp_size,
                "block_size": block_size,
                "layers": payload_layers,
                "imm_id": imm_id,
                "fail_imm_id": _fail_imm_id(imm_id),
                "abort_imm_id": _abort_imm_id(imm_id),
                "expected_imm_count": self._expected_imm_count_for_rank(rank),
            }
            if compact:
                payload["block_ids"] = list(shared_block_ids)
            result.append(
                payload
            )
        return result

    def _expected_imm_count_for_local_rank(self) -> int:
        return self._expected_imm_count_for_rank(self._w.tp_rank)

    def _expected_imm_count_for_rank(self, rank: int) -> int:
        if not self._expected_imm_counts:
            self._refresh_expected_imm_counts()
        return self._expected_imm_counts.get(rank, 1)

    def _refresh_peer_layer_templates(self) -> None:
        self._peer_layer_templates = {}
        self._peer_block_size = next(iter(self._w.layouts.values())).block_size
        for rank, peer_layouts in self._peer_layouts.items():
            peer_mr_descs = self._peer_mr_descs[rank]
            layers = []
            for layer_idx, name in enumerate(self._w.layer_names):
                layer = replace(
                    peer_layouts[name].remote_layout(layer_idx, (0,)),
                    mr_desc=peer_mr_descs.get(name),
                )
                layers.append(layer_layout_to_compact_dict(layer))
            self._peer_layer_templates[rank] = tuple(layers)
        self._refresh_expected_imm_counts()

    def _refresh_expected_imm_counts(self) -> None:
        if self._w.use_mla:
            self._expected_imm_counts = dict.fromkeys(range(self._w.tp_size), 1)
            return

        local_layout = self._peer_layouts.get(self._w.tp_rank, self._w.layouts)
        first_layout = next(iter(local_layout.values()))
        remote_heads = int(getattr(first_layout, "num_kv_heads", 1))
        prefill_tp_size = int(
            extra_config_value(
                self._w.vllm_config,
                "pegaflow.pd.prefill_tp_size",
                self._w.tp_size,
            )
            or self._w.tp_size
        )
        total_heads = _total_num_kv_heads_from_config(self._w.vllm_config)
        if total_heads is None:
            total_heads = remote_heads * self._w.tp_size
        local_heads = _ceil_div(total_heads, prefill_tp_size)
        source_counts = decode_rank_source_counts(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=self._w.tp_size,
            local_num_kv_heads=local_heads,
            remote_num_kv_heads=remote_heads,
            total_num_kv_heads=total_heads,
            use_mla=False,
        )
        self._expected_imm_counts = {
            rank: source_counts.get(rank, 1) for rank in range(self._w.tp_size)
        }

    def _alloc_imm_id(self) -> int:
        imm_id = self._next_imm_id
        self._next_imm_id += 1
        if self._next_imm_id > 0xFFFF_FFFF:
            self._next_imm_id = 1
        return imm_id

    def _remote_layout_with_mr_desc(
        self,
        layer_name: str,
        layer_idx: int,
        block_ids: tuple[int, ...],
    ) -> LayerRemoteLayout:
        layout = self._w.layouts[layer_name].remote_layout(layer_idx, block_ids)
        registered = self._w._registered_layers.get(layer_name)
        if registered is None:
            return layout
        return replace(layout, mr_desc=registered.mr_desc)


@dataclass(frozen=True)
class _RdmaWaitTask:
    req_id: str
    generation: int
    remote_request_id: str
    done_request_id: str
    prefill_url: str | None
    rank: int
    block_count: int
    queued_ts_ns: int


class _AsyncRdmaDoneWaiter:
    """Background pool blocking on request RDMA IMM completion."""

    def __init__(
        self,
        rdma: RdmaPort,
        failure_callback: Any | None = None,
        success_callback: Any | None = None,
        max_workers: int = 16,
    ) -> None:
        self.rdma = rdma
        self._failure_callback = failure_callback
        self._success_callback = success_callback
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="pd-rdma-done-waiter",
        )
        self._submitted: dict[str, int] = {}
        self._cancelled: dict[str, set[int]] = {}
        self._next_generation: dict[str, int] = {}
        self._lock = threading.Lock()

    def submit(self, task: _RdmaWaitTask) -> _RdmaWaitTask | None:
        with self._lock:
            if task.req_id in self._submitted:
                return None
            generation = self._next_generation.get(task.req_id, 0) + 1
            self._next_generation[task.req_id] = generation
            task = replace(task, generation=generation)
            self._submitted[task.req_id] = generation
        logger.info(
            "[PdConnector] D RDMA wait queued req=%s remote_req=%s done_req=%s rank=%d blocks=%d prefill_url=%s queue_depth=%d",
            task.req_id,
            task.remote_request_id,
            task.done_request_id,
            task.rank,
            task.block_count,
            task.prefill_url or "<oob>",
            len(self._submitted),
        )
        self._executor.submit(self._run_task, task)
        return task

    def cancel(self, req_id: str) -> None:
        with self._lock:
            generation = self._submitted.pop(req_id, None)
            if generation is None:
                return
            self._cancelled.setdefault(req_id, set()).add(generation)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_task(self, task: _RdmaWaitTask) -> None:
        try:
            if self._is_cancelled(task.req_id, task.generation):
                logger.info(
                    "[PdConnector] D RDMA done wait cancelled before start req=%s remote_req=%s done_req=%s rank=%d blocks=%d",
                    task.req_id,
                    task.remote_request_id,
                    task.done_request_id,
                    task.rank,
                    task.block_count,
                )
                return
            start_ts_ns = time.time_ns()
            self.rdma.wait_done(task.req_id)
            done_ts_ns = time.time_ns()
            if self._is_cancelled(task.req_id, task.generation):
                logger.info(
                    "[PdConnector] D RDMA done wait cancelled req=%s remote_req=%s done_req=%s rank=%d blocks=%d queue_wait_ms=%.3f wait_ms=%.3f ts_ns=%d",
                    task.req_id,
                    task.remote_request_id,
                    task.done_request_id,
                    task.rank,
                    task.block_count,
                    (start_ts_ns - task.queued_ts_ns) / 1_000_000,
                    (done_ts_ns - start_ts_ns) / 1_000_000,
                    done_ts_ns,
                )
                return
            logger.info(
                "[PdConnector] D received RDMA done req=%s remote_req=%s done_req=%s rank=%d blocks=%d queue_wait_ms=%.3f wait_ms=%.3f ts_ns=%d",
                task.req_id,
                task.remote_request_id,
                task.done_request_id,
                task.rank,
                task.block_count,
                (start_ts_ns - task.queued_ts_ns) / 1_000_000,
                (done_ts_ns - start_ts_ns) / 1_000_000,
                done_ts_ns,
            )
            if self._success_callback is not None:
                self._success_callback(task.req_id, (done_ts_ns - start_ts_ns) / 1_000_000_000)
        except Exception as exc:
            logger.exception(
                "[PdConnector] D RDMA done wait failed req=%s remote_req=%s done_req=%s rank=%d blocks=%d",
                task.req_id,
                task.remote_request_id,
                task.done_request_id,
                task.rank,
                task.block_count,
            )
            if self._failure_callback is not None:
                self._failure_callback(task.req_id, exc)
        finally:
            with self._lock:
                if self._submitted.get(task.req_id) == task.generation:
                    self._submitted.pop(task.req_id, None)
                cancelled = self._cancelled.get(task.req_id)
                if cancelled is not None:
                    cancelled.discard(task.generation)
                    if not cancelled:
                        self._cancelled.pop(task.req_id, None)

    def _is_cancelled(self, req_id: str, generation: int) -> bool:
        with self._lock:
            return generation in self._cancelled.get(req_id, set())


def _all_gather_peer_info(
    layouts: dict[str, KvCacheLayout],
    mr_descs: dict[str, Any],
    tp_size: int,
) -> list[tuple[dict[str, KvCacheLayout], dict[str, Any]]]:
    import torch.distributed as dist

    gathered: list[tuple[dict[str, KvCacheLayout], dict[str, Any]] | None] = [None] * tp_size
    dist.all_gather_object(gathered, (layouts, mr_descs))
    return gathered  # type: ignore[return-value]


def _elapsed_ms(start_ts_ns: int, end_ts_ns: int) -> float:
    if start_ts_ns <= 0 or end_ts_ns <= 0 or end_ts_ns < start_ts_ns:
        return -1.0
    return (end_ts_ns - start_ts_ns) / 1_000_000


def _elapsed_seconds(start_ts_ns: int, end_ts_ns: int) -> float:
    if start_ts_ns <= 0 or end_ts_ns <= 0 or end_ts_ns < start_ts_ns:
        return 0.0
    return (end_ts_ns - start_ts_ns) / 1_000_000_000


def _total_num_kv_heads_from_config(vllm_config: Any) -> int | None:
    model_config = getattr(vllm_config, "model_config", None)
    getter = getattr(model_config, "get_total_num_kv_heads", None)
    if getter is not None:
        value = getter()
        return int(value) if value is not None else None
    hf_config = getattr(model_config, "hf_text_config", None)
    for config in (model_config, hf_config):
        value = getattr(config, "num_key_value_heads", None)
        if value is not None:
            return int(value)
    return None


def _ceil_div(value: int, divisor: int) -> int:
    assert value > 0
    assert divisor > 0
    return (value + divisor - 1) // divisor


def _fail_imm_id(imm_id: int) -> int:
    return imm_id ^ 0x8000_0000


def _abort_imm_id(imm_id: int) -> int:
    return imm_id ^ 0x4000_0000
