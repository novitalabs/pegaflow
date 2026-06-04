"""D-side (decode) worker logic — receives KV via RDMA."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.layout import KvCacheLayout
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdHandshake,
    WaitReqMeta,
    flatten_block_ids,
    layer_layout_to_compact_dict,
)
from pegaflow.pd_connector.layout_mapping import (
    decode_rank_source_counts,
)
from pegaflow.pd_connector.prefill import AsyncPrefillSender, PrefillHttpTask
from pegaflow.pd_connector.rdma import RdmaPort

if TYPE_CHECKING:
    from pegaflow.pd_connector.worker import PdWorkerConnector

from pegaflow.pd_connector.config import extra_config_value

logger = get_connector_logger()


class DecodeHandler:
    """Handles D-side (decode) requests: RDMA receive, handshake, prefill dispatch."""

    def __init__(
        self,
        worker: PdWorkerConnector,
        prefill_sender: Any | None = None,
    ) -> None:
        self._w = worker
        self._wait_reqs: dict[str, WaitReqMeta] = {}
        self._peer_layouts: dict[int, dict[str, KvCacheLayout]] = {}
        self._peer_mr_descs: dict[int, dict[str, Any]] = {}
        self._peer_layer_templates: dict[int, tuple[dict[str, Any], ...]] = {}
        self._expected_imm_counts: dict[int, int] = {}
        self._peer_block_size: int | None = None
        self._next_imm_id = 1
        self._rdma_waiter: _AsyncRdmaDoneWaiter | None = (
            _AsyncRdmaDoneWaiter(worker.rdma) if worker.rdma is not None else None
        )
        self._prefill_sender = prefill_sender or AsyncPrefillSender()

    def init_rdma_waiter(self) -> None:
        """Called after RDMA port is built (during register_kv_caches)."""
        if self._rdma_waiter is None and self._w.rdma is not None:
            self._rdma_waiter = _AsyncRdmaDoneWaiter(self._w.rdma)

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
            self._wait_reqs[req_id] = req
            block_ids = flatten_block_ids(req.local_block_ids)
            imm_id = self._alloc_imm_id()
            wait_handshake = self._build_wait_handshake(
                req.done_request_id,
                block_ids,
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
                self._dispatch_prefill(req, block_ids, imm_id)

    def release(self, req_id: str) -> None:
        req = self._wait_reqs.pop(req_id, None)
        cancel_prefill = getattr(self._prefill_sender, "cancel", None)
        if req is not None and cancel_prefill is not None:
            cancel_prefill(req.remote_request_id)
        if self._rdma_waiter is not None:
            self._rdma_waiter.cancel(req_id)

    def finish_recving(self, finished_recving: set[str]) -> None:
        for req_id in finished_recving:
            self._wait_reqs.pop(req_id, None)
            self._w.rdma.close_request(req_id)

    def shutdown(self) -> None:
        self._wait_reqs.clear()
        if self._rdma_waiter is not None:
            self._rdma_waiter.close()
        close = getattr(self._prefill_sender, "close", None)
        if close is not None:
            close()

    @property
    def wait_reqs(self) -> dict[str, WaitReqMeta]:
        return self._wait_reqs

    def is_idle(self) -> bool:
        return not self._wait_reqs

    def _build_wait_handshake(
        self,
        req_id: str,
        block_ids: set[int],
        imm_id: int,
    ) -> PdHandshake:
        assert block_ids, f"PdConnector D wait req={req_id} has no local KV blocks"
        first_layer_name = self._w.layer_names[0]
        first_block_id = min(block_ids)
        return PdHandshake(
            request_id=req_id,
            engine_id=self._w.engine_id,
            tp_rank=self._w.tp_rank,
            tp_size=self._w.tp_size,
            block_size=next(iter(self._w.layouts.values())).block_size,
            layers=(
                self._remote_layout_with_mr_desc(
                    first_layer_name,
                    0,
                    (first_block_id,),
                ),
            ),
            imm_id=imm_id,
            expected_imm_count=self._expected_imm_count_for_local_rank(),
        )

    def _dispatch_prefill(
        self,
        req: WaitReqMeta,
        block_ids: set[int],
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
        logger.info(
            "[PdConnector] D rank0 dispatched prefill req=%s remote_req=%s ranks=%d blocks=%d "
            "build_ms=%.3f proxy_to_dispatch_ms=%.3f matched_to_dispatch_ms=%.3f "
            "scheduler_wait_to_dispatch_ms=%.3f ts_ns=%d",
            req.remote_request_id,
            req.done_request_id,
            len(all_handshakes),
            len(block_ids),
            (submitted_ts_ns - started_ts_ns) / 1_000_000,
            _elapsed_ms(req.proxy_start_ts_ns, submitted_ts_ns),
            _elapsed_ms(req.matched_ts_ns, submitted_ts_ns),
            _elapsed_ms(req.scheduler_wait_ts_ns, submitted_ts_ns),
            submitted_ts_ns,
        )

    def _build_all_rank_handshake_dicts(
        self,
        req_id: str,
        block_ids: set[int],
        imm_id: int,
    ) -> list[dict[str, Any]]:
        result = []
        block_size = self._peer_block_size
        assert block_size is not None, "peer layer templates are not initialized"
        ordered_block_ids = sorted(block_ids)
        for rank in range(self._w.tp_size):
            result.append(
                {
                    "request_id": req_id,
                    "engine_id": self._w.engine_id,
                    "tp_rank": rank,
                    "tp_size": self._w.tp_size,
                    "block_size": block_size,
                    "block_ids": list(ordered_block_ids),
                    "layers": list(self._peer_layer_templates[rank]),
                    "imm_id": imm_id,
                    "expected_imm_count": self._expected_imm_count_for_rank(rank),
                }
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
            self._expected_imm_counts = {rank: 1 for rank in range(self._w.tp_size)}
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
    """One background thread blocking on each request's RDMA IMM completion.

    submit() runs on the vLLM worker thread; _run() is the sole consumer.
    _submitted is shared by those two threads, hence the lock. A request is
    dropped from it once its wait resolves so the set tracks only in-flight waits.
    """

    def __init__(self, rdma: RdmaPort) -> None:
        self.rdma = rdma
        self._queue: queue.Queue[_RdmaWaitTask | None] = queue.Queue()
        self._submitted: dict[str, int] = {}
        self._cancelled: dict[str, set[int]] = {}
        self._next_generation: dict[str, int] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="pd-rdma-done-waiter", daemon=True)
        self._thread.start()

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
            self._queue.qsize(),
        )
        self._queue.put(task)
        return task

    def cancel(self, req_id: str) -> None:
        with self._lock:
            generation = self._submitted.pop(req_id, None)
            if generation is None:
                return
            self._cancelled.setdefault(req_id, set()).add(generation)

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
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
                    continue
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
                    continue
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
            except Exception:
                logger.exception(
                    "[PdConnector] D RDMA done wait failed req=%s remote_req=%s done_req=%s rank=%d blocks=%d",
                    task.req_id,
                    task.remote_request_id,
                    task.done_request_id,
                    task.rank,
                    task.block_count,
                )
            finally:
                with self._lock:
                    if self._submitted.get(task.req_id) == task.generation:
                        self._submitted.pop(task.req_id, None)
                    cancelled = self._cancelled.get(task.req_id)
                    if cancelled is not None:
                        cancelled.discard(task.generation)
                        if not cancelled:
                            self._cancelled.pop(task.req_id, None)
                self._queue.task_done()

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
