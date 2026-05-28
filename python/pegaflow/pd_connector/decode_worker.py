"""D-side (decode) worker logic — receives KV via RDMA."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.kv_params import ProducerKvParams
from pegaflow.pd_connector.layout import KvCacheLayout
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdHandshake,
    WaitReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.prefill import AsyncPrefillSender, PrefillHttpTask
from pegaflow.pd_connector.rdma import RdmaPort

if TYPE_CHECKING:
    from pegaflow.pd_connector.worker import PdWorkerConnector

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
            return
        try:
            gathered = _all_gather_peer_info(self._w.layouts, mr_descs, self._w.tp_size)
        except Exception:
            logger.warning(
                "[PdConnector] all_gather_object unavailable, rank 0 dispatch limited to local rank",
            )
            self._peer_layouts = {self._w.tp_rank: self._w.layouts}
            self._peer_mr_descs = {self._w.tp_rank: mr_descs}
            return
        for rank, (layouts, descs) in enumerate(gathered):
            self._peer_layouts[rank] = layouts
            self._peer_mr_descs[rank] = descs

    def process_wait_reqs(self, reqs_to_wait: dict[str, WaitReqMeta]) -> None:
        assert self._w.rdma is not None, "PdConnector RDMA port is not initialized"
        assert self._rdma_waiter is not None, "PdConnector RDMA waiter is not initialized"
        for req_id, req in reqs_to_wait.items():
            if req_id in self._wait_reqs:
                logger.info("[PdConnector] D wait req=%s already registered", req_id)
                continue
            self._wait_reqs[req_id] = req
            handshake = self._build_handshake(
                req.done_request_id,
                flatten_block_ids(req.local_block_ids),
            )
            self._w.rdma.open_request(req_id, handshake)
            local_block_count = len(flatten_block_ids(req.local_block_ids))
            self._rdma_waiter.submit(
                _RdmaWaitTask(
                    req_id=req_id,
                    remote_request_id=req.remote_request_id,
                    done_request_id=req.done_request_id,
                    prefill_url=req.prefill_url,
                    rank=self._w.tp_rank,
                    block_count=local_block_count,
                    queued_ts_ns=time.time_ns(),
                )
            )
            logger.info(
                "[PdConnector] D queued async wait req=%s remote_req=%s prefill_url=%s rank=%d blocks=%d ts_ns=%d",
                req_id,
                req.remote_request_id,
                req.prefill_url or "<oob>",
                self._w.tp_rank,
                local_block_count,
                time.time_ns(),
            )
            if req.prefill_url and self._w.tp_rank == 0:
                self._dispatch_prefill(req, handshake.imm_id)

    def release(self, req_id: str) -> None:
        self._wait_reqs.pop(req_id, None)

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

    def _build_handshake(self, req_id: str, block_ids: set[int]) -> PdHandshake:
        imm_id = self._alloc_imm_id()
        return PdHandshake(
            request_id=req_id,
            engine_id=self._w.engine_id,
            tp_rank=self._w.tp_rank,
            tp_size=self._w.tp_size,
            block_size=next(iter(self._w.layouts.values())).block_size,
            layers=tuple(
                self._remote_layout_with_mr_desc(layer_name, layer_idx, block_ids)
                for layer_idx, layer_name in enumerate(self._w.layer_names)
            ),
            imm_id=imm_id,
        )

    def _dispatch_prefill(self, req: WaitReqMeta, imm_id: int) -> None:
        block_ids = flatten_block_ids(req.local_block_ids)
        all_handshakes = self._build_all_rank_handshakes(
            req.done_request_id,
            block_ids,
            imm_id,
        )
        params = ProducerKvParams(
            target_engine_id=self._w.engine_id,
            target_request_id=req.done_request_id,
            handshakes=all_handshakes,
        )
        task = PrefillHttpTask(
            request_id=req.remote_request_id,
            prefill_url=req.prefill_url,
            model=req.model,
            prompt_token_ids=req.prompt_token_ids,
            max_tokens=req.prefill_max_tokens,
            kv_transfer_params=params.to_dict(),
        )
        self._prefill_sender.submit(task)
        logger.info(
            "[PdConnector] D rank0 dispatched prefill req=%s remote_req=%s ranks=%d ts_ns=%d",
            req.remote_request_id,
            req.done_request_id,
            len(all_handshakes),
            time.time_ns(),
        )

    def _build_all_rank_handshakes(
        self,
        req_id: str,
        block_ids: set[int],
        imm_id: int,
    ) -> tuple[PdHandshake, ...]:
        result = []
        block_size = next(iter(self._w.layouts.values())).block_size
        for rank in range(self._w.tp_size):
            peer_layouts = self._peer_layouts[rank]
            peer_mr_descs = self._peer_mr_descs[rank]
            layers = tuple(
                replace(
                    peer_layouts[name].remote_layout(layer_idx, block_ids),
                    mr_desc=peer_mr_descs.get(name),
                )
                for layer_idx, name in enumerate(self._w.layer_names)
            )
            result.append(
                PdHandshake(
                    request_id=req_id,
                    engine_id=self._w.engine_id,
                    tp_rank=rank,
                    tp_size=self._w.tp_size,
                    block_size=block_size,
                    layers=layers,
                    imm_id=imm_id,
                )
            )
        return tuple(result)

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
        block_ids: set[int],
    ) -> LayerRemoteLayout:
        layout = self._w.layouts[layer_name].remote_layout(layer_idx, block_ids)
        registered = self._w._registered_layers.get(layer_name)
        if registered is None:
            return layout
        return replace(layout, mr_desc=registered.mr_desc)


@dataclass(frozen=True)
class _RdmaWaitTask:
    req_id: str
    remote_request_id: str
    done_request_id: str
    prefill_url: str | None
    rank: int
    block_count: int
    queued_ts_ns: int


class _AsyncRdmaDoneWaiter:
    def __init__(self, rdma: RdmaPort) -> None:
        self.rdma = rdma
        self._queue: queue.Queue[_RdmaWaitTask | None] = queue.Queue()
        self._submitted: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="pd-rdma-done-waiter", daemon=True)
        self._thread.start()

    def submit(self, task: _RdmaWaitTask) -> None:
        if task.req_id in self._submitted:
            return
        self._submitted.add(task.req_id)
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

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            task = item
            start_ts_ns = time.time_ns()
            try:
                self.rdma.wait_done(task.req_id)
                done_ts_ns = time.time_ns()
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
                failed_ts_ns = time.time_ns()
                logger.exception(
                    "[PdConnector] D RDMA done wait failed req=%s remote_req=%s done_req=%s rank=%d blocks=%d queue_wait_ms=%.3f wait_ms=%.3f",
                    task.req_id,
                    task.remote_request_id,
                    task.done_request_id,
                    task.rank,
                    task.block_count,
                    (start_ts_ns - task.queued_ts_ns) / 1_000_000,
                    (failed_ts_ns - start_ts_ns) / 1_000_000,
                )


def _all_gather_peer_info(
    layouts: dict[str, KvCacheLayout],
    mr_descs: dict[str, Any],
    tp_size: int,
) -> list[tuple[dict[str, KvCacheLayout], dict[str, Any]]]:
    import torch.distributed as dist

    gathered: list[tuple[dict[str, KvCacheLayout], dict[str, Any]] | None] = [None] * tp_size
    dist.all_gather_object(gathered, (layouts, mr_descs))
    return gathered  # type: ignore[return-value]
