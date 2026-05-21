"""Worker-side logic for the experimental P/D connector."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.layout import (
    FlashAttnHndLayout,
    LayerBlockSlices,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (
    PdConnectorMetadata,
    PdHandshake,
    PdPrefillRequest,
    PushReqMeta,
    WaitReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.oob import InMemoryOobPort
from pegaflow.pd_connector.rdma import NoopRdmaPort, RdmaPort

logger = get_connector_logger()


class PdWorkerConnector:
    def __init__(
        self,
        vllm_config: Any,
        rdma: RdmaPort | None = None,
        oob: InMemoryOobPort | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rdma = rdma or NoopRdmaPort()
        self.oob = oob or InMemoryOobPort()
        self.engine_id = (
            getattr(getattr(vllm_config, "kv_transfer_config", None), "engine_id", None) or ""
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self.tp_rank = int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0)
        self.tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        self.layouts: dict[str, FlashAttnHndLayout] = {}
        self.layer_names: list[str] = []
        self._wait_reqs: dict[str, WaitReqMeta] = {}
        self._push_reqs: dict[str, PushReqMeta] = {}
        self._tracker = ChunkTracker()
        self._failed_blocks: set[int] = set()
        self._done_sender = _AsyncDoneSender()
        self._done_receiver: _AsyncDoneReceiver | None = None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.layouts = {
            layer_name: FlashAttnHndLayout.from_tensor(layer_name, tensor)
            for layer_name, tensor in kv_caches.items()
        }
        self.layer_names = list(kv_caches.keys())
        self.rdma.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        logger.info(
            "[PdConnector] registered %d FlashAttention HND KV cache layers",
            len(self.layouts),
        )

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        for req_id, req in metadata.reqs_to_wait.items():
            self._wait_reqs[req_id] = req
            if req.remote.done_endpoint:
                self._ensure_done_receiver(req.remote.done_endpoint)
            handshake = self._build_handshake(req_id, flatten_block_ids(req.local_block_ids))
            self.oob.publish_prefill_request(
                PdPrefillRequest(
                    request_id=req.remote_request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    producer_kv_transfer_params=_producer_params(
                        target_engine_id=self.engine_id,
                        target_request_id=req_id,
                        done_endpoint=req.remote.done_endpoint,
                    ),
                    handshake=handshake,
                )
            )
            self.rdma.register_remote(req_id, handshake)
            logger.info(
                "[PdConnector] D queued async wait req=%s remote_req=%s done_endpoint=%s",
                req_id,
                req.remote_request_id,
                req.remote.done_endpoint or "<rdma>",
            )

        for req_id, req in metadata.reqs_to_push.items():
            self._push_reqs[req_id] = req
            self._tracker.add_request(req_id)
            prefill_request = self.oob.get_prefill_request(req.target_request_id)
            handshake = prefill_request.handshake if prefill_request is not None else None
            self.rdma.register_remote(req_id, handshake)
            logger.info(
                "[PdConnector] P queued async push req=%s target_req=%s done_endpoint=%s",
                req_id,
                req.target_request_id,
                req.target.done_endpoint or "<rdma>",
            )

        for req_id in metadata.reqs_to_release:
            self._wait_reqs.pop(req_id, None)
            self._push_reqs.pop(req_id, None)
            self._tracker.remove(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert layer_name in self.layouts, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        self._drain_done_receiver()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        layout = self.layouts.get(layer_name)
        assert layout is not None, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        # Re-assert the runtime tensor. CUDA graph or backend changes must not
        # silently swap in a different layout.
        runtime_layout = FlashAttnHndLayout.from_tensor(layer_name, kv_layer)
        assert runtime_layout.shape == layout.shape, (
            f"PdConnector KV shape changed for {layer_name}: "
            f"registered={layout.shape} runtime={runtime_layout.shape}"
        )

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            return
        touched_blocks = unique_blocks_from_slot_mapping(slot_mapping, layout.block_size)
        if not touched_blocks:
            return

        layer_idx = self._layer_idx(layer_name)
        is_last_layer = layer_idx == len(self.layer_names) - 1
        for req_id, req in list(self._push_reqs.items()):
            req_blocks = flatten_block_ids(req.local_block_ids)
            selected = sorted(touched_blocks & req_blocks)
            if not selected:
                continue
            block_slices: list[LayerBlockSlices] = [
                layout.block_slices(block_id) for block_id in selected
            ]
            self.rdma.push_layer(req_id, layer_idx, block_slices)
            self._tracker.mark_layer_pushed(req_id, layer_idx)
            self._tracker.mark_blocks_pushed(req_id, set(selected))
            if is_last_layer and self._tracker.has_pushed_all_blocks(req_id, req_blocks):
                self.rdma.push_done(req_id)
                if req.target.done_endpoint:
                    self._done_sender.notify(
                        req.target.done_endpoint,
                        {
                            "type": "pd.done",
                            "request_id": req.target_request_id,
                            "producer_request_id": req_id,
                            "engine_id": self.engine_id,
                            "tp_rank": self.tp_rank,
                            "layers": len(self.layer_names),
                            "blocks": len(req_blocks),
                        },
                    )
                self._tracker.mark_done(req_id)
                logger.info(
                    "[PdConnector] P finished fake RDMA push req=%s target_req=%s layers=%d blocks=%d",
                    req_id,
                    req.target_request_id,
                    len(self.layer_names),
                    len(req_blocks),
                )

    def wait_for_save(self) -> None:
        return None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        self._drain_done_receiver()
        finished_sending = self.rdma.pop_finished_sending()
        finished_recving = self.rdma.pop_finished_recving()
        for req_id in finished_sending:
            self._push_reqs.pop(req_id, None)
            self._tracker.remove(req_id)
        for req_id in finished_recving:
            self._wait_reqs.pop(req_id, None)
        return finished_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._failed_blocks
        self._failed_blocks = set()
        return failed

    def shutdown(self) -> None:
        self._wait_reqs.clear()
        self._push_reqs.clear()
        if self._done_receiver is not None:
            self._done_receiver.close()
            self._done_receiver = None
        self._done_sender.close()

    def _layer_idx(self, layer_name: str) -> int:
        try:
            return self.layer_names.index(layer_name)
        except ValueError as exc:
            raise AssertionError(f"unknown layer {layer_name}") from exc

    def _build_handshake(self, req_id: str, block_ids: set[int]) -> PdHandshake:
        return PdHandshake(
            request_id=req_id,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=next(iter(self.layouts.values())).block_size,
            kv_layout="HND",
            layers=tuple(
                self.layouts[layer_name].remote_layout(layer_idx, block_ids)
                for layer_idx, layer_name in enumerate(self.layer_names)
            ),
        )

    def _ensure_done_receiver(self, endpoint: str) -> None:
        if self._done_receiver is None:
            self._done_receiver = _AsyncDoneReceiver(endpoint)
            return
        assert self._done_receiver.endpoint == endpoint, (
            "PdConnector currently supports one fake-RDMA done endpoint per worker; "
            f"existing={self._done_receiver.endpoint} new={endpoint}"
        )

    def _drain_done_receiver(self) -> None:
        if self._done_receiver is None:
            return
        for req_id in self._done_receiver.drain():
            if req_id in self._wait_reqs:
                self.rdma.mark_done(req_id)
                logger.info("[PdConnector] D received fake RDMA done req=%s", req_id)


class _AsyncDoneReceiver:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._queue: queue.Queue[str] = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pd-done-receiver", daemon=True)
        self._thread.start()
        logger.info("[PdConnector] fake RDMA done receiver listening endpoint=%s", endpoint)

    def drain(self) -> set[str]:
        done: set[str] = set()
        while True:
            try:
                done.add(self._queue.get_nowait())
            except queue.Empty:
                return done

    def close(self) -> None:
        self._closed.set()

    def _run(self) -> None:
        try:
            import zmq  # type: ignore[import-not-found]
        except Exception:
            logger.exception("[PdConnector] pyzmq is required for fake RDMA done receiver")
            return

        context = zmq.Context.instance()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._closed.is_set():
                events = dict(poller.poll(100))
                if socket not in events:
                    continue
                message = socket.recv_json()
                req_id = str(message["request_id"])
                self._queue.put(req_id)
                logger.info("[PdConnector] fake RDMA done message=%s", message)
        finally:
            socket.close(linger=0)


class _AsyncDoneSender:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="pd-done-sender", daemon=True)
        self._thread.start()

    def notify(self, endpoint: str, message: dict[str, Any]) -> None:
        self._queue.put((endpoint, message))

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        sockets: dict[str, Any] = {}
        context = None
        zmq_mod = None
        while True:
            item = self._queue.get()
            if item is None:
                break
            endpoint, message = item
            try:
                if zmq_mod is None:
                    import zmq as zmq_mod  # type: ignore[import-not-found]

                    context = zmq_mod.Context.instance()
                assert context is not None
                socket = sockets.get(endpoint)
                if socket is None:
                    socket = context.socket(zmq_mod.PUSH)
                    socket.setsockopt(zmq_mod.LINGER, 0)
                    socket.connect(endpoint)
                    sockets[endpoint] = socket
                socket.send_string(json.dumps(message))
                logger.info(
                    "[PdConnector] P sent fake RDMA done endpoint=%s message=%s",
                    endpoint,
                    message,
                )
            except Exception:
                logger.exception(
                    "[PdConnector] failed to send fake RDMA done endpoint=%s message=%s",
                    endpoint,
                    message,
                )
            finally:
                self._queue.task_done()
        for socket in sockets.values():
            socket.close(linger=0)


def _producer_params(
    target_engine_id: str,
    target_request_id: str,
    done_endpoint: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "do_remote_prefill_sender": True,
        "target_engine_id": target_engine_id,
        "target_request_id": target_request_id,
    }
    if done_endpoint:
        params["done_endpoint"] = done_endpoint
    return params
