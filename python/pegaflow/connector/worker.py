"""
Worker-side connector logic.
"""

import pickle
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from pegaflow.connector.common import ConnectorContext, PegaConnectorMetadata, logger
from pegaflow.ipc_wrapper import CudaIPCWrapper
from pegaflow.logging_utils import timing_wrapper
from pegaflow.pegaflow import PyLoadState

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext


@dataclass
class SaveTask:
    layer_name: str
    attn_metadata: "AttentionMetadata"
    metadata: PegaConnectorMetadata
    request_ids: list[str]


class WorkerConnector:
    """Holds worker-only state and behaviors."""

    def __init__(self, context: ConnectorContext):
        self._ctx = context

        self._save_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()

        self._req_pending_layers: dict[str, int] = {}
        self._completed_saves: set[str] = set()
        self._save_completion_lock = threading.Lock()

        self._current_save_intents: set[str] = set()

        self._pending_loads: dict[str, PyLoadState] = {}
        self._pending_load_reqs: dict[str, set[str]] = {}
        self._load_completion_lock = threading.Lock()

        self._registered_layers: list[str] = []
        self._layer_name_to_id: dict[str, int] = {}

        self._finished_requests: set[str] = set()

    def shutdown(self) -> None:
        self.unregister_context()
        self._save_queue.put(None)
        self._save_thread.join()

    def unregister_context(self) -> None:
        if not self._registered_layers:
            return

        if self._ctx.tp_rank == 0:
            ok, message = self._ctx.engine_client.unregister_context(self._ctx.instance_id)
            if not ok:
                logger.warning("[PegaKVConnector] Unregister context failed: %s", message)

        self._registered_layers.clear()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert (
            self._ctx.device_id is not None
        ), "CUDA device id is unknown; cannot register KV caches"

        self._registered_layers = list(kv_caches.keys())

        self._layer_name_to_id.clear()
        for layer_id, layer_name in enumerate(kv_caches.keys()):
            self._layer_name_to_id[layer_name] = layer_id

        layout = "unknown"
        for layer_name, kv_cache in kv_caches.items():
            assert (
                kv_cache.storage_offset() == 0
            ), f"KV cache for {layer_name} must have zero storage offset"

            wrapper = CudaIPCWrapper(kv_cache)
            wrapper_bytes = pickle.dumps(wrapper)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            if len(shape) >= 2 and shape[0] == 2:
                num_blocks = shape[1]
                bytes_per_block = stride[1] * element_size
                kv_stride_bytes = stride[0] * element_size
                segments = 2
                layout = "KV-first"
            else:
                num_blocks = shape[0]
                bytes_per_block = stride[0] * element_size
                kv_stride_bytes = 0
                segments = 1
                layout = "blocks-first"

            assert (
                bytes_per_block != 0
            ), f"Invalid bytes_per_block for {layer_name}: stride={stride}"

            ok, message = self._ctx.engine_client.register_context(
                self._ctx.instance_id,
                self._ctx.namespace,
                self._ctx.tp_rank,
                self._ctx.tp_size,
                self._ctx.device_id,
                self._ctx.num_layers,
                layer_name,
                wrapper_bytes,
                num_blocks,
                bytes_per_block,
                kv_stride_bytes,
                segments,
            )

            if not ok:
                raise RuntimeError(f"Register context failed for {layer_name}: {message}")

        logger.info(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout) instance=%s",
            len(kv_caches),
            layout,
            self._ctx.instance_id,
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] | None = None
        finished_recving: set[str] | None = None

        with self._save_completion_lock:
            # 1. Add newly finished requests (if they have pending saves) to tracking
            self._finished_requests.update(finished_req_ids & self._req_pending_layers.keys())
            # 2. Identify requests whose saves have completed
            done_saves = self._completed_saves & self._finished_requests
            done_saves.update(self._completed_saves & finished_req_ids)

            if done_saves:
                # 3. Clean up completed requests
                self._completed_saves -= done_saves
                self._finished_requests -= done_saves
                finished_sending = done_saves

        with self._load_completion_lock:
            completed_reqs: set[str] = set()
            completed_shms: list[str] = []

            for shm_name, req_ids in self._pending_load_reqs.items():
                sample_req_id = next(iter(req_ids))
                load_state = self._pending_loads.get(sample_req_id)
                if load_state is None:
                    continue

                if load_state.is_ready():
                    state = load_state.get_state()
                    if state < 0:
                        logger.error(
                            "[PegaKVConnector] async load failed with state=%d for reqs=%s",
                            state,
                            req_ids,
                        )
                    else:
                        logger.debug(
                            "[PegaKVConnector] async load completed for %d reqs, shm=%s",
                            len(req_ids),
                            shm_name,
                        )
                    completed_reqs.update(req_ids)
                    completed_shms.append(shm_name)

            for shm_name in completed_shms:
                req_ids = self._pending_load_reqs.pop(shm_name, set())
                for req_id in req_ids:
                    self._pending_loads.pop(req_id, None)

            if completed_reqs:
                finished_recving = completed_reqs

        if finished_sending:
            logger.debug(
                "[PegaKVConnector] finished saving KV for requests: %s",
                finished_sending,
            )
        if finished_recving:
            logger.debug(
                "[PegaKVConnector] finished loading KV for requests: %s",
                finished_recving,
            )
        return (finished_sending, finished_recving)

    @timing_wrapper
    def start_load_kv(
        self,
        metadata: PegaConnectorMetadata,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        self._current_save_intents = set(metadata.save_intents.keys())

        if not metadata.load_intents:
            return

        total_requests = len(metadata.load_intents)
        load_start = time.perf_counter()

        all_block_ids: list[int] = []
        all_block_hashes: list[bytes] = []
        request_ids: list[str] = []

        for req_id, load_intent in metadata.load_intents.items():
            all_block_ids.extend(load_intent.block_ids)
            all_block_hashes.extend(load_intent.block_hashes)
            request_ids.append(req_id)

        if not all_block_ids:
            return

        target_layers: list[str] = []
        for layer_name, layer in forward_context.no_compile_layers.items():
            if hasattr(layer, "kv_cache"):
                target_layers.append(layer_name)

        if not target_layers:
            return

        load_state = PyLoadState()
        shm_name = load_state.shm_name()

        ok, message = self._ctx.engine_client.load(
            self._ctx.instance_id,
            self._ctx.tp_rank,
            self._ctx.device_id,
            shm_name,
            target_layers,
            all_block_ids,
            all_block_hashes,
        )

        if not ok:
            raise RuntimeError(f"Load request failed: {message}")

        num_layers = len(target_layers)
        num_blocks = len(all_block_ids)

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6

        with self._load_completion_lock:
            for req_id in request_ids:
                self._pending_loads[req_id] = load_state
            self._pending_load_reqs[shm_name] = set(request_ids)

        logger.debug(
            "[PegaKVConnector] started async load: %d blocks across %d layers for %d reqs, "
            "schedule %.0f us, shm=%s",
            num_blocks,
            num_layers,
            total_requests,
            schedule_time_us,
            shm_name,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        metadata: PegaConnectorMetadata,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        request_ids = list(metadata.save_intents.keys())
        if not request_ids:
            return

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id not in self._req_pending_layers:
                    self._req_pending_layers[req_id] = len(self._registered_layers)

        self._save_queue.put(
            SaveTask(
                layer_name=layer_name,
                attn_metadata=attn_metadata,
                metadata=metadata,
                request_ids=request_ids,
            )
        )

    @timing_wrapper
    def wait_for_save(self) -> None:
        skipped_requests: set[str] = set()

        with self._save_completion_lock:
            pending_layers = set(self._req_pending_layers.keys())
            skipped_requests = self._current_save_intents - pending_layers
            if skipped_requests:
                self._completed_saves.update(skipped_requests)

            self._current_save_intents = set()

            pending_reqs = len(self._req_pending_layers)
            if pending_reqs > 0:
                logger.debug(
                    "[PegaKVConnector] %d requests still have pending layer saves",
                    pending_reqs,
                )

        if skipped_requests:
            logger.debug(
                "[PegaKVConnector] Detected %d skipped saves (CUDA graph): %s",
                len(skipped_requests),
                skipped_requests,
            )
            self._handle_save_completion(skipped_requests, reason="CUDA graph skip")

    def _save_worker(self) -> None:
        logger.info("[PegaKVConnector] Save worker thread started")

        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break

            batch: list[SaveTask] = [task]
            while True:
                try:
                    t = self._save_queue.get_nowait()
                    if t is None:
                        self._process_save_batch(batch)
                        self._save_queue.task_done()
                        logger.info("[PegaKVConnector] Save worker thread stopped")
                        return
                    batch.append(t)
                except queue.Empty:
                    break

            self._process_save_batch(batch)
            for _ in batch:
                self._save_queue.task_done()

        logger.info("[PegaKVConnector] Save worker thread stopped")

    def _process_save_batch(self, batch: list[SaveTask]) -> None:
        saves_by_layer: dict[str, tuple[list[int], list[bytes]]] = {}
        all_request_ids: list[str] = []

        for task in batch:
            all_request_ids.extend(task.request_ids)

            if task.attn_metadata.block_table is None:
                continue

            for save_intent in task.metadata.save_intents.values():
                if not save_intent.block_ids:
                    continue

                if task.layer_name not in saves_by_layer:
                    saves_by_layer[task.layer_name] = ([], [])

                saves_by_layer[task.layer_name][0].extend(save_intent.block_ids)
                saves_by_layer[task.layer_name][1].extend(save_intent.block_hashes)

        if saves_by_layer:
            # Ensure all GPU kernels have completed before reading KV cache
            # Otherwise we may copy uninitialized memory (attention kernel is async)
            torch.cuda.synchronize()

            saves_list = [(name, ids, hashes) for name, (ids, hashes) in saves_by_layer.items()]

            try:
                ok, message = self._ctx.engine_client.save(
                    self._ctx.instance_id,
                    self._ctx.tp_rank,
                    self._ctx.device_id,
                    saves_list,
                )

                if not ok:
                    logger.error(
                        "[PegaKVConnector] Save batch failed: %s (continuing without save)",
                        message,
                    )
                else:
                    logger.debug(
                        "[PegaKVConnector] Batch saved %d layers, %d total blocks",
                        len(saves_list),
                        sum(len(ids) for _, ids, _ in saves_list),
                    )
            except Exception as e:
                logger.error(
                    "[PegaKVConnector] Save RPC exception: %s (continuing without save)",
                    e,
                )

        # Always decrement layer counter to release blocks, even if save failed
        self._decrement_layer_counter(all_request_ids)

    def _decrement_layer_counter(self, request_ids: list[str]) -> None:
        completed_reqs: list[str] = []

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id in self._req_pending_layers:
                    self._req_pending_layers[req_id] -= 1
                    assert (
                        self._req_pending_layers[req_id] >= 0
                    ), f"Layer count mismatch for request {req_id}: counter went negative"

                    if self._req_pending_layers[req_id] == 0:
                        self._completed_saves.add(req_id)
                        del self._req_pending_layers[req_id]
                        completed_reqs.append(req_id)

        self._handle_save_completion(completed_reqs)

    def _handle_save_completion(
        self, request_ids: Iterable[str], reason: str | None = None
    ) -> None:
        req_list = list(request_ids)
        if not req_list:
            return

        suffix = "" if not reason else f" ({reason})"
        layer_count = len(self._registered_layers) or self._ctx.num_layers
        for req_id in req_list:
            logger.debug(
                "[PegaKVConnector] Request %s all %d layers saved%s",
                req_id,
                layer_count,
                suffix,
            )


__all__ = ["WorkerConnector"]
