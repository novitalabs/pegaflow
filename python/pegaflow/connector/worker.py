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

from pegaflow.connector.common import (
    ConnectorContext,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    logger,
    parse_env_int,
)
from pegaflow.ipc_wrapper import CudaIPCWrapper
from pegaflow.pegaflow import PyLoadState

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext


_CROSS_LAYER_KEY = "ALL_LAYERS"

_LOAD_TIMEOUT_FLOOR_SECONDS = 30
_LOAD_TIMEOUT_RAW = parse_env_int("PEGA_LOAD_TIMEOUT_SECONDS", 120)
if _LOAD_TIMEOUT_RAW < _LOAD_TIMEOUT_FLOOR_SECONDS:
    logger.warning(
        "[PegaKVConnector] PEGA_LOAD_TIMEOUT_SECONDS=%d clamped to %d "
        "(minimum guard against load-path deadlock)",
        _LOAD_TIMEOUT_RAW,
        _LOAD_TIMEOUT_FLOOR_SECONDS,
    )
    _LOAD_TIMEOUT_RAW = _LOAD_TIMEOUT_FLOOR_SECONDS


@dataclass
class SaveTask:
    metadata: PegaConnectorMetadata
    request_ids: list[str]


class WorkerConnector:
    """Holds worker-only state and behaviors."""

    # Maximum time to wait for an in-flight load to reach terminal state before
    # giving up and reporting it as a load error to vLLM. Load is pure H2D once
    # prefetch has completed, so 120s is generous. Overridable via env var, but
    # values below _LOAD_TIMEOUT_FLOOR_SECONDS are clamped at module import time
    # to prevent production misconfiguration from dropping every in-flight load.
    LOAD_TIMEOUT_SECONDS: int = _LOAD_TIMEOUT_RAW

    def __init__(self, context: ConnectorContext):
        self._ctx = context

        self._save_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()

        self._req_pending_saves: set[str] = set()
        self._completed_saves: set[str] = set()
        self._save_completion_lock = threading.Lock()
        self._save_completion_events: dict[str, threading.Event] = {}
        self._current_metadata: PegaConnectorMetadata | None = None

        self._pending_loads: dict[str, PyLoadState] = {}
        self._pending_load_reqs: dict[str, set[str]] = {}
        self._pending_load_meta: dict[
            str, tuple[float, int, list[int]]
        ] = {}  # shm_name -> (start_time, num_blocks, block_ids)
        self._load_completion_lock = threading.Lock()

        # Failure surface for vLLM's get_block_ids_with_load_errors / get_finished.
        # Populated when start_load_kv fails synchronously or when an in-flight
        # load times out waiting for the server. Drained once per get_finished
        # and get_block_ids_with_load_errors call.
        self._failed_load_block_ids: set[int] = set()
        self._failed_load_reqs: set[str] = set()

        self._registered_layers: list[str] = []
        self._layer_name_to_id: dict[str, int] = {}
        self._torch_device: torch.device | None = None

        self._cross_layer_mode = False
        self._cross_layer_key = _CROSS_LAYER_KEY

        self._finished_requests: set[str] = set()

        # Stats collection
        self._stats = PegaKVConnectorStats()
        self._stats_lock = threading.Lock()

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
        assert self._ctx.device_id is not None, (
            "CUDA device id is unknown; cannot register KV caches"
        )

        self._registered_layers = list(kv_caches.keys())
        self._torch_device = next(iter(kv_caches.values())).device

        self._layer_name_to_id.clear()
        for layer_id, layer_name in enumerate(kv_caches.keys()):
            self._layer_name_to_id[layer_name] = layer_id

        # Use actual number of registered layers, not model's num_hidden_layers.
        # This is important for models like DSA where indexer layers are separate.
        # With PP>1 each rank registers only its local layers; the Rust engine
        # grows the instance-wide total dynamically via verify_topology.
        actual_num_layers = len(kv_caches)

        layout = "unknown"

        layer_names = []
        ipc_wrappers = []
        layer_num_blocks = []
        layer_bytes_per_block = []
        layer_kv_stride_bytes = []
        layer_segments = []

        for layer_name, kv_cache in kv_caches.items():
            assert kv_cache.storage_offset() == 0, (
                f"KV cache for {layer_name} must have zero storage offset"
            )

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

            assert bytes_per_block != 0, (
                f"Invalid bytes_per_block for {layer_name}: stride={stride}"
            )

            layer_names.append(layer_name)
            ipc_wrappers.append(wrapper_bytes)
            layer_num_blocks.append(num_blocks)
            layer_bytes_per_block.append(bytes_per_block)
            layer_kv_stride_bytes.append(kv_stride_bytes)
            layer_segments.append(segments)

        ok, message = self._ctx.engine_client.register_context_batch(
            self._ctx.instance_id,
            self._ctx.namespace,
            self._ctx.effective_tp_rank,
            self._ctx.effective_tp_size,
            self._ctx.world_size,
            self._ctx.device_id,
            actual_num_layers,
            layer_names,
            ipc_wrappers,
            layer_num_blocks,
            layer_bytes_per_block,
            layer_kv_stride_bytes,
            layer_segments,
        )

        if not ok:
            raise RuntimeError(f"Register context failed for {layer_name}: {message}")

        logger.info(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout) instance=%s",
            len(kv_caches),
            layout,
            self._ctx.instance_id,
        )

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        self._cross_layer_mode = True
        if self._ctx.pp_size > 1:
            self._cross_layer_key = f"{_CROSS_LAYER_KEY}_pp{self._ctx.pp_rank}"
        else:
            self._cross_layer_key = _CROSS_LAYER_KEY
        self.register_kv_caches({self._cross_layer_key: kv_cache})

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] | None = None
        finished_recving: set[str] | None = None

        with self._save_completion_lock:
            # 1. Add newly finished requests (if they have pending saves) to tracking
            self._finished_requests.update(finished_req_ids & self._req_pending_saves)
            # 2. Identify requests whose saves have completed
            done_saves = self._completed_saves & self._finished_requests
            done_saves.update(self._completed_saves & finished_req_ids)

            if done_saves:
                # 3. Clean up completed requests
                self._completed_saves -= done_saves
                self._finished_requests -= done_saves
                finished_sending = done_saves

        timeout_triggered = False
        with self._load_completion_lock:
            completed_reqs: set[str] = set()
            completed_shms: list[str] = []
            load_stats_to_record: list[tuple[float, int, bool]] = []
            now = time.perf_counter()

            for shm_name, req_ids in self._pending_load_reqs.items():
                sample_req_id = next(iter(req_ids))
                load_state = self._pending_loads.get(sample_req_id)
                if load_state is None:
                    continue

                meta = self._pending_load_meta.get(shm_name)
                ready = load_state.is_ready()
                timed_out = (
                    not ready and meta is not None and (now - meta[0]) > self.LOAD_TIMEOUT_SECONDS
                )

                if ready:
                    state = load_state.get_state()
                    success = state >= 0
                    if not success:
                        logger.error(
                            "[PegaKVConnector] async_load_failed: reqs=%s state=%d",
                            req_ids,
                            state,
                        )
                        if meta is not None:
                            self._failed_load_block_ids.update(meta[2])
                    else:
                        logger.info(
                            "[PegaKVConnector] async_load_completed: reqs=%s",
                            req_ids,
                        )

                    if meta is not None:
                        start_time, num_blocks, _ = meta
                        duration = now - start_time
                        load_stats_to_record.append((duration, num_blocks, success))

                    completed_reqs.update(req_ids)
                    completed_shms.append(shm_name)
                elif timed_out:
                    assert meta is not None
                    start_time, num_blocks, block_ids = meta
                    duration = now - start_time
                    logger.error(
                        "[PegaKVConnector] load_timeout: reqs=%s shm=%s elapsed=%.1fs "
                        "blocks=%d (reporting as load errors)",
                        req_ids,
                        shm_name,
                        duration,
                        num_blocks,
                    )
                    self._failed_load_block_ids.update(block_ids)
                    load_stats_to_record.append((duration, num_blocks, False))
                    completed_reqs.update(req_ids)
                    completed_shms.append(shm_name)
                    timeout_triggered = True

            for shm_name in completed_shms:
                shm_req_ids = self._pending_load_reqs.pop(shm_name, set())
                self._pending_load_meta.pop(shm_name, None)
                for req_id in shm_req_ids:
                    self._pending_loads.pop(req_id, None)

            # Drain sync-failure reqs recorded by start_load_kv so they also
            # reach vLLM as finished_recving in this pass.
            if self._failed_load_reqs:
                completed_reqs.update(self._failed_load_reqs)
                self._failed_load_reqs = set()

            if completed_reqs:
                finished_recving = completed_reqs

        if timeout_triggered:
            self._ctx.state_manager.mark_unavailable("load timeout")

        # Record load stats outside the lock
        if load_stats_to_record:
            with self._stats_lock:
                for duration, num_blocks, success in load_stats_to_record:
                    self._stats.record_load(duration, num_blocks, success)

        if finished_sending:
            logger.info(
                "[PegaKVConnector] async_save_completed: reqs=%s",
                finished_sending,
            )
        if finished_recving:
            logger.debug(
                "[PegaKVConnector] finished loading KV for requests: %s",
                finished_recving,
            )
        return (finished_sending, finished_recving)

    def start_load_kv(
        self,
        metadata: PegaConnectorMetadata,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        self._current_metadata = metadata

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

        if self._cross_layer_mode:
            target_layers = [self._cross_layer_key]
        else:
            target_layers = []
            for layer_name, layer in forward_context.no_compile_layers.items():
                if hasattr(layer, "kv_cache"):
                    target_layers.append(layer_name)

        if not target_layers:
            return

        load_state = PyLoadState()
        shm_name = load_state.shm_name()

        try:
            ok, message = self._ctx.engine_client.load(
                self._ctx.instance_id,
                self._ctx.effective_tp_rank,
                self._ctx.device_id,
                shm_name,
                target_layers,
                all_block_ids,
                all_block_hashes,
            )
        except Exception as e:
            logger.error(
                "[PegaKVConnector] Load RPC exception: %s (reqs=%s blocks=%d, "
                "marking blocks as load errors)",
                e,
                request_ids,
                len(all_block_ids),
            )
            self._record_load_failure(request_ids, all_block_ids, load_start)
            self._ctx.state_manager.mark_unavailable(f"load rpc exception: {e}")
            return

        if not ok:
            logger.error(
                "[PegaKVConnector] Load RPC failed: %s (reqs=%s blocks=%d, "
                "marking blocks as load errors)",
                message,
                request_ids,
                len(all_block_ids),
            )
            self._record_load_failure(request_ids, all_block_ids, load_start)
            self._ctx.state_manager.mark_unavailable(f"load rpc failed: {message}")
            return

        num_layers = len(target_layers)
        num_blocks = len(all_block_ids)

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6

        with self._load_completion_lock:
            for req_id in request_ids:
                self._pending_loads[req_id] = load_state
            self._pending_load_reqs[shm_name] = set(request_ids)
            # Keep load_start as the shared baseline so timeout and stats duration
            # are comparable to the sync-failure path (which also uses load_start).
            # all_block_ids is not mutated after this point; keep the reference
            # instead of an extra defensive copy.
            self._pending_load_meta[shm_name] = (
                load_start,
                num_blocks,
                all_block_ids,
            )

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

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs whose load failed since the last call, then clear.

        vLLM calls this each forward pass and re-schedules reported blocks for
        local recomputation. Failures may come from synchronous RPC errors in
        start_load_kv or from in-flight load timeouts detected in get_finished.
        """
        with self._load_completion_lock:
            failed = self._failed_load_block_ids
            self._failed_load_block_ids = set()
        return failed

    def _record_load_failure(
        self,
        request_ids: list[str],
        block_ids: list[int],
        start_time: float,
    ) -> None:
        """Record a synchronous load RPC failure for later reporting to vLLM."""
        duration = time.perf_counter() - start_time
        with self._load_completion_lock:
            self._failed_load_reqs.update(request_ids)
            self._failed_load_block_ids.update(block_ids)
        with self._stats_lock:
            self._stats.record_load(duration, len(block_ids), success=False)

    def save_kv_layer(
        self,
        metadata: PegaConnectorMetadata,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Save is metadata-driven and submitted from wait_for_save() outside
        # layer callbacks so CUDA graph replay cannot suppress it.
        pass

    def wait_for_save(self) -> None:
        metadata = self._current_metadata
        self._current_metadata = None
        if metadata is None or not metadata.save_intents:
            return

        request_ids = list(metadata.save_intents.keys())

        if self._should_skip_save_submission():
            logger.debug(
                "[PegaKVConnector] Skipping save submission on non-zero TP rank for MLA "
                "without DCP: tp_rank=%s reqs=%s",
                self._ctx.tp_rank,
                request_ids,
            )
            with self._save_completion_lock:
                for req_id in request_ids:
                    self._req_pending_saves.add(req_id)
            self._complete_save_requests(request_ids)
            return

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id not in self._req_pending_saves:
                    self._req_pending_saves.add(req_id)
                    self._save_completion_events[req_id] = threading.Event()

        self._save_queue.put(SaveTask(metadata=metadata, request_ids=request_ids))

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

            for save_intent in task.metadata.save_intents.values():
                if not save_intent.block_ids:
                    continue

                if self._cross_layer_mode:
                    target_layers = (self._cross_layer_key,)
                else:
                    assert self._registered_layers, (
                        "KV caches must be registered before submitting save intents"
                    )
                    target_layers = tuple(self._registered_layers)

                for layer_name in target_layers:
                    if layer_name not in saves_by_layer:
                        saves_by_layer[layer_name] = ([], [])

                    saves_by_layer[layer_name][0].extend(save_intent.block_ids)
                    saves_by_layer[layer_name][1].extend(save_intent.block_hashes)

        if saves_by_layer:
            # Ensure all GPU kernels have completed before reading KV cache
            # Otherwise we may copy uninitialized memory (attention kernel is async)
            torch.cuda.synchronize(self._torch_device)

            saves_list = [(name, ids, hashes) for name, (ids, hashes) in saves_by_layer.items()]
            total_blocks = sum(len(ids) for _, ids, _ in saves_list)

            save_start = time.perf_counter()
            success = False

            try:
                ok, message = self._ctx.engine_client.save(
                    self._ctx.instance_id,
                    self._ctx.effective_tp_rank,
                    self._ctx.device_id,
                    saves_list,
                )

                if not ok:
                    logger.error(
                        "[PegaKVConnector] Save batch failed: %s (continuing without save)",
                        message,
                    )
                else:
                    success = True
                    logger.debug(
                        "[PegaKVConnector] Batch saved %d layers, %d total blocks",
                        len(saves_list),
                        total_blocks,
                    )
            except Exception as e:
                logger.error(
                    "[PegaKVConnector] Save RPC exception: %s (continuing without save)",
                    e,
                )

            save_duration = time.perf_counter() - save_start

            # Record stats
            with self._stats_lock:
                self._stats.record_save(save_duration, total_blocks, success)

        # Always complete the request save lifecycle, even if save failed.
        self._complete_save_requests(all_request_ids)

    def _complete_save_requests(self, request_ids: list[str]) -> None:
        completed_reqs: list[str] = []

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id in self._req_pending_saves:
                    self._req_pending_saves.discard(req_id)
                    self._completed_saves.add(req_id)
                    completed_reqs.append(req_id)
                    event = self._save_completion_events.pop(req_id, None)
                    if event:
                        event.set()

        self._handle_save_completion(completed_reqs)

    def _handle_save_completion(
        self, request_ids: Iterable[str], reason: str | None = None
    ) -> None:
        req_list = list(request_ids)
        if not req_list:
            return

        suffix = "" if not reason else f" ({reason})"
        for req_id in req_list:
            logger.debug(
                "[PegaKVConnector] Request %s save completed%s",
                req_id,
                suffix,
            )

    def _should_skip_save_submission(self) -> bool:
        return self._ctx.is_mla and self._ctx.dcp_world_size == 1 and (self._ctx.tp_rank or 0) != 0

    def handle_preemptions(self, preempted_req_ids: set[str] | None) -> None:
        """Wait for preempted requests' saves to complete before blocks are reused.

        Called by vLLM BEFORE preempted blocks are overwritten. This prevents
        data corruption when async saves are still reading from blocks that
        will be reassigned to new requests.
        """
        if not preempted_req_ids:
            return

        events_to_wait: list[tuple[str, threading.Event]] = []
        with self._save_completion_lock:
            for req_id in preempted_req_ids:
                event = self._save_completion_events.get(req_id)
                if event:
                    events_to_wait.append((req_id, event))

        if events_to_wait:
            logger.info(
                "[PegaKVConnector] preemption: waiting for %d requests' saves: %s",
                len(events_to_wait),
                [req_id for req_id, _ in events_to_wait],
            )
            for req_id, event in events_to_wait:
                event.wait()
                logger.info("[PegaKVConnector] preemption: req=%s save completed", req_id)
        else:
            logger.info(
                "[PegaKVConnector] preemption: %d requests (no pending saves)",
                len(preempted_req_ids),
            )

    def get_stats(self) -> PegaKVConnectorStats | None:
        """Get and reset worker stats for the current interval."""
        with self._stats_lock:
            # Add current queue depth as gauge
            with self._save_completion_lock:
                self._stats.data["pending_save_requests"] = len(self._req_pending_saves)

            if self._stats.is_empty():
                return None
            return self._stats.clone_and_reset()


__all__ = ["WorkerConnector"]
