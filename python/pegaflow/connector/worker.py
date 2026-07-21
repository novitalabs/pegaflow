"""
Worker-side connector logic.
"""

import pickle
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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


_KVCacheLayout = Literal["KV-first", "blocks-first"]


@dataclass(frozen=True)
class _KVCacheRegistrationInfo:
    layout: _KVCacheLayout
    num_blocks: int
    bytes_per_block: int
    kv_stride_bytes: int
    segments: int
    physical_blocks_per_logical_block: int


def _infer_kv_cache_registration(
    kv_cache: torch.Tensor,
    logical_block_size: int,
    *,
    is_mla: bool = False,
) -> _KVCacheRegistrationInfo:
    """Infer the PegaFlow registration from a vLLM KV cache tensor.

    vLLM may split one scheduler/manager block into multiple kernel KV rows
    when the attention backend only supports a smaller kernel block size. For
    example, FlashMLA supports 64-token kernel blocks while the manager can use
    128-token blocks. PegaFlow stores hashes at scheduler-block granularity, so
    each registered PegaFlow block must cover all physical rows for that logical
    block.
    """
    shape = tuple(kv_cache.shape)
    stride = tuple(kv_cache.stride())
    element_size = kv_cache.element_size()

    if logical_block_size <= 0:
        raise ValueError(f"logical block size must be > 0, got {logical_block_size}")

    if not is_mla:
        if len(shape) >= 2 and shape[0] == 2:
            layout = "KV-first"
            num_blocks = shape[1]
            bytes_per_block = stride[1] * element_size
            kv_stride_bytes = stride[0] * element_size
            segments = 2
        else:
            layout = "blocks-first"
            num_blocks = shape[0]
            bytes_per_block = stride[0] * element_size
            kv_stride_bytes = 0
            segments = 1

        if num_blocks <= 0:
            raise ValueError(f"physical block count must be > 0, got {num_blocks}")
        if bytes_per_block == 0:
            raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")

        return _KVCacheRegistrationInfo(
            layout=layout,
            num_blocks=num_blocks,
            bytes_per_block=bytes_per_block,
            kv_stride_bytes=kv_stride_bytes,
            segments=segments,
            physical_blocks_per_logical_block=1,
        )

    layout = "blocks-first"
    physical_num_blocks = shape[0]
    physical_block_size = shape[1] if len(shape) >= 2 else logical_block_size
    physical_bytes_per_block = stride[0] * element_size
    kv_stride_bytes = 0
    segments = 1

    if physical_num_blocks <= 0:
        raise ValueError(f"physical block count must be > 0, got {physical_num_blocks}")
    if physical_block_size <= 0:
        raise ValueError(f"physical block size must be > 0, got {physical_block_size}")
    if logical_block_size % physical_block_size != 0:
        raise ValueError(
            "logical block size must be a multiple of physical block size "
            f"(logical={logical_block_size}, physical={physical_block_size})"
        )

    physical_blocks_per_logical_block = logical_block_size // physical_block_size
    if physical_num_blocks % physical_blocks_per_logical_block != 0:
        raise ValueError(
            "physical block count must be divisible by physical/logical split ratio "
            f"(physical_blocks={physical_num_blocks}, ratio={physical_blocks_per_logical_block})"
        )

    bytes_per_block = physical_bytes_per_block * physical_blocks_per_logical_block
    if bytes_per_block == 0:
        raise ValueError(f"Invalid bytes_per_block: shape={shape} stride={stride}")

    return _KVCacheRegistrationInfo(
        layout=layout,
        num_blocks=physical_num_blocks // physical_blocks_per_logical_block,
        bytes_per_block=bytes_per_block,
        kv_stride_bytes=kv_stride_bytes,
        segments=segments,
        physical_blocks_per_logical_block=physical_blocks_per_logical_block,
    )


class WorkerConnector:
    """Holds worker-only state and behaviors."""

    # Maximum time to wait for an in-flight load to reach terminal state before
    # giving up and reporting it as a load error to vLLM. Load is pure H2D once
    # prefetch has completed, so 120s is generous. Overridable via env var, but
    # values below _LOAD_TIMEOUT_FLOOR_SECONDS are clamped at module import time
    # to prevent production misconfiguration from dropping every in-flight load.
    LOAD_TIMEOUT_SECONDS: int = _LOAD_TIMEOUT_RAW

    def __init__(
        self,
        context: ConnectorContext,
        vllm_config=None,
        kv_cache_config=None,
    ):
        self._ctx = context
        self._kv_cache_config = kv_cache_config
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        self._use_mla_layer_split_registration = context.is_mla and bool(
            additional_config.get("mla_layer_split_kv_cache", False)
        )

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
        # Page-first storage: all layers of a block in one host page, one slot
        # per tp_rank. Saves distribute by block stripe instead of by layer.
        self._page_first: bool = False
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
        """Register exactly the KV caches vLLM built on this device.

        The engine derives the instance-wide layer-id space once every worker
        has registered, so optional speculative MTP layers, external drafters,
        and hybrid attention layouts need no connector-side layer accounting.
        """
        assert self._ctx.device_id is not None, (
            "CUDA device id is unknown; cannot register KV caches"
        )

        if self._use_mla_layer_split_registration:
            kv_cache_tensors = getattr(self._kv_cache_config, "kv_cache_tensors", None)
            if not kv_cache_tensors:
                raise RuntimeError(
                    "Layer-split KV cache registration requires kv_cache_config.kv_cache_tensors"
                )

            layer_names = [
                layer_name
                for kv_cache_tensor in kv_cache_tensors
                for layer_name in (getattr(kv_cache_tensor, "shared_by", None) or ())
            ]
            if not layer_names:
                raise RuntimeError("Layer-split KV cache registration selected no layers")

            missing_layer_names = [
                layer_name for layer_name in layer_names if layer_name not in kv_caches
            ]
            if missing_layer_names:
                raise RuntimeError(
                    "Layer-split KV cache registration is missing layers: "
                    f"{missing_layer_names[:8]}"
                )

            kv_caches = {layer_name: kv_caches[layer_name] for layer_name in layer_names}
        if not kv_caches:
            raise RuntimeError("No KV cache layers were selected for registration")

        self._registered_layers = list(kv_caches.keys())
        self._page_first = self._use_page_first()
        self._torch_device = next(iter(kv_caches.values())).device

        layout = "unknown"

        layer_names = []
        ipc_wrappers = []
        layer_num_blocks = []
        layer_bytes_per_block = []
        layer_kv_stride_bytes = []
        layer_segments = []
        split_layer_count = 0
        split_blocks_per_logical = 1
        split_logical_blocks = 0

        for layer_name, kv_cache in kv_caches.items():
            assert kv_cache.storage_offset() == 0, (
                f"KV cache for {layer_name} must have zero storage offset"
            )

            wrapper = CudaIPCWrapper(kv_cache)
            wrapper_bytes = pickle.dumps(wrapper)

            registration = _infer_kv_cache_registration(
                kv_cache,
                self._ctx.block_size,
                is_mla=self._ctx.is_mla,
            )
            layout = registration.layout

            layer_names.append(layer_name)
            ipc_wrappers.append(wrapper_bytes)
            layer_num_blocks.append(registration.num_blocks)
            layer_bytes_per_block.append(registration.bytes_per_block)
            layer_kv_stride_bytes.append(registration.kv_stride_bytes)
            layer_segments.append(registration.segments)

            if registration.physical_blocks_per_logical_block > 1:
                split_layer_count += 1
                split_blocks_per_logical = registration.physical_blocks_per_logical_block
                split_logical_blocks = registration.num_blocks
                logger.debug(
                    "[PegaKVConnector] Registered %s with virtual block split: "
                    "logical_block_size=%d physical_blocks_per_logical=%d logical_blocks=%d",
                    layer_name,
                    self._ctx.block_size,
                    registration.physical_blocks_per_logical_block,
                    registration.num_blocks,
                )

        torch.cuda.synchronize(self._torch_device)
        ok, message = self._ctx.engine_client.register_context_batch(
            self._ctx.instance_id,
            self._ctx.namespace,
            self._ctx.effective_tp_rank,
            self._ctx.pp_rank,
            self._ctx.effective_tp_size,
            self._ctx.world_size,
            self._ctx.device_id,
            layer_names,
            ipc_wrappers,
            layer_num_blocks,
            layer_bytes_per_block,
            layer_kv_stride_bytes,
            layer_segments,
            self._ctx.transfer_backend,
            self._page_first,
        )

        if not ok:
            if "PegaFlow version mismatch" in message:
                raise RuntimeError(f"Register context failed: {message}")
            raise RuntimeError(f"Register context batch failed for layers {layer_names}: {message}")

        if split_layer_count:
            logger.info(
                "[PegaKVConnector] Registered %d/%d KV cache layers with virtual "
                "block split: logical_block_size=%d physical_blocks_per_logical=%d "
                "logical_blocks=%d",
                split_layer_count,
                len(kv_caches),
                self._ctx.block_size,
                split_blocks_per_logical,
                split_logical_blocks,
            )

        logger.debug(
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
                        logger.debug(
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
            logger.debug(
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
        loads: list[tuple[bytes, list[int]]] = []
        request_ids: list[str] = []

        for req_id, load_intent in metadata.load_intents.items():
            block_ids = list(load_intent.block_ids)
            all_block_ids.extend(block_ids)
            loads.append((load_intent.lease, block_ids))
            request_ids.append(req_id)

        if not all_block_ids:
            return

        if self._cross_layer_mode:
            target_layers = [self._cross_layer_key]
        else:
            assert self._registered_layers, (
                "KV caches must be registered before submitting load intents"
            )
            target_layers = list(self._registered_layers)

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
                loads,
            )
        except Exception as e:
            logger.error(
                "[PegaKVConnector] Load RPC exception: %s (reqs=%s blocks=%d, "
                "marking blocks as load errors)",
                e,
                request_ids,
                len(all_block_ids),
            )
            self._release_load_leases(loads)
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
            self._release_load_leases(loads)
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

    def _release_load_leases(self, loads: list[tuple[bytes, list[int]]]) -> None:
        seen: set[bytes] = set()
        for lease, _block_ids in loads:
            if not lease or lease in seen:
                continue
            seen.add(lease)
            try:
                self._ctx.engine_client.release(lease)
            except Exception:
                logger.exception(
                    "[PegaKVConnector] load failure lease release exception: lease_len=%d",
                    len(lease),
                )

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

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id not in self._req_pending_saves:
                    self._req_pending_saves.add(req_id)
                    self._save_completion_events[req_id] = threading.Event()

        self._save_queue.put(SaveTask(metadata=metadata, request_ids=request_ids))

    def _save_worker(self) -> None:
        logger.debug("[PegaKVConnector] Save worker thread started")

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
                        logger.debug("[PegaKVConnector] Save worker thread stopped")
                        return
                    batch.append(t)
                except queue.Empty:
                    break

            self._process_save_batch(batch)
            for _ in batch:
                self._save_queue.task_done()

        logger.debug("[PegaKVConnector] Save worker thread stopped")

    def _process_save_batch(self, batch: list[SaveTask]) -> None:
        saves_by_layer: dict[str, tuple[list[int], list[bytes]]] = {}
        all_request_ids: list[str] = []

        for task in batch:
            all_request_ids.extend(task.request_ids)

            for save_intent in task.metadata.save_intents.values():
                if not save_intent.block_ids:
                    continue

                block_ids = save_intent.block_ids
                block_hashes = save_intent.block_hashes

                if self._cross_layer_mode:
                    target_layers = (self._cross_layer_key,)
                elif self._page_first:
                    # Page-first: a block's page holds a whole shard's layers, so
                    # this rank writes all its registered layers (its shard).
                    assert self._registered_layers, (
                        "KV caches must be registered before submitting save intents"
                    )
                    target_layers = tuple(self._registered_layers)
                    if not self._use_mla_layer_split_registration:
                        # Full-replica (one shard): every rank holds all layers,
                        # so spread the whole-page writes across ranks by block
                        # stripe. Layer-split ranks are each the sole writer of
                        # their shard and keep the full block set (no striping).
                        block_ids, block_hashes = self._block_shard(save_intent)
                else:
                    assert self._registered_layers, (
                        "KV caches must be registered before submitting save intents"
                    )
                    target_layers = tuple(self._registered_layers)

                if not block_ids:
                    continue

                for layer_name in target_layers:
                    if layer_name not in saves_by_layer:
                        saves_by_layer[layer_name] = ([], [])

                    saves_by_layer[layer_name][0].extend(block_ids)
                    saves_by_layer[layer_name][1].extend(block_hashes)

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
                    self._ctx.pp_rank,
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

    def _use_page_first(self) -> bool:
        """Whether this instance stores blocks page-first.

        Page-first packs each block's layers into contiguous host pages — one
        slot per *shard* (the set of layers one worker holds) instead of one
        slot per layer — cutting per-block metadata by ~num_layers / num_shards.
        Two MLA shapes qualify:

        * full-replica (every rank holds all layers): one shard, so a block's
          whole page is a single slot and ranks block-stripe the writes.
        * layer-split (each rank holds a disjoint subset): one shard per rank,
          and each rank writes its own sub-page as a slot.

        Pipeline parallelism is excluded: with pp_size > 1 each stage holds only
        some layers, but the engine seals one topology spanning the union of all
        stages, so the layers a single worker holds are not one of the sealed
        shards. The first save would seal a partial page and the other stages'
        same-hash saves dedup instead of repairing it. Page-first requires that
        one worker can write each shard's entire page. DCP is excluded because a
        DCP rank stores a different token slice of every layer, not a layer
        partition.
        """
        return self._ctx.is_mla and self._ctx.dcp_world_size == 1 and self._ctx.pp_size == 1

    def _block_shard(self, save_intent) -> tuple[list[int], list[bytes]]:
        """`(block_ids, hashes)` this rank saves under page-first: a block stripe.

        A page needs all layers, so page-first distributes save work by block
        rather than by layer: rank r saves physical blocks where
        `block_id % tp_size == r`, writing all their layers. The stripes are
        disjoint and complete, so every block's page is written exactly once.
        With tp_size == 1 this is the whole set.
        """
        tp_size = self._ctx.tp_size
        if tp_size <= 1:
            return list(save_intent.block_ids), list(save_intent.block_hashes)
        tp_rank = self._ctx.tp_rank or 0
        ids: list[int] = []
        hashes: list[bytes] = []
        for block_id, block_hash in zip(
            save_intent.block_ids, save_intent.block_hashes, strict=True
        ):
            if block_id % tp_size == tp_rank:
                ids.append(block_id)
                hashes.append(block_hash)
        return ids, hashes

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
            logger.debug(
                "[PegaKVConnector] preemption: waiting for %d requests' saves: %s",
                len(events_to_wait),
                [req_id for req_id, _ in events_to_wait],
            )
            for req_id, event in events_to_wait:
                event.wait()
                logger.debug("[PegaKVConnector] preemption: req=%s save completed", req_id)
        else:
            logger.debug(
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
