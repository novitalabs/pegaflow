"""
Worker-side connector logic.
"""

import pickle
import queue
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from pegaflow import (
    KvCacheLayer,
    KvCacheRegistration,
    LayerSave,
    LoadHandle,
    LoadItem,
    LoadRequest,
    LoadSourceKind,
    SaveRequest,
)
from pegaflow.connector.common import (
    ConnectorContext,
    KvEgressIntent,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    logger,
    parse_env_int,
)
from pegaflow.connector.egress import (
    EgressLayerRegistration,
    KvEgressManager,
)
from pegaflow.ipc_wrapper import CudaIPCWrapper

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata


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


@dataclass(frozen=True, slots=True)
class _SaveTask:
    metadata: PegaConnectorMetadata
    request_ids: list[str]


@dataclass(frozen=True, slots=True)
class _RegisteredLayer:
    cache_layer: KvCacheLayer
    egress_layer: EgressLayerRegistration
    layout: str


@dataclass(frozen=True, slots=True)
class _LoadBatch:
    request_ids: tuple[str, ...]
    items: tuple[LoadItem, ...]
    block_ids: tuple[int, ...]
    target_layers: tuple[str, ...]
    receive_rank: int | None

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[str, LoadItem]],
        target_layers: Sequence[str],
        receive_rank: int | None,
    ) -> "_LoadBatch":
        return cls(
            request_ids=tuple(req_id for req_id, _ in items),
            items=tuple(item for _, item in items),
            block_ids=tuple(int(block_id) for _, item in items for block_id in item.block_ids),
            target_layers=tuple(target_layers),
            receive_rank=receive_rank,
        )

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


@dataclass(slots=True)
class _PendingLoad:
    handle: LoadHandle
    request_ids: set[str]
    block_ids: tuple[int, ...]
    started_at: float

    @property
    def shm_name(self) -> str:
        return self.handle.shm_name

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def elapsed(self, now: float) -> float:
        return now - self.started_at

    def timed_out(self, now: float, timeout_seconds: int) -> bool:
        return not self.handle.done and self.elapsed(now) > timeout_seconds


@dataclass(frozen=True, slots=True)
class _TransferStat:
    duration: float
    num_blocks: int
    success: bool


@dataclass(frozen=True, slots=True)
class _LoadPollResult:
    request_ids: set[str] = field(default_factory=set)
    stats: list[_TransferStat] = field(default_factory=list)
    timed_out: bool = False


@dataclass(slots=True)
class _LayerSaveAccumulator:
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[bytes] = field(default_factory=list)

    def extend(self, block_ids: Iterable[int], block_hashes: Iterable[bytes]) -> None:
        self.block_ids.extend(block_ids)
        self.block_hashes.extend(block_hashes)

    def to_layer_save(self, layer_name: str) -> LayerSave:
        return LayerSave(
            layer_name=layer_name,
            block_ids=tuple(self.block_ids),
            block_hashes=tuple(self.block_hashes),
        )


@dataclass(frozen=True, slots=True)
class _SaveWork:
    request_ids: list[str]
    layer_saves: list[LayerSave]
    egress_items: list[tuple[str, KvEgressIntent]]

    @property
    def needs_cuda_sync(self) -> bool:
        return bool(self.layer_saves or self.egress_items)


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

        self._save_queue: queue.Queue[_SaveTask | None] = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()

        self._req_pending_saves: set[str] = set()
        self._completed_saves: set[str] = set()
        self._save_completion_lock = threading.Lock()
        self._save_completion_events: dict[str, threading.Event] = {}
        self._current_metadata: PegaConnectorMetadata | None = None

        self._pending_loads_by_shm: dict[str, _PendingLoad] = {}
        self._load_completion_lock = threading.Lock()

        # Failure surface for vLLM's get_block_ids_with_load_errors / get_finished.
        # Populated when start_load_kv fails synchronously or when an in-flight
        # load times out waiting for the server. Drained once per get_finished
        # and get_block_ids_with_load_errors call.
        self._failed_load_block_ids: set[int] = set()
        self._failed_load_reqs: set[str] = set()

        self._registered_layers: list[str] = []
        self._torch_device: torch.device | None = None

        self._cross_layer_mode = False
        self._cross_layer_key = _CROSS_LAYER_KEY
        self._egress = KvEgressManager(
            instance_id=self._ctx.instance_id,
            device_id=self._ctx.device_id,
            receive_rank=self._ctx.effective_tp_rank,
        )

        self._finished_requests: set[str] = set()

        # Stats collection
        self._stats = PegaKVConnectorStats()
        self._stats_lock = threading.Lock()

    def shutdown(self) -> None:
        self.unregister_context()
        self._save_queue.put(None)
        self._save_thread.join()

    def unregister_context(self) -> None:
        if not self._registered_layers and not self._egress.has_layers:
            return

        self._egress.unregister_layers()

        if self._registered_layers and self._ctx.tp_rank == 0:
            try:
                self._ctx.client._unregister_context(self._ctx.instance_id)
            except Exception as e:
                logger.warning("[PegaKVConnector] Unregister context failed: %s", e)

        self._registered_layers.clear()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._ctx.device_id is not None, (
            "CUDA device id is unknown; cannot register KV caches"
        )
        assert kv_caches, "KV cache registration requires at least one layer"

        self._registered_layers = list(kv_caches.keys())
        self._torch_device = next(iter(kv_caches.values())).device

        # Use actual number of registered layers, not model's num_hidden_layers.
        # This is important for models like DSA where indexer layers are separate.
        # With PP>1 each rank registers only its local layers; the Rust engine
        # grows the instance-wide total dynamically via verify_topology.
        registrations = [
            _build_layer_registration(layer_name, kv_cache)
            for layer_name, kv_cache in kv_caches.items()
        ]

        self._ctx.client.register_kv_cache(
            KvCacheRegistration(
                instance_id=self._ctx.instance_id,
                namespace=self._ctx.namespace,
                tp_rank=self._ctx.effective_tp_rank,
                tp_size=self._ctx.effective_tp_size,
                world_size=self._ctx.world_size,
                device_id=self._ctx.device_id,
                num_layers=len(registrations),
                layers=tuple(registration.cache_layer for registration in registrations),
            )
        )

        self._egress.register_layers(
            {
                registration.cache_layer.name: registration.egress_layer
                for registration in registrations
            }
        )

        logger.debug(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout) instance=%s",
            len(kv_caches),
            _summarize_layout(registration.layout for registration in registrations),
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
        finished_sending = self._collect_finished_saves(finished_req_ids)
        load_result = self._collect_finished_loads()
        finished_recving = load_result.request_ids or None

        if load_result.timed_out:
            self._ctx.state_manager.mark_unavailable("load timeout")

        if load_result.stats:
            with self._stats_lock:
                for stat in load_result.stats:
                    self._stats.record_load(stat.duration, stat.num_blocks, stat.success)

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

    def _collect_finished_saves(self, finished_req_ids: set[str]) -> set[str] | None:
        with self._save_completion_lock:
            self._finished_requests.update(finished_req_ids & self._req_pending_saves)
            done_saves = (self._completed_saves & self._finished_requests) | (
                self._completed_saves & finished_req_ids
            )
            if not done_saves:
                return None

            self._completed_saves -= done_saves
            self._finished_requests -= done_saves
            return done_saves

    def _collect_finished_loads(self) -> _LoadPollResult:
        with self._load_completion_lock:
            completed_reqs: set[str] = set()
            completed_shms: list[str] = []
            stats: list[_TransferStat] = []
            timeout_triggered = False
            now = time.perf_counter()

            for shm_name, pending in self._pending_loads_by_shm.items():
                if pending.handle.done:
                    success = pending.handle.state >= 0
                    self._record_async_load_done(pending, now, stats, success)
                    completed_reqs.update(pending.request_ids)
                    completed_shms.append(shm_name)
                    continue

                if pending.timed_out(now, self.LOAD_TIMEOUT_SECONDS):
                    self._record_async_load_timeout(pending, now, stats)
                    completed_reqs.update(pending.request_ids)
                    completed_shms.append(shm_name)
                    timeout_triggered = True

            for shm_name in completed_shms:
                self._drop_pending_load(shm_name)

            # Drain sync-failure reqs recorded by start_load_kv so they also
            # reach vLLM as finished_recving in this pass.
            if self._failed_load_reqs:
                completed_reqs.update(self._failed_load_reqs)
                self._failed_load_reqs.clear()

        return _LoadPollResult(
            request_ids=completed_reqs,
            stats=stats,
            timed_out=timeout_triggered,
        )

    def _record_async_load_done(
        self,
        pending: _PendingLoad,
        now: float,
        stats: list[_TransferStat],
        success: bool,
    ) -> None:
        if success:
            logger.debug(
                "[PegaKVConnector] async_load_completed: reqs=%s",
                pending.request_ids,
            )
        else:
            logger.error(
                "[PegaKVConnector] async_load_failed: reqs=%s state=%d",
                pending.request_ids,
                pending.handle.state,
            )
            self._failed_load_block_ids.update(pending.block_ids)

        stats.append(
            _TransferStat(
                duration=pending.elapsed(now),
                num_blocks=pending.num_blocks,
                success=success,
            )
        )

    def _record_async_load_timeout(
        self,
        pending: _PendingLoad,
        now: float,
        stats: list[_TransferStat],
    ) -> None:
        duration = pending.elapsed(now)
        logger.error(
            "[PegaKVConnector] load_timeout: reqs=%s shm=%s elapsed=%.1fs "
            "blocks=%d (reporting as load errors)",
            pending.request_ids,
            pending.shm_name,
            duration,
            pending.num_blocks,
        )
        self._failed_load_block_ids.update(pending.block_ids)
        stats.append(
            _TransferStat(
                duration=duration,
                num_blocks=pending.num_blocks,
                success=False,
            )
        )

    def start_load_kv(
        self,
        metadata: PegaConnectorMetadata,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        self._current_metadata = metadata

        if not metadata.load_intents:
            return

        target_layers = self._load_target_layers(forward_context)
        if not target_layers:
            return

        normal_items: list[tuple[str, LoadItem]] = []
        staged_items: list[tuple[str, LoadItem]] = []

        for req_id, load_intent in metadata.load_intents.items():
            if not load_intent.block_ids:
                continue
            item = LoadItem(
                plan=load_intent.plan,
                block_ids=tuple(load_intent.block_ids),
            )
            if load_intent.plan.source is LoadSourceKind.STAGED:
                staged_items.append((req_id, item))
            else:
                normal_items.append((req_id, item))

        if normal_items:
            self._submit_load(
                _LoadBatch.from_items(
                    normal_items,
                    target_layers,
                    receive_rank=None,
                )
            )

        if staged_items:
            self._submit_load(
                _LoadBatch.from_items(
                    staged_items,
                    target_layers,
                    receive_rank=self._ctx.effective_tp_rank,
                )
            )

    def _load_target_layers(self, forward_context: "ForwardContext") -> tuple[str, ...]:
        if self._cross_layer_mode:
            return (self._cross_layer_key,)

        return tuple(
            layer_name
            for layer_name, layer in forward_context.no_compile_layers.items()
            if hasattr(layer, "kv_cache")
        )

    def _submit_load(self, batch: _LoadBatch) -> None:
        load_start = time.perf_counter()
        device_id = self._require_device_id()

        try:
            load_state = self._ctx.client.load(
                LoadRequest(
                    instance_id=self._ctx.instance_id,
                    tp_rank=self._ctx.effective_tp_rank,
                    device_id=device_id,
                    layer_names=batch.target_layers,
                    items=batch.items,
                    receive_rank=batch.receive_rank,
                )
            )
        except Exception as e:
            logger.error(
                "[PegaKVConnector] Load RPC exception: %s (reqs=%s blocks=%d, "
                "marking blocks as load errors)",
                e,
                list(batch.request_ids),
                batch.num_blocks,
            )
            self._record_load_failure(batch.request_ids, batch.block_ids, load_start)
            self._ctx.state_manager.mark_unavailable(f"load rpc exception: {e}")
            return

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6
        pending = _PendingLoad(
            handle=load_state,
            request_ids=set(batch.request_ids),
            block_ids=batch.block_ids,
            started_at=load_start,
        )

        with self._load_completion_lock:
            self._track_pending_load(pending)

        logger.debug(
            "[PegaKVConnector] started async load: %d blocks across %d layers for %d reqs, "
            "schedule %.0f us, shm=%s",
            batch.num_blocks,
            len(batch.target_layers),
            len(batch.request_ids),
            schedule_time_us,
            pending.shm_name,
        )

    def _track_pending_load(self, pending: _PendingLoad) -> None:
        self._pending_loads_by_shm[pending.shm_name] = pending

    def _drop_pending_load(self, shm_name: str) -> None:
        self._pending_loads_by_shm.pop(shm_name, None)

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
        request_ids: Iterable[str],
        block_ids: Sequence[int],
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
        if metadata is None or (not metadata.save_intents and not metadata.egress_intents):
            return

        request_ids = list(
            dict.fromkeys([*metadata.egress_intents.keys(), *metadata.save_intents.keys()])
        )

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

        self._save_queue.put(_SaveTask(metadata=metadata, request_ids=request_ids))

    def _save_worker(self) -> None:
        logger.debug("[PegaKVConnector] Save worker thread started")

        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break

            batch, stop_after_batch = self._drain_save_batch(task)
            self._process_save_batch(batch)
            for _ in batch:
                self._save_queue.task_done()
            if stop_after_batch:
                break

        logger.debug("[PegaKVConnector] Save worker thread stopped")

    def _drain_save_batch(self, first_task: _SaveTask) -> tuple[list[_SaveTask], bool]:
        batch = [first_task]
        stop_after_batch = False

        while True:
            try:
                task = self._save_queue.get_nowait()
            except queue.Empty:
                break

            if task is None:
                self._save_queue.task_done()
                stop_after_batch = True
                break
            batch.append(task)

        return batch, stop_after_batch

    def _process_save_batch(self, batch: list[_SaveTask]) -> None:
        work = self._collect_save_work(batch)

        if work.needs_cuda_sync and self._torch_device is not None:
            # Ensure all GPU kernels have completed before reading KV cache.
            # Otherwise we may copy uninitialized memory (attention kernel is async).
            torch.cuda.synchronize(self._torch_device)

        if work.egress_items:
            self._egress.execute_batch(work.egress_items)

        if work.layer_saves:
            self._submit_layer_saves(work.layer_saves)

        # Always complete the request save lifecycle, even if save failed.
        self._complete_save_requests(work.request_ids)

    def _collect_save_work(self, batch: list[_SaveTask]) -> _SaveWork:
        saves_by_layer: dict[str, _LayerSaveAccumulator] = {}
        egress_items: list[tuple[str, KvEgressIntent]] = []
        all_request_ids: list[str] = []

        for task in batch:
            all_request_ids.extend(task.request_ids)
            egress_items.extend(task.metadata.egress_intents.items())

            for save_intent in task.metadata.save_intents.values():
                if not save_intent.block_ids:
                    continue

                for layer_name in self._save_target_layers():
                    accumulator = saves_by_layer.setdefault(
                        layer_name,
                        _LayerSaveAccumulator(),
                    )
                    accumulator.extend(save_intent.block_ids, save_intent.block_hashes)

        return _SaveWork(
            request_ids=all_request_ids,
            layer_saves=[
                accumulator.to_layer_save(layer_name)
                for layer_name, accumulator in saves_by_layer.items()
            ],
            egress_items=egress_items,
        )

    def _save_target_layers(self) -> tuple[str, ...]:
        if self._cross_layer_mode:
            return (self._cross_layer_key,)

        assert self._registered_layers, (
            "KV caches must be registered before submitting save intents"
        )
        return tuple(self._registered_layers)

    def _submit_layer_saves(self, layer_saves: list[LayerSave]) -> None:
        total_blocks = sum(len(layer.block_ids) for layer in layer_saves)
        save_start = time.perf_counter()
        device_id = self._require_device_id()
        success = False

        try:
            self._ctx.client.save(
                SaveRequest(
                    instance_id=self._ctx.instance_id,
                    tp_rank=self._ctx.effective_tp_rank,
                    device_id=device_id,
                    layers=tuple(layer_saves),
                )
            )
            success = True
            logger.debug(
                "[PegaKVConnector] Batch saved %d layers, %d total blocks",
                len(layer_saves),
                total_blocks,
            )
        except Exception as e:
            logger.error(
                "[PegaKVConnector] Save RPC exception: %s (continuing without save)",
                e,
            )

        save_duration = time.perf_counter() - save_start
        with self._stats_lock:
            self._stats.record_save(save_duration, total_blocks, success)

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

    def _require_device_id(self) -> int:
        assert self._ctx.device_id is not None, "CUDA device id is required"
        return self._ctx.device_id

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


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    storage = tensor.untyped_storage()
    return int(storage.nbytes())


def _build_layer_registration(layer_name: str, kv_cache: torch.Tensor) -> _RegisteredLayer:
    assert kv_cache.storage_offset() == 0, (
        f"KV cache for {layer_name} must have zero storage offset"
    )

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

    assert bytes_per_block != 0, f"Invalid bytes_per_block for {layer_name}: stride={stride}"

    return _RegisteredLayer(
        cache_layer=KvCacheLayer(
            name=layer_name,
            wrapper_bytes=pickle.dumps(CudaIPCWrapper(kv_cache)),
            num_blocks=int(num_blocks),
            bytes_per_block=int(bytes_per_block),
            kv_stride_bytes=int(kv_stride_bytes),
            segments=int(segments),
        ),
        egress_layer=EgressLayerRegistration(
            base_ptr=int(kv_cache.data_ptr()),
            size_bytes=_tensor_storage_nbytes(kv_cache),
            num_blocks=int(num_blocks),
            bytes_per_block=int(bytes_per_block),
            kv_stride_bytes=int(kv_stride_bytes),
            segments=int(segments),
        ),
        layout=layout,
    )


def _summarize_layout(layouts: Iterable[str]) -> str:
    unique_layouts = tuple(dict.fromkeys(layouts))
    if len(unique_layouts) == 1:
        return unique_layouts[0]
    return "mixed:" + ",".join(unique_layouts)


__all__ = ["WorkerConnector"]
