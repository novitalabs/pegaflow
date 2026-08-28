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
    CacheGroupLayout,
    ConnectorContext,
    LocalRecurrentLoad,
    LocalRecurrentSave,
    LocalSaveRef,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    PegaWorkerMetadata,
    SaveIntent,
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


@dataclass
class _LocalRecurrentLayer:
    group_index: int
    pages: torch.Tensor
    pool: torch.Tensor
    generations: list[int]


def _raw_page_view(tensor: torch.Tensor, page_size_bytes: int) -> torch.Tensor:
    """Expose a block-first tensor allocation as ``[num_blocks, page_bytes]``."""
    registration_tensor = _registration_tensor(tensor)
    actual_page_bytes = registration_tensor.stride(0) * registration_tensor.element_size()
    if actual_page_bytes != page_size_bytes:
        raise RuntimeError(
            "recurrent-state page stride does not match MambaSpec.page_size_bytes: "
            f"actual={actual_page_bytes} declared={page_size_bytes}"
        )
    pages = torch.empty(0, dtype=torch.uint8, device=registration_tensor.device)
    return pages.set_(
        registration_tensor.untyped_storage(),
        registration_tensor.storage_offset() * registration_tensor.element_size(),
        (registration_tensor.shape[0], page_size_bytes),
        (page_size_bytes, 1),
    )


def _infer_kv_cache_registration(
    kv_cache: torch.Tensor,
    logical_block_size: int,
    *,
    is_mla: bool = False,
    is_recurrent_state: bool = False,
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

    if is_recurrent_state or not is_mla:
        if not is_recurrent_state and len(shape) >= 2 and shape[0] == 2:
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


def _registration_tensor(kv_cache) -> torch.Tensor:
    if not isinstance(kv_cache, (tuple, list)):
        return kv_cache

    states = tuple(kv_cache)
    if not states or not all(isinstance(state, torch.Tensor) for state in states):
        raise TypeError("KV cache must be a tensor or a non-empty sequence of state tensors")

    first = states[0]
    if first.storage_offset() != 0:
        raise RuntimeError("the first recurrent-state tensor must start at storage offset zero")
    storage_ptr = first.untyped_storage().data_ptr()
    num_blocks = first.shape[0]
    page_bytes = first.stride(0) * first.element_size()
    for state in states:
        if state.untyped_storage().data_ptr() != storage_ptr:
            raise RuntimeError("recurrent-state tensors must share one CUDA allocation")
        if state.shape[0] != num_blocks:
            raise RuntimeError("recurrent-state tensors must have the same block count")
        if state.stride(0) * state.element_size() != page_bytes:
            raise RuntimeError("recurrent-state tensors must have one common page stride")

    return first


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
        self._cache_groups = CacheGroupLayout.from_config(kv_cache_config)
        self._layer_to_group = self._cache_groups.layer_to_group()
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        self._use_mla_layer_split_registration = context.is_mla and bool(
            additional_config.get("mla_layer_split_kv_cache", False)
        )

        self._save_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()

        self._req_pending_save_tasks: dict[str, int] = {}
        self._completed_saves: set[str] = set()
        self._save_completion_lock = threading.Lock()
        self._save_completion_events: dict[str, threading.Event] = {}
        self._worker_meta = PegaWorkerMetadata()
        self._worker_meta_lock = threading.Lock()
        self._current_metadata: PegaConnectorMetadata | None = None

        self._pending_loads: dict[str, PyLoadState] = {}
        self._pending_load_reqs: dict[str, set[str]] = {}
        self._pending_load_meta: dict[
            str, tuple[float, int, list[int]]
        ] = {}  # shm_name -> (start_time, num_blocks, block_ids)
        self._pending_local_load_events: dict[str, torch.cuda.Event] = {}
        self._remote_load_completed: set[str] = set()
        self._load_completion_lock = threading.Lock()

        # Failure surface for vLLM's get_block_ids_with_load_errors / get_finished.
        # Populated when start_load_kv fails synchronously or when an in-flight
        # load times out waiting for the server. Drained once per get_finished
        # and get_block_ids_with_load_errors call.
        self._failed_load_block_ids: set[int] = set()
        self._failed_load_reqs: set[str] = set()

        self._registered_layers: list[str] = []
        self._local_recurrent_layers: dict[str, _LocalRecurrentLayer] = {}
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
        self._save_queue.put(None)
        self._save_thread.join()
        if self._pending_local_load_events:
            torch.cuda.synchronize(self._torch_device)
            self._pending_local_load_events.clear()
        with self._worker_meta_lock:
            self._worker_meta = PegaWorkerMetadata()
        self.unregister_context()
        self._local_recurrent_layers.clear()

    def unregister_context(self) -> None:
        if not self._registered_layers:
            return

        if self._ctx.local_physical_tp_rank == 0:
            ok, message = self._ctx.engine_client.unregister_context(self._ctx.instance_id)
            if not ok:
                logger.warning("[PegaKVConnector] Unregister context failed: %s", message)

        self._registered_layers.clear()

    def register_kv_caches(self, kv_caches: dict[str, Any]):
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

        self._page_first = self._use_page_first()
        first_tensor = _registration_tensor(next(iter(kv_caches.values())))
        self._torch_device = first_tensor.device

        if self._cache_groups.group_count > 1:
            unmapped_layers = [name for name in kv_caches if name not in self._layer_to_group]
            if unmapped_layers:
                raise RuntimeError(
                    f"HMA registration contains layers outside cache groups: {unmapped_layers[:8]}"
                )

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
        local_mode = bool(self._ctx.num_linear_state_cache_slots)
        local_slots = self._ctx.num_linear_state_cache_slots
        groups = tuple(getattr(self._kv_cache_config, "kv_cache_groups", ()) or ())

        for layer_name, kv_cache in kv_caches.items():
            is_recurrent_state = (
                layer_name in self._cache_groups.recurrent_layer_names
                or isinstance(kv_cache, (tuple, list))
            )
            registration_tensor = _registration_tensor(kv_cache)
            assert registration_tensor.storage_offset() == 0, (
                f"KV cache for {layer_name} must have zero storage offset"
            )
            if local_mode and is_recurrent_state:
                group_index = self._layer_to_group[layer_name]
                page_size_bytes = int(groups[group_index].kv_cache_spec.page_size_bytes)
                pages = _raw_page_view(kv_cache, page_size_bytes)
                try:
                    pool = torch.empty(
                        (local_slots, page_size_bytes),
                        dtype=torch.uint8,
                        device=registration_tensor.device,
                    )
                except torch.OutOfMemoryError as exc:
                    self._local_recurrent_layers.clear()
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "failed to allocate connector-owned linear-state GPU pool; "
                        "pegaflow.num_linear_state_cache_slots consumes additional "
                        "memory outside vLLM KV-cache planning"
                    ) from exc
                self._local_recurrent_layers[layer_name] = _LocalRecurrentLayer(
                    group_index=group_index,
                    pages=pages,
                    pool=pool,
                    generations=[0] * local_slots,
                )
                continue

            wrapper = CudaIPCWrapper(registration_tensor)
            wrapper_bytes = pickle.dumps(wrapper)

            registration = _infer_kv_cache_registration(
                registration_tensor,
                self._ctx.block_size,
                is_mla=self._ctx.is_mla,
                is_recurrent_state=is_recurrent_state,
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

        layer_group_ids = [
            self._cache_groups.storage_group_ids[self._layer_to_group.get(name, 0)]
            for name in layer_names
        ]

        ok, message = self._ctx.engine_client.register_context_batch(
            self._ctx.instance_id,
            self._ctx.namespace,
            self._ctx.effective_tp_rank,
            self._ctx.pp_rank,
            self._ctx.effective_tp_size,
            self._ctx.effective_world_size,
            self._ctx.device_id,
            layer_names,
            ipc_wrappers,
            layer_num_blocks,
            layer_bytes_per_block,
            layer_kv_stride_bytes,
            layer_segments,
            self._ctx.transfer_backend,
            self._page_first,
            layer_group_ids=layer_group_ids,
        )

        if not ok:
            if "PegaFlow version mismatch" in message:
                raise RuntimeError(f"Register context failed: {message}")
            raise RuntimeError(f"Register context batch failed for layers {layer_names}: {message}")

        self._registered_layers = layer_names

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
            self._finished_requests.update(
                finished_req_ids.intersection(self._req_pending_save_tasks)
            )
            done_saves = self._completed_saves & self._finished_requests
            done_saves.update(self._completed_saves & finished_req_ids)

            if done_saves:
                self._completed_saves -= done_saves
                self._finished_requests -= done_saves
                finished_sending = done_saves

        timeout_triggered = False
        hma_load_failure: str | None = None
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
                        if self._cache_groups.group_count > 1:
                            hma_load_failure = (
                                f"async load failed for requests {sorted(req_ids)}: state={state}"
                            )
                        elif meta is not None:
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
                    if self._cache_groups.group_count > 1:
                        hma_load_failure = (
                            f"load timed out for requests {sorted(req_ids)} after {duration:.1f}s"
                        )
                    else:
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

            if self._failed_load_reqs:
                completed_reqs.update(self._failed_load_reqs)
                self._failed_load_reqs = set()

            self._remote_load_completed.update(completed_reqs)
            jointly_completed = {
                req_id
                for req_id in self._remote_load_completed
                if req_id not in self._pending_local_load_events
                or self._pending_local_load_events[req_id].query()
            }
            for req_id in jointly_completed:
                self._pending_local_load_events.pop(req_id, None)
            self._remote_load_completed -= jointly_completed
            if jointly_completed:
                finished_recving = jointly_completed

        if timeout_triggered:
            self._ctx.state_manager.mark_unavailable("load timeout")

        if load_stats_to_record:
            with self._stats_lock:
                for duration, num_blocks, success in load_stats_to_record:
                    self._stats.record_load(duration, num_blocks, success)

        if hma_load_failure is not None:
            self._ctx.state_manager.mark_unavailable(hma_load_failure)
            raise RuntimeError(
                f"PegaFlow HMA load failed; vLLM 0.26 cannot recover failed "
                f"loads for multiple cache groups: {hma_load_failure}"
            )

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

        if metadata.ready_save_intents:
            if not self._cache_groups.has_recurrent_state:
                raise RuntimeError("ready save intents are only valid for HMA")
            self._process_save_batch([self._make_save_task(metadata.ready_save_intents)])

        if not metadata.load_intents:
            return

        total_requests = len(metadata.load_intents)
        load_start = time.perf_counter()

        all_block_ids: list[int] = []
        loads: list[tuple[bytes, list[list[int | None]]]] = []
        request_ids: list[str] = []

        for req_id, load_intent in metadata.load_intents.items():
            if load_intent.local_recurrent is not None:
                self._start_local_recurrent_load(req_id, load_intent.local_recurrent)
            if len(load_intent.leases) != self._ctx.tp_shard_count:
                raise RuntimeError(
                    f"load intent has {len(load_intent.leases)} TP shard leases; "
                    f"expected {self._ctx.tp_shard_count}"
                )
            block_ids_by_group = [list(group) for group in load_intent.block_ids_by_group]
            hold = load_intent.recurrent_hold
            recurrent_groups = sorted(self._cache_groups.recurrent_group_indices)
            if hold is not None:
                # The attention lease pins group-0 (dense prefix) blocks only;
                # recurrent destinations travel with the membership leases.
                for group_index in recurrent_groups:
                    block_ids_by_group[group_index] = [None] * len(block_ids_by_group[group_index])
            for block_ids in block_ids_by_group:
                all_block_ids.extend(block_id for block_id in block_ids if block_id is not None)
            loads.append((load_intent.leases[self._ctx.tp_shard_index], block_ids_by_group))
            if hold is not None:
                shard = self._ctx.tp_shard_index
                for slot, group_index in enumerate(recurrent_groups):
                    positions = hold.hit_positions[slot][shard]
                    if hold.checkpoint not in positions:
                        raise RuntimeError(
                            f"req {req_id}: recurrent group {group_index} shard {shard} "
                            f"lease has no checkpoint at query position {hold.checkpoint}"
                        )
                    destination = next(
                        (
                            block_id
                            for block_id in reversed(load_intent.block_ids_by_group[group_index])
                            if block_id is not None
                        ),
                        None,
                    )
                    if destination is None:
                        raise RuntimeError(
                            f"req {req_id}: no recurrent destination block in group "
                            f"{group_index} for checkpoint {hold.checkpoint}"
                        )
                    # The membership lease pins `[hit_positions]` blocks; only
                    # the chosen checkpoint has a physical destination.
                    vectors: list[list[int | None]] = [
                        [None] * len(positions) for _ in range(self._cache_groups.group_count)
                    ]
                    vectors[group_index][positions.index(hold.checkpoint)] = destination
                    loads.append((hold.leases[slot][shard], vectors))
                    all_block_ids.append(destination)
            request_ids.append(req_id)

        if not all_block_ids:
            return

        if self._cross_layer_mode:
            layer_groups = [[self._cross_layer_key]]
        else:
            assert self._registered_layers, (
                "KV caches must be registered before submitting load intents"
            )
            layer_groups = [[] for _ in range(self._cache_groups.group_count)]
            for layer_name in self._registered_layers:
                group_index = self._layer_to_group.get(layer_name, 0)
                layer_groups[group_index].append(layer_name)

        if not any(layer_groups):
            return

        load_state = PyLoadState()
        shm_name = load_state.shm_name()

        try:
            ok, message = self._ctx.engine_client.load(
                self._ctx.instance_id,
                self._ctx.effective_tp_rank,
                self._ctx.device_id,
                shm_name,
                layer_groups,
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
            self._ctx.state_manager.mark_unavailable(f"load rpc exception: {e}")
            if self._cache_groups.group_count > 1:
                raise RuntimeError(
                    "PegaFlow HMA load failed; vLLM 0.26 cannot recover failed "
                    "loads for multiple cache groups"
                ) from e
            self._record_load_failure(request_ids, all_block_ids, load_start)
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
            self._ctx.state_manager.mark_unavailable(f"load rpc failed: {message}")
            if self._cache_groups.group_count > 1:
                raise RuntimeError(
                    "PegaFlow HMA load failed; vLLM 0.26 cannot recover failed "
                    f"loads for multiple cache groups: {message}"
                )
            self._record_load_failure(request_ids, all_block_ids, load_start)
            return

        num_layers = sum(len(group) for group in layer_groups)
        num_blocks = len(all_block_ids)

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6

        with self._load_completion_lock:
            for req_id in request_ids:
                self._pending_loads[req_id] = load_state
            self._pending_load_reqs[shm_name] = set(request_ids)
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

    def _start_local_recurrent_load(self, req_id: str, intent: LocalRecurrentLoad) -> None:
        if not self._local_recurrent_layers:
            raise RuntimeError("local recurrent load requested before GPU pool registration")
        destinations = dict(intent.destination_block_ids)
        for layer_name, layer in self._local_recurrent_layers.items():
            ref = intent.source
            if ref.slot < 0 or ref.slot >= layer.pool.shape[0]:
                raise RuntimeError(f"req {req_id}: local recurrent slot {ref.slot} is out of range")
            if layer.generations[ref.slot] != ref.generation:
                raise RuntimeError(
                    f"req {req_id}: stale local recurrent slot {ref.slot}/{ref.generation} "
                    f"for layer {layer_name}; worker has generation {layer.generations[ref.slot]}"
                )
            destination = destinations.get(layer.group_index)
            if destination is None or destination < 0 or destination >= layer.pages.shape[0]:
                raise RuntimeError(
                    f"req {req_id}: recurrent destination {destination} is out of range "
                    f"for layer {layer_name}"
                )
            layer.pages[destination].copy_(layer.pool[ref.slot], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending_local_load_events[req_id] = event

    def _save_local_recurrent(self, save: LocalRecurrentSave) -> None:
        if not self._local_recurrent_layers:
            raise RuntimeError("local recurrent save requested before GPU pool registration")
        sources = dict(save.source_block_ids)
        for layer_name, layer in self._local_recurrent_layers.items():
            ref = save.target
            if ref.slot < 0 or ref.slot >= layer.pool.shape[0]:
                raise RuntimeError(f"local recurrent slot {ref.slot} is out of range")
            if ref.generation <= layer.generations[ref.slot]:
                raise RuntimeError(
                    f"stale local recurrent save {ref.slot}/{ref.generation} for layer "
                    f"{layer_name}; worker has generation {layer.generations[ref.slot]}"
                )
            source = sources.get(layer.group_index)
            if source is None or source < 0 or source >= layer.pages.shape[0]:
                raise RuntimeError(
                    f"recurrent source {source} is out of range for layer {layer_name}"
                )
            layer.pool[ref.slot].copy_(layer.pages[source], non_blocking=True)
            layer.generations[ref.slot] = ref.generation

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def _release_load_leases(self, loads: list[tuple[bytes, list[list[int | None]]]]) -> None:
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

        task = self._make_save_task(metadata.save_intents)
        if self._cache_groups.has_recurrent_state:
            # Align-mode recurrent states can reuse their only live block on
            # the next scheduler step. Finish D2H before returning so the
            # saved boundary state cannot be overwritten underneath the copy.
            self._process_save_batch([task])
        else:
            self._save_queue.put(task)

    def _make_save_task(self, save_intents: dict[str, SaveIntent]) -> SaveTask:
        request_ids = list(save_intents)

        with self._save_completion_lock:
            for req_id in request_ids:
                pending_tasks = self._req_pending_save_tasks.get(req_id, 0)
                if pending_tasks == 0:
                    self._completed_saves.discard(req_id)
                    self._save_completion_events[req_id] = threading.Event()
                self._req_pending_save_tasks[req_id] = pending_tasks + 1

        return SaveTask(
            metadata=PegaConnectorMetadata(save_intents=save_intents),
            request_ids=request_ids,
        )

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
        request_ids = [req_id for task in batch for req_id in task.request_ids]
        local_refs = [
            LocalSaveRef.from_save(req_id, save)
            for task in batch
            for req_id, intent in task.metadata.save_intents.items()
            for save in intent.local_recurrent_saves
        ]
        try:
            self._process_save_batch_inner(batch)
        except Exception:
            logger.exception("[PegaKVConnector] local/remote save batch failed")
            self._record_local_save_results(failed=local_refs)
        finally:
            self._complete_save_requests(request_ids)

    def _process_save_batch_inner(self, batch: list[SaveTask]) -> None:
        saves_by_layer: dict[str, tuple[list[int], list[bytes]]] = {}
        local_saves: list[tuple[LocalSaveRef, LocalRecurrentSave]] = []

        for task in batch:
            for req_id, save_intent in task.metadata.save_intents.items():
                local_saves.extend(
                    (LocalSaveRef.from_save(req_id, save), save)
                    for save in save_intent.local_recurrent_saves
                )
                if not any(save_intent.block_ids_by_group):
                    continue

                if self._cross_layer_mode:
                    target_layers = (self._cross_layer_key,)
                else:
                    assert self._registered_layers, (
                        "KV caches must be registered before submitting save intents"
                    )
                    target_layers = tuple(self._registered_layers)

                for layer_name in target_layers:
                    group_index = self._layer_to_group.get(layer_name, 0)
                    try:
                        block_ids = save_intent.block_ids_by_group[group_index]
                    except IndexError as exc:
                        raise RuntimeError(
                            f"save intent is missing cache group {group_index} for {layer_name}"
                        ) from exc
                    block_hashes = save_intent.block_hashes
                    if len(block_ids) != len(block_hashes):
                        raise RuntimeError(
                            f"save block/hash count mismatch for {layer_name}: "
                            f"blocks={len(block_ids)} hashes={len(block_hashes)}"
                        )
                    non_null = tuple(
                        (block_id, block_hash)
                        for block_id, block_hash in zip(block_ids, block_hashes, strict=True)
                        if block_id != 0
                    )
                    block_ids = tuple(block_id for block_id, _ in non_null)
                    block_hashes = tuple(block_hash for _, block_hash in non_null)
                    if self._page_first and not self._use_mla_layer_split_registration:
                        block_ids, block_hashes = self._block_shard(block_ids, block_hashes)
                    if not block_ids:
                        continue
                    if layer_name not in saves_by_layer:
                        saves_by_layer[layer_name] = ([], [])
                    saves_by_layer[layer_name][0].extend(block_ids)
                    saves_by_layer[layer_name][1].extend(block_hashes)

        if saves_by_layer or local_saves:
            torch.cuda.synchronize(self._torch_device)
        for _ref, save in local_saves:
            self._save_local_recurrent(save)
        if local_saves:
            torch.cuda.synchronize(self._torch_device)
            self._record_local_save_results(succeeded=[ref for ref, _save in local_saves])

        if saves_by_layer:
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
            with self._stats_lock:
                self._stats.record_save(save_duration, total_blocks, success)

    def _record_local_save_results(
        self,
        *,
        succeeded: Iterable[LocalSaveRef] = (),
        failed: Iterable[LocalSaveRef] = (),
    ) -> None:
        with self._worker_meta_lock:
            for ref in succeeded:
                self._worker_meta.succeeded[ref] = 1
            for ref in failed:
                self._worker_meta.failed[ref] = 1

    def build_connector_worker_meta(self) -> PegaWorkerMetadata | None:
        with self._worker_meta_lock:
            if not self._worker_meta.succeeded and not self._worker_meta.failed:
                return None
            metadata = self._worker_meta
            self._worker_meta = PegaWorkerMetadata()
            return metadata

    def _complete_save_requests(self, request_ids: list[str]) -> None:
        completed_reqs: list[str] = []

        with self._save_completion_lock:
            for req_id in request_ids:
                pending_tasks = self._req_pending_save_tasks.get(req_id)
                if pending_tasks is None:
                    continue
                if pending_tasks > 1:
                    self._req_pending_save_tasks[req_id] = pending_tasks - 1
                    continue

                del self._req_pending_save_tasks[req_id]
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

        Page-first requires one worker to write every layer in its page shard.
        PP workers hold only part of a sealed shard, DCP workers hold different
        token slices, and HMA groups can save different block sets, so none can
        safely use one common block set across every layer in a page.
        """
        return (
            self._ctx.is_mla
            and self._cache_groups.group_count == 1
            and self._ctx.dcp_world_size == 1
            and self._ctx.pp_size == 1
        )

    def _block_shard(
        self,
        block_ids: Iterable[int],
        block_hashes: Iterable[bytes],
    ) -> tuple[list[int], list[bytes]]:
        """`(block_ids, hashes)` this rank saves under page-first: a block stripe.

        A page needs all layers, so page-first distributes save work by block
        rather than by layer: rank r saves physical blocks where
        `block_id % tp_size == r`, writing all their layers. The stripes are
        disjoint and complete, so every block's page is written exactly once.
        With tp_size == 1 this is the whole set.
        """
        tp_size = self._ctx.local_physical_tp_size
        if tp_size <= 1:
            return list(block_ids), list(block_hashes)
        tp_rank = self._ctx.local_physical_tp_rank
        ids: list[int] = []
        hashes: list[bytes] = []
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
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
            with self._save_completion_lock:
                self._stats.data["pending_save_requests"] = len(self._req_pending_save_tasks)

            if self._stats.is_empty():
                return None
            return self._stats.clone_and_reset()


__all__ = ["WorkerConnector"]
