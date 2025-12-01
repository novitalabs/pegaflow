from __future__ import annotations

"""PegaFlow KV connector for vLLM v1.

This module defines :class:`PegaKVConnector`, a subclass of
``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``.
"""

import os
import pickle
import queue
import threading
import time
import uuid
import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgpack
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from pegaflow.ipc_wrapper import CudaIPCWrapper
from pegaflow.logging_utils import get_connector_logger, timing_wrapper
from pegaflow.pegaflow import PyLoadState

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    # Type alias for copy operation (defined in vllm internals)
    CopyBlocksOp = Any

logger = get_connector_logger()

# Engine server endpoint (independent process)
_ENGINE_ENDPOINT = os.environ.get("PEGAFLOW_ENGINE_ENDPOINT", "ipc:///tmp/pega_engine.sock")


# ==============================================================================
# Request Tracking: Phase, Intents, and Tracker
# ==============================================================================


class RequestPhase(enum.Enum):
    """
    Represents the lifecycle phase of a request in the KV connector.

    Phase transitions:
        LOOKUP -> LOADING -> ACTIVE -> DRAINING -> DONE
                    |          ^
                    +----------+  (if no external KV needed)
    """
    LOOKUP = "lookup"       # Waiting for lookup result from external storage
    LOADING = "loading"     # Need to load KV from external storage
    ACTIVE = "active"       # Actively generating (may be saving concurrently)
    DRAINING = "draining"   # Generation done, waiting for async save to complete
    DONE = "done"           # Fully completed


@dataclass(frozen=True)
class LoadIntent:
    """
    Immutable intent for a KV load operation.

    Produced by RequestTracker.consume_load_intent() and consumed by worker
    to load KV cache from external storage into GPU memory.
    """
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    num_tokens: int


@dataclass(frozen=True)
class SaveIntent:
    """
    Immutable intent for a KV save operation.

    Produced by RequestTracker.consume_save_intent() and consumed by worker
    to save KV cache from GPU memory to external storage.
    """
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


class RequestTracker:
    """
    Tracks the KV cache state for a single request.

    Design principles:
    - Event-driven: External code notifies state changes via on_xxx() methods
    - Intent pattern: Use consume_xxx_intent() to get operations to perform
    - Single consumption: Load intent can only be consumed once
    - Phase is derived: No manual state machine, phase is computed from internal state

    Usage:
        tracker = RequestTracker(req_id, block_hashes, block_size, num_layers)

        # On lookup result
        tracker.on_lookup(hit_blocks=10, computed_blocks=5)

        # On block allocation
        tracker.on_alloc(block_ids=[1,2,3], num_external_tokens=320)

        # When scheduled
        tracker.on_scheduled(num_tokens=128)

        # Get load intent (consumed once)
        if load := tracker.consume_load_intent():
            # Execute load operation

        # Get save intent (can be called multiple times)
        if save := tracker.consume_save_intent():
            # Execute save operation
    """

    __slots__ = (
        'request_id', '_block_hashes', '_block_size',
        # Lookup state
        '_hit_blocks', '_computed_blocks', '_lookup_done',
        # Allocation state
        '_allocated_blocks', '_external_tokens',
        # Progress tracking
        '_scheduled_tokens', '_stored_blocks',
        # Save tracking
        '_total_layers', '_saved_layers',
        # Flags
        '_load_consumed', '_finished',
    )

    def __init__(
        self,
        request_id: str,
        block_hashes: list[bytes],
        block_size: int,
        num_layers: int,
    ):
        """
        Initialize a new request tracker.

        Args:
            request_id: Unique identifier for the request
            block_hashes: List of block hashes for this request
            block_size: Number of tokens per block
            num_layers: Total number of KV cache layers
        """
        self.request_id = request_id
        self._block_hashes = tuple(block_hashes)
        self._block_size = block_size

        # Lookup state
        self._hit_blocks: int = 0
        self._computed_blocks: int = 0
        self._lookup_done: bool = False

        # Allocation state
        self._allocated_blocks: list[int] = []
        self._external_tokens: int = 0

        # Progress tracking
        self._scheduled_tokens: int = 0
        self._stored_blocks: int = 0

        # Save tracking
        self._total_layers = num_layers
        self._saved_layers: int = 0

        # Flags
        self._load_consumed: bool = False
        self._finished: bool = False

    # ========== Properties ==========

    @property
    def phase(self) -> RequestPhase:
        """
        Current lifecycle phase (read-only, derived from internal state).

        Returns:
            RequestPhase indicating current state
        """
        if not self._lookup_done:
            return RequestPhase.LOOKUP
        if self._needs_load and not self._load_consumed:
            return RequestPhase.LOADING
        if not self._finished:
            return RequestPhase.ACTIVE
        if self._saved_layers < self._total_layers:
            return RequestPhase.DRAINING
        return RequestPhase.DONE

    @property
    def _needs_load(self) -> bool:
        """Check if this request needs to load KV from external storage."""
        return self._external_tokens > 0 and self._hit_blocks > self._computed_blocks

    @property
    def block_hashes(self) -> tuple[bytes, ...]:
        """Get the block hashes for this request."""
        return self._block_hashes

    # ========== Event Methods ==========

    def on_lookup(self, hit_blocks: int, computed_blocks: int) -> None:
        """
        Handle lookup result from external storage.

        Args:
            hit_blocks: Number of blocks available in external storage
            computed_blocks: Number of blocks already computed locally (vLLM hit)

        Notes:
            This method can be called multiple times by vLLM scheduler.
            Each call updates the lookup state.
        """
        old_phase = self.phase
        self._hit_blocks = hit_blocks
        self._computed_blocks = computed_blocks
        
        if not self._lookup_done:
            # First lookup
            self._lookup_done = True
            logger.info(
                "[RequestTracker] %s on_lookup: hit=%d computed=%d phase=%s->%s",
                self.request_id, hit_blocks, computed_blocks, old_phase.value, self.phase.value,
            )

    def on_alloc(self, block_ids: list[int], num_external_tokens: int) -> None:
        """
        Handle block allocation event.

        Args:
            block_ids: List of newly allocated block IDs
            num_external_tokens: Number of tokens to load from external storage
        """
        old_phase = self.phase
        self._allocated_blocks.extend(block_ids)
        if num_external_tokens > 0:
            self._external_tokens = num_external_tokens
        if block_ids:
            logger.info(
                "[RequestTracker] %s on_alloc: +%d blocks (total=%d) external_tokens=%d phase=%s->%s",
                self.request_id, len(block_ids), len(self._allocated_blocks),
                self._external_tokens, old_phase.value, self.phase.value,
            )

    def on_scheduled(self, num_tokens: int) -> None:
        """
        Handle scheduling event.

        Args:
            num_tokens: Number of tokens scheduled for this step
        """
        self._scheduled_tokens += num_tokens
        logger.debug(
            "[RequestTracker] %s on_scheduled: +%d tokens (total=%d)",
            self.request_id, num_tokens, self._scheduled_tokens,
        )

    def on_layer_saved(self) -> None:
        """
        Handle layer save completion event.

        Called when one layer of KV cache has been saved to external storage.
        """
        old_phase = self.phase
        self._saved_layers += 1
        assert self._saved_layers <= self._total_layers, \
            f"saved {self._saved_layers} > total {self._total_layers} for {self.request_id}"
        logger.debug(
            "[RequestTracker] %s on_layer_saved: %d/%d phase=%s->%s",
            self.request_id, self._saved_layers, self._total_layers,
            old_phase.value, self.phase.value,
        )

    def on_finished(self) -> None:
        """
        Handle request generation completion event.

        Called when the request has finished generating all tokens.
        """
        old_phase = self.phase
        self._finished = True
        logger.info(
            "[RequestTracker] %s on_finished: phase=%s->%s",
            self.request_id, old_phase.value, self.phase.value,
        )

    # ========== Intent Methods (Single Consumption) ==========

    def consume_load_intent(self) -> LoadIntent | None:
        """
        Get the load operation intent.

        This method can only return a non-None value once. Subsequent calls
        will return None after the intent has been consumed.

        Returns:
            LoadIntent if load is needed and not yet consumed, None otherwise
        """
        if self._load_consumed or not self._needs_load:
            return None

        num_blocks = min(
            self._hit_blocks,
            len(self._allocated_blocks),
            len(self._block_hashes),
        )
        load_blocks = num_blocks - self._computed_blocks

        if load_blocks <= 0:
            return None

        # Mark as consumed
        old_phase = self.phase
        self._load_consumed = True

        # Load from computed_blocks to hit_blocks
        start = self._computed_blocks
        end = start + load_blocks

        intent = LoadIntent(
            block_ids=tuple(self._allocated_blocks[start:end]),
            block_hashes=self._block_hashes[start:end],
            num_tokens=load_blocks * self._block_size,
        )
        logger.info(
            "[RequestTracker] %s consume_load_intent: %d blocks [%d:%d] phase=%s->%s",
            self.request_id, load_blocks, start, end, old_phase.value, self.phase.value,
        )
        return intent

    def consume_save_intent(self) -> SaveIntent | None:
        """
        Get the save operation intent.

        This method can be called multiple times. Each call returns newly
        available blocks that haven't been saved yet.

        Returns:
            SaveIntent if there are new blocks to save, None otherwise
        """
        # Calculate saveable blocks based on scheduled tokens
        saveable = min(
            len(self._block_hashes),
            len(self._allocated_blocks),
            self._scheduled_tokens // self._block_size,
        )

        new_blocks = saveable - self._stored_blocks
        if new_blocks <= 0:
            return None

        start = self._stored_blocks
        end = start + new_blocks

        # Update stored count
        self._stored_blocks = end

        intent = SaveIntent(
            block_ids=tuple(self._allocated_blocks[start:end]),
            block_hashes=self._block_hashes[start:end],
        )
        logger.info(
            "[RequestTracker] %s consume_save_intent: %d blocks [%d:%d] total_stored=%d",
            self.request_id, new_blocks, start, end, self._stored_blocks,
        )
        return intent

    # ========== Query Methods ==========

    def should_hold_blocks(self) -> bool:
        """
        Check if blocks should be held (not freed) for async save.

        Returns:
            True if request is finished and has blocks to save that aren't complete.
            Uses _stored_blocks > 0 to detect if save intents were consumed (scheduler-side).
        """
        return self._finished and self._stored_blocks > 0 and self._saved_layers < self._total_layers

    def is_done(self) -> bool:
        """
        Check if request is fully completed.

        Returns:
            True if both generation and saving are complete
        """
        return self.phase == RequestPhase.DONE

    def is_saving(self) -> bool:
        """
        Check if request has started saving.

        Returns:
            True if at least one save intent has been consumed
        """
        return self._stored_blocks > 0

    def __repr__(self) -> str:
        return (
            f"RequestTracker(id={self.request_id}, phase={self.phase.value}, "
            f"hit={self._hit_blocks}, computed={self._computed_blocks}, "
            f"allocated={len(self._allocated_blocks)}, stored={self._stored_blocks}, "
            f"saved_layers={self._saved_layers}/{self._total_layers})"
        )


class PegaConnectorMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for KV cache operations.

    Contains lists of load and save intents that the worker should execute.
    """

    def __init__(
        self,
        load_intents: Optional[Dict[str, LoadIntent]] = None,
        save_intents: Optional[Dict[str, SaveIntent]] = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: Dict[str, LoadIntent] = load_intents or {}
        self.save_intents: Dict[str, SaveIntent] = save_intents or {}

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, "
            f"saves={len(self.save_intents)})"
        )


class PegaKVConnector(KVConnectorBase_V1):
    """
    v1 KV connector for PegaFlow.

    The class provides the following primitives:
        Scheduler-side: runs in the scheduler, binds metadata, which
        is used by the worker-side to load/save KV cache.
            get_num_new_matched_tokens() - IMPLEMENTED
            update_state_after_alloc() - IMPLEMENTED
            build_connector_meta() - IMPLEMENTED
            update_connector_output() - NOT IMPLEMENTED (uses base default)
            request_finished() - NOT IMPLEMENTED (uses base default: returns False, None)
            take_events() - NOT IMPLEMENTED (uses base default: returns empty tuple)

        Worker-side: runs in each worker, loads/saves KV cache to/from
        the Connector based on the metadata.
            start_load_kv() - IMPLEMENTED
            wait_for_layer_load() - IMPLEMENTED
            save_kv_layer() - IMPLEMENTED
            wait_for_save() - IMPLEMENTED
            get_finished() - NOT IMPLEMENTED (uses base default: returns None, None)
            register_kv_caches() - IMPLEMENTED (PegaFlow-specific)
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

        # Determine instance_id with DP rank support
        instance_id = vllm_config.kv_transfer_config.engine_id
        if instance_id:
            logger.info("[PegaKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        else:
            instance_id = vllm_config.instance_id or os.environ.get("PEGAFLOW_INSTANCE_ID", "")
            if not instance_id:
                instance_id = uuid.uuid4().hex
                logger.info("[PegaKVConnector] No instance_id from vLLM; generated fallback %s", instance_id)

            # Append DP rank if data parallelism is enabled
            parallel_config = vllm_config.parallel_config
            if parallel_config.data_parallel_size > 1:
                local_dp_rank = parallel_config.data_parallel_rank_local
                if local_dp_rank is not None:
                    instance_id = f"{instance_id}_dp{local_dp_rank}"
                    logger.info(
                        "[PegaKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                        instance_id, parallel_config.data_parallel_size, local_dp_rank,
                    )

        self._instance_id = instance_id

        # Extract TP info and model metadata
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        # Use total number of layers across all PP ranks for correct slot calculation
        # TODO: Use get_total_num_hidden_layers() when available in next vLLM release
        self._num_layers = getattr(
            vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )

        self._tp_rank: Optional[int] = None
        self._device_id: Optional[int] = None
        if role == KVConnectorRole.WORKER:
            self._tp_rank = get_tensor_model_parallel_rank()
            if torch.cuda.is_available():
                self._device_id = torch.cuda.current_device()

        logger.info(
            "[PegaKVConnector] Initialized role=%s instance_id=%s device=%s tp_rank=%s tp_size=%d layers=%d",
            role.name, self._instance_id,
            self._device_id if self._device_id is not None else "cpu",
            self._tp_rank if self._tp_rank is not None else "N/A",
            self._tp_size, self._num_layers,
        )

        # ZMQ client for connecting to engine server
        self._engine_endpoint = _ENGINE_ENDPOINT
        self._engine_context: Optional[zmq.Context] = None
        self._engine_socket = None
        self._engine_lock = threading.Lock()

        # Async save worker
        self._save_queue = queue.Queue()
        self._save_exception: Optional[Exception] = None
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="PegaSaveWorker"
        )
        self._save_thread.start()

        # Request tracking (Scheduler side)
        self._trackers: Dict[str, RequestTracker] = {}
        self._registered_layers: list[str] = []
        # Pending load intents (Scheduler side) - populated in update_state_after_alloc,
        # consumed in build_connector_meta. This is needed for async loading where
        # the request enters WAITING_FOR_REMOTE_KVS state before being scheduled.
        self._pending_load_intents: Dict[str, LoadIntent] = {}
        # Requests whose blocks we told vLLM to hold while async save completes (scheduler side)
        self._held_requests: set[str] = set()

        # Block size
        self._block_size = vllm_config.cache_config.block_size
        self._layer_name_to_id: Dict[str, int] = {}

        # Async save completion tracking (Worker side)
        self._req_pending_layers: Dict[str, int] = {}  # req_id -> remaining layer count
        self._completed_saves: set[str] = set()  # req_ids with all layers saved
        self._save_completion_lock = threading.Lock()

        # Track requests with save intents in current step (Worker side)
        # Set in start_load_kv, checked in wait_for_save to detect skipped saves
        self._current_save_intents: set[str] = set()

        # Async load tracking (Worker side)
        # Maps req_id -> PyLoadState for pending async loads
        self._pending_loads: Dict[str, "PyLoadState"] = {}
        self._pending_load_reqs: Dict[str, set[str]] = {}  # load_state.shm_name -> set of req_ids
        self._load_completion_lock = threading.Lock()

    # ==============================
    # Engine client helper methods
    # ==============================

    def _ensure_engine_socket(self) -> None:
        """Ensure engine socket is connected."""
        if self._engine_socket is not None:
            return

        if self._engine_context is None:
            self._engine_context = zmq.Context()

        self._engine_socket = self._engine_context.socket(zmq.REQ)
        self._engine_socket.setsockopt(zmq.RCVTIMEO, 20000)
        self._engine_socket.setsockopt(zmq.SNDTIMEO, 20000)
        self._engine_socket.connect(self._engine_endpoint)
        logger.info("[PegaKVConnector] Connected to engine server at %s", self._engine_endpoint)

    def _send_engine_request(self, command: str, payload: dict) -> dict:
        """Send request to engine server and get response."""
        # Prepare payload
        payload = dict(payload)
        payload.setdefault("instance_id", self._instance_id)
        if self._tp_rank is not None:
            payload.setdefault("tp_rank", self._tp_rank)
        if self._device_id is not None:
            payload.setdefault("device_id", self._device_id)

        with self._engine_lock:
            self._ensure_engine_socket()
            return self._zmq_send_recv(self._engine_socket, command, payload)

    def _zmq_send_recv(self, socket: zmq.Socket, command: str, payload: dict) -> dict:
        """Helper for ZMQ send/recv logic."""
        command_bytes = msgpack.packb(command)
        payload_bytes = msgpack.packb(payload, use_bin_type=True)
        socket.send_multipart([command_bytes, payload_bytes])

        response_parts = socket.recv_multipart()
        assert len(response_parts) == 1, f"Invalid response format: expected 1 part, got {len(response_parts)}"

        response = msgpack.unpackb(response_parts[0], raw=False)
        assert response.get('status') == 'success', f"Engine request failed: {response.get('message', 'Unknown error')}"
        return response

    def _save_worker(self) -> None:
        """Background worker for handling async save requests with batching."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 20000)
        socket.setsockopt(zmq.SNDTIMEO, 20000)
        socket.connect(self._engine_endpoint)
        logger.info("[PegaKVConnector] Save worker thread started")

        try:
            while True:
                # Block wait for first task
                task = self._save_queue.get()
                if task is None:
                    self._save_queue.task_done()
                    break

                # Batch pop: collect all available tasks
                batch = [task]
                while True:
                    try:
                        t = self._save_queue.get_nowait()
                        if t is None:
                            # Got shutdown signal, process batch then exit
                            self._process_save_batch(batch, socket)
                            self._save_queue.task_done()
                            return
                        batch.append(t)
                    except queue.Empty:
                        break

                try:
                    self._process_save_batch(batch, socket)
                except Exception as e:
                    logger.error(f"[PegaKVConnector] Save worker error: {e}", exc_info=True)
                    self._save_exception = e
                finally:
                    # Mark all tasks as done
                    for _ in batch:
                        self._save_queue.task_done()

        except Exception as e:
            logger.critical(f"[PegaKVConnector] Save worker crashed: {e}", exc_info=True)
            self._save_exception = e
        finally:
            socket.close()
            context.term()
            logger.info("[PegaKVConnector] Save worker thread stopped")

    def _process_save_batch(self, batch: list[dict], socket: zmq.Socket) -> None:
        """Process a batch of save tasks, sending them in a single request."""
        if not batch:
            return

        # Aggregate saves from all tasks
        # Key: (layer_name) -> {'block_ids': [], 'block_hashes': []}
        saves_by_layer: Dict[str, Dict[str, list]] = {}
        all_request_ids: list[str] = []

        for task in batch:
            metadata: PegaConnectorMetadata = task['metadata']
            attn_metadata = task['attn_metadata']
            layer_name = task['layer_name']
            request_ids = task.get('request_ids', [])

            all_request_ids.extend(request_ids)

            if attn_metadata.block_table is None:
                continue

            for _, save_intent in metadata.save_intents.items():
                if not save_intent.block_ids:
                    continue

                if layer_name not in saves_by_layer:
                    saves_by_layer[layer_name] = {'block_ids': [], 'block_hashes': []}

                saves_by_layer[layer_name]['block_ids'].extend(save_intent.block_ids)
                saves_by_layer[layer_name]['block_hashes'].extend(save_intent.block_hashes)

        # Send batch request if there are saves
        if saves_by_layer:
            saves_list = [
                {
                    'layer_name': layer_name,
                    'block_ids': list(data['block_ids']),
                    'block_hashes': list(data['block_hashes']),
                }
                for layer_name, data in saves_by_layer.items()
            ]

            payload = {
                'instance_id': self._instance_id,
                'saves': saves_list,
            }
            if self._tp_rank is not None:
                payload['tp_rank'] = self._tp_rank
            if self._device_id is not None:
                payload['device_id'] = self._device_id

            self._zmq_send_recv(socket, 'SAVE', payload)

            logger.debug(
                "[PegaKVConnector] Batch saved %d layers, %d total blocks",
                len(saves_list),
                sum(len(s['block_ids']) for s in saves_list),
            )

        # Update completion tracking for all requests in batch
        self._decrement_layer_counter(all_request_ids)

    def _decrement_layer_counter(self, request_ids: list[str]) -> None:
        """Decrement layer counter for requests and mark as completed if all layers done."""
        completed_reqs: list[str] = []

        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id in self._req_pending_layers:
                    # Decrement counter directly (let it crash if count mismatch)
                    self._req_pending_layers[req_id] -= 1
                    assert self._req_pending_layers[req_id] >= 0, \
                        f"Layer count mismatch for request {req_id}: counter went negative"

                    # Check if all layers complete for this request
                    if self._req_pending_layers[req_id] == 0:
                        # All layers processed, mark request complete
                        self._completed_saves.add(req_id)
                        del self._req_pending_layers[req_id]
                        completed_reqs.append(req_id)

        self._handle_save_completion(completed_reqs)

    def _handle_save_completion(self, request_ids: Iterable[str], reason: str | None = None) -> None:
        """
        Log save completion for requests.

        Args:
            request_ids: Iterable of request ids that have finished saving.
            reason: Optional suffix describing why completion was triggered.
        """
        req_list = list(request_ids)
        if not req_list:
            return

        suffix = "" if not reason else f" ({reason})"
        layer_count = len(self._registered_layers) or self._num_layers
        for req_id in req_list:
            logger.info(
                "[PegaKVConnector] Request %s all %d layers saved%s",
                req_id,
                layer_count,
                suffix,
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have finished
        generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        finished_sending: set[str] | None = None
        finished_recving: set[str] | None = None

        # Check for completed saves
        with self._save_completion_lock:
            # Find which completed saves are in the finished set
            done_saves = self._completed_saves & finished_req_ids
            if done_saves:
                # Remove from completed set (avoid duplicate reporting)
                self._completed_saves -= done_saves
                finished_sending = done_saves

        # Check for completed loads (async recv)
        with self._load_completion_lock:
            completed_reqs: set[str] = set()
            completed_shms: list[str] = []

            # Check each pending load state
            for shm_name, req_ids in self._pending_load_reqs.items():
                # Get load_state from any req_id in the group
                sample_req_id = next(iter(req_ids))
                load_state = self._pending_loads.get(sample_req_id)
                if load_state is None:
                    continue

                # Non-blocking check if load completed
                if load_state.is_ready():
                    state = load_state.get_state()
                    if state < 0:
                        logger.error(
                            "[PegaKVConnector] async load failed with state=%d for reqs=%s",
                            state, req_ids,
                        )
                        # TODO: report invalid_block_ids
                    else:
                        logger.info(
                            "[PegaKVConnector] async load completed for %d reqs, shm=%s",
                            len(req_ids), shm_name,
                        )
                    completed_reqs.update(req_ids)
                    completed_shms.append(shm_name)

            # Clean up completed loads
            for shm_name in completed_shms:
                req_ids = self._pending_load_reqs.pop(shm_name, set())
                for req_id in req_ids:
                    self._pending_loads.pop(req_id, None)

            if completed_reqs:
                finished_recving = completed_reqs

        if finished_sending:
            logger.info(f"[PegaKVConnector] finished saving KV for requests: {finished_sending}")
        if finished_recving:
            logger.info(f"[PegaKVConnector] finished loading KV for requests: {finished_recving}")

        return (finished_sending, finished_recving)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return set()

    def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
        """
        Get the KV connector stats collected during the last interval.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    def get_handshake_metadata(self) -> "KVConnectorHandshakeMetadata" | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    # Base class provides implementation for these methods:
    # - bind_connector_metadata(connector_metadata: KVConnectorMetadata)
    # - clear_connector_metadata()
    # We use the base class implementation directly.

    def set_host_xfer_buffer_ops(self, copy_operation: "CopyBlocksOp"):
        """
        Set the xPU-specific ops for copying KV between host and device.
        Needed when host buffer is used for kv transfer (e.g., in NixlConnector)

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return

    @timing_wrapper
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged KV buffer.

        This is an ASYNCHRONOUS operation: we submit async transfers to the engine,
        create a LoadState for synchronization, and return immediately. The actual
        transfer completion is tracked via get_finished().

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PegaConnectorMetadata):
            self._current_save_intents = set()
            return

        # Record save intents for this step (used by wait_for_save to detect skipped saves)
        self._current_save_intents = set(metadata.save_intents.keys())

        if not metadata.load_intents:
            return

        total_requests = len(metadata.load_intents)
        load_start = time.perf_counter()

        # Aggregate all blocks from all load intents
        all_block_ids: List[int] = []
        all_block_hashes: List[bytes] = []
        request_ids: List[str] = []

        for req_id, load_intent in metadata.load_intents.items():
            all_block_ids.extend(load_intent.block_ids)
            all_block_hashes.extend(load_intent.block_hashes)
            request_ids.append(req_id)

        if not all_block_ids:
            return

        # Identify all KV cache layers
        target_layers: List[str] = []
        for layer_name, layer in forward_context.no_compile_layers.items():
            if hasattr(layer, 'kv_cache'):
                target_layers.append(layer_name)

        if not target_layers:
            return

        # Create LoadState for synchronization
        load_state = PyLoadState()
        shm_name = load_state.shm_name()

        self._send_engine_request('LOAD', {
            'load_state_shm': shm_name,
            'layer_names': target_layers,
            'block_ids': all_block_ids,
            'block_hashes': all_block_hashes,
        })

        num_layers = len(target_layers)
        num_blocks = len(all_block_ids)

        schedule_end = time.perf_counter()
        schedule_time_us = (schedule_end - load_start) * 1e6

        # Track pending loads for async completion detection
        with self._load_completion_lock:
            for req_id in request_ids:
                self._pending_loads[req_id] = load_state
            self._pending_load_reqs[shm_name] = set(request_ids)

        logger.info(
            "[PegaKVConnector] started async load: %d blocks across %d layers for %d reqs, "
            "schedule %.0f us, shm=%s",
            num_blocks, num_layers, total_requests, schedule_time_us, shm_name,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        NOTE: This is now a no-op since start_load_kv is fully synchronous.
        All transfers are complete before start_load_kv returns.

        Args:
            layer_name: the name of that layer
        """
        # Synchronous path: all loading done in start_load_kv
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PegaConnectorMetadata):
            return

        # Extract request IDs from save intents
        request_ids = list(metadata.save_intents.keys())
        if not request_ids:
            return

        # Initialize pending layer counter for each request (first layer seen)
        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id not in self._req_pending_layers:
                    # Initialize with total number of registered layers
                    self._req_pending_layers[req_id] = len(self._registered_layers)

        self._save_queue.put({
            'layer_name': layer_name,
            'attn_metadata': attn_metadata,
            'metadata': metadata,
            'request_ids': request_ids,
        })

    @timing_wrapper
    def wait_for_save(self) -> None:
        """
        Non-blocking check for save errors and skipped saves.

        Detects requests that had save intents but save_kv_layer() was never called
        (e.g., due to CUDA graph), and marks them as completed to avoid deadlock.
        """
        skipped_requests: set[str] = set()

        with self._save_completion_lock:
            pending_layers = set(self._req_pending_layers.keys())
            # Detect skipped saves: requests with save intent but not in pending layers
            # This happens when CUDA graph skips save_kv_layer()
            skipped_requests = self._current_save_intents - pending_layers
            if skipped_requests:
                # Mark skipped saves as completed
                self._completed_saves.update(skipped_requests)

            # Clear for next step
            self._current_save_intents = set()

            pending_reqs = len(self._req_pending_layers)
            if pending_reqs > 0:
                logger.info(f"[PegaKVConnector] {pending_reqs} requests still have pending layer saves")

        if skipped_requests:
            logger.info(
                "[PegaKVConnector] Detected %d skipped saves (CUDA graph): %s",
                len(skipped_requests),
                skipped_requests,
            )
            self._handle_save_completion(skipped_requests, reason="CUDA graph skip")


    # ==============================
    # Scheduler-side methods
    # ==============================

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Process finished sends - notify trackers and clean up
        for req_id in connector_output.finished_sending or []:
            tracker = self._trackers.get(req_id)
            if tracker:
                # Mark all layers as saved (worker confirmed completion)
                while tracker._saved_layers < tracker._total_layers:
                    tracker.on_layer_saved()
                logger.info(f"[PegaKVConnector] Request {req_id} save completed, phase={tracker.phase.value}")

                # Clean up tracker if fully done
                if tracker.is_done():
                    del self._trackers[req_id]
                    logger.debug(f"[PegaKVConnector] Cleaned up tracker for {req_id}")

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        tracker = self._trackers.get(req_id)

        if tracker:
            # Notify tracker that generation is finished
            tracker.on_finished()

            # Check if we need to hold blocks for async save
            if tracker.should_hold_blocks():
                # Track this request so get_finished() can report it later
                self._held_requests.add(req_id)
                logger.info(f"[PegaKVConnector] Request {req_id} blocks held for async save")
                return (True, None)

        return (False, None)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return ()

    @timing_wrapper
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        req_id = request.request_id
        num_tokens = request.num_tokens
        block_hashes = request.block_hashes

        # Get or create tracker for this request
        tracker = self._get_or_create_tracker(request)

        lookup_start = time.perf_counter()
        hit_blocks = self._count_available_block_prefix(block_hashes)
        lookup_end = time.perf_counter()
        elapsed_us = (lookup_end - lookup_start) * 1e6

        computed_blocks = num_computed_tokens // self._block_size

        # Notify tracker of lookup result
        tracker.on_lookup(hit_blocks, computed_blocks)

        # Calculate tokens to load
        num_hit_tokens = hit_blocks * self._block_size - num_computed_tokens

        if num_hit_tokens <= 0:
            return (0, False)

        # Avoid loading all tokens (need at least 1 token to compute)
        if num_hit_tokens >= num_tokens:
            num_hit_tokens = num_tokens - 1

        logger.info(
            "[PegaKVConnector] scheduler_lookup req=%s hit_blocks=%d computed_blocks=%d "
            "hit_tokens=%d elapsed_us=%.0f async=True",
            req_id, hit_blocks, computed_blocks, num_hit_tokens, elapsed_us,
        )
        return (num_hit_tokens, True)  # async loading enabled

    @timing_wrapper
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        req_id = request.request_id
        tracker = self._trackers.get(req_id)
        if tracker is None:
            logger.warning("[PegaKVConnector] No tracker for request %s in update_state_after_alloc", req_id)
            return

        # Extract block IDs from the allocation
        block_ids = []
        if blocks is not None:
            raw_block_ids = blocks.get_block_ids()
            if raw_block_ids:
                # blocks.get_block_ids() returns tuple[list[int], ...] for each KV group
                block_ids = list(raw_block_ids[0]) if raw_block_ids[0] else []

        # Notify tracker of allocation
        tracker.on_alloc(block_ids, num_external_tokens)

        # For async loading: generate LoadIntent immediately and store it.
        # This is consumed by build_connector_meta() in the same scheduler step,
        # even though the request enters WAITING_FOR_REMOTE_KVS state.
        if num_external_tokens > 0:
            load_intent = tracker.consume_load_intent()
            if load_intent is not None:
                self._pending_load_intents[req_id] = load_intent
                logger.info(
                    "[PegaKVConnector] update_state_after_alloc req=%s created LoadIntent: "
                    "%d blocks, %d tokens",
                    req_id, len(load_intent.block_ids), load_intent.num_tokens,
                )

        logger.debug(
            "[PegaKVConnector] update_state_after_alloc req=%s blocks=%d external_tokens=%d phase=%s",
            req_id, len(block_ids), num_external_tokens, tracker.phase.value,
        )

    @timing_wrapper
    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        save_intents: Dict[str, SaveIntent] = {}

        # Consume pending load intents (created in update_state_after_alloc for async loading)
        # This handles requests that enter WAITING_FOR_REMOTE_KVS state
        load_intents = self._pending_load_intents
        self._pending_load_intents = {}

        # Process new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            tracker = self._trackers.get(req_id)
            if tracker is None:
                continue

            # Update scheduled tokens
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            tracker.on_scheduled(num_tokens)

            # Load intent already consumed in update_state_after_alloc for async loading
            # Only try here for sync loading (which we don't use anymore, but keep for safety)
            if req_id not in load_intents:
                if load_intent := tracker.consume_load_intent():
                    load_intents[req_id] = load_intent

            # Try to get save intent
            if save_intent := tracker.consume_save_intent():
                save_intents[req_id] = save_intent

        # Process cached requests (continuing generation)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            tracker = self._trackers.get(req_id)
            if tracker is None:
                continue

            # Update scheduled tokens from num_scheduled_tokens (not num_computed_tokens!)
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            tracker.on_scheduled(num_tokens)

            # Update block IDs if new blocks were allocated
            new_block_ids = cached_reqs.new_block_ids[idx]
            if new_block_ids:
                block_ids = list(new_block_ids[0]) if new_block_ids[0] else []
                tracker.on_alloc(block_ids, 0)

            # Try to get save intent
            if save_intent := tracker.consume_save_intent():
                save_intents[req_id] = save_intent

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves",
            len(load_intents), len(save_intents),
        )

        return PegaConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
        )

    def _get_or_create_tracker(self, request: "Request") -> RequestTracker:
        """
        Get existing tracker or create a new one for the request.

        Args:
            request: The vLLM request object

        Returns:
            RequestTracker for this request
        """
        req_id = request.request_id
        if req_id not in self._trackers:
            self._trackers[req_id] = RequestTracker(
                request_id=req_id,
                block_hashes=list(request.block_hashes),
                block_size=self._block_size,
                num_layers=self._num_layers,
            )
        return self._trackers[req_id]

    def _count_available_block_prefix(self, block_hashes: List[bytes]) -> int:
        """
        Return length of contiguous prefix available in CPU storage.

        Note: This is a PegaFlow-specific helper method.
        """
        if not block_hashes:
            return 0

        response = self._send_engine_request('QUERY', {'block_hashes': block_hashes})
        return response.get('hit_blocks', 0)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        assert self._device_id is not None, "CUDA device id is unknown; cannot register KV caches"

        self._registered_layers = list(kv_caches.keys())

        # Build layer_name -> layer_id mapping
        self._layer_name_to_id.clear()
        for layer_id, layer_name in enumerate(kv_caches.keys()):
            self._layer_name_to_id[layer_name] = layer_id

        layout = "unknown"
        for layer_name, kv_cache in kv_caches.items():
            assert kv_cache.storage_offset() == 0, f"KV cache for {layer_name} must have zero storage offset"

            wrapper = CudaIPCWrapper(kv_cache)
            wrapper_bytes = pickle.dumps(wrapper)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            # Detect KV cache layout
            if len(shape) >= 2 and shape[0] == 2:
                # KV-first layout: (2, num_blocks, ...)
                num_blocks = shape[1]
                bytes_per_block = stride[1] * element_size
                kv_stride_bytes = stride[0] * element_size
                segments = 2
                layout = "KV-first"
            else:
                # Blocks-first layout: (num_blocks, ...)
                num_blocks = shape[0]
                bytes_per_block = stride[0] * element_size
                kv_stride_bytes = 0
                segments = 1
                layout = "blocks-first"

            assert bytes_per_block != 0, f"Invalid bytes_per_block for {layer_name}: stride={stride}"

            self._send_engine_request('REGISTER_CONTEXT', {
                'layer_name': layer_name,
                'wrapper_bytes': wrapper_bytes,
                'num_blocks': num_blocks,
                'bytes_per_block': bytes_per_block,
                'kv_stride_bytes': kv_stride_bytes,
                'segments': segments,
                'tp_size': self._tp_size,
                'num_layers': self._num_layers,
                'device_id': self._device_id,
            })

        logger.info(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout) instance=%s",
            len(kv_caches), layout, self._instance_id,
        )

    def unregister_context(self) -> None:
        """
        Unregister the active inference context from the engine server.

        Note: This is a PegaFlow-specific method not in the base class.
        """
        if not self._registered_layers:
            return

        if self._tp_rank == 0:
            self._send_engine_request('UNREGISTER_CONTEXT', {})

        self._registered_layers.clear()

    # ==============================
    # Additional base class methods NOT IMPLEMENTED in PegaFlow
    # ==============================

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional["KVConnectorStats"]:
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, "KVConnectorHandshakeMetadata"]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (KVConnectorHandshakeMetadata): the handshake metadata to set.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ) -> Optional["KVConnectorPromMetrics"]:
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.

        Note: NOT IMPLEMENTED in PegaFlow - uses base class default.
        """
        return None

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        self.unregister_context()

        # Stop save worker
        self._save_queue.put(None)
        self._save_thread.join()

        if self._engine_socket is not None:
            self._engine_socket.close(0)
            self._engine_socket = None

        if self._engine_context is not None:
            self._engine_context.term()
            self._engine_context = None


__all__ = ["PegaKVConnector", "KVConnectorRole"]
