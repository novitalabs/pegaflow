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
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

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
from pegaflow.pegaflow import PyLayerSyncState

logger = get_connector_logger()

# Engine server endpoint (independent process)
_ENGINE_ENDPOINT = os.environ.get("PEGAFLOW_ENGINE_ENDPOINT", "ipc:///tmp/pega_engine.sock")

ReqId = str

class PegaConnectorMetadata(KVConnectorMetadata):
    def __init__(
        self,
        block_hashes: Optional[Dict[str, List[bytes]]] = None,
        requests_to_load: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.block_hashes = block_hashes or {}
        self.requests_to_load = requests_to_load or {}


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
        self._num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)

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

        self._request_block_hashes = {}  # req_id -> list[bytes]
        self._requests_to_load = {}  # req_id -> dict with load info
        self._registered_layers: list[str] = []

        # Block size and sync state
        self._block_size = vllm_config.cache_config.block_size
        self._sync_state: Optional[PyLayerSyncState] = None
        self._layer_name_to_id: Dict[str, int] = {}
        self._load_in_progress: bool = False

        # Async save completion tracking (Worker side)
        self._req_pending_layers: Dict[str, int] = {}  # req_id -> remaining layer count
        self._completed_saves: set[str] = set()  # req_ids with all layers saved
        self._save_completion_lock = threading.Lock()

        # Scheduler side: track requests being saved
        self._requests_being_saved: set[str] = set()  # req_ids currently being saved

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
        """Background worker for handling async save requests."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 20000)
        socket.setsockopt(zmq.SNDTIMEO, 20000)
        socket.connect(self._engine_endpoint)
        logger.info("[PegaKVConnector] Save worker thread started")

        try:
            while True:
                task = self._save_queue.get()
                if task is None:
                    self._save_queue.task_done()
                    break

                try:
                    self._process_save_task(task, socket)
                except Exception as e:
                    logger.error(f"[PegaKVConnector] Save worker error: {e}", exc_info=True)
                    self._save_exception = e
                finally:
                    self._save_queue.task_done()
        except Exception as e:
            logger.critical(f"[PegaKVConnector] Save worker crashed: {e}", exc_info=True)
            self._save_exception = e
        finally:
            socket.close()
            context.term()
            logger.info("[PegaKVConnector] Save worker thread stopped")

    def _process_save_task(self, task: dict, socket: zmq.Socket) -> None:
        """Process a single save task in the worker thread."""
        layer_name = task['layer_name']
        attn_metadata = task['attn_metadata']
        metadata = task['metadata']
        request_ids = task.get('request_ids', [])

        if attn_metadata.block_table is None:
            # Still need to decrement counter even if no blocks to save
            self._decrement_layer_counter(request_ids)
            return

        try:
            # Execute save logic
            block_table = attn_metadata.block_table
            seq_lens = attn_metadata.seq_lens
            layer_blocks_saved = 0

            # Common payload fields
            payload_base = {
                "instance_id": self._instance_id,
                "layer_name": layer_name,
            }
            if self._tp_rank is not None:
                payload_base["tp_rank"] = self._tp_rank
            if self._device_id is not None:
                payload_base["device_id"] = self._device_id
            for seq_idx in range(block_table.shape[0]):
                if seq_lens is not None:
                    seq_len = seq_lens[seq_idx].item()
                    num_blocks = (seq_len + self._block_size - 1) // self._block_size
                else:
                    num_blocks = (block_table[seq_idx] != 0).sum().item()

                if num_blocks == 0:
                    continue

                active_blocks = block_table[seq_idx, :num_blocks].cpu().tolist()

                # Find matching block hashes from metadata
                block_hashes_for_seq = None
                for req_id, hashes in metadata.block_hashes.items():
                    if len(hashes) > 0:
                        num_use = min(num_blocks, len(hashes))
                        block_hashes_for_seq = hashes[:num_use]
                        active_blocks = active_blocks[:num_use]
                        break

                if block_hashes_for_seq is None:
                    continue

                # Send SAVE request
                payload = payload_base.copy()
                payload.update({
                    'block_ids': active_blocks,
                    'block_hashes': block_hashes_for_seq,
                })
                self._zmq_send_recv(socket, 'SAVE', payload)
                layer_blocks_saved += len(block_hashes_for_seq)

        except Exception as e:
            logger.error(f"[PegaKVConnector] Save failed for layer {layer_name}: {e}", exc_info=True)
            # Continue processing even on failure (mark failed but continue)

        # Update completion tracking (success or failure, always decrement)
        self._decrement_layer_counter(request_ids)

    def _decrement_layer_counter(self, request_ids: list[str]) -> None:
        """Decrement layer counter for requests and mark as completed if all layers done."""
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
                        logger.info(f"[PegaKVConnector] Request {req_id} all {len(self._registered_layers)} layers saved")

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
        with self._save_completion_lock:
            # Find which completed saves are in the finished set
            finished_sending = self._completed_saves & finished_req_ids

            # Remove from completed set (avoid duplicate reporting)
            self._completed_saves -= finished_sending

        if finished_sending:
            logger.info(f"[PegaKVConnector] Finished saving KV for requests: {finished_sending}")
            return (finished_sending, None)
        return (None, None)

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
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self._load_in_progress = False

        metadata = self._get_connector_metadata()

        if not metadata.requests_to_load:
            logger.info("not metadata to load")
            return

        total_requests = len(metadata.requests_to_load)
        load_start = time.perf_counter()

        # Aggregate all blocks from all requests
        all_block_ids: List[int] = []
        all_block_hashes: List[bytes] = []

        for req_id, load_info in metadata.requests_to_load.items():
            all_block_ids.extend(load_info['block_ids'])
            all_block_hashes.extend(load_info['block_hashes'])

        if not all_block_ids:
            logger.info("not block")
            return

        # Identify all KV cache layers
        target_layers: List[str] = []
        for layer_name, layer in forward_context.no_compile_layers.items():
            if hasattr(layer, 'kv_cache'):
                target_layers.append(layer_name)

        if not target_layers:
            logger.info("not layers")
            return

        self._load_in_progress = True

        response = self._send_engine_request('LOAD', {
            'layer_names': target_layers,
            'block_ids': all_block_ids,
            'block_hashes': all_block_hashes,
        })

        num_layers_loaded = response.get('num_layers_loaded', 0)
        total_bytes = response.get('total_bytes', 0)
        total_blocks = len(all_block_ids) * num_layers_loaded

        transfer_end = time.perf_counter()
        total_time_us = (transfer_end - load_start) * 1e6
        total_time_s = total_time_us / 1e6
        bandwidth_gbps = (total_bytes / 1e9) / total_time_s if total_time_s > 0 else 0.0

        logger.info(
            "[PegaKVConnector] queued %d blocks (%.2f GB) across %d layers for %d reqs, schedule %.0f us (%.2f GB/s)",
            total_blocks, total_bytes / 1e9, num_layers_loaded, total_requests, total_time_us, bandwidth_gbps,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if not self._load_in_progress:
            return

        if self._sync_state is not None and layer_name in self._layer_name_to_id:
            layer_id = self._layer_name_to_id[layer_name]
            self._sync_state.wait_layer(layer_id)
            return

        raise NotImplementedError("wait_for_layer_load through ZMQ not supported")

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

        # Extract request IDs from metadata
        request_ids = list(metadata.block_hashes.keys())
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
        Non-blocking check for save errors.
        Actual save completion is tracked via get_finished().

        This prevents overwrites of paged KV buffer before saving done.
        """
        with self._save_completion_lock:
            pending_reqs = len(self._req_pending_layers)
            if pending_reqs > 0:
                logger.info(f"[PegaKVConnector] {pending_reqs} requests still have pending layer saves")


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
        # Process finished sends - remove from being_saved set
        for req_id in connector_output.finished_sending or []:
            self._requests_being_saved.discard(req_id)
            logger.info(f"[PegaKVConnector] Request {req_id} save completed, blocks can be freed")

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

        # If this request is being saved, delay block freeing
        if req_id in self._requests_being_saved:
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

        lookup_start = time.perf_counter()
        matched_blocks = self._count_available_block_prefix(block_hashes)
        lookup_end = time.perf_counter()

        elapsed_us = (lookup_end - lookup_start) * 1e6

        num_hit_tokens = matched_blocks * self._block_size - num_computed_tokens

        if num_hit_tokens <= 0:
            return (0, False)

        if num_hit_tokens == num_tokens:
            num_hit_tokens -= 1

        logger.info(
            "[PegaKVConnector] scheduler_lookup req=%s hit_tokens=%d, num_computed_tokens=%d, elapsed_us=%d",
            req_id, num_hit_tokens, num_computed_tokens, elapsed_us,
        )
        return (num_hit_tokens, False)

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
        self._request_block_hashes[req_id] = request.block_hashes

        if num_external_tokens > 0:
            self._requests_to_load[req_id] = {
                'request': request,
                'blocks': blocks,
                'num_external_tokens': num_external_tokens,
            }

    @timing_wrapper
    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        block_hashes = {}

        # Process new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            if req_id in self._request_block_hashes:
                block_hashes[req_id] = self._request_block_hashes[req_id]

        # Process cached requests
        for i, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            if req_id in self._request_block_hashes:
                block_hashes[req_id] = self._request_block_hashes[req_id]

        # Mark requests with block_hashes as being saved
        for req_id, hashes in block_hashes.items():
            if hashes:
                self._requests_being_saved.add(req_id)

        # Process requests that need to load from CPU storage
        requests_to_load = {}
        for req_id, load_info in self._requests_to_load.items():
            num_external_tokens = load_info['num_external_tokens']

            for req in scheduler_output.scheduled_new_reqs:
                if req.req_id == req_id:
                    block_ids = list(req.block_ids[0]) if req.block_ids else []
                    num_blocks = (num_external_tokens + self._block_size - 1) // self._block_size
                    saved_hashes = self._request_block_hashes.get(req_id, [])
                    num_blocks = min(num_blocks, len(saved_hashes))

                    if num_blocks > 0 and len(block_ids) >= num_blocks:
                        requests_to_load[req_id] = {
                            'block_ids': block_ids[:num_blocks],
                            'block_hashes': saved_hashes[:num_blocks],
                            'num_tokens': num_external_tokens,
                        }
                    break

        self._requests_to_load.clear()

        return PegaConnectorMetadata(
            block_hashes=block_hashes,
            requests_to_load=requests_to_load,
        )

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

        # Create shared-memory sync state for async layer loading
        self._sync_state = PyLayerSyncState(self._num_layers)
        shm_name = self._sync_state.shm_name()

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
                'shm_name': shm_name,
                'device_id': self._device_id,
            })

        logger.info(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout) instance=%s shm=%s",
            len(kv_caches), layout, self._instance_id, shm_name,
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
