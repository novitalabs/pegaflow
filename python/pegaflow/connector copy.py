from __future__ import annotations

"""Baseline vLLM v1 KV connector for local development.

This module defines :class:`PegaKVConnector`, a thin subclass of
``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``.

At the moment it only mirrors the abstract API and raises
``NotImplementedError`` in all required methods, so that we have a
self-contained place inside this repo to start iterating on our own
PegaFlow-backed connector implementation.

Usage example (scheduler/worker side)::

    from pegaflow import PegaKVConnector, KVConnectorRole

    connector = PegaKVConnector(vllm_config, KVConnectorRole.WORKER)

Later we can register this class as a dynamic connector in vLLM by
referencing it via its full import path.
"""

import functools
import logging
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

# Import CUDA IPC wrapper for cross-process tensor sharing
from pegaflow.ipc_wrapper import CudaIPCWrapper

logger = logging.getLogger(__name__)
# Enable INFO logs by default so required operational logs are visible even if
# the host application doesn't configure logging.
logger.setLevel(logging.INFO)

# Environment variable to control timing logging
_ENABLE_TIMING = os.environ.get("PEGAFLOW_ENABLE_TIMING", "1") == "1"


def timing_wrapper(func):
    """Decorator to log function name and execution time when enabled.

    Enable by setting environment variable: PEGAFLOW_ENABLE_TIMING=1
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _ENABLE_TIMING:
            return func(*args, **kwargs)

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "[PegaKVConnector] %s took %.2f ms",
                func.__name__,
                elapsed_ms,
            )
    return wrapper

if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.NOTSET)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    # Prevent duplicate output while using the fallback handler.
    logger.propagate = False

_LOOKUP_ENDPOINT = os.environ.get("PEGAFLOW_KV_LOOKUP_ENDPOINT")
if _LOOKUP_ENDPOINT is None:
    unique_id = getattr(os, "getuid", os.getpid)()
    _LOOKUP_ENDPOINT = f"ipc:///tmp/pegaflow_kv_lookup_{unique_id}.sock"

# Engine server endpoint (independent process)
_ENGINE_ENDPOINT = os.environ.get("PEGAFLOW_ENGINE_ENDPOINT", "ipc:///tmp/pega_engine.sock")


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata for PegaFlow KV connector.

    Contains information needed to save/load KV cache blocks:
    - block_hashes: content hashes for each block
    - requests_to_load: mapping from request ID to load information
    """

    def __init__(
        self,
        block_hashes: Optional[Dict[str, List[bytes]]] = None,
        requests_to_load: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.block_hashes = block_hashes or {}
        self.requests_to_load = requests_to_load or {}

class PegaKVConnector(KVConnectorBase_V1):
    """Skeleton v1 KV connector for PegaFlow.

    This class intentionally keeps the same method signatures as
    :class:`KVConnectorBase_V1` so that it can be used as a drop-in
    implementation once we fill in the logic. All abstract methods
    currently raise :class:`NotImplementedError`.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        """Create a new PegaKVConnector.

        Args:
            vllm_config: vLLM configuration object.
            role: Whether this connector instance runs in the scheduler
                process or the worker process.
        """
        super().__init__(vllm_config, role)

        # ZMQ client for connecting to engine server (independent process)
        self._engine_endpoint = _ENGINE_ENDPOINT
        self._engine_context: Optional[zmq.Context] = None
        self._engine_socket = None
        self._engine_lock = threading.Lock()  # Protect socket access

        # Track block hashes for each request across steps
        self._request_block_hashes = {}  # req_id -> list[bytes]

        # Track pending save operations
        self._pending_saves = []  # list[dict]

        # Track requests that need to load KV cache from CPU
        self._requests_to_load = {}  # req_id -> dict with load info

        # Track registered KV cache layers
        self._registered_layers: list[str] = []

        # Scheduler/worker lookup channel state
        self._lookup_endpoint = _LOOKUP_ENDPOINT
        self._lookup_context: Optional[zmq.Context] = None
        self._lookup_server_socket = None
        self._lookup_server_thread: Optional[threading.Thread] = None
        self._lookup_stop_event = threading.Event()
        self._lookup_client = None

        # Get block size from vllm_config
        self._block_size = vllm_config.cache_config.block_size
        # NOTE: KV cache layout is detected in register_kv_caches() by checking tensor shape.
        # vLLM uses KV-first layout: (2, num_blocks, block_size, num_heads, head_dim)
        # where the first dimension (2) represents K and V separately.

        # Only worker rank 0 needs to host the lookup server
        parallel_config = getattr(vllm_config, "parallel_config", None)
        data_parallel_rank = getattr(parallel_config, "data_parallel_rank", 0)
        if role == KVConnectorRole.WORKER and data_parallel_rank == 0:
            self._start_lookup_server()

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
        self._engine_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self._engine_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._engine_socket.connect(self._engine_endpoint)
        logger.info("[PegaKVConnector] Connected to engine server at %s", self._engine_endpoint)

    def _send_engine_request(self, command: str, payload: dict) -> dict:
        """Send request to engine server and get response.

        Args:
            command: Command name (REGISTER, SAVE, LOAD, QUERY, etc.)
            payload: Command payload dict

        Returns:
            Response dict with 'status' and optional result data

        Raises:
            RuntimeError: If request fails or times out
        """
        with self._engine_lock:
            try:
                self._ensure_engine_socket()

                # Send request
                request = (command, payload)
                self._engine_socket.send(pickle.dumps(request))

                # Receive response
                response_bytes = self._engine_socket.recv()
                response = pickle.loads(response_bytes)

                if response.get('status') != 'success':
                    error_msg = response.get('message', 'Unknown error')
                    raise RuntimeError(f"Engine request failed: {error_msg}")

                return response

            except zmq.error.Again:
                raise RuntimeError(f"Engine request timeout: {command}")
            except Exception as e:
                # Try to reconnect on next request
                if self._engine_socket is not None:
                    try:
                        self._engine_socket.close()
                    except Exception:
                        pass
                    self._engine_socket = None
                raise RuntimeError(f"Engine request error: {e}") from e

    # ==============================
    # Worker-side methods
    # ==============================

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
        # ============================================================
        # STEP 1: Get connector metadata
        # ============================================================
        metadata = self._get_connector_metadata()

        if not isinstance(metadata, PegaConnectorMetadata):
            return

        # ============================================================
        # STEP 2: Check if there are requests to load
        # ============================================================
        if not metadata.requests_to_load:
            return

        total_requests = len(metadata.requests_to_load)

        # ============================================================
        # STEP 3: Load KV blocks for each request and each layer
        # ============================================================
        try:
            load_start = time.perf_counter()

            # Aggregate all blocks from all requests
            all_block_ids: List[int] = []
            all_block_hashes: List[bytes] = []

            for req_id, load_info in metadata.requests_to_load.items():
                block_ids = load_info['block_ids']
                block_hashes = load_info['block_hashes']
                num_tokens = load_info['num_tokens']

                all_block_ids.extend(block_ids)
                all_block_hashes.extend(block_hashes)
            
            if not all_block_ids:
                return

            # Identify all KV cache layers in the order provided by vLLM
            target_layers: List[str] = []
            for layer_name, layer in forward_context.no_compile_layers.items():
                if hasattr(layer, 'kv_cache'):
                    target_layers.append(layer_name)

            if not target_layers:
                return

            # Batch load for all layers with async transfers
            response = self._send_engine_request('LOAD', {
                'layer_names': target_layers,
                'block_ids': all_block_ids,
                'block_hashes': all_block_hashes
            })

            num_layers_loaded = response.get('num_layers_loaded', 0)
            total_bytes = response.get('total_bytes', 0)

            total_blocks = len(all_block_ids) * num_layers_loaded
            total_layers = num_layers_loaded

            transfer_end = time.perf_counter()
            total_time_us = (transfer_end - load_start) * 1e6
            total_time_s = total_time_us / 1e6
            bandwidth_gbps = (total_bytes / 1e9) / total_time_s if total_time_s > 0 else 0.0

            logger.info(
                "[PegaKVConnector] queued %d blocks (%.2f GB) across %d layers for %d reqs, "
                "schedule %.0f us (%.2f GB/s)",
                total_blocks,
                total_bytes / 1e9,
                num_layers_loaded,
                total_requests,
                total_time_us,
                bandwidth_gbps,
            )

        except Exception as e:
            logger.debug(
                "[PegaKVConnector] Error in start_load_kv: %s",
                e,
                exc_info=True,
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
        try:
            self._send_engine_request('WAIT_LAYER', {'layer_name': layer_name})
        except Exception as exc:
            logger.debug(
                "[PegaKVConnector] wait_for_layer_load failed for %s: %s",
                layer_name,
                exc,
                exc_info=True,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",  # type: ignore[name-defined]
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

        # Store for later processing in wait_for_save
        self._pending_saves.append({
            'layer_name': layer_name,
            'attn_metadata': attn_metadata,
        })

    @timing_wrapper
    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        import time

        # ============================================================
        # STEP 1: Check if there are pending saves
        # ============================================================
        if len(self._pending_saves) == 0:
            return

        try:
            total_start = time.perf_counter()

            # ============================================================
            # STEP 2: Get connector metadata
            # ============================================================
            metadata = self._get_connector_metadata()

            if not isinstance(metadata, PegaConnectorMetadata):
                return

            # ============================================================
            # STEP 3: Create CUDA event for synchronization
            # ============================================================
            with torch.cuda.stream(torch.cuda.current_stream()):
                event = torch.cuda.Event(interprocess=True)
                event.record()

            # ============================================================
            # STEP 4: Process each layer's save operation
            # ============================================================
            total_blocks_saved = 0
            total_layers_saved = 0

            for save_info in self._pending_saves:
                layer_name = save_info['layer_name']
                attn_metadata = save_info['attn_metadata']

                # Skip if block_table is missing or None
                if attn_metadata.block_table is None:
                    continue

                block_table = attn_metadata.block_table  # [num_seqs, max_blocks]
                seq_lens = attn_metadata.seq_lens

                layer_blocks_saved = 0

                # Process each sequence in the batch
                for seq_idx in range(block_table.shape[0]):
                    # Calculate number of blocks needed for this sequence
                    if seq_lens is not None:
                        seq_len = seq_lens[seq_idx].item()
                        num_blocks = (seq_len + self._block_size - 1) // self._block_size
                    else:
                        # Fallback: count non-zero blocks
                        num_blocks = (block_table[seq_idx] != 0).sum().item()

                    if num_blocks == 0:
                        continue

                    # Get active block IDs for this sequence
                    active_blocks = block_table[seq_idx, :num_blocks].cpu().tolist()

                    # Find matching block hashes from metadata
                    # TODO: Improve mapping between seq_idx and req_id
                    block_hashes_for_seq = None
                    matched_req_id = None
                    for req_id, hashes in metadata.block_hashes.items():
                        if len(hashes) > 0:
                            num_use = min(num_blocks, len(hashes))
                            block_hashes_for_seq = hashes[:num_use]
                            active_blocks = active_blocks[:num_use]
                            matched_req_id = req_id
                            break

                    if block_hashes_for_seq is None:
                        continue

                    # Save blocks to storage via Rust backend
                    try:
                        self._send_engine_request('SAVE', {
                            'layer_name': layer_name,
                            'block_ids': active_blocks,
                            'block_hashes': block_hashes_for_seq
                        })
                        layer_blocks_saved += len(block_hashes_for_seq)
                    except Exception:
                        # Silently skip failed saves
                        pass

                if layer_blocks_saved > 0:
                    total_blocks_saved += layer_blocks_saved
                    total_layers_saved += 1

            # ============================================================
            # STEP 5: Wait for CUDA operations to complete
            # ============================================================
            event.synchronize()
            total_end = time.perf_counter()
            total_time_ms = (total_end - total_start) * 1000

            if total_blocks_saved > 0:
                logger.debug(
                    "[PegaKVConnector] saved %d blocks across %d layers (%.2f ms)",
                    total_blocks_saved,
                    total_layers_saved,
                    total_time_ms,
                )

        except Exception:
            # Silently handle errors
            pass
        finally:
            # ============================================================
            # STEP 6: Clean up pending saves
            # ============================================================
            self._pending_saves.clear()

    # ==============================
    # Scheduler-side methods
    # ==============================

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
        prompt_token_ids = request.prompt_token_ids or []
        req_id = request.request_id
        num_tokens = len(prompt_token_ids)
        block_hashes = request.block_hashes

        matched_blocks = self._send_lookup_request(req_id, block_hashes)
        if matched_blocks <= 0:
            return (0, False)

        available_tokens = min(matched_blocks * self._block_size, num_tokens)
        if available_tokens <= 1:
            return (0, False)

        # Always leave at least one prompt token for the scheduler to compute
        reusable_tokens = available_tokens - 1
        num_new_tokens = reusable_tokens - num_computed_tokens

        if num_new_tokens <= 0:
            return (0, False)

        return (num_new_tokens, False)

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

        # block hashes is  a list[bytes]
        self._request_block_hashes[req_id] = request.block_hashes

        # If there are external tokens to load, record this request
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

        # ============================================================
        # STEP 1: Process new requests (first time scheduled)
        # ============================================================
        new_reqs = scheduler_output.scheduled_new_reqs
        for req in new_reqs:
            req_id = req.req_id

            # Use block hashes saved from update_state_after_alloc()
            # These are vLLM's content-based hashes computed from token sequences
            if req_id in self._request_block_hashes:
                saved_hashes = self._request_block_hashes[req_id]
                block_hashes[req_id] = saved_hashes

        # ============================================================
        # STEP 2: Process cached requests (already scheduled, now in decode phase)
        # ============================================================
        # Note: For cached requests, block_hashes are already updated in
        # update_state_after_alloc() when new blocks are allocated during decode.
        # We just need to retrieve them from our persistent state.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            # Use block hashes from persistent state (updated by update_state_after_alloc)
            if req_id in self._request_block_hashes:
                saved_hashes = self._request_block_hashes[req_id]
                block_hashes[req_id] = saved_hashes

        # ============================================================
        # STEP 3: Process requests that need to load from CPU storage
        # ============================================================
        requests_to_load = {}

        for req_id, load_info in self._requests_to_load.items():
            num_external_tokens = load_info['num_external_tokens']

            # Find this request in scheduler_output
            found = False
            for req in scheduler_output.scheduled_new_reqs:
                if req.req_id == req_id:
                    # Extract block IDs from the request
                    block_ids = list(req.block_ids[0]) if req.block_ids else []

                    # Calculate number of blocks needed, clamp to available hashes
                    num_blocks = (num_external_tokens + self._block_size - 1) // self._block_size
                    saved_hashes = self._request_block_hashes.get(req_id, [])
                    num_blocks = min(num_blocks, len(saved_hashes))

                    if num_blocks > 0 and len(block_ids) >= num_blocks:
                        load_hashes = saved_hashes[:num_blocks]

                        # Store load information
                        requests_to_load[req_id] = {
                            'block_ids': block_ids[:num_blocks],
                            'block_hashes': load_hashes,
                            'num_tokens': num_external_tokens,
                        }

                    found = True
                    break

        # Clear the requests_to_load after processing
        self._requests_to_load.clear()

        # ============================================================
        # STEP 4: Build and return metadata
        # ============================================================
        metadata = PegaConnectorMetadata(
            block_hashes=block_hashes,
            requests_to_load=requests_to_load,
        )

        return metadata

    def _start_lookup_server(self) -> None:
        """Start background REP server for scheduler lookup requests."""
        if self._lookup_server_thread is not None:
            return

        self._lookup_context = zmq.Context()
        self._lookup_stop_event.clear()
        self._lookup_server_socket = self._lookup_context.socket(zmq.REP)

        if self._lookup_endpoint.startswith("ipc://"):
            ipc_path = self._lookup_endpoint.replace("ipc://", "")
            try:
                os.unlink(ipc_path)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

        self._lookup_server_socket.bind(self._lookup_endpoint)

        thread = threading.Thread(target=self._lookup_server_loop, daemon=True)
        thread.start()
        self._lookup_server_thread = thread
        logger.info(
            "[PegaKVConnector] Lookup server started at %s",
            self._lookup_endpoint,
        )

    def _lookup_server_loop(self) -> None:
        """Handle lookup requests from the scheduler."""
        assert self._lookup_server_socket is not None

        while not self._lookup_stop_event.is_set():
            try:
                message = self._lookup_server_socket.recv()
            except zmq.error.ZMQError:
                if self._lookup_stop_event.is_set():
                    break
                continue

            hit_blocks = 0
            try:
                # Directly deserialize block_hashes list for faster deserialization
                block_hashes = msgpack.unpackb(message)
                hit_blocks = self._count_available_block_prefix(block_hashes)
            except Exception:
                hit_blocks = 0

            # Directly serialize hit_blocks int for faster serialization
            reply = msgpack.packb(hit_blocks)
            try:
                self._lookup_server_socket.send(reply)
            except zmq.error.ZMQError:
                if self._lookup_stop_event.is_set():
                    break

    def _count_available_block_prefix(self, block_hashes: List[bytes]) -> int:
        """Return length of contiguous prefix available in CPU storage."""
        if not block_hashes:
            return 0

        try:
            response = self._send_engine_request('QUERY', {
                'block_hashes': block_hashes
            })
            return response.get('hit_blocks', 0)
        except Exception:
            return 0

    def _ensure_lookup_client(self) -> None:
        if self._lookup_client is not None:
            return
        if self._lookup_context is None:
            self._lookup_context = zmq.Context()
        self._lookup_client = self._lookup_context.socket(zmq.REQ)
        # Avoid hanging forever if worker is not reachable
        self._lookup_client.setsockopt(zmq.RCVTIMEO, 2000)
        self._lookup_client.setsockopt(zmq.SNDTIMEO, 2000)
        self._lookup_client.connect(self._lookup_endpoint)

    def _send_lookup_request(self, req_id: str, block_hashes: List[bytes]) -> int:
        """Query worker for contiguous cached prefix length (in blocks)."""
        if not block_hashes:
            return 0

        try:
            self._ensure_lookup_client()
        except Exception:
            return 0

        # Directly serialize block_hashes list for faster serialization
        payload = msgpack.packb(block_hashes)

        lookup_start = time.perf_counter()
        try:
            assert self._lookup_client is not None
            self._lookup_client.send(payload)
            reply = self._lookup_client.recv()
        except (zmq.error.Again, zmq.error.ZMQError):
            return 0
        lookup_end = time.perf_counter()

        try:
            # Directly deserialize hit_blocks int for faster deserialization
            hit_blocks = int(msgpack.unpackb(reply))
        except Exception:
            return 0

        total_blocks = len(block_hashes)
        elapsed_us = (lookup_end - lookup_start) * 1e6
        logger.info(
            "[PegaKVConnector] scheduler_lookup req=%s hit_blocks=%d/%d (%.1f%%) cost=%.0f us",
            req_id,
            hit_blocks,
            total_blocks,
            (hit_blocks / total_blocks * 100) if total_blocks > 0 else 0.0,
            elapsed_us,
        )
        return hit_blocks

    def _stop_lookup_server(self) -> None:
        if self._lookup_server_thread is None:
            return
        self._lookup_stop_event.set()
        if self._lookup_server_socket is not None:
            try:
                self._lookup_server_socket.close(0)
            except Exception:
                pass
            self._lookup_server_socket = None
        self._lookup_server_thread.join(timeout=1.0)
        self._lookup_server_thread = None
        if self._lookup_context is not None:
            self._lookup_context.term()
            self._lookup_context = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors with the PegaEngine via IPC.

        Args:
            kv_caches: Dictionary mapping layer names to KV cache tensors
        """
        self._registered_layers = list(kv_caches.keys())

        for layer_name, kv_cache in kv_caches.items():
            if kv_cache.storage_offset() != 0:
                raise RuntimeError(
                    f"KV cache for {layer_name} must have zero storage offset"
                )

            # Create CUDA IPC wrapper for cross-process sharing
            wrapper = CudaIPCWrapper(kv_cache)
            wrapper_bytes = pickle.dumps(wrapper)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            # Detect KV cache layout:
            # - KV-first layout: shape = (2, num_blocks, block_size, num_heads, head_dim)
            #   where shape[0] = 2 for K and V
            # - Blocks-first layout: shape = (num_blocks, block_size, num_heads, head_dim)
            #   where shape[0] = num_blocks
            #
            # We detect this by checking if shape[0] == 2, which indicates KV-first layout.
            # In KV-first layout, the actual num_blocks is shape[1].
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

            if bytes_per_block == 0:
                raise RuntimeError(
                    f"Invalid bytes_per_block for {layer_name}: stride={stride}"
                )

            # Send to engine server for registration
            try:
                self._send_engine_request('REGISTER', {
                    'layer_name': layer_name,
                    'wrapper_bytes': wrapper_bytes,
                    'num_blocks': num_blocks,
                    'bytes_per_block': bytes_per_block,
                    'kv_stride_bytes': kv_stride_bytes,
                    'segments': segments,
                })
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register layer {layer_name} with engine: {e}"
                ) from e

        logger.info(
            "[PegaKVConnector] Registered %d KV cache layers (%s layout)",
            len(kv_caches),
            layout if kv_caches else "unknown",
        )

    def shutdown(self):
        """Shutdown the connector and unregister all KV caches."""
        self._stop_lookup_server()

        # Shutdown engine connection
        if self._engine_socket is not None:
            try:
                # Send shutdown command to engine (optional - engine can stay running)
                # self._send_engine_request('SHUTDOWN', {})
                self._engine_socket.close(0)
            except Exception:
                pass
            self._engine_socket = None

        if self._lookup_client is not None:
            try:
                self._lookup_client.close(0)
            except Exception:
                pass
            self._lookup_client = None

        if self._engine_context is not None:
            try:
                self._engine_context.term()
            except Exception:
                pass
            self._engine_context = None

        if self._lookup_context is not None:
            try:
                self._lookup_context.term()
            except Exception:
                pass
            self._lookup_context = None


__all__ = ["PegaKVConnector", "KVConnectorRole"]
