"""Type stubs for the pegaflow Rust extension module (PyO3 bindings).

This module provides high-performance KV cache storage and gRPC client
for distributed LLM inference with vLLM and SGLang.
"""

from typing import Any

# Custom exceptions for error classification

class PegaFlowError(Exception):
    """Base exception for all PegaFlow errors."""

    ...

class PegaFlowServiceError(PegaFlowError):
    """Service errors indicating server unavailability.

    These errors should trigger health checks and retry logic.
    Includes: UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL, ABORTED, CANCELLED.
    """

    ...

class PegaFlowBusinessError(PegaFlowError):
    """Business logic errors from the application layer.

    These errors indicate invalid requests or state and should be propagated.
    Includes: INVALID_ARGUMENT, FAILED_PRECONDITION, NOT_FOUND.
    """

    ...

class PegaEngine:
    """Local in-process KV cache storage engine.

    Manages GPU workers and KV cache storage with pinned memory.
    Use this for single-process deployments without gRPC overhead.
    """

    def __init__(self) -> None:
        """Create a new PegaEngine instance.

        Initializes the Rust logger and creates the core engine.
        """
        ...

    def register_context_layer(
        self,
        instance_id: str,
        namespace: str,
        device_id: int,
        layer_name: str,
        data_ptr: int,
        size_bytes: int,
        num_blocks: int,
        bytes_per_block: int,
        kv_stride_bytes: int,
        segments: int,
        tp_rank: int,
        tp_size: int,
        world_size: int,
        num_layers: int,
    ) -> None:
        """Register a context layer buffer with layout metadata.

        Args:
            instance_id: ID of the model instance.
            namespace: Namespace for model isolation.
            device_id: CUDA device ID.
            layer_name: Name of the layer (e.g., "layer_0").
            data_ptr: GPU data pointer as integer.
            size_bytes: Total size of the tensor in bytes.
            num_blocks: Total number of paged blocks for this layer.
            bytes_per_block: Size of each paged block in bytes.
            kv_stride_bytes: Byte stride between K and V (KV-first layout).
            segments: Number of segments per block (1=blocks-first, 2=KV-first).
            tp_rank: Tensor Parallel rank of the worker.
            tp_size: Total Tensor Parallel size.
            world_size: Total worker count (TP * PP * PCP).
            num_layers: Total number of layers in the model.

        Raises:
            RuntimeError: If registration fails.
        """
        ...

    def unregister_instance(self, instance_id: str) -> None:
        """Unregister the active inference context/instance.

        Args:
            instance_id: ID of the model instance to unregister.

        Raises:
            RuntimeError: If unregistration fails.
        """
        ...

    def batch_load_kv_blocks(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names: list[str],
        block_ids: list[int],
        block_hashes: list[bytes],
    ) -> None:
        """Batch load KV blocks for multiple layers.

        More efficient than individual loads as it avoids Python overhead
        and reduces hash table lookups from O(layers × blocks) to O(blocks).

        Args:
            instance_id: ID of the model instance.
            tp_rank: Tensor Parallel rank of the worker.
            device_id: CUDA device ID.
            load_state_shm: Shared memory name from PyLoadState.shm_name().
            layer_names: List of layer names to load.
            block_ids: GPU block IDs to load into.
            block_hashes: Content hashes for each block.

        Raises:
            RuntimeError: If loading fails.
        """
        ...

class EngineRpcClient:
    """gRPC client for remote PegaEngine server communication.

    Provides async RPC methods for KV cache operations including
    registration, save, load, and query with automatic retry support.
    """

    def __init__(self, endpoint: str | None = None) -> None:
        """Create a new gRPC client.

        Args:
            endpoint: gRPC server endpoint (default: "http://127.0.0.1:50055").

        Raises:
            PegaFlowServiceError: If connection to the server fails.
        """
        ...

    def endpoint(self) -> str:
        """Return the configured gRPC endpoint.

        Returns:
            The endpoint URL string.
        """
        ...

    def health(self) -> tuple[bool, str]:
        """Check if the engine server is healthy.

        Returns:
            Tuple of (ok, message) where ok indicates health status.
        """
        ...

    def register_context(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        num_layers: int,
        layer_name: str,
        wrapper_bytes: bytes,
        num_blocks: int,
        bytes_per_block: int,
        kv_stride_bytes: int,
        segments: int,
    ) -> tuple[bool, str]:
        """Register a context layer for KV cache operations.

        Args:
            instance_id: Model instance ID.
            namespace: Namespace for model isolation.
            tp_rank: Tensor parallel rank.
            tp_size: Total tensor parallel size.
            world_size: Total worker count (TP * PP * PCP).
            device_id: CUDA device ID.
            num_layers: Number of model layers.
            layer_name: Name of this layer.
            wrapper_bytes: Serialized CUDA IPC tensor wrapper (pickle bytes).
            num_blocks: Number of KV blocks.
            bytes_per_block: Size of each block in bytes.
            kv_stride_bytes: Stride between K and V segments.
            segments: Number of segments per block.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def save(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]:
        """Save KV blocks to the engine.

        Args:
            instance_id: Model instance ID.
            tp_rank: Tensor parallel rank.
            device_id: CUDA device ID.
            saves: List of (layer_name, block_ids, block_hashes) tuples.
                Each tuple specifies blocks to save for one layer.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names: list[str],
        block_ids: list[int],
        block_hashes: list[bytes],
    ) -> tuple[bool, str]:
        """Load KV blocks from the engine.

        Args:
            instance_id: Model instance ID.
            tp_rank: Tensor parallel rank.
            device_id: CUDA device ID.
            load_state_shm: Shared memory name from PyLoadState.shm_name().
            layer_names: List of layer names to load.
            block_ids: GPU block IDs to load into.
            block_hashes: Content hashes for blocks.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def query(
        self,
        instance_id: str,
        block_hashes: list[bytes],
    ) -> dict[str, Any]:
        """Query prefix cache hits with prefetch support.

        Args:
            instance_id: Model instance ID.
            block_hashes: List of block hashes to check.

        Returns:
            Dict with keys:
                - ok (bool): Whether the request succeeded.
                - message (str): Error message if failed.
                - hit_blocks (int): Number of blocks ready in cache.
                - prefetch_state (str): One of "ready", "loading", "done".
                - loading_blocks (int): Number of blocks being prefetched.
                - missing_blocks (int): Number of blocks not found.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def unpin(
        self,
        instance_id: str,
        block_hashes: list[bytes],
    ) -> tuple[bool, str]:
        """Unpin blocks that were pinned during query.

        Call this when load is cancelled or preempted before consumption
        to release pinned blocks and prevent memory leaks.

        Args:
            instance_id: Model instance ID.
            block_hashes: List of block hashes to unpin.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        """Unregister a context/instance.

        Args:
            instance_id: Model instance ID to unregister.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def shutdown(self) -> tuple[bool, str]:
        """Shutdown the engine server.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowServiceError: If server is unavailable.
        """
        ...

class PyLoadState:
    """Batch-level synchronization for async KV cache loading via shared memory.

    Created by connector worker before starting a load batch.
    Pass shm_name() to the server, then poll via get_state()/is_ready().

    State values:
        - 0: pending (load in progress)
        - 1: success (all transfers complete)
        - <0: error (transfer failed)
    """

    def __init__(self) -> None:
        """Create a new LoadState with shared memory.

        Initializes the state to PENDING (0).

        Raises:
            RuntimeError: If shared memory creation fails.
        """
        ...

    def shm_name(self) -> str:
        """Get the shared memory name to pass to the server.

        Returns:
            The shared memory identifier string.
        """
        ...

    def get_state(self) -> int:
        """Get current state value (non-blocking).

        Returns:
            0 for pending, 1 for success, negative for error.
        """
        ...

    def is_ready(self) -> bool:
        """Check if load is complete (non-blocking).

        Returns:
            True if state is non-zero (completed or error).
        """
        ...
