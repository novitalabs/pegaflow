"""Type stubs for the pegaflow Rust extension module (PyO3 bindings).

This module provides high-performance KV cache storage and gRPC client
for distributed LLM inference with vLLM and SGLang.
"""

from typing import Any

__version__: str

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

    def register_context_batch(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        num_layers: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
    ) -> tuple[bool, str]:
        """Register all KV cache layers on a GPU with a single RPC call.

        Args:
            instance_id: Model instance ID.
            namespace: Namespace for model isolation.
            tp_rank: Tensor parallel rank.
            tp_size: Total tensor parallel size.
            world_size: Total worker count (TP * PP * PCP).
            device_id: CUDA device ID.
            num_layers: Number of model layers.
            layer_names: List of layer names.
            wrapper_bytes_list: List of serialized CUDA IPC tensor wrappers.
            num_blocks_list: List of block counts per layer.
            bytes_per_block_list: List of block sizes per layer.
            kv_stride_bytes_list: List of K/V strides per layer.
            segments_list: List of segment counts per layer.

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

    def load_pd_receive(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names: list[str],
        block_ids: list[int],
        block_hashes: list[bytes],
        request_id: str,
        handle: str | None = None,
        receive_rank: int = -1,
    ) -> tuple[bool, str]:
        """Load KV blocks from a D-side P/D CPU-staging receive lease."""
        ...

    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
    ) -> dict[str, Any]:
        """Query prefix cache hits with SSD prefetch support.

        Checks memory cache and triggers SSD prefetch for missing blocks.
        Pins hit blocks for subsequent load operations.

        Args:
            instance_id: Model instance ID.
            block_hashes: List of block hashes to check.
            req_id: Request ID for tracking and prefetch correlation.

        Returns:
            Dict with keys:
                - ok (bool): Whether the request succeeded.
                - message (str): Error message if failed.
                - hit_blocks (int): Number of blocks ready in cache.
                - prefetch_state (str): One of "done", "loading".
                - loading_blocks (int): Number of blocks being prefetched from SSD.
                - missing_blocks (int): Number of blocks not found anywhere.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def prepare_pd_receive(
        self,
        instance_id: str,
        request_id: str,
        block_hashes: list[bytes],
        num_blocks: int,
        expected_imm_count: int = 0,
        expire_after_ms: int = 0,
    ) -> dict[str, Any]:
        """Prepare a D-side CPU-staging lease for P/D push.

        Args:
            instance_id: D-side model instance ID.
            request_id: Stable P/D rendezvous request ID.
            block_hashes: Optional block hashes for the external KV span.
            num_blocks: Number of blocks to stage.
            expected_imm_count: Number of expected WRITE_WITH_IMM completions;
                0 lets D derive receive-rank count times local NIC fanout.
            expire_after_ms: Lease TTL override; 0 uses server default.

        Returns:
            Dict with keys ok, message, handle, imm_data, expires_at_ms.

        Raises:
            PegaFlowServiceError: If server is unavailable.
            PegaFlowBusinessError: If request is invalid.
        """
        ...

    def get_pd_receive_descriptor(
        self,
        dst_instance_id: str,
        request_id: str,
        receive_rank: int = -1,
        handle: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a D-side P/D receive descriptor.

        Returns a dict with state, slabs, layers, block_hashes, imm_data,
        expires_at_ms, and data_ready. State is one of "pending", "ready",
        "failed", "expired"; data_ready flips after WRITE_WITH_IMM completion.
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

    def start_session_watcher(
        self,
        instance_id: str,
        namespace: str,
        tp_size: int,
        world_size: int,
    ) -> None:
        """Open a liveness Session stream to the engine server.

        The client holds a server-streaming RPC open for the lifetime of
        this process. When the process dies, the kernel closes the TCP
        socket; the server observes disconnect and auto-releases the
        instance's CUDA IPC mappings. No polling on the Python side — the
        hyper connection driver keeps the underlying HTTP/2 connection
        alive on its own.

        Call once, typically from the scheduler role. Calling again
        replaces the previous stream; the server supersedes the prior
        session and the old one becomes a no-op on close.

        Raises:
            PegaFlowServiceError: If the server is unavailable.
            PegaFlowBusinessError: If the request is rejected.
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


class KvEgressRuntime:
    """In-process P-side runtime for outbound KV transfer."""

    def __init__(self, nic_names: list[str]) -> None:
        """Create a runtime with selected RDMA NIC names."""
        ...
