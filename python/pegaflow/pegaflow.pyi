"""Type stubs for the pegaflow Rust extension module (PyO3 bindings).

This module provides high-performance KV cache storage and gRPC client
for distributed LLM inference with vLLM.
"""

__version__: str

# Custom exceptions for error classification

class PegaFlowError(Exception):
    """Base exception for all PegaFlow errors."""

    ...

class PegaflowInternal(PegaFlowError):
    """Internal server error."""

    ...

class QueryLoading:
    def __init__(self) -> None: ...

class QueryReady:
    num_hit_blocks: int
    lease: bytes
    def __init__(self, num_hit_blocks: int, lease: bytes) -> None: ...

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
            PegaFlowError: If connection to the server fails.
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
        pp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
        transfer_backend: str,
        page_first: bool,
    ) -> tuple[bool, str]:
        """Register all KV cache layers on a GPU with a single RPC call.

        Workers declare only the layers that actually exist on the device;
        the engine derives the instance-wide layer-id space once all
        world_size workers have registered.

        Contract: device_id must be non-negative; tp_size and world_size
        must be non-zero; tp_rank must be less than tp_size; per-layer
        metadata lists must have the same non-zero length.

        Args:
            instance_id: Model instance ID.
            namespace: Namespace for model isolation.
            tp_rank: Tensor parallel rank.
            pp_rank: Pipeline parallel rank.
            tp_size: Total tensor parallel size.
            world_size: Total worker count (TP * PP * PCP).
            device_id: CUDA device ID.
            layer_names: List of layer names.
            wrapper_bytes_list: List of serialized CUDA IPC tensor wrappers.
            num_blocks_list: List of block counts per layer.
            bytes_per_block_list: List of block sizes per layer.
            kv_stride_bytes_list: List of K/V strides per layer.
            segments_list: List of segment counts per layer.
            transfer_backend: H2D/D2H backend for this instance's GPU worker
                pools, "direct" or "kernel". Chosen by the connector per model.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowError: If server is unavailable.
            ValueError: If request is invalid or transfer_backend is unknown.
        """
        ...

    def save(
        self,
        instance_id: str,
        tp_rank: int,
        pp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]:
        """Save KV blocks to the engine.

        Contract: device_id must be non-negative, and each save tuple must
        have matching block_ids and block_hashes lengths.

        Args:
            instance_id: Model instance ID.
            tp_rank: Tensor parallel rank.
            pp_rank: Pipeline parallel rank.
            device_id: CUDA device ID.
            saves: List of (layer_name, block_ids, block_hashes) tuples.
                Each tuple specifies blocks to save for one layer.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowError: If server is unavailable.
            ValueError: If request is invalid.
        """
        ...

    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names: list[str],
        loads: list[tuple[bytes, list[int]]],
    ) -> tuple[bool, str]:
        """Load KV blocks from the engine.

        Contract: device_id must be non-negative; each lease must be returned
        by query_prefetch; each lease's block count must match its destination
        block_ids count.

        Args:
            instance_id: Model instance ID.
            tp_rank: Tensor parallel rank.
            device_id: CUDA device ID.
            load_state_shm: Shared memory name from PyLoadState.shm_name().
            layer_names: List of layer names to load.
            loads: List of (lease, destination block IDs) pairs.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowError: If server is unavailable.
            ValueError: If request is invalid.
        """
        ...

    def query_prefetch(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
    ) -> QueryLoading | QueryReady:
        """Query prefix cache hits with SSD prefetch support.

        Checks memory cache and triggers SSD prefetch for missing blocks.
        Ready hits are owned by an opaque lease consumed by load or release.
        Contract: instance_id must be registered, req_id must be non-empty
        and stable across retries for the same request, and block_hashes may
        be empty.

        Args:
            instance_id: Model instance ID.
            block_hashes: List of block hashes to check.
            req_id: Request ID for tracking and prefetch correlation.

        Returns:
            QueryLoading while backing fetch is in progress, otherwise QueryReady.

        Raises:
            PegaFlowError: If server is unavailable.
            ValueError: If request is invalid.
        """
        ...

    def release(
        self,
        lease: bytes,
    ) -> None:
        """Release a query lease.

        Args:
            lease: Opaque lease returned by QueryReady.

        Raises:
            PegaFlowError: If server is unavailable.
        """
        ...

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        """Unregister a context/instance.

        Args:
            instance_id: Model instance ID to unregister.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowError: If server is unavailable.
            ValueError: If request is invalid.
        """
        ...

    def shutdown(self) -> tuple[bool, str]:
        """Shutdown the engine server.

        Returns:
            Tuple of (ok, message) indicating success/failure.

        Raises:
            PegaFlowError: If server is unavailable.
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
            PegaFlowError: If the server is unavailable.
            ValueError: If the request is rejected.
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

class PdRdmaEngine:
    def __init__(
        self,
        *,
        cuda_device: int = 0,
        numa_node: int | None = None,
        domains: list[str] | None = None,
        device: str = "cuda",
        pin_worker_cpu: int | None = None,
    ) -> None: ...
    def register_local_layers(self, layers: list[dict]) -> list[dict]: ...
    def register_remote(self, req_id: str, handshake_json: str) -> None: ...
    def push_layer(self, req_id: str, layer_idx: int, blocks: list[dict]) -> None: ...
    def wait_for_pushes(self, req_id: str) -> None: ...
    def push_done(self, req_id: str) -> None: ...
    def write_stats(self, req_id: str) -> dict: ...
    def fail_request(self, req_id: str) -> None: ...
    def abort_request(self, req_id: str) -> None: ...
    def wait_done(self, req_id: str) -> None: ...
    def pop_finished_sending(self) -> set[str]: ...
    def pop_finished_recving(self) -> set[str]: ...
    def close_request(self, req_id: str) -> None: ...
    def num_domains(self) -> int: ...
    def num_groups(self) -> int: ...
    def aggregated_link_speed(self) -> int: ...
