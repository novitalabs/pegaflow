use pegaflow_core::LoadState;
use pegaflow_proto::proto::engine::{
    GetPdReceiveDescriptorRequest, HealthRequest, LoadPdReceiveRequest, LoadRequest,
    PdReceiveDescriptorState, PreparePdReceiveRequest, QueryRequest, RdmaHandshakeRequest,
    RegisterContextRequest, ResponseStatus, SaveLayer, SaveRequest, SessionEvent, SessionRequest,
    ShutdownRequest, UnpinRequest, UnregisterRequest, engine_client::EngineClient,
};
use pegaflow_transfer::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
};
use std::{
    future::Future,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use tokio::runtime::{Handle, Runtime};
use tonic::{
    Code, Status as GrpcStatus, Streaming,
    transport::{Channel, Endpoint},
};

// Custom Python exceptions for error classification
create_exception!(pegaflow, PegaFlowError, PyException);
create_exception!(pegaflow, PegaFlowServiceError, PegaFlowError);
create_exception!(pegaflow, PegaFlowBusinessError, PegaFlowError);

static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get or create the global Tokio runtime (shared across all RPC calls)
fn get_runtime() -> PyResult<&'static Runtime> {
    // First check if already initialized (fast path)
    if let Some(rt) = TOKIO_RUNTIME.get() {
        return Ok(rt);
    }

    // Try to initialize - only one thread will succeed
    let rt = Runtime::new().map_err(runtime_creation_error)?;

    // Try to set it; if another thread beat us, that's fine - use theirs
    let _ = TOKIO_RUNTIME.set(rt);

    // Return whatever is now in the cell
    TOKIO_RUNTIME
        .get()
        .ok_or_else(|| PyRuntimeError::new_err("failed to initialize Tokio runtime"))
}

fn runtime_creation_error(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("failed to create Tokio runtime: {err}"))
}

fn transport_connect_error(endpoint: &str, err: impl std::fmt::Display) -> PyErr {
    PegaFlowServiceError::new_err(format!(
        "failed to connect to engine server at {endpoint}: {err}"
    ))
}

/// Classify gRPC status codes into service vs business errors.
///
/// Service errors (server unavailable, should trigger health check):
/// - UNAVAILABLE: Server not reachable
/// - DEADLINE_EXCEEDED: Request timed out
/// - INTERNAL: Server internal error
/// - ABORTED: Operation aborted
/// - CANCELLED: Operation cancelled
///
/// Business errors (application logic errors, should propagate):
/// - INVALID_ARGUMENT: Bad request parameters
/// - FAILED_PRECONDITION: State precondition not met
/// - NOT_FOUND: Resource not found
/// - All other codes
fn is_service_error(code: Code) -> bool {
    matches!(
        code,
        Code::Unavailable
            | Code::DeadlineExceeded
            | Code::Internal
            | Code::Aborted
            | Code::Cancelled
    )
}

fn rpc_status_error(method: &str, err: GrpcStatus) -> PyErr {
    let msg = format!("{method} RPC failed: {err}");
    if is_service_error(err.code()) {
        PegaFlowServiceError::new_err(msg)
    } else {
        PegaFlowBusinessError::new_err(msg)
    }
}

fn expect_status(method: &str, status: Option<ResponseStatus>) -> PyResult<ResponseStatus> {
    status.ok_or_else(|| PyRuntimeError::new_err(format!("{method} response missing status")))
}

fn status_tuple(method: &str, status: Option<ResponseStatus>) -> PyResult<(bool, String)> {
    let status = expect_status(method, status)?;
    Ok((status.ok, status.message))
}

fn u64_to_usize(value: u64, field: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyRuntimeError::new_err(format!("{field}={value} exceeds usize range")))
}

fn non_null_from_usize(ptr: usize, field: &str) -> PyResult<NonNull<u8>> {
    NonNull::new(ptr as *mut u8)
        .ok_or_else(|| PyRuntimeError::new_err(format!("{field} must not be null")))
}

#[pyclass]
struct EngineRpcClient {
    endpoint: String,
    client: EngineClient<Channel>,
    rt_handle: Handle,
    /// Holds the server-streaming Session response if `start_session_watcher`
    /// was called. We never poll it — the hyper connection driver keeps the
    /// underlying HTTP/2 connection alive on its own. Keeping the Streaming
    /// alive here (instead of dropping it) is what keeps the stream open on
    /// the wire, so server-side disconnect detection only fires when this
    /// process actually dies.
    session_stream: Mutex<Option<Streaming<SessionEvent>>>,
}

impl EngineRpcClient {
    /// Execute an RPC call with shared boilerplate:
    /// - get global runtime
    /// - clone channel
    /// - create client
    /// - block_on the async closure
    fn call<F, Fut, T>(&self, py: Python<'_>, method: &'static str, f: F) -> PyResult<T>
    where
        F: FnOnce(EngineClient<Channel>) -> Fut + Send,
        Fut: Future<Output = Result<T, GrpcStatus>> + Send,
        T: Send,
    {
        let rt_handle = self.rt_handle.clone();
        let client = self.client.clone();
        py.detach(move || {
            rt_handle
                .block_on(async move { f(client).await.map_err(|e| rpc_status_error(method, e)) })
        })
    }
}

#[pymethods]
impl EngineRpcClient {
    #[new]
    #[pyo3(signature = (endpoint = None))]
    fn new(py: Python<'_>, endpoint: Option<String>) -> PyResult<Self> {
        let endpoint = endpoint.unwrap_or_else(|| "http://127.0.0.1:50055".to_string());
        let rt = get_runtime()?;

        // Avoid per-RPC overhead by eager-connecting and reusing a warmed client handle.
        let endpoint_cfg = Endpoint::from_shared(endpoint.clone())
            .map_err(|err| transport_connect_error(&endpoint, err))?
            .connect_timeout(Duration::from_millis(500))
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_while_idle(true);

        let channel = py.detach({
            let endpoint = endpoint.clone();
            move || {
                rt.block_on(endpoint_cfg.connect())
                    .map_err(|err| transport_connect_error(&endpoint, err))
            }
        })?;
        // Match server's 64 MiB limit to avoid Status::resource_exhausted on large payloads
        const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
        let client = EngineClient::new(channel)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        let rt_handle = rt.handle().clone();

        Ok(Self {
            endpoint,
            client,
            rt_handle,
            session_stream: Mutex::new(None),
        })
    }

    /// Return the configured endpoint.
    fn endpoint(&self) -> String {
        self.endpoint.clone()
    }

    /// Check if the engine server is healthy.
    ///
    /// Returns: (ok: bool, message: str)
    fn health(&self, py: Python<'_>) -> PyResult<(bool, String)> {
        self.call(py, "health", |mut c| async move {
            let resp = c.health(HealthRequest {}).await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("health", r.status))
    }

    /// Register all KV cache layers on a GPU with a single RPC call.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     namespace: Namespace for block hash isolation
    ///     tp_rank: Tensor parallel rank of the worker
    ///     tp_size: Total Tensor Parallel size
    ///     world_size: Total worker count (TP * PP * PCP)
    ///     device_id: CUDA device ID of the worker
    ///     num_layers: Total number of layers in the model
    ///     wrapper_bytes_list: List of serialized CUDA tensor wrappers
    ///     num_blocks_list: List of block counts per layer
    ///     bytes_per_block_list: List of block sizes per layer
    ///     kv_stride_bytes_list: List of K/V strides per layer
    ///     segments_list: List of segment counts per layer
    ///
    /// Returns: (ok: bool, message: str)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (instance_id, namespace, tp_rank, tp_size, world_size, device_id, num_layers, layer_names, wrapper_bytes_list, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list))]
    fn register_context_batch(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_rank: u32,
        tp_size: u32,
        world_size: u32,
        device_id: i32,
        num_layers: u32,
        layer_names: Vec<String>,
        wrapper_bytes_list: Vec<Vec<u8>>,
        num_blocks_list: Vec<u64>,
        bytes_per_block_list: Vec<u64>,
        kv_stride_bytes_list: Vec<u64>,
        segments_list: Vec<u32>,
    ) -> PyResult<(bool, String)> {
        self.call(py, "register_context_batch", |mut c| async move {
            let resp = c
                .register_context_batch(RegisterContextRequest {
                    instance_id,
                    namespace,
                    tp_rank,
                    tp_size,
                    world_size,
                    device_id,
                    num_layers,
                    layer_names,
                    wrapper_bytes: wrapper_bytes_list,
                    num_blocks: num_blocks_list,
                    bytes_per_block: bytes_per_block_list,
                    kv_stride_bytes: kv_stride_bytes_list,
                    segments: segments_list,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("register_context_batch", r.status))
    }

    /// Save KV blocks to the engine.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     tp_rank: Tensor parallel rank
    ///     device_id: CUDA device ID
    ///     saves: List of (layer_name, block_ids, block_hashes) tuples
    ///
    /// Returns: (ok: bool, message: str)
    fn save(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<i32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String)> {
        let saves = saves
            .into_iter()
            .map(|(layer_name, block_ids, block_hashes)| SaveLayer {
                layer_name,
                block_ids,
                block_hashes,
            })
            .collect();
        self.call(py, "save", |mut c| async move {
            let resp = c
                .save(SaveRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    saves,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("save", r.status))
    }

    /// Load KV blocks from the engine.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     tp_rank: Tensor parallel rank
    ///     device_id: CUDA device ID
    ///     load_state_shm: Shared memory name for load state sync
    ///     layer_names: List of layer names to load
    ///     block_ids: GPU block IDs to load into
    ///     block_hashes: Content hashes for blocks
    ///
    /// Returns: (ok: bool, message: str)
    #[allow(clippy::too_many_arguments)]
    fn load(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        load_state_shm: String,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<(bool, String)> {
        self.call(py, "load", |mut c| async move {
            let resp = c
                .load(LoadRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    load_state_shm,
                    layer_names,
                    block_ids,
                    block_hashes,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("load", r.status))
    }

    /// Load KV blocks from a D-side P/D CPU-staging receive lease.
    ///
    /// Args are the normal load destination plus P/D rendezvous identity.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (instance_id, tp_rank, device_id, load_state_shm, layer_names, block_ids, block_hashes, request_id, handle = None, receive_rank = -1))]
    fn load_pd_receive(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        load_state_shm: String,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        request_id: String,
        handle: Option<String>,
        receive_rank: i32,
    ) -> PyResult<(bool, String)> {
        self.call(py, "load_pd_receive", |mut c| async move {
            let resp = c
                .load_pd_receive(LoadPdReceiveRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    load_state_shm,
                    layer_names,
                    block_ids,
                    block_hashes,
                    request_id,
                    handle: handle.unwrap_or_default(),
                    receive_rank,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("load_pd_receive", r.status))
    }

    /// Query prefix cache hits with SSD prefetch support.
    ///
    /// Checks memory cache and triggers SSD prefetch for missing blocks.
    /// Pins hit blocks for subsequent load operations.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     block_hashes: List of block hashes to check
    ///
    /// Returns: dict with keys:
    ///     - ok: bool - whether the request succeeded
    ///     - message: str - error message if failed
    ///     - hit_blocks: int - number of blocks ready in cache
    ///     - prefetch_state: str - one of "done", "loading"
    ///     - loading_blocks: int - number of blocks being prefetched
    ///     - missing_blocks: int - number of blocks not found
    fn query_prefetch(
        &self,
        py: Python<'_>,
        instance_id: String,
        block_hashes: Vec<Vec<u8>>,
        req_id: String,
    ) -> PyResult<Py<pyo3::types::PyAny>> {
        use pegaflow_proto::proto::engine::PrefetchState;

        self.call(py, "query_prefetch", |mut c| async move {
            let resp = c
                .query_prefetch(QueryRequest {
                    instance_id,
                    block_hashes,
                    req_id,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| {
            let (ok, msg) = status_tuple("query_prefetch", r.status)?;
            let hit = u64_to_usize(r.hit_blocks, "hit_blocks")?;
            let loading = u64_to_usize(r.loading_blocks, "loading_blocks")?;
            let missing = u64_to_usize(r.missing_blocks, "missing_blocks")?;

            let prefetch_state = match PrefetchState::try_from(r.prefetch_state) {
                Ok(PrefetchState::PrefetchDone) => "done",
                Ok(PrefetchState::PrefetchLoading) => "loading",
                _ => "done", // Default to done for unknown states
            };

            Python::attach(|py| {
                use pyo3::types::PyDict;
                let dict = PyDict::new(py);
                dict.set_item("ok", ok)?;
                dict.set_item("message", msg)?;
                dict.set_item("hit_blocks", hit)?;
                dict.set_item("prefetch_state", prefetch_state)?;
                dict.set_item("loading_blocks", loading)?;
                dict.set_item("missing_blocks", missing)?;
                Ok(dict.into())
            })
        })
    }

    /// Prepare a D-side CPU-staging lease for P/D push.
    ///
    /// Returns: dict with keys:
    ///     - ok: bool
    ///     - message: str
    ///     - handle: str
    ///     - imm_data: int
    ///     - expires_at_ms: int
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (instance_id, request_id, block_hashes, num_blocks, expected_imm_count = 0, expire_after_ms = 0))]
    fn prepare_pd_receive(
        &self,
        py: Python<'_>,
        instance_id: String,
        request_id: String,
        block_hashes: Vec<Vec<u8>>,
        num_blocks: u64,
        expected_imm_count: u32,
        expire_after_ms: u64,
    ) -> PyResult<Py<pyo3::types::PyAny>> {
        self.call(py, "prepare_pd_receive", |mut c| async move {
            let resp = c
                .prepare_pd_receive(PreparePdReceiveRequest {
                    instance_id,
                    request_id,
                    block_hashes,
                    num_blocks,
                    expected_imm_count,
                    expire_after_ms,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| {
            let (ok, msg) = status_tuple("prepare_pd_receive", r.status)?;
            Python::attach(|py| {
                use pyo3::types::PyDict;
                let dict = PyDict::new(py);
                dict.set_item("ok", ok)?;
                dict.set_item("message", msg)?;
                dict.set_item("handle", r.handle)?;
                dict.set_item("imm_data", r.imm_data)?;
                dict.set_item("expires_at_ms", r.expires_at_ms)?;
                Ok(dict.into())
            })
        })
    }

    /// Fetch a D-side P/D receive descriptor.
    ///
    /// P-side in-process push uses this against D PegaFlow after prefill.
    ///
    /// Returns: dict with keys:
    ///     - ok: bool
    ///     - message: str
    ///     - state: str ("pending", "ready", "failed", "expired")
    ///     - handle: str
    ///     - slabs: list[dict]
    ///     - layers: list[dict]
    ///     - block_hashes: list[bytes]
    ///     - imm_data: int
    ///     - expires_at_ms: int
    ///     - data_ready: bool
    #[pyo3(signature = (dst_instance_id, request_id, receive_rank = -1, handle = None))]
    fn get_pd_receive_descriptor(
        &self,
        py: Python<'_>,
        dst_instance_id: String,
        request_id: String,
        receive_rank: i32,
        handle: Option<String>,
    ) -> PyResult<Py<pyo3::types::PyAny>> {
        self.call(py, "get_pd_receive_descriptor", |mut c| async move {
            let resp = c
                .get_pd_receive_descriptor(GetPdReceiveDescriptorRequest {
                    dst_instance_id,
                    request_id,
                    handle: handle.unwrap_or_default(),
                    receive_rank,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| {
            let (ok, msg) = status_tuple("get_pd_receive_descriptor", r.status)?;
            Python::attach(|py| {
                use pyo3::types::{PyBytes, PyDict, PyList};

                let dict = PyDict::new(py);
                dict.set_item("ok", ok)?;
                dict.set_item("message", msg)?;
                let state = match PdReceiveDescriptorState::try_from(r.state) {
                    Ok(PdReceiveDescriptorState::PdDescriptorPending) => "pending",
                    Ok(PdReceiveDescriptorState::PdDescriptorReady) => "ready",
                    Ok(PdReceiveDescriptorState::PdDescriptorFailed) => "failed",
                    Ok(PdReceiveDescriptorState::PdDescriptorExpired) => "expired",
                    _ => "pending",
                };
                dict.set_item("state", state)?;
                dict.set_item("handle", r.handle)?;
                dict.set_item("imm_data", r.imm_data)?;
                dict.set_item("expires_at_ms", r.expires_at_ms)?;
                dict.set_item("data_ready", r.data_ready)?;

                let ranks = PyList::empty(py);
                for rank in r.ranks {
                    let item = PyDict::new(py);
                    item.set_item("receive_rank", rank.receive_rank)?;
                    item.set_item("device_id", rank.device_id)?;
                    item.set_item("tp_rank", rank.tp_rank)?;
                    item.set_item("slab_index", rank.slab_index)?;
                    item.set_item("numa_node", rank.numa_node)?;
                    ranks.append(item)?;
                }
                dict.set_item("ranks", ranks)?;

                let slabs = PyList::empty(py);
                for slab in r.slabs {
                    let item = PyDict::new(py);
                    item.set_item("base_ptr", slab.base_ptr)?;
                    item.set_item("size", slab.size)?;
                    item.set_item("numa_node", slab.numa_node)?;
                    slabs.append(item)?;
                }
                dict.set_item("slabs", slabs)?;

                let layers = PyList::empty(py);
                for layer in r.layers {
                    let item = PyDict::new(py);
                    item.set_item("layer_name", layer.layer_name)?;
                    item.set_item("slab_index", layer.slab_index)?;
                    item.set_item("layer_offset", layer.layer_offset)?;
                    item.set_item("block_stride", layer.block_stride)?;
                    item.set_item("segment_count", layer.segment_count)?;
                    item.set_item("segment_size", layer.segment_size)?;
                    item.set_item("padded_segment_stride", layer.padded_segment_stride)?;
                    item.set_item("num_blocks", layer.num_blocks)?;
                    item.set_item("slot_id", layer.slot_id)?;
                    item.set_item("receive_rank", layer.receive_rank)?;
                    layers.append(item)?;
                }
                dict.set_item("layers", layers)?;

                let block_hashes = PyList::empty(py);
                for hash in r.block_hashes {
                    block_hashes.append(PyBytes::new(py, &hash))?;
                }
                dict.set_item("block_hashes", block_hashes)?;

                Ok(dict.into())
            })
        })
    }

    /// Unpin blocks that were pinned during query.
    ///
    /// This is used when load is cancelled or preempted before consumption.
    /// Call this to release pinned blocks and prevent memory leaks.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     block_hashes: List of block hashes to unpin
    ///
    /// Returns: (ok: bool, message: str)
    fn unpin(
        &self,
        py: Python<'_>,
        instance_id: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<(bool, String)> {
        self.call(py, "unpin", |mut c| async move {
            let resp = c
                .unpin(UnpinRequest {
                    instance_id,
                    block_hashes,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("unpin", r.status))
    }

    /// Unregister a context/instance.
    ///
    /// Args:
    ///     instance_id: Model instance ID to unregister
    ///
    /// Returns: (ok: bool, message: str)
    fn unregister_context(&self, py: Python<'_>, instance_id: String) -> PyResult<(bool, String)> {
        self.call(py, "unregister_context", |mut c| async move {
            let resp = c
                .unregister_context(UnregisterRequest { instance_id })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("unregister_context", r.status))
    }

    /// Shutdown the engine server.
    ///
    /// Returns: (ok: bool, message: str)
    fn shutdown(&self, py: Python<'_>) -> PyResult<(bool, String)> {
        self.call(py, "shutdown", |mut c| async move {
            let resp = c.shutdown(ShutdownRequest {}).await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("shutdown", r.status))
    }

    /// Open a liveness session with the engine server.
    ///
    /// Sends one `Session` RPC and retains the resulting server-streaming
    /// response. No polling — the hyper connection driver keeps the HTTP/2
    /// connection alive. When this process dies, the kernel closes the TCP
    /// socket; the server observes disconnect and auto-releases CUDA IPC
    /// mappings for `instance_id`.
    ///
    /// Call once per client, typically from the scheduler role. Calling
    /// again replaces the previous stream (the old Streaming drops and the
    /// server's old session is superseded by the new one).
    #[pyo3(signature = (instance_id, namespace, tp_size, world_size))]
    fn start_session_watcher(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> PyResult<()> {
        let mut client = self.client.clone();
        let req = SessionRequest {
            instance_id,
            namespace,
            tp_size,
            world_size,
        };
        let stream = self.call(py, "session", move |_| async move {
            let resp = client.session(req).await?;
            Ok(resp.into_inner())
        })?;
        *self
            .session_stream
            .lock()
            .map_err(|_| PyRuntimeError::new_err("session_stream mutex poisoned"))? = Some(stream);
        Ok(())
    }
}

/// Python wrapper for LoadState (batch-level sync for async KV cache loading)
///
/// Created by connector worker before starting a load batch.
/// Pass shm_name() to the server, then poll via get()/is_ready() for completion.
///
/// State values:
/// - 0: pending (load in progress)
/// - 1: success (all transfers complete)
/// - <0: error (transfer failed)
#[pyclass]
struct PyLoadState {
    inner: Arc<LoadState>,
}

#[pymethods]
impl PyLoadState {
    /// Create a new LoadState (creates shared memory, initializes to PENDING).
    #[new]
    fn new() -> PyResult<Self> {
        let inner = LoadState::new()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create LoadState: {e}")))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Get the shared memory name to pass to the server.
    fn shm_name(&self) -> String {
        self.inner.shm_name().to_string()
    }

    /// Get current state value (non-blocking).
    ///
    /// Returns: 0=pending, 1=success, <0=error
    fn get_state(&self) -> i64 {
        self.inner.get()
    }

    /// Check if load is complete (non-blocking).
    ///
    /// Returns True if state is non-zero (completed or error).
    fn is_ready(&self) -> bool {
        self.inner.get() != 0
    }
}

/// In-process P-side runtime for outbound KV transfer.
///
/// This object lives in the vLLM worker process. It owns the transfer engine
/// used by future P/D D2H -> RDMA WRITE -> WRITE_WITH_IMM tasks.
#[pyclass]
struct KvEgressRuntime {
    transfer: Arc<TransferEngine>,
    rt_handle: Handle,
}

#[pymethods]
impl KvEgressRuntime {
    /// Create a KV egress runtime with the selected RDMA NIC names.
    #[new]
    fn new(py: Python<'_>, nic_names: Vec<String>) -> PyResult<Self> {
        if nic_names.is_empty() {
            return Err(PyRuntimeError::new_err(
                "KvEgressRuntime requires at least one RDMA NIC",
            ));
        }
        let rt = get_runtime()?;
        let transfer = py.detach({
            let nic_names = nic_names.clone();
            move || {
                TransferEngine::new(&nic_names)
                    .map_err(|err| PyRuntimeError::new_err(format!("create TransferEngine: {err}")))
            }
        })?;
        Ok(Self {
            transfer: Arc::new(transfer),
            rt_handle: rt.handle().clone(),
        })
    }

    /// Register local CPU/pinned memory for future RDMA operations.
    fn _register_memory(&self, py: Python<'_>, ptr: usize, len: usize) -> PyResult<()> {
        if len == 0 {
            return Err(PyRuntimeError::new_err("len must be non-zero"));
        }
        let transfer = Arc::clone(&self.transfer);
        py.detach(move || {
            let ptr = non_null_from_usize(ptr, "ptr")?;
            transfer
                .register_memory(&[MemoryRegion { ptr, len }])
                .map_err(|err| PyRuntimeError::new_err(format!("register_memory: {err}")))
        })
    }

    fn _unregister_memory(&self, py: Python<'_>, ptr: usize) -> PyResult<()> {
        let transfer = Arc::clone(&self.transfer);
        py.detach(move || {
            let ptr = non_null_from_usize(ptr, "ptr")?;
            transfer
                .unregister_memory(&[ptr])
                .map_err(|err| PyRuntimeError::new_err(format!("unregister_memory: {err}")))
        })
    }

    /// Establish an RDMA connection to D PegaFlow through its existing
    /// RdmaHandshake RPC. `remote_addr` is the transfer-layer peer key and
    /// should match the D PegaFlow endpoint key used by the descriptor path.
    fn _ensure_connected(
        &self,
        py: Python<'_>,
        remote_addr: String,
        requester_id: String,
        engine_client: PyRef<'_, EngineRpcClient>,
    ) -> PyResult<()> {
        let transfer = Arc::clone(&self.transfer);
        let mut client = engine_client.client.clone();
        let rt_handle = self.rt_handle.clone();

        py.detach(move || {
            rt_handle.block_on(async move {
                let local_meta = match transfer.get_or_prepare(&remote_addr) {
                    Ok(ConnectionStatus::Existing) => return Ok(()),
                    Ok(ConnectionStatus::Connecting) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "RDMA handshake to {remote_addr} already in progress"
                        )));
                    }
                    Ok(ConnectionStatus::Prepared(meta)) => meta,
                    Err(err) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "RDMA get_or_prepare({remote_addr}): {err}"
                        )));
                    }
                };

                let response = match client
                    .rdma_handshake(RdmaHandshakeRequest {
                        requester_id,
                        handshake_metadata: local_meta.to_bytes(),
                    })
                    .await
                {
                    Ok(resp) => resp.into_inner(),
                    Err(err) => {
                        transfer.abort_handshake(&remote_addr, &local_meta);
                        return Err(rpc_status_error("rdma_handshake", err));
                    }
                };

                let (ok, message) = status_tuple("rdma_handshake", response.status)?;
                if !ok {
                    transfer.abort_handshake(&remote_addr, &local_meta);
                    return Err(PegaFlowBusinessError::new_err(format!(
                        "rdma_handshake rejected: {message}"
                    )));
                }

                let remote_meta = HandshakeMetadata::from_bytes(&response.handshake_metadata)
                    .map_err(|err| {
                        transfer.abort_handshake(&remote_addr, &local_meta);
                        PyRuntimeError::new_err(format!("invalid remote handshake metadata: {err}"))
                    })?;
                transfer
                    .complete_handshake(&remote_addr, &local_meta, &remote_meta)
                    .map_err(|err| {
                        transfer.abort_handshake(&remote_addr, &local_meta);
                        PyRuntimeError::new_err(format!("complete_handshake: {err}"))
                    })?;
                Ok(())
            })
        })
    }

    /// RDMA WRITE from already-registered local memory to already-registered
    /// remote memory. This is the low-level primitive the P/D push task will
    /// use after D2H staging and descriptor resolution.
    fn _write_registered(
        &self,
        py: Python<'_>,
        remote_addr: String,
        descs: Vec<(usize, usize, usize)>,
    ) -> PyResult<usize> {
        if descs.is_empty() {
            return Ok(0);
        }
        for (_, _, len) in &descs {
            if *len == 0 {
                return Err(PyRuntimeError::new_err("transfer len must be non-zero"));
            }
        }

        let transfer = Arc::clone(&self.transfer);
        let rt_handle = self.rt_handle.clone();
        py.detach(move || {
            let mut transfer_descs = Vec::with_capacity(descs.len());
            for (local_ptr, remote_ptr, len) in descs {
                transfer_descs.push(TransferDesc {
                    local_ptr: non_null_from_usize(local_ptr, "local_ptr")?,
                    remote_ptr: non_null_from_usize(remote_ptr, "remote_ptr")?,
                    len,
                });
            }

            let receivers = transfer
                .batch_transfer_async(TransferOp::Write, &remote_addr, &transfer_descs)
                .map_err(|err| PyRuntimeError::new_err(format!("RDMA write submit: {err}")))?;
            rt_handle.block_on(async move {
                let mut total = 0usize;
                for rx in receivers {
                    let bytes = rx
                        .await
                        .map_err(|err| {
                            PyRuntimeError::new_err(format!("RDMA write dropped: {err}"))
                        })?
                        .map_err(|err| {
                            PyRuntimeError::new_err(format!("RDMA write failed: {err}"))
                        })?;
                    total = total.saturating_add(bytes);
                }
                Ok(total)
            })
        })
    }

    /// Send the final RDMA WRITE-with-immediate readiness signal and wait for
    /// local send completions.
    fn _write_imm(&self, py: Python<'_>, remote_addr: String, imm_data: u32) -> PyResult<usize> {
        let transfer = Arc::clone(&self.transfer);
        let rt_handle = self.rt_handle.clone();
        py.detach(move || {
            let receivers = transfer
                .write_imm_async(&remote_addr, imm_data)
                .map_err(|err| PyRuntimeError::new_err(format!("WRITE_WITH_IMM submit: {err}")))?;
            rt_handle.block_on(async move {
                let mut total = 0usize;
                for rx in receivers {
                    let bytes = rx
                        .await
                        .map_err(|err| {
                            PyRuntimeError::new_err(format!("WRITE_WITH_IMM dropped: {err}"))
                        })?
                        .map_err(|err| {
                            PyRuntimeError::new_err(format!("WRITE_WITH_IMM failed: {err}"))
                        })?;
                    total = total.saturating_add(bytes);
                }
                Ok(total)
            })
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pegaflow_common::logging::init_stderr("info,pegaflow_core=info");
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<EngineRpcClient>()?;
    m.add_class::<PyLoadState>()?;
    m.add_class::<KvEgressRuntime>()?;
    // Register custom exceptions for error classification
    m.add("PegaFlowError", m.py().get_type::<PegaFlowError>())?;
    m.add(
        "PegaFlowServiceError",
        m.py().get_type::<PegaFlowServiceError>(),
    )?;
    m.add(
        "PegaFlowBusinessError",
        m.py().get_type::<PegaFlowBusinessError>(),
    )?;

    Ok(())
}
