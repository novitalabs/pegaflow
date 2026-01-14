use pegaflow_core::{LoadState, PegaEngine as CoreEngine};
use pegaflow_proto::proto::engine::{
    engine_client::EngineClient, HealthRequest, LoadRequest, QueryRequest, RegisterContextRequest,
    ResponseStatus, SaveLayer, SaveRequest, ShutdownRequest, UnregisterRequest,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
};
use std::{
    future::Future,
    sync::{Arc, Once, OnceLock},
    time::Duration,
};
use tokio::runtime::{Handle, Runtime};
use tonic::{
    transport::{Channel, Endpoint},
    Code, Status as GrpcStatus,
};

// Custom Python exceptions for error classification
create_exception!(pegaflow, PegaFlowError, PyException);
create_exception!(pegaflow, PegaFlowServiceError, PegaFlowError);
create_exception!(pegaflow, PegaFlowBusinessError, PegaFlowError);
use tracing_subscriber::{
    fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

static INIT_TRACING: Once = Once::new();
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

fn init_tracing() {
    INIT_TRACING.call_once(|| {
        // Default to info for most crates, debug for core if RUST_LOG not set.
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| "info,pegaflow_core=info".parse())
            .unwrap_or_else(|_| EnvFilter::new("info"));

        let fmt_layer = tracing_subscriber::fmt::layer().with_span_events(FmtSpan::CLOSE);

        // Ignore errors if already initialized by embedding app.
        let _ = tracing_subscriber::registry()
            .with(fmt_layer)
            .with(env_filter)
            .try_init();
    });
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

/// Python wrapper for PegaEngine
#[pyclass]
struct PegaEngine {
    engine: CoreEngine,
}

#[pyclass]
struct EngineRpcClient {
    endpoint: String,
    client: EngineClient<Channel>,
    rt_handle: Handle,
}

impl EngineRpcClient {
    /// Execute an RPC call with shared boilerplate:
    /// - get global runtime
    /// - clone channel
    /// - create client
    /// - block_on the async closure
    fn call<F, Fut, T>(&self, py: Python<'_>, f: F) -> PyResult<T>
    where
        F: FnOnce(EngineClient<Channel>) -> Fut + Send,
        Fut: Future<Output = Result<T, GrpcStatus>> + Send,
        T: Send,
    {
        let rt_handle = self.rt_handle.clone();
        let client = self.client.clone();
        py.detach(move || {
            rt_handle
                .block_on(async move { f(client).await.map_err(|e| rpc_status_error("rpc", e)) })
        })
    }
}

#[pymethods]
impl PegaEngine {
    /// Create a new PegaEngine instance
    #[new]
    fn new() -> Self {
        init_tracing();
        PegaEngine {
            engine: CoreEngine::new(),
        }
    }

    /// Register a context layer buffer along with its layout metadata.
    ///
    /// Args:
    ///     instance_id: ID of the model instance
    ///     namespace: Namespace for model isolation
    ///     device_id: CUDA device ID
    ///     layer_name: Name of the layer
    ///     data_ptr: GPU data pointer (as u64)
    ///     size_bytes: Total size of the tensor in bytes
    ///     num_blocks: Total number of paged blocks for this layer
    ///     bytes_per_block: Size of each paged block in bytes
    ///     kv_stride_bytes: Byte stride between K and V when KV-first layout is used
    ///     segments: Number of segments per block (1 for blocks-first, 2 for KV-first)
    ///     tp_rank: Tensor Parallel rank of the worker
    ///     device_id: CUDA device ID of the worker
    ///     tp_size: Total Tensor Parallel size
    ///     num_layers: Total number of layers in the model
    #[allow(clippy::too_many_arguments)]
    fn register_context_layer(
        &mut self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
        tp_rank: usize,
        tp_size: usize,
        num_layers: usize,
    ) -> PyResult<()> {
        self.engine
            .register_context_layer(
                instance_id,
                namespace,
                device_id,
                layer_name,
                data_ptr,
                size_bytes,
                num_blocks,
                bytes_per_block,
                kv_stride_bytes,
                segments,
                tp_rank,
                tp_size,
                num_layers,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Unregister the active inference context/instance
    fn unregister_instance(&mut self, instance_id: &str) -> PyResult<()> {
        self.engine
            .unregister_instance(instance_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Batch load KV blocks for multiple layers using the same block mapping.
    ///
    /// This is much more efficient than calling load_kv_blocks_to_ipc in a loop
    /// from Python, as it avoids Python overhead, data copying, and redundant hash lookups.
    ///
    /// The optimization reduces hash table lookups from O(layers × blocks) to O(blocks)
    /// by performing all lookups once and then extracting blocks for each layer.
    ///
    /// Args:
    ///     instance_id: ID of the model instance
    ///     tp_rank: Tensor Parallel rank of the worker
    ///     load_state_shm: Shared memory name from PyLoadState.shm_name() for sync
    ///     layer_names: List of layer names to load
    ///     block_ids: GPU block IDs to load into (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn batch_load_kv_blocks(
        &self,
        py: Python<'_>,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        let instance_id_owned = instance_id.to_string();
        let load_state_shm_owned = load_state_shm.to_string();
        let engine = &self.engine;
        py.detach(move || {
            let layer_name_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            engine.batch_load_kv_blocks_multi_layer(
                &instance_id_owned,
                tp_rank,
                device_id,
                &load_state_shm_owned,
                &layer_name_refs,
                &block_ids,
                &block_hashes,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymethods]
impl EngineRpcClient {
    #[new]
    #[pyo3(signature = (endpoint = None))]
    fn new(endpoint: Option<String>) -> PyResult<Self> {
        let endpoint = endpoint.unwrap_or_else(|| "http://127.0.0.1:50055".to_string());
        let rt = get_runtime()?;

        // Avoid per-RPC overhead by eager-connecting and reusing a warmed client handle.
        let endpoint_cfg = Endpoint::from_shared(endpoint.clone())
            .map_err(|err| transport_connect_error(&endpoint, err))?
            .connect_timeout(Duration::from_millis(500))
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_while_idle(true);

        let channel = rt
            .block_on(endpoint_cfg.connect())
            .map_err(|err| transport_connect_error(&endpoint, err))?;
        let client = EngineClient::new(channel);
        let rt_handle = rt.handle().clone();

        Ok(Self {
            endpoint,
            client,
            rt_handle,
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
        self.call(py, |mut c| async move {
            let resp = c.health(HealthRequest {}).await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("health", r.status))
    }

    /// Register a context layer for KV cache operations.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     namespace: Namespace for model isolation
    ///     tp_rank: Tensor parallel rank
    ///     tp_size: Total tensor parallel size
    ///     device_id: CUDA device ID
    ///     num_layers: Number of model layers
    ///     layer_name: Name of this layer
    ///     wrapper_bytes: Serialized CUDA tensor wrapper
    ///     num_blocks: Number of KV blocks
    ///     bytes_per_block: Size of each block in bytes
    ///     kv_stride_bytes: Stride between K and V
    ///     segments: Number of segments per block
    ///
    /// Returns: (ok: bool, message: str)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (instance_id, namespace, tp_rank, tp_size, device_id, num_layers, layer_name, wrapper_bytes, num_blocks, bytes_per_block, kv_stride_bytes, segments))]
    fn register_context(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_rank: u32,
        tp_size: u32,
        device_id: i32,
        num_layers: u32,
        layer_name: String,
        wrapper_bytes: Vec<u8>,
        num_blocks: u64,
        bytes_per_block: u64,
        kv_stride_bytes: u64,
        segments: u32,
    ) -> PyResult<(bool, String)> {
        self.call(py, |mut c| async move {
            let resp = c
                .register_context(RegisterContextRequest {
                    instance_id,
                    tp_rank,
                    tp_size,
                    device_id,
                    num_layers,
                    layer_name,
                    wrapper_bytes,
                    num_blocks,
                    bytes_per_block,
                    kv_stride_bytes,
                    segments,
                    namespace,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("register_context", r.status))
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
        self.call(py, |mut c| async move {
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
        self.call(py, |mut c| async move {
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

    /// Query prefix cache hits with prefetch support.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     block_hashes: List of block hashes to check
    ///
    /// Returns: dict with keys:
    ///     - ok: bool - whether the request succeeded
    ///     - message: str - error message if failed
    ///     - hit_blocks: int - number of blocks ready in cache
    ///     - prefetch_state: str - one of "ready", "loading", "partial_miss"
    ///     - loading_blocks: int - number of blocks being prefetched
    ///     - missing_blocks: int - number of blocks not found in DFS
    fn query(
        &self,
        py: Python<'_>,
        instance_id: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<Py<pyo3::types::PyAny>> {
        use pegaflow_proto::proto::engine::PrefetchState;

        self.call(py, |mut c| async move {
            let resp = c
                .query(QueryRequest {
                    instance_id,
                    block_hashes,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| {
            let (ok, msg) = status_tuple("query", r.status)?;
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

    /// Unregister a context/instance.
    ///
    /// Args:
    ///     instance_id: Model instance ID to unregister
    ///
    /// Returns: (ok: bool, message: str)
    fn unregister_context(&self, py: Python<'_>, instance_id: String) -> PyResult<(bool, String)> {
        self.call(py, |mut c| async move {
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
        self.call(py, |mut c| async move {
            let resp = c.shutdown(ShutdownRequest {}).await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("shutdown", r.status))
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

/// A Python module implemented in Rust.
/// This module is named "pegaflow" and will be imported as: from pegaflow import PegaEngine
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();
    m.add_class::<PegaEngine>()?;
    m.add_class::<EngineRpcClient>()?;
    m.add_class::<PyLoadState>()?;

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
