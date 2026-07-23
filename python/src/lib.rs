use pegaflow_common::grpc::{GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL, GRPC_CONNECT_TIMEOUT};
use pegaflow_core::LoadState;
use pegaflow_proto::proto::engine::{
    HealthRequest, LeaseLoad, LoadRequest, QueryRequest, RegisterContextRequest, ReleaseRequest,
    ResponseStatus, SaveLayer, SaveRequest, SessionEvent, SessionRequest, ShutdownRequest,
    TransferMode, UnregisterRequest, engine_client::EngineClient, query_response,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    future::Future,
    sync::{Arc, Mutex, OnceLock},
};
use tokio::runtime::{Handle, Runtime};
use tonic::{
    Code, Status as GrpcStatus, Streaming,
    transport::{Channel, Endpoint},
};

#[cfg(feature = "rdma")]
mod pd_rdma;
#[cfg(feature = "rdma")]
mod rdma_v1;

// Custom Python exceptions for error classification
create_exception!(pegaflow, PegaFlowError, PyException);
create_exception!(pegaflow, PegaflowInternal, PegaFlowError);

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
    PegaFlowError::new_err(format!(
        "failed to connect to engine server at {endpoint}: {err}"
    ))
}

fn rpc_status_error(method: &str, err: GrpcStatus) -> PyErr {
    let msg = format!("{method} RPC failed: {err}");
    match err.code() {
        Code::InvalidArgument => PyValueError::new_err(msg),
        Code::Internal => PegaflowInternal::new_err(msg),
        _ => PegaFlowError::new_err(msg),
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

#[pyclass(frozen)]
struct QueryLoading {}

#[pymethods]
impl QueryLoading {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "QueryLoading()".to_string()
    }
}

#[derive(Clone)]
struct PyQueryLease(Vec<u8>);

impl PyQueryLease {
    fn into_proto(self) -> Vec<u8> {
        self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyQueryLease {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let bytes: Vec<u8> = obj.extract()?;
        Ok(Self(bytes))
    }
}

impl<'py> IntoPyObject<'py> for PyQueryLease {
    type Target = pyo3::types::PyBytes;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(pyo3::types::PyBytes::new(py, &self.0))
    }
}

#[pyclass(frozen)]
struct QueryReady {
    #[pyo3(get)]
    num_hit_blocks: usize,
    lease: PyQueryLease,
}

#[pymethods]
impl QueryReady {
    #[new]
    fn new(num_hit_blocks: usize, lease: PyQueryLease) -> Self {
        Self {
            num_hit_blocks,
            lease,
        }
    }

    #[getter]
    fn lease<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        self.lease.clone().into_pyobject(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryReady(num_hit_blocks={}, has_lease={})",
            self.num_hit_blocks,
            !self.lease.0.is_empty()
        )
    }
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
    fn new(endpoint: Option<String>) -> PyResult<Self> {
        let endpoint = endpoint.unwrap_or_else(|| "http://127.0.0.1:50055".to_string());
        let rt = get_runtime()?;

        // Avoid per-RPC overhead by eager-connecting and reusing a warmed client handle.
        let endpoint_cfg = Endpoint::from_shared(endpoint.clone())
            .map_err(|err| transport_connect_error(&endpoint, err))?
            .connect_timeout(GRPC_CONNECT_TIMEOUT)
            .tcp_nodelay(true)
            .http2_keep_alive_interval(GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL)
            .keep_alive_while_idle(true);

        let channel = rt
            .block_on(endpoint_cfg.connect())
            .map_err(|err| transport_connect_error(&endpoint, err))?;
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
    /// Workers declare only the layers that actually exist on the device; the
    /// engine derives the instance-wide layer-id space once all `world_size`
    /// workers have registered.
    ///
    /// Argument contract:
    /// - `device_id` must be non-negative.
    /// - `tp_size` and `world_size` must be non-zero.
    /// - `tp_rank` must be less than `tp_size`.
    /// - Per-layer metadata lists must have the same non-zero length.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     namespace: Namespace for block hash isolation
    ///     tp_rank: Tensor parallel rank of the worker
    ///     pp_rank: Pipeline parallel rank of the worker
    ///     tp_size: Total Tensor Parallel size
    ///     world_size: Total worker count (TP * PP * PCP)
    ///     device_id: CUDA device ID of the worker
    ///     layer_names: List of layer names
    ///     wrapper_bytes_list: List of serialized CUDA tensor wrappers
    ///     num_blocks_list: List of block counts per layer
    ///     bytes_per_block_list: List of block sizes per layer
    ///     kv_stride_bytes_list: List of K/V strides per layer
    ///     segments_list: List of segment counts per layer
    ///
    /// Returns: (ok: bool, message: str)
    #[allow(
        clippy::too_many_arguments,
        reason = "PyO3 binding mirrors the public batch registration call shape"
    )]
    #[pyo3(signature = (instance_id, namespace, tp_rank, pp_rank, tp_size, world_size, device_id, layer_names, wrapper_bytes_list, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list, transfer_backend, page_first))]
    fn register_context_batch(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_rank: u32,
        pp_rank: u32,
        tp_size: u32,
        world_size: u32,
        device_id: i32,
        layer_names: Vec<String>,
        wrapper_bytes_list: Vec<Vec<u8>>,
        num_blocks_list: Vec<u64>,
        bytes_per_block_list: Vec<u64>,
        kv_stride_bytes_list: Vec<u64>,
        segments_list: Vec<u32>,
        transfer_backend: &str,
        page_first: bool,
    ) -> PyResult<(bool, String)> {
        let transfer_mode = match transfer_backend {
            "direct" => TransferMode::Direct,
            "kernel" => TransferMode::Kernel,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown transfer_backend '{other}' (expected 'direct' or 'kernel')"
                )));
            }
        };
        self.call(py, "register_context_batch", |mut c| async move {
            let resp = c
                .register_context_batch(RegisterContextRequest {
                    instance_id,
                    namespace,
                    client_version: pegaflow_proto::VERSION.to_string(),
                    tp_rank,
                    tp_size,
                    world_size,
                    device_id,
                    layer_names,
                    wrapper_bytes: wrapper_bytes_list,
                    num_blocks: num_blocks_list,
                    bytes_per_block: bytes_per_block_list,
                    kv_stride_bytes: kv_stride_bytes_list,
                    segments: segments_list,
                    pp_rank,
                    transfer_mode: transfer_mode as i32,
                    page_first,
                    native_kv_tensors: Vec::new(),
                    native_alloc_size: 0,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("register_context_batch", r.status))
    }

    /// Save KV blocks to the engine.
    ///
    /// Argument contract:
    /// - `device_id` must be non-negative.
    /// - Each save tuple must have matching `block_ids` and `block_hashes` lengths.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     tp_rank: Tensor parallel rank
    ///     pp_rank: Pipeline parallel rank
    ///     device_id: CUDA device ID
    ///     saves: List of (layer_name, block_ids, block_hashes) tuples
    ///
    /// Returns: (ok: bool, message: str)
    #[pyo3(signature = (instance_id, tp_rank, pp_rank, device_id, saves))]
    fn save(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
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
                    pp_rank,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("save", r.status))
    }

    /// Load KV blocks from the engine.
    ///
    /// Argument contract:
    /// - `device_id` must be non-negative.
    /// - Each lease must be a query lease returned by `query_prefetch`.
    /// - Each lease's block count must match its destination block_ids count.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     tp_rank: Tensor parallel rank
    ///     device_id: CUDA device ID
    ///     load_state_shm: Shared memory name for load state sync
    ///     layer_names: List of layer names to load
    ///     loads: List of (lease, block_ids) pairs
    ///
    /// Returns: (ok: bool, message: str)
    #[allow(
        clippy::too_many_arguments,
        reason = "PyO3 binding mirrors the connector load request shape"
    )]
    fn load(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        load_state_shm: String,
        layer_names: Vec<String>,
        loads: Vec<(Vec<u8>, Vec<u32>)>,
    ) -> PyResult<(bool, String)> {
        let loads = loads
            .into_iter()
            .map(|(lease, block_ids)| LeaseLoad { lease, block_ids })
            .collect();
        self.call(py, "load", |mut c| async move {
            let resp = c
                .load(LoadRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    load_state_shm,
                    layer_names,
                    loads,
                    wait_for_completion: false,
                })
                .await?;
            Ok(resp.into_inner())
        })
        .and_then(|r| status_tuple("load", r.status))
    }

    /// Query prefix cache hits with SSD prefetch support.
    ///
    /// Checks memory cache and triggers backing-store prefetch for missing blocks.
    /// Ready blocks are owned by an opaque lease.
    ///
    /// Argument contract:
    /// - `instance_id` must identify a registered model instance.
    /// - `req_id` must be non-empty and stable across retries for the same request.
    /// - `block_hashes` may be empty; an empty list returns `QueryReady(0, empty lease)`.
    ///
    /// Args:
    ///     instance_id: Model instance ID
    ///     block_hashes: List of block hashes to check
    ///     req_id: Request ID for prefetch correlation
    ///
    /// Returns:
    ///     QueryLoading while backing fetch is in progress, otherwise QueryReady.
    fn query_prefetch(
        &self,
        py: Python<'_>,
        instance_id: String,
        block_hashes: Vec<Vec<u8>>,
        req_id: String,
    ) -> PyResult<Py<PyAny>> {
        let result = py.detach(|| {
            self.rt_handle.block_on(async {
                let mut client = self.client.clone();
                client
                    .query_prefetch(QueryRequest {
                        instance_id,
                        block_hashes,
                        req_id,
                    })
                    .await
                    .map(|resp| resp.into_inner())
            })
        });

        let response = match result {
            Ok(response) => response,
            Err(status) if matches!(status.code(), Code::Unavailable | Code::DeadlineExceeded) => {
                return Python::attach(|py| {
                    Py::new(
                        py,
                        QueryReady {
                            num_hit_blocks: 0,
                            lease: PyQueryLease(Vec::new()),
                        },
                    )
                    .map(|obj| obj.into_any())
                });
            }
            Err(status) => return Err(rpc_status_error("query_prefetch", status)),
        };

        Python::attach(|py| match response.outcome {
            Some(query_response::Outcome::Loading(_)) => {
                Py::new(py, QueryLoading {}).map(|obj| obj.into_any())
            }
            Some(query_response::Outcome::Ready(ready)) => Py::new(
                py,
                QueryReady {
                    num_hit_blocks: u64_to_usize(ready.num_hit_blocks, "num_hit_blocks")?,
                    lease: PyQueryLease(ready.lease),
                },
            )
            .map(|obj| obj.into_any()),
            None => Err(PyRuntimeError::new_err(
                "query_prefetch response missing outcome",
            )),
        })
    }

    /// Release a query lease.
    fn release(&self, py: Python<'_>, lease: PyQueryLease) -> PyResult<()> {
        self.call(py, "release", |mut c| async move {
            c.release(ReleaseRequest {
                lease: lease.into_proto(),
            })
            .await?;
            Ok(())
        })
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

/// A Python module implemented in Rust.
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pegaflow_common::logging::init_stderr("info,pegaflow_core=info");
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<EngineRpcClient>()?;
    m.add_class::<PyLoadState>()?;
    #[cfg(feature = "rdma")]
    pd_rdma::add_classes(m)?;
    #[cfg(feature = "rdma")]
    rdma_v1::add_classes(m)?;
    // Register custom exceptions for error classification
    m.add("PegaFlowError", m.py().get_type::<PegaFlowError>())?;
    m.add("PegaflowInternal", m.py().get_type::<PegaflowInternal>())?;
    m.add_class::<QueryLoading>()?;
    m.add_class::<QueryReady>()?;

    Ok(())
}
