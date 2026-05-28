use pegaflow_core::LoadState;
use pegaflow_proto::proto::engine::{
    HealthRequest, LeaseLoad, LoadRequest, QueryRequest, RegisterContextRequest, ReleaseRequest,
    ResponseStatus, SaveLayer, SaveRequest, SessionEvent, SessionRequest, ShutdownRequest,
    UnregisterRequest, engine_client::EngineClient, query_response,
};
use pegaflow_transfer::v2::{
    CudaDeviceId, CudaDeviceMemory, Device, DomainAddress, DomainGroupRouting, FabricLibError,
    GroupTransferRouting, ImmCounter, ImmTransferRequest, MemoryRegionDescriptor,
    MemoryRegionHandle, MemoryRegionRemoteKey, RdmaEngine, ScatterTarget, ScatterTransferRequest,
    SmallVec, TransferCallback, TransferEngine, TransferEngineBuilder, TransferRequest,
    detect_topology,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    future::Future,
    ptr::NonNull,
    sync::atomic::{AtomicI64, Ordering},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use tokio::runtime::{Handle, Runtime};
use tonic::{
    Code, Status as GrpcStatus, Streaming,
    transport::{Channel, Endpoint},
};

// Custom Python exceptions for error classification
create_exception!(pegaflow, PegaFlowError, PyException);
create_exception!(pegaflow, PegaflowInternal, PegaFlowError);

static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
const PD_RDMA_WRITE_WINDOW: i64 = 64;

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

fn pd_rdma_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PegaFlowError::new_err(format!("{context}: {err}"))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn nonnull_from_u64(ptr: u64, field: &str) -> PyResult<NonNull<c_void>> {
    NonNull::new(ptr as *mut c_void)
        .ok_or_else(|| PyValueError::new_err(format!("{field} must be non-zero")))
}

fn py_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    let value = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?;
    value.extract()
}

fn wait_atomic_count(counter: &AtomicI64, target: i64, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while counter.load(Ordering::Acquire) < target {
        if Instant::now() >= deadline {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

fn wait_write_window(
    submitted: &AtomicI64,
    completed: &AtomicI64,
    limit: i64,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    while submitted.load(Ordering::Acquire) - completed.load(Ordering::Acquire) >= limit {
        if Instant::now() >= deadline {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

fn pd_imm(req_id: &str) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in req_id.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[derive(Clone)]
struct PdLocalLayer {
    base_addr: u64,
    mr: MemoryRegionHandle,
    mr_desc: MemoryRegionDescriptor,
}

#[derive(Clone)]
struct PdRemoteRegion {
    base_addr: u64,
    block_len: u64,
}

#[derive(Clone)]
struct PdRemoteLayer {
    mr_desc: MemoryRegionDescriptor,
    allowed_block_ids: HashSet<u64>,
    regions: Vec<PdRemoteRegion>,
}

struct PdRemoteRequest {
    remote_request_id: String,
    imm_data: u32,
    layers: HashMap<u64, PdRemoteLayer>,
}

#[derive(Clone)]
struct PendingRdmaWrites {
    submitted: i64,
    completed: Arc<AtomicI64>,
    errors: Arc<AtomicI64>,
}

impl PendingRdmaWrites {
    fn new() -> Self {
        Self {
            submitted: 0,
            completed: Arc::new(AtomicI64::new(0)),
            errors: Arc::new(AtomicI64::new(0)),
        }
    }
}

#[pyclass]
struct PdRdmaTestBuffer {
    mem: CudaDeviceMemory,
}

#[pymethods]
impl PdRdmaTestBuffer {
    #[new]
    #[pyo3(signature = (*, size, cuda_device = 0))]
    fn new(size: usize, cuda_device: u8) -> PyResult<Self> {
        let mem = CudaDeviceMemory::device_on(size, cuda_device)
            .map_err(|err| pd_rdma_error("cuda test buffer allocation failed", err))?;
        Ok(Self { mem })
    }

    fn ptr(&self) -> u64 {
        self.mem.ptr().as_ptr() as u64
    }

    fn size(&self) -> usize {
        self.mem.size()
    }

    fn fill(&self, value: u8) -> PyResult<()> {
        self.mem
            .fill(value)
            .map_err(|err| pd_rdma_error("cuda test buffer fill failed", err))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self
            .mem
            .copy_to_vec()
            .map_err(|err| pd_rdma_error("cuda test buffer copy_to_vec failed", err))?;
        Ok(PyBytes::new(py, &data))
    }
}

#[pyclass]
struct PdRdmaEngine {
    engine: Arc<TransferEngine>,
    device: Device,
    imm_domain: DomainGroupRouting,
    local_layers: Mutex<HashMap<u64, PdLocalLayer>>,
    remote_requests: Mutex<HashMap<String, PdRemoteRequest>>,
    imm_counters: Mutex<HashMap<String, ImmCounter>>,
    pending_writes: Mutex<HashMap<String, PendingRdmaWrites>>,
    write_submitted: AtomicI64,
    write_completed: Arc<AtomicI64>,
    imm_to_req: Arc<Mutex<HashMap<u32, String>>>,
    finished_sending: Arc<Mutex<HashSet<String>>>,
    finished_recving: Arc<Mutex<HashSet<String>>>,
    pin_worker_cpu: u16,
}

impl Drop for PdRdmaEngine {
    fn drop(&mut self) {
        self.engine.stop();
    }
}

#[pymethods]
impl PdRdmaEngine {
    #[new]
    #[pyo3(signature = (*, cuda_device = 0, numa_node = None, domains = None, device = "cuda", pin_worker_cpu = None))]
    fn new(
        cuda_device: u8,
        numa_node: Option<u8>,
        domains: Option<Vec<String>>,
        device: &str,
        pin_worker_cpu: Option<u16>,
    ) -> PyResult<Self> {
        let topology =
            detect_topology().map_err(|err| pd_rdma_error("v2 topology detect failed", err))?;
        log::info!(
            "[PdRdmaEngine] topology detected groups={} cuda_device={} numa={:?}",
            topology.len(),
            cuda_device,
            numa_node
        );
        let requested_domains = domains
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();
        let group = topology
            .iter()
            .find(|group| {
                group.cuda_device == cuda_device
                    && numa_node.is_none_or(|numa| group.numa == numa)
                    && !group.domains.is_empty()
            })
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "no RDMA topology group found for cuda_device={cuda_device} numa={numa_node:?}"
                ))
            })?;
        let selected_domains = group
            .domains
            .iter()
            .filter(|domain| {
                requested_domains.is_empty() || requested_domains.contains(&*domain.name())
            })
            .cloned()
            .collect::<Vec<_>>();
        if selected_domains.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "no RDMA domains matched cuda_device={cuda_device} numa={numa_node:?}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] selected cuda={} numa={} domains=[{}]",
            group.cuda_device,
            group.numa,
            selected_domains
                .iter()
                .map(|domain| domain.name().into_owned())
                .collect::<Vec<_>>()
                .join(",")
        );

        let worker_cpu = match pin_worker_cpu {
            Some(cpu) if group.cpus.contains(&cpu) => cpu,
            Some(cpu) => {
                return Err(PyValueError::new_err(format!(
                    "pin_worker_cpu {cpu} is not in the selected RDMA topology CPU set"
                )));
            }
            None => group.cpus.first().copied().ok_or_else(|| {
                PyRuntimeError::new_err("selected RDMA topology group has no CPU")
            })?,
        };
        let mut builder = TransferEngineBuilder::default();
        builder.add_gpu_domains(cuda_device, selected_domains, worker_cpu);
        let engine = Arc::new(
            builder
                .build()
                .map_err(|err| pd_rdma_error("v2 transfer engine build failed", err))?,
        );
        let imm_to_req = Arc::new(Mutex::new(HashMap::<u32, String>::new()));
        let finished_recving = Arc::new(Mutex::new(HashSet::<String>::new()));
        let callback_imm_to_req = Arc::clone(&imm_to_req);
        let callback_finished_recving = Arc::clone(&finished_recving);
        engine.add_imm_callback(Box::new(move |imm| {
            if let Some(req_id) = callback_imm_to_req.lock().unwrap().get(&imm).cloned() {
                callback_finished_recving.lock().unwrap().insert(req_id);
            }
            Ok(())
        }));
        let device = match device {
            "cuda" => Device::Cuda(CudaDeviceId(cuda_device)),
            "host" => Device::Host,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported RDMA device {other}"
                )));
            }
        };
        Ok(Self {
            engine,
            device,
            imm_domain: DomainGroupRouting::Pinned { domain_idx: 0 },
            local_layers: Mutex::new(HashMap::new()),
            remote_requests: Mutex::new(HashMap::new()),
            imm_counters: Mutex::new(HashMap::new()),
            pending_writes: Mutex::new(HashMap::new()),
            write_submitted: AtomicI64::new(0),
            write_completed: Arc::new(AtomicI64::new(0)),
            imm_to_req,
            finished_sending: Arc::new(Mutex::new(HashSet::new())),
            finished_recving,
            pin_worker_cpu: worker_cpu,
        })
    }

    fn register_local_layers(
        &self,
        py: Python<'_>,
        layers: Vec<Py<PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut registered_layers = Vec::with_capacity(layers.len());
        let mut local_layers = self.local_layers.lock().unwrap();
        for layer in layers {
            let layer = layer.bind(py);
            let layer_idx: u64 = py_get(layer, "layer_idx")?;
            let block_ids: Vec<u64> = py_get(layer, "block_ids")?;
            if block_ids.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "local layer {layer_idx} has no block_ids"
                )));
            }
            let max_block_id = block_ids
                .iter()
                .copied()
                .max()
                .ok_or_else(|| PyValueError::new_err("local layer has no block_ids"))?;
            let regions_any = layer
                .get_item("regions")?
                .ok_or_else(|| PyValueError::new_err("local layer missing regions"))?;
            let mut min_addr = u64::MAX;
            let mut max_addr = 0_u64;
            let mut region_count = 0_usize;
            for (expected_region_idx, region_any) in regions_any.try_iter()?.enumerate() {
                let region = region_any?.cast_into::<PyDict>()?;
                let region_idx: usize = py_get(&region, "region_idx")?;
                if region_idx != expected_region_idx {
                    return Err(PyValueError::new_err(format!(
                        "local layer {layer_idx} regions must be ordered by region_idx"
                    )));
                }
                let base_addr: u64 = py_get(&region, "base_addr")?;
                let block_len: u64 = py_get(&region, "block_len")?;
                if base_addr == 0 || block_len == 0 {
                    return Err(PyValueError::new_err(format!(
                        "local layer {layer_idx} region {region_idx} has invalid address range"
                    )));
                }
                let end = max_block_id
                    .checked_add(1)
                    .and_then(|block_count| block_count.checked_mul(block_len))
                    .and_then(|byte_len| base_addr.checked_add(byte_len))
                    .ok_or_else(|| PyValueError::new_err("local layer address range overflow"))?;
                min_addr = min_addr.min(base_addr);
                max_addr = max_addr.max(end);
                region_count += 1;
            }
            if region_count == 0 {
                return Err(PyValueError::new_err(format!(
                    "local layer {layer_idx} has no regions"
                )));
            }
            let len = max_addr
                .checked_sub(min_addr)
                .filter(|len| *len > 0)
                .ok_or_else(|| PyValueError::new_err("invalid layer address range"))?;
            let ptr = nonnull_from_u64(min_addr, "base_addr")?;
            let (mr, mr_desc) = self
                .engine
                .register_memory_allow_remote(ptr, u64_to_usize(len, "layer length")?, self.device)
                .map_err(|err| pd_rdma_error("register_memory_allow_remote failed", err))?;
            local_layers.insert(
                layer_idx,
                PdLocalLayer {
                    base_addr: min_addr,
                    mr,
                    mr_desc: mr_desc.clone(),
                },
            );
            log::info!(
                "[PdRdmaEngine] register local layer layer={} blocks={} regions={} base=0x{:x} len={} domains={} link_speed={}",
                layer_idx,
                block_ids.len(),
                region_count,
                min_addr,
                len,
                self.engine.num_domains(),
                self.engine.aggregated_link_speed(),
            );

            let out = PyDict::new(py);
            for (key, value) in layer.iter() {
                out.set_item(key, value)?;
            }
            out.set_item("mr_desc", mr_desc_to_py(py, &mr_desc)?)?;
            registered_layers.push(out.into_any().unbind());
        }
        Ok(registered_layers)
    }

    fn register_remote(
        &self,
        py: Python<'_>,
        req_id: String,
        handshake: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let Some(handshake) = handshake else {
            return Ok(());
        };
        let handshake = handshake.bind(py);
        let remote_request_id: String = py_get(handshake, "request_id")?;
        let imm = match handshake.get_item("imm_id")? {
            Some(value) if !value.is_none() => value.extract()?,
            _ => pd_imm(&remote_request_id),
        };
        self.imm_to_req.lock().unwrap().insert(imm, req_id.clone());
        self.imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm));

        let layers_any = handshake
            .get_item("layers")?
            .ok_or_else(|| PyValueError::new_err("handshake missing layers"))?;
        let mut layers = HashMap::new();
        for layer_any in layers_any.try_iter()? {
            let layer = layer_any?.cast_into::<PyDict>()?;
            let layer_idx: u64 = py_get(&layer, "layer_idx")?;
            let block_ids: Vec<u64> = py_get(&layer, "block_ids")?;
            if block_ids.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "remote layer {layer_idx} has no block_ids"
                )));
            }
            let allowed_block_ids = block_ids.iter().copied().collect::<HashSet<_>>();
            if allowed_block_ids.len() != block_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "remote layer {layer_idx} has duplicate block_ids"
                )));
            }
            let regions_any = layer
                .get_item("regions")?
                .ok_or_else(|| PyValueError::new_err("remote layer missing regions"))?;
            let mut regions = Vec::new();
            for (expected_region_idx, region_any) in regions_any.try_iter()?.enumerate() {
                let region = region_any?.cast_into::<PyDict>()?;
                let region_idx: usize = py_get(&region, "region_idx")?;
                if region_idx != expected_region_idx {
                    return Err(PyValueError::new_err(format!(
                        "remote layer {layer_idx} regions must be ordered by region_idx"
                    )));
                }
                let base_addr: u64 = py_get(&region, "base_addr")?;
                let block_len: u64 = py_get(&region, "block_len")?;
                if base_addr == 0 || block_len == 0 {
                    return Err(PyValueError::new_err(format!(
                        "remote layer {layer_idx} region {region_idx} has invalid address range"
                    )));
                }
                regions.push(PdRemoteRegion {
                    base_addr,
                    block_len,
                });
            }
            if regions.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "remote layer {layer_idx} has no regions"
                )));
            }
            let mr_desc_any = layer
                .get_item("mr_desc")?
                .ok_or_else(|| PyValueError::new_err("remote layer missing mr_desc"))?;
            let mr_desc = mr_desc_from_py(&mr_desc_any.cast_into::<PyDict>()?)?;
            layers.insert(
                layer_idx,
                PdRemoteLayer {
                    mr_desc,
                    allowed_block_ids,
                    regions,
                },
            );
        }
        let layer_count = layers.len();
        let blocks_per_layer = layers
            .values()
            .next()
            .map(|layer| layer.allowed_block_ids.len())
            .unwrap_or(0);
        let regions_per_layer = layers
            .values()
            .next()
            .map(|layer| layer.regions.len())
            .unwrap_or(0);
        log::info!(
            "[PdRdmaEngine] register remote req={} remote_req={} imm={} layers={} blocks_per_layer={} regions_per_layer={} domains={}",
            req_id,
            remote_request_id,
            imm,
            layer_count,
            blocks_per_layer,
            regions_per_layer,
            self.engine.num_domains(),
        );
        self.remote_requests.lock().unwrap().insert(
            req_id,
            PdRemoteRequest {
                remote_request_id,
                imm_data: imm,
                layers,
            },
        );
        Ok(())
    }

    fn push_layer(
        &self,
        py: Python<'_>,
        req_id: String,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<()> {
        let total_start = Instant::now();
        let input_blocks = blocks.len();
        let local = self
            .local_layers
            .lock()
            .unwrap()
            .get(&layer_idx)
            .cloned()
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("local layer {layer_idx} is not registered"))
            })?;
        let remote = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .and_then(|request| request.layers.get(&layer_idx).cloned())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "remote layer {layer_idx} for req {req_id} is not registered"
                ))
            })?;
        let mut dsts = Vec::with_capacity(blocks.len() * remote.regions.len());
        let mut dst_bytes = 0_u64;
        for block in blocks {
            let block = block.bind(py);
            let regions_any = block
                .get_item("regions")?
                .ok_or_else(|| PyValueError::new_err("block missing regions"))?;
            for (expected_region_idx, region_any) in regions_any.try_iter()?.enumerate() {
                let region = region_any?.cast_into::<PyDict>()?;
                let region_idx: usize = py_get(&region, "region_idx")?;
                if region_idx != expected_region_idx {
                    return Err(PyValueError::new_err(
                        "block regions must be ordered by region_idx",
                    ));
                }
                let target =
                    self.block_slice_to_scatter_target(&local, &remote, &region, region_idx)?;
                dst_bytes = dst_bytes.checked_add(target.length).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "RDMA scatter byte count overflow for req={req_id} layer={layer_idx}"
                    ))
                })?;
                dsts.push(target);
            }
        }
        if dsts.is_empty() {
            return Ok(());
        }
        let convert_ms = duration_ms(total_start.elapsed());
        let window_start = Instant::now();
        if !py.detach(|| {
            wait_write_window(
                &self.write_submitted,
                &self.write_completed,
                PD_RDMA_WRITE_WINDOW,
                Duration::from_secs(30),
            )
        }) {
            log::error!(
                "[PdRdmaEngine] RDMA WRITE window timeout req={} layer={} submitted={} completed={} window_wait_ms={:.3}",
                req_id,
                layer_idx,
                self.write_submitted.load(Ordering::Acquire),
                self.write_completed.load(Ordering::Acquire),
                duration_ms(window_start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE window timed out for req={req_id} layer={layer_idx} submitted={} completed={}",
                self.write_submitted.load(Ordering::Acquire),
                self.write_completed.load(Ordering::Acquire)
            )));
        }
        let window_ms = duration_ms(window_start.elapsed());
        let (req_completed, req_errors) = {
            let mut pending = self.pending_writes.lock().unwrap();
            let state = pending
                .entry(req_id.clone())
                .or_insert_with(PendingRdmaWrites::new);
            state.submitted += 1;
            (Arc::clone(&state.completed), Arc::clone(&state.errors))
        };
        let scatter_targets = dsts.len();
        let submitted_after = self.write_submitted.fetch_add(1, Ordering::Release) + 1;
        let global_completed = Arc::clone(&self.write_completed);
        let done_completed = Arc::clone(&req_completed);
        let error_completed = Arc::clone(&req_completed);
        let error_counter = Arc::clone(&req_errors);
        let error_global_completed = Arc::clone(&global_completed);
        let submit_start = Instant::now();
        let callback_start = Instant::now();
        let done_req_id = req_id.clone();
        let error_req_id = req_id.clone();
        let done_bytes = dst_bytes;
        let error_bytes = dst_bytes;
        let done_layer_idx = layer_idx;
        let error_layer_idx = layer_idx;
        self.engine
            .submit_transfer(
                TransferRequest::Scatter(ScatterTransferRequest {
                    src_mr: local.mr,
                    dst_handle: None,
                    dsts: Arc::new(dsts),
                    imm_data: None,
                    domain: GroupTransferRouting::AllDomainsShardBytes,
                }),
                TransferCallback {
                    on_done: Box::new(move || {
                        done_completed.fetch_add(1, Ordering::Release);
                        global_completed.fetch_add(1, Ordering::Release);
                        log::info!(
                            "[PdRdmaEngine] RDMA WRITE completed req={} layer={} bytes={} latency_ms={:.3}",
                            done_req_id,
                            done_layer_idx,
                            done_bytes,
                            duration_ms(callback_start.elapsed()),
                        );
                        Ok(())
                    }),
                    on_error: Box::new(move |err: FabricLibError| {
                        error_counter.fetch_add(1, Ordering::Release);
                        error_completed.fetch_add(1, Ordering::Release);
                        error_global_completed.fetch_add(1, Ordering::Release);
                        log::error!(
                            "[PdRdmaEngine] RDMA WRITE completion error req={} layer={} bytes={} latency_ms={:.3} err={err}",
                            error_req_id,
                            error_layer_idx,
                            error_bytes,
                            duration_ms(callback_start.elapsed()),
                        );
                        Ok(())
                    }),
                },
            )
            .map_err(|err| {
                req_errors.fetch_add(1, Ordering::Release);
                req_completed.fetch_add(1, Ordering::Release);
                self.write_completed.fetch_add(1, Ordering::Release);
                log::error!(
                    "[PdRdmaEngine] RDMA WRITE submit failed req={} layer={} input_blocks={} scatter_targets={} bytes={} domains={} err={err}",
                    req_id,
                    layer_idx,
                    input_blocks,
                    scatter_targets,
                    dst_bytes,
                    self.engine.num_domains(),
                );
                pd_rdma_error("submit RDMA WRITE failed", err)
            })?;
        log::info!(
            "[PdRdmaEngine] RDMA WRITE submitted req={} layer={} input_blocks={} scatter_targets={} bytes={} domains={} convert_ms={:.3} window_wait_ms={:.3} submit_ms={:.3} submitted={} completed={}",
            req_id,
            layer_idx,
            input_blocks,
            scatter_targets,
            dst_bytes,
            self.engine.num_domains(),
            convert_ms,
            window_ms,
            duration_ms(submit_start.elapsed()),
            submitted_after,
            self.write_completed.load(Ordering::Acquire),
        );
        Ok(())
    }

    fn wait_for_pushes(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_writes(py, &req_id, Duration::from_secs(30))
    }

    fn push_done(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_writes(py, &req_id, Duration::from_secs(30))?;
        let start = Instant::now();
        let remote = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} is not registered"))
            })?
            .layers
            .values()
            .next()
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err(format!("remote req {req_id} has no layers")))?;
        let imm_data = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .map(|request| request.imm_data)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} is not registered"))
            })?;
        let tx_counter = Arc::new(AtomicI64::new(0));
        let err_counter = Arc::new(AtomicI64::new(0));
        log::info!(
            "[PdRdmaEngine] RDMA IMM submit start req={} imm={} domains={}",
            req_id,
            imm_data,
            self.engine.num_domains(),
        );
        self.engine
            .submit_transfer_atomic(
                TransferRequest::Imm(ImmTransferRequest {
                    imm_data,
                    dst_mr: remote.mr_desc,
                    domain: self.imm_domain,
                }),
                Arc::clone(&tx_counter),
                Arc::clone(&err_counter),
            )
            .map_err(|err| {
                log::error!(
                    "[PdRdmaEngine] RDMA IMM submit failed req={} imm={} err={err}",
                    req_id,
                    imm_data,
                );
                pd_rdma_error("submit IMM failed", err)
            })?;
        if !py.detach(|| wait_atomic_count(&tx_counter, 1, Duration::from_secs(30))) {
            log::error!(
                "[PdRdmaEngine] RDMA IMM timed out req={} imm={} completed={} errors={} wait_ms={:.3}",
                req_id,
                imm_data,
                tx_counter.load(Ordering::Acquire),
                err_counter.load(Ordering::Acquire),
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM timed out for req={req_id} completed={}",
                tx_counter.load(Ordering::Acquire)
            )));
        }
        let errors = err_counter.load(Ordering::Acquire);
        if errors != 0 {
            log::error!(
                "[PdRdmaEngine] RDMA IMM failed req={} imm={} completed={} errors={} wait_ms={:.3}",
                req_id,
                imm_data,
                tx_counter.load(Ordering::Acquire),
                errors,
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM failed for req={req_id} errors={errors}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] RDMA IMM done req={} imm={} wait_ms={:.3}",
            req_id,
            imm_data,
            duration_ms(start.elapsed()),
        );
        self.finished_sending.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn wait_done(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        let (remote_request_id, imm_data) = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .map(|request| (request.remote_request_id.clone(), request.imm_data))
            .unwrap_or_else(|| {
                let fallback = pd_imm(&req_id);
                (req_id.clone(), fallback)
            });
        if self.finished_recving.lock().unwrap().remove(&req_id) {
            self.finished_recving.lock().unwrap().insert(req_id.clone());
            log::info!(
                "[PdRdmaEngine] RDMA IMM wait already done req={} remote_req={} imm={}",
                req_id,
                remote_request_id,
                imm_data,
            );
            return Ok(());
        }
        let counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm_data))
            .clone();
        let start = Instant::now();
        log::info!(
            "[PdRdmaEngine] RDMA IMM wait start req={} remote_req={} imm={}",
            req_id,
            remote_request_id,
            imm_data,
        );
        let done = py.detach(|| counter.wait_timeout(1, Duration::from_secs(30)));
        if !done {
            log::error!(
                "[PdRdmaEngine] RDMA IMM wait timed out req={} remote_req={} imm={} wait_ms={:.3}",
                req_id,
                remote_request_id,
                imm_data,
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM wait timed out for req={req_id} remote_request_id={remote_request_id}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] RDMA IMM wait done req={} remote_req={} imm={} wait_ms={:.3}",
            req_id,
            remote_request_id,
            imm_data,
            duration_ms(start.elapsed()),
        );
        self.finished_recving.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn poll_done(&self, req_id: String) -> bool {
        if self.finished_recving.lock().unwrap().contains(&req_id) {
            return true;
        }
        let imm_data = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .map(|request| request.imm_data)
            .unwrap_or_else(|| pd_imm(&req_id));
        let counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm_data))
            .clone();
        if counter.wait_timeout(1, Duration::ZERO) {
            self.finished_recving.lock().unwrap().insert(req_id);
            return true;
        }
        false
    }

    fn mark_done(&self, req_id: String) {
        self.finished_recving.lock().unwrap().insert(req_id);
    }

    fn pop_finished_sending(&self) -> HashSet<String> {
        std::mem::take(&mut *self.finished_sending.lock().unwrap())
    }

    fn pop_finished_recving(&self) -> HashSet<String> {
        std::mem::take(&mut *self.finished_recving.lock().unwrap())
    }

    fn close_request(&self, req_id: String) {
        if let Some(request) = self.remote_requests.lock().unwrap().remove(&req_id) {
            self.engine.remove_imm_count(request.imm_data);
            self.imm_to_req.lock().unwrap().remove(&request.imm_data);
        }
        self.imm_counters.lock().unwrap().remove(&req_id);
        self.pending_writes.lock().unwrap().remove(&req_id);
        self.finished_sending.lock().unwrap().remove(&req_id);
        self.finished_recving.lock().unwrap().remove(&req_id);
    }

    fn main_address(&self) -> String {
        self.engine.main_address().to_string()
    }

    fn num_domains(&self) -> usize {
        self.engine.num_domains()
    }

    fn num_groups(&self) -> usize {
        self.engine.num_groups()
    }

    fn aggregated_link_speed(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }

    fn pin_worker_cpu(&self) -> u16 {
        self.pin_worker_cpu
    }
}

impl PdRdmaEngine {
    fn wait_for_request_writes(
        &self,
        py: Python<'_>,
        req_id: &str,
        timeout: Duration,
    ) -> PyResult<()> {
        let Some(state) = self.pending_writes.lock().unwrap().get(req_id).cloned() else {
            return Ok(());
        };
        let submitted = state.submitted;
        if submitted == 0 {
            return Ok(());
        }
        let start = Instant::now();
        log::info!(
            "[PdRdmaEngine] RDMA WRITE wait start req={} submitted={} completed={} errors={}",
            req_id,
            submitted,
            state.completed.load(Ordering::Acquire),
            state.errors.load(Ordering::Acquire),
        );
        if !py.detach(|| wait_atomic_count(&state.completed, submitted, timeout)) {
            log::error!(
                "[PdRdmaEngine] RDMA WRITE wait timeout req={} submitted={} completed={} errors={} wait_ms={:.3}",
                req_id,
                submitted,
                state.completed.load(Ordering::Acquire),
                state.errors.load(Ordering::Acquire),
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE timed out for req={req_id} submitted={submitted} completed={}",
                state.completed.load(Ordering::Acquire)
            )));
        }
        let errors = state.errors.load(Ordering::Acquire);
        if errors != 0 {
            log::error!(
                "[PdRdmaEngine] RDMA WRITE wait failed req={} submitted={} completed={} errors={} wait_ms={:.3}",
                req_id,
                submitted,
                state.completed.load(Ordering::Acquire),
                errors,
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE failed for req={req_id} errors={errors}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] RDMA WRITE wait done req={} submitted={} completed={} wait_ms={:.3}",
            req_id,
            submitted,
            state.completed.load(Ordering::Acquire),
            duration_ms(start.elapsed()),
        );
        Ok(())
    }

    fn block_slice_to_scatter_target(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        block: &Bound<'_, PyDict>,
        region_idx: usize,
    ) -> PyResult<ScatterTarget> {
        let block_id: u64 = py_get(block, "block_id")?;
        let src_offset: u64 = py_get(block, "src_offset_bytes")?;
        let bytes: u64 = py_get(block, "bytes")?;
        if bytes == 0 {
            return Err(PyValueError::new_err("block slice bytes must be positive"));
        }
        let region = remote.regions.get(region_idx).ok_or_else(|| {
            PyValueError::new_err(format!("remote region {region_idx} is not registered"))
        })?;
        if !bytes.is_multiple_of(region.block_len) {
            return Err(PyValueError::new_err(format!(
                "block slice bytes {bytes} must be a positive multiple of remote region block_len {}",
                region.block_len
            )));
        }
        let block_count = bytes / region.block_len;
        for offset in 0..block_count {
            let remote_block_id = block_id
                .checked_add(offset)
                .ok_or_else(|| PyValueError::new_err("remote block id overflow"))?;
            if !remote.allowed_block_ids.contains(&remote_block_id) {
                return Err(PyRuntimeError::new_err(format!(
                    "remote block {remote_block_id} is not registered"
                )));
            }
        }
        let remote_addr = block_id
            .checked_mul(region.block_len)
            .and_then(|offset| region.base_addr.checked_add(offset))
            .ok_or_else(|| PyValueError::new_err("remote block address overflow"))?;
        let dst_offset = remote_addr
            .checked_sub(remote.mr_desc.ptr)
            .ok_or_else(|| PyRuntimeError::new_err("remote block address is below MR base"))?;
        let src_absolute = local
            .base_addr
            .checked_add(src_offset)
            .ok_or_else(|| PyValueError::new_err("source offset overflow"))?;
        if src_absolute < local.mr_desc.ptr {
            return Err(PyValueError::new_err(
                "source block address is below local MR base",
            ));
        }
        let src_mr_offset = src_absolute - local.mr_desc.ptr;
        Ok(ScatterTarget {
            dst_mr: remote.mr_desc.clone(),
            length: bytes,
            src_offset: src_mr_offset,
            dst_offset,
        })
    }
}

fn mr_desc_to_py(py: Python<'_>, mr_desc: &MemoryRegionDescriptor) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    out.set_item("ptr", mr_desc.ptr)?;
    let addr_rkey_list = PyList::empty(py);
    for (addr, rkey) in &mr_desc.addr_rkey_list {
        addr_rkey_list.append((addr.to_string(), rkey.0))?;
    }
    out.set_item("addr_rkey_list", addr_rkey_list)?;
    Ok(out.into_any().unbind())
}

fn mr_desc_from_py(dict: &Bound<'_, PyDict>) -> PyResult<MemoryRegionDescriptor> {
    let ptr: u64 = py_get(dict, "ptr")?;
    let addr_rkey_list: Vec<(String, u64)> = py_get(dict, "addr_rkey_list")?;
    let mut pairs = SmallVec::new();
    for (addr, rkey) in addr_rkey_list {
        let addr = addr
            .parse::<DomainAddress>()
            .map_err(|err| pd_rdma_error("invalid DomainAddress", err))?;
        pairs.push((addr, MemoryRegionRemoteKey(rkey)));
    }
    Ok(MemoryRegionDescriptor {
        ptr,
        addr_rkey_list: pairs,
    })
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
            .connect_timeout(Duration::from_millis(500))
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
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
    /// Argument contract:
    /// - `device_id` must be non-negative.
    /// - `num_layers`, `tp_size`, and `world_size` must be non-zero.
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
    ///     num_layers: Total number of layers in the model
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
    #[pyo3(signature = (instance_id, namespace, tp_rank, pp_rank, tp_size, world_size, device_id, num_layers, layer_names, wrapper_bytes_list, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list))]
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
                    pp_rank,
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
        loads: Vec<(Vec<u8>, Vec<i32>)>,
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
    m.add_class::<PdRdmaEngine>()?;
    m.add_class::<PdRdmaTestBuffer>()?;
    // Register custom exceptions for error classification
    m.add("PegaFlowError", m.py().get_type::<PegaFlowError>())?;
    m.add("PegaflowInternal", m.py().get_type::<PegaflowInternal>())?;
    m.add_class::<QueryLoading>()?;
    m.add_class::<QueryReady>()?;

    Ok(())
}
