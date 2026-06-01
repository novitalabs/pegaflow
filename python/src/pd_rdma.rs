use crate::{PegaFlowError, u64_to_usize};

use pegaflow_transfer::v2::{
    CudaDeviceId, CudaDeviceMemory, Device, DomainAddress, DomainGroupRouting, FabricLibError,
    GroupTransferRouting, ImmCounter, ImmTransferRequest, MemoryRegionDescriptor,
    MemoryRegionHandle, MemoryRegionRemoteKey, RdmaEngine, ScatterTarget, ScatterTransferRequest,
    SmallVec, TransferCallback, TransferEngine, TransferEngineBuilder, TransferRequest,
    detect_topology,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::atomic::{AtomicI64, Ordering},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const PD_RDMA_WRITE_WINDOW: i64 = 64;
const WAIT_SPINS_BEFORE_YIELD: u32 = 64;
const WAIT_YIELDS_BEFORE_SLEEP: u32 = 512;

fn pd_rdma_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PegaFlowError::new_err(format!("{context}: {err}"))
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
    let mut waits = 0;
    while counter.load(Ordering::Acquire) < target {
        if Instant::now() >= deadline {
            return false;
        }
        wait_backoff(&mut waits);
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
    let mut waits = 0;
    while submitted.load(Ordering::Acquire) - completed.load(Ordering::Acquire) >= limit {
        if Instant::now() >= deadline {
            return false;
        }
        wait_backoff(&mut waits);
    }
    true
}

fn wait_backoff(waits: &mut u32) {
    if *waits < WAIT_SPINS_BEFORE_YIELD {
        std::hint::spin_loop();
    } else if *waits < WAIT_YIELDS_BEFORE_SLEEP {
        std::thread::yield_now();
    } else {
        std::thread::sleep(Duration::from_micros(50));
    }
    *waits = waits.saturating_add(1);
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
    block_bytes: u64,
    mr: MemoryRegionHandle,
    mr_desc: MemoryRegionDescriptor,
}

#[derive(Clone)]
struct PdRemoteLayer {
    mr_desc: MemoryRegionDescriptor,
    k_by_block: HashMap<u64, u64>,
    v_by_block: HashMap<u64, u64>,
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
            let base_addr: u64 = py_get(layer, "base_addr")?;
            let block_bytes: u64 = py_get(layer, "block_bytes")?;
            let k_block_addrs: Vec<u64> = py_get(layer, "k_block_addrs")?;
            let v_block_addrs: Vec<u64> = py_get(layer, "v_block_addrs")?;
            let max_addr = k_block_addrs
                .iter()
                .chain(v_block_addrs.iter())
                .copied()
                .max()
                .ok_or_else(|| PyValueError::new_err("layer has no block addresses"))?;
            let len = max_addr
                .checked_add(block_bytes)
                .and_then(|end| end.checked_sub(base_addr))
                .ok_or_else(|| PyValueError::new_err("invalid layer address range"))?;
            let ptr = nonnull_from_u64(base_addr, "base_addr")?;
            let (mr, mr_desc) = self
                .engine
                .register_memory_allow_remote(ptr, u64_to_usize(len, "layer length")?, self.device)
                .map_err(|err| pd_rdma_error("register_memory_allow_remote failed", err))?;
            local_layers.insert(
                layer_idx,
                PdLocalLayer {
                    base_addr,
                    block_bytes,
                    mr,
                    mr_desc: mr_desc.clone(),
                },
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
            let k_block_addrs: Vec<u64> = py_get(&layer, "k_block_addrs")?;
            let v_block_addrs: Vec<u64> = py_get(&layer, "v_block_addrs")?;
            if block_ids.len() != k_block_addrs.len() || block_ids.len() != v_block_addrs.len() {
                return Err(PyValueError::new_err(format!(
                    "remote layer {layer_idx} block_ids and addresses length mismatch"
                )));
            }
            let mr_desc_any = layer
                .get_item("mr_desc")?
                .ok_or_else(|| PyValueError::new_err("remote layer missing mr_desc"))?;
            let mr_desc = mr_desc_from_py(&mr_desc_any.cast_into::<PyDict>()?)?;
            let k_by_block = block_ids
                .iter()
                .copied()
                .zip(k_block_addrs.iter().copied())
                .collect::<HashMap<_, _>>();
            let v_by_block = block_ids
                .iter()
                .copied()
                .zip(v_block_addrs.iter().copied())
                .collect::<HashMap<_, _>>();
            layers.insert(
                layer_idx,
                PdRemoteLayer {
                    mr_desc,
                    k_by_block,
                    v_by_block,
                },
            );
        }
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
        let mut dsts = Vec::with_capacity(blocks.len() * 2);
        for block in blocks {
            let block = block.bind(py);
            let k = block
                .get_item("k")?
                .ok_or_else(|| PyValueError::new_err("block missing k"))?
                .cast_into::<PyDict>()?;
            let v = block
                .get_item("v")?
                .ok_or_else(|| PyValueError::new_err("block missing v"))?
                .cast_into::<PyDict>()?;
            dsts.push(self.block_slice_to_scatter_target(&local, &remote, &k, true)?);
            dsts.push(self.block_slice_to_scatter_target(&local, &remote, &v, false)?);
        }
        if dsts.is_empty() {
            return Ok(());
        }
        if !py.detach(|| {
            wait_write_window(
                &self.write_submitted,
                &self.write_completed,
                PD_RDMA_WRITE_WINDOW,
                Duration::from_secs(30),
            )
        }) {
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE window timed out for req={req_id} layer={layer_idx} submitted={} completed={}",
                self.write_submitted.load(Ordering::Acquire),
                self.write_completed.load(Ordering::Acquire)
            )));
        }
        let (req_completed, req_errors) = {
            let mut pending = self.pending_writes.lock().unwrap();
            let state = pending
                .entry(req_id.clone())
                .or_insert_with(PendingRdmaWrites::new);
            state.submitted += 1;
            (Arc::clone(&state.completed), Arc::clone(&state.errors))
        };
        self.write_submitted.fetch_add(1, Ordering::Release);
        let global_completed = Arc::clone(&self.write_completed);
        let done_completed = Arc::clone(&req_completed);
        let error_completed = Arc::clone(&req_completed);
        let error_counter = Arc::clone(&req_errors);
        let error_global_completed = Arc::clone(&global_completed);
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
                        Ok(())
                    }),
                    on_error: Box::new(move |err: FabricLibError| {
                        error_counter.fetch_add(1, Ordering::Release);
                        error_completed.fetch_add(1, Ordering::Release);
                        error_global_completed.fetch_add(1, Ordering::Release);
                        log::error!("[PdRdmaEngine] RDMA WRITE completion error: {err}");
                        Ok(())
                    }),
                },
            )
            .map_err(|err| {
                req_errors.fetch_add(1, Ordering::Release);
                req_completed.fetch_add(1, Ordering::Release);
                self.write_completed.fetch_add(1, Ordering::Release);
                pd_rdma_error("submit RDMA WRITE failed", err)
            })?;
        Ok(())
    }

    fn wait_for_pushes(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_writes(py, &req_id, Duration::from_secs(30))
    }

    fn push_done(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_writes(py, &req_id, Duration::from_secs(30))?;
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
            .map_err(|err| pd_rdma_error("submit IMM failed", err))?;
        if !py.detach(|| wait_atomic_count(&tx_counter, 1, Duration::from_secs(30))) {
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM timed out for req={req_id} completed={}",
                tx_counter.load(Ordering::Acquire)
            )));
        }
        let errors = err_counter.load(Ordering::Acquire);
        if errors != 0 {
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM failed for req={req_id} errors={errors}"
            )));
        }
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
        if self.finished_recving.lock().unwrap().contains(&req_id) {
            return Ok(());
        }
        let counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm_data))
            .clone();
        let done = py.detach(|| counter.wait_timeout(1, Duration::from_secs(30)));
        if !done {
            return Err(PegaFlowError::new_err(format!(
                "RDMA IMM wait timed out for req={req_id} remote_request_id={remote_request_id}"
            )));
        }
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
        if !py.detach(|| wait_atomic_count(&state.completed, submitted, timeout)) {
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE timed out for req={req_id} submitted={submitted} completed={}",
                state.completed.load(Ordering::Acquire)
            )));
        }
        let errors = state.errors.load(Ordering::Acquire);
        if errors != 0 {
            return Err(PegaFlowError::new_err(format!(
                "RDMA WRITE failed for req={req_id} errors={errors}"
            )));
        }
        Ok(())
    }

    fn block_slice_to_scatter_target(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        block: &Bound<'_, PyDict>,
        is_k: bool,
    ) -> PyResult<ScatterTarget> {
        let block_id: u64 = py_get(block, "block_id")?;
        let src_offset: u64 = py_get(block, "src_offset_bytes")?;
        let bytes: u64 = py_get(block, "bytes")?;
        if bytes == 0 || !bytes.is_multiple_of(local.block_bytes) {
            return Err(PyValueError::new_err(format!(
                "block slice bytes {bytes} must be a positive multiple of layer block_bytes {}",
                local.block_bytes
            )));
        }
        let remote_addr = if is_k {
            remote.k_by_block.get(&block_id)
        } else {
            remote.v_by_block.get(&block_id)
        }
        .copied()
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("remote block {block_id} is not registered"))
        })?;
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

pub(crate) fn add_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PdRdmaEngine>()?;
    m.add_class::<PdRdmaTestBuffer>()?;
    Ok(())
}
