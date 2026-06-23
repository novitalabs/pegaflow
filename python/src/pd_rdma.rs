use crate::{PegaFlowError, u64_to_usize};

use pegaflow_pd_wire as pd_wire;
use pegaflow_transfer::v2::{
    CudaDeviceId, Device, DomainAddress, DomainGroupRouting, FabricLibError, GroupTransferRouting,
    ImmCounter, ImmTransferRequest, MemoryRegionDescriptor, MemoryRegionHandle,
    MemoryRegionRemoteKey, RdmaEngine, ScatterTarget, ScatterTransferRequest, SmallVec,
    TransferCallback, TransferEngine, TransferEngineBuilder, TransferRequest, detect_topology,
};
use pegaflow_transfer::{
    ConnectionStatus as V1ConnectionStatus, HandshakeMetadata as V1HandshakeMetadata,
    MemoryRegion as V1MemoryRegion, TransferDesc as V1TransferDesc,
    TransferEngine as V1TransferEngine, TransferOp as V1TransferOp,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};
use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::atomic::{AtomicI64, Ordering},
    sync::{Arc, Mutex},
    task::{Context, Poll, Wake},
    time::{Duration, Instant},
};

const PD_RDMA_WRITE_WINDOW: i64 = 64;
const WAIT_SPINS_BEFORE_YIELD: u32 = 64;
const WAIT_YIELDS_BEFORE_SLEEP: u32 = 512;

fn pd_rdma_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PegaFlowError::new_err(format!("{context}: {err}"))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn instant_elapsed_ms(start: Instant, end: Instant) -> f64 {
    end.checked_duration_since(start)
        .map(duration_ms)
        .unwrap_or(0.0)
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

fn wait_imm_or_fail(
    done_counter: &ImmCounter,
    fail_counter: &ImmCounter,
    abort_counter: &ImmCounter,
    expected_done: u32,
    timeout: Duration,
) -> PdImmWaitResult {
    let deadline = Instant::now() + timeout;
    let mut waits = 0;
    loop {
        if fail_counter.wait_timeout(1, Duration::ZERO) {
            return PdImmWaitResult::Failed;
        }
        if abort_counter.wait_timeout(1, Duration::ZERO) {
            return PdImmWaitResult::Aborted;
        }
        if done_counter.wait_timeout(expected_done, Duration::ZERO) {
            return PdImmWaitResult::Done;
        }
        if Instant::now() >= deadline {
            return PdImmWaitResult::TimedOut;
        }
        wait_backoff(&mut waits);
    }
}

enum PdImmWaitResult {
    Done,
    Failed,
    Aborted,
    TimedOut,
}

#[derive(Clone, Copy)]
enum PdRequestImmKind {
    Done,
    Fail,
    Abort,
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

fn block_recv<T>(rx: mea::oneshot::Receiver<T>) -> Result<T, mea::oneshot::RecvError> {
    use std::future::IntoFuture;
    use std::pin::pin;

    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Arc::new(ThreadWaker(std::thread::current())).into();
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(rx.into_future());

    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn fail_counter_key(req_id: &str) -> String {
    format!("{req_id}#fail")
}

fn abort_counter_key(req_id: &str) -> String {
    format!("{req_id}#abort")
}

#[derive(Clone)]
struct PdLocalLayer {
    base_addr: u64,
    mr: MemoryRegionHandle,
    mr_desc: MemoryRegionDescriptor,
    regions: Vec<PdRemoteRegion>,
}

#[derive(Clone)]
struct PdRemoteRegion {
    base_addr: u64,
    block_len: u64,
    block_stride: u64,
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
    fail_imm_data: u32,
    abort_imm_data: u32,
    expected_imm_count: u32,
    layers: HashMap<u64, PdRemoteLayer>,
}

#[derive(Clone)]
struct PendingRdmaTransfers {
    submitted: i64,
    completed: Arc<AtomicI64>,
    errors: Arc<AtomicI64>,
    stats: Arc<Mutex<PendingRdmaTransferStats>>,
}

impl PendingRdmaTransfers {
    fn new() -> Self {
        Self {
            submitted: 0,
            completed: Arc::new(AtomicI64::new(0)),
            errors: Arc::new(AtomicI64::new(0)),
            stats: Arc::new(Mutex::new(PendingRdmaTransferStats::default())),
        }
    }
}

#[derive(Default)]
struct PendingRdmaTransferStats {
    bytes: u64,
    first_submit: Option<Instant>,
    last_submit: Option<Instant>,
    first_complete: Option<Instant>,
    last_complete: Option<Instant>,
    transfer_latency_count: u64,
    transfer_latency_sum_ms: f64,
    transfer_latency_max_ms: f64,
}

impl PendingRdmaTransferStats {
    fn record_submit(&mut self, bytes: u64, at: Instant) {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.first_submit.is_none() {
            self.first_submit = Some(at);
        }
        self.last_submit = Some(at);
    }

    fn record_complete(&mut self, at: Instant, latency_ms: f64) {
        if self.first_complete.is_none() {
            self.first_complete = Some(at);
        }
        self.last_complete = Some(at);
        self.transfer_latency_count = self.transfer_latency_count.saturating_add(1);
        self.transfer_latency_sum_ms += latency_ms;
        self.transfer_latency_max_ms = self.transfer_latency_max_ms.max(latency_ms);
    }

    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bytes", self.bytes)?;
        dict.set_item("has_submit", self.first_submit.is_some())?;
        dict.set_item("has_complete", self.last_complete.is_some())?;
        dict.set_item("transfer_latency_count", self.transfer_latency_count)?;
        dict.set_item("transfer_latency_sum_ms", self.transfer_latency_sum_ms)?;
        dict.set_item("transfer_latency_max_ms", self.transfer_latency_max_ms)?;
        dict.set_item("write_latency_count", self.transfer_latency_count)?;
        dict.set_item("write_latency_sum_ms", self.transfer_latency_sum_ms)?;
        dict.set_item("write_latency_max_ms", self.transfer_latency_max_ms)?;
        let active_s = self.transfer_latency_sum_ms / 1000.0;
        let active_gbps = if active_s > 0.0 {
            self.bytes as f64 * 8.0 / active_s / 1_000_000_000.0
        } else {
            0.0
        };
        dict.set_item("active_gbps", active_gbps)?;
        if let Some(first_submit) = self.first_submit {
            if let Some(last_submit) = self.last_submit {
                dict.set_item(
                    "submit_span_ms",
                    instant_elapsed_ms(first_submit, last_submit),
                )?;
            }
            if let Some(first_complete) = self.first_complete {
                dict.set_item(
                    "first_complete_ms",
                    instant_elapsed_ms(first_submit, first_complete),
                )?;
            }
            if let Some(last_complete) = self.last_complete {
                let xfer_window_ms = instant_elapsed_ms(first_submit, last_complete);
                dict.set_item("xfer_window_ms", xfer_window_ms)?;
                let xfer_s = xfer_window_ms / 1000.0;
                let xfer_gbps = if xfer_s > 0.0 {
                    self.bytes as f64 * 8.0 / xfer_s / 1_000_000_000.0
                } else {
                    0.0
                };
                dict.set_item("xfer_gbps", xfer_gbps)?;
            }
            if let (Some(last_submit), Some(last_complete)) = (self.last_submit, self.last_complete)
            {
                dict.set_item(
                    "completion_tail_ms",
                    instant_elapsed_ms(last_submit, last_complete),
                )?;
            }
        }
        Ok(dict)
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
    pending_writes: Mutex<HashMap<String, PendingRdmaTransfers>>,
    pending_reads: Mutex<HashMap<String, PendingRdmaTransfers>>,
    write_submitted: AtomicI64,
    write_completed: Arc<AtomicI64>,
    read_submitted: AtomicI64,
    read_completed: Arc<AtomicI64>,
    imm_to_req: Arc<Mutex<HashMap<u32, String>>>,
    finished_sending: Arc<Mutex<HashSet<String>>>,
    finished_recving: Arc<Mutex<HashSet<String>>>,
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
            pending_reads: Mutex::new(HashMap::new()),
            write_submitted: AtomicI64::new(0),
            write_completed: Arc::new(AtomicI64::new(0)),
            read_submitted: AtomicI64::new(0),
            read_completed: Arc::new(AtomicI64::new(0)),
            imm_to_req,
            finished_sending: Arc::new(Mutex::new(HashSet::new())),
            finished_recving,
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
            let mut regions = Vec::new();
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
                let block_stride = region
                    .get_item("block_stride")?
                    .map(|value| value.extract::<u64>())
                    .transpose()?
                    .unwrap_or(block_len);
                if block_stride < block_len {
                    return Err(PyValueError::new_err(format!(
                        "local layer {layer_idx} region {region_idx} block_stride={block_stride} is smaller than block_len={block_len}"
                    )));
                }
                let end = max_block_id
                    .checked_mul(block_stride)
                    .and_then(|byte_offset| byte_offset.checked_add(block_len))
                    .and_then(|byte_len| base_addr.checked_add(byte_len))
                    .ok_or_else(|| PyValueError::new_err("local layer address range overflow"))?;
                min_addr = min_addr.min(base_addr);
                max_addr = max_addr.max(end);
                regions.push(PdRemoteRegion {
                    base_addr,
                    block_len,
                    block_stride,
                });
            }
            if regions.is_empty() {
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
            let region_count = regions.len();
            local_layers.insert(
                layer_idx,
                PdLocalLayer {
                    base_addr: min_addr,
                    mr,
                    mr_desc: mr_desc.clone(),
                    regions,
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
        handshake_json: String,
    ) -> PyResult<()> {
        // The wire crate owns the schema: parsing implies full validation.
        let handshake = py
            .detach(|| pd_wire::Handshake::from_json(&handshake_json))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let remote_request_id = handshake.request_id.clone();
        let imm = handshake.imm_id;
        let fail_imm = handshake.fail_imm_id();
        let abort_imm = handshake.abort_imm_id();
        let expected_imm_count = handshake.expected_imm_count.get();

        let mut layers = HashMap::new();
        for layer in &handshake.layers {
            let block_ids = handshake.layer_block_ids(layer).ok_or_else(|| {
                PyValueError::new_err(format!("remote layer {} has no block_ids", layer.layer_idx))
            })?;
            let mr_desc = layer.mr_desc.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!("remote layer {} missing mr_desc", layer.layer_idx))
            })?;
            layers.insert(
                layer.layer_idx,
                PdRemoteLayer {
                    mr_desc: mr_desc_from_wire(mr_desc)?,
                    allowed_block_ids: block_ids.iter().copied().collect::<HashSet<_>>(),
                    regions: layer
                        .regions
                        .iter()
                        .map(|region| PdRemoteRegion {
                            base_addr: region.base_addr,
                            block_len: region.block_len,
                            block_stride: region.block_stride(),
                        })
                        .collect(),
                },
            );
        }

        self.imm_to_req.lock().unwrap().insert(imm, req_id.clone());
        self.imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm));
        self.imm_counters
            .lock()
            .unwrap()
            .entry(fail_counter_key(&req_id))
            .or_insert_with(|| self.engine.get_imm_counter(fail_imm));
        self.imm_counters
            .lock()
            .unwrap()
            .entry(abort_counter_key(&req_id))
            .or_insert_with(|| self.engine.get_imm_counter(abort_imm));

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
                fail_imm_data: fail_imm,
                abort_imm_data: abort_imm,
                expected_imm_count,
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
                let targets =
                    self.block_slice_to_scatter_targets(&local, &remote, &region, region_idx)?;
                for target in targets {
                    dst_bytes = dst_bytes.checked_add(target.length).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "RDMA scatter byte count overflow for req={req_id} layer={layer_idx}"
                        ))
                    })?;
                    dsts.push(target);
                }
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
        self.write_submitted.fetch_add(1, Ordering::Release);
        let window_ms = duration_ms(window_start.elapsed());
        let submit_at = Instant::now();
        let (req_completed, req_errors, req_stats) = {
            let mut pending = self.pending_writes.lock().unwrap();
            let state = pending
                .entry(req_id.clone())
                .or_insert_with(PendingRdmaTransfers::new);
            state.submitted += 1;
            state
                .stats
                .lock()
                .unwrap()
                .record_submit(dst_bytes, submit_at);
            (
                Arc::clone(&state.completed),
                Arc::clone(&state.errors),
                Arc::clone(&state.stats),
            )
        };
        let scatter_targets = dsts.len();
        let global_completed = Arc::clone(&self.write_completed);
        let done_completed = Arc::clone(&req_completed);
        let error_completed = Arc::clone(&req_completed);
        let error_counter = Arc::clone(&req_errors);
        let done_stats = Arc::clone(&req_stats);
        let error_stats = Arc::clone(&req_stats);
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
                        let latency_ms = duration_ms(callback_start.elapsed());
                        done_stats
                            .lock()
                            .unwrap()
                            .record_complete(Instant::now(), latency_ms);
                        done_completed.fetch_add(1, Ordering::Release);
                        global_completed.fetch_add(1, Ordering::Release);
                        log::debug!(
                            "[PdRdmaEngine] RDMA WRITE completed req={} layer={} bytes={} latency_ms={:.3}",
                            done_req_id,
                            done_layer_idx,
                            done_bytes,
                            latency_ms,
                        );
                        Ok(())
                    }),
                    on_error: Box::new(move |err: FabricLibError| {
                        let latency_ms = duration_ms(callback_start.elapsed());
                        error_stats
                            .lock()
                            .unwrap()
                            .record_complete(Instant::now(), latency_ms);
                        error_counter.fetch_add(1, Ordering::Release);
                        error_completed.fetch_add(1, Ordering::Release);
                        error_global_completed.fetch_add(1, Ordering::Release);
                        log::error!(
                            "[PdRdmaEngine] RDMA WRITE completion error req={} layer={} bytes={} latency_ms={:.3} err={err}",
                            error_req_id,
                            error_layer_idx,
                            error_bytes,
                            latency_ms,
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
        log::debug!(
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
            self.write_submitted.load(Ordering::Acquire),
            self.write_completed.load(Ordering::Acquire),
        );
        Ok(())
    }

    fn wait_for_pushes(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_transfers(
            py,
            &self.pending_writes,
            "WRITE",
            &req_id,
            Duration::from_secs(30),
        )
    }

    fn push_done(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_transfers(
            py,
            &self.pending_writes,
            "WRITE",
            &req_id,
            Duration::from_secs(30),
        )?;
        self.send_request_imm(py, &req_id, PdRequestImmKind::Done)?;
        self.finished_sending.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn write_stats<'py>(&self, py: Python<'py>, req_id: String) -> PyResult<Bound<'py, PyDict>> {
        self.transfer_stats(py, &self.pending_writes, &req_id)
    }

    fn pull_layer(
        &self,
        py: Python<'_>,
        req_id: String,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<()> {
        let total_start = Instant::now();
        let input_blocks = blocks.len();
        let (local, reads, read_bytes) =
            self.collect_read_targets(py, &req_id, layer_idx, blocks)?;
        if reads.is_empty() {
            return Ok(());
        }
        self.submit_read_batch(
            py,
            &req_id,
            layer_idx,
            input_blocks,
            reads,
            read_bytes,
            local.mr,
            duration_ms(total_start.elapsed()),
        )
    }

    fn pull_layers(
        &self,
        py: Python<'_>,
        req_id: String,
        layers: Vec<(u64, Vec<Py<PyDict>>)>,
    ) -> PyResult<()> {
        let total_start = Instant::now();
        let mut submitted_layers = 0_usize;
        for (layer_idx, blocks) in layers {
            let input_blocks = blocks.len();
            let (local, reads, read_bytes) =
                self.collect_read_targets(py, &req_id, layer_idx, blocks)?;
            if reads.is_empty() {
                continue;
            }
            self.submit_read_batch(
                py,
                &req_id,
                layer_idx,
                input_blocks,
                reads,
                read_bytes,
                local.mr,
                duration_ms(total_start.elapsed()),
            )?;
            submitted_layers += 1;
        }
        log::debug!(
            "[PdRdmaEngine] RDMA READ layers submitted req={} layers={} total_ms={:.3} submitted={} completed={}",
            req_id,
            submitted_layers,
            duration_ms(total_start.elapsed()),
            self.read_submitted.load(Ordering::Acquire),
            self.read_completed.load(Ordering::Acquire),
        );
        Ok(())
    }

    fn wait_for_pulls(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.wait_for_request_transfers(
            py,
            &self.pending_reads,
            "READ",
            &req_id,
            Duration::from_secs(30),
        )?;
        self.finished_recving.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn read_stats<'py>(&self, py: Python<'py>, req_id: String) -> PyResult<Bound<'py, PyDict>> {
        self.transfer_stats(py, &self.pending_reads, &req_id)
    }

    fn fail_request(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.send_request_imm(py, &req_id, PdRequestImmKind::Fail)
    }

    fn abort_request(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        self.send_request_imm(py, &req_id, PdRequestImmKind::Abort)
    }

    fn wait_done(&self, py: Python<'_>, req_id: String) -> PyResult<()> {
        if self.finished_recving.lock().unwrap().contains(&req_id) {
            log::info!("[PdRdmaEngine] RDMA IMM wait already done req={}", req_id);
            return Ok(());
        }
        let (remote_request_id, imm_data, fail_imm_data, abort_imm_data, expected_imm_count) = self
            .remote_requests
            .lock()
            .unwrap()
            .get(&req_id)
            .map(|request| {
                (
                    request.remote_request_id.clone(),
                    request.imm_data,
                    request.fail_imm_data,
                    request.abort_imm_data,
                    request.expected_imm_count,
                )
            })
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} is not registered"))
            })?;
        let counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(req_id.clone())
            .or_insert_with(|| self.engine.get_imm_counter(imm_data))
            .clone();
        let start = Instant::now();
        log::info!(
            "[PdRdmaEngine] RDMA IMM wait start req={} remote_req={} imm={} fail_imm={} abort_imm={} expected={}",
            req_id,
            remote_request_id,
            imm_data,
            fail_imm_data,
            abort_imm_data,
            expected_imm_count,
        );
        let fail_counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(fail_counter_key(&req_id))
            .or_insert_with(|| self.engine.get_imm_counter(fail_imm_data))
            .clone();
        let abort_counter = self
            .imm_counters
            .lock()
            .unwrap()
            .entry(abort_counter_key(&req_id))
            .or_insert_with(|| self.engine.get_imm_counter(abort_imm_data))
            .clone();
        let result = py.detach(|| {
            wait_imm_or_fail(
                &counter,
                &fail_counter,
                &abort_counter,
                expected_imm_count,
                Duration::from_secs(30),
            )
        });
        match result {
            PdImmWaitResult::Done => {}
            PdImmWaitResult::Failed => {
                log::error!(
                    "[PdRdmaEngine] RDMA IMM wait failed req={} remote_req={} imm={} fail_imm={} expected={} wait_ms={:.3}",
                    req_id,
                    remote_request_id,
                    imm_data,
                    fail_imm_data,
                    expected_imm_count,
                    duration_ms(start.elapsed()),
                );
                return Err(PegaFlowError::new_err(format!(
                    "RDMA IMM wait failed for req={req_id} remote_request_id={remote_request_id}"
                )));
            }
            PdImmWaitResult::Aborted => {
                log::info!(
                    "[PdRdmaEngine] RDMA IMM wait abort-acked req={} remote_req={} imm={} abort_imm={} wait_ms={:.3}",
                    req_id,
                    remote_request_id,
                    imm_data,
                    abort_imm_data,
                    duration_ms(start.elapsed()),
                );
            }
            PdImmWaitResult::TimedOut => {
                log::error!(
                    "[PdRdmaEngine] RDMA IMM wait timed out req={} remote_req={} imm={} fail_imm={} expected={} wait_ms={:.3}",
                    req_id,
                    remote_request_id,
                    imm_data,
                    fail_imm_data,
                    expected_imm_count,
                    duration_ms(start.elapsed()),
                );
                return Err(PegaFlowError::new_err(format!(
                    "RDMA IMM wait timed out for req={req_id} remote_request_id={remote_request_id}"
                )));
            }
        }
        log::info!(
            "[PdRdmaEngine] RDMA IMM wait done req={} remote_req={} imm={} expected={} wait_ms={:.3}",
            req_id,
            remote_request_id,
            imm_data,
            expected_imm_count,
            duration_ms(start.elapsed()),
        );
        self.finished_recving.lock().unwrap().insert(req_id);
        Ok(())
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
            self.engine.remove_imm_count(request.fail_imm_data);
            self.engine.remove_imm_count(request.abort_imm_data);
            self.imm_to_req.lock().unwrap().remove(&request.imm_data);
        }
        self.imm_counters.lock().unwrap().remove(&req_id);
        self.imm_counters
            .lock()
            .unwrap()
            .remove(&fail_counter_key(&req_id));
        self.imm_counters
            .lock()
            .unwrap()
            .remove(&abort_counter_key(&req_id));
        self.pending_writes.lock().unwrap().remove(&req_id);
        self.pending_reads.lock().unwrap().remove(&req_id);
        self.finished_sending.lock().unwrap().remove(&req_id);
        self.finished_recving.lock().unwrap().remove(&req_id);
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
}

impl PdRdmaEngine {
    fn send_request_imm(
        &self,
        py: Python<'_>,
        req_id: &str,
        kind: PdRequestImmKind,
    ) -> PyResult<()> {
        let start = Instant::now();
        let (remote, imm_data) = {
            let requests = self.remote_requests.lock().unwrap();
            let request = requests.get(req_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} is not registered"))
            })?;
            let remote = request.layers.values().next().cloned().ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} has no layers"))
            })?;
            let imm_data = match kind {
                PdRequestImmKind::Done => request.imm_data,
                PdRequestImmKind::Fail => request.fail_imm_data,
                PdRequestImmKind::Abort => request.abort_imm_data,
            };
            (remote, imm_data)
        };
        let label = match kind {
            PdRequestImmKind::Done => "IMM",
            PdRequestImmKind::Fail => "fail IMM",
            PdRequestImmKind::Abort => "abort IMM",
        };
        let tx_counter = Arc::new(AtomicI64::new(0));
        let err_counter = Arc::new(AtomicI64::new(0));
        log::info!(
            "[PdRdmaEngine] RDMA {} submit start req={} imm={} domains={}",
            label,
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
                    "[PdRdmaEngine] RDMA {} submit failed req={} imm={} err={err}",
                    label,
                    req_id,
                    imm_data,
                );
                pd_rdma_error(&format!("submit {label} failed"), err)
            })?;
        if !py.detach(|| wait_atomic_count(&tx_counter, 1, Duration::from_secs(30))) {
            log::error!(
                "[PdRdmaEngine] RDMA {} timed out req={} imm={} completed={} errors={} wait_ms={:.3}",
                label,
                req_id,
                imm_data,
                tx_counter.load(Ordering::Acquire),
                err_counter.load(Ordering::Acquire),
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA {label} timed out for req={req_id} completed={}",
                tx_counter.load(Ordering::Acquire)
            )));
        }
        let errors = err_counter.load(Ordering::Acquire);
        if errors != 0 {
            log::error!(
                "[PdRdmaEngine] RDMA {} failed req={} imm={} completed={} errors={} wait_ms={:.3}",
                label,
                req_id,
                imm_data,
                tx_counter.load(Ordering::Acquire),
                errors,
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA {label} failed for req={req_id} errors={errors}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] RDMA {} done req={} imm={} wait_ms={:.3}",
            label,
            req_id,
            imm_data,
            duration_ms(start.elapsed()),
        );
        Ok(())
    }

    fn wait_for_request_transfers(
        &self,
        py: Python<'_>,
        pending: &Mutex<HashMap<String, PendingRdmaTransfers>>,
        label: &str,
        req_id: &str,
        timeout: Duration,
    ) -> PyResult<()> {
        let Some(state) = pending.lock().unwrap().get(req_id).cloned() else {
            return Ok(());
        };
        let submitted = state.submitted;
        if submitted == 0 {
            return Ok(());
        }
        let start = Instant::now();
        log::info!(
            "[PdRdmaEngine] RDMA {} wait start req={} submitted={} completed={} errors={}",
            label,
            req_id,
            submitted,
            state.completed.load(Ordering::Acquire),
            state.errors.load(Ordering::Acquire),
        );
        if !py.detach(|| wait_atomic_count(&state.completed, submitted, timeout)) {
            log::error!(
                "[PdRdmaEngine] RDMA {} wait timeout req={} submitted={} completed={} errors={} wait_ms={:.3}",
                label,
                req_id,
                submitted,
                state.completed.load(Ordering::Acquire),
                state.errors.load(Ordering::Acquire),
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA {label} timed out for req={req_id} submitted={submitted} completed={}",
                state.completed.load(Ordering::Acquire)
            )));
        }
        let errors = state.errors.load(Ordering::Acquire);
        if errors != 0 {
            log::error!(
                "[PdRdmaEngine] RDMA {} wait failed req={} submitted={} completed={} errors={} wait_ms={:.3}",
                label,
                req_id,
                submitted,
                state.completed.load(Ordering::Acquire),
                errors,
                duration_ms(start.elapsed()),
            );
            return Err(PegaFlowError::new_err(format!(
                "RDMA {label} failed for req={req_id} errors={errors}"
            )));
        }
        log::info!(
            "[PdRdmaEngine] RDMA {} wait done req={} submitted={} completed={} wait_ms={:.3}",
            label,
            req_id,
            submitted,
            state.completed.load(Ordering::Acquire),
            duration_ms(start.elapsed()),
        );
        Ok(())
    }

    fn collect_read_targets(
        &self,
        py: Python<'_>,
        req_id: &str,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<(PdLocalLayer, Vec<ScatterTarget>, u64)> {
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
            .get(req_id)
            .and_then(|request| request.layers.get(&layer_idx).cloned())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "remote layer {layer_idx} for req {req_id} is not registered"
                ))
            })?;

        let mut reads = Vec::with_capacity(blocks.len() * remote.regions.len());
        let mut read_bytes = 0_u64;
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
                let targets =
                    self.block_slice_to_read_targets(&local, &remote, &region, region_idx)?;
                for target in targets {
                    read_bytes = read_bytes.checked_add(target.length).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "RDMA read byte count overflow for req={req_id} layer={layer_idx}"
                        ))
                    })?;
                    reads.push(target);
                }
            }
        }
        Ok((local, reads, read_bytes))
    }

    fn submit_read_batch(
        &self,
        _py: Python<'_>,
        _req_id: &str,
        _layer_idx: u64,
        input_blocks: usize,
        reads: Vec<ScatterTarget>,
        read_bytes: u64,
        local_mr: MemoryRegionHandle,
        convert_ms: f64,
    ) -> PyResult<()> {
        let _ = (input_blocks, reads, read_bytes, local_mr, convert_ms);
        Err(PyRuntimeError::new_err(
            "PdRdmaEngine v2 does not support RDMA READ; use PdRdmaV1Engine for Pega NIXL pull",
        ))
    }

    fn transfer_stats<'py>(
        &self,
        py: Python<'py>,
        pending: &Mutex<HashMap<String, PendingRdmaTransfers>>,
        req_id: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let Some(state) = pending.lock().unwrap().get(req_id).cloned() else {
            let dict = PyDict::new(py);
            dict.set_item("submitted", 0_i64)?;
            dict.set_item("completed", 0_i64)?;
            dict.set_item("errors", 0_i64)?;
            dict.set_item("bytes", 0_u64)?;
            dict.set_item("has_submit", false)?;
            dict.set_item("has_complete", false)?;
            return Ok(dict);
        };
        let dict = state.stats.lock().unwrap().to_py_dict(py)?;
        dict.set_item("submitted", state.submitted)?;
        dict.set_item("completed", state.completed.load(Ordering::Acquire))?;
        dict.set_item("errors", state.errors.load(Ordering::Acquire))?;
        Ok(dict)
    }

    fn block_slice_to_scatter_targets(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        block: &Bound<'_, PyDict>,
        region_idx: usize,
    ) -> PyResult<Vec<ScatterTarget>> {
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
        if block_count == 1 || region.block_stride == region.block_len {
            return Ok(vec![self.make_scatter_target(
                local, remote, block_id, src_offset, bytes, region,
            )?]);
        }

        let mut targets = Vec::with_capacity(u64_to_usize(block_count, "block_count")?);
        for offset in 0..block_count {
            let remote_block_id = block_id
                .checked_add(offset)
                .ok_or_else(|| PyValueError::new_err("remote block id overflow"))?;
            let source_offset = src_offset
                .checked_add(
                    offset
                        .checked_mul(region.block_len)
                        .ok_or_else(|| PyValueError::new_err("source offset overflow"))?,
                )
                .ok_or_else(|| PyValueError::new_err("source offset overflow"))?;
            targets.push(self.make_scatter_target(
                local,
                remote,
                remote_block_id,
                source_offset,
                region.block_len,
                region,
            )?);
        }
        Ok(targets)
    }

    fn block_slice_to_read_targets(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        block: &Bound<'_, PyDict>,
        region_idx: usize,
    ) -> PyResult<Vec<ScatterTarget>> {
        let local_block_id: u64 = py_get(block, "block_id")?;
        let remote_offset: u64 = py_get(block, "src_offset_bytes")?;
        let bytes: u64 = py_get(block, "bytes")?;
        if bytes == 0 {
            return Err(PyValueError::new_err("block slice bytes must be positive"));
        }
        let region = remote.regions.get(region_idx).ok_or_else(|| {
            PyValueError::new_err(format!("remote region {region_idx} is not registered"))
        })?;
        let local_region = local.regions.get(region_idx).ok_or_else(|| {
            PyValueError::new_err(format!("local region {region_idx} is not registered"))
        })?;
        if !bytes.is_multiple_of(region.block_len) {
            return Err(PyValueError::new_err(format!(
                "block slice bytes {bytes} must be a positive multiple of remote region block_len {}",
                region.block_len
            )));
        }
        let block_count = bytes / region.block_len;
        if block_count == 1 || region.block_stride == region.block_len {
            return Ok(vec![self.make_read_scatter_target(
                local,
                remote,
                local_region,
                local_block_id,
                remote_offset,
                bytes,
                region,
            )?]);
        }

        let mut targets = Vec::with_capacity(u64_to_usize(block_count, "block_count")?);
        for offset in 0..block_count {
            let local_block_id = local_block_id
                .checked_add(offset)
                .ok_or_else(|| PyValueError::new_err("local block id overflow"))?;
            let remote_offset = remote_offset
                .checked_add(
                    offset
                        .checked_mul(region.block_len)
                        .ok_or_else(|| PyValueError::new_err("remote offset overflow"))?,
                )
                .ok_or_else(|| PyValueError::new_err("remote offset overflow"))?;
            targets.push(self.make_read_scatter_target(
                local,
                remote,
                local_region,
                local_block_id,
                remote_offset,
                region.block_len,
                region,
            )?);
        }
        Ok(targets)
    }

    fn make_scatter_target(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        block_id: u64,
        src_offset: u64,
        bytes: u64,
        region: &PdRemoteRegion,
    ) -> PyResult<ScatterTarget> {
        let remote_addr = block_id
            .checked_mul(region.block_stride)
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

    fn make_read_scatter_target(
        &self,
        local: &PdLocalLayer,
        remote: &PdRemoteLayer,
        local_region: &PdRemoteRegion,
        local_block_id: u64,
        remote_offset: u64,
        bytes: u64,
        region: &PdRemoteRegion,
    ) -> PyResult<ScatterTarget> {
        let remote_source_absolute = remote
            .mr_desc
            .ptr
            .checked_add(remote_offset)
            .ok_or_else(|| PyValueError::new_err("remote source address overflow"))?;
        let remote_block_offset = remote_source_absolute
            .checked_sub(region.base_addr)
            .ok_or_else(|| PyRuntimeError::new_err("remote source address is below region base"))?;
        if remote_block_offset % region.block_stride != 0 {
            return Err(PyValueError::new_err(
                "remote source address is not aligned to remote block stride",
            ));
        }
        let remote_block_id = remote_block_offset / region.block_stride;
        let block_count = bytes / region.block_len;
        for offset in 0..block_count {
            let block_id = remote_block_id
                .checked_add(offset)
                .ok_or_else(|| PyValueError::new_err("remote block id overflow"))?;
            if !remote.allowed_block_ids.contains(&block_id) {
                return Err(PyRuntimeError::new_err(format!(
                    "remote block {block_id} is not registered"
                )));
            }
        }
        let local_addr = local_block_id
            .checked_mul(local_region.block_stride)
            .and_then(|offset| local_region.base_addr.checked_add(offset))
            .ok_or_else(|| PyValueError::new_err("local block address overflow"))?;
        if local_addr < local.mr_desc.ptr {
            return Err(PyValueError::new_err(
                "local block address is below local MR base",
            ));
        }
        Ok(ScatterTarget {
            dst_mr: remote.mr_desc.clone(),
            dst_offset: remote_offset,
            src_offset: local_addr - local.mr_desc.ptr,
            length: bytes,
        })
    }
}

#[derive(Clone)]
struct V1LocalLayer {
    regions: Vec<PdRemoteRegion>,
}

#[derive(Clone)]
struct V1RemoteLayer {
    allowed_block_ids: HashSet<u64>,
    regions: Vec<PdRemoteRegion>,
}

struct V1RemoteRequest {
    peer_key: String,
    layers: HashMap<u64, V1RemoteLayer>,
}

#[pyclass]
struct PdRdmaV1Engine {
    engine: Arc<V1TransferEngine>,
    local_layers: Mutex<HashMap<u64, V1LocalLayer>>,
    remote_requests: Mutex<HashMap<String, V1RemoteRequest>>,
    pending_local_meta: Mutex<HashMap<String, V1HandshakeMetadata>>,
    pending_writes: Mutex<HashMap<String, PendingRdmaTransfers>>,
    pending_reads: Mutex<HashMap<String, PendingRdmaTransfers>>,
    finished_sending: Arc<Mutex<HashSet<String>>>,
    finished_recving: Arc<Mutex<HashSet<String>>>,
}

#[pymethods]
impl PdRdmaV1Engine {
    #[new]
    #[pyo3(signature = (*, cuda_device = 0, domains = None, qps_per_peer = 1, local_peer_key = "rank:0"))]
    fn new(
        cuda_device: u8,
        domains: Option<Vec<String>>,
        qps_per_peer: usize,
        local_peer_key: &str,
    ) -> PyResult<Self> {
        let domains = domains.unwrap_or_default();
        if domains.is_empty() {
            return Err(PyValueError::new_err("PdRdmaV1Engine requires domains"));
        }
        if local_peer_key.is_empty() {
            return Err(PyValueError::new_err(
                "PdRdmaV1Engine local_peer_key must not be empty",
            ));
        }
        let qps_per_peer = qps_per_peer.max(1);
        let engine = Arc::new(
            V1TransferEngine::new(&domains, qps_per_peer)
                .map_err(|err| pd_rdma_error("v1 transfer engine build failed", err))?,
        );
        log::info!(
            "[PdRdmaV1Engine] selected cuda={} domains=[{}] qps_per_peer={}",
            cuda_device,
            domains.join(","),
            qps_per_peer,
        );
        Ok(Self {
            engine,
            local_layers: Mutex::new(HashMap::new()),
            remote_requests: Mutex::new(HashMap::new()),
            pending_local_meta: Mutex::new(HashMap::new()),
            pending_writes: Mutex::new(HashMap::new()),
            pending_reads: Mutex::new(HashMap::new()),
            finished_sending: Arc::new(Mutex::new(HashSet::new())),
            finished_recving: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    fn register_local_layers(
        &self,
        py: Python<'_>,
        layers: Vec<Py<PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut parsed_layers = Vec::with_capacity(layers.len());
        let mut memory_regions = Vec::with_capacity(layers.len());
        for layer in &layers {
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
            let mut regions = Vec::new();
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
                let block_stride = region
                    .get_item("block_stride")?
                    .map(|value| value.extract::<u64>())
                    .transpose()?
                    .unwrap_or(block_len);
                let end = max_block_id
                    .checked_mul(block_stride)
                    .and_then(|byte_offset| byte_offset.checked_add(block_len))
                    .and_then(|byte_len| base_addr.checked_add(byte_len))
                    .ok_or_else(|| PyValueError::new_err("local layer address range overflow"))?;
                min_addr = min_addr.min(base_addr);
                max_addr = max_addr.max(end);
                regions.push(PdRemoteRegion {
                    base_addr,
                    block_len,
                    block_stride,
                });
            }
            if regions.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "local layer {layer_idx} has no regions"
                )));
            }
            let len = max_addr
                .checked_sub(min_addr)
                .filter(|len| *len > 0)
                .ok_or_else(|| PyValueError::new_err("invalid layer address range"))?;
            let ptr = nonnull_from_u64(min_addr, "base_addr")?.cast::<u8>();
            memory_regions.push(V1MemoryRegion {
                ptr,
                len: u64_to_usize(len, "layer length")?,
            });
            parsed_layers.push((layer_idx, V1LocalLayer { regions }));
        }

        self.engine
            .register_memory(&memory_regions)
            .map_err(|err| pd_rdma_error("v1 register_memory failed", err))?;
        let mut local_layers = self.local_layers.lock().unwrap();
        let mut registered_layers = Vec::with_capacity(layers.len());
        for (layer, (layer_idx, parsed)) in layers.into_iter().zip(parsed_layers.into_iter()) {
            local_layers.insert(layer_idx, parsed);
            let layer = layer.bind(py);
            let out = PyDict::new(py);
            for (key, value) in layer.iter() {
                out.set_item(key, value)?;
            }
            registered_layers.push(out.into_any().unbind());
        }
        Ok(registered_layers)
    }

    fn prepare_peer(&self, peer_key: String) -> PyResult<String> {
        if peer_key.is_empty() {
            return Err(PyValueError::new_err("v1 peer_key must not be empty"));
        }
        let local_meta = self
            .prepare_or_get_local_meta(&peer_key)
            .map_err(|err| pd_rdma_error("v1 prepare_peer failed", err))?;
        Ok(hex_encode(&local_meta.to_bytes()))
    }

    fn complete_peer(&self, peer_key: String, remote_metadata: String) -> PyResult<()> {
        if peer_key.is_empty() {
            return Err(PyValueError::new_err("v1 peer_key must not be empty"));
        }
        let remote_meta = v1_handshake_metadata_from_hex(&remote_metadata)?;
        let local_meta = self
            .prepare_or_get_local_meta(&peer_key)
            .map_err(|err| pd_rdma_error("v1 prepare local handshake failed", err))?;
        self.engine
            .complete_handshake(&peer_key, &local_meta, &remote_meta)
            .map_err(|err| pd_rdma_error("v1 complete_peer failed", err))?;
        self.pending_local_meta.lock().unwrap().remove(&peer_key);
        Ok(())
    }

    fn register_remote(
        &self,
        py: Python<'_>,
        req_id: String,
        handshake_json: String,
        peer_key: String,
    ) -> PyResult<()> {
        if peer_key.is_empty() {
            return Err(PyValueError::new_err("v1 peer_key must not be empty"));
        }
        let handshake = py
            .detach(|| pd_wire::Handshake::from_json(&handshake_json))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut layers = HashMap::new();
        for layer in &handshake.layers {
            let block_ids = handshake.layer_block_ids(layer).ok_or_else(|| {
                PyValueError::new_err(format!("remote layer {} has no block_ids", layer.layer_idx))
            })?;
            layers.insert(
                layer.layer_idx,
                V1RemoteLayer {
                    allowed_block_ids: block_ids.iter().copied().collect::<HashSet<_>>(),
                    regions: layer
                        .regions
                        .iter()
                        .map(|region| PdRemoteRegion {
                            base_addr: region.base_addr,
                            block_len: region.block_len,
                            block_stride: region.block_stride(),
                        })
                        .collect(),
                },
            );
        }
        self.remote_requests
            .lock()
            .unwrap()
            .insert(req_id, V1RemoteRequest { peer_key, layers });
        Ok(())
    }

    fn push_layer(
        &self,
        py: Python<'_>,
        req_id: String,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<()> {
        let descs = self.collect_write_descs(py, &req_id, layer_idx, blocks)?;
        self.submit_v1_transfer(&req_id, V1TransferOp::Write, descs, &self.pending_writes)
    }

    fn wait_for_pushes(&self, _py: Python<'_>, _req_id: String) -> PyResult<()> {
        Ok(())
    }

    fn pull_layer(
        &self,
        py: Python<'_>,
        req_id: String,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<()> {
        let descs = self.collect_read_descs(py, &req_id, layer_idx, blocks)?;
        self.submit_v1_transfer(&req_id, V1TransferOp::Read, descs, &self.pending_reads)
    }

    fn pull_layers(
        &self,
        py: Python<'_>,
        req_id: String,
        layers: Vec<(u64, Vec<Py<PyDict>>)>,
    ) -> PyResult<()> {
        let mut all_descs = Vec::new();
        for (layer_idx, blocks) in layers {
            all_descs.extend(self.collect_read_descs(py, &req_id, layer_idx, blocks)?);
        }
        self.submit_v1_transfer(&req_id, V1TransferOp::Read, all_descs, &self.pending_reads)
    }

    fn wait_for_pulls(&self, _py: Python<'_>, req_id: String) -> PyResult<()> {
        self.finished_recving.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn push_done(&self, _py: Python<'_>, req_id: String) -> PyResult<()> {
        self.finished_sending.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn write_stats<'py>(&self, py: Python<'py>, req_id: String) -> PyResult<Bound<'py, PyDict>> {
        self.transfer_stats(py, &self.pending_writes, &req_id)
    }

    fn read_stats<'py>(&self, py: Python<'py>, req_id: String) -> PyResult<Bound<'py, PyDict>> {
        self.transfer_stats(py, &self.pending_reads, &req_id)
    }

    fn fail_request(&self, _py: Python<'_>, _req_id: String) -> PyResult<()> {
        Ok(())
    }

    fn abort_request(&self, _py: Python<'_>, req_id: String) -> PyResult<()> {
        self.finished_recving.lock().unwrap().insert(req_id);
        Ok(())
    }

    fn wait_done(&self, _py: Python<'_>, _req_id: String) -> PyResult<()> {
        Ok(())
    }

    fn pop_finished_sending(&self) -> HashSet<String> {
        std::mem::take(&mut *self.finished_sending.lock().unwrap())
    }

    fn pop_finished_recving(&self) -> HashSet<String> {
        std::mem::take(&mut *self.finished_recving.lock().unwrap())
    }

    fn close_request(&self, req_id: String) {
        self.remote_requests.lock().unwrap().remove(&req_id);
        self.pending_writes.lock().unwrap().remove(&req_id);
        self.pending_reads.lock().unwrap().remove(&req_id);
        self.finished_sending.lock().unwrap().remove(&req_id);
        self.finished_recving.lock().unwrap().remove(&req_id);
    }

    fn num_domains(&self) -> usize {
        1
    }

    fn num_groups(&self) -> usize {
        1
    }

    fn aggregated_link_speed(&self) -> u64 {
        400_000_000_000
    }
}

impl PdRdmaV1Engine {
    fn prepare_or_get_local_meta(
        &self,
        peer_key: &str,
    ) -> pegaflow_transfer::Result<V1HandshakeMetadata> {
        if let Some(meta) = self.engine.local_meta_for(peer_key) {
            return Ok(meta);
        }
        if let Some(meta) = self
            .pending_local_meta
            .lock()
            .unwrap()
            .get(peer_key)
            .cloned()
        {
            return Ok(meta);
        }
        match self.engine.get_or_prepare(peer_key)? {
            V1ConnectionStatus::Existing => self.engine.local_meta_for(peer_key).ok_or(
                pegaflow_transfer::TransferError::InvalidArgument("v1 local metadata unavailable"),
            ),
            V1ConnectionStatus::Connecting => self
                .pending_local_meta
                .lock()
                .unwrap()
                .get(peer_key)
                .cloned()
                .ok_or(pegaflow_transfer::TransferError::InvalidArgument(
                    "v1 pending local metadata unavailable",
                )),
            V1ConnectionStatus::Prepared(meta) => {
                self.pending_local_meta
                    .lock()
                    .unwrap()
                    .insert(peer_key.to_string(), meta.clone());
                Ok(meta)
            }
        }
    }

    fn submit_v1_transfer(
        &self,
        req_id: &str,
        op: V1TransferOp,
        descs: Vec<V1TransferDesc>,
        pending: &Mutex<HashMap<String, PendingRdmaTransfers>>,
    ) -> PyResult<()> {
        if descs.is_empty() {
            return Ok(());
        }
        let peer_key = self
            .remote_requests
            .lock()
            .unwrap()
            .get(req_id)
            .map(|request| request.peer_key.clone())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("remote req {req_id} is not registered"))
            })?;
        let bytes = descs.iter().map(|desc| desc.len as u64).sum::<u64>();
        let submit_at = Instant::now();
        let (req_completed, req_errors, req_stats) = {
            let mut pending = pending.lock().unwrap();
            let state = pending
                .entry(req_id.to_string())
                .or_insert_with(PendingRdmaTransfers::new);
            state.submitted += 1;
            state.stats.lock().unwrap().record_submit(bytes, submit_at);
            (
                Arc::clone(&state.completed),
                Arc::clone(&state.errors),
                Arc::clone(&state.stats),
            )
        };
        let callback_start = Instant::now();
        let receivers = self
            .engine
            .batch_transfer_async(op, &peer_key, &descs)
            .map_err(|err| pd_rdma_error("v1 batch_transfer_async failed", err))?;
        for rx in receivers {
            match block_recv(rx) {
                Ok(Ok(_bytes)) => {}
                Ok(Err(err)) => {
                    req_errors.fetch_add(1, Ordering::Release);
                    return Err(pd_rdma_error("v1 transfer failed", err));
                }
                Err(err) => {
                    req_errors.fetch_add(1, Ordering::Release);
                    return Err(pd_rdma_error("v1 transfer completion dropped", err));
                }
            }
        }
        req_stats
            .lock()
            .unwrap()
            .record_complete(Instant::now(), duration_ms(callback_start.elapsed()));
        req_completed.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn collect_write_descs(
        &self,
        py: Python<'_>,
        req_id: &str,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<Vec<V1TransferDesc>> {
        let local = self.local_layer(layer_idx)?;
        let remote = self.remote_layer(req_id, layer_idx)?;
        let mut descs = Vec::new();
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
                let block_id: u64 = py_get(&region, "block_id")?;
                let src_offset: u64 = py_get(&region, "src_offset_bytes")?;
                let bytes: u64 = py_get(&region, "bytes")?;
                let remote_region = remote.regions.get(region_idx).ok_or_else(|| {
                    PyValueError::new_err(format!("remote region {region_idx} is not registered"))
                })?;
                let local_region = local.regions.get(region_idx).ok_or_else(|| {
                    PyValueError::new_err(format!("local region {region_idx} is not registered"))
                })?;
                let local_ptr = local_region
                    .base_addr
                    .checked_add(src_offset)
                    .ok_or_else(|| PyValueError::new_err("local source address overflow"))?;
                self.push_v1_descs_for_blocks(
                    &mut descs,
                    local_ptr,
                    block_id,
                    bytes,
                    remote_region,
                    &remote,
                    true,
                )?;
            }
        }
        Ok(descs)
    }

    fn collect_read_descs(
        &self,
        py: Python<'_>,
        req_id: &str,
        layer_idx: u64,
        blocks: Vec<Py<PyDict>>,
    ) -> PyResult<Vec<V1TransferDesc>> {
        let local = self.local_layer(layer_idx)?;
        let remote = self.remote_layer(req_id, layer_idx)?;
        let mut descs = Vec::new();
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
                let local_block_id: u64 = py_get(&region, "block_id")?;
                let remote_offset: u64 = py_get(&region, "src_offset_bytes")?;
                let bytes: u64 = py_get(&region, "bytes")?;
                let local_region = local.regions.get(region_idx).ok_or_else(|| {
                    PyValueError::new_err(format!("local region {region_idx} is not registered"))
                })?;
                let remote_region = remote.regions.get(region_idx).ok_or_else(|| {
                    PyValueError::new_err(format!("remote region {region_idx} is not registered"))
                })?;
                let local_ptr = local_block_id
                    .checked_mul(local_region.block_stride)
                    .and_then(|offset| local_region.base_addr.checked_add(offset))
                    .ok_or_else(|| PyValueError::new_err("local block address overflow"))?;
                let remote_ptr = remote_region
                    .base_addr
                    .checked_add(remote_offset)
                    .ok_or_else(|| PyValueError::new_err("remote source address overflow"))?;
                descs.push(v1_transfer_desc(local_ptr, remote_ptr, bytes)?);
            }
        }
        Ok(descs)
    }

    fn push_v1_descs_for_blocks(
        &self,
        descs: &mut Vec<V1TransferDesc>,
        local_ptr: u64,
        remote_block_id: u64,
        bytes: u64,
        remote_region: &PdRemoteRegion,
        remote: &V1RemoteLayer,
        validate_blocks: bool,
    ) -> PyResult<()> {
        if bytes == 0 {
            return Err(PyValueError::new_err("block slice bytes must be positive"));
        }
        if !bytes.is_multiple_of(remote_region.block_len) {
            return Err(PyValueError::new_err(
                "block slice bytes must be a multiple of remote block_len",
            ));
        }
        let block_count = bytes / remote_region.block_len;
        for offset in 0..block_count {
            let block_id = remote_block_id
                .checked_add(offset)
                .ok_or_else(|| PyValueError::new_err("remote block id overflow"))?;
            if validate_blocks && !remote.allowed_block_ids.contains(&block_id) {
                return Err(PyRuntimeError::new_err(format!(
                    "remote block {block_id} is not registered"
                )));
            }
            let source = local_ptr
                .checked_add(
                    offset
                        .checked_mul(remote_region.block_len)
                        .ok_or_else(|| PyValueError::new_err("local source address overflow"))?,
                )
                .ok_or_else(|| PyValueError::new_err("local source address overflow"))?;
            let target = block_id
                .checked_mul(remote_region.block_stride)
                .and_then(|byte_offset| remote_region.base_addr.checked_add(byte_offset))
                .ok_or_else(|| PyValueError::new_err("remote block address overflow"))?;
            descs.push(v1_transfer_desc(source, target, remote_region.block_len)?);
        }
        Ok(())
    }

    fn local_layer(&self, layer_idx: u64) -> PyResult<V1LocalLayer> {
        self.local_layers
            .lock()
            .unwrap()
            .get(&layer_idx)
            .cloned()
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("local layer {layer_idx} is not registered"))
            })
    }

    fn remote_layer(&self, req_id: &str, layer_idx: u64) -> PyResult<V1RemoteLayer> {
        self.remote_requests
            .lock()
            .unwrap()
            .get(req_id)
            .and_then(|request| request.layers.get(&layer_idx).cloned())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "remote layer {layer_idx} for req {req_id} is not registered"
                ))
            })
    }

    fn transfer_stats<'py>(
        &self,
        py: Python<'py>,
        pending: &Mutex<HashMap<String, PendingRdmaTransfers>>,
        req_id: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let Some(state) = pending.lock().unwrap().get(req_id).cloned() else {
            let dict = PyDict::new(py);
            dict.set_item("submitted", 0_i64)?;
            dict.set_item("completed", 0_i64)?;
            dict.set_item("errors", 0_i64)?;
            dict.set_item("bytes", 0_u64)?;
            dict.set_item("has_submit", false)?;
            dict.set_item("has_complete", false)?;
            return Ok(dict);
        };
        let dict = state.stats.lock().unwrap().to_py_dict(py)?;
        dict.set_item("submitted", state.submitted)?;
        dict.set_item("completed", state.completed.load(Ordering::Acquire))?;
        dict.set_item("errors", state.errors.load(Ordering::Acquire))?;
        Ok(dict)
    }
}

fn v1_transfer_desc(local_ptr: u64, remote_ptr: u64, len: u64) -> PyResult<V1TransferDesc> {
    Ok(V1TransferDesc {
        local_ptr: nonnull_from_u64(local_ptr, "local_ptr")?.cast::<u8>(),
        remote_ptr: nonnull_from_u64(remote_ptr, "remote_ptr")?.cast::<u8>(),
        len: u64_to_usize(len, "transfer length")?,
    })
}

fn v1_handshake_metadata_from_hex(hex: &str) -> PyResult<V1HandshakeMetadata> {
    let bytes = hex_decode(hex)?;
    let meta = V1HandshakeMetadata::from_bytes(&bytes)
        .map_err(|err| pd_rdma_error("invalid v1 handshake metadata", err))?;
    Ok(meta)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(input: &str) -> PyResult<Vec<u8>> {
    if !input.len().is_multiple_of(2) {
        return Err(PyValueError::new_err("hex string length must be even"));
    }
    let mut bytes = Vec::with_capacity(input.len() / 2);
    let raw = input.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        let hi = hex_value(raw[i])?;
        let lo = hex_value(raw[i + 1])?;
        bytes.push((hi << 4) | lo);
        i += 2;
    }
    Ok(bytes)
}

fn hex_value(byte: u8) -> PyResult<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(PyValueError::new_err("invalid hex digit")),
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

fn mr_desc_from_wire(mr_desc: &pd_wire::MrDesc) -> PyResult<MemoryRegionDescriptor> {
    let mut pairs = SmallVec::new();
    for (addr, rkey) in &mr_desc.addr_rkey_list {
        let addr = addr
            .parse::<DomainAddress>()
            .map_err(|err| pd_rdma_error("invalid DomainAddress", err))?;
        pairs.push((addr, MemoryRegionRemoteKey(*rkey)));
    }
    Ok(MemoryRegionDescriptor {
        ptr: mr_desc.ptr,
        addr_rkey_list: pairs,
    })
}

pub(crate) fn add_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PdRdmaEngine>()?;
    m.add_class::<PdRdmaV1Engine>()?;
    Ok(())
}
