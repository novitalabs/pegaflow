use std::{sync::Arc, thread::JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use cudarc::driver::{CudaContext, CudaStream};
use log::{debug, error, info, warn};
use logforth::diagnostic::ThreadLocalDiagnostic;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::block::RawBlock;
use crate::metrics::core_metrics;
use crate::pinned_pool::MappedPinnedPtr;
use crate::sync_state::LoadState;
use crate::transfer::{
    self, CopyDesc, KernelBackend, MemcpyBackend, TransferBackend, TransferMode,
};
use crate::{EngineError, KVCacheRegistration};
use pegaflow_common::{NumaNode, pin_thread_to_numa_node};

/// A task to load KV blocks from CPU to GPU for multiple layers
pub(crate) struct LoadTask {
    pub layers: Vec<LayerLoadData>,
    pub load_state_shm: String,
}

/// Data for loading a single layer
pub(crate) struct LayerLoadData {
    pub layer_name: String,
    pub registration: KVCacheRegistration,
    pub blocks: Vec<LoadBlock>,
}

pub(crate) struct LoadBlock {
    pub block_idx: usize,
    pub block: Arc<RawBlock>,
}

/// A task to save KV blocks from GPU to CPU for multiple layers.
/// Caller pre-allocates pinned memory, worker does the GPU->CPU copy.
/// All layers are copied on the same CUDA stream with a single synchronization.
pub(crate) struct SaveTask {
    pub layers: Vec<SaveLayerData>,
    pub reply: oneshot::Sender<Result<(), EngineError>>,
    #[cfg(feature = "tracing")]
    pub trace_ctx: Option<::fastrace::prelude::SpanContext>,
}

/// Data for saving a single layer's blocks from GPU to CPU.
pub(crate) struct SaveLayerData {
    pub layer_name: String,
    pub registration: KVCacheRegistration,
    pub blocks: Vec<SaveBlock>,
}

pub(crate) struct SaveBlock {
    pub block_idx: usize,
    pub k_dst: MappedPinnedPtr,
    pub v_dst: Option<MappedPinnedPtr>,
}

// SAFETY: SaveBlock contains raw pointers to pinned memory that is managed
// by PinnedAllocation. The caller ensures the backing memory stays alive
// until the task completes.
unsafe impl Send for SaveBlock {}
// SAFETY: SaveLayerData contains SaveBlocks which hold raw pointers to pinned
// memory. Same safety guarantees as SaveBlock apply.
unsafe impl Send for SaveLayerData {}

/// Per-GPU worker pool with dedicated load and save threads
pub(crate) struct GpuWorkerPool {
    device_id: i32,
    load: Worker<LoadTask>,
    save: Worker<SaveTask>,
}

impl GpuWorkerPool {
    /// Spawn a new worker pool for the given GPU device.
    ///
    /// Worker threads will be pinned to the specified NUMA node for optimal
    /// memory locality during D2H/H2D transfers. If `numa_node` is unknown
    /// or pinning fails, the worker will continue without NUMA affinity.
    ///
    /// `transfer_mode` selects the H2D/D2H backend the workers use for their
    /// lifetime.
    pub(crate) fn spawn(
        device_id: i32,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
    ) -> Result<Self, EngineError> {
        let load = spawn_worker(
            "load",
            device_id,
            numa_node,
            transfer_mode,
            load_worker_loop,
        )?;
        let save = spawn_worker(
            "save",
            device_id,
            numa_node,
            transfer_mode,
            save_worker_loop,
        )?;

        info!(
            "GPU worker pool started: device={}, numa_node={}, transfer_mode={:?}",
            device_id, numa_node, transfer_mode
        );
        Ok(Self {
            device_id,
            load,
            save,
        })
    }

    /// Submit a load task (CPU -> GPU) - fire and forget
    pub(crate) fn submit_load(&self, task: LoadTask) -> Result<(), EngineError> {
        self.load.submit(task)
    }

    /// Submit a multi-layer save task (GPU -> CPU) and wait for completion.
    /// All layers are copied on the same CUDA stream with a single synchronization,
    /// which is significantly faster than submitting individual per-layer tasks.
    pub(crate) async fn save_layers(&self, layers: Vec<SaveLayerData>) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let task = SaveTask {
            layers,
            reply: reply_tx,
            #[cfg(feature = "tracing")]
            trace_ctx: ::fastrace::prelude::SpanContext::current_local_parent(),
        };

        self.save.submit(task)?;

        // Await the result (this is async, won't block tokio runtime)
        reply_rx.await.map_err(|_| {
            EngineError::Storage(format!(
                "Save worker reply channel closed for device {}",
                self.device_id
            ))
        })?
    }

    /// Stop accepting new tasks, drain queued work, and join worker threads.
    ///
    /// Callable through a shared reference and idempotent: only the first call
    /// observes the running threads. Joining guarantees no worker is still
    /// touching GPU memory when this returns, so callers may release CUDA
    /// tensors afterwards without risking a use-after-unmap.
    pub(crate) fn shutdown(&self) -> Result<(), EngineError> {
        // Attempt both regardless of the first result so neither thread leaks.
        let load = self.load.shutdown();
        let save = self.save.shutdown();
        let load = load?;
        let save = save?;
        if load || save {
            debug!("GPU worker pool shut down: device={}", self.device_id);
        }
        Ok(())
    }
}

impl Drop for GpuWorkerPool {
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            error!("{err}");
        }
    }
}

struct WorkerRuntime {
    stream: Arc<CudaStream>,
    backend: Box<dyn TransferBackend>,
}

/// A worker thread plus the channel feeding it.
///
/// `tx` and `handle` live behind locks so that [`Worker::shutdown`] can close
/// the channel and join the thread through a shared reference, without needing
/// to own the `Worker`. This is what lets an instance be torn down while other
/// tasks still hold an `Arc` to its `GpuContext`.
struct Worker<T> {
    name: &'static str,
    device_id: i32,
    tx: Mutex<Option<Sender<T>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl<T> Worker<T> {
    fn new(name: &'static str, device_id: i32, tx: Sender<T>, handle: JoinHandle<()>) -> Self {
        Self {
            name,
            device_id,
            tx: Mutex::new(Some(tx)),
            handle: Mutex::new(Some(handle)),
        }
    }

    fn submit(&self, task: T) -> Result<(), EngineError> {
        let guard = self.tx.lock();
        let tx = guard.as_ref().ok_or_else(|| {
            EngineError::Storage(format!(
                "{} worker is shut down for device {}",
                self.name, self.device_id
            ))
        })?;

        tx.send(task).map_err(|_| {
            EngineError::Storage(format!(
                "{} worker channel closed for device {}",
                self.name, self.device_id
            ))
        })
    }

    /// Close the channel and join the worker thread.
    ///
    /// Returns `Ok(true)` if this call joined a live thread, `Ok(false)` if the
    /// worker was already shut down. Dropping the sole `Sender` makes the loop
    /// exit only after its queue drains, so in-flight tasks finish first.
    fn shutdown(&self) -> Result<bool, EngineError> {
        drop(self.tx.lock().take());

        let Some(handle) = self.handle.lock().take() else {
            return Ok(false);
        };
        handle.join().map_err(|_| {
            EngineError::Storage(format!(
                "{} worker panicked on device {}",
                self.name, self.device_id
            ))
        })?;
        Ok(true)
    }
}

impl<T> Drop for Worker<T> {
    fn drop(&mut self) {
        // Safety net for a `Worker` dropped without an explicit shutdown, e.g.
        // when pool construction fails after this worker was already spawned.
        if let Err(err) = self.shutdown() {
            error!("{err}");
        }
    }
}

fn spawn_worker<T, F>(
    name: &'static str,
    device_id: i32,
    numa_node: NumaNode,
    transfer_mode: TransferMode,
    run_loop: F,
) -> Result<Worker<T>, EngineError>
where
    T: Send + 'static,
    F: FnOnce(i32, Receiver<T>, WorkerRuntime) + Send + 'static,
{
    let (tx, rx) = crossbeam_channel::unbounded();
    // One-shot handshake: the worker reports CUDA-init success/failure exactly once.
    let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);

    let handle = std::thread::Builder::new()
        .name(format!("gpu{}-{name}", device_id))
        .spawn(move || {
            if numa_node.is_valid()
                && let Err(e) = pin_thread_to_numa_node(numa_node)
            {
                warn!("Failed to pin {name} worker to {numa_node}: {e}");
            }

            match init_worker(device_id, transfer_mode) {
                Ok(runtime) => {
                    let _ = ready_tx.send(Ok(()));
                    run_loop(device_id, rx, runtime);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .map_err(|e| EngineError::CudaInit(format!("Failed to spawn {name} worker: {e}")))?;

    if let Err(err) = wait_worker_ready(name, device_id, ready_rx) {
        drop(tx);
        if handle.join().is_err() {
            return Err(EngineError::CudaInit(format!(
                "{name} worker panicked during CUDA initialization on device {device_id}"
            )));
        }
        return Err(err);
    }

    Ok(Worker::new(name, device_id, tx, handle))
}

fn wait_worker_ready(
    worker_name: &str,
    device_id: i32,
    rx: Receiver<Result<(), EngineError>>,
) -> Result<(), EngineError> {
    rx.recv().map_err(|_| {
        EngineError::CudaInit(format!(
            "{worker_name} worker exited before CUDA initialization on device {device_id}"
        ))
    })?
}

/// Build the configured transfer backend for a worker. The kernel is only
/// compiled when actually selected.
fn build_backend(
    mode: TransferMode,
    ctx: &std::sync::Arc<CudaContext>,
) -> Result<Box<dyn TransferBackend>, EngineError> {
    match mode {
        TransferMode::Direct => Ok(Box::new(MemcpyBackend)),
        TransferMode::Kernel => {
            let kernel = KernelBackend::new(ctx)
                .map_err(|e| EngineError::CudaInit(format!("kernel backend init failed: {e}")))?;
            Ok(Box::new(kernel))
        }
    }
}

fn init_worker(device_id: i32, transfer_mode: TransferMode) -> Result<WorkerRuntime, EngineError> {
    // Initialize CUDA context for this thread
    let ctx = CudaContext::new(device_id as usize)
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA context: {e:?}")))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA stream: {e:?}")))?;

    // Set thread-local diagnostic info
    ThreadLocalDiagnostic::insert("device_id", device_id.to_string());

    let backend = build_backend(transfer_mode, &ctx)?;

    info!(
        "GPU worker initialized: device={} backend={}",
        device_id,
        backend.name()
    );

    Ok(WorkerRuntime { stream, backend })
}

/// Load worker thread main loop
fn load_worker_loop(device_id: i32, rx: Receiver<LoadTask>, runtime: WorkerRuntime) {
    while let Ok(task) = rx.recv() {
        let result = process_load_task(&task, &runtime.stream, runtime.backend.as_ref());

        // Attach to LoadState and signal completion
        match LoadState::attach(&task.load_state_shm) {
            Ok(load_state) => match result {
                Ok(()) => load_state.set_completed(),
                Err(ref e) => {
                    error!("Load task failed: device={} error={:?}", device_id, e);
                    core_metrics().load_failures.add(1, &[]);
                    load_state.set_error();
                }
            },
            Err(e) => {
                error!(
                    "Failed to attach to LoadState: device={} shm={} error={:?}",
                    device_id, task.load_state_shm, e
                );
                core_metrics().load_failures.add(1, &[]);
            }
        }
    }

    info!("Load worker shutting down: device={}", device_id);
}

/// Save worker thread main loop
fn save_worker_loop(device_id: i32, rx: Receiver<SaveTask>, runtime: WorkerRuntime) {
    while let Ok(task) = rx.recv() {
        let result = process_save_task(&task, &runtime.stream, runtime.backend.as_ref());
        let _ = task.reply.send(result);
    }

    info!("Save worker shutting down: device={}", device_id);
}

fn device_addr(
    registration: &KVCacheRegistration,
    offset: usize,
    size: usize,
    layer_name: &str,
) -> Result<u64, EngineError> {
    let end = offset.checked_add(size).ok_or_else(|| {
        EngineError::Storage(format!(
            "layer {layer_name}: GPU copy range overflow: offset={offset} size={size}"
        ))
    })?;
    if end > registration.size_bytes {
        return Err(EngineError::Storage(format!(
            "layer {layer_name}: GPU copy range exceeds registration: offset={offset} size={size} registration_size={}",
            registration.size_bytes
        )));
    }

    let addr = registration
        .data_ptr
        .checked_add(offset as u64)
        .ok_or_else(|| {
            EngineError::Storage(format!(
                "layer {layer_name}: GPU pointer overflow: base=0x{:x} offset={offset}",
                registration.data_ptr
            ))
        })?;
    addr.checked_add(size as u64).ok_or_else(|| {
        EngineError::Storage(format!(
            "layer {layer_name}: GPU pointer range overflow: addr=0x{addr:x} size={size}"
        ))
    })?;
    Ok(addr)
}

/// Process a load task: copy blocks from CPU pinned memory to GPU for multiple
/// layers. All layers and segments are collected into one descriptor batch,
/// handed to a single backend (memcpy or kernel), then synchronized once.
fn process_load_task(
    task: &LoadTask,
    stream: &Arc<CudaStream>,
    backend: &dyn TransferBackend,
) -> Result<(), EngineError> {
    trace_root!("gpu.load_task", _root);
    let start = std::time::Instant::now();
    let mut total_bytes = 0usize;
    // Use the first layer's block count as the physical block count (all layers have the same)
    let total_blocks = task.layers.first().map(|l| l.blocks.len()).unwrap_or(0);
    let metrics = core_metrics();

    let mut copies: Vec<CopyDesc> = Vec::new();

    for layer_data in &task.layers {
        let registration = &layer_data.registration;
        let layer_name = &layer_data.layer_name;

        if layer_data.blocks.is_empty() {
            continue;
        }

        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            // Layer-first layout with KV stride: K and V live in separate
            // regions. Actual (unpadded) GPU segment size matches GPU layout.
            let segment_size = registration.bytes_per_block;

            for block in &layer_data.blocks {
                let k_offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                let v_offset = transfer::segment_offset(registration, block.block_idx, 1)
                    .map_err(EngineError::Storage)?;

                let k_ptr = block.block.segment_mapped_ptr(0).unwrap();
                // SAFETY: For contiguous layout (segment 1 absent), the allocation
                // is 2 * segment_size bytes, so k_host + segment_size is in bounds.
                let v_ptr = block
                    .block
                    .segment_mapped_ptr(1)
                    .unwrap_or_else(|| k_ptr.add(segment_size));

                copies.push(CopyDesc {
                    device: device_addr(registration, k_offset, segment_size, layer_name)?,
                    host: k_ptr.host().as_ptr(),
                    host_device: k_ptr.device().as_ptr() as u64,
                    size: segment_size,
                });
                copies.push(CopyDesc {
                    device: device_addr(registration, v_offset, segment_size, layer_name)?,
                    host: v_ptr.host().as_ptr(),
                    host_device: v_ptr.device().as_ptr() as u64,
                    size: segment_size,
                });
            }

            total_bytes += layer_data.blocks.len() * segment_size * 2;
        } else {
            // Contiguous or single-segment layout. Actual (unpadded) block size.
            let block_size = registration.block_size_bytes;

            for block in &layer_data.blocks {
                let offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                let ptr = block.block.segment_mapped_ptr(0).unwrap();
                copies.push(CopyDesc {
                    device: device_addr(registration, offset, block_size, layer_name)?,
                    host: ptr.host().as_ptr(),
                    host_device: ptr.device().as_ptr() as u64,
                    size: block_size,
                });
            }

            total_bytes += layer_data.blocks.len() * block_size;
        }
    }

    backend.h2d(&copies, stream).map_err(EngineError::Storage)?;

    // Wait for all transfers to complete
    let event = stream
        .record_event(None)
        .map_err(|e| EngineError::Storage(format!("Failed to record event: {e:?}")))?;
    event
        .synchronize()
        .map_err(|e| EngineError::Storage(format!("Failed to synchronize: {e:?}")))?;

    let elapsed = start.elapsed();
    let bandwidth_gbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes as f64 / 1e9) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    if total_blocks > 0 {
        metrics.load_bytes.add(total_bytes as u64, &[]);
        metrics
            .load_duration_seconds
            .record(elapsed.as_secs_f64(), &[]);
    }

    info!(
        "Load task completed: layers={} blocks={} copies={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} backend={}",
        task.layers.len(),
        total_blocks,
        copies.len(),
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        backend.name()
    );

    Ok(())
}

/// Process a save task: copy blocks from GPU to CPU pinned memory. All layers
/// and segments are collected into one descriptor batch, handed to a single
/// backend, then synchronized once.
fn process_save_task(
    task: &SaveTask,
    stream: &Arc<CudaStream>,
    backend: &dyn TransferBackend,
) -> Result<(), EngineError> {
    trace_child!("gpu.save_task", task.trace_ctx);
    let start = std::time::Instant::now();
    let mut total_bytes = 0usize;
    let mut total_blocks = 0usize;

    let mut copies: Vec<CopyDesc> = Vec::new();

    for layer in &task.layers {
        let registration = &layer.registration;
        let layer_name = &layer.layer_name;

        if layer.blocks.is_empty() {
            continue;
        }
        total_blocks += layer.blocks.len();

        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            // Layer-first layout: K and V segments stored separately. Actual
            // (unpadded) GPU segment size — pinned memory may use a larger
            // padded stride, but CUDA copies only the real data.
            let segment_size = registration.bytes_per_block;

            for block in &layer.blocks {
                let k_offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                let v_offset = transfer::segment_offset(registration, block.block_idx, 1)
                    .map_err(EngineError::Storage)?;
                let v_dst = block.v_dst.unwrap_or_else(|| block.k_dst.add(segment_size));

                copies.push(CopyDesc {
                    device: device_addr(registration, k_offset, segment_size, layer_name)?,
                    host: block.k_dst.host().as_ptr(),
                    host_device: block.k_dst.device().as_ptr() as u64,
                    size: segment_size,
                });
                copies.push(CopyDesc {
                    device: device_addr(registration, v_offset, segment_size, layer_name)?,
                    host: v_dst.host().as_ptr(),
                    host_device: v_dst.device().as_ptr() as u64,
                    size: segment_size,
                });
            }

            total_bytes += layer.blocks.len() * segment_size * 2;
        } else {
            // Contiguous or single-segment layout. Actual (unpadded) block size.
            let block_size = registration.block_size_bytes;

            for block in &layer.blocks {
                let offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                copies.push(CopyDesc {
                    device: device_addr(registration, offset, block_size, layer_name)?,
                    host: block.k_dst.host().as_ptr(),
                    host_device: block.k_dst.device().as_ptr() as u64,
                    size: block_size,
                });
            }

            total_bytes += layer.blocks.len() * block_size;
        }
    }

    backend.d2h(&copies, stream).map_err(EngineError::Storage)?;

    // Single synchronization for all layers
    let event = stream
        .record_event(None)
        .map_err(|e| EngineError::Storage(format!("Failed to record event: {e:?}")))?;
    event
        .synchronize()
        .map_err(|e| EngineError::Storage(format!("Failed to synchronize: {e:?}")))?;

    let elapsed = start.elapsed();
    let bandwidth_gbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes as f64 / 1e9) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    debug!(
        "Save task completed: layers={} blocks={} copies={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} backend={}",
        task.layers.len(),
        total_blocks,
        copies.len(),
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        backend.name()
    );

    Ok(())
}
