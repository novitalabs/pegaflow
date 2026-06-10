use std::sync::{Arc, mpsc as std_mpsc};

use cudarc::driver::{CudaContext, CudaStream};
use log::{debug, error, info, warn};
use logforth::diagnostic::ThreadLocalDiagnostic;
use tokio::sync::{mpsc, oneshot};

use crate::EngineError;
use crate::block::RawBlock;
use crate::layout::{BlockCopies, KVCacheLayout};
use crate::metrics::core_metrics;
use crate::sync_state::LoadState;
use crate::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend, TransferMode};
use pegaflow_common::{NumaNode, pin_thread_to_numa_node};

/// A task to load KV blocks from CPU to GPU for multiple layers
pub(crate) struct LoadTask {
    pub layers: Vec<LayerTransferData>,
    pub completion: LoadCompletion,
}

/// How a finished [`LoadTask`] hands its result back to the submitter.
///
/// Loads run on the GPU worker thread, so the outcome has to cross back to
/// whoever requested it. Cross-process callers (the vLLM Python worker) watch a
/// shared-memory `LoadState`; in-process Rust callers await a oneshot, the same
/// way [`SaveTask`] already replies. Exactly one of these fires when the task
/// settles.
pub(crate) enum LoadCompletion {
    /// Signal a shared-memory `LoadState` the caller polls (cross-process).
    Shm(String),
    /// Resolve a oneshot the caller awaits or polls (in-process).
    Channel(oneshot::Sender<Result<(), EngineError>>),
}

impl LoadCompletion {
    /// Deliver the task's terminal result. Consumes self so it fires once.
    pub(crate) fn signal(self, result: Result<(), EngineError>) {
        match self {
            LoadCompletion::Shm(shm) => match LoadState::attach(&shm) {
                Ok(load_state) => match result {
                    Ok(()) => load_state.set_completed(),
                    Err(_) => load_state.set_error(),
                },
                Err(e) => error!("Failed to attach to LoadState shm={shm}: {e:?}"),
            },
            LoadCompletion::Channel(reply) => {
                let _ = reply.send(result);
            }
        }
    }
}

/// One layer in a transfer task (either direction): GPU layout plus the
/// blocks to move.
pub(crate) struct LayerTransferData {
    pub layer_name: String,
    pub layout: KVCacheLayout,
    pub blocks: Vec<TransferBlock>,
}

/// One block in a transfer: the GPU slot index and its host-side image.
/// Loads copy `block` -> GPU slot, saves copy GPU slot -> `block`.
pub(crate) struct TransferBlock {
    pub block_idx: usize,
    pub block: Arc<RawBlock>,
}

/// A task to save KV blocks from GPU to CPU for multiple layers.
/// Caller pre-allocates the host blocks, worker does the GPU->CPU copy.
/// All layers are copied on the same CUDA stream with a single synchronization.
pub(crate) struct SaveTask {
    pub layers: Vec<LayerTransferData>,
    pub reply: oneshot::Sender<Result<(), EngineError>>,
    #[cfg(feature = "tracing")]
    pub trace_ctx: Option<::fastrace::prelude::SpanContext>,
}

/// Per-GPU worker pool with dedicated load and save threads
pub(crate) struct GpuWorkerPool {
    device_id: i32,
    load_tx: mpsc::UnboundedSender<LoadTask>,
    save_tx: mpsc::UnboundedSender<SaveTask>,
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
        let (load_tx, load_rx) = mpsc::unbounded_channel();
        let (save_tx, save_rx) = mpsc::unbounded_channel();
        let (load_ready_tx, load_ready_rx) = std_mpsc::channel();
        let (save_ready_tx, save_ready_rx) = std_mpsc::channel();

        // Spawn load worker thread
        let load_device_id = device_id;
        let load_numa = numa_node;
        std::thread::Builder::new()
            .name(format!("gpu{}-load", device_id))
            .spawn(move || {
                // Pin thread to NUMA node before any allocations
                if load_numa.is_valid()
                    && let Err(e) = pin_thread_to_numa_node(load_numa)
                {
                    warn!("Failed to pin load worker to {}: {}", load_numa, e);
                }
                match init_worker(load_device_id, transfer_mode) {
                    Ok(runtime) => {
                        let _ = load_ready_tx.send(Ok(()));
                        load_worker_loop(load_device_id, load_rx, runtime);
                    }
                    Err(e) => {
                        let _ = load_ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| EngineError::CudaInit(format!("Failed to spawn load worker: {e}")))?;

        // Spawn save worker thread
        let save_device_id = device_id;
        let save_numa = numa_node;
        std::thread::Builder::new()
            .name(format!("gpu{}-save", device_id))
            .spawn(move || {
                // Pin thread to NUMA node before any allocations
                if save_numa.is_valid()
                    && let Err(e) = pin_thread_to_numa_node(save_numa)
                {
                    warn!("Failed to pin save worker to {}: {}", save_numa, e);
                }
                match init_worker(save_device_id, transfer_mode) {
                    Ok(runtime) => {
                        let _ = save_ready_tx.send(Ok(()));
                        save_worker_loop(save_device_id, save_rx, runtime);
                    }
                    Err(e) => {
                        let _ = save_ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| EngineError::CudaInit(format!("Failed to spawn save worker: {e}")))?;

        wait_worker_ready("load", device_id, load_ready_rx)?;
        wait_worker_ready("save", device_id, save_ready_rx)?;

        info!(
            "GPU worker pool started: device={}, numa_node={}, transfer_mode={:?}",
            device_id, numa_node, transfer_mode
        );
        Ok(Self {
            device_id,
            load_tx,
            save_tx,
        })
    }

    /// Submit a load task (CPU -> GPU) - fire and forget
    pub(crate) fn submit_load(&self, task: LoadTask) -> Result<(), EngineError> {
        self.load_tx.send(task).map_err(|_| {
            EngineError::Storage(format!(
                "Load worker channel closed for device {}",
                self.device_id
            ))
        })
    }

    /// Submit a multi-layer save task (GPU -> CPU) - async, wait for completion.
    /// All layers are copied on the same CUDA stream with a single synchronization,
    /// which is significantly faster than submitting individual per-layer tasks.
    pub(crate) async fn batch_save(
        &self,
        layers: Vec<LayerTransferData>,
    ) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let task = SaveTask {
            layers,
            reply: reply_tx,
            #[cfg(feature = "tracing")]
            trace_ctx: ::fastrace::prelude::SpanContext::current_local_parent(),
        };

        self.save_tx.send(task).map_err(|_| {
            EngineError::Storage(format!(
                "Save worker channel closed for device {}",
                self.device_id
            ))
        })?;

        // Await the result (this is async, won't block tokio runtime)
        reply_rx.await.map_err(|_| {
            EngineError::Storage(format!(
                "Save worker reply channel closed for device {}",
                self.device_id
            ))
        })?
    }
}

struct WorkerRuntime {
    stream: Arc<CudaStream>,
    backend: Box<dyn TransferBackend>,
}

fn wait_worker_ready(
    worker_name: &str,
    device_id: i32,
    rx: std_mpsc::Receiver<Result<(), EngineError>>,
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
fn load_worker_loop(
    device_id: i32,
    mut rx: mpsc::UnboundedReceiver<LoadTask>,
    runtime: WorkerRuntime,
) {
    while let Some(task) = rx.blocking_recv() {
        let LoadTask { layers, completion } = task;
        let result = process_load_task(&layers, &runtime.stream, runtime.backend.as_ref());

        if let Err(ref e) = result {
            error!("Load task failed: device={device_id} error={e:?}");
            core_metrics().load_failures.add(1, &[]);
        }
        completion.signal(result);
    }

    info!("Load worker shutting down: device={}", device_id);
}

/// Save worker thread main loop
fn save_worker_loop(
    device_id: i32,
    mut rx: mpsc::UnboundedReceiver<SaveTask>,
    runtime: WorkerRuntime,
) {
    while let Some(task) = rx.blocking_recv() {
        let result = process_save_task(&task, &runtime.stream, runtime.backend.as_ref());
        let _ = task.reply.send(result);
    }

    info!("Save worker shutting down: device={}", device_id);
}

/// Build one `CopyDesc` per GPU segment of every block across all layers,
/// pairing device ranges from the layout with the host segments of each
/// block's `RawBlock`. Direction-agnostic: load and save submit the same
/// descriptors to `h2d`/`d2h` respectively.
///
/// Returns `(copies, total_bytes)`.
fn build_copy_descs(layers: &[LayerTransferData]) -> Result<(Vec<CopyDesc>, usize), EngineError> {
    let mut copies: Vec<CopyDesc> = Vec::new();
    let mut total_bytes = 0usize;

    for layer in layers {
        let layer_name = &layer.layer_name;

        for block in &layer.blocks {
            let block_copies = layer
                .layout
                .block_copies(block.block_idx)
                .map_err(|e| EngineError::Storage(format!("layer {layer_name}: {e}")))?;

            match block_copies {
                BlockCopies::Split { k, v } => {
                    let k_ptr = block.block.segment_mapped_ptr(0).unwrap();
                    // SAFETY: For a contiguous host block (segment 1 absent), the
                    // allocation is 2 * segment size, so k + k.bytes is in bounds.
                    let v_ptr = block
                        .block
                        .segment_mapped_ptr(1)
                        .unwrap_or_else(|| k_ptr.add(k.bytes));

                    copies.push(CopyDesc {
                        device: k.addr,
                        host: k_ptr.host().as_ptr(),
                        host_device: k_ptr.device().as_ptr() as u64,
                        size: k.bytes,
                    });
                    copies.push(CopyDesc {
                        device: v.addr,
                        host: v_ptr.host().as_ptr(),
                        host_device: v_ptr.device().as_ptr() as u64,
                        size: v.bytes,
                    });
                    total_bytes += k.bytes + v.bytes;
                }
                BlockCopies::Contiguous(c) => {
                    let ptr = block.block.segment_mapped_ptr(0).unwrap();
                    copies.push(CopyDesc {
                        device: c.addr,
                        host: ptr.host().as_ptr(),
                        host_device: ptr.device().as_ptr() as u64,
                        size: c.bytes,
                    });
                    total_bytes += c.bytes;
                }
            }
        }
    }

    Ok((copies, total_bytes))
}

/// Process a load task: copy blocks from CPU pinned memory to GPU for multiple
/// layers. All layers and segments are collected into one descriptor batch,
/// handed to a single backend (memcpy or kernel), then synchronized once.
fn process_load_task(
    layers: &[LayerTransferData],
    stream: &Arc<CudaStream>,
    backend: &dyn TransferBackend,
) -> Result<(), EngineError> {
    trace_root!("gpu.load_task", _root);
    let start = std::time::Instant::now();
    // Use the first layer's block count as the physical block count (all layers have the same)
    let total_blocks = layers.first().map(|l| l.blocks.len()).unwrap_or(0);
    let metrics = core_metrics();

    let (copies, total_bytes) = build_copy_descs(layers)?;

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
        layers.len(),
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
    let total_blocks: usize = task.layers.iter().map(|l| l.blocks.len()).sum();

    let (copies, total_bytes) = build_copy_descs(&task.layers)?;

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
