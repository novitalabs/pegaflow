use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use log::{debug, error, info};
use logforth::diagnostic::ThreadLocalDiagnostic;
use tokio::sync::{mpsc, oneshot};

use crate::block::LayerBlock;
use crate::metrics::core_metrics;
use crate::sync_state::LoadState;
use crate::{transfer, EngineError, KVCacheRegistration};

/// A task to load KV blocks from CPU to GPU for multiple layers
pub struct LoadTask {
    pub layers: Vec<LayerLoadData>,
    pub load_state_shm: String,
}

/// Data for loading a single layer
pub struct LayerLoadData {
    pub layer_name: String,
    pub registration: KVCacheRegistration,
    pub blocks: Vec<LoadBlock>,
}

pub struct LoadBlock {
    pub block_idx: usize,
    pub layer_block: Arc<LayerBlock>,
}

/// A task to save KV blocks from GPU to CPU
/// Caller pre-allocates pinned memory, worker does the GPU->CPU copy
pub struct SaveTask {
    pub registration: KVCacheRegistration,
    pub blocks: Vec<SaveBlock>,
    pub reply: oneshot::Sender<Result<(), EngineError>>,
}

pub struct SaveBlock {
    pub block_idx: usize,
    /// Pre-allocated K segment destination pointer (pinned memory)
    pub k_dst_ptr: *mut u8,
    /// Pre-allocated V segment destination pointer (pinned memory), if split layout
    pub v_dst_ptr: Option<*mut u8>,
}

// SAFETY: SaveBlock contains raw pointers to pinned memory that is managed
// by PinnedAllocation. The caller ensures the backing memory stays alive
// until the task completes.
unsafe impl Send for SaveBlock {}

/// Per-GPU worker pool with dedicated load and save threads
pub struct GpuWorkerPool {
    device_id: i32,
    load_tx: mpsc::UnboundedSender<LoadTask>,
    save_tx: mpsc::UnboundedSender<SaveTask>,
}

impl GpuWorkerPool {
    /// Spawn a new worker pool for the given GPU device
    pub fn spawn(device_id: i32) -> Result<Self, EngineError> {
        let (load_tx, load_rx) = mpsc::unbounded_channel();
        let (save_tx, save_rx) = mpsc::unbounded_channel();

        // Spawn load worker thread
        let load_device_id = device_id;
        std::thread::Builder::new()
            .name(format!("gpu{}-load", device_id))
            .spawn(move || {
                if let Err(e) = load_worker_loop(load_device_id, load_rx) {
                    error!(
                        "Load worker failed: device={} error={:?}",
                        load_device_id, e
                    );
                }
            })
            .map_err(|e| EngineError::CudaInit(format!("Failed to spawn load worker: {e}")))?;

        // Spawn save worker thread
        let save_device_id = device_id;
        std::thread::Builder::new()
            .name(format!("gpu{}-save", device_id))
            .spawn(move || {
                if let Err(e) = save_worker_loop(save_device_id, save_rx) {
                    error!(
                        "Save worker failed: device={} error={:?}",
                        save_device_id, e
                    );
                }
            })
            .map_err(|e| EngineError::CudaInit(format!("Failed to spawn save worker: {e}")))?;

        info!("GPU worker pool started: device={}", device_id);
        Ok(Self {
            device_id,
            load_tx,
            save_tx,
        })
    }

    /// Submit a load task (CPU -> GPU) - fire and forget
    pub fn submit_load(&self, task: LoadTask) -> Result<(), EngineError> {
        self.load_tx.send(task).map_err(|_| {
            EngineError::Storage(format!(
                "Load worker channel closed for device {}",
                self.device_id
            ))
        })
    }

    /// Submit a save task (GPU -> CPU) - async, wait for completion
    /// TODO: make sure the source of the blocks is valid until the task is completed
    pub async fn save(
        &self,
        registration: KVCacheRegistration,
        blocks: Vec<SaveBlock>,
    ) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let task = SaveTask {
            registration,
            blocks,
            reply: reply_tx,
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

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

/// Load worker thread main loop
fn load_worker_loop(
    device_id: i32,
    mut rx: mpsc::UnboundedReceiver<LoadTask>,
) -> Result<(), EngineError> {
    // Initialize CUDA context for this thread
    let ctx = CudaContext::new(device_id as usize)
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA context: {e:?}")))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA stream: {e:?}")))?;

    // Set thread-local diagnostic info
    ThreadLocalDiagnostic::insert("device_id", device_id.to_string());

    info!("Load worker initialized: device={}", device_id);

    while let Some(task) = rx.blocking_recv() {
        let result = process_load_task(&task, &stream);

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
    Ok(())
}

/// Save worker thread main loop
fn save_worker_loop(
    device_id: i32,
    mut rx: mpsc::UnboundedReceiver<SaveTask>,
) -> Result<(), EngineError> {
    // Initialize CUDA context for this thread
    let ctx = CudaContext::new(device_id as usize)
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA context: {e:?}")))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA stream: {e:?}")))?;

    // Set thread-local diagnostic info
    ThreadLocalDiagnostic::insert("device_id", device_id.to_string());

    info!("Save worker initialized: device={}", device_id);

    while let Some(task) = rx.blocking_recv() {
        let result = process_save_task(&task, &stream);
        let _ = task.reply.send(result);
    }

    info!("Save worker shutting down: device={}", device_id);
    Ok(())
}

/// Process a load task: copy blocks from CPU pinned memory to GPU for multiple layers
fn process_load_task(task: &LoadTask, stream: &CudaStream) -> Result<(), EngineError> {
    let start = std::time::Instant::now();
    let mut total_bytes = 0usize;
    let mut memcpy_calls = 0usize;
    // Use the first layer's block count as the physical block count (all layers have the same)
    let total_blocks = task.layers.first().map(|l| l.blocks.len()).unwrap_or(0);
    let metrics = core_metrics();

    for layer_data in &task.layers {
        let registration = &layer_data.registration;

        if layer_data.blocks.is_empty() {
            continue;
        }

        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            // Layer-first layout with KV stride: batch K and V separately
            let segment_size = registration.bytes_per_block;

            let mut k_transfers = Vec::with_capacity(layer_data.blocks.len());
            let mut v_transfers = Vec::with_capacity(layer_data.blocks.len());

            for block in &layer_data.blocks {
                let k_gpu_offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                let v_gpu_offset = transfer::segment_offset(registration, block.block_idx, 1)
                    .map_err(EngineError::Storage)?;

                let k_cpu_ptr = block.layer_block.k_ptr();
                let v_cpu_ptr = block
                    .layer_block
                    .v_ptr()
                    .unwrap_or_else(|| unsafe { k_cpu_ptr.add(segment_size) });

                k_transfers.push((k_gpu_offset, k_cpu_ptr));
                v_transfers.push((v_gpu_offset, v_cpu_ptr));
            }

            memcpy_calls += transfer::batch_copy_segments_to_gpu(
                &k_transfers,
                segment_size,
                registration,
                stream,
            )
            .map_err(|e| {
                EngineError::Storage(format!(
                    "K segment transfer failed for layer {}: {e}",
                    layer_data.layer_name
                ))
            })?;

            memcpy_calls += transfer::batch_copy_segments_to_gpu(
                &v_transfers,
                segment_size,
                registration,
                stream,
            )
            .map_err(|e| {
                EngineError::Storage(format!(
                    "V segment transfer failed for layer {}: {e}",
                    layer_data.layer_name
                ))
            })?;

            total_bytes += layer_data.blocks.len() * segment_size * 2;
        } else {
            // Contiguous or single-segment layout - use batch copy for better performance
            let block_size = registration.block_size_bytes;
            let mut transfers = Vec::with_capacity(layer_data.blocks.len());

            for block in &layer_data.blocks {
                let gpu_offset = transfer::segment_offset(registration, block.block_idx, 0)
                    .map_err(EngineError::Storage)?;
                let cpu_ptr = block.layer_block.k_ptr();
                transfers.push((gpu_offset, cpu_ptr));
            }

            memcpy_calls +=
                transfer::batch_copy_segments_to_gpu(&transfers, block_size, registration, stream)
                    .map_err(|e| {
                        EngineError::Storage(format!(
                            "Batch transfer failed for layer {}: {e}",
                            layer_data.layer_name
                        ))
                    })?;

            total_bytes += layer_data.blocks.len() * block_size;
        }
    }

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
        "Load task completed: layers={} blocks={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} memcpy_calls={}",
        task.layers.len(),
        total_blocks,
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        memcpy_calls
    );

    Ok(())
}

/// Process a save task: copy blocks from GPU to CPU pinned memory
fn process_save_task(task: &SaveTask, stream: &CudaStream) -> Result<(), EngineError> {
    let registration = &task.registration;
    let start = std::time::Instant::now();

    let (total_bytes, memcpy_calls) = if registration.segments == 2
        && registration.kv_stride_bytes > registration.bytes_per_block
    {
        // Layer-first layout: K and V segments stored separately
        let segment_size = registration.bytes_per_block;

        // Build transfer lists for batch copy
        let mut k_transfers = Vec::with_capacity(task.blocks.len());
        let mut v_transfers = Vec::with_capacity(task.blocks.len());

        for block in &task.blocks {
            let k_offset = transfer::segment_offset(registration, block.block_idx, 0)
                .map_err(EngineError::Storage)?;
            let v_offset = transfer::segment_offset(registration, block.block_idx, 1)
                .map_err(EngineError::Storage)?;
            let v_ptr = block
                .v_dst_ptr
                .unwrap_or_else(|| unsafe { block.k_dst_ptr.add(segment_size) });

            k_transfers.push((k_offset, block.k_dst_ptr));
            v_transfers.push((v_offset, v_ptr));
        }

        // Batch copy K segments
        let k_batches = transfer::batch_copy_segments_from_gpu(
            &k_transfers,
            segment_size,
            registration,
            stream,
        )
        .map_err(|e| EngineError::Storage(format!("K batch copy failed: {e}")))?;

        // Batch copy V segments
        let v_batches = transfer::batch_copy_segments_from_gpu(
            &v_transfers,
            segment_size,
            registration,
            stream,
        )
        .map_err(|e| EngineError::Storage(format!("V batch copy failed: {e}")))?;

        (task.blocks.len() * segment_size * 2, k_batches + v_batches)
    } else {
        // Contiguous or single-segment layout - build transfer list for batch copy
        let block_size = registration.block_size_bytes;
        let mut transfers = Vec::with_capacity(task.blocks.len());

        for block in &task.blocks {
            let gpu_offset = transfer::segment_offset(registration, block.block_idx, 0)
                .map_err(EngineError::Storage)?;
            transfers.push((gpu_offset, block.k_dst_ptr));
        }

        let memcpy_calls =
            transfer::batch_copy_segments_from_gpu(&transfers, block_size, registration, stream)
                .map_err(|e| EngineError::Storage(format!("Batch copy failed: {e}")))?;

        (task.blocks.len() * block_size, memcpy_calls)
    };

    // Synchronize stream to ensure all copies are complete
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
        "Save task completed: blocks={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} memcpy_calls={}",
        task.blocks.len(),
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        memcpy_calls
    );

    Ok(())
}
