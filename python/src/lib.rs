use std::sync::{Arc, Once};

use pega_core::{LoadState, PegaEngine as CoreEngine};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tracing_subscriber::{
    fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

static INIT_TRACING: Once = Once::new();

fn init_tracing() {
    INIT_TRACING.call_once(|| {
        // Default to info for most crates, debug for core if RUST_LOG not set.
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| "info,pega_core=info".parse())
            .unwrap_or_else(|_| EnvFilter::new("info"));

        let fmt_layer = tracing_subscriber::fmt::layer().with_span_events(FmtSpan::CLOSE);

        // Ignore errors if already initialized by embedding app.
        let _ = tracing_subscriber::registry()
            .with(fmt_layer)
            .with(env_filter)
            .try_init();
    });
}

/// Python wrapper for PegaEngine
#[pyclass]
struct PegaEngine {
    engine: CoreEngine,
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
    ///     device_id: CUDA device ID
    ///     layer_name: Name of the layer
    ///     data_ptr: GPU data pointer (as u64)
    ///     size_bytes: Total size of the tensor in bytes
    ///     num_blocks: Total number of paged blocks for this layer
    ///     bytes_per_block: Size of each paged block in bytes
    ///     kv_stride_bytes: Byte stride between K and V when KV-first layout is used
    ///     segments: Number of segments per block (1 for blocks-first, 2 for KV-first)
    ///     tp_rank: Tensor Parallel rank of the worker
    ///     tp_size: Total Tensor Parallel size
    ///     num_layers: Total number of layers in the model
    #[allow(clippy::too_many_arguments)]
    fn register_context_layer(
        &mut self,
        instance_id: &str,
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

    /// Save KV blocks from GPU via IPC handle to CPU memory
    ///
    /// Args:
    ///     instance_id: ID of the model instance
    ///     tp_rank: Tensor Parallel rank of the worker
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to copy (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn save_kv_blocks_from_ipc(
        &self,
        py: Python<'_>,
        instance_id: &str,
        tp_rank: usize,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        let instance_id_owned = instance_id.to_string();
        let layer_name_owned = layer_name;
        let engine = &self.engine;
        py.allow_threads(move || {
            engine.save_kv_blocks_from_ipc(
                &instance_id_owned,
                tp_rank,
                &layer_name_owned,
                block_ids,
                block_hashes,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Count how many blocks from the prefix are available in CPU storage
    ///
    /// Returns the number of contiguous blocks available from the start.
    /// Stops counting at the first unavailable block by inspecting the
    /// CPU cache completion status directly (no GPU context required).
    ///
    /// Args:
    ///     block_hashes: List of block hashes to check (list of bytes)
    ///
    /// Returns:
    ///     Number of contiguous blocks available from the prefix (int)
    fn count_prefix_hit_blocks(
        &self,
        py: Python<'_>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<usize> {
        let engine = &self.engine;
        py.allow_threads(move || engine.count_prefix_hit_blocks(&block_hashes))
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
        load_state_shm: &str,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        let instance_id_owned = instance_id.to_string();
        let load_state_shm_owned = load_state_shm.to_string();
        let engine = &self.engine;
        py.allow_threads(move || {
            let layer_name_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            engine.batch_load_kv_blocks_multi_layer(
                &instance_id_owned,
                tp_rank,
                &load_state_shm_owned,
                &layer_name_refs,
                &block_ids,
                &block_hashes,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Python wrapper for LoadState (batch-level sync for async KV cache loading)
///
/// Created by connector worker before starting a load batch.
/// Pass shm_name() to the server, then use wait() to spin-wait for completion.
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
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create LoadState: {e:?}")))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Get the shared memory name to pass to the server.
    fn shm_name(&self) -> String {
        self.inner.shm_name().to_string()
    }

    /// Reset state to PENDING (call before starting a new load batch).
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get current state value (non-blocking).
    ///
    /// Returns: 0=pending, 1=success, <0=error
    fn get(&self) -> i64 {
        self.inner.get()
    }

    /// Alias for get() - used by connector for clarity.
    fn get_state(&self) -> i64 {
        self.inner.get()
    }

    /// Check if load is complete (non-blocking).
    ///
    /// Returns True if state is non-zero (completed or error).
    fn is_ready(&self) -> bool {
        self.inner.get() != 0
    }

    /// Spin-wait until state becomes non-zero (completed or error).
    ///
    /// Returns the final state value (1 for success, <0 for error).
    fn wait(&self, py: Python<'_>) -> i64 {
        py.allow_threads(|| self.inner.wait())
    }
}

/// A Python module implemented in Rust.
/// This module is named "pegaflow" and will be imported as: from pegaflow import PegaEngine
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();
    m.add_class::<PegaEngine>()?;
    m.add_class::<PyLoadState>()?;
    Ok(())
}
