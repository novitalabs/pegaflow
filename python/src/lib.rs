use std::sync::Once;

use pega_core::PegaEngine as CoreEngine;
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

    /// Wait until the async transfer for `layer_name` completes.
    fn wait_for_layer_transfer(
        &self,
        py: Python<'_>,
        instance_id: &str,
        tp_rank: usize,
        layer_name: String,
    ) -> PyResult<()> {
        let instance_id_owned = instance_id.to_string();
        let engine = &self.engine;
        py.allow_threads(move || {
            engine.wait_for_layer_transfer(&instance_id_owned, tp_rank, &layer_name)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Batch load KV blocks for multiple layers using the same block mapping
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
    ///     layer_names: List of layer names to load
    ///     block_ids: GPU block IDs to load into (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn batch_load_kv_blocks(
        &self,
        py: Python<'_>,
        instance_id: &str,
        tp_rank: usize,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<(usize, usize)> {
        let instance_id_owned = instance_id.to_string();
        let engine = &self.engine;
        py.allow_threads(move || {
            let layer_name_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            engine
                .batch_load_kv_blocks_multi_layer(
                    &instance_id_owned,
                    tp_rank,
                    &layer_name_refs,
                    &block_ids,
                    &block_hashes,
                )
                .map(|results| {
                    let total_layers = results.len();
                    let total_bytes = results.iter().map(|(_, bytes)| bytes).sum();
                    (total_layers, total_bytes)
                })
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// A Python module implemented in Rust.
/// This module is named "pegaflow" and will be imported as: from pegaflow import PegaEngine
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();
    m.add_class::<PegaEngine>()?;
    Ok(())
}
