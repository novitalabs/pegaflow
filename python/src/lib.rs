use std::sync::Once;

use pega_core::PegaEngine as CoreEngine;
use pyo3::prelude::*;
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

    /// Register a KV cache buffer along with its layout metadata.
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     data_ptr: GPU data pointer (as u64)
    ///     size_bytes: Total size of the tensor in bytes
    ///     num_blocks: Total number of paged blocks for this layer
    ///     bytes_per_block: Size of each paged block in bytes
    ///     kv_stride_bytes: Byte stride between K and V when KV-first layout is used
    ///     segments: Number of segments per block (1 for blocks-first, 2 for KV-first)
    fn register_kv_cache(
        &self,
        py: Python<'_>,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) {
        py.allow_threads(|| {
            self.engine.register_kv_cache(
                layer_name,
                data_ptr,
                size_bytes,
                num_blocks,
                bytes_per_block,
                kv_stride_bytes,
                segments,
            );
        })
    }

    /// Unregister all KV cache handles
    fn unregister_all_kv_caches(&self, py: Python<'_>) {
        py.allow_threads(|| {
            self.engine.unregister_all_kv_caches();
        })
    }

    /// Save KV blocks from GPU via IPC handle to CPU memory
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to copy (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn save_kv_blocks_from_ipc(
        &self,
        py: Python<'_>,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) {
        py.allow_threads(|| {
            self.engine
                .save_kv_blocks_from_ipc(layer_name, block_ids, block_hashes)
        })
    }

    /// Count how many blocks from the prefix are available in CPU storage
    ///
    /// Returns the number of contiguous blocks available from the start.
    /// Stops counting at the first unavailable block.
    /// Uses the first registered layer (layer_id = 0) for availability check.
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
    ) -> usize {
        py.allow_threads(|| self.engine.count_prefix_hit_blocks(&block_hashes))
    }

    /// Wait until the async transfer for `layer_name` completes.
    fn wait_for_layer_transfer(&self, py: Python<'_>, layer_name: String) -> PyResult<()> {
        py.allow_threads(|| self.engine.wait_for_layer_transfer(&layer_name))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
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
    ///     layer_names: List of layer names to load
    ///     block_ids: GPU block IDs to load into (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn batch_load_kv_blocks(
        &self,
        py: Python<'_>,
        layer_names: Vec<String>,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<(usize, usize)> {
        py.allow_threads(|| {
            // Convert Vec<String> to Vec<&str> for the engine call
            let layer_name_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            match self.engine.batch_load_kv_blocks_multi_layer(
                &layer_name_refs,
                &block_ids,
                &block_hashes,
            ) {
                Ok(results) => {
                    let total_layers = results.len();
                    let total_bytes = results.iter().map(|(_, bytes)| bytes).sum();
                    Ok((total_layers, total_bytes))
                }
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to batch load KV blocks: {}",
                    e
                ))),
            }
        })
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
