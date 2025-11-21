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
        &mut self,
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
    fn unregister_all_kv_caches(&mut self, py: Python<'_>) {
        py.allow_threads(|| {
            self.engine.unregister_all_kv_caches();
        })
    }

    /// Get the number of registered KV caches
    fn num_registered_kv_caches(&self) -> usize {
        self.engine.num_registered_kv_caches()
    }

    /// Save KV blocks from GPU via IPC handle to CPU memory
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to copy (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn save_kv_blocks_from_ipc(
        &mut self,
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

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    fn get_storage_stats(&self) -> (usize, usize) {
        self.engine.get_storage_stats()
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_hashes: List of block hashes to check (list of bytes)
    ///
    /// Returns:
    ///     List of booleans indicating availability for each block
    fn check_kv_blocks_availability(
        &self,
        py: Python<'_>,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        py.allow_threads(|| {
            self.engine
                .check_kv_blocks_availability(layer_name, block_hashes)
        })
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to load into (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn load_kv_blocks_to_ipc(
        &self,
        py: Python<'_>,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<usize> {
        py.allow_threads(|| {
            self.engine
                .load_kv_blocks_to_ipc(layer_name, block_ids, block_hashes)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Get pinned memory usage statistics
    /// Returns (used_bytes, total_bytes)
    fn get_pinned_memory_usage(&self) -> (usize, usize) {
        self.engine.get_pinned_memory_usage()
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
