use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use log::{error, info};
use pegaflow_transfer::{MemoryRegion, TransferEngine};

use crate::pinned_pool::PinnedAllocator;

/// Thin wrapper around [`TransferEngine`] that manages RDMA memory registration
/// for pinned memory pools.
pub(crate) struct RdmaTransport {
    engine: TransferEngine,
    /// Base pointers of registered regions, kept for unregister on drop.
    registered_ptrs: Vec<NonNull<u8>>,
}

// SAFETY: The registered pointers point to CUDA-pinned memory that is
// fixed in physical memory and safe to access from any thread. The Vec
// is only read during Drop, which is exclusive.
unsafe impl Send for RdmaTransport {}
unsafe impl Sync for RdmaTransport {}

impl RdmaTransport {
    /// Access the underlying transfer engine for active RDMA operations.
    pub(crate) fn engine(&self) -> &TransferEngine {
        &self.engine
    }

    /// Create a new RDMA transport and register all pinned memory regions.
    fn new(nic_names: &[String], allocator: &PinnedAllocator) -> Result<Self, String> {
        let t0 = Instant::now();
        let engine = TransferEngine::new(nic_names).map_err(|e| e.to_string())?;

        let regions: Vec<(NonNull<u8>, usize)> = allocator.memory_regions();
        let mr_descs: Vec<MemoryRegion> = regions
            .iter()
            .map(|&(ptr, len)| MemoryRegion { ptr, len })
            .collect();

        engine
            .register_memory(&mr_descs)
            .map_err(|e| e.to_string())?;

        let registered_ptrs: Vec<NonNull<u8>> = regions.iter().map(|&(ptr, _)| ptr).collect();

        info!(
            "RDMA transport initialised: nics={}, registered {} memory region(s), elapsed={:?}",
            nic_names.len(),
            registered_ptrs.len(),
            t0.elapsed(),
        );

        Ok(Self {
            engine,
            registered_ptrs,
        })
    }
}

impl Drop for RdmaTransport {
    fn drop(&mut self) {
        if let Err(e) = self.engine.unregister_memory(&self.registered_ptrs) {
            error!("Failed to unregister RDMA memory regions: {e}");
        }
    }
}

/// Create an [`RdmaTransport`], returning `None` on failure (logs the error).
pub(crate) fn new_rdma(
    nic_names: &[String],
    allocator: &PinnedAllocator,
) -> Option<Arc<RdmaTransport>> {
    match RdmaTransport::new(nic_names, allocator) {
        Ok(t) => Some(Arc::new(t)),
        Err(e) => {
            error!("Failed to initialise RDMA transport: {e}");
            None
        }
    }
}
