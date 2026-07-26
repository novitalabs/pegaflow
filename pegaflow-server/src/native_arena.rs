//! Server-owned KV arenas for native (non-Python) clients.
//!
//! A native client does not export memory to us — we allocate the arena in
//! this process with `cuMemAlloc` and hand back a CUDA IPC handle in the
//! registration response. Owning the allocation is what makes the follow-up
//! RDMA work possible: `ibv_reg_mr`/dma-buf registration works on memory this
//! process allocated, while an IPC-*imported* pointer can never be registered
//! into a NIC. The client side only runs compute kernels on its imported
//! mapping, which CUDA IPC fully supports.
//!
//! One arena per context; layer views are plain `base + offset` arithmetic.
//! The arena is freed when its context leaves the registry, strictly after the
//! engine forgot the raw pointers derived from it.

use cudarc::driver::{CudaContext, result::DriverError, sys};
use std::collections::HashMap;
use std::sync::Arc;

use crate::registry::TensorMetadata;

/// One layer's requested view into the arena, straight from the wire.
pub(crate) struct NativeLayerView {
    pub layer_name: String,
    pub offset_bytes: u64,
    pub size_bytes: u64,
}

/// Result of a native registration: raw pointers for the engine plus the IPC
/// handle the client imports.
pub(crate) struct NativeRegistration {
    pub metadatas: Vec<TensorMetadata>,
    pub arena_ipc_handle: Vec<u8>,
}

/// A device allocation owned by this process, shared to one client via CUDA
/// IPC. Freed on drop, on the registry actor thread.
struct NativeArena {
    context: Arc<CudaContext>,
    base_ptr: sys::CUdeviceptr,
    size_bytes: usize,
    ipc_handle: sys::CUipcMemHandle,
}

// SAFETY: the raw device pointer and IPC handle are plain values; all CUDA
// calls bind the context to the calling thread first.
unsafe impl Send for NativeArena {}

impl NativeArena {
    fn allocate(device_id: i32, size_bytes: usize) -> Result<Self, String> {
        let device = usize::try_from(device_id)
            .map_err(|_| format!("device_id {device_id} must be >= 0"))?;
        let context = CudaContext::new(device).map_err(|e| cuda_error("retain CUDA context", e))?;
        context
            .bind_to_thread()
            .map_err(|e| cuda_error("bind CUDA context", e))?;

        let mut base_ptr: sys::CUdeviceptr = 0;
        // SAFETY: base_ptr receives the allocation; size is non-zero
        // (validated by the caller).
        unsafe { sys::cuMemAlloc_v2(&mut base_ptr, size_bytes).result() }
            .map_err(|e| cuda_error("allocate KV arena", e))?;

        // Zero the arena so a KV hit never reads stale device memory. The
        // memset is async with respect to the host and CUDA IPC gives the
        // importing process no cross-process stream ordering, so synchronize
        // before the handle leaves this function.
        // SAFETY: base_ptr..base_ptr+size_bytes was just allocated.
        let zeroed = unsafe { sys::cuMemsetD8_v2(base_ptr, 0, size_bytes).result() }
            .and_then(|_| unsafe { sys::cuCtxSynchronize().result() });
        if let Err(e) = zeroed {
            unsafe { sys::cuMemFree_v2(base_ptr).result().ok() };
            return Err(cuda_error("zero KV arena", e));
        }

        let mut ipc_handle = sys::CUipcMemHandle { reserved: [0; 64] };
        // SAFETY: base_ptr is a live cuMemAlloc allocation, the only kind
        // cuIpcGetMemHandle accepts.
        if let Err(e) = unsafe { sys::cuIpcGetMemHandle(&mut ipc_handle, base_ptr).result() } {
            unsafe { sys::cuMemFree_v2(base_ptr).result().ok() };
            return Err(cuda_error("export KV arena IPC handle", e));
        }

        Ok(Self {
            context,
            base_ptr,
            size_bytes,
            ipc_handle,
        })
    }
}

impl Drop for NativeArena {
    fn drop(&mut self) {
        self.context
            .bind_to_thread()
            .expect("bind CUDA context before freeing native KV arena");
        // SAFETY: base_ptr is the live allocation from `allocate`; the engine
        // dropped its raw pointers before the registry released this arena.
        unsafe { sys::cuMemFree_v2(self.base_ptr).result() }.expect("free native KV arena");
    }
}

fn cuda_error(operation: &str, error: DriverError) -> String {
    format!("{operation}: {error}")
}

struct NativeContext {
    #[allow(
        dead_code,
        reason = "owning the arena keeps the registered addresses alive; freed on drop"
    )]
    arena: NativeArena,
    layer_count: usize,
}

/// Native contexts keyed by the same `instance:tp:pp:dev` context key as the
/// Python registry, kept in a separate map so the Python bookkeeping stays
/// untouched.
#[derive(Default)]
pub(crate) struct NativeArenaMap {
    contexts: HashMap<String, NativeContext>,
}

impl NativeArenaMap {
    pub(crate) fn contains(&self, context_key: &str) -> bool {
        self.contexts.contains_key(context_key)
    }

    /// Allocate an arena on `device_id`, carve the requested layer views out
    /// of it, and record the context. Runs on the registry actor thread.
    pub(crate) fn register(
        &mut self,
        context_key: &str,
        device_id: i32,
        layers: &[NativeLayerView],
        alloc_size: usize,
    ) -> Result<NativeRegistration, String> {
        if self.contexts.contains_key(context_key) {
            return Err(format!("context {context_key} is already registered"));
        }
        let mut seen = std::collections::HashSet::with_capacity(layers.len());
        for layer in layers {
            if !seen.insert(layer.layer_name.as_str()) {
                return Err(format!(
                    "layer {} appears more than once in context {context_key}",
                    layer.layer_name
                ));
            }
        }

        let arena = NativeArena::allocate(device_id, alloc_size)?;
        let mut metadatas = Vec::with_capacity(layers.len());
        for layer in layers {
            let size = usize::try_from(layer.size_bytes)
                .map_err(|_| format!("layer {} size does not fit usize", layer.layer_name))?;
            layer
                .offset_bytes
                .checked_add(layer.size_bytes)
                .filter(|end| *end <= arena.size_bytes as u64)
                .ok_or_else(|| {
                    format!(
                        "layer {} view [{} +{}] is outside its {}-byte arena",
                        layer.layer_name, layer.offset_bytes, layer.size_bytes, arena.size_bytes
                    )
                })?;
            metadatas.push(TensorMetadata {
                data_ptr: arena.base_ptr + layer.offset_bytes,
                size_bytes: size,
                device_id,
            });
        }

        let arena_ipc_handle: Vec<u8> = arena
            .ipc_handle
            .reserved
            .iter()
            .map(|&byte| byte as u8)
            .collect();
        self.contexts.insert(
            context_key.to_string(),
            NativeContext {
                arena,
                layer_count: layers.len(),
            },
        );
        Ok(NativeRegistration {
            metadatas,
            arena_ipc_handle,
        })
    }

    /// Free the arena of one context; returns the number of layers dropped.
    pub(crate) fn drop_context(&mut self, context_key: &str) -> usize {
        self.contexts
            .remove(context_key)
            .map(|ctx| ctx.layer_count)
            .unwrap_or(0)
    }

    /// Free every arena belonging to `instance_id`; returns layers dropped.
    pub(crate) fn drop_instance(&mut self, instance_id: &str) -> usize {
        let prefix = format!("{instance_id}:");
        let keys: Vec<String> = self
            .contexts
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();
        keys.iter().map(|key| self.drop_context(key)).sum()
    }

    /// Free every arena; returns layers dropped.
    pub(crate) fn clear(&mut self) -> usize {
        let count = self.contexts.values().map(|ctx| ctx.layer_count).sum();
        self.contexts.clear();
        count
    }
}
