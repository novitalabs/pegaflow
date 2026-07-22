use cudarc::driver::{CudaContext, result::DriverError, sys};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub device_id: i32,
}

pub(crate) enum TensorRegistration {
    Python(Vec<u8>),
    /// One strided layer view into the batch's shared VMM allocation. The
    /// backing fd is not here — it arrives out-of-band on the fd side-channel
    /// and is imported once per batch (see [`CudaTensorRegistry::register_layers`]).
    Native { offset_bytes: u64, size_bytes: u64 },
}

#[allow(dead_code, reason = "owners keep registered CUDA addresses valid")]
enum TensorOwner {
    Python(Py<PyAny>),
    Vmm(Arc<VmmMapping>),
}

struct LayerTensor {
    owner: TensorOwner,
    metadata: TensorMetadata,
}

/// A VMM allocation imported from a client's exported POSIX file descriptor and
/// mapped into this process's address space. Owns the reserved VA range and the
/// imported physical handle; `base_ptr` is the mapped device pointer other
/// layers offset into. Unlike a CUDA IPC mapping, this pointer can be handed to
/// `ibv_reg_dmabuf_mr` for GPUDirect RDMA (registration wiring is a follow-up).
struct VmmMapping {
    context: Arc<CudaContext>,
    handle: sys::CUmemGenericAllocationHandle,
    base_ptr: sys::CUdeviceptr,
    size_bytes: usize,
}

impl VmmMapping {
    /// Import `fd` (a `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` exported by the
    /// client), reserve a VA range of `alloc_size`, map the physical memory in,
    /// and grant this device read/write access. `alloc_size` must match the
    /// client's allocation (already rounded up to the VMM granularity).
    fn import(device_id: i32, fd: i32, alloc_size: usize) -> PyResult<Arc<Self>> {
        let device = usize::try_from(device_id)
            .map_err(|_| PyValueError::new_err("device_id must be non-negative"))?;
        let context = CudaContext::new(device).map_err(|e| cuda_error("retain context", e))?;
        context
            .bind_to_thread()
            .map_err(|e| cuda_error("bind context", e))?;

        // Import the physical allocation from the client's fd. CUDA dups the fd
        // internally, so the caller still owns (and closes) its own copy.
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: `handle` is valid output storage; `fd` is a live POSIX fd for a
        // VMM allocation the client exported; the type tag matches the export.
        unsafe {
            sys::cuMemImportFromShareableHandle(
                &mut handle,
                fd as usize as *mut std::ffi::c_void,
                sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
            .result()
            .map_err(|e| cuda_error("import VMM shareable handle", e))?;
        }

        // Reserve a VA range and map the physical handle into it.
        let mut base_ptr: sys::CUdeviceptr = 0;
        // SAFETY: standard VMM reserve; outputs are valid, alignment 0 = default.
        if let Err(e) = unsafe {
            sys::cuMemAddressReserve(&mut base_ptr, alloc_size, 0, 0, 0).result()
        } {
            // SAFETY: handle was successfully imported above and not yet mapped.
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("reserve VMM address range", e));
        }
        // SAFETY: base_ptr..+alloc_size is freshly reserved; handle is imported.
        if let Err(e) = unsafe { sys::cuMemMap(base_ptr, alloc_size, 0, handle, 0).result() } {
            unsafe { sys::cuMemAddressFree(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("map VMM allocation", e));
        }

        // Grant this device read/write access to the mapped range.
        let access = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: base_ptr..+alloc_size is mapped; one access descriptor.
        if let Err(e) = unsafe { sys::cuMemSetAccess(base_ptr, alloc_size, &access, 1).result() } {
            unsafe { sys::cuMemUnmap(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemAddressFree(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("set VMM access", e));
        }

        Ok(Arc::new(Self {
            context,
            handle,
            base_ptr,
            size_bytes: alloc_size,
        }))
    }
}

impl Drop for VmmMapping {
    fn drop(&mut self) {
        self.context
            .bind_to_thread()
            .expect("bind CUDA context before releasing VMM mapping");
        // SAFETY: this object owns one successful import+reserve+map, torn down
        // in reverse order: unmap, free VA, release physical handle.
        unsafe {
            sys::cuMemUnmap(self.base_ptr, self.size_bytes)
                .result()
                .expect("unmap VMM allocation");
            sys::cuMemAddressFree(self.base_ptr, self.size_bytes)
                .result()
                .expect("free VMM address range");
            sys::cuMemRelease(self.handle)
                .result()
                .expect("release VMM handle");
        }
    }
}

fn cuda_error(operation: &str, error: DriverError) -> PyErr {
    PyRuntimeError::new_err(format!("{operation}: {error}"))
}

struct ContextState {
    device_id: i32,
    tensors: HashMap<String, LayerTensor>,
}

impl ContextState {
    fn new(device_id: i32) -> Self {
        Self {
            device_id,
            tensors: HashMap::new(),
        }
    }
}

pub struct CudaTensorRegistry {
    contexts: HashMap<String, ContextState>,
}

impl CudaTensorRegistry {
    pub fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let torch = py.import("torch")?;
            let cuda = torch.getattr("cuda")?;
            cuda.call_method0("init")?;
            Ok(Self {
                contexts: HashMap::new(),
            })
        })
    }

    pub fn empty() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Register a batch of layers under `context_key`.
    ///
    /// `native_fd` is `Some((fd, alloc_size))` for native VMM clients: the whole
    /// batch shares one fused allocation imported once from that fd. It is `None`
    /// for Python clients, whose per-layer tensors carry their own storage.
    fn register_layers(
        &mut self,
        context_key: &str,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        native_fd: Option<(i32, usize)>,
    ) -> PyResult<Vec<TensorMetadata>> {
        if self.contexts.contains_key(context_key) {
            return Err(PyValueError::new_err(format!(
                "context {context_key} is already registered"
            )));
        }

        let mut seen_layers = HashSet::with_capacity(layers.len());
        for (layer_name, _) in &layers {
            if !seen_layers.insert(layer_name.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "layer {layer_name} appears more than once in context {context_key}"
                )));
            }
        }

        // Native batches import their one shared allocation up front; every layer
        // is a strided view offsetting into it.
        let mapping = match native_fd {
            Some((fd, alloc_size)) => Some(VmmMapping::import(device_id, fd, alloc_size)?),
            None => None,
        };

        let mut context = ContextState::new(device_id);
        let mut metadatas = Vec::with_capacity(layers.len());
        for (layer_name, registration) in layers {
            let layer_tensor = match registration {
                TensorRegistration::Python(bytes) => {
                    Self::materialize_python_tensor(device_id, &bytes)?
                }
                TensorRegistration::Native {
                    offset_bytes,
                    size_bytes,
                } => {
                    let mapping = mapping.as_ref().ok_or_else(|| {
                        PyValueError::new_err(
                            "native tensor registration without an imported fd",
                        )
                    })?;
                    Self::materialize_native_tensor(
                        device_id,
                        mapping,
                        offset_bytes,
                        size_bytes,
                    )?
                }
            };
            let metadata = layer_tensor.metadata.clone();

            if context.device_id != metadata.device_id {
                return Err(PyValueError::new_err(format!(
                    "context {context_key} is pinned to device {} but got {}",
                    context.device_id, metadata.device_id
                )));
            }

            context.tensors.insert(layer_name, layer_tensor);
            metadatas.push(metadata);
        }

        self.contexts.insert(context_key.to_string(), context);
        Ok(metadatas)
    }

    fn drop_context(&mut self, context_key: &str) -> usize {
        self.release_contexts(vec![context_key.to_string()])
    }

    fn drop_instance(&mut self, instance_id: &str) -> usize {
        let prefix = format!("{instance_id}:");
        let keys: Vec<String> = self
            .contexts
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();
        self.release_contexts(keys)
    }

    /// Clear all contexts and return the total number of tensors removed.
    fn clear_and_count(&mut self) -> usize {
        let keys: Vec<String> = self.contexts.keys().cloned().collect();
        self.release_contexts(keys)
    }

    /// Remove `keys` from the registry, returning the number of CUDA IPC tensors
    /// released.
    ///
    /// This is the single place that decides whether GIL + CUDA teardown is
    /// needed, so the decision can't drift between callers: it acquires the GIL
    /// and runs `gc.collect()` + `torch.cuda.empty_cache()` ONLY when the
    /// removed contexts actually hold live tensors. Dropping empty contexts is
    /// pure Rust, so an idle/empty registry never forces a CUDA device sync —
    /// which would block forever on a wedged GPU.
    fn release_contexts(&mut self, keys: Vec<String>) -> usize {
        let tensor_count: usize = keys
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .map(|ctx| ctx.tensors.len())
            .sum();

        if tensor_count == 0 {
            for key in &keys {
                self.contexts.remove(key);
            }
            return 0;
        }

        let needs_python = keys
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .flat_map(|context| context.tensors.values())
            .any(|tensor| matches!(&tensor.owner, TensorOwner::Python(_)));
        let removed: Vec<_> = keys
            .iter()
            .filter_map(|key| self.contexts.remove(key))
            .collect();

        if needs_python {
            Python::attach(|py| {
                drop(removed);
                let gc = py.import("gc").expect("gc module");
                let _ = gc.call_method0("collect");

                let torch = py.import("torch").expect("torch module");
                let cuda = torch.getattr("cuda").expect("torch.cuda");
                let _ = cuda.call_method0("empty_cache");
            });
        }

        tensor_count
    }

    fn materialize_python_tensor(device_id: i32, wrapper_bytes: &[u8]) -> PyResult<LayerTensor> {
        Python::attach(|py| {
            let torch = py.import("torch")?;
            let pickle = py.import("pickle")?;
            let cuda = torch.getattr("cuda")?;

            cuda.call_method1("set_device", (device_id,))?;

            let py_bytes = PyBytes::new(py, wrapper_bytes);
            let wrapper = pickle.call_method1("loads", (py_bytes,))?;
            let tensor = wrapper.call_method0("to_tensor")?;

            let data_ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;
            let device_attr = tensor.getattr("device")?;
            let device_index: Option<i32> = device_attr.getattr("index")?.extract()?;
            let resolved_device = device_index.unwrap_or(device_id);

            let storage = tensor.call_method0("untyped_storage")?;
            let size_bytes: usize = storage.call_method0("nbytes")?.extract()?;

            let tensor_owned = tensor.unbind();

            Ok(LayerTensor {
                owner: TensorOwner::Python(tensor_owned),
                metadata: TensorMetadata {
                    data_ptr,
                    size_bytes,
                    device_id: resolved_device,
                },
            })
        })
    }

    /// Build one layer view as a strided window into the batch's shared VMM
    /// mapping. `offset_bytes`/`size_bytes` are validated against the mapping's
    /// imported size so a malformed client can't point us outside the allocation.
    fn materialize_native_tensor(
        device_id: i32,
        mapping: &Arc<VmmMapping>,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> PyResult<LayerTensor> {
        let offset = usize::try_from(offset_bytes)
            .map_err(|_| PyValueError::new_err("native view offset does not fit usize"))?;
        let size = usize::try_from(size_bytes)
            .map_err(|_| PyValueError::new_err("native view size does not fit usize"))?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| PyValueError::new_err("native view range overflows usize"))?;
        if size == 0 || end > mapping.size_bytes {
            return Err(PyValueError::new_err(
                "native view is outside its allocation",
            ));
        }
        let data_ptr = mapping
            .base_ptr
            .checked_add(offset_bytes)
            .ok_or_else(|| PyValueError::new_err("native view pointer overflows u64"))?;
        Ok(LayerTensor {
            owner: TensorOwner::Vmm(Arc::clone(mapping)),
            metadata: TensorMetadata {
                data_ptr,
                size_bytes: size,
                device_id,
            },
        })
    }
}

/// Work submitted to the dedicated registry thread. Each carries a `oneshot`
/// the actor uses to hand the result back to the awaiting caller.
enum RegistryCommand {
    RegisterLayers {
        context_key: String,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        /// `Some((fd, alloc_size))` for native VMM batches; the actor imports the
        /// shared allocation from `fd` and closes its own fd copy afterward.
        native_fd: Option<(std::os::fd::OwnedFd, usize)>,
        // The `PyErr` is stringified on the actor thread (which holds the GIL),
        // so callers never need to touch the GIL to read an error message.
        reply: oneshot::Sender<Result<Vec<TensorMetadata>, String>>,
    },
    DropInstance {
        instance_id: String,
        reply: oneshot::Sender<usize>,
    },
    DropContext {
        context_key: String,
        reply: oneshot::Sender<usize>,
    },
    Clear {
        reply: oneshot::Sender<usize>,
    },
}

/// Async handle to a [`CudaTensorRegistry`] that lives on its own OS thread.
///
/// Every mutating op takes the GIL and may run a blocking
/// `torch.cuda.empty_cache()` that performs a CUDA device sync — which never
/// returns if the GPU is wedged. Confining the registry to one dedicated thread
/// keeps that blocking, GIL-bearing work off the async runtime *by
/// construction*: handlers only ever `.await` a reply, so a wedged CUDA call
/// pins this single thread instead of starving tokio workers (the outage where
/// a few `cleanup` calls hung every endpoint, `/health` and `/metrics`
/// included). Serializing on one thread also matches the GIL's own
/// serialization — registry ops were never able to run concurrently anyway.
#[derive(Clone)]
pub struct RegistryHandle {
    tx: mpsc::Sender<RegistryCommand>,
}

impl RegistryHandle {
    /// Move `registry` onto a dedicated `cuda-registry` thread and return an
    /// async handle to it. The thread runs until every handle is dropped.
    pub fn spawn(registry: CudaTensorRegistry) -> Self {
        // Bounds how many register/cleanup requests queue before callers await
        // for space; the actor drains them one at a time under the GIL.
        let (tx, rx) = mpsc::channel(64);
        std::thread::Builder::new()
            .name("cuda-registry".to_string())
            .spawn(move || registry_actor(registry, rx))
            .expect("spawn cuda-registry thread");
        Self { tx }
    }

    /// Materialize and register a batch of layers under `context_key`. Returns
    /// per-layer metadata in input order. The batch is transactional with
    /// respect to the registry: an existing context is rejected before any
    /// tensor is materialized, and a materialization failure does not publish a
    /// partial context. `native_fd` is `Some` for native VMM clients (the fused
    /// allocation's fd, already received over the fd side-channel) and `None`
    /// for Python clients.
    pub(crate) async fn register_layers(
        &self,
        context_key: String,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        native_fd: Option<(std::os::fd::OwnedFd, usize)>,
    ) -> Result<Vec<TensorMetadata>, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::RegisterLayers {
            context_key,
            device_id,
            layers,
            native_fd,
            reply,
        })
        .await;
        rx.await.expect("cuda-registry thread dropped reply")
    }

    /// Drop all CUDA tensors belonging to `instance_id`; returns the count
    /// released.
    pub async fn drop_instance(&self, instance_id: String) -> usize {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropInstance { instance_id, reply })
            .await;
        rx.await.expect("cuda-registry thread dropped reply")
    }

    /// Drop exactly one context; returns the number of CUDA tensors released.
    pub async fn drop_context(&self, context_key: String) -> usize {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropContext { context_key, reply })
            .await;
        rx.await.expect("cuda-registry thread dropped reply")
    }

    /// Drop every registered tensor; returns the count released.
    pub async fn clear(&self) -> usize {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::Clear { reply }).await;
        rx.await.expect("cuda-registry thread dropped reply")
    }

    async fn dispatch(&self, cmd: RegistryCommand) {
        self.tx
            .send(cmd)
            .await
            .expect("cuda-registry thread is gone");
    }
}

/// Owns the registry and drains commands serially. Each op runs synchronously
/// on this thread, so a GIL/CUDA stall blocks only here.
fn registry_actor(mut registry: CudaTensorRegistry, mut rx: mpsc::Receiver<RegistryCommand>) {
    while let Some(cmd) = rx.blocking_recv() {
        match cmd {
            RegistryCommand::RegisterLayers {
                context_key,
                device_id,
                layers,
                native_fd,
                reply,
            } => {
                // Borrow the raw fd for the import; `OwnedFd` closes our copy
                // when it drops at the end of this arm (CUDA dups it internally).
                use std::os::fd::AsRawFd;
                let native = native_fd
                    .as_ref()
                    .map(|(fd, size)| (fd.as_raw_fd(), *size));
                let result = registry
                    .register_layers(&context_key, device_id, layers, native)
                    // Stringify here, on the GIL-owning thread, so the gRPC
                    // handler never needs `Python::attach` just to read the message.
                    .map_err(|err| Python::attach(|py| err.value(py).to_string()));
                let _ = reply.send(result);
            }
            RegistryCommand::DropInstance { instance_id, reply } => {
                let _ = reply.send(registry.drop_instance(&instance_id));
            }
            RegistryCommand::DropContext { context_key, reply } => {
                let _ = reply.send(registry.drop_context(&context_key));
            }
            RegistryCommand::Clear { reply } => {
                let _ = reply.send(registry.clear_and_count());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_context_removes_only_that_context() {
        let mut registry = CudaTensorRegistry::empty();
        registry
            .contexts
            .insert("instance-a:tp0:pp0:dev0".to_string(), ContextState::new(0));
        registry
            .contexts
            .insert("instance-a:tp0:pp1:dev1".to_string(), ContextState::new(1));

        assert_eq!(registry.drop_context("instance-a:tp0:pp0:dev0"), 0);

        assert!(!registry.contexts.contains_key("instance-a:tp0:pp0:dev0"));
        assert!(registry.contexts.contains_key("instance-a:tp0:pp1:dev1"));
    }

    #[test]
    fn register_layers_rejects_existing_context_before_materializing() {
        let mut registry = CudaTensorRegistry::empty();
        registry
            .contexts
            .insert("instance-a:tp0:pp0:dev0".to_string(), ContextState::new(7));

        let err = registry
            .register_layers("instance-a:tp0:pp0:dev0", 0, Vec::new(), None)
            .expect_err("existing context must be rejected");

        let message = Python::attach(|py| err.value(py).to_string());
        assert!(message.contains("already registered"));
        assert_eq!(
            registry
                .contexts
                .get("instance-a:tp0:pp0:dev0")
                .expect("existing context remains")
                .device_id,
            7
        );
    }
}
