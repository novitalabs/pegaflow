use cudarc::driver::{CudaContext, result::DriverError, sys};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet};
use std::mem::MaybeUninit;
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
    CudaIpc {
        handle: Vec<u8>,
        offset_bytes: u64,
        size_bytes: u64,
    },
}

#[allow(dead_code, reason = "owners keep registered CUDA addresses valid")]
enum TensorOwner {
    Python(Py<PyAny>),
    CudaIpc(Arc<CudaIpcMapping>),
}

struct LayerTensor {
    owner: TensorOwner,
    metadata: TensorMetadata,
}

struct CudaIpcMapping {
    context: Arc<CudaContext>,
    base_ptr: u64,
    size_bytes: usize,
}

impl CudaIpcMapping {
    fn open(device_id: i32, bytes: &[u8]) -> PyResult<Arc<Self>> {
        if bytes.len() != std::mem::size_of::<sys::CUipcMemHandle>() {
            return Err(PyValueError::new_err("invalid CUDA IPC handle size"));
        }
        let device = usize::try_from(device_id)
            .map_err(|_| PyValueError::new_err("device_id must be non-negative"))?;
        let context = CudaContext::new(device).map_err(|e| cuda_error("retain context", e))?;
        context
            .bind_to_thread()
            .map_err(|e| cuda_error("bind context", e))?;

        let mut handle = MaybeUninit::<sys::CUipcMemHandle>::uninit();
        // SAFETY: the length check above covers the entire opaque handle.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                handle.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        let mut base_ptr = 0;
        // SAFETY: CUDA initializes base_ptr from the fully copied handle.
        unsafe {
            sys::cuIpcOpenMemHandle_v2(
                &mut base_ptr,
                handle.assume_init(),
                sys::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
            )
            .result()
            .map_err(|e| cuda_error("open CUDA IPC handle", e))?;
        }
        let mut mapping = Self {
            context,
            base_ptr,
            size_bytes: 0,
        };

        let mut actual_device = -1i32;
        // SAFETY: base_ptr is a live CUDA mapping and output storage is valid.
        unsafe {
            sys::cuPointerGetAttribute(
                (&mut actual_device as *mut i32).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                base_ptr,
            )
            .result()
            .map_err(|e| cuda_error("query mapping device", e))?;
        }
        if actual_device != device_id {
            return Err(PyValueError::new_err(format!(
                "CUDA IPC allocation belongs to device {actual_device}, not {device_id}"
            )));
        }

        let mut allocation_base = 0;
        // SAFETY: base_ptr is a live CUDA mapping and output storage is valid.
        unsafe {
            sys::cuMemGetAddressRange_v2(&mut allocation_base, &mut mapping.size_bytes, base_ptr)
                .result()
                .map_err(|e| cuda_error("query mapping range", e))?;
        }
        if allocation_base != base_ptr {
            return Err(PyRuntimeError::new_err(
                "CUDA IPC mapping is not allocation-aligned",
            ));
        }
        Ok(Arc::new(mapping))
    }
}

impl Drop for CudaIpcMapping {
    fn drop(&mut self) {
        self.context
            .bind_to_thread()
            .expect("bind CUDA context before closing IPC mapping");
        // SAFETY: this object owns one successful CUDA IPC open.
        unsafe { sys::cuIpcCloseMemHandle(self.base_ptr).result() }
            .expect("close CUDA IPC mapping");
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

    fn register_layers(
        &mut self,
        context_key: &str,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
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

        let mut context = ContextState::new(device_id);
        let mut metadatas = Vec::with_capacity(layers.len());
        let mut mappings = HashMap::new();
        for (layer_name, registration) in layers {
            let layer_tensor = match registration {
                TensorRegistration::Python(bytes) => {
                    Self::materialize_python_tensor(device_id, &bytes)?
                }
                TensorRegistration::CudaIpc {
                    handle,
                    offset_bytes,
                    size_bytes,
                } => Self::materialize_cuda_ipc_tensor(
                    device_id,
                    handle,
                    offset_bytes,
                    size_bytes,
                    &mut mappings,
                )?,
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

    fn materialize_cuda_ipc_tensor(
        device_id: i32,
        handle: Vec<u8>,
        offset_bytes: u64,
        size_bytes: u64,
        mappings: &mut HashMap<Vec<u8>, Arc<CudaIpcMapping>>,
    ) -> PyResult<LayerTensor> {
        let offset = usize::try_from(offset_bytes)
            .map_err(|_| PyValueError::new_err("CUDA IPC offset does not fit usize"))?;
        let size = usize::try_from(size_bytes)
            .map_err(|_| PyValueError::new_err("CUDA IPC size does not fit usize"))?;
        let mapping = if let Some(mapping) = mappings.get(&handle) {
            Arc::clone(mapping)
        } else {
            let mapping = CudaIpcMapping::open(device_id, &handle)?;
            mappings.insert(handle, Arc::clone(&mapping));
            mapping
        };
        let end = offset
            .checked_add(size)
            .ok_or_else(|| PyValueError::new_err("CUDA IPC view range overflows usize"))?;
        if size == 0 || end > mapping.size_bytes {
            return Err(PyValueError::new_err(
                "CUDA IPC view is outside its allocation",
            ));
        }
        let data_ptr = mapping
            .base_ptr
            .checked_add(offset_bytes)
            .ok_or_else(|| PyValueError::new_err("CUDA IPC view pointer overflows u64"))?;
        Ok(LayerTensor {
            owner: TensorOwner::CudaIpc(mapping),
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
    /// partial context.
    pub(crate) async fn register_layers(
        &self,
        context_key: String,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
    ) -> Result<Vec<TensorMetadata>, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::RegisterLayers {
            context_key,
            device_id,
            layers,
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
                reply,
            } => {
                let result = registry
                    .register_layers(&context_key, device_id, layers)
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
            .register_layers("instance-a:tp0:pp0:dev0", 0, Vec::new())
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
