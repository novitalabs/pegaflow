use cudarc::driver::{CudaContext, result::DriverError, sys};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, mpsc, oneshot};

mod cleanup;
pub(crate) use cleanup::RegistryCleanup;
use cleanup::RemovedContexts;

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub device_id: i32,
}

pub(crate) enum TensorRegistration {
    Python(Vec<u8>),
    /// Strided view into the batch's shared VMM allocation (fd arrives out-of-band).
    Native {
        offset_bytes: u64,
        size_bytes: u64,
    },
}

#[allow(dead_code, reason = "owners keep registered CUDA addresses valid")]
enum TensorOwner {
    Python(Py<PyAny>),
    Vmm(Arc<VmmRegistration>),
}

struct LayerTensor {
    owner: TensorOwner,
    metadata: TensorMetadata,
}

/// Imported client VMM allocation mapped into this process (`base_ptr` + size).
struct VmmMapping {
    context: Arc<CudaContext>,
    handle: sys::CUmemGenericAllocationHandle,
    base_ptr: sys::CUdeviceptr,
    size_bytes: usize,
}

impl VmmMapping {
    /// Import POSIX-fd VMM handle, reserve/map `alloc_size`, set device access.
    fn import(device_id: i32, fd: i32, alloc_size: usize) -> PyResult<Self> {
        let device = usize::try_from(device_id)
            .map_err(|_| PyValueError::new_err("device_id must be non-negative"))?;
        let context = CudaContext::new(device).map_err(|e| cuda_error("retain context", e))?;
        context
            .bind_to_thread()
            .map_err(|e| cuda_error("bind context", e))?;

        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: live VMM POSIX fd; CUDA dups it so the caller still owns `fd`.
        unsafe {
            sys::cuMemImportFromShareableHandle(
                &mut handle,
                fd as usize as *mut std::ffi::c_void,
                sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
            .result()
            .map_err(|e| cuda_error("import VMM shareable handle", e))?;
        }

        let mut base_ptr: sys::CUdeviceptr = 0;
        // SAFETY: reserve output is written to base_ptr.
        if let Err(e) =
            unsafe { sys::cuMemAddressReserve(&mut base_ptr, alloc_size, 0, 0, 0).result() }
        {
            // SAFETY: handle imported, not yet mapped.
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("reserve VMM address range", e));
        }
        // SAFETY: base_ptr reserved; handle imported.
        if let Err(e) = unsafe { sys::cuMemMap(base_ptr, alloc_size, 0, handle, 0).result() } {
            unsafe { sys::cuMemAddressFree(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("map VMM allocation", e));
        }

        let access = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            },
            flags: sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: range is mapped; one access desc.
        if let Err(e) = unsafe { sys::cuMemSetAccess(base_ptr, alloc_size, &access, 1).result() } {
            unsafe { sys::cuMemUnmap(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemAddressFree(base_ptr, alloc_size).result().ok() };
            unsafe { sys::cuMemRelease(handle).result().ok() };
            return Err(cuda_error("set VMM access", e));
        }

        Ok(Self {
            context,
            handle,
            base_ptr,
            size_bytes: alloc_size,
        })
    }
}

impl Drop for VmmMapping {
    fn drop(&mut self) {
        self.context
            .bind_to_thread()
            .expect("bind CUDA context before releasing VMM mapping");
        // SAFETY: reverse of import: unmap, free VA, release handle.
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

struct VmmRegistration {
    instance_id: String,
    device_id: i32,
    mapping: VmmMappingOwner,
    operations: OperationGate,
}

enum VmmMappingOwner {
    Cuda(VmmMapping),
    #[cfg(test)]
    Stub,
}

impl VmmMappingOwner {
    fn cuda(&self) -> &VmmMapping {
        match self {
            Self::Cuda(mapping) => mapping,
            #[cfg(test)]
            Self::Stub => panic!("test-only VMM registration has no CUDA mapping"),
        }
    }
}

#[derive(Clone, Default)]
struct OperationGate(Arc<RwLock<()>>);

impl OperationGate {
    fn enter(&self) -> Result<OwnedRwLockReadGuard<()>, ()> {
        Arc::clone(&self.0).try_read_owned().map_err(|_| ())
    }

    async fn close(&self) -> OwnedRwLockWriteGuard<()> {
        Arc::clone(&self.0).write_owned().await
    }
}

impl VmmRegistration {
    fn new(instance_id: String, device_id: i32, mapping: VmmMapping) -> Self {
        Self {
            instance_id,
            device_id,
            mapping: VmmMappingOwner::Cuda(mapping),
            operations: OperationGate::default(),
        }
    }

    #[cfg(test)]
    fn stub(instance_id: &str, device_id: i32) -> Self {
        Self {
            instance_id: instance_id.to_string(),
            device_id,
            mapping: VmmMappingOwner::Stub,
            operations: OperationGate::default(),
        }
    }

    fn acquire(
        self: &Arc<Self>,
        instance_id: &str,
        device_id: i32,
    ) -> Result<RegistrationGuard, String> {
        if self.instance_id != instance_id || self.device_id != device_id {
            return Err(format!(
                "native mapping belongs to instance {} device {}, not instance {instance_id} device {device_id}",
                self.instance_id, self.device_id
            ));
        }
        let active = self
            .operations
            .enter()
            .map_err(|_| format!("native instance {} is closing", self.instance_id))?;
        Ok(RegistrationGuard {
            _active: active,
            _registration: Arc::clone(self),
        })
    }

    async fn drain(&self) -> OwnedRwLockWriteGuard<()> {
        self.operations.close().await
    }
}

pub(crate) struct RegistrationGuard {
    _active: OwnedRwLockReadGuard<()>,
    _registration: Arc<VmmRegistration>,
}

fn cuda_error(operation: &str, error: DriverError) -> PyErr {
    PyRuntimeError::new_err(format!("{operation}: {error}"))
}

struct ContextState {
    device_id: i32,
    tensors: HashMap<String, LayerTensor>,
    native_registration: Option<Arc<VmmRegistration>>,
}

impl ContextState {
    fn new(device_id: i32) -> Self {
        Self {
            device_id,
            tensors: HashMap::new(),
            native_registration: None,
        }
    }
}

pub struct CudaTensorRegistry {
    contexts: HashMap<String, ContextState>,
    native_instances: HashMap<String, Arc<VmmRegistration>>,
    closing_native_instances: HashSet<String>,
}

impl CudaTensorRegistry {
    pub fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let torch = py.import("torch")?;
            let cuda = torch.getattr("cuda")?;
            cuda.call_method0("init")?;
            Ok(Self {
                contexts: HashMap::new(),
                native_instances: HashMap::new(),
                closing_native_instances: HashSet::new(),
            })
        })
    }

    pub fn empty() -> Self {
        Self {
            contexts: HashMap::new(),
            native_instances: HashMap::new(),
            closing_native_instances: HashSet::new(),
        }
    }

    fn register_layers(
        &mut self,
        context_key: &str,
        instance_id: &str,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        native_fd: Option<(i32, usize)>,
    ) -> PyResult<(Vec<TensorMetadata>, Option<RegistrationGuard>)> {
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

        let instance_prefix = format!("{instance_id}:");
        let has_instance_context = self
            .contexts
            .keys()
            .any(|key| key.starts_with(&instance_prefix));
        let registration = match native_fd {
            Some((fd, alloc_size)) => {
                if self.native_instances.contains_key(instance_id)
                    || self.closing_native_instances.contains(instance_id)
                {
                    return Err(PyValueError::new_err(format!(
                        "native instance {instance_id} is already registered"
                    )));
                }
                if has_instance_context {
                    return Err(PyValueError::new_err(format!(
                        "instance {instance_id} already has Python contexts"
                    )));
                }
                let mapping = VmmMapping::import(device_id, fd, alloc_size)?;
                Some(Arc::new(VmmRegistration::new(
                    instance_id.to_string(),
                    device_id,
                    mapping,
                )))
            }
            None => {
                if self.native_instances.contains_key(instance_id) {
                    return Err(PyValueError::new_err(format!(
                        "instance {instance_id} is registered through native VMM"
                    )));
                }
                None
            }
        };

        let mut context = ContextState::new(device_id);
        let mut metadatas = Vec::with_capacity(layers.len());
        for (layer_name, layer_registration) in layers {
            let layer_tensor = match layer_registration {
                TensorRegistration::Python(bytes) => {
                    Self::materialize_python_tensor(device_id, &bytes)?
                }
                TensorRegistration::Native {
                    offset_bytes,
                    size_bytes,
                } => {
                    let registration = registration.as_ref().ok_or_else(|| {
                        PyValueError::new_err("native tensor registration without an imported fd")
                    })?;
                    Self::materialize_native_tensor(
                        device_id,
                        registration,
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

        let guard = if let Some(registration) = registration {
            let guard = registration
                .acquire(instance_id, device_id)
                .map_err(PyRuntimeError::new_err)?;
            context.native_registration = Some(Arc::clone(&registration));
            self.native_instances
                .insert(instance_id.to_string(), registration);
            Some(guard)
        } else {
            None
        };
        self.contexts.insert(context_key.to_string(), context);
        Ok((metadatas, guard))
    }

    fn drop_context(&mut self, context_key: &str) -> RemovedContexts {
        let removed = self.release_contexts(vec![context_key.to_string()]);
        for registration in &removed.registrations {
            self.native_instances.remove(&registration.instance_id);
            self.closing_native_instances
                .insert(registration.instance_id.clone());
        }
        removed
    }

    fn drop_instance(&mut self, instance_id: &str) -> RemovedContexts {
        let prefix = format!("{instance_id}:");
        let keys: Vec<String> = self
            .contexts
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();
        let mut removed = self.release_contexts(keys);
        if let Some(registration) = self.native_instances.remove(instance_id)
            && !removed
                .registrations
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &registration))
        {
            removed.registrations.push(registration);
        }
        if !removed.registrations.is_empty() {
            self.closing_native_instances
                .insert(instance_id.to_string());
        }
        removed
    }

    /// Clear all contexts and return the total number of tensors removed.
    fn clear(&mut self) -> RemovedContexts {
        let keys: Vec<String> = self.contexts.keys().cloned().collect();
        let mut removed = self.release_contexts(keys);
        for (_, registration) in self.native_instances.drain() {
            if !removed
                .registrations
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &registration))
            {
                removed.registrations.push(registration);
            }
        }
        self.closing_native_instances.extend(
            removed
                .registrations
                .iter()
                .map(|registration| registration.instance_id.clone()),
        );
        removed
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
    fn release_contexts(&mut self, keys: Vec<String>) -> RemovedContexts {
        let tensor_count: usize = keys
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .map(|ctx| ctx.tensors.len())
            .sum();

        let needs_python = keys
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .flat_map(|context| context.tensors.values())
            .any(|tensor| matches!(&tensor.owner, TensorOwner::Python(_)));
        let removed: Vec<_> = keys
            .iter()
            .filter_map(|key| self.contexts.remove(key))
            .collect();
        let registrations = removed
            .iter()
            .filter_map(|context| context.native_registration.as_ref())
            .cloned()
            .collect::<Vec<_>>();

        if needs_python {
            Python::attach(|py| {
                drop(removed);
                let gc = py.import("gc").expect("gc module");
                let _ = gc.call_method0("collect");

                let torch = py.import("torch").expect("torch module");
                let cuda = torch.getattr("cuda").expect("torch.cuda");
                let _ = cuda.call_method0("empty_cache");
            });
        } else {
            drop(removed);
        }

        RemovedContexts::new(tensor_count, registrations)
    }

    fn acquire_registration(
        &self,
        instance_id: &str,
        device_id: i32,
    ) -> Result<Option<RegistrationGuard>, String> {
        if let Some(registration) = self.native_instances.get(instance_id) {
            return registration.acquire(instance_id, device_id).map(Some);
        }
        if self.closing_native_instances.contains(instance_id) {
            return Err(format!("native instance {instance_id} is closing"));
        }
        Ok(None)
    }

    fn finish_cleanup(&mut self, cleanup: RegistryCleanup) {
        for registration in &cleanup._registrations {
            self.closing_native_instances
                .remove(&registration.instance_id);
        }
        drop(cleanup);
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

    fn materialize_native_tensor(
        device_id: i32,
        registration: &Arc<VmmRegistration>,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> PyResult<LayerTensor> {
        let mapping = registration.mapping.cuda();
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
            owner: TensorOwner::Vmm(Arc::clone(registration)),
            metadata: TensorMetadata {
                data_ptr,
                size_bytes: size,
                device_id,
            },
        })
    }
}

type RegisterLayersResult = Result<(Vec<TensorMetadata>, Option<RegistrationGuard>), String>;
type RegistrationPermit = OwnedRwLockReadGuard<()>;

enum RegistryCommand {
    RegisterLayers {
        context_key: String,
        instance_id: String,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        /// `Some((fd, alloc_size))` for native VMM; `None` for Python.
        native_fd: Option<(std::os::fd::OwnedFd, usize)>,
        // Stringified on the actor thread (holds GIL).
        reply: oneshot::Sender<RegisterLayersResult>,
    },
    DropInstance {
        instance_id: String,
        reply: oneshot::Sender<RemovedContexts>,
    },
    DropContext {
        context_key: String,
        reply: oneshot::Sender<RemovedContexts>,
    },
    Clear {
        reply: oneshot::Sender<RemovedContexts>,
    },
    AcquireRegistration {
        instance_id: String,
        device_id: i32,
        reply: oneshot::Sender<Result<Option<RegistrationGuard>, String>>,
    },
    FinishCleanup {
        cleanup: RegistryCleanup,
        reply: oneshot::Sender<()>,
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
    registration_barrier: Arc<RwLock<()>>,
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
        Self {
            tx,
            registration_barrier: Arc::new(RwLock::new(())),
        }
    }

    /// Register layers under `context_key`. `native_fd`: VMM fd + size, or `None` for Python.
    pub(crate) async fn register_layers(
        &self,
        context_key: String,
        instance_id: String,
        device_id: i32,
        layers: Vec<(String, TensorRegistration)>,
        native_fd: Option<(std::os::fd::OwnedFd, usize)>,
    ) -> Result<
        (
            Vec<TensorMetadata>,
            Option<RegistrationGuard>,
            RegistrationPermit,
        ),
        String,
    > {
        let permit = Arc::clone(&self.registration_barrier).read_owned().await;
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::RegisterLayers {
            context_key,
            instance_id,
            device_id,
            layers,
            native_fd,
            reply,
        })
        .await;
        rx.await
            .expect("cuda-registry thread dropped reply")
            .map(|(metadatas, registration)| (metadatas, registration, permit))
    }

    /// Drain all CUDA tensors belonging to `instance_id`.
    ///
    /// The returned cleanup guard owns native mappings until the caller removes
    /// the corresponding engine instance and drops the guard.
    pub(crate) async fn drop_instance(&self, instance_id: String) -> RegistryCleanup {
        let barrier = Arc::clone(&self.registration_barrier).write_owned().await;
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropInstance { instance_id, reply })
            .await;
        self.finish_removal(
            rx.await.expect("cuda-registry thread dropped reply"),
            Some(barrier),
        )
        .await
    }

    /// Drain exactly one context, retaining native mappings in the returned guard.
    pub(crate) async fn drop_context(&self, context_key: String) -> RegistryCleanup {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::DropContext { context_key, reply })
            .await;
        self.finish_removal(rx.await.expect("cuda-registry thread dropped reply"), None)
            .await
    }

    /// Drain every registered tensor, retaining native mappings in the returned guard.
    pub(crate) async fn clear(&self) -> RegistryCleanup {
        let barrier = Arc::clone(&self.registration_barrier).write_owned().await;
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::Clear { reply }).await;
        self.finish_removal(
            rx.await.expect("cuda-registry thread dropped reply"),
            Some(barrier),
        )
        .await
    }

    /// Release a drained cleanup on the dedicated registry thread.
    ///
    /// Call this only after the engine has discarded the instance's raw CUDA
    /// addresses. Once dispatched, completion is independent of caller
    /// cancellation; the reply confirms that CUDA teardown has finished.
    pub(crate) async fn finish_cleanup(&self, cleanup: RegistryCleanup) {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::FinishCleanup { cleanup, reply })
            .await;
        rx.await
            .expect("cuda-registry thread dropped cleanup reply");
    }

    pub(crate) async fn acquire_registration(
        &self,
        instance_id: String,
        device_id: i32,
    ) -> Result<Option<RegistrationGuard>, String> {
        let (reply, rx) = oneshot::channel();
        self.dispatch(RegistryCommand::AcquireRegistration {
            instance_id,
            device_id,
            reply,
        })
        .await;
        rx.await.expect("cuda-registry thread dropped reply")
    }

    async fn finish_removal(
        &self,
        removed: RemovedContexts,
        registration_barrier: Option<OwnedRwLockWriteGuard<()>>,
    ) -> RegistryCleanup {
        let RemovedContexts {
            tensor_count,
            mut registrations,
        } = removed;
        registrations.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
        registrations.dedup_by(|left, right| Arc::ptr_eq(left, right));
        let mut drained = Vec::with_capacity(registrations.len());
        for registration in &registrations {
            drained.push(registration.drain().await);
        }
        RegistryCleanup {
            tensor_count,
            _drained: drained,
            _registrations: registrations,
            _registration_barrier: registration_barrier,
        }
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
                instance_id,
                device_id,
                layers,
                native_fd,
                reply,
            } => {
                use std::os::fd::AsRawFd;
                let native = native_fd.as_ref().map(|(fd, size)| (fd.as_raw_fd(), *size));
                let result = registry
                    .register_layers(&context_key, &instance_id, device_id, layers, native)
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
                let _ = reply.send(registry.clear());
            }
            RegistryCommand::AcquireRegistration {
                instance_id,
                device_id,
                reply,
            } => {
                let _ = reply.send(registry.acquire_registration(&instance_id, device_id));
            }
            RegistryCommand::FinishCleanup { cleanup, reply } => {
                registry.finish_cleanup(cleanup);
                let _ = reply.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests;
