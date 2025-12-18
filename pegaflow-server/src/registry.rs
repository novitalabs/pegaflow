use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub device_id: i32,
}

struct LayerTensor {
    tensor: Py<PyAny>,
    metadata: TensorMetadata,
}

// Note: No custom Drop impl needed for LayerTensor.
// PyO3's Py<PyAny> will automatically:
// 1. Acquire the GIL when dropped
// 2. Decrement the Python object's reference count
// 3. Let Python's garbage collector handle the actual cleanup
// This is the correct way to release CUDA IPC tensors - the mapped memory
// will be unmapped when the tensor's storage is garbage collected.

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
            Ok(Self { contexts: HashMap::new() })
        })
    }

    pub fn register_layer(
        &mut self,
        context_key: &str,
        layer_name: &str,
        device_id: i32,
        wrapper_bytes: &[u8],
    ) -> PyResult<TensorMetadata> {
        let layer_tensor = Self::materialize_tensor(device_id, wrapper_bytes)?;
        let metadata = layer_tensor.metadata.clone();

        let context = self
            .contexts
            .entry(context_key.to_string())
            .or_insert_with(|| ContextState::new(device_id));

        if context.device_id != metadata.device_id {
            return Err(PyValueError::new_err(format!(
                "context {context_key} is pinned to device {} but got {}",
                context.device_id, metadata.device_id
            )));
        }

        context.tensors.insert(layer_name.to_string(), layer_tensor);

        Ok(metadata)
    }

    pub fn drop_instance(&mut self, instance_id: &str) -> usize {
        let prefix = format!("{instance_id}:");

        // Collect keys to remove first
        let keys_to_remove: Vec<String> = self
            .contexts
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();

        // Count total tensors across all contexts being removed
        let tensor_count: usize = keys_to_remove
            .iter()
            .filter_map(|key| self.contexts.get(key))
            .map(|ctx| ctx.tensors.len())
            .sum();

        if tensor_count == 0 {
            return 0;
        }

        // Remove contexts under GIL to ensure proper Python object cleanup.
        // The Py<PyAny> inside LayerTensor will be dropped here, which will
        // decrement the reference count and allow Python to garbage collect
        // the CUDA IPC tensors, releasing the mapped GPU memory.
        Python::attach(|py| {
            for key in keys_to_remove {
                self.contexts.remove(&key);
            }

            // Force garbage collection to release CUDA IPC memory immediately.
            // Without this, Python's GC may defer cleanup, leaving GPU memory mapped.
            let gc = py.import("gc").expect("gc module");
            let _ = gc.call_method0("collect");

            // Clear CUDA memory cache to return memory to the device
            let torch = py.import("torch").expect("torch module");
            let cuda = torch.getattr("cuda").expect("torch.cuda");
            let _ = cuda.call_method0("empty_cache");
        });

        tensor_count
    }

    pub fn clear(&mut self) {
        // Clear all contexts under GIL to ensure proper Python object cleanup
        Python::attach(|py| {
            self.contexts.clear();

            // Force garbage collection and clear CUDA cache
            let gc = py.import("gc").expect("gc module");
            let _ = gc.call_method0("collect");

            let torch = py.import("torch").expect("torch module");
            let cuda = torch.getattr("cuda").expect("torch.cuda");
            let _ = cuda.call_method0("empty_cache");
        });
    }

    fn materialize_tensor(device_id: i32, wrapper_bytes: &[u8]) -> PyResult<LayerTensor> {
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
                tensor: tensor_owned,
                metadata: TensorMetadata {
                    data_ptr,
                    size_bytes,
                    device_id: resolved_device,
                },
            })
        })
    }
}
