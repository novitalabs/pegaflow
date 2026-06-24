use crate::{PegaFlowError, u64_to_usize};

use pegaflow_transfer::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

fn rdma_v1_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PegaFlowError::new_err(format!("{context}: {err}"))
}

fn nonnull_from_u64(ptr: u64, field: &str) -> PyResult<NonNull<u8>> {
    NonNull::new(ptr as *mut u8)
        .ok_or_else(|| PyValueError::new_err(format!("{field} must be non-zero")))
}

fn py_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    let value = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?;
    value.extract()
}

#[pyclass(frozen)]
struct PegaRdmaV1Handshake {
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    has_metadata: bool,
    metadata: Option<Vec<u8>>,
}

#[pymethods]
impl PegaRdmaV1Handshake {
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Option<Bound<'py, pyo3::types::PyBytes>> {
        self.metadata
            .as_ref()
            .map(|metadata| pyo3::types::PyBytes::new(py, metadata))
    }

    fn __repr__(&self) -> String {
        format!(
            "PegaRdmaV1Handshake(status={:?}, has_metadata={})",
            self.status, self.has_metadata
        )
    }
}

struct PendingRead {
    receivers: Vec<Option<mea::oneshot::Receiver<pegaflow_transfer::Result<usize>>>>,
    bytes_done: usize,
}

#[pyclass]
struct PegaRdmaV1Engine {
    engine: Arc<TransferEngine>,
    pending_reads: Mutex<HashMap<u64, PendingRead>>,
    next_handle: Mutex<u64>,
}

#[pymethods]
impl PegaRdmaV1Engine {
    #[new]
    #[pyo3(signature = (*, nics, qps_per_peer = 4))]
    fn new(nics: Vec<String>, qps_per_peer: usize) -> PyResult<Self> {
        if nics.is_empty() {
            return Err(PyValueError::new_err("nics must not be empty"));
        }
        let engine = TransferEngine::new(&nics, qps_per_peer)
            .map_err(|err| rdma_v1_error("v1 transfer engine init failed", err))?;
        Ok(Self {
            engine: Arc::new(engine),
            pending_reads: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
        })
    }

    fn register_memory(&self, py: Python<'_>, regions: Vec<Py<PyDict>>) -> PyResult<()> {
        let mut native = Vec::with_capacity(regions.len());
        for region in regions {
            let region = region.bind(py);
            let addr: u64 = py_get(region, "addr")?;
            let len: u64 = py_get(region, "len")?;
            if len == 0 {
                return Err(PyValueError::new_err("memory region len must be positive"));
            }
            native.push(MemoryRegion {
                ptr: nonnull_from_u64(addr, "addr")?,
                len: u64_to_usize(len, "memory region len")?,
            });
        }
        self.engine
            .register_memory(&native)
            .map_err(|err| rdma_v1_error("register_memory failed", err))
    }

    fn unregister_memory(&self, addrs: Vec<u64>) -> PyResult<()> {
        let mut ptrs = Vec::with_capacity(addrs.len());
        for addr in addrs {
            ptrs.push(nonnull_from_u64(addr, "addr")?);
        }
        self.engine
            .unregister_memory(&ptrs)
            .map_err(|err| rdma_v1_error("unregister_memory failed", err))
    }

    fn get_or_prepare(&self, remote_addr: String) -> PyResult<PegaRdmaV1Handshake> {
        let status = self
            .engine
            .get_or_prepare(&remote_addr)
            .map_err(|err| rdma_v1_error("get_or_prepare failed", err))?;
        Ok(match status {
            ConnectionStatus::Existing => PegaRdmaV1Handshake {
                status: "existing".to_string(),
                has_metadata: false,
                metadata: None,
            },
            ConnectionStatus::Connecting => PegaRdmaV1Handshake {
                status: "connecting".to_string(),
                has_metadata: false,
                metadata: None,
            },
            ConnectionStatus::Prepared(metadata) => PegaRdmaV1Handshake {
                status: "prepared".to_string(),
                has_metadata: true,
                metadata: Some(metadata.to_bytes()),
            },
        })
    }

    fn local_meta_for<'py>(
        &self,
        py: Python<'py>,
        remote_addr: String,
    ) -> Option<Bound<'py, pyo3::types::PyBytes>> {
        self.engine
            .local_meta_for(&remote_addr)
            .map(|metadata| pyo3::types::PyBytes::new(py, &metadata.to_bytes()))
    }

    fn complete_handshake(
        &self,
        remote_addr: String,
        local_metadata: Vec<u8>,
        remote_metadata: Vec<u8>,
    ) -> PyResult<()> {
        let local = HandshakeMetadata::from_bytes(&local_metadata)
            .map_err(|err| rdma_v1_error("decode local handshake failed", err))?;
        let remote = HandshakeMetadata::from_bytes(&remote_metadata)
            .map_err(|err| rdma_v1_error("decode remote handshake failed", err))?;
        self.engine
            .complete_handshake(&remote_addr, &local, &remote)
            .map_err(|err| rdma_v1_error("complete_handshake failed", err))
    }

    fn abort_handshake(&self, remote_addr: String, local_metadata: Vec<u8>) -> PyResult<()> {
        let local = HandshakeMetadata::from_bytes(&local_metadata)
            .map_err(|err| rdma_v1_error("decode local handshake failed", err))?;
        self.engine.abort_handshake(&remote_addr, &local);
        Ok(())
    }

    fn read_async(
        &self,
        py: Python<'_>,
        remote_addr: String,
        descs: Vec<Py<PyDict>>,
    ) -> PyResult<u64> {
        let mut native = Vec::with_capacity(descs.len());
        for desc in descs {
            let desc = desc.bind(py);
            let local_addr: u64 = py_get(desc, "local_addr")?;
            let remote_addr_value: u64 = py_get(desc, "remote_addr")?;
            let len: u64 = py_get(desc, "len")?;
            if len == 0 {
                return Err(PyValueError::new_err("transfer desc len must be positive"));
            }
            native.push(TransferDesc {
                local_ptr: nonnull_from_u64(local_addr, "local_addr")?,
                remote_ptr: nonnull_from_u64(remote_addr_value, "remote_addr")?,
                len: u64_to_usize(len, "transfer desc len")?,
            });
        }
        let receivers = self
            .engine
            .batch_transfer_async(TransferOp::Read, &remote_addr, &native)
            .map_err(|err| rdma_v1_error("submit RDMA READ failed", err))?;
        let mut next_handle = self
            .next_handle
            .lock()
            .map_err(|_| PyRuntimeError::new_err("next_handle mutex poisoned"))?;
        let handle = *next_handle;
        *next_handle = next_handle
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("RDMA read handle overflow"))?;
        self.pending_reads
            .lock()
            .map_err(|_| PyRuntimeError::new_err("pending_reads mutex poisoned"))?
            .insert(
                handle,
                PendingRead {
                    receivers: receivers.into_iter().map(Some).collect(),
                    bytes_done: 0,
                },
            );
        Ok(handle)
    }

    fn check_read(&self, handle: u64) -> PyResult<String> {
        let mut pending = self
            .pending_reads
            .lock()
            .map_err(|_| PyRuntimeError::new_err("pending_reads mutex poisoned"))?;
        let read = pending
            .get_mut(&handle)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RDMA read handle {handle}")))?;
        for slot in &mut read.receivers {
            let Some(rx) = slot.as_ref() else {
                continue;
            };
            match rx.try_recv() {
                Ok(Ok(bytes)) => {
                    read.bytes_done = read.bytes_done.saturating_add(bytes);
                    *slot = None;
                }
                Ok(Err(err)) => {
                    return Err(rdma_v1_error("RDMA READ failed", err));
                }
                Err(mea::oneshot::TryRecvError::Empty) => {
                    return Ok("pending".to_string());
                }
                Err(mea::oneshot::TryRecvError::Disconnected) => {
                    return Err(PegaFlowError::new_err(
                        "RDMA READ completion channel closed",
                    ));
                }
            }
        }
        if read.receivers.iter().any(Option::is_some) {
            return Ok("pending".to_string());
        }
        let total = read.bytes_done;
        pending.remove(&handle);
        log::debug!("[PegaRdmaV1Engine] RDMA READ done handle={handle} bytes={total}");
        Ok("done".to_string())
    }

    fn release_read(&self, handle: u64) -> PyResult<()> {
        self.pending_reads
            .lock()
            .map_err(|_| PyRuntimeError::new_err("pending_reads mutex poisoned"))?
            .remove(&handle);
        Ok(())
    }

    fn invalidate_connection(&self, remote_addr: String) {
        self.engine.invalidate_connection(&remote_addr);
    }

    fn num_qps(&self) -> usize {
        self.engine.num_qps()
    }
}

pub(crate) fn add_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PegaRdmaV1Engine>()?;
    m.add_class::<PegaRdmaV1Handshake>()?;
    Ok(())
}
