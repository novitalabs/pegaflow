use pegaflow_transfer::{DomainAddress, MooncakeTransferEngine};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

#[pyclass(name = "TransferEngine")]
pub(crate) struct PyTransferEngine {
    engine: Arc<Mutex<MooncakeTransferEngine>>,
}

fn parse_local_hostname_port(local_hostname: &str) -> Option<u16> {
    if let Some(stripped) = local_hostname.strip_prefix('[') {
        let (_, port_str) = stripped.rsplit_once("]:")?;
        return port_str.parse::<u16>().ok().filter(|port| *port != 0);
    }

    let (_, port_str) = local_hostname.rsplit_once(':')?;
    port_str.parse::<u16>().ok().filter(|port| *port != 0)
}

fn resolve_transfer_rpc_port(local_hostname: &str) -> Option<u16> {
    parse_local_hostname_port(local_hostname)
}

fn resolve_transfer_nic(device_name: &str) -> Option<String> {
    if !device_name.trim().is_empty() {
        return Some(device_name.trim().to_string());
    }

    std::env::var("PEGAFLOW_TRANSFER_NIC")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[pymethods]
impl PyTransferEngine {
    #[new]
    fn new() -> Self {
        pegaflow_transfer::init_logging();
        Self {
            engine: Arc::new(Mutex::new(MooncakeTransferEngine::new())),
        }
    }

    #[pyo3(signature = (local_hostname, metadata_server, protocol, device_name))]
    fn initialize(
        &self,
        py: Python<'_>,
        local_hostname: &str,
        metadata_server: &str,
        protocol: &str,
        device_name: &str,
    ) -> i32 {
        let _ = metadata_server;
        let _ = protocol;

        py.detach(move || {
            let local_hostname = local_hostname.to_string();
            let device_name = device_name.to_string();
            let Some(nic_name) = resolve_transfer_nic(&device_name) else {
                log::error!("transfer initialize failed: NIC not configured");
                return -1;
            };
            let Some(rpc_port) = resolve_transfer_rpc_port(&local_hostname) else {
                log::error!(
                    "transfer initialize failed: local_hostname must include a non-zero port, got `{local_hostname}`"
                );
                return -1;
            };
            let mut guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("transfer initialize failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.initialize(nic_name, rpc_port) {
                Ok(()) => 0,
                Err(error) => {
                    log::error!("transfer initialize failed: {error}");
                    -1
                }
            }
        })
    }

    fn get_rpc_port(&self, py: Python<'_>) -> i32 {
        py.detach(|| {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("get_rpc_port failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.get_rpc_port() {
                Ok(port) => i32::from(port),
                Err(error) => {
                    log::error!("get_rpc_port failed: {error}");
                    -1
                }
            }
        })
    }

    fn get_session_id(&self, py: Python<'_>) -> Vec<u8> {
        py.detach(|| {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("get_session_id failed: engine mutex poisoned");
                    return Vec::new();
                }
            };
            guard.get_session_id().to_bytes().to_vec()
        })
    }

    fn register_memory(&self, py: Python<'_>, ptr: u64, len: usize) -> i32 {
        py.detach(move || {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("register_memory failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.register_memory(ptr, len) {
                Ok(()) => 0,
                Err(error) => {
                    log::error!("register_memory failed: ptr={ptr:#x} len={len}, error={error}");
                    -1
                }
            }
        })
    }

    fn unregister_memory(&self, py: Python<'_>, ptr: u64) -> i32 {
        py.detach(move || {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("unregister_memory failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.unregister_memory(ptr) {
                Ok(()) => 0,
                Err(error) => {
                    log::error!("unregister_memory failed: ptr={ptr:#x}, error={error}");
                    -1
                }
            }
        })
    }

    fn batch_register_memory(&self, py: Python<'_>, ptrs: Vec<u64>, lens: Vec<usize>) -> i32 {
        py.detach(move || {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("batch_register_memory failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.batch_register_memory(&ptrs, &lens) {
                Ok(()) => 0,
                Err(error) => {
                    log::error!(
                        "batch_register_memory failed: ptrs={}, lens={}, error={error}",
                        ptrs.len(),
                        lens.len()
                    );
                    -1
                }
            }
        })
    }

    fn batch_unregister_memory(&self, py: Python<'_>, ptrs: Vec<u64>) -> i32 {
        py.detach(move || {
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("batch_unregister_memory failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.batch_unregister_memory(&ptrs) {
                Ok(()) => 0,
                Err(error) => {
                    log::error!(
                        "batch_unregister_memory failed: ptrs={}, error={error}",
                        ptrs.len()
                    );
                    -1
                }
            }
        })
    }

    fn transfer_sync_write(
        &self,
        py: Python<'_>,
        session_id: Vec<u8>,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> i64 {
        py.detach(move || {
            let Some(session_id) = DomainAddress::from_bytes(&session_id) else {
                log::error!("transfer_sync_write failed: invalid session_id length");
                return -1;
            };
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("transfer_sync_write failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.transfer_sync_write(&session_id, local_ptr, remote_ptr, len) {
                Ok(bytes) => i64::try_from(bytes).unwrap_or(-1),
                Err(error) => {
                    log::error!(
                        "transfer_sync_write failed: local_ptr={local_ptr:#x} remote_ptr={remote_ptr:#x} len={len}, error={error}"
                    );
                    -1
                }
            }
        })
    }

    fn batch_transfer_sync_write(
        &self,
        py: Python<'_>,
        session_id: Vec<u8>,
        local_ptrs: Vec<u64>,
        remote_ptrs: Vec<u64>,
        lens: Vec<usize>,
    ) -> i64 {
        py.detach(move || {
            let Some(session_id) = DomainAddress::from_bytes(&session_id) else {
                log::error!("batch_transfer_sync_write failed: invalid session_id length");
                return -1;
            };
            let guard = match self.engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::error!("batch_transfer_sync_write failed: engine mutex poisoned");
                    return -1;
                }
            };
            match guard.batch_transfer_sync_write(&session_id, &local_ptrs, &remote_ptrs, &lens) {
                Ok(bytes) => i64::try_from(bytes).unwrap_or(-1),
                Err(error) => {
                    log::error!(
                        "batch_transfer_sync_write failed: local_ptrs={} remote_ptrs={} lens={}, error={error}",
                        local_ptrs.len(),
                        remote_ptrs.len(),
                        lens.len()
                    );
                    -1
                }
            }
        })
    }
}

pub(crate) fn register_py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransferEngine>()?;
    Ok(())
}
