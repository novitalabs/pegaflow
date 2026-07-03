use crate::{PegaFlowError, u64_to_usize};

use pegaflow_transfer::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
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

#[derive(Clone, Copy)]
struct BlockEntry {
    // Keep block table entries as plain integers.  The PyO3 class can then stay
    // Send-friendly, and pointers are converted to NonNull only while building
    // the native transfer descriptors for a submitted READ.
    addr: u64,
    len: usize,
}

struct EngineState {
    pending_handshakes: HashMap<String, HandshakeMetadata>,
    pending_reads: HashMap<u64, PendingRead>,
    next_handle: u64,
    next_local_table_id: u64,
    local_block_tables: HashMap<u64, Vec<BlockEntry>>,
    remote_block_tables: HashMap<(String, usize), Vec<BlockEntry>>,
}

#[pyclass]
struct PegaRdmaV1Engine {
    engine: Arc<TransferEngine>,
    state: Mutex<EngineState>,
}

#[pymethods]
impl PegaRdmaV1Engine {
    #[new]
    #[pyo3(signature = (*, nics, qps_per_peer = 4))]
    fn new(nics: Vec<String>, qps_per_peer: usize) -> PyResult<Self> {
        // Create one RDMA v1 transfer engine bound to the connector-provided
        // NIC list.  Python owns the lifecycle through the connector worker.
        if nics.is_empty() {
            return Err(PyValueError::new_err("nics must not be empty"));
        }
        let engine = TransferEngine::new(&nics, qps_per_peer)
            .map_err(|err| rdma_v1_error("v1 transfer engine init failed", err))?;
        Ok(Self {
            engine: Arc::new(engine),
            state: Mutex::new(EngineState {
                pending_handshakes: HashMap::new(),
                pending_reads: HashMap::new(),
                next_handle: 1,
                next_local_table_id: 1,
                local_block_tables: HashMap::new(),
                remote_block_tables: HashMap::new(),
            }),
        })
    }

    fn register_memory(&self, py: Python<'_>, regions: Vec<Py<PyDict>>) -> PyResult<()> {
        // Register local memory regions that Pega RDMA v1 may use as READ
        // destinations or sources, depending on which worker side this engine
        // is running on.
        let mut raw_regions = Vec::with_capacity(regions.len());
        for region in regions {
            let region = region.bind(py);
            let addr: u64 = py_get(region, "addr")?;
            let len: u64 = py_get(region, "len")?;
            if len == 0 {
                return Err(PyValueError::new_err("memory region len must be positive"));
            }
            raw_regions.push((addr, u64_to_usize(len, "memory region len")?));
        }
        let engine = Arc::clone(&self.engine);
        py.detach(move || -> PyResult<()> {
            let mut native = Vec::with_capacity(raw_regions.len());
            for (addr, len) in raw_regions {
                native.push(MemoryRegion {
                    ptr: nonnull_from_u64(addr, "addr")?,
                    len,
                });
            }
            engine
                .register_memory(&native)
                .map_err(|err| rdma_v1_error("register_memory failed", err))
        })
    }

    fn unregister_memory(&self, py: Python<'_>, addrs: Vec<u64>) -> PyResult<()> {
        // Drop registrations for local memory regions during connector
        // shutdown.  The transfer engine owns the actual deregistration logic.
        let engine = Arc::clone(&self.engine);
        py.detach(move || -> PyResult<()> {
            let mut ptrs = Vec::with_capacity(addrs.len());
            for addr in addrs {
                ptrs.push(nonnull_from_u64(addr, "addr")?);
            }
            engine
                .unregister_memory(&ptrs)
                .map_err(|err| rdma_v1_error("unregister_memory failed", err))
        })
    }

    fn prepare_handshake(
        &self,
        py: Python<'_>,
        remote_addr: String,
    ) -> PyResult<PegaRdmaV1Handshake> {
        // Expose TransferEngine's initiator state machine without leaking Rust
        // enums into Python. "prepared" means Python must send this opaque RDMA
        // metadata to its peer; "connecting" means another request is already
        // doing that for the same peer.
        self.prepare_handshake_inner(py, &remote_addr)
    }

    fn accept_handshake<'py>(
        &self,
        py: Python<'py>,
        remote_addr: String,
        remote_metadata: Vec<u8>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Responder-side equivalent of PegaEngine::rdma_accept_handshake. The
        // Python connector only transports opaque bytes; stale connection
        // invalidation, local QP preparation, and completion stay in Rust.
        let local = self.accept_handshake_inner(py, &remote_addr, &remote_metadata)?;
        Ok(pyo3::types::PyBytes::new(py, &local))
    }

    fn finish_handshake(
        &self,
        py: Python<'_>,
        remote_addr: String,
        remote_metadata: Vec<u8>,
    ) -> PyResult<()> {
        // Initiator-side completion. The local metadata was cached by
        // prepare_handshake, so Python does not need to keep or decode it.
        self.finish_handshake_inner(py, &remote_addr, &remote_metadata)
    }

    fn abort_handshake(&self, py: Python<'_>, remote_addr: String) -> PyResult<()> {
        // Abort a prepared-but-incomplete initiator handshake after the Python
        // connector's timeout expires. If there is no prepared local metadata
        // left, the peer is already connected or gone, so abort is a no-op.
        let local = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .pending_handshakes
            .remove(&remote_addr);
        if let Some(local) = local {
            py.detach(|| self.engine.abort_handshake(&remote_addr, &local));
        }
        Ok(())
    }

    fn register_local_blocks(&self, blocks: Vec<(u64, u64, u64)>) -> PyResult<u64> {
        // Blocks are stable after NIXL registers KV cache memory. Rust owns a
        // monotonically increasing table id so Python does not need to expose
        // object addresses such as id(nixl_handle) across the PyO3 boundary.
        let table = Self::build_block_table(blocks)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?;
        let table_id = state.next_local_table_id;
        state.next_local_table_id = state
            .next_local_table_id
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("local block table id overflow"))?;
        state.local_block_tables.insert(table_id, table);
        Ok(table_id)
    }

    fn unregister_local_blocks(&self, table_id: u64) -> PyResult<()> {
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .local_block_tables
            .remove(&table_id);
        Ok(())
    }

    fn register_remote_blocks(
        &self,
        engine_id: String,
        tp_rank: usize,
        blocks: Vec<(u64, u64, u64)>,
    ) -> PyResult<()> {
        let table = Self::build_block_table(blocks)?;
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .remote_block_tables
            .insert((engine_id, tp_rank), table);
        Ok(())
    }

    #[pyo3(signature = (engine_id, tp_rank = None))]
    fn unregister_remote_blocks(&self, engine_id: String, tp_rank: Option<usize>) -> PyResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?;
        if let Some(tp_rank) = tp_rank {
            state.remote_block_tables.remove(&(engine_id, tp_rank));
        } else {
            state
                .remote_block_tables
                .retain(|(table_engine_id, _), _| table_engine_id != &engine_id);
        }
        Ok(())
    }

    fn read_async_indices(
        &self,
        py: Python<'_>,
        remote_addr: String,
        local_table_id: u64,
        remote_engine_id: String,
        remote_tp_rank: usize,
        local_desc_ids: Vec<usize>,
        remote_desc_ids: Vec<usize>,
    ) -> PyResult<(u64, usize, usize)> {
        // Submit one batch READ by looking up NIXL-produced descriptor indices
        // in cached local/remote block tables.  Returns the Python handle plus
        // transfer statistics used by the inherited NIXL metrics path.
        if local_desc_ids.len() != remote_desc_ids.len() {
            return Err(PyValueError::new_err(
                "local_desc_ids and remote_desc_ids must have the same length",
            ));
        }

        let prepared = {
            let state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?;
            let local_table = state
                .local_block_tables
                .get(&local_table_id)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("unknown local block table id {local_table_id}"))
                })?;
            let remote_key = (remote_engine_id.clone(), remote_tp_rank);
            let remote_table = state.remote_block_tables.get(&remote_key).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown remote block table for engine={} rank={}",
                    remote_engine_id, remote_tp_rank
                ))
            })?;

            let mut native = Vec::with_capacity(local_desc_ids.len());
            let mut bytes = 0usize;
            for (local_idx, remote_idx) in local_desc_ids.into_iter().zip(remote_desc_ids) {
                let local = local_table.get(local_idx).ok_or_else(|| {
                    PyIndexError::new_err(format!("local desc index out of range: {local_idx}"))
                })?;
                let remote = remote_table.get(remote_idx).ok_or_else(|| {
                    PyIndexError::new_err(format!("remote desc index out of range: {remote_idx}"))
                })?;
                let len = local.len.min(remote.len);
                if len == 0 {
                    continue;
                }
                // NIXL owns descriptor ordering and index selection; this layer
                // only materializes those indices into native RDMA READ
                // descriptors for the PegaFlow v1 transfer engine.
                native.push((local.addr, remote.addr, len));
                bytes = bytes.saturating_add(len);
            }
            (native, bytes)
        };

        let (native, bytes) = prepared;
        let desc_count = native.len();
        let handle = self.submit_read_native(py, remote_addr, native)?;
        Ok((handle, bytes, desc_count))
    }

    fn check_read(&self, handle: u64) -> PyResult<String> {
        // Poll all per-descriptor completion receivers under a single Python
        // handle.  The connector keeps polling until every descriptor finishes,
        // then sends the regular NIXL completion notification.
        let mut pending = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?;
        let read = pending
            .pending_reads
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
        pending.pending_reads.remove(&handle);
        log::debug!("[PegaRdmaV1Engine] RDMA READ done handle={handle} bytes={total}");
        Ok("done".to_string())
    }

    fn release_read(&self, handle: u64) -> PyResult<()> {
        // Forget a pending READ after request failure or shutdown.  Completed
        // READs are removed by check_read when it returns "done".
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .pending_reads
            .remove(&handle);
        Ok(())
    }

    fn invalidate_connection(&self, remote_addr: String) {
        // Mark the peer connection unusable after a transfer failure so the
        // next request does a fresh transport handshake.
        self.engine.invalidate_connection(&remote_addr);
    }

    fn num_qps(&self) -> usize {
        // Expose the native queue-pair count for diagnostics.
        self.engine.num_qps()
    }
}

impl PegaRdmaV1Engine {
    fn build_block_table(blocks: Vec<(u64, u64, u64)>) -> PyResult<Vec<BlockEntry>> {
        let mut table = Vec::with_capacity(blocks.len());
        for (addr, len, _device_id) in blocks {
            if len == 0 {
                return Err(PyValueError::new_err("block len must be positive"));
            }
            table.push(BlockEntry {
                addr,
                len: u64_to_usize(len, "block len")?,
            });
        }
        Ok(table)
    }

    fn prepare_handshake_inner(
        &self,
        py: Python<'_>,
        remote_addr: &str,
    ) -> PyResult<PegaRdmaV1Handshake> {
        let status = py
            .detach(|| self.engine.get_or_prepare(remote_addr))
            .map_err(|err| rdma_v1_error("prepare_handshake failed", err))?;
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
            ConnectionStatus::Prepared(metadata) => {
                self.state
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
                    .pending_handshakes
                    .insert(remote_addr.to_string(), metadata.clone());
                PegaRdmaV1Handshake {
                    status: "prepared".to_string(),
                    has_metadata: true,
                    metadata: Some(metadata.to_bytes()),
                }
            }
        })
    }

    fn accept_handshake_inner(
        &self,
        py: Python<'_>,
        remote_addr: &str,
        remote_metadata: &[u8],
    ) -> PyResult<Vec<u8>> {
        let remote = HandshakeMetadata::from_bytes(remote_metadata)
            .map_err(|err| rdma_v1_error("decode remote handshake failed", err))?;

        // Match the server-side PegaEngine RDMA handshake behavior: a peer
        // that sends fresh metadata is asking for a fresh connection, so drop
        // stale local state before preparing response QPs.
        py.detach(|| self.engine.invalidate_connection(remote_addr));

        let local = match py
            .detach(|| self.engine.get_or_prepare(remote_addr))
            .map_err(|err| rdma_v1_error("accept_handshake prepare failed", err))?
        {
            ConnectionStatus::Prepared(metadata) => metadata,
            ConnectionStatus::Existing => {
                unreachable!("just invalidated connection for {remote_addr}")
            }
            ConnectionStatus::Connecting => {
                return Err(PegaFlowError::new_err(format!(
                    "handshake to {remote_addr} already in progress"
                )));
            }
        };

        if let Err(err) = py.detach(|| self.engine.complete_handshake(remote_addr, &local, &remote))
        {
            py.detach(|| self.engine.abort_handshake(remote_addr, &local));
            return Err(rdma_v1_error("accept_handshake complete failed", err));
        }
        Ok(local.to_bytes())
    }

    fn finish_handshake_inner(
        &self,
        py: Python<'_>,
        remote_addr: &str,
        remote_metadata: &[u8],
    ) -> PyResult<()> {
        let remote = HandshakeMetadata::from_bytes(remote_metadata)
            .map_err(|err| rdma_v1_error("decode remote handshake failed", err))?;
        let local = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .pending_handshakes
            .get(remote_addr)
            .cloned()
            .ok_or_else(|| {
                PegaFlowError::new_err(format!(
                    "finish_handshake called without prepared local metadata for {remote_addr}"
                ))
            })?;
        if let Err(err) = py.detach(|| self.engine.complete_handshake(remote_addr, &local, &remote))
        {
            py.detach(|| self.engine.abort_handshake(remote_addr, &local));
            return Err(rdma_v1_error("finish_handshake complete failed", err));
        }
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?
            .pending_handshakes
            .remove(remote_addr);
        Ok(())
    }

    fn submit_read_native(
        &self,
        py: Python<'_>,
        remote_addr: String,
        raw_descs: Vec<(u64, u64, usize)>,
    ) -> PyResult<u64> {
        // TransferEngine returns one completion receiver per descriptor.  Python
        // sees a single monotonically increasing handle so the worker can store
        // request-level metadata alongside the native completions.
        let engine = Arc::clone(&self.engine);
        let receivers = py.detach(move || -> PyResult<_> {
            let mut native = Vec::with_capacity(raw_descs.len());
            for (local_addr, remote_addr, len) in raw_descs {
                native.push(TransferDesc {
                    local_ptr: nonnull_from_u64(local_addr, "local_addr")?,
                    remote_ptr: nonnull_from_u64(remote_addr, "remote_addr")?,
                    len,
                });
            }
            engine
                .batch_transfer_async(TransferOp::Read, &remote_addr, &native)
                .map_err(|err| rdma_v1_error("submit RDMA READ failed", err))
        })?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PegaRdmaV1Engine state mutex poisoned"))?;
        let handle = state.next_handle;
        state.next_handle = state
            .next_handle
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("RDMA read handle overflow"))?;
        state.pending_reads.insert(
            handle,
            PendingRead {
                receivers: receivers.into_iter().map(Some).collect(),
                bytes_done: 0,
            },
        );
        Ok(handle)
    }
}

pub(crate) fn add_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PegaRdmaV1Engine>()?;
    m.add_class::<PegaRdmaV1Handshake>()?;
    Ok(())
}
