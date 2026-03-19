mod runtime;
mod session;
mod state;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use log::debug;
use parking_lot::Mutex;
use sideway::ibverbs::AccessFlags;

use self::runtime::RcRuntime;
use self::session::{RcSession, RdmaOp};
use self::state::{RcBackendState, RegisteredMemoryEntry};
use crate::engine::{RcEndpoint, RegisteredMemoryRegion, TransferOp};
use crate::error::{Result, TransferError};

pub(crate) struct RcBackend {
    runtime: Arc<RcRuntime>,
    state: Arc<Mutex<RcBackendState>>,
    psn_counter: AtomicU64,
}

impl RcBackend {
    pub(crate) fn new(nic_name: &str) -> Result<Self> {
        crate::init_logging();
        if nic_name.trim().is_empty() {
            return Err(TransferError::InvalidArgument("nic_name is empty"));
        }
        log::info!("rc backend initialize: nic={}", nic_name);
        let runtime = RcRuntime::open(nic_name)?;
        log::info!("rc backend initialized: nic={}", nic_name);
        Ok(Self {
            runtime,
            state: Arc::new(Mutex::new(RcBackendState::default())),
            psn_counter: AtomicU64::new(1),
        })
    }

    pub(crate) fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }

        let mr = unsafe {
            self.runtime.pd.reg_mr(
                ptr as usize,
                len,
                AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
            )
        }
        .map_err(|error| TransferError::Backend(error.to_string()))?;

        let mut state = self.state.lock();
        state.registered.insert(
            ptr,
            RegisteredMemoryEntry {
                base_ptr: ptr,
                len,
                mr,
            },
        );
        debug!("memory registered: ptr={:#x}, len={}", ptr, len);
        Ok(())
    }

    pub(crate) fn unregister_memory(&self, ptr: u64) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        let mut state = self.state.lock();
        let removed = state.registered.remove(&ptr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr });
        }
        debug!("memory unregistered: ptr={:#x}", ptr);
        Ok(())
    }

    pub(crate) fn snapshot_registered_memory(&self) -> Vec<RegisteredMemoryRegion> {
        let state = self.state.lock();
        let mut regions: Vec<RegisteredMemoryRegion> = state
            .registered
            .values()
            .map(|entry| RegisteredMemoryRegion {
                base_ptr: entry.base_ptr,
                len: entry.len as u64,
                rkey: entry.mr.rkey(),
            })
            .collect();
        regions.sort_unstable_by_key(|entry| entry.base_ptr);
        regions
    }

    /// Create an RC QP in INIT state, push it to the pending queue, and return its endpoint.
    pub(crate) fn prepare_handshake(&self) -> Result<RcEndpoint> {
        let psn_seed = self.psn_counter.fetch_add(1, Ordering::Relaxed);
        let session = RcSession::create(&self.runtime, psn_seed)?;
        let endpoint = session.local_endpoint;
        self.state.lock().pending.push_back(session);
        Ok(endpoint)
    }

    /// Pop the oldest pending QP, connect it to the remote peer (RTR→RTS),
    /// and cache the remote memory regions. Called by the responder.
    pub(crate) fn accept_handshake(
        &self,
        remote: &RcEndpoint,
        remote_memory_regions: &[RegisteredMemoryRegion],
    ) -> Result<()> {
        let session = self
            .state
            .lock()
            .pending
            .pop_front()
            .ok_or(TransferError::Backend(
                "no pending session to accept".to_string(),
            ))?;

        session.connect(&self.runtime, remote)?;

        let mut state = self.state.lock();
        state.cache_remote_memory(remote.qp_num, remote_memory_regions)?;
        state.sessions.insert(remote.qp_num, session);
        Ok(())
    }

    pub(crate) fn batch_transfer(
        &self,
        op: TransferOp,
        remote: &RcEndpoint,
        remote_memory_regions: &[RegisteredMemoryRegion],
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
        if lens.is_empty() {
            return Ok(0);
        }

        let remote_qpn = remote.qp_num;

        // --- Lazy connect + lookup in one lock acquisition (hot path) ---
        let lookup_start = Instant::now();
        let (session, prepared_ops) = {
            let mut state = self.state.lock();

            // Lazy connect: if no session yet, pop a pending QP.
            // Check + pop under one lock to avoid TOCTOU.
            if !state.sessions.contains_key(&remote_qpn) {
                let pending_session = state.pending.pop_front().ok_or(TransferError::Backend(
                    "no pending session for lazy connect".to_string(),
                ))?;
                // Must drop lock before blocking connect().
                drop(state);

                pending_session.connect(&self.runtime, remote)?;

                state = self.state.lock();
                if !state.sessions.contains_key(&remote_qpn) {
                    state.cache_remote_memory(remote_qpn, remote_memory_regions)?;
                    state.sessions.insert(remote_qpn, pending_session);
                }
            }

            let session = Arc::clone(
                state
                    .sessions
                    .get(&remote_qpn)
                    .expect("session must exist after ensure_connected"),
            );

            let mut prepared = Vec::with_capacity(lens.len());
            for ((local_ptr, remote_ptr), len) in local_ptrs
                .iter()
                .copied()
                .zip(remote_ptrs.iter().copied())
                .zip(lens.iter().copied())
            {
                if local_ptr == 0 {
                    return Err(TransferError::InvalidArgument("local_ptr must be non-zero"));
                }
                if remote_ptr == 0 {
                    return Err(TransferError::InvalidArgument(
                        "remote_ptr must be non-zero",
                    ));
                }
                if len == 0 {
                    return Err(TransferError::InvalidArgument("len must be non-zero"));
                }

                let local_mr = state
                    .find_local_mr(local_ptr, len)
                    .ok_or(TransferError::MemoryNotRegistered { ptr: local_ptr })?;

                let remote_rkey = state.find_remote_rkey(remote_qpn, remote_ptr, len).ok_or(
                    TransferError::InvalidArgument("remote memory not found in handshake snapshot"),
                )?;

                prepared.push(RdmaOp {
                    local_mr,
                    local_ptr,
                    remote_ptr,
                    len,
                    remote_rkey,
                });
            }
            (session, prepared)
        };
        let lookup_dur = lookup_start.elapsed();

        // --- Submit outside lock ---
        let submit_start = Instant::now();
        let transferred = session.transfer_batch(prepared_ops, op)?;
        let submit_dur = submit_start.elapsed();

        debug!(
            "batch_transfer_{:?}: remote_qpn={}, bytes={}, chunks={}, lookup_ms={:.3}, submit_ms={:.3}, total_ms={:.3}",
            op,
            remote_qpn,
            transferred,
            lens.len(),
            lookup_dur.as_secs_f64() * 1000.0,
            submit_dur.as_secs_f64() * 1000.0,
            (lookup_dur + submit_dur).as_secs_f64() * 1000.0,
        );
        Ok(transferred)
    }
}
