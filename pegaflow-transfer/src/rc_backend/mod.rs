mod runtime;
mod session;
mod state;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use log::debug;
use parking_lot::Mutex;
use sideway::ibverbs::AccessFlags;

use self::runtime::RcRuntime;
use self::session::{RcSession, RdmaOp};
use self::state::{RcBackendState, RegisteredMemoryEntry};
use std::ptr::NonNull;

use crate::engine::{NicHandshake, RegisteredMemoryRegion, TransferDesc, TransferOp};
use crate::error::{Result, TransferError};

pub(crate) struct RcBackend {
    runtimes: Vec<Arc<RcRuntime>>,
    state: Arc<Mutex<RcBackendState>>,
    psn_counter: AtomicU64,
    rr_counter: AtomicUsize,
}

impl RcBackend {
    pub(crate) fn new(nic_names: &[String]) -> Result<Self> {
        crate::init_logging();
        if nic_names.is_empty() {
            return Err(TransferError::InvalidArgument("nic_names is empty"));
        }
        let mut runtimes = Vec::with_capacity(nic_names.len());
        for name in nic_names {
            if name.trim().is_empty() {
                return Err(TransferError::InvalidArgument("nic_name is empty"));
            }
            log::info!("rc backend initialize: nic={}", name);
            let runtime = RcRuntime::open(name)?;
            log::info!("rc backend initialized: nic={}", name);
            runtimes.push(runtime);
        }
        let nic_count = runtimes.len();
        Ok(Self {
            runtimes,
            state: Arc::new(Mutex::new(RcBackendState::new(nic_count))),
            psn_counter: AtomicU64::new(1),
            rr_counter: AtomicUsize::new(0),
        })
    }

    fn nic_count(&self) -> usize {
        self.runtimes.len()
    }

    pub(crate) fn register_memory(&self, ptr: NonNull<u8>, len: usize) -> Result<()> {
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }
        let raw = ptr.as_ptr() as u64;

        // Register on every NIC's PD.
        let mut mrs = Vec::with_capacity(self.nic_count());
        for runtime in &self.runtimes {
            let mr = unsafe {
                runtime.pd.reg_mr(
                    raw as usize,
                    len,
                    AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
                )
            }
            .map_err(|error| TransferError::Backend(error.to_string()))?;
            mrs.push(mr);
        }

        let mut state = self.state.lock();
        state.registered.insert(
            raw,
            RegisteredMemoryEntry {
                base_ptr: raw,
                len,
                mrs,
            },
        );
        debug!(
            "memory registered: ptr={:#x}, len={}, nics={}",
            raw,
            len,
            self.nic_count()
        );
        Ok(())
    }

    pub(crate) fn unregister_memory(&self, ptr: NonNull<u8>) -> Result<()> {
        let raw = ptr.as_ptr() as u64;
        let mut state = self.state.lock();
        let removed = state.registered.remove(&raw);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr: raw });
        }
        debug!("memory unregistered: ptr={:#x}", raw);
        Ok(())
    }

    /// Snapshot registered memory for each NIC. Outer vec indexed by nic_idx.
    fn snapshot_registered_memory(&self) -> Vec<Vec<RegisteredMemoryRegion>> {
        let state = self.state.lock();
        (0..self.nic_count())
            .map(|nic_idx| {
                let mut regions: Vec<RegisteredMemoryRegion> = state
                    .registered
                    .values()
                    .map(|entry| RegisteredMemoryRegion {
                        base_ptr: entry.base_ptr,
                        len: entry.len as u64,
                        rkey: entry.mrs[nic_idx].rkey(),
                    })
                    .collect();
                regions.sort_unstable_by_key(|entry| entry.base_ptr);
                regions
            })
            .collect()
    }

    /// Create one RC QP per NIC in INIT state, push to per-NIC pending queues,
    /// return per-NIC handshake data.
    pub(crate) fn prepare_handshake(&self) -> Result<Vec<NicHandshake>> {
        let snapshots = self.snapshot_registered_memory();
        let mut nic_handshakes = Vec::with_capacity(self.nic_count());

        // Create all QPs before locking state.
        let mut sessions = Vec::with_capacity(self.nic_count());
        for runtime in &self.runtimes {
            let psn_seed = self.psn_counter.fetch_add(1, Ordering::Relaxed);
            let session = RcSession::create(runtime, psn_seed)?;
            sessions.push(session);
        }

        let mut state = self.state.lock();
        for (nic_idx, session) in sessions.into_iter().enumerate() {
            let endpoint = session.local_endpoint;
            state.nics[nic_idx].pending.push_back(session);
            nic_handshakes.push(NicHandshake {
                endpoint,
                memory_regions: snapshots[nic_idx].clone(),
            });
        }

        Ok(nic_handshakes)
    }

    /// Connect all pending QPs to the remote peer (responder path).
    pub(crate) fn accept_handshake(&self, remote_nics: &[NicHandshake]) -> Result<()> {
        let nic_count = self.nic_count();
        if remote_nics.len() != nic_count {
            return Err(TransferError::InvalidArgument(
                "remote NIC count mismatch in handshake",
            ));
        }

        // Pop all pending sessions under one lock.
        let pending_sessions: Vec<Arc<RcSession>> = {
            let mut state = self.state.lock();
            let mut sessions = Vec::with_capacity(nic_count);
            for nic_idx in 0..nic_count {
                let session =
                    state.nics[nic_idx]
                        .pending
                        .pop_front()
                        .ok_or(TransferError::Backend(
                            "no pending session to accept".to_string(),
                        ))?;
                sessions.push(session);
            }
            sessions
        };

        // Connect all outside lock.
        for (nic_idx, session) in pending_sessions.iter().enumerate() {
            session.connect(&self.runtimes[nic_idx], &remote_nics[nic_idx].endpoint)?;
        }

        // Re-lock and insert all sessions + cache remote memory.
        let mut state = self.state.lock();
        for (nic_idx, session) in pending_sessions.into_iter().enumerate() {
            let remote = &remote_nics[nic_idx];
            let remote_qpn = remote.endpoint.qp_num;
            state.nics[nic_idx].cache_remote_memory(remote_qpn, &remote.memory_regions)?;
            state.nics[nic_idx].sessions.insert(remote_qpn, session);
        }
        Ok(())
    }

    pub(crate) fn batch_transfer_async(
        &self,
        op: TransferOp,
        remote_nics: &[NicHandshake],
        descs: &[TransferDesc],
    ) -> Result<std::sync::mpsc::Receiver<Result<usize>>> {
        let nic_count = self.nic_count();
        if remote_nics.len() != nic_count {
            return Err(TransferError::InvalidArgument(
                "remote NIC count mismatch in transfer",
            ));
        }

        if descs.is_empty() {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let _ = tx.send(Ok(0));
            return Ok(rx);
        }

        // Round-robin assignment.
        let rr_base = self.rr_counter.fetch_add(descs.len(), Ordering::Relaxed);

        // Group ops by NIC index.
        let mut per_nic: Vec<Vec<TransferDesc>> = (0..nic_count).map(|_| Vec::new()).collect();
        for (i, &desc) in descs.iter().enumerate() {
            per_nic[(rr_base + i) % nic_count].push(desc);
        }

        // --- Lock: lazy connect + prepare ops ---
        let lookup_start = Instant::now();
        let mut nic_work: Vec<(Arc<RcSession>, Vec<RdmaOp>)> = Vec::new();
        {
            let mut state = self.state.lock();

            // Lazy connect: if NIC 0 has no session for the remote peer, connect all NICs.
            let nic0_remote_qpn = remote_nics[0].endpoint.qp_num;
            if !state.nics[0].sessions.contains_key(&nic0_remote_qpn) {
                // Pop pending from all NICs.
                let mut pending_sessions = Vec::with_capacity(nic_count);
                for nic_idx in 0..nic_count {
                    let session =
                        state.nics[nic_idx]
                            .pending
                            .pop_front()
                            .ok_or(TransferError::Backend(
                                "no pending session for lazy connect".to_string(),
                            ))?;
                    pending_sessions.push(session);
                }
                drop(state);

                // Connect all outside lock.
                for (nic_idx, session) in pending_sessions.iter().enumerate() {
                    session.connect(&self.runtimes[nic_idx], &remote_nics[nic_idx].endpoint)?;
                }

                state = self.state.lock();
                for (nic_idx, session) in pending_sessions.into_iter().enumerate() {
                    let remote_qpn = remote_nics[nic_idx].endpoint.qp_num;
                    if !state.nics[nic_idx].sessions.contains_key(&remote_qpn) {
                        state.nics[nic_idx].cache_remote_memory(
                            remote_qpn,
                            &remote_nics[nic_idx].memory_regions,
                        )?;
                        state.nics[nic_idx].sessions.insert(remote_qpn, session);
                    }
                }
            }

            // Prepare ops for each NIC that has work.
            for nic_idx in 0..nic_count {
                let nic_descs = &per_nic[nic_idx];
                if nic_descs.is_empty() {
                    continue;
                }

                let remote_qpn = remote_nics[nic_idx].endpoint.qp_num;
                let session = Arc::clone(
                    state.nics[nic_idx]
                        .sessions
                        .get(&remote_qpn)
                        .expect("session must exist after lazy connect"),
                );

                let mut prepared = Vec::with_capacity(nic_descs.len());
                for desc in nic_descs {
                    let local_ptr = desc.local_ptr.as_ptr() as u64;
                    let remote_ptr = desc.remote_ptr.as_ptr() as u64;
                    let len = desc.len;

                    if len == 0 {
                        return Err(TransferError::InvalidArgument("len must be non-zero"));
                    }

                    let local_mr = state
                        .find_local_mr(nic_idx, local_ptr, len)
                        .ok_or(TransferError::MemoryNotRegistered { ptr: local_ptr })?;

                    let remote_rkey = state.nics[nic_idx]
                        .find_remote_rkey(remote_qpn, remote_ptr, len)
                        .ok_or(TransferError::InvalidArgument(
                            "remote memory not found in handshake snapshot",
                        ))?;

                    prepared.push(RdmaOp {
                        local_mr,
                        local_ptr,
                        remote_ptr,
                        len,
                        remote_rkey,
                    });
                }
                nic_work.push((session, prepared));
            }
        }
        let lookup_dur = lookup_start.elapsed();

        debug!(
            "batch_transfer_async_{:?}: nics_active={}/{}, chunks={}, lookup_ms={:.3}",
            op,
            nic_work.len(),
            nic_count,
            descs.len(),
            lookup_dur.as_secs_f64() * 1000.0,
        );

        // --- Submit outside lock ---
        if nic_work.len() == 1 {
            // Single NIC path: return its receiver directly.
            let (session, prepared) = nic_work.into_iter().next().unwrap();
            return session.transfer_batch_async(prepared, op);
        }

        // Multi-NIC path: submit to all NICs and aggregate results.
        let mut receivers = Vec::with_capacity(nic_work.len());
        for (session, prepared) in nic_work {
            let rx = session.transfer_batch_async(prepared, op)?;
            receivers.push(rx);
        }

        let (agg_tx, agg_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("pegaflow-transfer-agg".to_string())
            .spawn(move || {
                let mut total_bytes = 0usize;
                for rx in receivers {
                    match rx.recv() {
                        Ok(Ok(bytes)) => total_bytes += bytes,
                        Ok(Err(e)) => {
                            let _ = agg_tx.send(Err(e));
                            return;
                        }
                        Err(_) => {
                            let _ = agg_tx.send(Err(TransferError::Backend(
                                "session worker channel dropped during aggregation".to_string(),
                            )));
                            return;
                        }
                    }
                }
                let _ = agg_tx.send(Ok(total_bytes));
            })
            .map_err(|e| TransferError::Backend(format!("failed to spawn aggregator: {e}")))?;

        Ok(agg_rx)
    }
}
