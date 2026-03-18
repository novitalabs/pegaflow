mod runtime;
mod session;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use log::debug;
use parking_lot::Mutex;
use sideway::ibverbs::AccessFlags;
use sideway::ibverbs::memory_region::MemoryRegion;

use self::runtime::RcRuntime;
use self::session::{RcSession, RdmaOp};
use crate::engine::{RcEndpoint, RegisteredMemoryRegion, TransferOp};
use crate::error::{Result, TransferError};

struct RegisteredMemoryEntry {
    base_ptr: u64,
    len: usize,
    mr: Arc<MemoryRegion>,
}

#[derive(Clone, Copy)]
struct RemoteMemoryEntry {
    base_ptr: u64,
    end_ptr: u64,
    rkey: u32,
}

#[derive(Default)]
struct RcBackendState {
    registered: HashMap<u64, RegisteredMemoryEntry>,
    /// Pre-connect sessions in FIFO order (first prepared, first connected).
    pending: VecDeque<Arc<RcSession>>,
    /// Connected sessions keyed by remote QP number.
    sessions: HashMap<u32, Arc<RcSession>>,
    /// Remote memory cache keyed by remote QP number.
    remote_memory: HashMap<u32, Vec<RemoteMemoryEntry>>,
}

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
        Self::cache_remote_memory(&mut state, remote.qp_num, remote_memory_regions)?;
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
        self.ensure_connected(remote, remote_memory_regions)?;

        let local_lookup_start = Instant::now();
        let (session, mut prepared_ops): (Arc<RcSession>, Vec<RdmaOp>) = {
            let state = self.state.lock();
            let session =
                state
                    .sessions
                    .get(&remote_qpn)
                    .cloned()
                    .ok_or(TransferError::Backend(format!(
                        "no connected session for remote_qpn={}",
                        remote_qpn
                    )))?;

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
                let (_, _, local_mr) = Self::find_registered_entry(&state, local_ptr, len)
                    .ok_or(TransferError::MemoryNotRegistered { ptr: local_ptr })?;
                prepared.push(RdmaOp {
                    local_mr,
                    local_ptr,
                    remote_ptr,
                    len,
                    remote_rkey: 0,
                });
            }
            (session, prepared)
        };
        let local_lookup_dur = local_lookup_start.elapsed();

        let remote_lookup_start = Instant::now();
        {
            let state = self.state.lock();
            for rdma_op in prepared_ops.iter_mut() {
                let (remote_rkey, remote_available) = Self::find_cached_remote_memory(
                    &state,
                    remote_qpn,
                    rdma_op.remote_ptr,
                    rdma_op.len,
                )
                .ok_or(TransferError::InvalidArgument(
                    "remote memory not found in handshake snapshot",
                ))?;
                if rdma_op.len > remote_available {
                    return Err(TransferError::InvalidArgument(
                        "len exceeds remote registered memory",
                    ));
                }
                rdma_op.remote_rkey = remote_rkey;
            }
        }
        let remote_lookup_dur = remote_lookup_start.elapsed();

        let submit_start = Instant::now();
        let transferred = session.submit_batch(prepared_ops, op)?;
        let submit_dur = submit_start.elapsed();

        let total_dur = local_lookup_dur + remote_lookup_dur + submit_dur;
        debug!(
            "batch_transfer_{:?} profile: remote_qpn={}, bytes={}, chunks={}, local_lookup_ms={:.3}, remote_lookup_ms={:.3}, submit_wait_ms={:.3}, total_ms={:.3}",
            op,
            remote_qpn,
            transferred,
            lens.len(),
            local_lookup_dur.as_secs_f64() * 1000.0,
            remote_lookup_dur.as_secs_f64() * 1000.0,
            submit_dur.as_secs_f64() * 1000.0,
            total_dur.as_secs_f64() * 1000.0
        );
        Ok(transferred)
    }

    /// Lazy connect: if no session exists for the remote peer, pop a pending QP and connect it.
    fn ensure_connected(
        &self,
        remote: &RcEndpoint,
        remote_memory_regions: &[RegisteredMemoryRegion],
    ) -> Result<()> {
        if self.state.lock().sessions.contains_key(&remote.qp_num) {
            return Ok(());
        }

        let session = self
            .state
            .lock()
            .pending
            .pop_front()
            .ok_or(TransferError::Backend(
                "no pending session for lazy connect".to_string(),
            ))?;

        session.connect(&self.runtime, remote)?;

        let mut state = self.state.lock();
        Self::cache_remote_memory(&mut state, remote.qp_num, remote_memory_regions)?;
        state.sessions.insert(remote.qp_num, session);
        Ok(())
    }

    fn find_registered_entry(
        state: &RcBackendState,
        ptr: u64,
        len: usize,
    ) -> Option<(u64, usize, Arc<MemoryRegion>)> {
        let end = ptr.checked_add(len as u64)?;
        state.registered.values().find_map(|entry| {
            let entry_end = entry.base_ptr.checked_add(entry.len as u64)?;
            if ptr >= entry.base_ptr && end <= entry_end {
                Some((entry.base_ptr, entry.len, Arc::clone(&entry.mr)))
            } else {
                None
            }
        })
    }

    fn find_cached_remote_memory(
        state: &RcBackendState,
        remote_qpn: u32,
        remote_ptr: u64,
        len: usize,
    ) -> Option<(u32, usize)> {
        let end = remote_ptr.checked_add(len as u64)?;
        let entries = state.remote_memory.get(&remote_qpn)?;
        let index = match entries.binary_search_by_key(&remote_ptr, |entry| entry.base_ptr) {
            Ok(index) => index,
            Err(0) => return None,
            Err(index) => index - 1,
        };
        let entry = &entries[index];
        if remote_ptr >= entry.base_ptr && end <= entry.end_ptr {
            let available = usize::try_from(entry.end_ptr - remote_ptr).ok()?;
            return Some((entry.rkey, available));
        }
        None
    }

    fn cache_remote_memory(
        state: &mut RcBackendState,
        remote_qpn: u32,
        remote_memory_regions: &[RegisteredMemoryRegion],
    ) -> Result<()> {
        let mut cached = Vec::with_capacity(remote_memory_regions.len());
        for entry in remote_memory_regions.iter().copied() {
            if entry.len == 0 {
                return Err(TransferError::Backend(
                    "handshake response contains zero-length memory region".to_string(),
                ));
            }
            let Some(end_ptr) = entry.base_ptr.checked_add(entry.len) else {
                return Err(TransferError::Backend(
                    "handshake response contains memory region overflow".to_string(),
                ));
            };
            cached.push(RemoteMemoryEntry {
                base_ptr: entry.base_ptr,
                end_ptr,
                rkey: entry.rkey,
            });
        }
        cached.sort_unstable_by_key(|entry| entry.base_ptr);
        for pair in cached.windows(2) {
            if pair[1].base_ptr < pair[0].end_ptr {
                return Err(TransferError::Backend(
                    "handshake response contains overlapping memory regions".to_string(),
                ));
            }
        }
        state.remote_memory.insert(remote_qpn, cached);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_remote_memory_rejects_overlapping_regions() {
        let mut state = RcBackendState::default();
        let regions = vec![
            RegisteredMemoryRegion {
                base_ptr: 0x1000,
                len: 0x200,
                rkey: 1,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x1100,
                len: 0x100,
                rkey: 2,
            },
        ];

        let error = RcBackend::cache_remote_memory(&mut state, 1, &regions)
            .expect_err("overlap should fail");
        assert_eq!(
            error,
            TransferError::Backend(
                "handshake response contains overlapping memory regions".to_string()
            )
        );
    }

    #[test]
    fn find_cached_remote_memory_uses_sorted_snapshot() {
        let mut state = RcBackendState::default();
        let remote_qpn = 42;
        let regions = vec![
            RegisteredMemoryRegion {
                base_ptr: 0x3000,
                len: 0x100,
                rkey: 3,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x1000,
                len: 0x100,
                rkey: 1,
            },
            RegisteredMemoryRegion {
                base_ptr: 0x2000,
                len: 0x100,
                rkey: 2,
            },
        ];
        RcBackend::cache_remote_memory(&mut state, remote_qpn, &regions).expect("snapshot cache");

        let hit = RcBackend::find_cached_remote_memory(&state, remote_qpn, 0x2080, 0x10);
        assert_eq!(hit, Some((2, 0x80)));

        let miss = RcBackend::find_cached_remote_memory(&state, remote_qpn, 0x2500, 0x10);
        assert!(miss.is_none());
    }
}
