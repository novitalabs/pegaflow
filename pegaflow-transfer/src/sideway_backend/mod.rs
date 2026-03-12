use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{Arc, atomic::AtomicU64, mpsc as std_mpsc},
    time::{Duration, Instant},
};

use log::{debug, info};
use parking_lot::{Condvar, Mutex};
use rdma_mummy_sys::{ibv_ah, ibv_destroy_ah};
use sideway::ibverbs::{
    AccessFlags,
    address::Gid,
    completion::GenericCompletionQueue,
    device_context::{DeviceContext, LinkLayer, Mtu},
    memory_region::MemoryRegion,
    protection_domain::ProtectionDomain,
    queue_pair::{GenericQueuePair, WorkRequestOperationType},
};

use crate::{
    api::WorkerConfig,
    control_protocol::{RcEndpoint, RegisteredMemoryRegion},
    domain_address::DomainAddress,
    error::{Result, TransferError},
    logging,
};

mod control;
mod qp;
mod rdma_op;
mod runtime;
mod session;

const UD_QKEY: u32 = 0x1111_1111;
const UD_RECV_SLOTS: usize = 64;
const UD_BUFFER_BYTES: usize = 4096;
const UD_GRH_BYTES: usize = 40;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_INFLIGHT_OPS: usize = 96;

#[derive(Default)]
struct ControlPlane {
    next_request_id: AtomicU64,
    pending_replies: Mutex<HashMap<u64, Option<crate::control_protocol::ControlMessage>>>,
    reply_cv: Condvar,
}

struct AddressHandle {
    ah: NonNull<ibv_ah>,
}

impl Drop for AddressHandle {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_ah(self.ah.as_ptr());
        }
    }
}

unsafe impl Send for AddressHandle {}
unsafe impl Sync for AddressHandle {}

struct UdRecvSlot {
    bytes: Box<[u8]>,
    mr: Arc<MemoryRegion>,
}

struct UdSendSlot {
    bytes: Box<[u8]>,
    mr: Arc<MemoryRegion>,
    next_wr_id: u64,
}

struct SidewayRuntime {
    _device_ctx: Arc<DeviceContext>,
    pd: Arc<ProtectionDomain>,
    port_num: u8,
    gid_index: u8,
    link_layer: LinkLayer,
    mtu: Mtu,
    local_gid: Gid,
    local_lid: u16,
    local_ud: DomainAddress,
    max_rd_atomic: u8,
    ud_qp: Arc<Mutex<GenericQueuePair>>,
    ud_cq: GenericCompletionQueue,
    recv_slots: Vec<UdRecvSlot>,
    send_slot: Mutex<UdSendSlot>,
    ah_cache: Mutex<HashMap<DomainAddress, Arc<AddressHandle>>>,
    control: Arc<ControlPlane>,
}

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

struct ActiveSession {
    qp: Mutex<GenericQueuePair>,
    send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
    local_rc: RcEndpoint,
    cmd_tx: std_mpsc::Sender<SessionCommand>,
}

struct RdmaOp {
    local_mr: Arc<MemoryRegion>,
    local_ptr: u64,
    remote_ptr: u64,
    len: usize,
    remote_rkey: u32,
}

enum SessionCommand {
    SubmitBatch {
        ops: Vec<RdmaOp>,
        opcode: WorkRequestOperationType,
        done_tx: std_mpsc::Sender<Result<usize>>,
    },
}

#[derive(Default)]
struct SidewayState {
    config: Option<WorkerConfig>,
    runtime: Option<Arc<SidewayRuntime>>,
    registered: HashMap<u64, RegisteredMemoryEntry>,
    sessions: HashMap<DomainAddress, Arc<ActiveSession>>,
    remote_memory_cache: HashMap<DomainAddress, Vec<RemoteMemoryEntry>>,
}

#[derive(Default)]
pub(crate) struct SidewayBackend {
    state: Arc<Mutex<SidewayState>>,
}

// ---------------------------------------------------------------------------
// Memory helpers (shared across sub-modules)
// ---------------------------------------------------------------------------

impl SidewayBackend {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn ensure_initialized(state: &SidewayState) -> Result<&WorkerConfig> {
        state.config.as_ref().ok_or(TransferError::NotInitialized)
    }

    fn find_registered_entry(
        state: &SidewayState,
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
        state: &SidewayState,
        peer_ud: &DomainAddress,
        remote_ptr: u64,
        len: usize,
    ) -> Option<(u32, usize)> {
        let end = remote_ptr.checked_add(len as u64)?;
        let entries = state.remote_memory_cache.get(peer_ud)?;
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

    fn snapshot_registered_memory(state: &SidewayState) -> Vec<RegisteredMemoryRegion> {
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

    fn cache_remote_memory_snapshot(
        state: &mut SidewayState,
        peer_ud: &DomainAddress,
        remote_memory_regions: &[RegisteredMemoryRegion],
    ) -> Result<()> {
        let mut cached = Vec::with_capacity(remote_memory_regions.len());
        for entry in remote_memory_regions.iter().copied() {
            if entry.len == 0 {
                return Err(TransferError::Backend(
                    "connect response contains zero-length memory region".to_string(),
                ));
            }
            let Some(end_ptr) = entry.base_ptr.checked_add(entry.len) else {
                return Err(TransferError::Backend(
                    "connect response contains memory region overflow".to_string(),
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
                    "connect response contains overlapping memory regions".to_string(),
                ));
            }
        }
        state.remote_memory_cache.insert(peer_ud.clone(), cached);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl SidewayBackend {
    pub(crate) fn initialize(&self, config: WorkerConfig) -> Result<()> {
        logging::ensure_initialized();
        if config.nic_name.trim().is_empty() {
            return Err(TransferError::InvalidArgument("nic_name is empty"));
        }
        if config.rpc_port == 0 {
            return Err(TransferError::InvalidArgument("rpc_port must be non-zero"));
        }
        info!(
            "transfer backend initialize: nic={}, rpc_port={}",
            config.nic_name, config.rpc_port
        );

        let runtime = Self::create_runtime(&config, Arc::clone(&self.state))?;
        let mut state = self.state.lock();
        state.config = Some(config);
        state.runtime = Some(Arc::clone(&runtime));
        state.registered.clear();
        state.sessions.clear();
        state.remote_memory_cache.clear();
        info!(
            "transfer backend initialized: session_id={}",
            runtime.local_ud
        );
        Ok(())
    }

    pub(crate) fn rpc_port(&self) -> Result<u16> {
        let state = self.state.lock();
        Ok(Self::ensure_initialized(&state)?.rpc_port)
    }

    pub(crate) fn session_id(&self) -> DomainAddress {
        let state = self.state.lock();
        let runtime = state
            .runtime
            .as_ref()
            .expect("transfer backend not initialized");
        runtime.local_ud.clone()
    }

    pub(crate) fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }

        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let runtime = state
            .runtime
            .as_ref()
            .ok_or(TransferError::NotInitialized)?;
        let mr = unsafe {
            runtime.pd.reg_mr(
                ptr as usize,
                len,
                AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
            )
        }
        .map_err(|error| TransferError::Backend(error.to_string()))?;

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
        Self::ensure_initialized(&state)?;
        let removed = state.registered.remove(&ptr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr });
        }
        debug!("memory unregistered: ptr={:#x}", ptr);
        Ok(())
    }

    pub(crate) fn transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        self.batch_transfer_sync_write(session_id, &[local_ptr], &[remote_ptr], &[len])
    }

    pub(crate) fn transfer_sync_read(
        &self,
        session_id: &DomainAddress,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        self.batch_transfer_sync_read(session_id, &[local_ptr], &[remote_ptr], &[len])
    }

    pub(crate) fn batch_transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
        self.batch_transfer_sync(
            WorkRequestOperationType::Write,
            session_id,
            local_ptrs,
            remote_ptrs,
            lens,
        )
    }

    pub(crate) fn batch_transfer_sync_read(
        &self,
        session_id: &DomainAddress,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
        self.batch_transfer_sync(
            WorkRequestOperationType::Read,
            session_id,
            local_ptrs,
            remote_ptrs,
            lens,
        )
    }

    fn batch_transfer_sync(
        &self,
        opcode: WorkRequestOperationType,
        session_id: &DomainAddress,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
        if session_id.to_bytes().len() != DomainAddress::BYTES {
            return Err(TransferError::InvalidArgument(
                "session_id has invalid DomainAddress bytes",
            ));
        }
        if local_ptrs.len() != remote_ptrs.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: remote_ptrs.len(),
            });
        }
        if local_ptrs.len() != lens.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: lens.len(),
            });
        }
        if lens.is_empty() {
            return Ok(0);
        }

        let op_name = match opcode {
            WorkRequestOperationType::Write => "write",
            WorkRequestOperationType::Read => "read",
            _ => unreachable!("only Write and Read opcodes are used for RDMA transfers"),
        };

        let peer_ud = session_id;
        let local_lookup_start = Instant::now();
        let (runtime, mut prepared_ops): (Arc<SidewayRuntime>, Vec<RdmaOp>) = {
            let state = self.state.lock();
            Self::ensure_initialized(&state)?;
            let runtime = state
                .runtime
                .as_ref()
                .ok_or(TransferError::NotInitialized)?;

            let mut prepared_ops = Vec::with_capacity(lens.len());
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
                let Some((_, _, local_mr)) = Self::find_registered_entry(&state, local_ptr, len)
                else {
                    return Err(TransferError::MemoryNotRegistered { ptr: local_ptr });
                };
                prepared_ops.push(RdmaOp {
                    local_mr,
                    local_ptr,
                    remote_ptr,
                    len,
                    remote_rkey: 0,
                });
            }
            (Arc::clone(runtime), prepared_ops)
        };
        let local_lookup_dur = local_lookup_start.elapsed();

        let ensure_session_start = Instant::now();
        let session = Self::ensure_active_session(&runtime, &self.state, peer_ud)?;
        let ensure_session_dur = ensure_session_start.elapsed();

        let remote_lookup_start = Instant::now();
        let mut remote_cache_hits = 0usize;
        {
            let state = self.state.lock();
            for op in prepared_ops.iter_mut() {
                let Some((remote_rkey, remote_available)) =
                    Self::find_cached_remote_memory(&state, peer_ud, op.remote_ptr, op.len)
                else {
                    return Err(TransferError::InvalidArgument(
                        "remote memory not found in connect snapshot",
                    ));
                };
                remote_cache_hits += 1;
                if op.len > remote_available {
                    return Err(TransferError::InvalidArgument(
                        "len exceeds remote registered memory",
                    ));
                }
                op.remote_rkey = remote_rkey;
            }
        }
        let remote_lookup_dur = remote_lookup_start.elapsed();
        let submit_start = Instant::now();
        let transferred = Self::submit_session_batch(&session, prepared_ops, opcode)?;
        let submit_dur = submit_start.elapsed();

        let total_dur = local_lookup_dur + ensure_session_dur + remote_lookup_dur + submit_dur;
        debug!(
            "batch_transfer_sync_{} profile: peer={}, bytes={}, chunks={}, local_lookup_ms={:.3}, ensure_session_ms={:.3}, remote_lookup_ms={:.3}, remote_cache_hits={}, submit_wait_ms={:.3}, total_ms={:.3}",
            op_name,
            peer_ud,
            transferred,
            lens.len(),
            local_lookup_dur.as_secs_f64() * 1000.0,
            ensure_session_dur.as_secs_f64() * 1000.0,
            remote_lookup_dur.as_secs_f64() * 1000.0,
            remote_cache_hits,
            submit_dur.as_secs_f64() * 1000.0,
            total_dur.as_secs_f64() * 1000.0
        );
        Ok(transferred)
    }
}

#[cfg(test)]
mod tests {
    use super::{SidewayBackend, SidewayState};
    use crate::{
        api::WorkerConfig, control_protocol::RegisteredMemoryRegion, domain_address::DomainAddress,
        error::TransferError,
    };

    fn sample_addr(seed: u8) -> DomainAddress {
        DomainAddress::from_parts([seed; 16], seed as u16, seed as u32, 0x1111_1111)
    }

    #[test]
    fn initialize_rejects_invalid_input() {
        let backend = SidewayBackend::new();
        let error = backend
            .initialize(WorkerConfig {
                nic_name: "".to_string(),
                rpc_port: 50055,
            })
            .expect_err("must fail");
        assert_eq!(error, TransferError::InvalidArgument("nic_name is empty"));
    }

    #[test]
    fn cache_remote_snapshot_rejects_overlapping_regions() {
        let mut state = SidewayState::default();
        let peer = sample_addr(1);
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

        let error = SidewayBackend::cache_remote_memory_snapshot(&mut state, &peer, &regions)
            .expect_err("overlap should fail");
        assert_eq!(
            error,
            TransferError::Backend(
                "connect response contains overlapping memory regions".to_string()
            )
        );
    }

    #[test]
    fn find_cached_remote_memory_uses_sorted_snapshot() {
        let mut state = SidewayState::default();
        let peer = sample_addr(2);
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
        SidewayBackend::cache_remote_memory_snapshot(&mut state, &peer, &regions)
            .expect("snapshot cache");

        let hit = SidewayBackend::find_cached_remote_memory(&state, &peer, 0x2080, 0x10);
        assert_eq!(hit, Some((2, 0x80)));

        let miss = SidewayBackend::find_cached_remote_memory(&state, &peer, 0x2500, 0x10);
        assert!(miss.is_none());
    }
}
