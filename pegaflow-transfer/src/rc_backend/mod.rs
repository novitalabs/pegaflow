mod runtime;
mod session;
mod state;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use log::{debug, info, warn};
use parking_lot::Mutex;
use sideway::ibverbs::AccessFlags;

use self::runtime::RcRuntime;
use self::session::{RcSession, RdmaOp};
use self::state::{AddrConnection, RcBackendState, RegisteredMemoryEntry};
use std::ptr::NonNull;

use mea::oneshot;

use crate::engine::{NicHandshake, RegisteredMemoryRegion, TransferDesc, TransferOp};
use crate::error::{Result, TransferError};
use pegaflow_common::NumaNode;

struct NicGroup {
    nic_indices: Vec<usize>,
    rr_counter: AtomicUsize,
}

impl NicGroup {
    fn next(&self) -> usize {
        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed);
        self.nic_indices[idx % self.nic_indices.len()]
    }
}

struct NumaRoundRobin {
    /// NUMA node → NIC group on that node.
    groups: HashMap<NumaNode, NicGroup>,
    /// Fallback for unknown NUMA or unmatched nodes (all NICs).
    fallback: NicGroup,
    /// True when all NICs are on a single NUMA node (skip move_pages).
    single_numa: bool,
}

impl NumaRoundRobin {
    fn from_runtimes(runtimes: &[Arc<RcRuntime>]) -> Self {
        let all_indices: Vec<usize> = (0..runtimes.len()).collect();

        // Group NIC indices by NUMA node.
        let mut map: HashMap<NumaNode, Vec<usize>> = HashMap::new();
        for (i, rt) in runtimes.iter().enumerate() {
            if rt.numa_node.is_valid() {
                map.entry(rt.numa_node).or_default().push(i);
            }
        }

        let single_numa = map.len() <= 1;

        let groups: HashMap<NumaNode, NicGroup> = map
            .into_iter()
            .map(|(node, indices)| {
                (
                    node,
                    NicGroup {
                        nic_indices: indices,
                        rr_counter: AtomicUsize::new(0),
                    },
                )
            })
            .collect();

        let fallback = NicGroup {
            nic_indices: all_indices,
            rr_counter: AtomicUsize::new(0),
        };

        Self {
            groups,
            fallback,
            single_numa,
        }
    }

    fn pick(&self, numa: NumaNode) -> usize {
        if numa.is_valid()
            && let Some(group) = self.groups.get(&numa)
        {
            return group.next();
        }
        self.fallback.next()
    }
}

pub(crate) enum GetOrPrepareResult {
    Existing,
    AlreadyConnecting,
    NeedHandshake(Vec<NicHandshake>),
}

pub(crate) struct RcBackend {
    runtimes: Vec<Arc<RcRuntime>>,
    state: Arc<Mutex<RcBackendState>>,
    psn_counter: AtomicU64,
    numa_rr: NumaRoundRobin,
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
            let runtime = RcRuntime::open(name)?;
            runtimes.push(runtime);
        }
        let nic_count = runtimes.len();
        let numa_rr = NumaRoundRobin::from_runtimes(&runtimes);
        Ok(Self {
            runtimes,
            state: Arc::new(Mutex::new(RcBackendState::new(nic_count))),
            psn_counter: AtomicU64::new(1),
            numa_rr,
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
    fn prepare_handshake(&self) -> Result<Vec<NicHandshake>> {
        let snapshots = self.snapshot_registered_memory();
        let mut nic_handshakes = Vec::with_capacity(self.nic_count());

        // Create all QPs before locking state.
        let mut sessions = Vec::with_capacity(self.nic_count());
        for runtime in &self.runtimes {
            let psn_seed = self.psn_counter.fetch_add(1, Ordering::Relaxed);
            let session = RcSession::create(runtime, psn_seed).map_err(|e| {
                TransferError::Backend(format!(
                    "QP creation failed on nic={}: {e}",
                    runtime.nic_name
                ))
            })?;
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

    /// Check if connected to remote_addr; if not, prepare local QPs.
    pub(crate) fn get_or_prepare(&self, remote_addr: &str) -> Result<GetOrPrepareResult> {
        {
            let mut state = self.state.lock();
            if state.addr_connections.contains_key(remote_addr) {
                return Ok(GetOrPrepareResult::Existing);
            }
            if state.connecting.contains(remote_addr) {
                return Ok(GetOrPrepareResult::AlreadyConnecting);
            }
            state.connecting.insert(remote_addr.to_string());
        }
        match self.prepare_handshake() {
            Ok(nics) => Ok(GetOrPrepareResult::NeedHandshake(nics)),
            Err(e) => {
                let removed = self.state.lock().connecting.remove(remote_addr);
                debug_assert!(removed, "connecting set should contain {remote_addr}");
                Err(e)
            }
        }
    }

    /// Complete a connection after handshake exchange.
    pub(crate) fn complete_handshake_for(
        &self,
        remote_addr: &str,
        local_nics: Vec<NicHandshake>,
        remote_nics: &[NicHandshake],
    ) -> Result<()> {
        let nic_count = self.nic_count();
        if remote_nics.len() != nic_count {
            return Err(TransferError::InvalidArgument(
                "remote NIC count mismatch in handshake",
            ));
        }

        // Pop pending sessions by matching QPN from local_nics (not blind FIFO).
        // Concurrent prepare/complete for different remote addrs could reorder the
        // queue, so we must find our own sessions by QPN.
        let pending: Vec<Arc<RcSession>> = {
            let mut state = self.state.lock();
            // If already connected (concurrent request beat us), remove our pending sessions
            if state.addr_connections.contains_key(remote_addr) {
                for (nic_idx, nic) in local_nics.iter().enumerate() {
                    state.nics[nic_idx].remove_pending_by_qpn(nic.endpoint.qp_num);
                }
                state.connecting.remove(remote_addr);
                info!("handshake won by concurrent path: remote={remote_addr}");
                return Ok(());
            }
            let mut sessions = Vec::with_capacity(nic_count);
            for (nic_idx, nic) in local_nics.iter().enumerate() {
                let session = state.nics[nic_idx]
                    .remove_pending_by_qpn(nic.endpoint.qp_num)
                    .ok_or(TransferError::Backend(
                        "no pending session to complete".to_string(),
                    ))?;
                sessions.push(session);
            }
            sessions
        };

        // Connect outside lock
        for (nic_idx, session) in pending.iter().enumerate() {
            session.connect(&self.runtimes[nic_idx], &remote_nics[nic_idx].endpoint)?;
        }

        // Store sessions + addr_connections
        let mut state = self.state.lock();
        let mut remote_qpns = Vec::with_capacity(nic_count);
        for (nic_idx, session) in pending.into_iter().enumerate() {
            let remote = &remote_nics[nic_idx];
            let remote_qpn = remote.endpoint.qp_num;
            state.nics[nic_idx].cache_remote_memory(remote_qpn, &remote.memory_regions)?;
            state.nics[nic_idx].sessions.insert(remote_qpn, session);
            remote_qpns.push(remote_qpn);
        }
        let removed = state.connecting.remove(remote_addr);
        debug_assert!(removed, "connecting set should contain {remote_addr}");
        let local_qpns: Vec<u32> = local_nics.iter().map(|n| n.endpoint.qp_num).collect();
        info!(
            "RDMA connection established: remote={remote_addr}, local_qpns={local_qpns:?}, remote_qpns={remote_qpns:?}"
        );
        state.addr_connections.insert(
            remote_addr.to_string(),
            AddrConnection {
                remote_qpns,
                local_nics,
            },
        );
        Ok(())
    }

    /// Drop pending sessions created by get_or_prepare when handshake failed.
    pub(crate) fn abort_handshake(&self, remote_addr: &str, local_nics: &[NicHandshake]) {
        let mut state = self.state.lock();
        let removed = state.connecting.remove(remote_addr);
        debug_assert!(removed, "connecting set should contain {remote_addr}");
        for (nic_idx, nic) in local_nics.iter().enumerate() {
            state.nics[nic_idx].remove_pending_by_qpn(nic.endpoint.qp_num);
        }
        warn!("handshake aborted: remote={remote_addr}");
    }

    /// Get local NicHandshake metadata for an established connection.
    pub(crate) fn local_meta_for_addr(&self, remote_addr: &str) -> Option<Vec<NicHandshake>> {
        let state = self.state.lock();
        state
            .addr_connections
            .get(remote_addr)
            .map(|c| c.local_nics.clone())
    }

    /// Remove connection state on transfer failure.
    pub(crate) fn invalidate_connection(&self, remote_addr: &str) {
        let mut state = self.state.lock();
        if let Some(conn) = state.addr_connections.remove(remote_addr) {
            for (nic_idx, &remote_qpn) in conn.remote_qpns.iter().enumerate() {
                state.nics[nic_idx].cleanup_connection(remote_qpn);
            }
            info!("connection invalidated: remote={remote_addr}");
        }
    }

    pub(crate) fn num_qps(&self) -> usize {
        self.state.lock().num_qps()
    }

    /// One receiver per NIC that had work; each yields bytes transferred on that NIC.
    pub(crate) fn batch_transfer_async(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc],
    ) -> Result<Vec<oneshot::Receiver<Result<usize>>>> {
        if descs.is_empty() {
            return Ok(Vec::new());
        }

        let nic_count = self.nic_count();

        // NUMA-aware NIC assignment: query the NUMA node of each descriptor's
        // first page and route to a NIC on the same NUMA node.
        let mut per_nic: Vec<Vec<TransferDesc>> = (0..nic_count).map(|_| Vec::new()).collect();
        if self.numa_rr.single_numa {
            // All NICs on one NUMA node — skip move_pages, plain round-robin.
            for &desc in descs {
                let nic_idx = self.numa_rr.fallback.next();
                per_nic[nic_idx].push(desc);
            }
        } else {
            let addrs: Vec<*const u8> = descs
                .iter()
                .map(|d| d.local_ptr.as_ptr() as *const u8)
                .collect();
            let numa_nodes = pegaflow_common::query_pages_numa(&addrs);
            for (i, &desc) in descs.iter().enumerate() {
                let nic_idx = self.numa_rr.pick(numa_nodes[i]);
                per_nic[nic_idx].push(desc);
            }
        }

        // --- Lock: look up connection + prepare ops ---
        let lookup_start = Instant::now();
        let mut nic_work: Vec<(usize, Arc<RcSession>, Vec<RdmaOp>)> = Vec::new();
        let mut nic_debug = String::new();
        {
            let state = self.state.lock();

            let conn = state
                .addr_connections
                .get(remote_addr)
                .ok_or(TransferError::Backend(format!(
                    "no connection for remote addr {remote_addr}"
                )))?;
            let remote_qpns = &conn.remote_qpns;

            // Prepare ops for each NIC that has work.
            for nic_idx in 0..nic_count {
                let nic_descs = &per_nic[nic_idx];
                if nic_descs.is_empty() {
                    continue;
                }

                let remote_qpn = remote_qpns[nic_idx];
                let session = Arc::clone(
                    state.nics[nic_idx]
                        .sessions
                        .get(&remote_qpn)
                        .expect("session must exist for established connection"),
                );

                let mut prepared = Vec::with_capacity(nic_descs.len());
                let mut nic_bytes = 0usize;
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

                    nic_bytes = nic_bytes.saturating_add(len);
                    prepared.push(RdmaOp {
                        local_mr,
                        local_ptr,
                        remote_ptr,
                        len,
                        remote_rkey,
                    });
                }
                if !nic_debug.is_empty() {
                    nic_debug.push_str(", ");
                }
                let _ = write!(
                    nic_debug,
                    "nic{nic_idx}: {} ops/{:.1} MiB",
                    prepared.len(),
                    nic_bytes as f64 / (1024.0 * 1024.0)
                );
                nic_work.push((nic_idx, session, prepared));
            }
        }
        let lookup_dur = lookup_start.elapsed();

        debug!(
            "batch_transfer_async_{:?}: nics_active={}/{}, chunks={}, lookup_ms={:.3}, per_nic=[{}]",
            op,
            nic_work.len(),
            nic_count,
            descs.len(),
            lookup_dur.as_secs_f64() * 1000.0,
            nic_debug,
        );

        // --- Submit outside lock ---
        let mut receivers = Vec::with_capacity(nic_work.len());
        for (_nic_idx, session, prepared) in nic_work {
            receivers.push(session.transfer_batch_async(prepared, op)?);
        }
        Ok(receivers)
    }
}
