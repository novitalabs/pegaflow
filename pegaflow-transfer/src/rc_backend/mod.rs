mod runtime;
mod session;
mod state;

use std::collections::HashMap;
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

/// Max bytes per RDMA work request; larger `TransferDesc`s are split at the engine entry.
const MAX_RDMA_TRANSFER_BYTES: usize = 128 * 1024 * 1024;

fn mr_access_flags() -> AccessFlags {
    AccessFlags::LocalWrite
        | AccessFlags::RemoteWrite
        | AccessFlags::RemoteRead
        | AccessFlags::RelaxedOrdering
}

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
    qps_per_peer: usize,
}

impl RcBackend {
    pub(crate) fn new(nic_names: &[String], qps_per_peer: usize) -> Result<Self> {
        crate::init_logging();
        if nic_names.is_empty() {
            return Err(TransferError::InvalidArgument("nic_names is empty"));
        }
        if qps_per_peer == 0 {
            return Err(TransferError::InvalidArgument("qps_per_peer must be > 0"));
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
        info!(
            "RC backend init: nics={}, qps_per_peer={}",
            nic_count, qps_per_peer
        );
        Ok(Self {
            runtimes,
            state: Arc::new(Mutex::new(RcBackendState::new(nic_count))),
            psn_counter: AtomicU64::new(1),
            numa_rr,
            qps_per_peer,
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
            let mr = unsafe { runtime.pd.reg_mr(raw as usize, len, mr_access_flags()) }
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

    /// Create N RC QPs per NIC in INIT state, push to per-NIC pending queues,
    /// return per-NIC handshake data (each NIC carries N endpoints).
    fn prepare_handshake(&self) -> Result<Vec<NicHandshake>> {
        let snapshots = self.snapshot_registered_memory();
        let mut nic_handshakes = Vec::with_capacity(self.nic_count());

        // Create all NIC × N QPs before locking state.
        let mut sessions_per_nic: Vec<Vec<Arc<RcSession>>> =
            (0..self.nic_count()).map(|_| Vec::new()).collect();
        for (nic_idx, runtime) in self.runtimes.iter().enumerate() {
            for qp_idx in 0..self.qps_per_peer {
                let psn_seed = self.psn_counter.fetch_add(1, Ordering::Relaxed);
                let session = RcSession::create(runtime, psn_seed).map_err(|e| {
                    TransferError::Backend(format!(
                        "QP creation failed on nic={} qp_idx={qp_idx}: {e}",
                        runtime.nic_name
                    ))
                })?;
                sessions_per_nic[nic_idx].push(session);
            }
        }

        let mut state = self.state.lock();
        for (nic_idx, sessions) in sessions_per_nic.into_iter().enumerate() {
            let endpoints: Vec<_> = sessions.iter().map(|s| s.local_endpoint).collect();
            for s in sessions {
                state.nics[nic_idx].pending.push_back(s);
            }
            nic_handshakes.push(NicHandshake {
                endpoints,
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
        for (nic_idx, nic) in remote_nics.iter().enumerate() {
            if nic.endpoints.len() != self.qps_per_peer {
                return Err(TransferError::Backend(format!(
                    "remote qps_per_peer mismatch on nic={nic_idx}: local={}, remote={}",
                    self.qps_per_peer,
                    nic.endpoints.len()
                )));
            }
        }

        // Pop pending sessions by matching QPN from local_nics (not blind FIFO).
        // Concurrent prepare/complete for different remote addrs could reorder the
        // queue, so we must find our own sessions by QPN.
        let pending: Vec<Vec<Arc<RcSession>>> = {
            let mut state = self.state.lock();
            // If already connected (concurrent request beat us), remove our pending sessions
            if state.addr_connections.contains_key(remote_addr) {
                for (nic_idx, nic) in local_nics.iter().enumerate() {
                    for ep in &nic.endpoints {
                        state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num);
                    }
                }
                state.connecting.remove(remote_addr);
                info!("handshake won by concurrent path: remote={remote_addr}");
                return Ok(());
            }
            let mut sessions_per_nic: Vec<Vec<Arc<RcSession>>> = (0..nic_count)
                .map(|_| Vec::with_capacity(self.qps_per_peer))
                .collect();
            for (nic_idx, nic) in local_nics.iter().enumerate() {
                for ep in &nic.endpoints {
                    let session = state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num).ok_or(
                        TransferError::Backend("no pending session to complete".to_string()),
                    )?;
                    sessions_per_nic[nic_idx].push(session);
                }
            }
            sessions_per_nic
        };

        // Connect outside lock — pair local QP i with remote QP i in handshake order.
        for (nic_idx, sessions) in pending.iter().enumerate() {
            let remote_eps = &remote_nics[nic_idx].endpoints;
            for (qp_idx, session) in sessions.iter().enumerate() {
                session.connect(&self.runtimes[nic_idx], &remote_eps[qp_idx])?;
            }
        }

        // Store sessions + addr_connections
        let mut state = self.state.lock();
        let mut remote_first_qpns = Vec::with_capacity(nic_count);
        for (nic_idx, sessions) in pending.into_iter().enumerate() {
            let remote = &remote_nics[nic_idx];
            let remote_first_qpn = remote.endpoints[0].qp_num;
            state.nics[nic_idx].cache_remote_memory(remote_first_qpn, &remote.memory_regions)?;
            state.nics[nic_idx]
                .sessions
                .insert(remote_first_qpn, sessions);
            remote_first_qpns.push(remote_first_qpn);
        }
        let removed = state.connecting.remove(remote_addr);
        debug_assert!(removed, "connecting set should contain {remote_addr}");
        let local_qpns: Vec<Vec<u32>> = local_nics
            .iter()
            .map(|n| n.endpoints.iter().map(|e| e.qp_num).collect())
            .collect();
        let remote_qpns: Vec<Vec<u32>> = remote_nics
            .iter()
            .map(|n| n.endpoints.iter().map(|e| e.qp_num).collect())
            .collect();
        info!(
            "RDMA connection established: remote={remote_addr}, qps_per_peer={}, local_qpns={local_qpns:?}, remote_qpns={remote_qpns:?}",
            self.qps_per_peer
        );
        let rr_counters = (0..nic_count).map(|_| AtomicUsize::new(0)).collect();
        state.addr_connections.insert(
            remote_addr.to_string(),
            AddrConnection {
                remote_first_qpns,
                local_nics,
                rr_counters,
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
            for ep in &nic.endpoints {
                state.nics[nic_idx].remove_pending_by_qpn(ep.qp_num);
            }
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
            for (nic_idx, &remote_first_qpn) in conn.remote_first_qpns.iter().enumerate() {
                state.nics[nic_idx].cleanup_connection(remote_first_qpn);
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
        let mut nic_work: Vec<(Arc<RcSession>, Vec<RdmaOp>)> = Vec::new();
        {
            let state = self.state.lock();

            let conn = state
                .addr_connections
                .get(remote_addr)
                .ok_or(TransferError::Backend(format!(
                    "no connection for remote addr {remote_addr}"
                )))?;

            // Per-WQE round-robin across the N sessions per NIC so a single
            // batch can fill all N QPs (per-call RR would leave the rest idle).
            for (nic_idx, nic_descs) in per_nic.iter().enumerate() {
                if nic_descs.is_empty() {
                    continue;
                }

                let remote_first_qpn = conn.remote_first_qpns[nic_idx];
                let sessions = state.nics[nic_idx]
                    .sessions
                    .get(&remote_first_qpn)
                    .expect("session vec must exist for established connection");
                let n = sessions.len();
                // Rotate the starting bucket each call so small batches still
                // hit different QPs across calls.
                let rot = conn.rr_counters[nic_idx].fetch_add(1, Ordering::Relaxed);

                let per_bucket = nic_descs.len().div_ceil(n);
                let mut buckets: Vec<Vec<RdmaOp>> =
                    (0..n).map(|_| Vec::with_capacity(per_bucket)).collect();

                for (i, desc) in nic_descs.iter().enumerate() {
                    let base_local = desc.local_ptr.as_ptr() as u64;
                    let base_remote = desc.remote_ptr.as_ptr() as u64;
                    let len = desc.len;

                    if len == 0 {
                        return Err(TransferError::InvalidArgument("len must be non-zero"));
                    }

                    let bucket = rot.wrapping_add(i) % n;

                    let mut offset = 0usize;
                    while offset < len {
                        let chunk_len = (len - offset).min(MAX_RDMA_TRANSFER_BYTES);
                        let local_ptr = base_local.checked_add(offset as u64).ok_or(
                            TransferError::InvalidArgument("local_ptr + offset overflow"),
                        )?;
                        let remote_ptr = base_remote.checked_add(offset as u64).ok_or(
                            TransferError::InvalidArgument("remote_ptr + offset overflow"),
                        )?;

                        let local_mr = state
                            .find_local_mr(nic_idx, local_ptr, chunk_len)
                            .ok_or(TransferError::MemoryNotRegistered { ptr: local_ptr })?;

                        let remote_rkey = state.nics[nic_idx]
                            .find_remote_rkey(remote_first_qpn, remote_ptr, chunk_len)
                            .ok_or(TransferError::InvalidArgument(
                                "remote memory not found in handshake snapshot",
                            ))?;

                        buckets[bucket].push(RdmaOp {
                            local_mr,
                            local_ptr,
                            remote_ptr,
                            len: chunk_len,
                            remote_rkey,
                        });
                        offset += chunk_len;
                    }
                }

                for (q_idx, prepared) in buckets.into_iter().enumerate() {
                    if prepared.is_empty() {
                        continue;
                    }
                    nic_work.push((Arc::clone(&sessions[q_idx]), prepared));
                }
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
        let mut receivers = Vec::with_capacity(nic_work.len());
        for (session, prepared) in nic_work {
            receivers.push(session.transfer_batch_async(prepared, op)?);
        }
        Ok(receivers)
    }
}
