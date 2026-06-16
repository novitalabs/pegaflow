// RDMA remote block fetch: MetaServer query -> gRPC QueryBlocksForTransfer -> RDMA READ.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use log::{debug, info, warn};
use mea::singleflight::Group;
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, RdmaHandshakeRequest,
    ReleaseTransferLockRequest,
};
use pegaflow_transfer::{ConnectionStatus, HandshakeMetadata, TransferOp};
use pegaflow_transfer_wire::TransferPlan;
use tonic::transport::{Channel, Endpoint};

use opentelemetry::KeyValue;

use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;
use crate::transfer_plan::{materialize_transfer_plan, rebuild_sealed_blocks};

/// Minimum usable transfer timeout. If the server's lock timeout minus the
/// safety margin falls below this, we use this floor to avoid instant timeouts.
const MIN_TRANSFER_TIMEOUT: Duration = Duration::from_secs(10);

/// Safety margin subtracted from the server's lock timeout. The client must
/// finish the RDMA transfer before the server releases the lock.
const LOCK_TIMEOUT_MARGIN: Duration = Duration::from_secs(60);

/// RDMA remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries MetaServer for their
/// location, picks the best remote node, and uses gRPC + RDMA READ to fetch them.
pub(crate) struct RdmaFetchStore {
    metaserver_client: Arc<MetaServerClient>,
    rdma_transport: Arc<RdmaTransport>,
    allocate_fn: AllocateFn,
    advertise_addr: String,
    /// Lazy gRPC channel cache keyed by remote address. Tonic channels multiplex
    /// requests over a single HTTP/2 connection; cloning is cheap.
    grpc_channels: Arc<DashMap<String, EngineClient<Channel>>>,
    /// Singleflight group to deduplicate concurrent RDMA handshakes to the
    /// same remote address. Without this, N concurrent fetches to the same
    /// peer would each create QPs and race on the server, causing all but
    /// the last handshake's QPs to be invalidated.
    connect_group: Arc<Group<String, ()>>,
}

impl RdmaFetchStore {
    pub(crate) fn new(
        metaserver_client: Arc<MetaServerClient>,
        rdma_transport: Arc<RdmaTransport>,
        allocate_fn: AllocateFn,
        advertise_addr: String,
    ) -> Self {
        info!("RDMA remote fetch enabled (advertise={})", advertise_addr);
        Self {
            metaserver_client,
            rdma_transport,
            allocate_fn,
            advertise_addr,
            grpc_channels: Arc::new(DashMap::new()),
            connect_group: Arc::new(Group::new()),
        }
    }

    /// Query MetaServer for the best remote node that holds a prefix of `hashes`.
    /// Returns `(node_addr, prefix_len)`, or `None` if no remote node has any.
    pub(crate) async fn query_prefix(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Option<(String, usize)> {
        if hashes.is_empty() {
            return None;
        }

        let nodes = match self.metaserver_client.query_prefix(namespace, hashes).await {
            Ok(n) => n,
            Err(e) => {
                warn!("MetaServer query failed for remote fetch: {e}");
                return None;
            }
        };

        let best = nodes
            .iter()
            .filter(|n| n.node != self.advertise_addr)
            .max_by_key(|n| n.prefix_len)?;

        let prefix_len = best.prefix_len as usize;
        if prefix_len == 0 {
            return None;
        }

        debug!(
            "Remote prefix query: namespace={namespace} best_node={} prefix={prefix_len}/{}",
            best.node,
            hashes.len()
        );

        Some((best.node.clone(), prefix_len))
    }

    /// Fetch `hashes` from `remote_addr`.
    pub(crate) async fn fetch_blocks(
        &self,
        remote_addr: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchResult {
        rdma_fetch_task(
            &self.rdma_transport,
            &self.allocate_fn,
            &self.grpc_channels,
            &self.connect_group,
            remote_addr,
            req_id,
            &self.advertise_addr,
            namespace,
            hashes,
        )
        .await
    }
}

/// Execute RDMA fetch against a single remote node.
///
/// 1. Ensure RDMA connection (singleflight per remote_addr)
/// 2. gRPC QueryBlocksForTransfer RPC (connection reuse, no handshake)
/// 3. RDMA READ all block segments + build SealedBlocks
/// 4. ReleaseTransferLock (fire-and-forget, non-blocking)
#[allow(
    clippy::too_many_arguments,
    reason = "RDMA task arguments are the per-fetch context passed from the scheduler"
)]
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    connect_group: &Group<String, ()>,
    remote_addr: &str,
    req_id: &str,
    advertise_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> PrefetchResult {
    let t0 = Instant::now();

    // 1. Ensure RDMA connection (singleflight: at most one handshake per remote_addr)
    let connect_start = Instant::now();
    if let Err(e) = ensure_connected(
        connect_group,
        rdma,
        grpc_channels,
        remote_addr,
        advertise_addr,
    )
    .await
    {
        warn!("RDMA connect to {remote_addr} failed: {e}");
        core_metrics()
            .rdma_fetch_total
            .add(1, &[KeyValue::new("status", "error")]);
        return Vec::new();
    }
    let connect_elapsed = connect_start.elapsed();

    // 2. gRPC QueryBlocksForTransfer (connection already established)
    let query_start = Instant::now();
    let (client, response) = match query_remote_blocks(
        grpc_channels,
        remote_addr,
        namespace,
        block_hashes,
        advertise_addr,
    )
    .await
    {
        Ok(cr) => cr,
        Err(e) => {
            warn!("Remote query to {remote_addr} failed: {e}");
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };
    let query_elapsed = query_start.elapsed();

    let transfer_session_id = response.transfer_session_id.clone();

    // 3. RDMA READ all blocks + build SealedBlocks
    let transfer_timeout = transfer_timeout_from_server(response.lock_timeout_secs);
    let plan = match TransferPlan::decode_from_slice(&response.transfer_plan) {
        Ok(plan) => plan,
        Err(e) => {
            warn!("Invalid transfer_plan from {remote_addr}: {e}");
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };
    let total_bytes = plan.total_remote_bytes();
    let (result, transfer_timing) = match fetch_blocks_via_rdma(
        rdma,
        allocate_fn,
        namespace,
        remote_addr,
        &plan,
        transfer_timeout,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("RDMA transfer from {remote_addr} failed: {e}");
            rdma.engine().invalidate_connection(remote_addr);
            spawn_release_lock(client, transfer_session_id);
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };

    // 4. Release transfer lock (fire-and-forget: spawns a detached task)
    spawn_release_lock(client, transfer_session_id);

    let elapsed = t0.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
        mb / elapsed.as_secs_f64()
    } else {
        0.0
    };
    info!(
        "RDMA fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} chunks={} placements={} numa_slabs={} descs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
        result.len(),
        block_hashes.len(),
        plan.remote_chunks.len(),
        plan.placements.len(),
        transfer_timing.numa_slab_count,
        transfer_timing.transfer_desc_count,
    );
    info!(
        "RDMA fetch stages: req_id={req_id} remote={remote_addr} connect_ms={:.2} query_ms={:.2} build_transfer_tasks_ms={:.2} submit_transfer_ms={:.2} rdma_wait_ms={:.2} rebuild_ms={:.2}",
        connect_elapsed.as_secs_f64() * 1000.0,
        query_elapsed.as_secs_f64() * 1000.0,
        transfer_timing.build_transfer_tasks.as_secs_f64() * 1000.0,
        transfer_timing.submit_transfer.as_secs_f64() * 1000.0,
        transfer_timing.rdma_wait.as_secs_f64() * 1000.0,
        transfer_timing.rebuild.as_secs_f64() * 1000.0,
    );
    let m = core_metrics();
    let ok = &[KeyValue::new("status", "ok")];
    m.rdma_fetch_total.add(1, ok);
    m.rdma_fetch_duration_seconds
        .record(elapsed.as_secs_f64(), ok);
    m.rdma_fetch_bytes.add(total_bytes, ok);
    result
}

/// Ensure an RDMA connection to `remote_addr` exists, using singleflight to
/// deduplicate concurrent handshakes to the same peer.
///
/// On the first call for a given remote, one task performs the full handshake
/// (prepare QPs → gRPC metadata exchange → complete connection). Concurrent
/// callers wait for that handshake to finish and then reuse the connection.
/// If the handshake fails, the error is returned to the leader and waiting
/// callers retry independently (mea try_work semantics).
async fn ensure_connected(
    connect_group: &Group<String, ()>,
    rdma: &RdmaTransport,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    advertise_addr: &str,
) -> Result<(), String> {
    connect_group
        .try_work(remote_addr.to_string(), async || {
            // Fast path: already connected
            let local_meta = match rdma.engine().get_or_prepare(remote_addr) {
                Ok(ConnectionStatus::Existing) => return Ok(()),
                Ok(ConnectionStatus::Connecting) => {
                    return Err("handshake to this peer already in progress".into());
                }
                Ok(ConnectionStatus::Prepared(m)) => m,
                Err(e) => return Err(format!("RDMA prepare: {e}")),
            };

            // Exchange handshake metadata via the dedicated RdmaHandshake RPC.
            let mut client = get_or_create_channel(grpc_channels, remote_addr)
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?;

            let request = RdmaHandshakeRequest {
                requester_id: advertise_addr.to_string(),
                handshake_metadata: local_meta.to_bytes(),
            };
            let response = client
                .rdma_handshake(request)
                .await
                .map_err(|e| format!("RdmaHandshake RPC failed: {e}"))
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?
                .into_inner();

            // Complete the RDMA connection with the server's QP info.
            finish_handshake(rdma, remote_addr, &local_meta, &response.handshake_metadata)
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?;

            Ok(())
        })
        .await
}

/// Allocate local memory, build TransferDescs, execute RDMA READ, build SealedBlocks.
async fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_addr: &str,
    plan: &TransferPlan,
    transfer_timeout: Duration,
) -> Result<(PrefetchResult, TransferTiming), String> {
    if plan.block_hashes.is_empty() {
        return Ok((Vec::new(), TransferTiming::default()));
    }

    let build_start = Instant::now();
    let (receivers, mut timing, rebuild_ctx) = {
        let materialized = materialize_transfer_plan(plan, allocate_fn)?;
        let transfer_desc_count = materialized.rdma_descs.len();
        let numa_slab_count = materialized.rebuild.slabs.len();
        if materialized.rdma_descs.is_empty() {
            let timing = TransferTiming {
                build_transfer_tasks: build_start.elapsed(),
                transfer_desc_count,
                numa_slab_count,
                ..TransferTiming::default()
            };
            return Ok((Vec::new(), timing));
        }

        let submit_start = Instant::now();
        let receivers = rdma
            .engine()
            .batch_transfer_async(TransferOp::Read, remote_addr, &materialized.rdma_descs)
            .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?;
        let submit_transfer = submit_start.elapsed();

        let timing = TransferTiming {
            build_transfer_tasks: build_start.elapsed().saturating_sub(submit_transfer),
            submit_transfer,
            transfer_desc_count,
            numa_slab_count,
            ..TransferTiming::default()
        };
        let rebuild_ctx = materialized.rebuild;
        (receivers, timing, rebuild_ctx)
    };

    let wait_start = Instant::now();
    tokio::time::timeout(transfer_timeout, async {
        for rx in receivers {
            rx.await
                .map_err(|_| "RDMA transfer channel closed".to_string())?
                .map_err(|e| format!("RDMA transfer failed: {e}"))?;
        }
        Ok::<(), String>(())
    })
    .await
    .map_err(|_| "RDMA transfer timed out".to_string())??;
    timing.rdma_wait = wait_start.elapsed();

    let rebuild_start = Instant::now();
    let result = rebuild_sealed_blocks(plan, &rebuild_ctx, namespace)?;
    timing.rebuild = rebuild_start.elapsed();

    Ok((result, timing))
}

#[derive(Default)]
struct TransferTiming {
    build_transfer_tasks: Duration,
    submit_transfer: Duration,
    rdma_wait: Duration,
    rebuild: Duration,
    transfer_desc_count: usize,
    numa_slab_count: usize,
}

fn get_or_create_channel(
    cache: &DashMap<String, EngineClient<Channel>>,
    addr: &str,
) -> Result<EngineClient<Channel>, String> {
    if let Some(client) = cache.get(addr) {
        return Ok(client.clone());
    }
    let url = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    let channel = Endpoint::from_shared(url)
        .map_err(|e| format!("invalid remote address: {e}"))?
        .connect_timeout(Duration::from_secs(5))
        .connect_lazy();
    // Match the engine server's 64 MiB message cap.
    const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
    let client = EngineClient::new(channel)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
    cache.insert(addr.to_string(), client.clone());
    Ok(client)
}

/// Get/create gRPC channel and call QueryBlocksForTransfer.
async fn query_remote_blocks(
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
    advertise_addr: &str,
) -> Result<(EngineClient<Channel>, QueryBlocksForTransferResponse), String> {
    let mut client = get_or_create_channel(grpc_channels, remote_addr)?;

    let request = QueryBlocksForTransferRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.to_vec(),
        requester_id: advertise_addr.to_string(),
    };

    let response = client
        .query_blocks_for_transfer(request)
        .await
        .map_err(|e| format!("QueryBlocksForTransfer RPC failed: {e}"))?
        .into_inner();

    if let Some(st) = &response.status
        && !st.ok
    {
        return Err(format!("remote returned error: {}", st.message));
    }

    Ok((client, response))
}

/// Decode remote handshake metadata and complete the RDMA connection.
fn finish_handshake(
    rdma: &RdmaTransport,
    remote_addr: &str,
    local_meta: &HandshakeMetadata,
    remote_bytes: &[u8],
) -> Result<(), String> {
    if remote_bytes.is_empty() {
        return Err("server returned empty handshake_metadata".into());
    }
    let remote_meta = HandshakeMetadata::from_bytes(remote_bytes)
        .map_err(|e| format!("invalid metadata: {e}"))?;
    rdma.engine()
        .complete_handshake(remote_addr, local_meta, &remote_meta)
        .map_err(|e| format!("{e}"))
}

/// Compute client-side transfer timeout from server's lock timeout.
/// Returns `max(server_timeout - 60s, 10s)` so the client always finishes
/// before the server force-releases the lock.
fn transfer_timeout_from_server(lock_timeout_secs: u32) -> Duration {
    let server = Duration::from_secs(lock_timeout_secs as u64);
    server
        .saturating_sub(LOCK_TIMEOUT_MARGIN)
        .max(MIN_TRANSFER_TIMEOUT)
}

/// Release a transfer lock in a detached task. Does not block the caller.
fn spawn_release_lock(mut client: EngineClient<Channel>, transfer_session_id: String) {
    if transfer_session_id.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let req = ReleaseTransferLockRequest {
            transfer_session_id: transfer_session_id.clone(),
        };
        if let Err(e) = client.release_transfer_lock(req).await {
            warn!("ReleaseTransferLock failed for session {transfer_session_id}: {e}");
        }
    });
}

#[cfg(test)]
mod tests {
    use pegaflow_transfer_wire::{RemoteChunk, TransferPlan};

    #[test]
    fn transfer_plan_total_remote_bytes_sums_chunks() {
        let plan = TransferPlan {
            slot_schemas: vec![],
            remote_chunks: vec![
                RemoteChunk {
                    base_ptr: 0x1000,
                    length: 100,
                    numa_node: 0,
                },
                RemoteChunk {
                    base_ptr: 0x2000,
                    length: 200,
                    numa_node: 0,
                },
            ],
            placements: vec![],
            block_hashes: vec![],
        };
        assert_eq!(plan.total_remote_bytes(), 300);
    }
}
