use std::collections::HashMap;
use std::sync::Weak;

use log::{debug, error, info, warn};
use pegaflow_common::grpc::{GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL, GRPC_CONNECT_TIMEOUT};
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient as MetaServerGrpcClient;
use pegaflow_proto::proto::engine::{
    HeartbeatNodeRequest, InsertBlockHashesRequest, RemoveBlockHashesRequest, UnregisterNodeRequest,
};
#[cfg(feature = "rdma")]
use pegaflow_proto::proto::engine::{NodePrefixResult, QueryPrefixBlocksRequest};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, Instant};
use tonic::Code;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

use crate::metrics::core_metrics;
use crate::storage::ReadCache;

// Shared insert/remove command channel depth. Eviction bursts outrun the single
// consumer's per-RPC drain, so a shallow queue silently drops removals.
pub const DEFAULT_METASERVER_QUEUE_DEPTH: usize = 4096;

// Cap hashes per insert/remove RPC. The consumer coalesces a whole queue drain
// per namespace, so without a cap one RPC could reach queue_depth * batch_size
// hashes and blow past the MetaServer's gRPC decode limit. 32-byte sha256 hashes
// keep a full chunk near 0.5 MiB, well under tonic's 4 MiB default.
const MAX_HASHES_PER_RPC: usize = 16_384;

/// Error type for MetaServer client operations.
#[cfg(feature = "rdma")]
#[derive(Debug)]
pub(crate) enum ClientError {
    RpcFailed(String),
}

#[cfg(feature = "rdma")]
impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::RpcFailed(msg) => write!(f, "RPC failed: {msg}"),
        }
    }
}

const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 30_000;
const MIN_HEARTBEAT_INTERVAL_SECS: u64 = 1;
const UNREGISTER_TIMEOUT_SECS: u64 = 3;

struct HeartbeatState {
    node_registered: bool,
    backoff_ms: u64,
    period: Duration,
    next_at: Instant,
}

impl HeartbeatState {
    fn new() -> Self {
        Self {
            node_registered: false,
            backoff_ms: INITIAL_BACKOFF_MS,
            period: Duration::from_secs(MIN_HEARTBEAT_INTERVAL_SECS),
            next_at: Instant::now(),
        }
    }
}

pub struct MetaServerClientConfig {
    pub metaserver_addr: String,
    pub advertise_addr: String,
    pub queue_depth: usize,
}

impl MetaServerClientConfig {
    pub fn new(metaserver_addr: String, advertise_addr: String) -> Self {
        Self {
            metaserver_addr,
            advertise_addr,
            queue_depth: DEFAULT_METASERVER_QUEUE_DEPTH,
        }
    }

    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth;
        self
    }
}

/// A batch of block hashes grouped by namespace for MetaServer operations.
struct BlockHashBatch {
    groups: Vec<(String, Vec<Vec<u8>>)>,
}

impl BlockHashBatch {
    fn from_entries(entries: Vec<(String, Vec<u8>)>) -> Self {
        let mut groups: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for (namespace, hash) in entries {
            groups.entry(namespace).or_default().push(hash);
        }
        Self {
            groups: groups.into_iter().collect(),
        }
    }

    fn single_namespace(namespace: String, hashes: Vec<Vec<u8>>) -> Self {
        Self {
            groups: vec![(namespace, hashes)],
        }
    }

    fn count(&self) -> usize {
        self.groups.iter().map(|(_, hashes)| hashes.len()).sum()
    }
}

/// Command sent to the background MetaServer loop.
enum MetaServerCommand {
    Insert(BlockHashBatch),
    Remove(BlockHashBatch),
    /// Barrier: acked once every insert/remove enqueued before it has been
    /// delivered to the MetaServer (or dropped after a failed attempt).
    Flush(oneshot::Sender<()>),
    Shutdown(oneshot::Sender<()>),
}

fn metaserver_endpoint(metaserver_addr: String) -> Endpoint {
    // keep_alive_timeout is 20s by default on both client and server side (tonic).
    Endpoint::from_shared(metaserver_addr)
        .expect("valid metaserver_addr URI")
        .connect_timeout(GRPC_CONNECT_TIMEOUT)
        .http2_keep_alive_interval(GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL)
        .keep_alive_while_idle(true)
}

async fn connect_metaserver_client(
    endpoint: &Endpoint,
) -> Result<MetaServerGrpcClient<Channel>, tonic::transport::Error> {
    endpoint.connect().await.map(MetaServerGrpcClient::new)
}

/// Unified MetaServer client handling both insert (fire-and-forget) and query (direct RPC).
pub struct MetaServerClient {
    /// Fire-and-forget command channel for insert/remove operations.
    command_tx: mpsc::Sender<MetaServerCommand>,
    /// Lazy-connect query client
    #[cfg(feature = "rdma")]
    query_client: MetaServerGrpcClient<Channel>,
}

impl MetaServerClient {
    /// Create a new client and spawn the background registration loop.
    ///
    /// Must be called from within a tokio runtime context.
    pub(crate) fn new(config: MetaServerClientConfig, read_cache: Weak<ReadCache>) -> Self {
        let endpoint = metaserver_endpoint(config.metaserver_addr.clone());
        let (command_tx, rx) = mpsc::channel(config.queue_depth);

        tokio::spawn(registration_loop(
            rx,
            config.metaserver_addr.clone(),
            endpoint.clone(),
            config.advertise_addr,
            read_cache,
        ));

        // Lazy-connect query client: connects on first RPC, not here
        #[cfg(feature = "rdma")]
        let query_client = {
            let channel = endpoint.connect_lazy();
            MetaServerGrpcClient::new(channel)
        };

        info!(
            "MetaServer client started (queue_depth={}, addr={})",
            config.queue_depth, config.metaserver_addr
        );

        Self {
            command_tx,
            #[cfg(feature = "rdma")]
            query_client,
        }
    }

    /// Fire-and-forget registration of block hashes.
    ///
    /// Accepts one namespace and its block hashes so callers can preserve their
    /// hot-path grouping and enqueue a single MetaServer command.
    pub(crate) fn try_register_namespace(&self, namespace: String, hashes: Vec<Vec<u8>>) {
        if hashes.is_empty() {
            return;
        }
        let batch = BlockHashBatch::single_namespace(namespace, hashes);
        let count = batch.count();
        match self.command_tx.try_send(MetaServerCommand::Insert(batch)) {
            Ok(()) => {
                core_metrics()
                    .metaserver_registration_blocks
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!(
                    "MetaServer registration queue full, dropping {} hashes",
                    count
                );
                core_metrics()
                    .metaserver_registration_queue_full
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!(
                    "MetaServer registration loop has exited, dropping {} hashes",
                    count
                );
                core_metrics()
                    .metaserver_registration_queue_full
                    .add(count as u64, &[]);
            }
        }
    }

    /// Fire-and-forget removal of block hashes.
    ///
    /// Called after LRU eviction to notify MetaServer that this node no longer
    /// holds these blocks. Losing an occasional remove message is acceptable;
    /// the node lifecycle sweep is the fallback for node failures.
    pub(crate) fn try_unregister(&self, entries: Vec<(String, Vec<u8>)>) {
        if entries.is_empty() {
            return;
        }
        self.try_send_unregister_batch(BlockHashBatch::from_entries(entries));
    }

    fn try_send_unregister_batch(&self, batch: BlockHashBatch) {
        let count = batch.count();
        match self.command_tx.try_send(MetaServerCommand::Remove(batch)) {
            Ok(()) => {
                core_metrics()
                    .metaserver_removal_blocks
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("MetaServer removal queue full, dropping {} hashes", count);
                core_metrics()
                    .metaserver_removal_queue_full
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!(
                    "MetaServer removal loop has exited, dropping {} hashes",
                    count
                );
                core_metrics()
                    .metaserver_removal_queue_full
                    .add(count as u64, &[]);
            }
        }
    }

    /// Best-effort graceful unregister of this server's MetaServer node session.
    pub async fn shutdown(&self) {
        let (done_tx, done_rx) = oneshot::channel();
        let shutdown_timeout = tokio::time::Duration::from_secs(UNREGISTER_TIMEOUT_SECS + 1);
        match tokio::time::timeout(
            shutdown_timeout,
            self.command_tx.send(MetaServerCommand::Shutdown(done_tx)),
        )
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(_)) | Err(_) => return,
        }
        let _ = tokio::time::timeout(shutdown_timeout, done_rx).await;
    }

    /// Barrier: resolves once every registration enqueued before this call has
    /// been delivered to the MetaServer — or dropped after a failed attempt.
    /// "Attempted" is the strongest contract the fire-and-forget queue can
    /// offer; on ack, a subsequent MetaServer query observes every hash whose
    /// insert RPC succeeded.
    ///
    /// Returns immediately if the registration loop has already exited.
    pub async fn flush(&self) {
        let (done_tx, done_rx) = oneshot::channel();
        if self
            .command_tx
            .send(MetaServerCommand::Flush(done_tx))
            .await
            .is_err()
        {
            return;
        }
        let _ = done_rx.await;
    }

    /// Query MetaServer for the longest prefix of blocks that exist remotely.
    /// Returns per-node prefix lengths.
    #[cfg(feature = "rdma")]
    pub(crate) async fn query_prefix(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<NodePrefixResult>, ClientError> {
        let request = QueryPrefixBlocksRequest {
            namespace: namespace.to_string(),
            block_hashes: hashes.to_vec(),
        };

        let response = self
            .query_client
            .clone()
            .query_prefix_blocks(request)
            .await
            .map_err(|e| ClientError::RpcFailed(format!("MetaServer query failed: {e}")))?;

        let resp = response.into_inner();

        debug!(
            "MetaServer query_prefix: namespace={} nodes={}",
            namespace,
            resp.nodes.len()
        );

        Ok(resp.nodes)
    }
}

async fn registration_loop(
    mut rx: mpsc::Receiver<MetaServerCommand>,
    metaserver_addr: String,
    endpoint: Endpoint,
    advertise_addr: String,
    read_cache: Weak<ReadCache>,
) {
    let mut client: Option<MetaServerGrpcClient<Channel>> = None;
    let node_id = Uuid::new_v4().to_string();
    let mut heartbeat = HeartbeatState::new();

    loop {
        let heartbeat_sleep = tokio::time::sleep_until(heartbeat.next_at);
        tokio::pin!(heartbeat_sleep);
        let cmd = tokio::select! {
            cmd = rx.recv() => match cmd {
                Some(cmd) => cmd,
                None => break,
            },
            _ = &mut heartbeat_sleep => {
                match send_heartbeat(
                    &mut client,
                    &mut heartbeat,
                    &metaserver_addr,
                    &endpoint,
                    &advertise_addr,
                    &node_id,
                ).await {
                    Ok(next_period) => {
                        heartbeat.period = next_period;
                        heartbeat.next_at = Instant::now() + heartbeat.period;
                    }
                    Err(retry_after) => {
                        heartbeat.next_at = Instant::now() + retry_after;
                    }
                }
                continue;
            }
        };

        if let MetaServerCommand::Shutdown(done) = cmd {
            unregister_current_session(
                &mut client,
                &metaserver_addr,
                &endpoint,
                &advertise_addr,
                &node_id,
            )
            .await;
            let _ = done.send(());
            break;
        }

        let mut inserts: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut removes: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut mixed_ops: Option<HashMap<(String, Vec<u8>), bool>> = None; // true=insert
        let mut saw_insert = false;
        let mut saw_remove = false;
        // Flush barriers drained in this batch; acked after the sends below, so
        // an ack means "everything enqueued before the flush has been attempted".
        let mut flush_acks: Vec<oneshot::Sender<()>> = Vec::new();

        // Drain all pending commands. Pure insert/remove batches stay grouped by
        // namespace; mixed streams switch to last-write-wins netting.
        for cmd in std::iter::once(cmd).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
            match cmd {
                MetaServerCommand::Insert(batch) => {
                    saw_insert = true;
                    if saw_remove {
                        let net = mixed_ops.get_or_insert_with(|| {
                            build_net(std::mem::take(&mut inserts), std::mem::take(&mut removes))
                        });
                        insert_groups_into_net(net, batch.groups, true);
                    } else {
                        append_groups(&mut inserts, batch.groups);
                    }
                }
                MetaServerCommand::Remove(batch) => {
                    saw_remove = true;
                    if saw_insert {
                        let net = mixed_ops.get_or_insert_with(|| {
                            build_net(std::mem::take(&mut inserts), std::mem::take(&mut removes))
                        });
                        insert_groups_into_net(net, batch.groups, false);
                    } else {
                        append_groups(&mut removes, batch.groups);
                    }
                }
                MetaServerCommand::Flush(done) => {
                    flush_acks.push(done);
                }
                MetaServerCommand::Shutdown(done) => {
                    unregister_current_session(
                        &mut client,
                        &metaserver_addr,
                        &endpoint,
                        &advertise_addr,
                        &node_id,
                    )
                    .await;
                    let _ = done.send(());
                    info!("MetaServer registration loop shutting down");
                    return;
                }
            }
        }

        if let Some(net) = mixed_ops {
            for ((namespace, hash), is_insert) in net {
                if is_insert {
                    inserts.entry(namespace).or_default().push(hash);
                } else {
                    removes.entry(namespace).or_default().push(hash);
                }
            }
        }

        let insert_total: usize = inserts.values().map(|v| v.len()).sum();
        let remove_total: usize = removes.values().map(|v| v.len()).sum();
        if ensure_heartbeat_registered(
            &mut client,
            &metaserver_addr,
            &endpoint,
            &advertise_addr,
            &node_id,
            &mut heartbeat,
        )
        .await
        .is_err()
        {
            core_metrics()
                .metaserver_registration_failures
                .add(insert_total as u64, &[]);
            core_metrics()
                .metaserver_removal_failures
                .add(remove_total as u64, &[]);
            // The batched hashes are dropped, not retried: the barrier's
            // "delivered or dropped" contract is met, so ack the flushes.
            ack_flushes(flush_acks);
            continue;
        }

        let c = client.as_mut().expect("client is Some after lazy-connect");

        // Process inserts
        let insert_namespaces: Vec<(String, Vec<Vec<u8>>)> = inserts.into_iter().collect();
        let mut insert_failed_at: Option<(usize, usize)> = None;

        'insert: for (i, (namespace, hashes)) in insert_namespaces.iter().enumerate() {
            for (chunk_idx, chunk) in hashes.chunks(MAX_HASHES_PER_RPC).enumerate() {
                let count = chunk.len();
                let request = InsertBlockHashesRequest {
                    namespace: namespace.clone(),
                    block_hashes: chunk.to_vec(),
                    node: advertise_addr.clone(),
                    node_id: node_id.clone(),
                };

                match c.insert_block_hashes(request).await {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        if !inner.reclaimable_hashes.is_empty()
                            && let Some(cache) = read_cache.upgrade()
                        {
                            cache.mark_reclaimable_hashes(namespace, &inner.reclaimable_hashes);
                        }
                        debug!(
                            "Registered {} block hashes with MetaServer (namespace={}, inserted={}, reclaimable={})",
                            count,
                            namespace,
                            inner.inserted_count,
                            inner.reclaimable_hashes.len()
                        );
                    }
                    Err(e) => {
                        error!(
                            "MetaServer insert_block_hashes failed (namespace={}, count={}): {e}",
                            namespace, count
                        );
                        if e.code() == Code::FailedPrecondition {
                            warn!(
                                "MetaServer insert rejected current session; resetting node registration: node={} node_id={}",
                                advertise_addr, node_id
                            );
                            core_metrics().metaserver_session_resets.add(1, &[]);
                            heartbeat.node_registered = false;
                        }
                        insert_failed_at = Some((i, chunk_idx * MAX_HASHES_PER_RPC));
                        break 'insert;
                    }
                }
            }
        }

        if let Some((idx, offset)) = insert_failed_at {
            let dropped = unsent_after_failure(&insert_namespaces, idx, offset);
            core_metrics()
                .metaserver_registration_failures
                .add(dropped as u64, &[]);
            let remove_total: usize = removes.values().map(|v| v.len()).sum();
            if remove_total > 0 {
                core_metrics()
                    .metaserver_removal_failures
                    .add(remove_total as u64, &[]);
            }
            client = None;
            ack_flushes(flush_acks);
            continue;
        }

        // Process removes
        let remove_namespaces: Vec<(String, Vec<Vec<u8>>)> = removes.into_iter().collect();
        let mut remove_failed_at: Option<(usize, usize)> = None;

        'remove: for (i, (namespace, hashes)) in remove_namespaces.iter().enumerate() {
            for (chunk_idx, chunk) in hashes.chunks(MAX_HASHES_PER_RPC).enumerate() {
                let count = chunk.len();
                let request = RemoveBlockHashesRequest {
                    namespace: namespace.clone(),
                    block_hashes: chunk.to_vec(),
                    node: advertise_addr.clone(),
                    node_id: node_id.clone(),
                };

                match c.remove_block_hashes(request).await {
                    Ok(resp) => {
                        let inner = resp.into_inner();
                        debug!(
                            "Removed {} block hashes from MetaServer (namespace={}, removed={})",
                            count, namespace, inner.removed_count
                        );
                    }
                    Err(e) => {
                        error!(
                            "MetaServer remove_block_hashes failed (namespace={}, count={}): {e}",
                            namespace, count
                        );
                        if e.code() == Code::FailedPrecondition {
                            warn!(
                                "MetaServer remove rejected current session; resetting node registration: node={} node_id={}",
                                advertise_addr, node_id
                            );
                            core_metrics().metaserver_session_resets.add(1, &[]);
                            heartbeat.node_registered = false;
                        }
                        remove_failed_at = Some((i, chunk_idx * MAX_HASHES_PER_RPC));
                        break 'remove;
                    }
                }
            }
        }

        if let Some((idx, offset)) = remove_failed_at {
            let dropped = unsent_after_failure(&remove_namespaces, idx, offset);
            core_metrics()
                .metaserver_removal_failures
                .add(dropped as u64, &[]);
            client = None;
        }
        ack_flushes(flush_acks);
    }

    info!("MetaServer registration loop shutting down");
}

fn ack_flushes(acks: Vec<oneshot::Sender<()>>) {
    for done in acks {
        let _ = done.send(());
    }
}

/// Hashes that did not reach the MetaServer after a chunked send failed at
/// `failed_offset` within namespace `failed_idx`: the unsent tail of that
/// namespace (earlier chunks already landed) plus every later namespace.
fn unsent_after_failure(
    namespaces: &[(String, Vec<Vec<u8>>)],
    failed_idx: usize,
    failed_offset: usize,
) -> usize {
    let unsent_in_ns = namespaces[failed_idx].1.len().saturating_sub(failed_offset);
    let later: usize = namespaces[failed_idx + 1..]
        .iter()
        .map(|(_, h)| h.len())
        .sum();
    unsent_in_ns + later
}

fn append_groups(target: &mut HashMap<String, Vec<Vec<u8>>>, groups: Vec<(String, Vec<Vec<u8>>)>) {
    for (namespace, mut hashes) in groups {
        target.entry(namespace).or_default().append(&mut hashes);
    }
}

fn build_net(
    inserts: HashMap<String, Vec<Vec<u8>>>,
    removes: HashMap<String, Vec<Vec<u8>>>,
) -> HashMap<(String, Vec<u8>), bool> {
    let mut net = HashMap::new();
    insert_map_into_net(&mut net, inserts, true);
    insert_map_into_net(&mut net, removes, false);
    net
}

fn insert_map_into_net(
    net: &mut HashMap<(String, Vec<u8>), bool>,
    grouped: HashMap<String, Vec<Vec<u8>>>,
    is_insert: bool,
) {
    for (namespace, hashes) in grouped {
        insert_groups_into_net(net, vec![(namespace, hashes)], is_insert);
    }
}

fn insert_groups_into_net(
    net: &mut HashMap<(String, Vec<u8>), bool>,
    groups: Vec<(String, Vec<Vec<u8>>)>,
    is_insert: bool,
) {
    for (namespace, hashes) in groups {
        for hash in hashes {
            net.insert((namespace.clone(), hash), is_insert);
        }
    }
}

async fn ensure_heartbeat_registered(
    client: &mut Option<MetaServerGrpcClient<Channel>>,
    metaserver_addr: &str,
    endpoint: &Endpoint,
    advertise_addr: &str,
    node_id: &str,
    heartbeat: &mut HeartbeatState,
) -> Result<(), ()> {
    if heartbeat.node_registered && client.is_some() {
        return Ok(());
    }

    if Instant::now() < heartbeat.next_at {
        return Err(());
    }

    match send_heartbeat(
        client,
        heartbeat,
        metaserver_addr,
        endpoint,
        advertise_addr,
        node_id,
    )
    .await
    {
        Ok(next_period) => {
            heartbeat.period = next_period;
            heartbeat.next_at = Instant::now() + next_period;
            Ok(())
        }
        Err(retry_after) => {
            heartbeat.next_at = Instant::now() + retry_after;
            Err(())
        }
    }
}

async fn send_heartbeat(
    client: &mut Option<MetaServerGrpcClient<Channel>>,
    heartbeat: &mut HeartbeatState,
    metaserver_addr: &str,
    endpoint: &Endpoint,
    advertise_addr: &str,
    node_id: &str,
) -> Result<Duration, Duration> {
    if client.is_none() {
        match connect_metaserver_client(endpoint).await {
            Ok(c) => {
                info!("Connected to MetaServer at {}", metaserver_addr);
                *client = Some(c);
                heartbeat.backoff_ms = INITIAL_BACKOFF_MS;
            }
            Err(e) => {
                error!("Failed to connect to MetaServer: {e}");
                core_metrics().metaserver_heartbeat_failures.add(1, &[]);
                return Err(heartbeat.advance_backoff());
            }
        }
    }

    let c = client.as_mut().expect("client is connected");
    match c
        .heartbeat_node(HeartbeatNodeRequest {
            node: advertise_addr.to_string(),
            node_id: node_id.to_string(),
        })
        .await
    {
        Ok(resp) => {
            let heartbeat_period =
                heartbeat_period_from_stale_after(resp.into_inner().stale_after_secs);
            if !heartbeat.node_registered {
                info!(
                    "MetaServer heartbeat established: node={advertise_addr} node_id={node_id} next_in={:?}",
                    heartbeat_period
                );
            } else {
                debug!(
                    "Heartbeat accepted by MetaServer: node={advertise_addr} node_id={node_id} next_in={:?}",
                    heartbeat_period
                );
            }
            heartbeat.node_registered = true;
            heartbeat.backoff_ms = INITIAL_BACKOFF_MS;
            Ok(heartbeat_period)
        }
        Err(e) => {
            warn!("MetaServer heartbeat failed: {e}");
            core_metrics().metaserver_heartbeat_failures.add(1, &[]);
            if e.code() == Code::FailedPrecondition {
                warn!(
                    "MetaServer heartbeat rejected current session; resetting node registration: node={advertise_addr} node_id={node_id}"
                );
                core_metrics().metaserver_session_resets.add(1, &[]);
                heartbeat.node_registered = false;
            }
            *client = None;
            Err(heartbeat.advance_backoff())
        }
    }
}

fn heartbeat_period_from_stale_after(stale_after_secs: u64) -> Duration {
    Duration::from_secs((stale_after_secs / 2).max(MIN_HEARTBEAT_INTERVAL_SECS))
}

impl HeartbeatState {
    fn advance_backoff(&mut self) -> Duration {
        let delay = Duration::from_millis(self.backoff_ms);
        self.backoff_ms = (self.backoff_ms * 2).min(MAX_BACKOFF_MS);
        delay
    }
}

async fn unregister_current_session(
    client: &mut Option<MetaServerGrpcClient<Channel>>,
    metaserver_addr: &str,
    endpoint: &Endpoint,
    advertise_addr: &str,
    node_id: &str,
) {
    if client.is_none() {
        match connect_metaserver_client(endpoint).await {
            Ok(c) => *client = Some(c),
            Err(e) => {
                warn!("Failed to connect to MetaServer for unregister at {metaserver_addr}: {e}");
                core_metrics().metaserver_unregister_failures.add(1, &[]);
                return;
            }
        }
    }

    let Some(c) = client.as_mut() else {
        return;
    };
    let request = UnregisterNodeRequest {
        node: advertise_addr.to_string(),
        node_id: node_id.to_string(),
    };
    match tokio::time::timeout(
        tokio::time::Duration::from_secs(UNREGISTER_TIMEOUT_SECS),
        c.unregister_node(request),
    )
    .await
    {
        Ok(Ok(resp)) => {
            debug!(
                "Unregistered MetaServer node session: node={} removed_owners={}",
                advertise_addr,
                resp.into_inner().removed_owners
            );
        }
        Ok(Err(e)) => {
            warn!("MetaServer unregister_node failed: {e}");
            core_metrics().metaserver_unregister_failures.add(1, &[]);
        }
        Err(_) => {
            warn!("MetaServer unregister_node timed out");
            core_metrics().metaserver_unregister_failures.add(1, &[]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockKey, SealedBlock};
    use pegaflow_proto::proto::engine::meta_server_server::{MetaServer, MetaServerServer};
    use pegaflow_proto::proto::engine::{
        HeartbeatNodeResponse, InsertBlockHashesResponse, QueryPrefixBlocksRequest,
        QueryPrefixBlocksResponse, RemoveBlockHashesResponse, ResponseStatus,
        UnregisterNodeResponse,
    };
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::net::TcpListener;
    use tokio::sync::{Notify, oneshot};
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{Request, Response, Status, async_trait};

    type RequestLog = Mutex<Vec<(String, Vec<Vec<u8>>)>>;
    type RequestSet = BTreeMap<String, Vec<Vec<u8>>>;

    #[derive(Default)]
    struct FakeMetaServerState {
        heartbeat_count: AtomicUsize,
        insert_count: AtomicUsize,
        remove_count: AtomicUsize,
        unregister_count: AtomicUsize,
        fail_insert_with_stale_session: AtomicUsize,
        reclaimable_hashes: Mutex<Vec<Vec<u8>>>,
        insert_requests: RequestLog,
        remove_requests: RequestLog,
        heartbeat_notify: Notify,
        insert_notify: Notify,
        remove_notify: Notify,
        unregister_notify: Notify,
    }

    #[derive(Clone)]
    struct FakeMetaServer {
        state: Arc<FakeMetaServerState>,
    }

    #[async_trait]
    impl MetaServer for FakeMetaServer {
        async fn heartbeat_node(
            &self,
            _request: Request<HeartbeatNodeRequest>,
        ) -> Result<Response<HeartbeatNodeResponse>, Status> {
            self.state.heartbeat_count.fetch_add(1, Ordering::SeqCst);
            self.state.heartbeat_notify.notify_waiters();
            Ok(Response::new(HeartbeatNodeResponse {
                stale_after_secs: 2,
            }))
        }

        async fn unregister_node(
            &self,
            _request: Request<UnregisterNodeRequest>,
        ) -> Result<Response<UnregisterNodeResponse>, Status> {
            self.state.unregister_count.fetch_add(1, Ordering::SeqCst);
            self.state.unregister_notify.notify_waiters();
            Ok(Response::new(UnregisterNodeResponse { removed_owners: 0 }))
        }

        async fn insert_block_hashes(
            &self,
            request: Request<InsertBlockHashesRequest>,
        ) -> Result<Response<InsertBlockHashesResponse>, Status> {
            let request = request.into_inner();
            self.state.insert_count.fetch_add(1, Ordering::SeqCst);
            if self
                .state
                .fail_insert_with_stale_session
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    (remaining > 0).then(|| remaining - 1)
                })
                .is_ok()
            {
                return Err(Status::failed_precondition("stale node session"));
            }
            let inserted_count = request.block_hashes.len() as u64;
            let reclaimable_hashes = self
                .state
                .reclaimable_hashes
                .lock()
                .unwrap()
                .iter()
                .filter(|hash| request.block_hashes.contains(hash))
                .cloned()
                .collect();
            self.state
                .insert_requests
                .lock()
                .unwrap()
                .push((request.namespace, request.block_hashes));
            self.state.insert_notify.notify_waiters();
            Ok(Response::new(InsertBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                inserted_count,
                reclaimable_hashes,
            }))
        }

        async fn remove_block_hashes(
            &self,
            request: Request<RemoveBlockHashesRequest>,
        ) -> Result<Response<RemoveBlockHashesResponse>, Status> {
            let request = request.into_inner();
            self.state.remove_count.fetch_add(1, Ordering::SeqCst);
            let removed_count = request.block_hashes.len() as u64;
            self.state
                .remove_requests
                .lock()
                .unwrap()
                .push((request.namespace, request.block_hashes));
            self.state.remove_notify.notify_waiters();
            Ok(Response::new(RemoveBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                removed_count,
            }))
        }

        async fn query_prefix_blocks(
            &self,
            _request: Request<QueryPrefixBlocksRequest>,
        ) -> Result<Response<QueryPrefixBlocksResponse>, Status> {
            Ok(Response::new(QueryPrefixBlocksResponse { nodes: vec![] }))
        }
    }

    async fn start_fake_metaserver() -> (String, Arc<FakeMetaServerState>, oneshot::Sender<()>) {
        let state = Arc::new(FakeMetaServerState::default());
        let service = FakeMetaServer {
            state: Arc::clone(&state),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let incoming = TcpListenerStream::new(listener);
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MetaServerServer::new(service))
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_rx.await;
                })
                .await
                .unwrap();
        });
        (format!("http://{addr}"), state, shutdown_tx)
    }

    fn collect_requests(requests: &RequestLog) -> RequestSet {
        let mut grouped = BTreeMap::new();
        for (namespace, hashes) in requests.lock().unwrap().iter() {
            let namespace_hashes: &mut Vec<Vec<u8>> = grouped.entry(namespace.clone()).or_default();
            namespace_hashes.extend(hashes.iter().cloned());
        }
        for hashes in grouped.values_mut() {
            hashes.sort();
        }
        grouped
    }

    fn expected_requests(entries: &[(&str, Vec<u8>)]) -> RequestSet {
        let mut grouped: RequestSet = BTreeMap::new();
        for (namespace, hash) in entries {
            grouped
                .entry((*namespace).to_string())
                .or_default()
                .push(hash.clone());
        }
        for hashes in grouped.values_mut() {
            hashes.sort();
        }
        grouped
    }

    async fn wait_for_count(notify: &Notify, count: &AtomicUsize, expected: usize) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            if count.load(Ordering::SeqCst) >= expected {
                return;
            }
            tokio::select! {
                _ = notify.notified() => {}
                _ = tokio::time::sleep_until(deadline) => {
                    panic!("timed out waiting for count {expected}");
                }
            }
        }
    }

    #[tokio::test]
    async fn heartbeat_loop_sends_initial_heartbeat_and_unregisters_on_shutdown() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Weak::new(),
        );

        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
        client.shutdown().await;
        wait_for_count(&service.unregister_notify, &service.unregister_count, 1).await;

        let _ = shutdown_tx.send(());
    }

    #[tokio::test]
    async fn stale_session_write_error_triggers_new_heartbeat() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        service
            .fail_insert_with_stale_session
            .store(1, Ordering::SeqCst);
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Weak::new(),
        );

        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
        client.try_register_namespace("ns".to_string(), vec![vec![1]]);
        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 2).await;

        client.shutdown().await;
        let _ = shutdown_tx.send(());
    }

    #[tokio::test]
    async fn flush_barrier_waits_for_prior_registrations() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Weak::new(),
        );

        client.try_register_namespace("ns".to_string(), vec![vec![1], vec![2]]);
        // On return, the insert enqueued above must have been delivered — no
        // wait_for_count polling; the barrier itself is the synchronization.
        client.flush().await;
        assert_eq!(
            collect_requests(&service.insert_requests),
            expected_requests(&[("ns", vec![1]), ("ns", vec![2])])
        );

        // Flush with an empty queue resolves promptly (no deadlock).
        client.flush().await;

        client.shutdown().await;
        // Flush after shutdown: loop has exited, must return, not hang.
        client.flush().await;
        let _ = shutdown_tx.send(());
    }

    #[tokio::test]
    async fn registration_applies_reclaimable_hashes() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        let read_cache = Arc::new(ReadCache::new(1 << 20, false, None));
        let hashes: Vec<Vec<u8>> = (0..=MAX_HASHES_PER_RPC as u32)
            .map(|value| value.to_le_bytes().to_vec())
            .collect();
        let hinted_hash = hashes[0].clone();
        let hinted_key = BlockKey::new("ns".to_string(), hinted_hash.clone());
        read_cache.insert_retained_for_test(
            hinted_key.clone(),
            Arc::new(SealedBlock::from_slots(Vec::new())),
        );
        *service.reclaimable_hashes.lock().unwrap() = vec![hinted_hash];
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Arc::downgrade(&read_cache),
        );

        client.try_register_namespace("ns".to_string(), hashes);
        client.flush().await;

        assert!(read_cache.is_reclaimable_for_test(&hinted_key));
        assert_eq!(service.insert_count.load(Ordering::SeqCst), 2);
        client.shutdown().await;
        let _ = shutdown_tx.send(());
        drop(service);
    }

    #[tokio::test]
    async fn flush_barrier_acks_even_when_insert_fails() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        // Fail both the insert attempt and the session-reset retry heartbeat
        // path's next insert, so the batch is dropped rather than delivered.
        service
            .fail_insert_with_stale_session
            .store(usize::MAX, Ordering::SeqCst);
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Weak::new(),
        );

        client.try_register_namespace("ns".to_string(), vec![vec![1]]);
        // The contract is "delivered or dropped": a failed insert drops the
        // batch and the flush must still resolve instead of hanging.
        client.flush().await;

        client.shutdown().await;
        let _ = shutdown_tx.send(());
    }

    #[test]
    fn unsent_after_failure_counts_only_unsent_tail() {
        let ns = |name: &str, n: usize| (name.to_string(), vec![vec![0u8]; n]);
        let namespaces = vec![ns("a", 100), ns("b", 50), ns("c", 30)];

        // First chunk of "b" failed (nothing of b sent yet): all of b + c.
        assert_eq!(unsent_after_failure(&namespaces, 1, 0), 50 + 30);
        // 40 hashes of "b" already landed before the failing chunk: b's tail + c.
        assert_eq!(unsent_after_failure(&namespaces, 1, 40), 10 + 30);
        // Failure in the last namespace: only its unsent tail, no later namespaces.
        assert_eq!(unsent_after_failure(&namespaces, 2, 20), 10);
        // Offset past the namespace length saturates to zero unsent.
        assert_eq!(unsent_after_failure(&namespaces, 2, 999), 0);
    }

    #[tokio::test]
    async fn large_namespace_removal_splits_into_bounded_rpcs() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        let client = MetaServerClient::new(
            MetaServerClientConfig::new(addr, "node-a:50055".to_string()),
            Weak::new(),
        );
        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;

        // One namespace coalesced past the per-RPC cap. Use sha256-sized (32B)
        // hashes so the full payload (~5 MiB) exceeds tonic's 4 MiB default
        // decode limit: without chunking this single RPC would be rejected.
        // Chunking holds each request to MAX_HASHES_PER_RPC * 32B = 512 KiB.
        let sha256_hash = |i: u32| {
            let mut h = vec![0u8; 32];
            h[..4].copy_from_slice(&i.to_le_bytes());
            h
        };
        let total = MAX_HASHES_PER_RPC * 10 + 1;
        let expected_chunks = total.div_ceil(MAX_HASHES_PER_RPC);
        let entries: Vec<(String, Vec<u8>)> = (0..total as u32)
            .map(|i| ("ns".to_string(), sha256_hash(i)))
            .collect();
        client.try_unregister(entries);

        wait_for_count(
            &service.remove_notify,
            &service.remove_count,
            expected_chunks,
        )
        .await;

        let logged = service.remove_requests.lock().unwrap().clone();
        assert_eq!(
            logged.len(),
            expected_chunks,
            "removal split into wrong RPC count"
        );
        let mut sent: Vec<Vec<u8>> = Vec::with_capacity(total);
        for (namespace, hashes) in &logged {
            assert_eq!(namespace, "ns");
            assert!(
                hashes.len() <= MAX_HASHES_PER_RPC,
                "chunk of {} hashes exceeds cap {MAX_HASHES_PER_RPC}",
                hashes.len()
            );
            sent.extend(hashes.iter().cloned());
        }
        sent.sort();
        let mut expected: Vec<Vec<u8>> = (0..total as u32).map(sha256_hash).collect();
        expected.sort();
        assert_eq!(sent, expected, "chunking lost or duplicated hashes");

        client.shutdown().await;
        let _ = shutdown_tx.send(());
    }

    #[tokio::test]
    async fn mixed_insert_remove_drain_preserves_last_write_per_namespace() {
        let (addr, service, shutdown_tx) = start_fake_metaserver().await;
        let (tx, rx) = mpsc::channel(16);

        let insert_then_remove = vec![0xa0];
        let remove_then_insert = vec![0xb0];
        let remove_only = vec![0xc0];
        let insert_only = vec![0xd0];

        tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
            "ns-first".to_string(),
            vec![insert_then_remove.clone()],
        )))
        .unwrap();
        tx.try_send(MetaServerCommand::Remove(BlockHashBatch::from_entries(
            vec![("ns-first".to_string(), insert_then_remove.clone())],
        )))
        .unwrap();
        tx.try_send(MetaServerCommand::Remove(BlockHashBatch::from_entries(
            vec![
                ("ns-second".to_string(), remove_then_insert.clone()),
                ("ns-remove".to_string(), remove_only.clone()),
            ],
        )))
        .unwrap();
        tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
            "ns-second".to_string(),
            vec![remove_then_insert.clone()],
        )))
        .unwrap();
        tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
            "ns-insert".to_string(),
            vec![insert_only.clone()],
        )))
        .unwrap();

        let endpoint = metaserver_endpoint(addr.clone());
        let loop_task = tokio::spawn(registration_loop(
            rx,
            addr,
            endpoint,
            "node-a:50055".to_string(),
            Weak::new(),
        ));

        wait_for_count(&service.insert_notify, &service.insert_count, 2).await;
        wait_for_count(&service.remove_notify, &service.remove_count, 2).await;

        assert_eq!(
            collect_requests(&service.insert_requests),
            expected_requests(&[
                ("ns-second", remove_then_insert),
                ("ns-insert", insert_only),
            ])
        );
        assert_eq!(
            collect_requests(&service.remove_requests),
            expected_requests(&[("ns-first", insert_then_remove), ("ns-remove", remove_only)])
        );

        let (done_tx, done_rx) = oneshot::channel();
        tx.send(MetaServerCommand::Shutdown(done_tx)).await.unwrap();
        done_rx.await.unwrap();
        loop_task.await.unwrap();
        let _ = shutdown_tx.send(());
    }
}
