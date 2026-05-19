use std::collections::HashMap;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient as MetaServerGrpcClient;
use pegaflow_proto::proto::engine::{
    HeartbeatNodeRequest, InsertBlockHashesRequest, NodePrefixResult, QueryPrefixBlocksRequest,
    RemoveBlockHashesRequest, UnregisterNodeRequest,
};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, Instant};
use tonic::Code;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

use crate::metrics::core_metrics;

pub const DEFAULT_METASERVER_QUEUE_DEPTH: usize = 256;

/// Error type for MetaServer client operations.
#[derive(Debug)]
pub(crate) enum ClientError {
    RpcFailed(String),
}

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
    Shutdown(oneshot::Sender<()>),
}

/// Unified MetaServer client handling both insert (fire-and-forget) and query (direct RPC).
pub struct MetaServerClient {
    /// Fire-and-forget command channel for insert/remove operations.
    command_tx: mpsc::Sender<MetaServerCommand>,
    /// Lazy-connect query client
    query_client: MetaServerGrpcClient<Channel>,
}

impl MetaServerClient {
    /// Create a new client and spawn the background registration loop.
    ///
    /// Must be called from within a tokio runtime context.
    pub fn new(config: MetaServerClientConfig) -> Self {
        let (command_tx, rx) = mpsc::channel(config.queue_depth);

        tokio::spawn(registration_loop(
            rx,
            config.metaserver_addr.clone(),
            config.advertise_addr,
        ));

        // Lazy-connect query client: connects on first RPC, not here
        let channel = Endpoint::from_shared(config.metaserver_addr.clone())
            .expect("valid metaserver_addr URI")
            .connect_lazy();
        let query_client = MetaServerGrpcClient::new(channel);

        info!(
            "MetaServer client started (queue_depth={}, addr={})",
            config.queue_depth, config.metaserver_addr
        );

        Self {
            command_tx,
            query_client,
        }
    }

    /// Fire-and-forget registration of block hashes.
    ///
    /// Accepts a flat list of (namespace, block_hash) pairs. The background loop
    /// groups by namespace before issuing gRPC calls.
    pub(crate) fn try_register_namespace(&self, namespace: String, hashes: Vec<Vec<u8>>) {
        if hashes.is_empty() {
            return;
        }
        self.try_send_register_batch(BlockHashBatch::single_namespace(namespace, hashes));
    }

    fn try_send_register_batch(&self, batch: BlockHashBatch) {
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

    /// Query MetaServer for the longest prefix of blocks that exist remotely.
    /// Returns per-node prefix lengths.
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
    advertise_addr: String,
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
            unregister_current_session(&mut client, &metaserver_addr, &advertise_addr, &node_id)
                .await;
            let _ = done.send(());
            break;
        }

        let mut inserts: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut removes: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut mixed_ops: Option<HashMap<(String, Vec<u8>), bool>> = None; // true=insert
        let mut saw_insert = false;
        let mut saw_remove = false;

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
                MetaServerCommand::Shutdown(done) => {
                    unregister_current_session(
                        &mut client,
                        &metaserver_addr,
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
            continue;
        }

        let c = client.as_mut().expect("client is Some after lazy-connect");

        // Process inserts
        let insert_namespaces: Vec<(String, Vec<Vec<u8>>)> = inserts.into_iter().collect();
        let mut insert_failed_at = None;

        for (i, (namespace, hashes)) in insert_namespaces.iter().enumerate() {
            let count = hashes.len();
            let request = InsertBlockHashesRequest {
                namespace: namespace.clone(),
                block_hashes: hashes.clone(),
                node: advertise_addr.clone(),
                node_id: node_id.clone(),
            };

            match c.insert_block_hashes(request).await {
                Ok(resp) => {
                    let inner = resp.into_inner();
                    debug!(
                        "Registered {} block hashes with MetaServer (namespace={}, inserted={})",
                        count, namespace, inner.inserted_count
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
                    insert_failed_at = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = insert_failed_at {
            let dropped: usize = insert_namespaces[idx..].iter().map(|(_, h)| h.len()).sum();
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
            continue;
        }

        // Process removes
        let remove_namespaces: Vec<(String, Vec<Vec<u8>>)> = removes.into_iter().collect();
        let mut remove_failed_at = None;

        for (i, (namespace, hashes)) in remove_namespaces.iter().enumerate() {
            let count = hashes.len();
            let request = RemoveBlockHashesRequest {
                namespace: namespace.clone(),
                block_hashes: hashes.clone(),
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
                    remove_failed_at = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = remove_failed_at {
            let dropped: usize = remove_namespaces[idx..].iter().map(|(_, h)| h.len()).sum();
            core_metrics()
                .metaserver_removal_failures
                .add(dropped as u64, &[]);
            client = None;
        }
    }

    info!("MetaServer registration loop shutting down");
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

    match send_heartbeat(client, heartbeat, metaserver_addr, advertise_addr, node_id).await {
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
    advertise_addr: &str,
    node_id: &str,
) -> Result<Duration, Duration> {
    if client.is_none() {
        match MetaServerGrpcClient::connect(metaserver_addr.to_string()).await {
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
    advertise_addr: &str,
    node_id: &str,
) {
    if client.is_none() {
        match MetaServerGrpcClient::connect(metaserver_addr.to_string()).await {
            Ok(c) => *client = Some(c),
            Err(e) => {
                warn!("Failed to connect to MetaServer for unregister: {e}");
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
    use pegaflow_proto::proto::engine::meta_server_server::{MetaServer, MetaServerServer};
    use pegaflow_proto::proto::engine::{
        HeartbeatNodeResponse, InsertBlockHashesResponse, QueryPrefixBlocksResponse,
        RemoveBlockHashesResponse, ResponseStatus, UnregisterNodeResponse,
    };
    use std::net::SocketAddr;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::net::TcpListener;
    use tokio::sync::{Notify, oneshot};
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{Request, Response, Status, async_trait};

    #[derive(Default)]
    struct FakeMetaServerState {
        heartbeat_count: AtomicUsize,
        insert_count: AtomicUsize,
        unregister_count: AtomicUsize,
        fail_insert_with_stale_session: AtomicUsize,
        heartbeat_notify: Notify,
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
            _request: Request<InsertBlockHashesRequest>,
        ) -> Result<Response<InsertBlockHashesResponse>, Status> {
            self.state.insert_count.fetch_add(1, Ordering::SeqCst);
            if self
                .state
                .fail_insert_with_stale_session
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    (remaining > 0).then_some(remaining - 1)
                })
                .is_ok()
            {
                return Err(Status::failed_precondition("stale node session"));
            }
            Ok(Response::new(InsertBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                inserted_count: 1,
            }))
        }

        async fn remove_block_hashes(
            &self,
            _request: Request<RemoveBlockHashesRequest>,
        ) -> Result<Response<RemoveBlockHashesResponse>, Status> {
            Ok(Response::new(RemoveBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                removed_count: 1,
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
        let client = MetaServerClient::new(MetaServerClientConfig::new(
            addr,
            "node-a:50055".to_string(),
        ));

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
        let client = MetaServerClient::new(MetaServerClientConfig::new(
            addr,
            "node-a:50055".to_string(),
        ));

        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
        client.try_register_namespace("ns".to_string(), vec![vec![1]]);
        wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 2).await;

        client.shutdown().await;
        let _ = shutdown_tx.send(());
    }
}
