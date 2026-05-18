use std::collections::HashMap;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient as MetaServerGrpcClient;
use pegaflow_proto::proto::engine::{
    HeartbeatNodeRequest, InsertBlockHashesRequest, NodePrefixResult, QueryPrefixBlocksRequest,
    RemoveBlockHashesRequest, UnregisterNodeRequest,
};
use tokio::sync::{mpsc, oneshot};
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
const HEARTBEAT_INTERVAL_SECS: u64 = 10;
const UNREGISTER_TIMEOUT_SECS: u64 = 3;

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

/// A batch of (namespace, block_hash) pairs for MetaServer operations.
struct BlockHashBatch {
    entries: Vec<(String, Vec<u8>)>,
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
    pub(crate) fn try_register(&self, entries: Vec<(String, Vec<u8>)>) {
        if entries.is_empty() {
            return;
        }
        let count = entries.len();
        let batch = BlockHashBatch { entries };
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
        let count = entries.len();
        let batch = BlockHashBatch { entries };
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
    let mut node_registered = false;
    let mut backoff_ms: u64 = INITIAL_BACKOFF_MS;
    let heartbeat_period = tokio::time::Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + heartbeat_period,
        heartbeat_period,
    );
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        let cmd = tokio::select! {
            cmd = rx.recv() => match cmd {
                Some(cmd) => cmd,
                None => break,
            },
            _ = heartbeat.tick() => {
                if send_heartbeat(
                    &mut client,
                    &mut node_registered,
                    &metaserver_addr,
                    &advertise_addr,
                    &node_id,
                    &mut backoff_ms,
                ).await.is_err() {
                    continue;
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

        // Drain all pending commands. For each (namespace, hash), keep only the last
        // operation (last-write-wins), so [Remove(X), Insert(X)] correctly resolves
        // to Insert(X) rather than being reversed by separate-bucket processing.
        let mut net: HashMap<(String, Vec<u8>), bool> = HashMap::new(); // true=insert, false=remove

        for cmd in std::iter::once(cmd).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
            match cmd {
                MetaServerCommand::Insert(batch) => {
                    for (ns, hash) in batch.entries {
                        net.insert((ns, hash), true);
                    }
                }
                MetaServerCommand::Remove(batch) => {
                    for (ns, hash) in batch.entries {
                        net.insert((ns, hash), false);
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

        let mut inserts: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut removes: HashMap<String, Vec<Vec<u8>>> = HashMap::new();

        for ((ns, hash), is_insert) in net {
            if is_insert {
                inserts.entry(ns).or_default().push(hash);
            } else {
                removes.entry(ns).or_default().push(hash);
            }
        }

        let insert_total: usize = inserts.values().map(|v| v.len()).sum();
        let remove_total: usize = removes.values().map(|v| v.len()).sum();
        if ensure_heartbeat_registered(
            &mut client,
            &mut node_registered,
            &metaserver_addr,
            &advertise_addr,
            &node_id,
            &mut backoff_ms,
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
                        core_metrics().metaserver_session_resets.add(1, &[]);
                        node_registered = false;
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
                        core_metrics().metaserver_session_resets.add(1, &[]);
                        node_registered = false;
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

async fn ensure_heartbeat_registered(
    client: &mut Option<MetaServerGrpcClient<Channel>>,
    node_registered: &mut bool,
    metaserver_addr: &str,
    advertise_addr: &str,
    node_id: &str,
    backoff_ms: &mut u64,
) -> Result<(), ()> {
    if *node_registered {
        return Ok(());
    }

    send_heartbeat(
        client,
        node_registered,
        metaserver_addr,
        advertise_addr,
        node_id,
        backoff_ms,
    )
    .await
}

async fn send_heartbeat(
    client: &mut Option<MetaServerGrpcClient<Channel>>,
    node_registered: &mut bool,
    metaserver_addr: &str,
    advertise_addr: &str,
    node_id: &str,
    backoff_ms: &mut u64,
) -> Result<(), ()> {
    if client.is_none() {
        match MetaServerGrpcClient::connect(metaserver_addr.to_string()).await {
            Ok(c) => {
                info!("Connected to MetaServer at {}", metaserver_addr);
                *client = Some(c);
                *backoff_ms = INITIAL_BACKOFF_MS;
            }
            Err(e) => {
                error!("Failed to connect to MetaServer: {e}");
                core_metrics().metaserver_heartbeat_failures.add(1, &[]);
                tokio::time::sleep(tokio::time::Duration::from_millis(*backoff_ms)).await;
                *backoff_ms = (*backoff_ms * 2).min(MAX_BACKOFF_MS);
                return Err(());
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
        Ok(_) => {
            debug!("Heartbeat accepted by MetaServer: node={advertise_addr} node_id={node_id}");
            *node_registered = true;
            *backoff_ms = INITIAL_BACKOFF_MS;
            Ok(())
        }
        Err(e) => {
            warn!("MetaServer heartbeat failed: {e}");
            core_metrics().metaserver_heartbeat_failures.add(1, &[]);
            if e.code() == Code::FailedPrecondition {
                core_metrics().metaserver_session_resets.add(1, &[]);
                *node_registered = false;
            }
            *client = None;
            tokio::time::sleep(tokio::time::Duration::from_millis(*backoff_ms)).await;
            *backoff_ms = (*backoff_ms * 2).min(MAX_BACKOFF_MS);
            Err(())
        }
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
                "Unregistered MetaServer node session: node={} removed_keys={}",
                advertise_addr,
                resp.into_inner().removed_keys
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
