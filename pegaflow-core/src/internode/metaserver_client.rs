use std::collections::HashMap;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient as MetaServerGrpcClient;
use pegaflow_proto::proto::engine::{
    InsertBlockHashesRequest, NodePrefixResult, QueryPrefixBlocksRequest,
    RemoveBlockHashesRequest,
};
use tokio::sync::mpsc;
use tonic::transport::{Channel, Endpoint};

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
    /// holds these blocks. Losing an occasional remove message is acceptable —
    /// TTL still serves as the ultimate fallback for node failures.
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
                warn!(
                    "MetaServer removal queue full, dropping {} hashes",
                    count
                );
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
    let mut backoff_ms: u64 = INITIAL_BACKOFF_MS;

    while let Some(cmd) = rx.recv().await {
        // Drain all pending commands for coalescing, separating inserts from removes.
        let mut inserts: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let mut removes: HashMap<String, Vec<Vec<u8>>> = HashMap::new();

        for cmd in std::iter::once(cmd).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
            match cmd {
                MetaServerCommand::Insert(batch) => {
                    for (ns, hash) in batch.entries {
                        inserts.entry(ns).or_default().push(hash);
                    }
                }
                MetaServerCommand::Remove(batch) => {
                    for (ns, hash) in batch.entries {
                        removes.entry(ns).or_default().push(hash);
                    }
                }
            }
        }

        // Lazy-connect with exponential backoff
        if client.is_none() {
            match MetaServerGrpcClient::connect(metaserver_addr.clone()).await {
                Ok(c) => {
                    info!("Connected to MetaServer at {}", metaserver_addr);
                    client = Some(c);
                    backoff_ms = INITIAL_BACKOFF_MS;
                }
                Err(e) => {
                    error!("Failed to connect to MetaServer: {e}");
                    let insert_total: usize = inserts.values().map(|v| v.len()).sum();
                    let remove_total: usize = removes.values().map(|v| v.len()).sum();
                    core_metrics()
                        .metaserver_registration_failures
                        .add(insert_total as u64, &[]);
                    core_metrics()
                        .metaserver_removal_failures
                        .add(remove_total as u64, &[]);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                    continue;
                }
            }
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
            client = None;
            // Re-borrow after setting client = None would fail, skip removes this round.
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
