use std::collections::HashMap;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient as MetaServerGrpcClient;
use pegaflow_proto::proto::engine::{
    InsertBlockHashesRequest, NodeBlockHashes, QueryBlockHashesRequest,
};
use tokio::sync::mpsc;
use tonic::transport::{Channel, Endpoint};

use crate::internode::types::ClientError;
use crate::metrics::core_metrics;

pub const DEFAULT_METASERVER_QUEUE_DEPTH: usize = 256;

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

/// A batch of (namespace, block_hash) pairs to register with MetaServer.
struct RegistrationBatch {
    entries: Vec<(String, Vec<u8>)>,
}

/// Unified MetaServer client handling both insert (fire-and-forget) and query (direct RPC).
pub struct MetaServerClient {
    /// Fire-and-forget insert channel (same pattern as old registrar)
    insert_tx: mpsc::Sender<RegistrationBatch>,
    /// Lazy-connect query client
    query_client: MetaServerGrpcClient<Channel>,
}

impl MetaServerClient {
    /// Create a new client and spawn the background registration loop.
    ///
    /// Must be called from within a tokio runtime context.
    pub fn new(config: MetaServerClientConfig) -> Self {
        let (insert_tx, rx) = mpsc::channel(config.queue_depth);

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
            insert_tx,
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
        let batch = RegistrationBatch { entries };
        match self.insert_tx.try_send(batch) {
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

    /// Query MetaServer for block locations. Returns node-grouped results.
    pub(crate) async fn query(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<NodeBlockHashes>, ClientError> {
        let request = QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: hashes.to_vec(),
        };

        let response = self
            .query_client
            .clone()
            .query_block_hashes(request)
            .await
            .map_err(|e| ClientError::RpcFailed(format!("MetaServer query failed: {e}")))?;

        let resp = response.into_inner();
        if let Some(status) = &resp.status
            && !status.ok
        {
            return Err(ClientError::ResponseError(format!(
                "MetaServer query error: {}",
                status.message
            )));
        }

        debug!(
            "MetaServer query: namespace={} queried={} found={}",
            namespace, resp.total_queried, resp.found_count
        );

        Ok(resp.node_blocks)
    }
}

async fn registration_loop(
    mut rx: mpsc::Receiver<RegistrationBatch>,
    metaserver_addr: String,
    advertise_addr: String,
) {
    let mut client: Option<MetaServerGrpcClient<Channel>> = None;
    let mut backoff_ms: u64 = INITIAL_BACKOFF_MS;

    while let Some(batch) = rx.recv().await {
        // Drain all pending batches for coalescing
        let mut all_entries = batch.entries;
        while let Ok(more) = rx.try_recv() {
            all_entries.extend(more.entries);
        }

        // Group by namespace
        let mut grouped: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for (namespace, hash) in all_entries {
            grouped.entry(namespace).or_default().push(hash);
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
                    let total: usize = grouped.values().map(|v| v.len()).sum();
                    core_metrics()
                        .metaserver_registration_failures
                        .add(total as u64, &[]);
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                    continue;
                }
            }
        }

        let c = client.as_mut().expect("client is Some after lazy-connect");

        let namespaces: Vec<(String, Vec<Vec<u8>>)> = grouped.into_iter().collect();
        let mut failed_at = None;

        for (i, (namespace, hashes)) in namespaces.iter().enumerate() {
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
                    failed_at = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = failed_at {
            let dropped: usize = namespaces[idx..].iter().map(|(_, h)| h.len()).sum();
            core_metrics()
                .metaserver_registration_failures
                .add(dropped as u64, &[]);
            client = None;
        }
    }

    info!("MetaServer registration loop shutting down");
}
