use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use log::{debug, error, info, warn};
use tokio::runtime::Handle;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tonic::transport::{Channel, Endpoint};

use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use pegaflow_proto::proto::engine::{InsertBlockHashesRequest, QueryBlockHashesRequest};

use crate::block::{BlockKey, SealedBlock};

use super::{BackingStore, BackingStoreKind, BakingStoreConfig, PrefetchResult};

/// Timeout for a single metaserver query RPC.
const QUERY_TIMEOUT: Duration = Duration::from_millis(200);

// ============================================================================
// Insert actor command
// ============================================================================

struct InsertCmd {
    namespace: String,
    block_hashes: Vec<Vec<u8>>,
}

// ============================================================================
// P2pBakingStore
// ============================================================================

pub(crate) struct P2pBakingStore {
    insert_tx: UnboundedSender<InsertCmd>,
    /// gRPC client for queries. Channel connects lazily on first RPC.
    query_client: MetaServerClient<Channel>,
}

impl P2pBakingStore {
    fn create(config: BakingStoreConfig) -> Option<Arc<dyn BackingStore>> {
        let handle = match Handle::try_current() {
            Ok(handle) => handle,
            Err(err) => {
                error!(
                    "Failed to initialize P2P baking store: no Tokio runtime available: {}",
                    err
                );
                return None;
            }
        };

        let (insert_tx, insert_rx) = mpsc::unbounded_channel();

        let coordinator = config.p2p_coordinator_addr.clone();
        let node = config.p2p_node_addr.clone();
        handle.spawn(insert_actor(insert_rx, coordinator.clone(), node));

        // Lazy channel: no TCP connection until the first RPC call.
        let channel = Endpoint::from_shared(coordinator)
            .expect("invalid coordinator address")
            .connect_lazy();
        let query_client = MetaServerClient::new(channel);

        info!(
            "P2P baking store configured (coordinator={}, node={})",
            config.p2p_coordinator_addr, config.p2p_node_addr
        );

        Some(Arc::new(Self {
            insert_tx,
            query_client,
        }))
    }
}

impl BackingStore for P2pBakingStore {
    fn kind(&self) -> BackingStoreKind {
        BackingStoreKind::P2p
    }

    fn ingest_batch(&self, blocks: Vec<(BlockKey, Weak<SealedBlock>)>) {
        if blocks.is_empty() {
            return;
        }
        let mut grouped: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for (key, _) in blocks {
            grouped.entry(key.namespace).or_default().push(key.hash);
        }
        for (namespace, block_hashes) in grouped {
            let _ = self.insert_tx.send(InsertCmd {
                namespace,
                block_hashes,
            });
        }
    }

    fn submit_prefix(&self, keys: Vec<BlockKey>) -> (usize, oneshot::Receiver<PrefetchResult>) {
        if !keys.is_empty() {
            let mut grouped: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
            for key in &keys {
                grouped
                    .entry(key.namespace.clone())
                    .or_default()
                    .push(key.hash.clone());
            }

            // Inline query: block_in_place lets tokio reschedule other tasks
            // off this worker so the runtime isn't starved.
            tokio::task::block_in_place(|| {
                Handle::current().block_on(async {
                    for (namespace, block_hashes) in grouped {
                        self.do_query(&namespace, block_hashes).await;
                    }
                });
            });
        }

        // No actual data pull — return 0 so the scheduler falls through to SSD.
        let (done_tx, done_rx) = oneshot::channel();
        let _ = done_tx.send(Vec::new());
        (0, done_rx)
    }
}

impl P2pBakingStore {
    async fn do_query(&self, namespace: &str, block_hashes: Vec<Vec<u8>>) {
        let queried = block_hashes.len();
        let mut client = self.query_client.clone();

        let req = QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes,
        };
        let t = std::time::Instant::now();
        match tokio::time::timeout(QUERY_TIMEOUT, client.query_block_hashes(req)).await {
            Ok(Ok(response)) => {
                let resp = response.into_inner();
                let nodes: Vec<(&str, usize)> = resp
                    .node_blocks
                    .iter()
                    .map(|nb| (nb.node.as_str(), nb.block_hashes.len()))
                    .collect();
                info!(
                    "p2p query: ns={} queried={} found={} rpc={:.1}ms nodes={:?}",
                    namespace,
                    queried,
                    resp.found_count,
                    t.elapsed().as_secs_f64() * 1000.0,
                    nodes,
                );
            }
            Ok(Err(err)) => {
                warn!("p2p query rpc failed: {}", err);
            }
            Err(_) => {
                warn!("p2p query timed out: ns={} queried={}", namespace, queried);
            }
        }
    }
}

// ============================================================================
// Insert actor (fire-and-forget, batched)
// ============================================================================

async fn insert_actor(
    mut rx: UnboundedReceiver<InsertCmd>,
    coordinator_addr: String,
    node_addr: String,
) {
    let mut client: Option<MetaServerClient<Channel>> = None;

    while let Some(cmd) = rx.recv().await {
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        let mut by_ns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for cmd in cmds {
            by_ns
                .entry(cmd.namespace)
                .or_default()
                .extend(cmd.block_hashes);
        }

        for (namespace, block_hashes) in by_ns {
            if client.is_none() {
                client = connect_client(&coordinator_addr).await;
            }
            let Some(c) = client.as_mut() else {
                continue;
            };

            let count = block_hashes.len();
            let req = InsertBlockHashesRequest {
                namespace: namespace.clone(),
                block_hashes,
                node: node_addr.clone(),
            };
            let t = std::time::Instant::now();
            match c.insert_block_hashes(req).await {
                Ok(resp) => {
                    debug!(
                        "p2p insert: ns={} sent={} inserted={} rpc={:.1}ms",
                        namespace,
                        count,
                        resp.into_inner().inserted_count,
                        t.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                Err(err) => {
                    warn!(
                        "p2p insert failed after {:.1}ms: {}",
                        t.elapsed().as_secs_f64() * 1000.0,
                        err,
                    );
                    client = None;
                }
            }
        }
    }

    info!("p2p insert actor shutting down");
}

// ============================================================================
// Shared helpers
// ============================================================================

async fn connect_client(addr: &str) -> Option<MetaServerClient<Channel>> {
    match MetaServerClient::connect(addr.to_string()).await {
        Ok(client) => Some(client),
        Err(err) => {
            warn!("Failed to connect P2P coordinator at {}: {}", addr, err);
            None
        }
    }
}

pub(super) fn new_p2p(config: BakingStoreConfig) -> Option<Arc<dyn BackingStore>> {
    P2pBakingStore::create(config)
}
