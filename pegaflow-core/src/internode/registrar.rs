use std::collections::HashMap;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::InsertBlockHashesRequest;
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use tokio::sync::mpsc;
use tonic::transport::Channel;

use crate::metrics::core_metrics;

pub const DEFAULT_METASERVER_QUEUE_DEPTH: usize = 256;

const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 30_000;

pub struct MetaServerRegistrarConfig {
    metaserver_addr: String,
    advertise_addr: String,
    queue_depth: usize,
}

impl MetaServerRegistrarConfig {
    pub const fn new(metaserver_addr: String, advertise_addr: String) -> Self {
        Self {
            metaserver_addr,
            advertise_addr,
            queue_depth: DEFAULT_METASERVER_QUEUE_DEPTH,
        }
    }

    pub const fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth;
        self
    }
}

/// A batch of (namespace, block_hash) pairs to register with MetaServer.
/// Sent as a single channel message to avoid consuming multiple queue slots.
struct RegistrationBatch {
    entries: Vec<(String, Vec<u8>)>,
}

pub struct MetaServerRegistrar {
    tx: mpsc::Sender<RegistrationBatch>,
}

impl MetaServerRegistrar {
    /// Create a new registrar and spawn the background registration loop.
    ///
    /// Must be called from within a tokio runtime context.
    pub fn new(config: MetaServerRegistrarConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_depth);

        tokio::spawn(registration_loop(
            rx,
            config.metaserver_addr,
            config.advertise_addr,
        ));

        info!(
            "MetaServer registrar started (queue_depth={})",
            config.queue_depth
        );

        Self { tx }
    }

    /// Fire-and-forget registration. Mirrors `SsdBackingStore::ingest_batch()`.
    ///
    /// Accepts a flat list of (namespace, block_hash) pairs. The background loop
    /// groups by namespace before issuing gRPC calls.
    pub(crate) fn try_register(&self, entries: Vec<(String, Vec<u8>)>) {
        if entries.is_empty() {
            return;
        }
        let count = entries.len();
        let batch = RegistrationBatch { entries };
        match self.tx.try_send(batch) {
            Ok(()) => {
                core_metrics()
                    .metaserver_registration_blocks
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("MetaServer registration queue full, dropping {count} hashes");
                core_metrics()
                    .metaserver_registration_queue_full
                    .add(count as u64, &[]);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!("MetaServer registration loop has exited, dropping {count} hashes");
                core_metrics()
                    .metaserver_registration_queue_full
                    .add(count as u64, &[]);
            }
        }
    }
}

async fn registration_loop(
    mut rx: mpsc::Receiver<RegistrationBatch>,
    metaserver_addr: String,
    advertise_addr: String,
) {
    let mut client: Option<MetaServerClient<Channel>> = None;
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
            match MetaServerClient::connect(metaserver_addr.clone()).await {
                Ok(c) => {
                    info!("Connected to MetaServer at {metaserver_addr}");
                    client = Some(c);
                    backoff_ms = INITIAL_BACKOFF_MS;
                }
                Err(e) => {
                    error!("Failed to connect to MetaServer: {e}");
                    let total: usize = grouped.values().map(Vec::len).sum();
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

        // Collect into Vec for indexed iteration so we can count remaining on failure
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
                        "Registered {count} block hashes with MetaServer (namespace={namespace}, inserted={})",
                        inner.inserted_count
                    );
                }
                Err(e) => {
                    error!(
                        "MetaServer insert_block_hashes failed (namespace={namespace}, count={count}): {e}"
                    );
                    failed_at = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = failed_at {
            // Count all hashes from the failed namespace onward as failures
            let dropped: usize = namespaces[idx..].iter().map(|(_, h)| h.len()).sum();
            core_metrics()
                .metaserver_registration_failures
                .add(dropped as u64, &[]);
            client = None;
        }
    }

    info!("MetaServer registration loop shutting down");
}
