use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::InsertBlockHashesRequest;
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use tokio::sync::mpsc;
use tonic::transport::Channel;

use crate::metrics::core_metrics;

const DEFAULT_QUEUE_DEPTH: usize = 16;

pub struct MetaServerRegistrarConfig {
    pub metaserver_addr: String,
    pub advertise_addr: String,
    pub queue_depth: usize,
}

impl MetaServerRegistrarConfig {
    pub fn new(metaserver_addr: String, advertise_addr: String) -> Self {
        Self {
            metaserver_addr,
            advertise_addr,
            queue_depth: DEFAULT_QUEUE_DEPTH,
        }
    }
}

struct RegistrationBatch {
    namespace: String,
    block_hashes: Vec<Vec<u8>>,
}

pub struct MetaServerRegistrar {
    tx: mpsc::Sender<RegistrationBatch>,
}

impl MetaServerRegistrar {
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
    pub(crate) fn try_register(&self, namespace: String, block_hashes: Vec<Vec<u8>>) {
        if block_hashes.is_empty() {
            return;
        }
        let count = block_hashes.len();
        let batch = RegistrationBatch {
            namespace,
            block_hashes,
        };
        if self.tx.try_send(batch).is_ok() {
            core_metrics()
                .metaserver_registration_blocks
                .add(count as u64, &[]);
        } else {
            warn!(
                "MetaServer registration queue full, dropping {} hashes",
                count
            );
            core_metrics()
                .metaserver_registration_queue_full
                .add(count as u64, &[]);
        }
    }
}

async fn registration_loop(
    mut rx: mpsc::Receiver<RegistrationBatch>,
    metaserver_addr: String,
    advertise_addr: String,
) {
    let mut client: Option<MetaServerClient<Channel>> = None;

    while let Some(batch) = rx.recv().await {
        // Drain all pending batches for batching
        let mut batches = vec![batch];
        while let Ok(more) = rx.try_recv() {
            batches.push(more);
        }

        // Group by namespace
        let mut grouped: std::collections::HashMap<String, Vec<Vec<u8>>> =
            std::collections::HashMap::new();
        for b in batches {
            grouped
                .entry(b.namespace)
                .or_default()
                .extend(b.block_hashes);
        }

        // Lazy-connect
        if client.is_none() {
            match MetaServerClient::connect(metaserver_addr.clone()).await {
                Ok(c) => {
                    info!("Connected to MetaServer at {}", metaserver_addr);
                    client = Some(c);
                }
                Err(e) => {
                    error!("Failed to connect to MetaServer: {e}");
                    let total: usize = grouped.values().map(|v| v.len()).sum();
                    core_metrics()
                        .metaserver_registration_failures
                        .add(total as u64, &[]);
                    continue;
                }
            }
        }

        let c = client.as_mut().unwrap();

        for (namespace, hashes) in grouped {
            let count = hashes.len();
            let request = InsertBlockHashesRequest {
                namespace: namespace.clone(),
                block_hashes: hashes,
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
                    core_metrics()
                        .metaserver_registration_failures
                        .add(count as u64, &[]);
                    // Reset client to force reconnect on next batch
                    client = None;
                    break;
                }
            }
        }
    }

    info!("MetaServer registration loop shutting down");
}
