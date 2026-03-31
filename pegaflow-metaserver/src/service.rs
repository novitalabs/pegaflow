use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    HealthRequest, HealthResponse, InsertBlockHashesRequest, InsertBlockHashesResponse,
    NodePrefixResult, QueryPrefixBlocksRequest, QueryPrefixBlocksResponse, ResponseStatus,
    ShutdownRequest, ShutdownResponse,
};
use crate::store::BlockHashStore;
use log::{debug, info};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;
use tonic::{Request, Response, Status, async_trait};

#[derive(Clone)]
pub struct GrpcMetaService {
    store: Arc<BlockHashStore>,
    shutdown: Arc<Notify>,
}

impl GrpcMetaService {
    pub fn new(store: Arc<BlockHashStore>, shutdown: Arc<Notify>) -> Self {
        Self { store, shutdown }
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn error_status(message: String) -> ResponseStatus {
        ResponseStatus { ok: false, message }
    }
}

#[async_trait]
impl MetaServer for GrpcMetaService {
    async fn insert_block_hashes(
        &self,
        request: Request<InsertBlockHashesRequest>,
    ) -> Result<Response<InsertBlockHashesResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let all_hashes: Vec<String> = req
            .block_hashes
            .iter()
            .map(|h| {
                h.iter()
                    .take(8)
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>()
            })
            .collect();
        debug!(
            "RPC [insert_block_hashes]: namespace={} node={} hashes_count={} hashes={:?}",
            req.namespace,
            req.node,
            req.block_hashes.len(),
            all_hashes
        );

        // Validate request
        if req.block_hashes.is_empty() {
            let response = InsertBlockHashesResponse {
                status: Some(Self::error_status(
                    "block_hashes cannot be empty".to_string(),
                )),
                inserted_count: 0,
            };
            return Ok(Response::new(response));
        }

        // Insert hashes with node ownership
        let inserted = self
            .store
            .insert_hashes(&req.namespace, &req.block_hashes, &req.node)
            .await;

        let elapsed = start.elapsed();
        info!(
            "RPC [insert_block_hashes]: namespace={} node={} inserted={} hashes in {:?} (store_entries={})",
            req.namespace, req.node, inserted, elapsed, self.store.entry_count()
        );

        let response = InsertBlockHashesResponse {
            status: Some(Self::ok_status()),
            inserted_count: inserted as u64,
        };

        Ok(Response::new(response))
    }

    /// Given an ordered list of block hashes, find the longest contiguous prefix
    /// that exists in the store (stop at the first hash no node owns), then for
    /// each node return how many consecutive hashes from h0 it holds.
    async fn query_prefix_blocks(
        &self,
        request: Request<QueryPrefixBlocksRequest>,
    ) -> Result<Response<QueryPrefixBlocksResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let all_hashes: Vec<String> = req
            .block_hashes
            .iter()
            .map(|h| {
                h.iter()
                    .take(8)
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>()
            })
            .collect();
        debug!(
            "RPC [query_prefix_blocks]: namespace={} hashes_count={} hashes={:?}",
            req.namespace,
            req.block_hashes.len(),
            all_hashes
        );

        if req.block_hashes.is_empty() {
            return Err(Status::invalid_argument("block_hashes cannot be empty"));
        }

        // Returns entries up to the first globally-missing hash.
        let existing = self
            .store
            .query_prefix(&req.namespace, &req.block_hashes)
            .await;

        let total_queried = req.block_hashes.len();
        let prefix_len = existing.len();

        let elapsed = start.elapsed();
        info!(
            "RPC [query_prefix_blocks]: namespace={} prefix={}/{} in {:?}",
            req.namespace, prefix_len, total_queried, elapsed
        );

        // Per-node prefix: `existing[i]` maps to `block_hashes[i]`.
        // A node's prefix = count of consecutive hashes from h0 it owns.
        let mut node_prefix: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();
        for (i, entry) in existing.iter().enumerate() {
            let count = node_prefix.entry(&entry.node).or_insert(0);
            // Only extend if every previous hash (0..i) was also on this node
            if *count == i as u32 {
                *count += 1;
            }
        }
        let nodes: Vec<NodePrefixResult> = node_prefix
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(node, count)| NodePrefixResult {
                node: node.to_string(),
                prefix_len: count,
            })
            .collect();

        let response = QueryPrefixBlocksResponse { nodes };

        Ok(Response::new(response))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let response = HealthResponse {
            status: Some(Self::ok_status()),
        };
        Ok(Response::new(response))
    }

    async fn shutdown(
        &self,
        _request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        info!("RPC [shutdown]: received shutdown request");
        self.shutdown.notify_one();

        let response = ShutdownResponse {
            status: Some(Self::ok_status()),
        };
        Ok(Response::new(response))
    }
}
