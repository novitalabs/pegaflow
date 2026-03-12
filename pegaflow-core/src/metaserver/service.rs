use pegaflow_proto::proto::engine::meta_server_server::MetaServer;
use pegaflow_proto::proto::engine::{
    HealthRequest, HealthResponse, InsertBlockHashesRequest, InsertBlockHashesResponse,
    NodeBlockHashes, QueryBlockHashesRequest, QueryBlockHashesResponse, ResponseStatus,
    ShutdownRequest, ShutdownResponse,
};

use super::store::BlockHashStore;
use log::{debug, info};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;
use tonic::{Request, Response, Status, async_trait};

const EMPTY_HASHES_MSG: &str = "block_hashes cannot be empty";

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

        debug!(
            "RPC [insert_block_hashes]: namespace={} node={} hashes_count={}",
            req.namespace,
            req.node,
            req.block_hashes.len()
        );

        // Validate request
        if req.block_hashes.is_empty() {
            let response = InsertBlockHashesResponse {
                status: Some(Self::error_status(EMPTY_HASHES_MSG.to_string())),
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
            "RPC [insert_block_hashes]: namespace={} node={} inserted={} hashes in {:?}",
            req.namespace, req.node, inserted, elapsed
        );

        let response = InsertBlockHashesResponse {
            status: Some(Self::ok_status()),
            inserted_count: inserted as u64,
        };

        Ok(Response::new(response))
    }

    async fn query_block_hashes(
        &self,
        request: Request<QueryBlockHashesRequest>,
    ) -> Result<Response<QueryBlockHashesResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!(
            "RPC [query_block_hashes]: namespace={} hashes_count={}",
            req.namespace,
            req.block_hashes.len()
        );

        // Validate request
        if req.block_hashes.is_empty() {
            let response = QueryBlockHashesResponse {
                status: Some(Self::error_status(EMPTY_HASHES_MSG.to_string())),
                existing_hashes: vec![],
                total_queried: 0,
                found_count: 0,
                node_blocks: vec![],
            };
            return Ok(Response::new(response));
        }

        // Query existing hashes (returns CrossNodeBlock entries)
        let existing = self
            .store
            .query_hashes(&req.namespace, &req.block_hashes)
            .await;

        let total_queried = req.block_hashes.len();
        let found_count = existing.len();

        let elapsed = start.elapsed();
        info!(
            "RPC [query_block_hashes]: namespace={} found={}/{} hashes in {:?}",
            req.namespace, found_count, total_queried, elapsed
        );

        // Build existing_hashes and node_blocks in a single pass.
        let mut node_map: HashMap<&str, Vec<Vec<u8>>> = HashMap::new();
        let mut existing_hashes = Vec::with_capacity(found_count);
        for entry in &existing {
            existing_hashes.push(entry.block_hash.clone());
            node_map
                .entry(&entry.node)
                .or_default()
                .push(entry.block_hash.clone());
        }
        let node_blocks: Vec<NodeBlockHashes> = node_map
            .into_iter()
            .map(|(node, block_hashes)| NodeBlockHashes {
                node: node.to_string(),
                block_hashes,
                domain_addresses: vec![],
            })
            .collect();

        let response = QueryBlockHashesResponse {
            status: Some(Self::ok_status()),
            existing_hashes,
            total_queried: total_queried as u64,
            found_count: found_count as u64,
            node_blocks,
        };

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
