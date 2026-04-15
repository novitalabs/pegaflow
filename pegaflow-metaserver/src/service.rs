use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    HealthRequest, HealthResponse, InsertBlockHashesRequest, InsertBlockHashesResponse,
    NodePrefixResult, QueryPrefixBlocksRequest, QueryPrefixBlocksResponse,
    RemoveBlockHashesRequest, RemoveBlockHashesResponse, ResponseStatus, ShutdownRequest,
    ShutdownResponse,
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

        debug!(
            "RPC [insert_block_hashes]: namespace={} node={} hashes_count={}",
            req.namespace,
            req.node,
            req.block_hashes.len()
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
            "RPC [insert_block_hashes]: namespace={} node={} inserted={} hashes in {:?}",
            req.namespace, req.node, inserted, elapsed
        );

        let response = InsertBlockHashesResponse {
            status: Some(Self::ok_status()),
            inserted_count: inserted as u64,
        };

        Ok(Response::new(response))
    }

    async fn remove_block_hashes(
        &self,
        request: Request<RemoveBlockHashesRequest>,
    ) -> Result<Response<RemoveBlockHashesResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!(
            "RPC [remove_block_hashes]: namespace={} node={} hashes_count={}",
            req.namespace,
            req.node,
            req.block_hashes.len()
        );

        if req.block_hashes.is_empty() {
            return Ok(Response::new(RemoveBlockHashesResponse {
                status: Some(Self::error_status(
                    "block_hashes cannot be empty".to_string(),
                )),
                removed_count: 0,
            }));
        }

        let removed = self
            .store
            .remove_hashes(&req.namespace, &req.block_hashes, &req.node)
            .await;

        let elapsed = start.elapsed();
        info!(
            "RPC [remove_block_hashes]: namespace={} node={} removed={} hashes in {:?}",
            req.namespace, req.node, removed, elapsed
        );

        Ok(Response::new(RemoveBlockHashesResponse {
            status: Some(Self::ok_status()),
            removed_count: removed as u64,
        }))
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

        debug!(
            "RPC [query_prefix_blocks]: namespace={} hashes_count={}",
            req.namespace,
            req.block_hashes.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BlockHashStore;

    fn make_service() -> GrpcMetaService {
        GrpcMetaService::new(Arc::new(BlockHashStore::new()), Arc::new(Notify::new()))
    }

    #[tokio::test]
    async fn test_remove_block_hashes_own_blocks() {
        let svc = make_service();

        // Insert
        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1, 2, 3]],
            node: "node-a".into(),
        }))
        .await
        .unwrap();

        // Remove with matching owner
        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
                node: "node-a".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(resp.status.unwrap().ok);
        assert_eq!(resp.removed_count, 1);

        // Verify gone
        let query_resp = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(query_resp.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_remove_block_hashes_wrong_owner_is_noop() {
        let svc = make_service();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1, 2, 3]],
            node: "node-b".into(),
        }))
        .await
        .unwrap();

        // Remove with non-matching owner
        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
                node: "node-a".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(resp.status.unwrap().ok);
        assert_eq!(resp.removed_count, 0);

        // Verify still present
        let query_resp = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(query_resp.nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_remove_block_hashes_empty_request_returns_error() {
        let svc = make_service();

        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![],
                node: "node-a".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(!resp.status.unwrap().ok);
        assert_eq!(resp.removed_count, 0);
    }
}
