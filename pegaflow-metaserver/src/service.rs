use crate::metric::{record_liveness_sweep, record_rpc_result};
use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    ByeRequest, ByeResponse, HeartbeatRequest, HeartbeatResponse, InsertBlockHashesRequest,
    InsertBlockHashesResponse, NodePrefixResult, QueryPrefixBlocksRequest,
    QueryPrefixBlocksResponse, RemoveBlockHashesRequest, RemoveBlockHashesResponse, ResponseStatus,
};
use crate::store::BlockHashStore;
use log::{debug, info};
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status, async_trait};

#[derive(Clone)]
pub struct GrpcMetaService {
    store: Arc<BlockHashStore>,
}

impl GrpcMetaService {
    pub fn new(store: Arc<BlockHashStore>) -> Self {
        Self { store }
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

    fn validate_node_epoch(node: &str, epoch: &str) -> Result<(), Status> {
        if node.is_empty() || epoch.is_empty() {
            Err(Status::invalid_argument("node and epoch are required"))
        } else {
            Ok(())
        }
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
            let result = Ok(Response::new(InsertBlockHashesResponse {
                status: Some(Self::error_status(
                    "block_hashes cannot be empty".to_string(),
                )),
                inserted_count: 0,
            }));
            record_rpc_result("insert_block_hashes", &result, start);
            return result;
        }

        // Insert hashes with node ownership
        let inserted = self
            .store
            .insert_hashes(&req.namespace, &req.block_hashes, &req.node);

        let elapsed = start.elapsed();
        info!(
            "RPC [insert_block_hashes]: namespace={} node={} inserted={} hashes in {:?}",
            req.namespace, req.node, inserted, elapsed
        );

        let result = Ok(Response::new(InsertBlockHashesResponse {
            status: Some(Self::ok_status()),
            inserted_count: inserted as u64,
        }));
        record_rpc_result("insert_block_hashes", &result, start);
        result
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
            let result = Ok(Response::new(RemoveBlockHashesResponse {
                status: Some(Self::error_status(
                    "block_hashes cannot be empty".to_string(),
                )),
                removed_count: 0,
            }));
            record_rpc_result("remove_block_hashes", &result, start);
            return result;
        }

        let removed = self
            .store
            .remove_hashes(&req.namespace, &req.block_hashes, &req.node);

        let elapsed = start.elapsed();
        info!(
            "RPC [remove_block_hashes]: namespace={} node={} removed={} hashes in {:?}",
            req.namespace, req.node, removed, elapsed
        );

        let result = Ok(Response::new(RemoveBlockHashesResponse {
            status: Some(Self::ok_status()),
            removed_count: removed as u64,
        }));
        record_rpc_result("remove_block_hashes", &result, start);
        result
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
            let result: Result<Response<QueryPrefixBlocksResponse>, Status> =
                Err(Status::invalid_argument("block_hashes cannot be empty"));
            record_rpc_result("query_prefix_blocks", &result, start);
            return result;
        }

        // Returns entries up to the first globally-missing hash.
        let existing = self.store.query_prefix(&req.namespace, &req.block_hashes);

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
            for node in &entry.nodes {
                let count = node_prefix.entry(node.as_ref()).or_insert(0);
                // Only extend if every previous hash (0..i) was also on this node
                if *count == i as u32 {
                    *count += 1;
                }
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

        let result = Ok(Response::new(QueryPrefixBlocksResponse { nodes }));
        record_rpc_result("query_prefix_blocks", &result, start);
        result
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!("RPC [heartbeat]: node={} epoch={}", req.node, req.epoch);

        if let Err(e) = Self::validate_node_epoch(&req.node, &req.epoch) {
            let result: Result<Response<HeartbeatResponse>, Status> = Err(e);
            record_rpc_result("heartbeat", &result, start);
            return result;
        }

        self.store.heartbeat(&req.node, &req.epoch);

        let result = Ok(Response::new(HeartbeatResponse {}));
        record_rpc_result("heartbeat", &result, start);
        result
    }

    async fn bye(&self, request: Request<ByeRequest>) -> Result<Response<ByeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        info!("RPC [bye]: node={} epoch={}", req.node, req.epoch);

        if let Err(e) = Self::validate_node_epoch(&req.node, &req.epoch) {
            let result: Result<Response<ByeResponse>, Status> = Err(e);
            record_rpc_result("bye", &result, start);
            return result;
        }

        let purged = self.store.bye(&req.node, &req.epoch);
        info!("RPC [bye]: node={} purged={} entries", req.node, purged);

        if purged > 0 {
            record_liveness_sweep(0, 1);
        }

        let result = Ok(Response::new(ByeResponse {}));
        record_rpc_result("bye", &result, start);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BlockHashStore;

    fn make_service() -> GrpcMetaService {
        GrpcMetaService::new(Arc::new(BlockHashStore::new()))
    }

    fn make_service_with_liveness(suspect_secs: u64, hard_delete_secs: u64) -> GrpcMetaService {
        GrpcMetaService::new(Arc::new(BlockHashStore::with_liveness_config(
            120,
            suspect_secs,
            hard_delete_secs,
        )))
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

    #[tokio::test]
    async fn test_query_prefix_blocks_shared_prefix_multi_owner() {
        let svc = make_service();
        let namespace = "model-a";

        let h1 = vec![1];
        let h2 = vec![2];
        let h3 = vec![3];
        let h4 = vec![4];
        let h5 = vec![5];

        let node_a = "node-a:50055";
        let node_b = "node-b:50055";

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: namespace.into(),
            block_hashes: vec![h1.clone(), h2.clone(), h3.clone(), h4.clone()],
            node: node_a.into(),
        }))
        .await
        .unwrap();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: namespace.into(),
            block_hashes: vec![h1.clone(), h2.clone(), h3.clone(), h5],
            node: node_b.into(),
        }))
        .await
        .unwrap();

        let response = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: namespace.into(),
                block_hashes: vec![h1, h2, h3, h4],
            }))
            .await
            .unwrap()
            .into_inner();

        let mut nodes: Vec<(String, u32)> = response
            .nodes
            .into_iter()
            .map(|entry| (entry.node, entry.prefix_len))
            .collect();
        nodes.sort();

        // node-a owns h1..h4 (prefix 4), node-b owns h1..h3 (prefix 3)
        assert_eq!(
            nodes,
            vec![(node_a.to_string(), 4), (node_b.to_string(), 3),]
        );
    }

    #[tokio::test]
    async fn test_heartbeat_rpc() {
        let svc = make_service();

        let resp = svc
            .heartbeat(Request::new(HeartbeatRequest {
                node: "node-a".into(),
                epoch: "epoch-1".into(),
            }))
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn test_heartbeat_rpc_empty_fields_rejected() {
        let svc = make_service();

        let resp = svc
            .heartbeat(Request::new(HeartbeatRequest {
                node: "".into(),
                epoch: "epoch-1".into(),
            }))
            .await;
        assert!(resp.is_err());
    }

    #[tokio::test]
    async fn test_bye_rpc_purges_entries() {
        let svc = make_service();

        // Register heartbeat + insert blocks
        svc.heartbeat(Request::new(HeartbeatRequest {
            node: "node-a".into(),
            epoch: "epoch-1".into(),
        }))
        .await
        .unwrap();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1], vec![2]],
            node: "node-a".into(),
        }))
        .await
        .unwrap();

        // Bye → purge
        let resp = svc
            .bye(Request::new(ByeRequest {
                node: "node-a".into(),
                epoch: "epoch-1".into(),
            }))
            .await;
        assert!(resp.is_ok());

        // Query should return empty
        let query_resp = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1], vec![2]],
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(query_resp.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_query_excludes_suspect_after_sweep() {
        // suspect_threshold=0 → immediately suspect
        let svc = make_service_with_liveness(0, 3600);

        svc.heartbeat(Request::new(HeartbeatRequest {
            node: "node-a".into(),
            epoch: "epoch-1".into(),
        }))
        .await
        .unwrap();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1]],
            node: "node-a".into(),
        }))
        .await
        .unwrap();

        // Trigger sweep → node-a becomes suspect
        svc.store.sweep_liveness();

        // Query should filter out suspect node
        let query_resp = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1]],
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(query_resp.nodes.is_empty());
    }
}
