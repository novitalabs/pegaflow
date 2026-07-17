use crate::metric::record_rpc_result;
use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    HeartbeatNodeRequest, HeartbeatNodeResponse, InsertBlockHashesRequest,
    InsertBlockHashesResponse, NodePrefixResult, QueryPrefixBlocksRequest,
    QueryPrefixBlocksResponse, RemoveBlockHashesRequest, RemoveBlockHashesResponse, ResponseStatus,
    UnregisterNodeRequest, UnregisterNodeResponse,
};
use crate::store::{BlockHashStore, StoreError};
use log::debug;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

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

    fn parse_node_id(node_id: &str) -> Result<Uuid, Status> {
        Uuid::parse_str(node_id)
            .map_err(|e| Status::invalid_argument(format!("invalid node_id: {e}")))
    }

    fn store_error_status(err: StoreError) -> Status {
        match err {
            StoreError::UnknownNode => Status::failed_precondition("unknown node"),
            StoreError::StaleSession => Status::failed_precondition("stale node session"),
        }
    }
}

#[async_trait]
impl MetaServer for GrpcMetaService {
    async fn heartbeat_node(
        &self,
        request: Request<HeartbeatNodeRequest>,
    ) -> Result<Response<HeartbeatNodeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        debug!(
            "RPC [heartbeat_node]: node={} node_id={}",
            req.node, req.node_id
        );
        let result = async {
            let node_id = Self::parse_node_id(&req.node_id)?;
            self.store
                .heartbeat_node(&req.node, node_id)
                .map_err(Self::store_error_status)?;
            Ok(Response::new(HeartbeatNodeResponse {
                stale_after_secs: self.store.config().node_stale_after.as_secs(),
            }))
        }
        .await;
        record_rpc_result("heartbeat_node", &result, start);
        result
    }

    async fn unregister_node(
        &self,
        request: Request<UnregisterNodeRequest>,
    ) -> Result<Response<UnregisterNodeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        debug!(
            "RPC [unregister_node]: node={} node_id={}",
            req.node, req.node_id
        );
        let result = async {
            let node_id = Self::parse_node_id(&req.node_id)?;
            let removed = self
                .store
                .unregister_node(&req.node, node_id)
                .map_err(Self::store_error_status)?;
            Ok(Response::new(UnregisterNodeResponse {
                removed_owners: removed as u64,
            }))
        }
        .await;
        if let Ok(resp) = &result {
            debug!(
                "RPC [unregister_node]: node={} removed_owners={} in {:?}",
                req.node,
                resp.get_ref().removed_owners,
                start.elapsed()
            );
        }
        record_rpc_result("unregister_node", &result, start);
        result
    }

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

        let node_id = match Self::parse_node_id(&req.node_id) {
            Ok(id) => id,
            Err(status) => {
                let result = Err(status);
                record_rpc_result("insert_block_hashes", &result, start);
                return result;
            }
        };

        let inserted =
            match self
                .store
                .insert_hashes(&req.namespace, &req.block_hashes, &req.node, node_id)
            {
                Ok(inserted) => inserted,
                Err(err) => {
                    let result = Err(Self::store_error_status(err));
                    record_rpc_result("insert_block_hashes", &result, start);
                    return result;
                }
            };

        let elapsed = start.elapsed();
        debug!(
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

        let node_id = match Self::parse_node_id(&req.node_id) {
            Ok(id) => id,
            Err(status) => {
                let result = Err(status);
                record_rpc_result("remove_block_hashes", &result, start);
                return result;
            }
        };

        let removed =
            match self
                .store
                .remove_hashes(&req.namespace, &req.block_hashes, &req.node, node_id)
            {
                Ok(removed) => removed,
                Err(err) => {
                    let result = Err(Self::store_error_status(err));
                    record_rpc_result("remove_block_hashes", &result, start);
                    return result;
                }
            };

        let elapsed = start.elapsed();
        debug!(
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
        debug!(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BlockHashStore;

    fn make_service() -> GrpcMetaService {
        GrpcMetaService::new(Arc::new(BlockHashStore::new()))
    }

    async fn heartbeat_node(svc: &GrpcMetaService, node: &str) -> String {
        let node_id = Uuid::new_v4().to_string();
        svc.heartbeat_node(Request::new(HeartbeatNodeRequest {
            node: node.into(),
            node_id: node_id.clone(),
        }))
        .await
        .unwrap();
        node_id
    }

    #[tokio::test]
    async fn test_remove_block_hashes_own_blocks() {
        let svc = make_service();
        let node_id = heartbeat_node(&svc, "node-a").await;

        // Insert
        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1, 2, 3]],
            node: "node-a".into(),
            node_id: node_id.clone(),
        }))
        .await
        .unwrap();

        // Remove with matching owner
        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
                node: "node-a".into(),
                node_id,
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
        let node_b_id = heartbeat_node(&svc, "node-b").await;
        let node_a_id = heartbeat_node(&svc, "node-a").await;

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1, 2, 3]],
            node: "node-b".into(),
            node_id: node_b_id,
        }))
        .await
        .unwrap();

        // Remove with non-matching owner
        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1, 2, 3]],
                node: "node-a".into(),
                node_id: node_a_id,
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
        let node_id = heartbeat_node(&svc, "node-a").await;

        let resp = svc
            .remove_block_hashes(Request::new(RemoveBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![],
                node: "node-a".into(),
                node_id,
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(!resp.status.unwrap().ok);
        assert_eq!(resp.removed_count, 0);
    }

    #[tokio::test]
    async fn test_heartbeat_node_accepts_current_session() {
        let svc = make_service();
        let node_id = heartbeat_node(&svc, "node-a").await;

        let resp = svc
            .heartbeat_node(Request::new(HeartbeatNodeRequest {
                node: "node-a".into(),
                node_id,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.stale_after_secs, 30);
    }

    #[tokio::test]
    async fn test_heartbeat_with_active_different_session_is_rejected() {
        let svc = make_service();
        let old_id = heartbeat_node(&svc, "node-a").await;
        let new_id = Uuid::new_v4().to_string();
        assert_ne!(old_id, new_id);

        let err = svc
            .heartbeat_node(Request::new(HeartbeatNodeRequest {
                node: "node-a".into(),
                node_id: new_id,
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn test_old_session_insert_is_rejected_after_stale_takeover() {
        let store = Arc::new(BlockHashStore::with_config(crate::store::StoreConfig {
            node_stale_after: std::time::Duration::ZERO,
            ttl: std::time::Duration::from_secs(60),
        }));
        let svc = GrpcMetaService::new(store);
        let old_id = heartbeat_node(&svc, "node-a").await;
        let new_id = heartbeat_node(&svc, "node-a").await;
        assert_ne!(old_id, new_id);

        let err = svc
            .insert_block_hashes(Request::new(InsertBlockHashesRequest {
                namespace: "ns".into(),
                block_hashes: vec![vec![1]],
                node: "node-a".into(),
                node_id: old_id,
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn test_unregister_node_removes_matching_owners() {
        let svc = make_service();
        let node_id = heartbeat_node(&svc, "node-a").await;

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: "ns".into(),
            block_hashes: vec![vec![1], vec![2]],
            node: "node-a".into(),
            node_id: node_id.clone(),
        }))
        .await
        .unwrap();

        let resp = svc
            .unregister_node(Request::new(UnregisterNodeRequest {
                node: "node-a".into(),
                node_id,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.removed_owners, 2);

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
        let node_a_id = heartbeat_node(&svc, node_a).await;
        let node_b_id = heartbeat_node(&svc, node_b).await;

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: namespace.into(),
            block_hashes: vec![h1.clone(), h2.clone(), h3.clone(), h4.clone()],
            node: node_a.into(),
            node_id: node_a_id,
        }))
        .await
        .unwrap();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: namespace.into(),
            block_hashes: vec![h1.clone(), h2.clone(), h3.clone(), h5],
            node: node_b.into(),
            node_id: node_b_id,
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
}
