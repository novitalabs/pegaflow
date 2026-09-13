use crate::metric::record_rpc_result;
use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    FetchSegment, HeartbeatNodeRequest, HeartbeatNodeResponse, InsertBlockHashesRequest,
    InsertBlockHashesResponse, QueryPrefixBlocksRequest, QueryPrefixBlocksResponse,
    RemoveBlockHashesRequest, RemoveBlockHashesResponse, ResponseStatus, UnregisterNodeRequest,
    UnregisterNodeResponse,
};
use crate::store::{BlockHashStore, CACHE_TIER_ALL, CACHE_TIER_RAM, PrefixEntry, StoreError};
use log::debug;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

fn plan_fetch_segments(
    entries: &[PrefixEntry],
    exclude_node: &str,
    required_tier_mask: u32,
) -> Result<Vec<FetchSegment>, &'static str> {
    let mut segments = Vec::new();
    let mut offset = 0usize;

    while offset < entries.len() {
        let mut best: Option<(&str, usize)> = None;
        for owner in &entries[offset].owners {
            if owner.tier_mask & required_tier_mask == 0 {
                continue;
            }
            let candidate = owner.node.as_ref();
            if candidate == exclude_node {
                continue;
            }

            let end = entries[offset..]
                .iter()
                .take_while(|entry| {
                    entry.owners.iter().any(|owner| {
                        owner.node.as_ref() == candidate
                            && owner.tier_mask & required_tier_mask != 0
                    })
                })
                .count()
                + offset;

            if best.is_none_or(|(best_node, best_end)| {
                end > best_end || (end == best_end && candidate < best_node)
            }) {
                best = Some((candidate, end));
            }
        }

        let Some((node, end)) = best else {
            break;
        };
        let block_count =
            u32::try_from(end - offset).map_err(|_| "fetch segment block count exceeds uint32")?;
        segments.push(FetchSegment {
            node: node.to_string(),
            block_count,
        });
        offset = end;
    }

    Ok(segments)
}

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

    fn insert_tier_mask(mask: u32) -> Result<u32, Status> {
        let mask = if mask == 0 { CACHE_TIER_RAM } else { mask };
        Self::validate_tier_mask(mask)
    }

    fn remove_tier_mask(mask: u32) -> Result<u32, Status> {
        let mask = if mask == 0 { CACHE_TIER_ALL } else { mask };
        Self::validate_tier_mask(mask)
    }

    fn query_tier_mask(mask: u32) -> Result<u32, Status> {
        let mask = if mask == 0 { CACHE_TIER_ALL } else { mask };
        Self::validate_tier_mask(mask)
    }

    fn validate_tier_mask(mask: u32) -> Result<u32, Status> {
        if mask == 0 || mask & !CACHE_TIER_ALL != 0 {
            return Err(Status::invalid_argument("tier mask contains unknown bits"));
        }
        Ok(mask)
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
                reclaimable_hashes: Vec::new(),
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
        let tier_mask = Self::insert_tier_mask(req.tier_mask)?;

        let inserted_count = req.block_hashes.len() as u64;
        let insert_result = match self.store.add_hash_tiers(
            &req.namespace,
            &req.block_hashes,
            &req.node,
            node_id,
            tier_mask,
        ) {
            Ok(result) => result,
            Err(err) => {
                let result = Err(Self::store_error_status(err));
                record_rpc_result("insert_block_hashes", &result, start);
                return result;
            }
        };

        let elapsed = start.elapsed();
        debug!(
            "RPC [insert_block_hashes]: namespace={} node={} inserted={} reclaimable={} in {:?}",
            req.namespace,
            req.node,
            inserted_count,
            insert_result.reclaimable_hashes.len(),
            elapsed
        );

        let result = Ok(Response::new(InsertBlockHashesResponse {
            status: Some(Self::ok_status()),
            inserted_count,
            reclaimable_hashes: insert_result.reclaimable_hashes,
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
        let tier_mask = Self::remove_tier_mask(req.tier_mask)?;

        let removed = match self.store.remove_hash_tiers(
            &req.namespace,
            &req.block_hashes,
            &req.node,
            node_id,
            tier_mask,
        ) {
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

    /// Build an ordered plan that covers the longest remotely available prefix.
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
        let required_tier_mask = Self::query_tier_mask(req.required_tier_mask)?;

        let total_queried = req.block_hashes.len();
        let prefix_len = existing.len();

        let elapsed = start.elapsed();
        debug!(
            "RPC [query_prefix_blocks]: namespace={} prefix={}/{} in {:?}",
            req.namespace, prefix_len, total_queried, elapsed
        );

        let segments = plan_fetch_segments(&existing, &req.exclude_node, required_tier_mask)
            .map_err(Status::invalid_argument)?;

        let result = Ok(Response::new(QueryPrefixBlocksResponse { segments }));
        record_rpc_result("query_prefix_blocks", &result, start);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BlockHashStore;
    use crate::store::{CACHE_TIER_SSD, PrefixOwner};

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
    async fn test_insert_block_hashes_returns_new_third_owner_hashes() {
        let svc = make_service();
        let hashes = vec![vec![1], vec![2], vec![1]];

        for (node, expected) in [
            ("node-a", vec![]),
            ("node-b", vec![]),
            ("node-c", vec![vec![1], vec![2]]),
        ] {
            let node_id = heartbeat_node(&svc, node).await;
            let response = svc
                .insert_block_hashes(Request::new(InsertBlockHashesRequest {
                    namespace: "ns".into(),
                    block_hashes: hashes.clone(),
                    node: node.into(),
                    node_id,
                    tier_mask: 0,
                }))
                .await
                .unwrap()
                .into_inner();

            assert_eq!(response.inserted_count, hashes.len() as u64);
            assert_eq!(response.reclaimable_hashes, expected);
        }
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
            tier_mask: 0,
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
                tier_mask: 0,
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
                exclude_node: String::new(),
                required_tier_mask: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(query_resp.segments.is_empty());
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
            tier_mask: 0,
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
                tier_mask: 0,
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
                exclude_node: String::new(),
                required_tier_mask: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(query_resp.segments.len(), 1);
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
                tier_mask: 0,
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
                tier_mask: 0,
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
            tier_mask: 0,
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
                exclude_node: String::new(),
                required_tier_mask: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(query_resp.segments.is_empty());
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
            tier_mask: 0,
        }))
        .await
        .unwrap();

        svc.insert_block_hashes(Request::new(InsertBlockHashesRequest {
            namespace: namespace.into(),
            block_hashes: vec![h1.clone(), h2.clone(), h3.clone(), h5],
            node: node_b.into(),
            node_id: node_b_id,
            tier_mask: 0,
        }))
        .await
        .unwrap();

        let response = svc
            .query_prefix_blocks(Request::new(QueryPrefixBlocksRequest {
                namespace: namespace.into(),
                block_hashes: vec![h1, h2, h3, h4],
                exclude_node: String::new(),
                required_tier_mask: 0,
            }))
            .await
            .unwrap()
            .into_inner();

        let segments: Vec<(String, u32)> = response
            .segments
            .into_iter()
            .map(|entry| (entry.node, entry.block_count))
            .collect();

        assert_eq!(segments, vec![(node_a.to_string(), 4)]);
    }

    fn prefix_entry(hash: u8, nodes: &[&str]) -> PrefixEntry {
        tiered_prefix_entry(
            hash,
            &nodes
                .iter()
                .map(|node| (*node, CACHE_TIER_ALL))
                .collect::<Vec<_>>(),
        )
    }

    fn tiered_prefix_entry(hash: u8, owners: &[(&str, u32)]) -> PrefixEntry {
        PrefixEntry {
            block_hash: vec![hash],
            nodes: owners
                .iter()
                .map(|(node, _)| Arc::<str>::from(*node))
                .collect(),
            owners: owners
                .iter()
                .map(|(node, tier_mask)| PrefixOwner {
                    node: Arc::<str>::from(*node),
                    tier_mask: *tier_mask,
                })
                .collect(),
        }
    }

    fn planned(
        entries: &[PrefixEntry],
        exclude_node: &str,
        required_tier_mask: u32,
    ) -> Vec<(String, u32)> {
        plan_fetch_segments(entries, exclude_node, required_tier_mask)
            .expect("small test plan should fit uint32")
            .into_iter()
            .map(|segment| (segment.node, segment.block_count))
            .collect()
    }

    #[test]
    fn planner_combines_fragmented_remote_prefix() {
        let entries = vec![
            prefix_entry(1, &["node-a"]),
            prefix_entry(2, &["node-a"]),
            prefix_entry(3, &["node-b"]),
            prefix_entry(4, &["node-b"]),
        ];

        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_ALL),
            vec![("node-a".into(), 2), ("node-b".into(), 2)]
        );
    }

    #[test]
    fn planner_chooses_farthest_owner_and_stable_tie_break() {
        let entries = vec![
            prefix_entry(1, &["node-c", "node-b", "node-a"]),
            prefix_entry(2, &["node-c", "node-b", "node-a"]),
            prefix_entry(3, &["node-c"]),
        ];

        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_ALL),
            vec![("node-c".into(), 3)]
        );
        assert_eq!(
            planned(&entries[..2], "requester", CACHE_TIER_ALL),
            vec![("node-a".into(), 2)]
        );
    }

    #[test]
    fn planner_excludes_requester_and_stops_at_remote_gap() {
        let entries = vec![
            prefix_entry(1, &["requester", "node-a"]),
            prefix_entry(2, &["requester"]),
            prefix_entry(3, &["node-b"]),
        ];

        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_ALL),
            vec![("node-a".into(), 1)]
        );
    }

    #[test]
    fn planner_keeps_single_owner_prefix_in_one_segment() {
        let entries = vec![
            prefix_entry(1, &["node-a"]),
            prefix_entry(2, &["node-a"]),
            prefix_entry(3, &["node-a"]),
        ];

        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_ALL),
            vec![("node-a".into(), 3)]
        );
    }

    #[test]
    fn planner_filters_tiers_without_losing_multi_node_segments() {
        let entries = vec![
            tiered_prefix_entry(1, &[("ram-a", CACHE_TIER_RAM), ("ssd-a", CACHE_TIER_SSD)]),
            tiered_prefix_entry(2, &[("ram-a", CACHE_TIER_RAM), ("ssd-a", CACHE_TIER_SSD)]),
            tiered_prefix_entry(3, &[("ssd-a", CACHE_TIER_SSD)]),
            tiered_prefix_entry(4, &[("ssd-b", CACHE_TIER_SSD)]),
        ];

        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_RAM),
            vec![("ram-a".into(), 2)]
        );
        assert_eq!(
            planned(&entries, "requester", CACHE_TIER_ALL),
            vec![("ssd-a".into(), 3), ("ssd-b".into(), 1)]
        );
    }
}
