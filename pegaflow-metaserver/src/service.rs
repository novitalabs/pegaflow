use crate::proto::engine::meta_server_server::MetaServer;
use crate::proto::engine::{
    HealthRequest, HealthResponse, InsertBlockHashesRequest, InsertBlockHashesResponse,
    QueryBlockHashesRequest, QueryBlockHashesResponse, ResponseStatus, ShutdownRequest,
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
            "RPC [insert_block_hashes]: namespace={} hashes_count={}",
            req.namespace,
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

        // Insert hashes
        let inserted = self
            .store
            .insert_hashes(&req.namespace, &req.block_hashes)
            .await;

        let elapsed = start.elapsed();
        info!(
            "RPC [insert_block_hashes]: namespace={} inserted={} hashes in {:?}",
            req.namespace, inserted, elapsed
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
                status: Some(Self::error_status(
                    "block_hashes cannot be empty".to_string(),
                )),
                existing_hashes: vec![],
                total_queried: 0,
                found_count: 0,
            };
            return Ok(Response::new(response));
        }

        // Query existing hashes
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

        let response = QueryBlockHashesResponse {
            status: Some(Self::ok_status()),
            existing_hashes: existing,
            total_queried: total_queried as u64,
            found_count: found_count as u64,
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
