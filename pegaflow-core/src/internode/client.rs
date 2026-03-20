//! gRPC client for inter-node PegaFlow communication.
//!
//! This module provides a high-level gRPC client for communicating with
//! remote PegaFlow engine instances, particularly for Query operations
//! in P/D disaggregation scenarios.

use std::sync::Arc;

use log::debug;
use pegaflow_proto::proto::engine::{
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, QueryRequest,
    ReleaseTransferLockRequest, engine_client::EngineClient,
};
use tonic::transport::{Channel, Endpoint};

use super::registry::InstanceRegistry;
use super::types::{ClientConfig, ClientError, PegaflowInstance, QueryPrefetchStatus, QueryResult};

/// gRPC client for a single PegaFlow instance.
///
/// Manages a persistent connection to a remote PegaFlow engine server.
#[derive(Clone)]
pub struct PegaflowClient {
    /// The endpoint URL.
    #[allow(dead_code)]
    endpoint: String,
    /// The underlying gRPC client.
    client: EngineClient<Channel>,
}

#[allow(dead_code)]
impl PegaflowClient {
    /// Connect to a PegaFlow instance at the given endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The gRPC endpoint URL (e.g., "http://10.0.0.1:50055")
    /// * `config` - Client configuration
    async fn connect(endpoint: &str, config: &ClientConfig) -> Result<Self, ClientError> {
        let endpoint_cfg = Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .tcp_nodelay(config.tcp_nodelay)
            .http2_keep_alive_interval(config.keep_alive_interval)
            .keep_alive_while_idle(true);

        let channel = endpoint_cfg
            .connect()
            .await
            .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?;

        let client = EngineClient::new(channel);

        debug!("Connected to PegaFlow instance at {endpoint}");

        Ok(Self {
            endpoint: endpoint.to_string(),
            client,
        })
    }

    /// Connect to a PegaFlow instance with default configuration.
    async fn connect_default(endpoint: &str) -> Result<Self, ClientError> {
        Self::connect(endpoint, &ClientConfig::default()).await
    }

    /// Connect to a discovered PegaFlow instance.
    async fn from_instance(
        instance: &PegaflowInstance,
        config: &ClientConfig,
    ) -> Result<Self, ClientError> {
        Self::connect(&instance.grpc_endpoint(), config).await
    }

    /// Get the endpoint URL.
    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Pure memory-only query on the remote instance.
    ///
    /// Checks if prefix blocks are in the remote node's memory cache
    /// without triggering SSD prefetch.
    ///
    /// # Arguments
    ///
    /// * `instance_id` - The model instance ID
    /// * `block_hashes` - List of block hashes to query
    ///
    /// # Returns
    ///
    /// `QueryResult` containing hit/miss counts (no loading state).
    async fn query(
        &self,
        instance_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<QueryResult, ClientError> {
        let request = QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes,
            req_id: String::new(),
        };

        let response = self
            .client
            .clone()
            .query(request)
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        let resp = response.into_inner();
        let status = resp
            .status
            .ok_or_else(|| ClientError::ResponseError("missing status in response".to_string()))?;

        Ok(QueryResult {
            ok: status.ok,
            message: status.message,
            status: QueryPrefetchStatus::Done {
                hit_blocks: resp.hit_blocks as usize,
                missing_blocks: resp.missing_blocks as usize,
            },
        })
    }

    /// Query prefix cache hits with SSD prefetch support on the remote instance.
    ///
    /// This is the main API for P/D disaggregation where a decode instance
    /// queries a prefill instance for cached KV blocks, triggering SSD prefetch
    /// for blocks not in memory.
    ///
    /// # Arguments
    ///
    /// * `instance_id` - The model instance ID
    /// * `block_hashes` - List of block hashes to query
    ///
    /// # Returns
    ///
    /// `QueryResult` containing hit/miss/loading counts.
    async fn query_prefetch(
        &self,
        instance_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<QueryResult, ClientError> {
        let request = QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes,
            req_id: String::new(),
        };

        let response = self
            .client
            .clone()
            .query_prefetch(request)
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        let resp = response.into_inner();
        let status = resp
            .status
            .ok_or_else(|| ClientError::ResponseError("missing status in response".to_string()))?;

        let prefetch_status = match resp.prefetch_state {
            // PrefetchLoading = 1
            1 => QueryPrefetchStatus::Loading {
                hit_blocks: resp.hit_blocks as usize,
                loading_blocks: resp.loading_blocks as usize,
            },
            // PrefetchDone = 0, and any unknown state
            _ => QueryPrefetchStatus::Done {
                hit_blocks: resp.hit_blocks as usize,
                missing_blocks: resp.missing_blocks as usize,
            },
        };

        Ok(QueryResult {
            ok: status.ok,
            message: status.message,
            status: prefetch_status,
        })
    }

    /// Query block RDMA metadata for cross-node transfer.
    pub(crate) async fn query_blocks_for_transfer(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        requester_id: &str,
    ) -> Result<QueryBlocksForTransferResponse, ClientError> {
        let request = QueryBlocksForTransferRequest {
            namespace: namespace.to_string(),
            block_hashes: block_hashes.to_vec(),
            requester_id: requester_id.to_string(),
        };

        let response = self
            .client
            .clone()
            .query_blocks_for_transfer(request)
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        let resp = response.into_inner();
        if let Some(status) = &resp.status
            && !status.ok
        {
            return Err(ClientError::RpcFailed(format!(
                "query_blocks_for_transfer failed: {}",
                status.message
            )));
        }
        Ok(resp)
    }

    /// Release transfer locks on a remote node.
    pub(crate) async fn release_transfer_lock(
        &self,
        transfer_session_id: &str,
    ) -> Result<(), ClientError> {
        let request = ReleaseTransferLockRequest {
            transfer_session_id: transfer_session_id.to_string(),
        };

        self.client
            .clone()
            .release_transfer_lock(request)
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        Ok(())
    }

    /// Check if the remote instance is healthy.
    async fn health(&self) -> Result<bool, ClientError> {
        use pegaflow_proto::proto::engine::HealthRequest;

        let response = self
            .client
            .clone()
            .health(HealthRequest {})
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        let resp = response.into_inner();
        Ok(resp.status.is_some_and(|s| s.ok))
    }
}

/// Connection pool keyed by endpoint URL (e.g. `http://10.0.0.1:50055`).
///
/// Used for targeted queries: the metaserver tells you which node owns a block,
/// then you call `get_or_connect(endpoint)` to get a reusable gRPC channel.
/// Stale entries are evicted when the backing registry no longer contains the
/// endpoint's instance.
pub struct PegaflowClientPool {
    /// Client configuration.
    config: ClientConfig,
    /// Instance registry for health checks.
    registry: Arc<InstanceRegistry>,
    /// Cached clients keyed by endpoint URL.
    clients: dashmap::DashMap<String, PegaflowClient>,
    /// Maximum number of cached connections before eviction.
    max_connections: usize,
}

#[allow(dead_code)]
impl PegaflowClientPool {
    /// Create a new client pool with the given registry.
    fn new(registry: Arc<InstanceRegistry>, config: ClientConfig) -> Self {
        Self {
            config,
            registry,
            clients: dashmap::DashMap::new(),
            max_connections: 64,
        }
    }

    /// Create a new client pool with default configuration.
    fn with_registry(registry: Arc<InstanceRegistry>) -> Self {
        Self::new(registry, ClientConfig::default())
    }

    /// Create a client pool without service-discovery health tracking.
    pub fn without_registry() -> Self {
        Self::with_registry(Arc::new(InstanceRegistry::new()))
    }

    /// Get a cached client or connect to the given endpoint.
    ///
    /// The endpoint is typically `http://ip:port` as returned by the metaserver.
    /// Cached connections whose endpoint is no longer healthy in the registry
    /// are evicted and reconnected.
    pub(crate) async fn get_or_connect(
        &self,
        endpoint: &str,
    ) -> Result<PegaflowClient, ClientError> {
        // Fast path: return cached client if the instance is still healthy.
        if let Some(client) = self.clients.get(endpoint) {
            if self.is_endpoint_healthy(endpoint) {
                return Ok(client.clone());
            }
            // Instance disappeared or became unhealthy — drop the stale entry.
            drop(client);
            self.clients.remove(endpoint);
            debug!("Evicted stale client for {endpoint}");
        }

        // Connect and cache.
        let client = PegaflowClient::connect(endpoint, &self.config).await?;
        self.clients.insert(endpoint.to_string(), client.clone());

        // Evict if over capacity
        while self.clients.len() > self.max_connections {
            if let Some(entry) = self.clients.iter().next() {
                let key = entry.key().clone();
                drop(entry);
                if key != endpoint {
                    self.clients.remove(&key);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(client)
    }

    /// Check whether any healthy registry instance matches this endpoint.
    fn is_endpoint_healthy(&self, endpoint: &str) -> bool {
        // If no registry entries exist (e.g. service discovery not used),
        // assume healthy — the caller knows the endpoint from the metaserver.
        if self.registry.is_empty() {
            return true;
        }
        self.registry
            .healthy_instances()
            .iter()
            .any(|i| i.grpc_endpoint() == endpoint)
    }

    /// Remove a client from the cache by endpoint.
    fn remove(&self, endpoint: &str) {
        self.clients.remove(endpoint);
    }

    /// Drop all cached clients.
    fn clear(&self) {
        self.clients.clear();
    }

    /// Number of cached connections.
    fn len(&self) -> usize {
        self.clients.len()
    }

    /// Whether the pool has no cached connections.
    fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

#[cfg(test)]
impl PegaflowClientPool {
    pub(crate) fn new_for_test() -> Self {
        Self::without_registry()
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    use pegaflow_proto::proto::engine::engine_server::{Engine as EngineService, EngineServer};
    use pegaflow_proto::proto::engine::{
        HealthRequest, HealthResponse, LoadRequest, LoadResponse, QueryBlocksForTransferRequest,
        QueryBlocksForTransferResponse, QueryRequest, QueryResponse, RegisterContextRequest,
        RegisterContextResponse, ReleaseTransferLockRequest, ReleaseTransferLockResponse,
        ResponseStatus, SaveRequest, SaveResponse, ShutdownRequest, ShutdownResponse, UnpinRequest,
        UnpinResponse, UnregisterRequest, UnregisterResponse,
    };
    use tonic::{Request, Response, Status};

    /// Mock Engine gRPC server for unit tests.
    pub(crate) struct MockEngine {
        pub transfer_response: QueryBlocksForTransferResponse,
    }

    impl MockEngine {
        pub(crate) fn new() -> Self {
            Self {
                transfer_response: QueryBlocksForTransferResponse {
                    status: Some(ResponseStatus {
                        ok: true,
                        message: String::new(),
                    }),
                    blocks: Vec::new(),
                    transfer_session_id: String::new(),
                    rdma_session_id: Vec::new(),
                },
            }
        }
    }

    #[tonic::async_trait]
    impl EngineService for MockEngine {
        async fn register_context_batch(
            &self,
            _: Request<RegisterContextRequest>,
        ) -> Result<Response<RegisterContextResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn save(&self, _: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn load(&self, _: Request<LoadRequest>) -> Result<Response<LoadResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn query(&self, _: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn query_prefetch(
            &self,
            _: Request<QueryRequest>,
        ) -> Result<Response<QueryResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn unpin(&self, _: Request<UnpinRequest>) -> Result<Response<UnpinResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn unregister_context(
            &self,
            _: Request<UnregisterRequest>,
        ) -> Result<Response<UnregisterResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn query_blocks_for_transfer(
            &self,
            _: Request<QueryBlocksForTransferRequest>,
        ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
            Ok(Response::new(self.transfer_response.clone()))
        }
        async fn release_transfer_lock(
            &self,
            _: Request<ReleaseTransferLockRequest>,
        ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
            Ok(Response::new(ReleaseTransferLockResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                released_blocks: 0,
            }))
        }
        async fn shutdown(
            &self,
            _: Request<ShutdownRequest>,
        ) -> Result<Response<ShutdownResponse>, Status> {
            Err(Status::unimplemented("not used in test"))
        }
        async fn health(
            &self,
            _: Request<HealthRequest>,
        ) -> Result<Response<HealthResponse>, Status> {
            Ok(Response::new(HealthResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
            }))
        }
    }

    /// Start a mock Engine gRPC server on a random port.
    /// Returns (address, join_handle).
    pub(crate) async fn start_mock_engine(
        mock: MockEngine,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(EngineServer::new(mock))
                .serve(addr)
                .await
                .unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (addr, handle)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pegaflow_proto::proto::engine::{
        QueryBlocksForTransferResponse, ResponseStatus, TransferBlockInfo, TransferSlotInfo,
    };

    use super::test_utils::*;
    use super::*;

    #[tokio::test]
    async fn test_client_pool_connect_unknown_endpoint() {
        let registry = Arc::new(InstanceRegistry::new());
        let pool = PegaflowClientPool::with_registry(registry);

        // Unreachable endpoint should fail with ConnectionFailed.
        let result = pool.get_or_connect("http://192.0.2.1:50055").await;
        assert!(matches!(result, Err(ClientError::ConnectionFailed(_))));
    }

    #[tokio::test]
    async fn client_pool_caches_connections() {
        let (addr, _) = start_mock_engine(MockEngine::new()).await;
        let endpoint = format!("http://127.0.0.1:{}", addr.port());
        let pool = PegaflowClientPool::with_registry(Arc::new(InstanceRegistry::new()));

        pool.get_or_connect(&endpoint).await.unwrap();
        pool.get_or_connect(&endpoint).await.unwrap();

        assert_eq!(pool.len(), 1);
    }

    #[tokio::test]
    async fn client_pool_empty_registry_healthy() {
        let (addr, _) = start_mock_engine(MockEngine::new()).await;
        let endpoint = format!("http://127.0.0.1:{}", addr.port());
        let pool = PegaflowClientPool::with_registry(Arc::new(InstanceRegistry::new()));

        let c1 = pool.get_or_connect(&endpoint).await.unwrap();
        let c2 = pool.get_or_connect(&endpoint).await.unwrap();

        assert_eq!(pool.len(), 1);
        assert_eq!(c1.endpoint(), c2.endpoint());
    }

    #[tokio::test]
    async fn client_pool_evicts_unhealthy() {
        let (addr, server_handle) = start_mock_engine(MockEngine::new()).await;
        let endpoint = format!("http://127.0.0.1:{}", addr.port());

        let registry = Arc::new(InstanceRegistry::new());
        registry.upsert(PegaflowInstance {
            name: "pega-1".into(),
            ip: "127.0.0.1".into(),
            namespace: "default".into(),
            grpc_port: addr.port(),
            status: "Running".into(),
            is_ready: true,
            labels: HashMap::new(),
        });
        // Dummy instance to keep registry non-empty after removing pega-1
        registry.upsert(PegaflowInstance {
            name: "pega-dummy".into(),
            ip: "192.168.99.1".into(),
            namespace: "default".into(),
            grpc_port: 50055,
            status: "Running".into(),
            is_ready: true,
            labels: HashMap::new(),
        });

        let pool = PegaflowClientPool::with_registry(registry.clone());
        pool.get_or_connect(&endpoint).await.unwrap();
        assert_eq!(pool.len(), 1);

        // Mark endpoint unhealthy + stop server
        registry.remove("pega-1");
        server_handle.abort();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Should evict stale client → reconnect fails (server down)
        let result = pool.get_or_connect(&endpoint).await;
        assert!(result.is_err());
        assert_eq!(pool.len(), 0);
    }

    #[tokio::test]
    async fn query_blocks_for_transfer_roundtrip() {
        let expected = vec![TransferBlockInfo {
            block_hash: vec![1, 2, 3],
            slots: vec![
                TransferSlotInfo {
                    k_ptr: 0x1000,
                    k_size: 512,
                    v_ptr: 0x2000,
                    v_size: 512,
                },
                TransferSlotInfo {
                    k_ptr: 0x3000,
                    k_size: 256,
                    v_ptr: 0,
                    v_size: 0,
                },
            ],
            rkey: 42,
        }];

        let mock = MockEngine {
            transfer_response: QueryBlocksForTransferResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                blocks: expected,
                transfer_session_id: "sess-123".to_string(),
                rdma_session_id: vec![0u8; 26],
            },
        };
        let (addr, _) = start_mock_engine(mock).await;
        let endpoint = format!("http://127.0.0.1:{}", addr.port());

        let client = PegaflowClient::connect_default(&endpoint).await.unwrap();
        let resp = client
            .query_blocks_for_transfer("ns", &[vec![1, 2, 3]], "test")
            .await
            .unwrap();

        assert_eq!(resp.blocks.len(), 1);
        assert_eq!(resp.blocks[0].block_hash, vec![1, 2, 3]);
        assert_eq!(resp.blocks[0].slots.len(), 2);
        assert_eq!(resp.blocks[0].slots[0].k_ptr, 0x1000);
        assert_eq!(resp.blocks[0].slots[0].v_ptr, 0x2000);
        assert_eq!(resp.blocks[0].slots[1].v_ptr, 0);
        assert_eq!(resp.transfer_session_id, "sess-123");
        assert_eq!(resp.rdma_session_id.len(), 26);
    }
}
