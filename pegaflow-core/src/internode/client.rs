//! gRPC client for inter-node PegaFlow communication.
//!
//! This module provides a high-level gRPC client for communicating with
//! remote PegaFlow engine instances, particularly for Query operations
//! in P/D disaggregation scenarios.

use std::sync::Arc;

use log::debug;
use pegaflow_proto::proto::engine::{QueryRequest, engine_client::EngineClient};
use tonic::transport::{Channel, Endpoint};

use super::registry::InstanceRegistry;
use super::types::{ClientConfig, ClientError, PegaflowInstance, QueryPrefetchStatus, QueryResult};

/// gRPC client for a single PegaFlow instance.
///
/// Manages a persistent connection to a remote PegaFlow engine server.
#[derive(Clone)]
pub struct PegaflowClient {
    /// The endpoint URL.
    endpoint: String,
    /// The underlying gRPC client.
    client: EngineClient<Channel>,
}

impl PegaflowClient {
    /// Connect to a PegaFlow instance at the given endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The gRPC endpoint URL (e.g., "http://10.0.0.1:50055")
    /// * `config` - Client configuration
    pub async fn connect(endpoint: &str, config: &ClientConfig) -> Result<Self, ClientError> {
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

        debug!("Connected to PegaFlow instance at {}", endpoint);

        Ok(Self {
            endpoint: endpoint.to_string(),
            client,
        })
    }

    /// Connect to a PegaFlow instance with default configuration.
    pub async fn connect_default(endpoint: &str) -> Result<Self, ClientError> {
        Self::connect(endpoint, &ClientConfig::default()).await
    }

    /// Connect to a discovered PegaFlow instance.
    pub async fn from_instance(
        instance: &PegaflowInstance,
        config: &ClientConfig,
    ) -> Result<Self, ClientError> {
        Self::connect(&instance.grpc_endpoint(), config).await
    }

    /// Get the endpoint URL.
    pub fn endpoint(&self) -> &str {
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
    pub async fn query(
        &self,
        instance_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<QueryResult, ClientError> {
        let request = QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes,
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
    pub async fn query_prefetch(
        &self,
        instance_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<QueryResult, ClientError> {
        let request = QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes,
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
            // PrefetchDone = 0
            0 => QueryPrefetchStatus::Done {
                hit_blocks: resp.hit_blocks as usize,
                missing_blocks: resp.missing_blocks as usize,
            },
            // PrefetchLoading = 1
            1 => QueryPrefetchStatus::Loading {
                hit_blocks: resp.hit_blocks as usize,
                loading_blocks: resp.loading_blocks as usize,
            },
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

    /// Check if the remote instance is healthy.
    pub async fn health(&self) -> Result<bool, ClientError> {
        use pegaflow_proto::proto::engine::HealthRequest;

        let response = self
            .client
            .clone()
            .health(HealthRequest {})
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        let resp = response.into_inner();
        Ok(resp.status.map(|s| s.ok).unwrap_or(false))
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
}

impl PegaflowClientPool {
    /// Create a new client pool with the given registry.
    pub fn new(registry: Arc<InstanceRegistry>, config: ClientConfig) -> Self {
        Self {
            config,
            registry,
            clients: dashmap::DashMap::new(),
        }
    }

    /// Create a new client pool with default configuration.
    pub fn with_registry(registry: Arc<InstanceRegistry>) -> Self {
        Self::new(registry, ClientConfig::default())
    }

    /// Get a cached client or connect to the given endpoint.
    ///
    /// The endpoint is typically `http://ip:port` as returned by the metaserver.
    /// Cached connections whose endpoint is no longer healthy in the registry
    /// are evicted and reconnected.
    pub async fn get_or_connect(&self, endpoint: &str) -> Result<PegaflowClient, ClientError> {
        // Fast path: return cached client if the instance is still healthy.
        if let Some(client) = self.clients.get(endpoint) {
            if self.is_endpoint_healthy(endpoint) {
                return Ok(client.clone());
            }
            // Instance disappeared or became unhealthy — drop the stale entry.
            drop(client);
            self.clients.remove(endpoint);
            debug!("Evicted stale client for {}", endpoint);
        }

        // Connect and cache.
        let client = PegaflowClient::connect(endpoint, &self.config).await?;
        self.clients.insert(endpoint.to_string(), client.clone());
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
    pub fn remove(&self, endpoint: &str) {
        self.clients.remove(endpoint);
    }

    /// Drop all cached clients.
    pub fn clear(&self) {
        self.clients.clear();
    }

    /// Number of cached connections.
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// Whether the pool has no cached connections.
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_pool_connect_unknown_endpoint() {
        let registry = Arc::new(InstanceRegistry::new());
        let pool = PegaflowClientPool::with_registry(registry);

        // Unreachable endpoint should fail with ConnectionFailed.
        let result = pool.get_or_connect("http://192.0.2.1:50055").await;
        assert!(matches!(result, Err(ClientError::ConnectionFailed(_))));
    }
}
