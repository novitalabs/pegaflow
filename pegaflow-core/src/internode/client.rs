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

    /// Query prefix cache hits on the remote instance.
    ///
    /// This is the main API for P/D disaggregation where a decode instance
    /// queries a prefill instance for cached KV blocks.
    ///
    /// # Arguments
    ///
    /// * `instance_id` - The model instance ID
    /// * `block_hashes` - List of block hashes to query
    ///
    /// # Returns
    ///
    /// `QueryResult` containing hit/miss/loading counts.
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

/// Client pool for managing connections to multiple PegaFlow instances.
///
/// Uses service discovery to automatically connect to discovered instances.
pub struct PegaflowClientPool {
    /// Client configuration.
    config: ClientConfig,
    /// Instance registry for service discovery.
    registry: Arc<InstanceRegistry>,
    /// Cached clients by instance name.
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

    /// Get or create a client for the given instance name.
    pub async fn get_client(&self, instance_name: &str) -> Result<PegaflowClient, ClientError> {
        // Check cache first
        if let Some(client) = self.clients.get(instance_name) {
            return Ok(client.clone());
        }

        // Look up instance in registry
        let instance = self
            .registry
            .get(instance_name)
            .ok_or_else(|| ClientError::InstanceNotFound(instance_name.to_string()))?;

        if !instance.is_healthy() {
            return Err(ClientError::InstanceNotFound(format!(
                "{} is not healthy",
                instance_name
            )));
        }

        // Create new client
        let client = PegaflowClient::from_instance(&instance, &self.config).await?;
        self.clients
            .insert(instance_name.to_string(), client.clone());

        Ok(client)
    }

    /// Get a client for any healthy instance.
    pub async fn get_any_client(&self) -> Result<PegaflowClient, ClientError> {
        let instances = self.registry.healthy_instances();
        if instances.is_empty() {
            return Err(ClientError::NoHealthyInstances);
        }

        // Try to get an existing client first
        for instance in &instances {
            if let Some(client) = self.clients.get(&instance.name) {
                return Ok(client.clone());
            }
        }

        // Create a new client for the first healthy instance
        let instance = &instances[0];
        let client = PegaflowClient::from_instance(instance, &self.config).await?;
        self.clients.insert(instance.name.clone(), client.clone());

        Ok(client)
    }

    /// Query across all healthy instances in parallel.
    ///
    /// Returns results from all instances that responded successfully.
    pub async fn query_all(
        &self,
        instance_id: &str,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<(String, Result<QueryResult, ClientError>)> {
        use futures::future::join_all;

        let instances = self.registry.healthy_instances();
        if instances.is_empty() {
            return vec![];
        }

        let futures: Vec<_> = instances
            .into_iter()
            .map(|instance| {
                let name = instance.name.clone();
                let endpoint = instance.grpc_endpoint();
                let config = self.config.clone();
                let instance_id = instance_id.to_string();
                let hashes = block_hashes.clone();

                async move {
                    let result = async {
                        let client = PegaflowClient::connect(&endpoint, &config).await?;
                        client.query(&instance_id, hashes).await
                    }
                    .await;

                    (name, result)
                }
            })
            .collect();

        join_all(futures).await
    }

    /// Remove a client from the cache.
    pub fn remove_client(&self, instance_name: &str) {
        self.clients.remove(instance_name);
    }

    /// Clear all cached clients.
    pub fn clear(&self) {
        self.clients.clear();
    }

    /// Get the number of cached clients.
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_pool_empty_registry() {
        let registry = Arc::new(InstanceRegistry::new());
        let pool = PegaflowClientPool::with_registry(registry);

        let result = pool.get_any_client().await;
        assert!(matches!(result, Err(ClientError::NoHealthyInstances)));
    }
}
