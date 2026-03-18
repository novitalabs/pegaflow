//! gRPC client for querying block locations from the MetaServer.
//!
//! Complements the existing `MetaServerRegistrar` (write-only) with read
//! capabilities needed for the cross-node remote fetch path.

use log::debug;
use pegaflow_proto::proto::engine::{
    QueryBlockHashesRequest, QueryBlockHashesResponse, meta_server_client::MetaServerClient,
};
use tonic::transport::{Channel, Endpoint};

use super::types::ClientError;

/// gRPC client for querying block hash locations from the MetaServer.
pub(crate) struct MetaServerQueryClient {
    client: MetaServerClient<Channel>,
}

impl MetaServerQueryClient {
    /// Connect to the MetaServer at the given address.
    pub(crate) async fn connect(addr: &str) -> Result<Self, ClientError> {
        let endpoint = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{}", addr)
        };

        let channel = Endpoint::from_shared(endpoint.clone())
            .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(5))
            .tcp_nodelay(true)
            .connect()
            .await
            .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?;

        debug!("Connected to MetaServer at {}", endpoint);

        Ok(Self {
            client: MetaServerClient::new(channel),
        })
    }

    /// Query which remote nodes hold the given block hashes.
    ///
    /// Returns `QueryBlockHashesResponse` with `node_blocks` grouping hashes by owning node.
    pub(crate) async fn query_block_hashes(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<QueryBlockHashesResponse, ClientError> {
        let request = QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: block_hashes.to_vec(),
        };

        let response = self
            .client
            .clone()
            .query_block_hashes(request)
            .await
            .map_err(|e| ClientError::RpcFailed(e.to_string()))?;

        Ok(response.into_inner())
    }
}

#[cfg(test)]
impl MetaServerQueryClient {
    pub(crate) fn from_channel(channel: Channel) -> Self {
        Self {
            client: MetaServerClient::new(channel),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use pegaflow_proto::proto::engine::meta_server_server::{
        MetaServer as MetaServerService, MetaServerServer,
    };
    use pegaflow_proto::proto::engine::{
        HealthRequest, HealthResponse, InsertBlockHashesRequest, InsertBlockHashesResponse,
        NodeBlockHashes, QueryBlockHashesRequest, QueryBlockHashesResponse, ResponseStatus,
        ShutdownRequest, ShutdownResponse,
    };
    use tonic::{Request, Response, Status};

    /// Mock MetaServer for unit tests. Pre-populate with `insert()`, then
    /// start with `start_mock_metaserver()`.
    pub(crate) struct MockMetaServer {
        /// (namespace, hash, node)
        entries: Mutex<Vec<(String, Vec<u8>, String)>>,
    }

    impl MockMetaServer {
        pub(crate) fn new() -> Self {
            Self {
                entries: Mutex::new(Vec::new()),
            }
        }

        pub(crate) fn insert(&self, namespace: &str, hash: Vec<u8>, node: &str) {
            self.entries
                .lock()
                .unwrap()
                .push((namespace.to_string(), hash, node.to_string()));
        }
    }

    #[tonic::async_trait]
    impl MetaServerService for MockMetaServer {
        async fn insert_block_hashes(
            &self,
            request: Request<InsertBlockHashesRequest>,
        ) -> Result<Response<InsertBlockHashesResponse>, Status> {
            let req = request.into_inner();
            let mut entries = self.entries.lock().unwrap();
            for hash in &req.block_hashes {
                entries.push((req.namespace.clone(), hash.clone(), req.node.clone()));
            }
            Ok(Response::new(InsertBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                inserted_count: req.block_hashes.len() as u64,
            }))
        }

        async fn query_block_hashes(
            &self,
            request: Request<QueryBlockHashesRequest>,
        ) -> Result<Response<QueryBlockHashesResponse>, Status> {
            let req = request.into_inner();
            let entries = self.entries.lock().unwrap();

            let mut found_count = 0u64;
            let mut existing_hashes = Vec::new();
            let mut node_map: HashMap<String, Vec<Vec<u8>>> = HashMap::new();

            for hash in &req.block_hashes {
                for (ns, stored_hash, node) in entries.iter() {
                    if ns == &req.namespace && stored_hash == hash {
                        found_count += 1;
                        existing_hashes.push(hash.clone());
                        node_map.entry(node.clone()).or_default().push(hash.clone());
                        break;
                    }
                }
            }

            let node_blocks: Vec<NodeBlockHashes> = node_map
                .into_iter()
                .map(|(node, block_hashes)| NodeBlockHashes { node, block_hashes })
                .collect();

            Ok(Response::new(QueryBlockHashesResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
                existing_hashes,
                total_queried: req.block_hashes.len() as u64,
                found_count,
                node_blocks,
            }))
        }

        async fn health(
            &self,
            _request: Request<HealthRequest>,
        ) -> Result<Response<HealthResponse>, Status> {
            Ok(Response::new(HealthResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
            }))
        }

        async fn shutdown(
            &self,
            _request: Request<ShutdownRequest>,
        ) -> Result<Response<ShutdownResponse>, Status> {
            Ok(Response::new(ShutdownResponse {
                status: Some(ResponseStatus {
                    ok: true,
                    message: String::new(),
                }),
            }))
        }
    }

    /// Start a mock MetaServer on a random port. Returns the bound address.
    pub(crate) async fn start_mock_metaserver(mock: MockMetaServer) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MetaServerServer::new(mock))
                .serve(addr)
                .await
                .unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::*;

    #[tokio::test]
    async fn query_returns_found_blocks() {
        let mock = MockMetaServer::new();
        mock.insert("ns", vec![1, 2, 3], "10.0.0.1:50055");
        mock.insert("ns", vec![4, 5, 6], "10.0.0.1:50055");

        let addr = start_mock_metaserver(mock).await;
        let client = MetaServerQueryClient::connect(&format!("127.0.0.1:{}", addr.port()))
            .await
            .unwrap();

        let resp = client
            .query_block_hashes("ns", &[vec![1, 2, 3], vec![4, 5, 6]])
            .await
            .unwrap();

        assert_eq!(resp.found_count, 2);
        assert_eq!(resp.node_blocks.len(), 1);
        assert_eq!(resp.node_blocks[0].node, "10.0.0.1:50055");
        assert_eq!(resp.node_blocks[0].block_hashes.len(), 2);
    }

    #[tokio::test]
    async fn query_returns_empty_for_unknown() {
        let mock = MockMetaServer::new();
        let addr = start_mock_metaserver(mock).await;
        let client = MetaServerQueryClient::connect(&format!("127.0.0.1:{}", addr.port()))
            .await
            .unwrap();

        let resp = client.query_block_hashes("ns", &[vec![99]]).await.unwrap();

        assert_eq!(resp.found_count, 0);
        assert!(resp.node_blocks.is_empty());
    }

    #[tokio::test]
    async fn query_groups_by_node() {
        let mock = MockMetaServer::new();
        mock.insert("ns", vec![1], "10.0.0.1:50055");
        mock.insert("ns", vec![2], "10.0.0.2:50055");

        let addr = start_mock_metaserver(mock).await;
        let client = MetaServerQueryClient::connect(&format!("127.0.0.1:{}", addr.port()))
            .await
            .unwrap();

        let resp = client
            .query_block_hashes("ns", &[vec![1], vec![2]])
            .await
            .unwrap();

        assert_eq!(resp.found_count, 2);
        assert_eq!(resp.node_blocks.len(), 2);
    }
}
