//! Common types for inter-node communication.
//!
//! This module contains shared type definitions used across the internode module,
//! including configuration structs and instance representations.

use std::{collections::HashMap, time::Duration};

use k8s_openapi::api::core::v1::Pod;
use log::warn;

/// Default gRPC port for PegaFlow engine instances.
pub const DEFAULT_GRPC_PORT: u16 = 50055;

/// Default label selector for PegaFlow pods.
pub const DEFAULT_PEGAFLOW_LABEL_KEY: &str = "novita.ai/pegaflow";
pub const DEFAULT_PEGAFLOW_LABEL_VALUE: &str = "app";

/// Default connection timeout for gRPC clients.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);

/// Default request timeout for gRPC calls.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

// ============================================================================
// Service Discovery Types
// ============================================================================

/// Configuration for Kubernetes service discovery.
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    /// Whether service discovery is enabled.
    pub enabled: bool,
    /// Label selector for discovering PegaFlow pods.
    /// Key-value pairs that must all match.
    pub selector: HashMap<String, String>,
    /// Interval between checking for pod changes (used on watcher restart).
    pub check_interval: Duration,
    /// gRPC port for PegaFlow engine communication.
    pub grpc_port: u16,
    /// Kubernetes namespace to watch. None means all namespaces.
    pub namespace: Option<String>,
    /// Annotation key for custom gRPC port override.
    pub grpc_port_annotation: String,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        let mut selector = HashMap::new();
        selector.insert(
            DEFAULT_PEGAFLOW_LABEL_KEY.to_string(),
            DEFAULT_PEGAFLOW_LABEL_VALUE.to_string(),
        );

        ServiceDiscoveryConfig {
            enabled: false,
            selector,
            check_interval: Duration::from_secs(60),
            grpc_port: DEFAULT_GRPC_PORT,
            namespace: None,
            grpc_port_annotation: "novita.ai/pegaflow-grpc-port".to_string(),
        }
    }
}

impl ServiceDiscoveryConfig {
    /// Create a new config with the default PegaFlow label selector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable service discovery.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set the gRPC port.
    pub fn with_grpc_port(mut self, port: u16) -> Self {
        self.grpc_port = port;
        self
    }

    /// Set the namespace to watch.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Add a label selector.
    pub fn with_selector(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.selector.insert(key.into(), value.into());
        self
    }
}

/// Represents a discovered PegaFlow engine instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PegaflowInstance {
    /// Pod name (unique identifier within namespace).
    pub name: String,
    /// Pod IP address for gRPC communication.
    pub ip: String,
    /// Kubernetes namespace.
    pub namespace: String,
    /// gRPC port for the PegaFlow engine.
    pub grpc_port: u16,
    /// Pod status (e.g., "Running", "Pending").
    pub status: String,
    /// Whether the pod is ready.
    pub is_ready: bool,
    /// Additional labels from the pod.
    pub labels: HashMap<String, String>,
}

impl PegaflowInstance {
    /// Get the gRPC endpoint URL for this instance.
    pub fn grpc_endpoint(&self) -> String {
        format!("http://{}:{}", self.ip, self.grpc_port)
    }

    /// Get the gRPC address (ip:port) for this instance.
    pub fn grpc_address(&self) -> String {
        format!("{}:{}", self.ip, self.grpc_port)
    }

    /// Check if this instance is healthy (ready and running).
    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    /// Check if a pod matches the given label selector.
    pub(crate) fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    /// Create a PegaflowInstance from a Kubernetes Pod.
    pub fn from_pod(pod: &Pod, config: &ServiceDiscoveryConfig) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let namespace = pod.metadata.namespace.clone().unwrap_or_default();
        let status = pod.status.clone()?;
        let pod_ip = status.pod_ip?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        // Check for custom gRPC port annotation
        let grpc_port = pod
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(&config.grpc_port_annotation))
            .and_then(|port_str| port_str.parse::<u16>().ok())
            .unwrap_or(config.grpc_port);

        // Extract labels
        let labels = pod
            .metadata
            .labels
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();

        Some(PegaflowInstance {
            name,
            ip: pod_ip,
            namespace,
            grpc_port,
            status: pod_status,
            is_ready,
            labels,
        })
    }

    /// Check if a pod should be included based on the selector.
    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.selector.is_empty() {
            warn!("Service discovery selector is empty, no pods will be discovered");
            return false;
        }
        Self::matches_selector(pod, &config.selector)
    }
}

// ============================================================================
// Client Types
// ============================================================================

/// Configuration for PegaFlow gRPC client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Request timeout.
    pub request_timeout: Duration,
    /// Enable TCP_NODELAY.
    pub tcp_nodelay: bool,
    /// HTTP/2 keep-alive interval.
    pub keep_alive_interval: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            tcp_nodelay: true,
            keep_alive_interval: Duration::from_secs(30),
        }
    }
}

/// Error types for PegaFlow client operations.
#[derive(Debug)]
pub enum ClientError {
    /// Failed to connect to the remote endpoint.
    ConnectionFailed(String),
    /// gRPC call failed.
    RpcFailed(String),
    /// No healthy instances available.
    NoHealthyInstances,
    /// Instance not found in registry.
    InstanceNotFound(String),
    /// Response indicates failure.
    ResponseError(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            ClientError::RpcFailed(msg) => write!(f, "RPC failed: {msg}"),
            ClientError::NoHealthyInstances => write!(f, "no healthy PegaFlow instances available"),
            ClientError::InstanceNotFound(name) => write!(f, "instance not found: {name}"),
            ClientError::ResponseError(msg) => write!(f, "response error: {msg}"),
        }
    }
}

impl std::error::Error for ClientError {}

/// Prefetch status from a Query response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryPrefetchStatus {
    /// All blocks are ready in cache.
    Done {
        /// Number of blocks hit in cache.
        hit_blocks: usize,
        /// Number of blocks missing.
        missing_blocks: usize,
    },
    /// Some blocks are being prefetched from SSD/DFS.
    Loading {
        /// Number of blocks hit in cache.
        hit_blocks: usize,
        /// Number of blocks being loaded.
        loading_blocks: usize,
    },
}

/// Result of a Query RPC call.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Whether the request succeeded.
    pub ok: bool,
    /// Error message if failed.
    pub message: String,
    /// Prefetch status.
    pub status: QueryPrefetchStatus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodSpec, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use std::collections::BTreeMap;

    fn create_pegaflow_pod(
        name: &str,
        ip: &str,
        namespace: &str,
        ready: bool,
        custom_port: Option<u16>,
    ) -> Pod {
        let mut labels = BTreeMap::new();
        labels.insert(
            DEFAULT_PEGAFLOW_LABEL_KEY.to_string(),
            DEFAULT_PEGAFLOW_LABEL_VALUE.to_string(),
        );

        let mut annotations = BTreeMap::new();
        if let Some(port) = custom_port {
            annotations.insert("novita.ai/pegaflow-grpc-port".to_string(), port.to_string());
        }

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                namespace: Some(namespace.to_string()),
                labels: Some(labels),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: if ready { "True" } else { "False" }.to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn test_pegaflow_instance_from_pod() {
        let config = ServiceDiscoveryConfig::default();
        let pod = create_pegaflow_pod("pega-1", "10.0.0.1", "default", true, None);

        let instance = PegaflowInstance::from_pod(&pod, &config).unwrap();
        assert_eq!(instance.name, "pega-1");
        assert_eq!(instance.ip, "10.0.0.1");
        assert_eq!(instance.namespace, "default");
        assert_eq!(instance.grpc_port, DEFAULT_GRPC_PORT);
        assert!(instance.is_healthy());
        assert_eq!(instance.grpc_endpoint(), "http://10.0.0.1:50055");
    }

    #[test]
    fn test_pegaflow_instance_with_custom_port() {
        let config = ServiceDiscoveryConfig::default();
        let pod = create_pegaflow_pod("pega-2", "10.0.0.2", "prod", true, Some(50060));

        let instance = PegaflowInstance::from_pod(&pod, &config).unwrap();
        assert_eq!(instance.grpc_port, 50060);
        assert_eq!(instance.grpc_endpoint(), "http://10.0.0.2:50060");
    }

    #[test]
    fn test_pegaflow_instance_not_ready() {
        let config = ServiceDiscoveryConfig::default();
        let pod = create_pegaflow_pod("pega-3", "10.0.0.3", "default", false, None);

        let instance = PegaflowInstance::from_pod(&pod, &config).unwrap();
        assert!(!instance.is_ready);
        assert!(!instance.is_healthy());
    }

    #[test]
    fn test_should_include_with_matching_labels() {
        let config = ServiceDiscoveryConfig::default();
        let pod = create_pegaflow_pod("pega-1", "10.0.0.1", "default", true, None);

        assert!(PegaflowInstance::should_include(&pod, &config));
    }

    #[test]
    fn test_should_include_with_non_matching_labels() {
        let config = ServiceDiscoveryConfig::default();

        // Pod without the required label
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("other-pod".to_string()),
                labels: Some(BTreeMap::new()),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some("10.0.0.1".to_string()),
                phase: Some("Running".to_string()),
                ..Default::default()
            }),
        };

        assert!(!PegaflowInstance::should_include(&pod, &config));
    }

    #[test]
    fn test_config_builder() {
        let config = ServiceDiscoveryConfig::new()
            .enable()
            .with_grpc_port(50060)
            .with_namespace("production")
            .with_selector("env", "prod");

        assert!(config.enabled);
        assert_eq!(config.grpc_port, 50060);
        assert_eq!(config.namespace, Some("production".to_string()));
        assert_eq!(config.selector.get("env"), Some(&"prod".to_string()));
    }

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
        assert_eq!(config.request_timeout, DEFAULT_REQUEST_TIMEOUT);
        assert!(config.tcp_nodelay);
    }

    #[test]
    fn test_query_prefetch_status() {
        let done = QueryPrefetchStatus::Done {
            hit_blocks: 10,
            missing_blocks: 2,
        };
        assert_eq!(
            done,
            QueryPrefetchStatus::Done {
                hit_blocks: 10,
                missing_blocks: 2
            }
        );

        let loading = QueryPrefetchStatus::Loading {
            hit_blocks: 8,
            loading_blocks: 4,
        };
        assert_eq!(
            loading,
            QueryPrefetchStatus::Loading {
                hit_blocks: 8,
                loading_blocks: 4
            }
        );
    }

    #[test]
    fn test_client_error_display() {
        let err = ClientError::ConnectionFailed("timeout".to_string());
        assert!(err.to_string().contains("connection failed"));

        let err = ClientError::NoHealthyInstances;
        assert!(err.to_string().contains("no healthy"));

        let err = ClientError::InstanceNotFound("pega-1".to_string());
        assert!(err.to_string().contains("pega-1"));
    }
}
