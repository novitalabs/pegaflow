//! Kubernetes service discovery for PegaFlow instances.
//!
//! This module provides automatic discovery of PegaFlow engine instances
//! running as Kubernetes pods. It watches pods with the label `novita.ai/pegaflow: app`
//! and maintains a registry of available instances for gRPC communication.

use std::{sync::Arc, time::Duration};

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    Client,
    api::Api,
    runtime::{
        WatchStreamExt,
        watcher::{Config, watcher},
    },
};
use log::{debug, error, info, warn};
use tokio::{task, time};

use super::registry::InstanceRegistry;
use super::types::{PegaflowInstance, ServiceDiscoveryConfig};

/// Start Kubernetes service discovery for PegaFlow instances.
///
/// This function starts a background task that watches for pods with the
/// configured label selector and maintains the instance registry.
///
/// # Arguments
///
/// * `config` - Service discovery configuration
/// * `registry` - Registry to update with discovered instances
///
/// # Returns
///
/// A `JoinHandle` for the background task, or an error if initialization fails.
///
/// # Example
///
/// ```rust,no_run
/// use pegaflow_core::internode::{
///     ServiceDiscoveryConfig, InstanceRegistry, start_service_discovery,
/// };
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let registry = Arc::new(InstanceRegistry::new());
///     let config = ServiceDiscoveryConfig::new()
///         .enable()
///         .with_namespace("default");
///
///     let handle = start_service_discovery(config, registry.clone()).await?;
///
///     // Get all healthy instances
///     for instance in registry.healthy_instances() {
///         println!("Found PegaFlow at: {}", instance.grpc_endpoint());
///     }
///
///     Ok(())
/// }
/// ```
pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    registry: Arc<InstanceRegistry>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Discovery(
            kube::error::DiscoveryError::MissingResource(
                "Service discovery is disabled".to_string(),
            ),
        ));
    }

    // Initialize rustls crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    let client = Client::try_default().await?;

    let label_selector = config
        .selector
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");

    info!(
        "Starting PegaFlow service discovery | selector: '{}' | namespace: {:?} | grpc_port: {}",
        label_selector,
        config.namespace.as_deref().unwrap_or("all"),
        config.grpc_port
    );

    let handle = task::spawn(async move {
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("PegaFlow service discovery initialized");

        let config_arc = Arc::new(config);

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

        loop {
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            let config_clone = Arc::clone(&config_arc);

            let filtered_stream = watcher_stream.filter_map(move |obj_res| {
                let config_inner = Arc::clone(&config_clone);

                async move {
                    match obj_res {
                        Ok(pod) => {
                            if PegaflowInstance::should_include(&pod, &config_inner) {
                                Some(Ok(pod))
                            } else {
                                None
                            }
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
            });

            let registry_clone = Arc::clone(&registry);
            let config_clone2 = Arc::clone(&config_arc);

            match filtered_stream
                .try_for_each(move |pod| {
                    let registry_inner = Arc::clone(&registry_clone);
                    let config_inner = Arc::clone(&config_clone2);

                    async move {
                        if let Some(instance) = PegaflowInstance::from_pod(&pod, &config_inner) {
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(&instance, &registry_inner);
                            } else {
                                handle_pod_event(&instance, &registry_inner);
                            }
                        }
                        Ok(())
                    }
                })
                .await
            {
                Ok(_) => {
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in PegaFlow service discovery watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            warn!(
                "PegaFlow service discovery watcher exited, restarting in {} seconds",
                config_arc.check_interval.as_secs()
            );
            time::sleep(config_arc.check_interval).await;
        }
    });

    Ok(handle)
}

/// Handle a pod event (add or update).
fn handle_pod_event(instance: &PegaflowInstance, registry: &InstanceRegistry) {
    if instance.is_healthy() {
        let existing = registry.get(&instance.name);
        let is_new = existing.is_none();

        registry.upsert(instance.clone());

        if is_new {
            info!(
                "Discovered PegaFlow instance: {} | endpoint: {} | namespace: {}",
                instance.name,
                instance.grpc_endpoint(),
                instance.namespace
            );
        } else {
            debug!(
                "Updated PegaFlow instance: {} | endpoint: {}",
                instance.name,
                instance.grpc_endpoint()
            );
        }
    } else {
        // Pod not healthy, remove from registry if present
        if registry.get(&instance.name).is_some() {
            registry.remove(&instance.name);
            info!(
                "PegaFlow instance {} became unhealthy (status: {}, ready: {}), removed",
                instance.name, instance.status, instance.is_ready
            );
        }
    }
}

/// Handle a pod deletion event.
fn handle_pod_deletion(instance: &PegaflowInstance, registry: &InstanceRegistry) {
    if let Some(removed) = registry.remove(&instance.name) {
        info!(
            "PegaFlow instance {} deleted | endpoint: {} | namespace: {}",
            removed.name,
            removed.grpc_endpoint(),
            removed.namespace
        );
    } else {
        debug!(
            "Pod deletion event for untracked instance: {}",
            instance.name
        );
    }
}
