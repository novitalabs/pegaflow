use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get, routing::post};
use log::{info, warn};
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use prometheus::{Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::registry::CudaTensorRegistry;

#[derive(Clone)]
struct AppState {
    engine: Arc<PegaEngine>,
    registry: Arc<Mutex<CudaTensorRegistry>>,
    prometheus_registry: Option<Registry>,
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let Some(ref registry) = state.prometheus_registry else {
        return (StatusCode::NOT_FOUND, "metrics not enabled".to_string());
    };
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    (
        StatusCode::OK,
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("# Error encoding metrics: {e}")),
    )
}

#[derive(Serialize)]
struct InstancesResponse {
    instances: Vec<String>,
}

async fn list_instances_handler(State(state): State<AppState>) -> Json<InstancesResponse> {
    let instances = state.engine.list_instance_ids();
    Json(InstancesResponse { instances })
}

#[derive(Deserialize)]
struct CleanupQuery {
    id: Option<String>,
}

#[derive(Serialize)]
struct CleanupResponse {
    removed_instances: Vec<String>,
    removed_tensors: usize,
}

/// POST /instances/cleanup[?id=<instance_id>]
///
/// Without `id`: remove all instances and release all CUDA IPC tensors.
/// With `id`:    remove only the specified instance.
async fn cleanup_handler(
    State(state): State<AppState>,
    Query(query): Query<CleanupQuery>,
) -> impl IntoResponse {
    match query.id {
        None => {
            let removed_tensors = {
                let mut registry = state.registry.lock();
                registry.clear_and_count()
            };
            let removed_instances = state.engine.unregister_all_instances();

            if !removed_instances.is_empty() || removed_tensors > 0 {
                warn!(
                    "Cleanup all: removed {:?}, {} CUDA tensor(s) released",
                    removed_instances, removed_tensors
                );
            } else {
                info!("Cleanup all: nothing to remove");
            }

            (
                StatusCode::OK,
                Json(CleanupResponse {
                    removed_instances,
                    removed_tensors,
                })
                .into_response(),
            )
        }
        Some(instance_id) => {
            let removed_tensors = {
                let mut registry = state.registry.lock();
                registry.drop_instance(&instance_id)
            };

            match state.engine.unregister_instance(&instance_id) {
                Ok(()) => {
                    warn!(
                        "Cleanup instance {}: {} CUDA tensor(s) released",
                        instance_id, removed_tensors
                    );
                    (
                        StatusCode::OK,
                        Json(CleanupResponse {
                            removed_instances: vec![instance_id],
                            removed_tensors,
                        })
                        .into_response(),
                    )
                }
                Err(_) if removed_tensors > 0 => {
                    warn!(
                        "Instance {} not in engine but cleaned {} CUDA tensor(s)",
                        instance_id, removed_tensors
                    );
                    (
                        StatusCode::OK,
                        Json(CleanupResponse {
                            removed_instances: vec![instance_id],
                            removed_tensors,
                        })
                        .into_response(),
                    )
                }
                Err(e) => (StatusCode::NOT_FOUND, format!("{e}").into_response()),
            }
        }
    }
}

/// Start HTTP server for health check, optional Prometheus metrics, and instance management.
pub async fn start_http_server(
    addr: std::net::SocketAddr,
    engine: Arc<PegaEngine>,
    registry: Arc<Mutex<CudaTensorRegistry>>,
    enable_prometheus: bool,
    prometheus_registry: Option<Registry>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        engine,
        registry,
        prometheus_registry: if enable_prometheus {
            prometheus_registry
        } else {
            None
        },
    };

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .route("/instances", get(list_instances_handler))
        .route("/instances/cleanup", post(cleanup_handler));

    if enable_prometheus {
        app = app.route("/metrics", get(metrics_handler));
        info!(
            "Starting HTTP server on {} (/health, /metrics, /instances, /instances/cleanup)",
            addr
        );
    } else {
        info!(
            "Starting HTTP server on {} (/health, /instances, /instances/cleanup)",
            addr
        );
    }

    let app = app.with_state(state);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await
            .ok();
    });

    Ok(handle)
}
