use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::delete, routing::get, routing::post};
use log::{info, warn};
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use prometheus::{Registry, TextEncoder};
use serde::Serialize;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::registry::CudaTensorRegistry;

/// Shared state for HTTP handlers that need engine access.
#[derive(Clone)]
struct AppState {
    engine: Arc<PegaEngine>,
    registry: Arc<Mutex<CudaTensorRegistry>>,
    prometheus_registry: Option<Registry>,
}

/// Handler for health check endpoint
async fn health_handler() -> &'static str {
    "ok"
}

/// Handler for Prometheus /metrics endpoint
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

/// Handler for listing all registered instances.
async fn list_instances_handler(State(state): State<AppState>) -> Json<InstancesResponse> {
    let instances = state.engine.list_instance_ids();
    Json(InstancesResponse { instances })
}

#[derive(Serialize)]
struct CleanupResponse {
    removed_instances: Vec<String>,
    removed_tensors: usize,
}

/// Handler for cleaning up all instances and their CUDA tensors.
async fn cleanup_instances_handler(State(state): State<AppState>) -> Json<CleanupResponse> {
    // 1. Drop all CUDA IPC tensors first (requires GIL, gc.collect, empty_cache)
    let removed_tensors = {
        let mut registry = state.registry.lock();
        registry.clear_and_count()
    };

    // 2. Remove all instances from the engine
    let removed_instances = state.engine.unregister_all_instances();

    if !removed_instances.is_empty() || removed_tensors > 0 {
        warn!(
            "Cleanup: removed {} instance(s) {:?}, {} CUDA tensor(s)",
            removed_instances.len(),
            removed_instances,
            removed_tensors
        );
    } else {
        info!("Cleanup: nothing to clean");
    }

    Json(CleanupResponse {
        removed_instances,
        removed_tensors,
    })
}

#[derive(Serialize)]
struct DeleteInstanceResponse {
    instance_id: String,
    removed_tensors: usize,
}

/// Handler for deleting a specific instance and its CUDA tensors.
async fn delete_instance_handler(
    State(state): State<AppState>,
    Path(instance_id): Path<String>,
) -> impl IntoResponse {
    // 1. Drop CUDA IPC tensors for this instance
    let removed_tensors = {
        let mut registry = state.registry.lock();
        registry.drop_instance(&instance_id)
    };

    // 2. Remove instance from the engine
    match state.engine.unregister_instance(&instance_id) {
        Ok(()) => {
            warn!(
                "Deleted instance {}, {} CUDA tensor(s) released",
                instance_id, removed_tensors
            );
            (
                StatusCode::OK,
                Json(DeleteInstanceResponse {
                    instance_id,
                    removed_tensors,
                })
                .into_response(),
            )
        }
        Err(_) if removed_tensors > 0 => {
            // Tensors were cleaned but instance was already gone from engine
            warn!(
                "Instance {} not in engine but cleaned {} CUDA tensor(s)",
                instance_id, removed_tensors
            );
            (
                StatusCode::OK,
                Json(DeleteInstanceResponse {
                    instance_id,
                    removed_tensors,
                })
                .into_response(),
            )
        }
        Err(e) => (StatusCode::NOT_FOUND, format!("{e}").into_response()),
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
        .route("/instances/cleanup", post(cleanup_instances_handler))
        .route("/instances/{instance_id}", delete(delete_instance_handler));

    if enable_prometheus {
        app = app.route("/metrics", get(metrics_handler));
        info!(
            "Starting HTTP server on {} (/health, /metrics, /instances, /instances/cleanup, /instances/:id)",
            addr
        );
    } else {
        info!(
            "Starting HTTP server on {} (/health, /instances, /instances/cleanup, /instances/:id)",
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
