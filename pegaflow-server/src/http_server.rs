use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get, routing::post};
use log::{info, warn};
use pegaflow_core::PegaEngine;
use prometheus::{Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::registry::RegistryHandle;

#[derive(Clone)]
struct AppState {
    engine: Arc<PegaEngine>,
    registry: RegistryHandle,
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

#[derive(Serialize)]
struct MemoryCacheCleanupResponse {
    evicted_blocks: usize,
    evicted_bytes: u64,
    reclaimed_bytes: u64,
    still_referenced_blocks: u64,
}

/// POST /instances/cleanup[?id=<instance_id>]
///
/// Without `id`: remove all instances and release all CUDA IPC tensors.
/// With `id`:    remove only the specified instance.
///
/// Releasing CUDA IPC tensors takes the GIL and runs a blocking
/// `torch.cuda.empty_cache()`. That work runs on the dedicated registry thread
/// behind [`RegistryHandle`]; the handler only `.await`s the reply, so a
/// slow/wedged cleanup never occupies an async worker (the outage where a few
/// `cleanup` calls hung every endpoint, `/health` and `/metrics` included).
async fn cleanup_handler(
    State(state): State<AppState>,
    Query(query): Query<CleanupQuery>,
) -> impl IntoResponse {
    match query.id {
        None => {
            let engine = Arc::clone(&state.engine);
            let registry = state.registry.clone();
            let (removed_instances, removed_tensors) = tokio::spawn(async move {
                let cleanup = registry.clear().await;
                let removed_instances = engine.unregister_all_instances();
                let removed_tensors = cleanup.tensor_count();
                registry.finish_cleanup(cleanup).await;
                (removed_instances, removed_tensors)
            })
            .await
            .expect("cleanup-all task failed");

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
            let engine = Arc::clone(&state.engine);
            let registry = state.registry.clone();
            let cleanup_id = instance_id.clone();
            let (removed_tensors, unregister) = tokio::spawn(async move {
                let cleanup = registry.drop_instance(cleanup_id.clone()).await;
                let removed_tensors = cleanup.tensor_count();
                let unregister = engine.unregister_instance(&cleanup_id);
                registry.finish_cleanup(cleanup).await;
                (removed_tensors, unregister)
            })
            .await
            .expect("instance cleanup task failed");
            match unregister {
                Ok(()) => {
                    warn!(
                        "Cleanup instance {}: {} CUDA tensor(s) released",
                        instance_id, removed_tensors
                    );
                    cleanup_ok_response(instance_id, removed_tensors)
                }
                Err(_) if removed_tensors > 0 => {
                    warn!(
                        "Instance {} not in engine but cleaned {} CUDA tensor(s)",
                        instance_id, removed_tensors
                    );
                    cleanup_ok_response(instance_id, removed_tensors)
                }
                Err(e) => (StatusCode::NOT_FOUND, format!("{e}").into_response()),
            }
        }
    }
}

fn cleanup_ok_response(
    instance_id: String,
    removed_tensors: usize,
) -> (StatusCode, axum::response::Response) {
    (
        StatusCode::OK,
        Json(CleanupResponse {
            removed_instances: vec![instance_id],
            removed_tensors,
        })
        .into_response(),
    )
}

/// POST /cache/memory/cleanup
///
/// Drops resident in-memory cache blocks while preserving backing-store data.
async fn cleanup_memory_cache_handler(
    State(state): State<AppState>,
) -> Json<MemoryCacheCleanupResponse> {
    let stats = state.engine.cleanup_memory_cache();
    Json(MemoryCacheCleanupResponse {
        evicted_blocks: stats.evicted_blocks,
        evicted_bytes: stats.evicted_bytes,
        reclaimed_bytes: stats.reclaimed_bytes,
        still_referenced_blocks: stats.still_referenced_blocks,
    })
}

/// Start HTTP server for health check, optional Prometheus metrics, and instance management.
pub async fn start_http_server(
    addr: std::net::SocketAddr,
    engine: Arc<PegaEngine>,
    registry: RegistryHandle,
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
        .route("/instances/cleanup", post(cleanup_handler))
        .route("/cache/memory/cleanup", post(cleanup_memory_cache_handler));

    if enable_prometheus {
        app = app.route("/metrics", get(metrics_handler));
        info!(
            "Starting HTTP server on {} (/health, /metrics, /instances, /instances/cleanup, /cache/memory/cleanup)",
            addr
        );
    } else {
        info!(
            "Starting HTTP server on {} (/health, /instances, /instances/cleanup, /cache/memory/cleanup)",
            addr
        );
    }

    let app = app.with_state(state);

    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await
        {
            warn!("HTTP server stopped with error: {err}");
        }
    });

    Ok(handle)
}
