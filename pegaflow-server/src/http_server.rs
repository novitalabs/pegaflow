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

use crate::teardown::InstanceTeardown;

#[derive(Clone)]
struct AppState {
    engine: Arc<PegaEngine>,
    teardown: Arc<InstanceTeardown>,
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
    /// Instances whose GPU workers failed to drain; their CUDA IPC mappings
    /// are deliberately leaked instead of being unmapped under stale tasks.
    leaked_instances: Vec<String>,
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
/// Both go through [`InstanceTeardown`]: fence the instance, drain its GPU
/// worker queues, then unmap. Releasing CUDA IPC tensors takes the GIL and
/// runs a blocking `torch.cuda.empty_cache()`. That work runs on the
/// dedicated registry thread; the handler only `.await`s the reply, so a
/// slow/wedged cleanup never occupies an async worker (the outage where a few
/// `cleanup` calls hung every endpoint, `/health` and `/metrics` included).
async fn cleanup_handler(
    State(state): State<AppState>,
    Query(query): Query<CleanupQuery>,
) -> impl IntoResponse {
    match query.id {
        None => {
            let outcome = state.teardown.cleanup_all("http cleanup").await;

            if !outcome.removed_instances.is_empty() || outcome.dropped_tensors > 0 {
                warn!(
                    "Cleanup all: removed {:?}, {} CUDA tensor(s) released, leaked {:?}",
                    outcome.removed_instances, outcome.dropped_tensors, outcome.leaked_instances
                );
            } else {
                info!("Cleanup all: nothing to remove");
            }

            let status = if outcome.leaked_instances.is_empty() {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(CleanupResponse {
                    removed_instances: outcome.removed_instances,
                    removed_tensors: outcome.dropped_tensors,
                    leaked_instances: outcome.leaked_instances,
                })
                .into_response(),
            )
        }
        Some(instance_id) => {
            let outcome = state.teardown.cleanup(&instance_id, "http cleanup").await;
            if !outcome.drained {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(CleanupResponse {
                        removed_instances: vec![instance_id.clone()],
                        removed_tensors: 0,
                        leaked_instances: vec![instance_id],
                    })
                    .into_response(),
                );
            }
            if outcome.instance_found || outcome.dropped_tensors > 0 {
                warn!(
                    "Cleanup instance {}: {} CUDA tensor(s) released",
                    instance_id, outcome.dropped_tensors
                );
                (
                    StatusCode::OK,
                    Json(CleanupResponse {
                        removed_instances: vec![instance_id],
                        removed_tensors: outcome.dropped_tensors,
                        leaked_instances: Vec::new(),
                    })
                    .into_response(),
                )
            } else {
                (
                    StatusCode::NOT_FOUND,
                    format!("instance {instance_id} not found").into_response(),
                )
            }
        }
    }
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
    teardown: Arc<InstanceTeardown>,
    enable_prometheus: bool,
    prometheus_registry: Option<Registry>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        engine,
        teardown,
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
