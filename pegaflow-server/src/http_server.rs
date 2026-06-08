use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get, routing::post};
use log::{info, warn};
use pegaflow_core::{EngineError, PegaEngine};
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
            let removed_instances =
                match unregister_all_engine_instances(Arc::clone(&state.engine)).await {
                    Ok(removed_instances) => removed_instances,
                    Err(err) => {
                        warn!("Cleanup all failed before CUDA tensor release: {err}");
                        return (cleanup_error_status(&err), format!("{err}").into_response());
                    }
                };
            let removed_tensors = state.registry.clear().await;

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
            match unregister_engine_instance(Arc::clone(&state.engine), instance_id.clone()).await {
                Ok(()) => {
                    let removed_tensors = state.registry.drop_instance(instance_id.clone()).await;
                    warn!(
                        "Cleanup instance {}: {} CUDA tensor(s) released",
                        instance_id, removed_tensors
                    );
                    cleanup_ok_response(instance_id, removed_tensors)
                }
                Err(EngineError::InstanceMissing(_)) => {
                    let removed_tensors = state.registry.drop_instance(instance_id.clone()).await;
                    if removed_tensors > 0 {
                        warn!(
                            "Instance {} not in engine but cleaned {} CUDA tensor(s)",
                            instance_id, removed_tensors
                        );
                        cleanup_ok_response(instance_id, removed_tensors)
                    } else {
                        (
                            StatusCode::NOT_FOUND,
                            format!("instance {instance_id} not found").into_response(),
                        )
                    }
                }
                Err(err) => {
                    warn!(
                        "Cleanup instance {} failed before CUDA tensor release: {}",
                        instance_id, err
                    );
                    (cleanup_error_status(&err), format!("{err}").into_response())
                }
            }
        }
    }
}

async fn unregister_engine_instance(
    engine: Arc<PegaEngine>,
    instance_id: String,
) -> Result<(), EngineError> {
    let task_instance_id = instance_id.clone();
    tokio::task::spawn_blocking(move || engine.unregister_instance(&task_instance_id))
        .await
        .map_err(|err| {
            EngineError::Storage(format!(
                "engine unregister join task failed for instance {instance_id}: {err}"
            ))
        })?
}

async fn unregister_all_engine_instances(
    engine: Arc<PegaEngine>,
) -> Result<Vec<String>, EngineError> {
    tokio::task::spawn_blocking(move || engine.unregister_all_instances())
        .await
        .map_err(|err| {
            EngineError::Storage(format!("engine unregister-all join task failed: {err}"))
        })
}

fn cleanup_error_status(err: &EngineError) -> StatusCode {
    match err {
        EngineError::InvalidArgument(_) => StatusCode::BAD_REQUEST,
        EngineError::InstanceMissing(_) => StatusCode::NOT_FOUND,
        EngineError::WorkerMissing(_, _)
        | EngineError::CudaInit(_)
        | EngineError::Storage(_)
        | EngineError::TopologyMismatch(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
