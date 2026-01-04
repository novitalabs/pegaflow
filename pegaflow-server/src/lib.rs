pub mod proto;
pub mod registry;
pub mod service;
mod utils;

pub use registry::CudaTensorRegistry;
pub use service::GrpcEngineService;

use clap::Parser;
use cudarc::driver::result as cuda_driver;
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use proto::engine::engine_server::EngineServer;
use pyo3::{types::PyAnyMethods, PyErr, Python};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber::{
    fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use utils::parse_memory_size;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(
    name = "pega-engine-server",
    about = "PegaEngine gRPC server with CUDA IPC registry"
)]
pub struct Cli {
    /// Address to bind, e.g. 0.0.0.0:50055
    #[arg(long, default_value = "127.0.0.1:50055")]
    pub addr: SocketAddr,

    /// Default CUDA device to bind for server-managed contexts
    #[arg(long, default_value_t = 0)]
    pub device: i32,

    /// Pinned memory pool size (supports units: kb, mb, gb, tb)
    /// Examples: "10gb", "500mb", "1tb"
    #[arg(long, default_value = "30gb", value_parser = parse_memory_size)]
    pub pool_size: usize,

    /// Use huge pages for pinned memory pool (faster allocation).
    /// Requires pre-configured huge pages via /proc/sys/vm/nr_hugepages
    #[arg(long, default_value_t = false)]
    pub use_hugepages: bool,

    /// Enable pre-eviction background monitoring
    #[arg(long, default_value_t = false)]
    pub pre_evict: bool,

    /// Pre-eviction threshold: start evicting when free space drops below this (supports units: kb, mb, gb, tb)
    #[arg(long, default_value = "5gb", value_parser = parse_memory_size)]
    pub pre_evict_threshold: usize,

    /// Pre-eviction target: evict until free space reaches this target (supports units: kb, mb, gb, tb)
    #[arg(long, default_value = "8gb", value_parser = parse_memory_size)]
    pub pre_evict_target: usize,

    /// Pre-eviction check interval in milliseconds
    #[arg(long, default_value_t = 100)]
    pub pre_evict_interval_ms: u64,

    /// Disable TinyLFU admission (falls back to plain LRU inserts)
    #[arg(long, default_value_t = false)]
    pub disable_lfu_admission: bool,

    /// Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317). Leave empty to disable.
    #[arg(long, default_value = "http://127.0.0.1:4321")]
    pub metrics_otel_endpoint: Option<String>,

    /// Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    #[arg(long, default_value_t = 5)]
    pub metrics_period_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Redis URL for seal offload. If set, sealed blocks' keys are written to Redis.
    #[arg(long)]
    pub redis_url: Option<String>,

    /// DFS root directory for block offload (e.g., /mnt/dfs/pega)
    #[arg(long, default_value = "/tmp/pega")]
    pub dfs_root: String,

    /// DFS quota in bytes (triggers eviction when exceeded)
    #[arg(long, default_value = "50gb", value_parser = parse_memory_size)]
    pub dfs_quota: usize,

    /// DFS quota scan interval in milliseconds
    #[arg(long, default_value_t = 100)]
    pub dfs_scan_interval_ms: u64,
}

fn init_tracing(log_level: &str) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| log_level.parse().unwrap());
    let fmt_layer = tracing_subscriber::fmt::layer().with_span_events(FmtSpan::CLOSE);
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init();
}

fn format_py_err(err: PyErr) -> String {
    Python::attach(|py| err.value(py).to_string())
}

fn init_cuda_driver() -> Result<(), std::io::Error> {
    cuda_driver::init()
        .map_err(|err| std::io::Error::other(format!("failed to initialize CUDA driver: {err}")))
}

fn init_python_cuda(device_id: i32) -> Result<(), std::io::Error> {
    Python::attach(|py| -> pyo3::PyResult<()> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        cuda.call_method0("init")?;
        cuda.call_method1("set_device", (device_id,))?;
        Ok(())
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to initialize python/tensor CUDA runtime: {}",
            format_py_err(err)
        ))
    })
}

fn init_metrics(
    endpoint: Option<String>,
    period_secs: u64,
) -> Result<Option<SdkMeterProvider>, Box<dyn Error>> {
    let Some(endpoint) = endpoint.filter(|s| !s.is_empty()) else {
        info!("OTLP metrics disabled (no endpoint configured or endpoint is empty)");
        return Ok(None);
    };

    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
        .with_interval(Duration::from_secs(period_secs))
        .build();

    let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();

    global::set_meter_provider(meter_provider.clone());
    info!("OTLP metrics exporter enabled (period={}s)", period_secs);

    Ok(Some(meter_provider))
}

/// Main entry point for pegaflow-server
pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    // Initialize CUDA in the main thread before starting Tokio runtime
    init_cuda_driver()?;
    init_python_cuda(cli.device)?;
    info!("CUDA runtime initialized for device {}", cli.device);

    let registry = CudaTensorRegistry::new().map_err(|err| {
        let msg = format_py_err(err);
        std::io::Error::other(format!("failed to initialize torch CUDA context: {msg}"))
    })?;
    let registry = Arc::new(Mutex::new(registry));

    info!(
        "Creating PegaEngine with pinned memory pool: {:.2} GiB ({} bytes), hugepages={}",
        cli.pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.pool_size,
        cli.use_hugepages
    );

    let pre_evict_config = if cli.pre_evict {
        // Validate: target must be greater than threshold to ensure eviction frees memory
        if cli.pre_evict_target <= cli.pre_evict_threshold {
            return Err(format!(
                "--pre-evict-target ({}) must be greater than --pre-evict-threshold ({})",
                cli.pre_evict_target, cli.pre_evict_threshold
            )
            .into());
        }
        // Validate: target must not exceed pool capacity
        if cli.pre_evict_target > cli.pool_size {
            return Err(format!(
                "--pre-evict-target ({}) must not exceed --pool-size ({})",
                cli.pre_evict_target, cli.pool_size
            )
            .into());
        }

        info!(
            "Pre-eviction enabled: threshold={:.2} GiB, target={:.2} GiB, interval={}ms",
            cli.pre_evict_threshold as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.pre_evict_target as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.pre_evict_interval_ms
        );
        pegaflow_core::PreEvictConfig::new(
            cli.pre_evict_threshold as u64,
            cli.pre_evict_target as u64,
            cli.pre_evict_interval_ms,
        )
    } else {
        pegaflow_core::PreEvictConfig::default()
    };

    let storage_config = pegaflow_core::StorageConfig {
        pre_evict_config,
        enable_lfu_admission: !cli.disable_lfu_admission,
    };

    if cli.disable_lfu_admission {
        info!("TinyLFU cache admission disabled; falling back to plain LRU inserts");
    }

    let (engine, seal_notify_rx) =
        PegaEngine::new_with_config(cli.pool_size, cli.use_hugepages, storage_config);
    let engine = Arc::new(engine);
    let shutdown = Arc::new(Notify::new());

    // Now create and start Tokio runtime after CUDA initialization
    // Using multi-thread runtime since GPU operations now run in dedicated worker threads
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let metrics_endpoint = cli.metrics_otel_endpoint.clone();
    let metrics_period = cli.metrics_period_secs;

    runtime.block_on(async move {
        // Create service - with or without DFS support depending on redis config
        let service = if let Some(url) = &cli.redis_url {
            // Spawn the DFS offload task
            let config = pegaflow_core::DfsOffloadConfig {
                dfs_root: cli.dfs_root.clone().into(),
                quota_bytes: cli.dfs_quota as u64,
                scan_interval_ms: cli.dfs_scan_interval_ms,
                ..Default::default()
            };
            let _offload_handles =
                pegaflow_core::spawn_dfs_offload_task(seal_notify_rx, url, config).await?;

            // Create Redis connection for query prefetch
            let redis_client = redis::Client::open(url.as_str()).map_err(|e| {
                std::io::Error::other(format!("Failed to create Redis client: {e}"))
            })?;
            let redis_conn = redis_client
                .get_multiplexed_async_connection()
                .await
                .map_err(|e| std::io::Error::other(format!("Failed to connect to Redis: {e}")))?;

            info!("DFS prefetch enabled with Redis at {}", url);
            GrpcEngineService::with_dfs(
                Arc::clone(&engine),
                Arc::clone(&registry),
                Arc::clone(&shutdown),
                redis_conn,
                cli.dfs_root.into(),
            )
        } else {
            // Drop the receiver since we're not using DFS offload
            drop(seal_notify_rx);
            GrpcEngineService::new(
                Arc::clone(&engine),
                Arc::clone(&registry),
                Arc::clone(&shutdown),
            )
        };

        // Initialize OTEL metrics (requires Tokio runtime for gRPC)
        let meter_provider = init_metrics(metrics_endpoint, metrics_period)?;

        let shutdown_signal = {
            let notify = Arc::clone(&shutdown);
            async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl+C received, shutting down");
                    }
                    _ = notify.notified() => {
                        info!("Shutdown requested via RPC");
                    }
                }
            }
        };

        info!("Starting PegaEngine gRPC server on {}", cli.addr);

        if let Err(err) = Server::builder()
            .add_service(EngineServer::new(service))
            .serve_with_shutdown(cli.addr, shutdown_signal)
            .await
        {
            error!("Server error: {err}");
            return Err(err.into());
        }

        info!("Server stopped");

        // Flush metrics before exit
        if let Some(provider) = meter_provider {
            if let Err(err) = provider.shutdown() {
                error!("Failed to shutdown metrics provider: {err}");
            }
        }

        Ok(())
    })
}
