pub mod metric;
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

    /// Hint for typical value size (supports units: kb, mb, gb, tb); tunes cache + allocator
    #[arg(long, value_parser = parse_memory_size)]
    pub hint_value_size: Option<usize>,

    /// Use huge pages for pinned memory pool (faster allocation).
    /// Requires pre-configured huge pages via /proc/sys/vm/nr_hugepages
    #[arg(long, default_value_t = false)]
    pub use_hugepages: bool,

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

    /// Enable SSD cache for sealed blocks. Provide the cache file path to enable.
    #[arg(long)]
    pub ssd_cache_path: Option<String>,

    /// SSD cache capacity (supports units: kb, mb, gb, tb). Default: 512gb
    #[arg(long, default_value = "512gb", value_parser = parse_memory_size)]
    pub ssd_cache_capacity: usize,

    /// SSD write queue depth (max pending write batches). Default: 8
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_WRITE_QUEUE_DEPTH)]
    pub ssd_write_queue_depth: usize,

    /// SSD prefetch queue depth (max pending prefetch batches). Default: 2
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_PREFETCH_QUEUE_DEPTH)]
    pub ssd_prefetch_queue_depth: usize,
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

    if let Some(hint_value_size) = cli.hint_value_size {
        if hint_value_size == 0 {
            return Err("--hint-value-size must be greater than zero when set".into());
        }
        info!("Value size hint set to {} bytes", hint_value_size);
    }

    info!(
        "Creating PegaEngine with pinned memory pool: {:.2} GiB ({} bytes), hugepages={}",
        cli.pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.pool_size,
        cli.use_hugepages
    );

    let ssd_cache_config = cli.ssd_cache_path.as_ref().map(|path| {
        info!(
            "SSD cache enabled: path={}, capacity={:.2} GiB, write_queue={}, prefetch_queue={}",
            path,
            cli.ssd_cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.ssd_write_queue_depth,
            cli.ssd_prefetch_queue_depth,
        );
        pegaflow_core::SsdCacheConfig {
            cache_path: path.into(),
            capacity_bytes: cli.ssd_cache_capacity as u64,
            write_queue_depth: cli.ssd_write_queue_depth,
            prefetch_queue_depth: cli.ssd_prefetch_queue_depth,
        }
    });

    let storage_config = pegaflow_core::StorageConfig {
        enable_lfu_admission: !cli.disable_lfu_admission,
        hint_value_size_bytes: cli.hint_value_size,
        ssd_cache_config,
    };

    if cli.disable_lfu_admission {
        info!("TinyLFU cache admission disabled; falling back to plain LRU inserts");
    }

    // Create Tokio runtime early - needed for OTLP metrics gRPC exporter
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Initialize OTEL metrics BEFORE creating PegaEngine, so that core metrics
    // (pool, cache, save/load) use the real meter provider instead of noop.
    let meter_provider = runtime.block_on(async {
        init_metrics(cli.metrics_otel_endpoint.clone(), cli.metrics_period_secs)
    })?;

    let shutdown = Arc::new(Notify::new());

    runtime.block_on(async move {
        // Create PegaEngine inside tokio runtime context (needed for SSD cache tokio::spawn)
        let (engine, _seal_notify_rx) =
            PegaEngine::new_with_config(cli.pool_size, cli.use_hugepages, storage_config);
        let engine = Arc::new(engine);
        let service = GrpcEngineService::new(
            Arc::clone(&engine),
            Arc::clone(&registry),
            Arc::clone(&shutdown),
        );

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
