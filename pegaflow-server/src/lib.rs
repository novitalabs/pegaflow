pub mod metric;
pub mod proto;
pub mod registry;
pub mod service;
mod utils;

pub use registry::CudaTensorRegistry;
pub use service::GrpcEngineService;

use axum::{Router, routing::get};
use clap::Parser;
use cudarc::driver::result as cuda_driver;
use log::{error, info, warn};
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use prometheus::{Registry, TextEncoder};
use proto::engine::engine_server::EngineServer;
use pyo3::{PyErr, Python, types::PyAnyMethods};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tonic::transport::Server;
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

    /// CUDA devices to initialize (comma-separated, e.g., "0,1,2,3").
    /// If not specified, auto-detects and initializes all available GPUs.
    #[arg(long, value_delimiter = ',')]
    pub devices: Vec<i32>,

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
    #[arg(long, default_value_t = true)]
    pub disable_lfu_admission: bool,

    /// Address for Prometheus metrics HTTP endpoint (e.g. 0.0.0.0:9091). Leave empty to disable.
    #[arg(long, default_value = "0.0.0.0:9091")]
    pub metrics_addr: Option<SocketAddr>,

    /// **DEPRECATED**: Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317).
    /// Use `--metrics-addr` for direct Prometheus export instead.
    #[arg(long)]
    pub metrics_otel_endpoint: Option<String>,

    /// **DEPRECATED**: Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    /// Use `--metrics-addr` for direct Prometheus export instead.
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

    /// SSD write inflight (max concurrent block writes). Default: 2
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_WRITE_INFLIGHT)]
    pub ssd_write_inflight: usize,

    /// SSD prefetch inflight (max concurrent block reads). Default: 16
    #[arg(long, default_value_t = pegaflow_core::DEFAULT_SSD_PREFETCH_INFLIGHT)]
    pub ssd_prefetch_inflight: usize,

    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch). Default: 1500
    #[arg(long, default_value_t = 800)]
    pub max_prefetch_blocks: usize,
}

fn format_py_err(err: PyErr) -> String {
    Python::attach(|py| err.value(py).to_string())
}

fn init_cuda_driver() -> Result<(), std::io::Error> {
    cuda_driver::init()
        .map_err(|err| std::io::Error::other(format!("failed to initialize CUDA driver: {err}")))
}

fn detect_cuda_devices() -> Result<Vec<i32>, std::io::Error> {
    Python::attach(|py| -> pyo3::PyResult<Vec<i32>> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        let device_count: i32 = cuda.call_method0("device_count")?.extract()?;

        // Probe each device ID from 0 to device_count-1 to see if it's available
        let mut available_devices = Vec::new();
        for device_id in 0..device_count {
            // Try to get device properties to verify it's accessible
            match cuda.call_method1("get_device_properties", (device_id,)) {
                Ok(_) => available_devices.push(device_id),
                Err(_) => continue, // Skip unavailable devices
            }
        }
        Ok(available_devices)
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to detect CUDA devices: {}",
            format_py_err(err)
        ))
    })
}

fn init_python_cuda(device_ids: &[i32]) -> Result<(), std::io::Error> {
    if device_ids.is_empty() {
        return Err(std::io::Error::other("no CUDA devices to initialize"));
    }

    Python::attach(|py| -> pyo3::PyResult<()> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        cuda.call_method0("init")?;

        // Initialize CUDA context for each device by performing a real CUDA operation
        // PyTorch uses lazy initialization, so we need to actually allocate something
        // to force context creation on each device
        for &device_id in device_ids {
            let start = std::time::Instant::now();
            cuda.call_method1("set_device", (device_id,))?;

            // Allocate a small tensor to force CUDA context creation on this device
            // This ensures the CUDA driver creates a context for the device
            let device_str = format!("cuda:{}", device_id);
            let empty_args = (vec![1i64],);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("device", device_str)?;
            let _ = torch.call_method("empty", empty_args, Some(&kwargs))?;

            // Synchronize to ensure context is fully initialized
            cuda.call_method0("synchronize")?;

            let elapsed = start.elapsed();
            log::info!(
                "Initialized CUDA context for device {} in {:.2}s",
                device_id,
                elapsed.as_secs_f64()
            );
        }

        // Set the first device as the default
        cuda.call_method1("set_device", (device_ids[0],))?;
        Ok(())
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to initialize python/tensor CUDA runtime: {}",
            format_py_err(err)
        ))
    })
}

struct MetricsState {
    meter_provider: Option<SdkMeterProvider>,
    prometheus_registry: Option<Registry>,
}

fn init_metrics(
    prometheus_enabled: bool,
    otlp_endpoint: Option<String>,
    otlp_period_secs: u64,
) -> Result<MetricsState, Box<dyn Error>> {
    let otlp_endpoint = otlp_endpoint.filter(|s| !s.is_empty());

    // If neither Prometheus nor OTLP is enabled, return empty state
    if !prometheus_enabled && otlp_endpoint.is_none() {
        info!("Metrics disabled (no Prometheus addr or OTLP endpoint configured)");
        return Ok(MetricsState {
            meter_provider: None,
            prometheus_registry: None,
        });
    }

    let mut builder = SdkMeterProvider::builder();
    let mut prometheus_registry = None;

    // Add Prometheus exporter if enabled
    if prometheus_enabled {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()?;
        builder = builder.with_reader(exporter);
        prometheus_registry = Some(registry);
        info!("Prometheus metrics exporter enabled");
    }

    // Add OTLP exporter if endpoint is configured (DEPRECATED)
    if let Some(endpoint) = otlp_endpoint {
        warn!(
            "DEPRECATED: --metrics-otel-endpoint is deprecated and will be removed in a future release. \
            Use --metrics-addr for direct Prometheus export instead."
        );

        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?;

        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(otlp_period_secs))
            .build();

        builder = builder.with_reader(reader);
        info!(
            "OTLP metrics exporter enabled (period={}s)",
            otlp_period_secs
        );
    }

    let meter_provider = builder.build();
    global::set_meter_provider(meter_provider.clone());

    Ok(MetricsState {
        meter_provider: Some(meter_provider),
        prometheus_registry,
    })
}

/// Handler for Prometheus /metrics endpoint
async fn metrics_handler(axum::extract::State(registry): axum::extract::State<Registry>) -> String {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    encoder
        .encode_to_string(&metric_families)
        .unwrap_or_else(|e| format!("# Error encoding metrics: {e}"))
}

/// Main entry point for pegaflow-server
pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    pegaflow_core::logging::init_stdout_colored(&cli.log_level);

    // Initialize CUDA in the main thread before starting Tokio runtime
    init_cuda_driver()?;

    // Determine which devices to initialize
    let devices = if cli.devices.is_empty() {
        // Auto-detect all available devices
        let detected = detect_cuda_devices()?;
        info!(
            "Auto-detected {} CUDA device(s): {:?}",
            detected.len(),
            detected
        );
        detected
    } else {
        info!("Using specified CUDA device(s): {:?}", cli.devices);
        cli.devices.clone()
    };

    if devices.is_empty() {
        return Err("No CUDA devices available".into());
    }

    init_python_cuda(&devices)?;
    info!(
        "CUDA runtime initialized for {} device(s): {:?}",
        devices.len(),
        devices
    );

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
            "SSD cache enabled: path={}, capacity={:.2} GiB, write_queue={}, prefetch_queue={}, write_inflight={}, prefetch_inflight={}",
            path,
            cli.ssd_cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.ssd_write_queue_depth,
            cli.ssd_prefetch_queue_depth,
            cli.ssd_write_inflight,
            cli.ssd_prefetch_inflight,
        );
        pegaflow_core::SsdCacheConfig {
            cache_path: path.into(),
            capacity_bytes: cli.ssd_cache_capacity as u64,
            write_queue_depth: cli.ssd_write_queue_depth,
            prefetch_queue_depth: cli.ssd_prefetch_queue_depth,
            write_inflight: cli.ssd_write_inflight,
            prefetch_inflight: cli.ssd_prefetch_inflight,
        }
    });

    let storage_config = pegaflow_core::StorageConfig {
        enable_lfu_admission: !cli.disable_lfu_admission,
        hint_value_size_bytes: cli.hint_value_size,
        max_prefetch_blocks: cli.max_prefetch_blocks,
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
    let metrics_addr = cli.metrics_addr;
    let metrics_state = runtime.block_on(async {
        init_metrics(
            metrics_addr.is_some(),
            cli.metrics_otel_endpoint.clone(),
            cli.metrics_period_secs,
        )
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

        // Spawn background GC task for stale inflight blocks
        {
            let engine = Arc::clone(&engine);
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                const GC_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
                const GC_MAX_AGE: Duration = Duration::from_secs(3600); // 60 minutes
                let mut interval = tokio::time::interval(GC_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let cleaned = engine.gc_stale_inflight(GC_MAX_AGE);
                            if cleaned > 0 {
                                info!("Inflight GC: cleaned {} stale blocks", cleaned);
                            }
                        }
                        _ = shutdown.notified() => {
                            info!("Inflight GC task shutting down");
                            break;
                        }
                    }
                }
            });
            info!("Inflight GC task started (interval=5m, max_age=60m)");
        }

        // Start Prometheus HTTP server if configured
        let metrics_server_handle = if let (Some(metrics_addr), Some(registry)) =
            (cli.metrics_addr, metrics_state.prometheus_registry.clone())
        {
            let app = Router::new()
                .route("/metrics", get(metrics_handler))
                .with_state(registry);

            let listener = tokio::net::TcpListener::bind(metrics_addr).await?;
            info!("Starting Prometheus metrics server on {}", metrics_addr);

            let shutdown = Arc::clone(&shutdown);
            let handle = tokio::spawn(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        shutdown.notified().await;
                    })
                    .await
                    .ok();
            });
            Some(handle)
        } else {
            None
        };

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

        info!("PegaEngine gRPC server listening on {}", cli.addr);

        if let Err(err) = Server::builder()
            .add_service(EngineServer::new(service))
            .serve_with_shutdown(cli.addr, shutdown_signal)
            .await
        {
            error!("Server error: {err}");
            return Err(err.into());
        }

        info!("Server stopped");

        // Stop metrics server if running
        if let Some(handle) = metrics_server_handle {
            shutdown.notify_waiters();
            let _ = handle.await;
        }

        // Flush metrics before exit
        if let Some(provider) = metrics_state.meter_provider
            && let Err(err) = provider.shutdown()
        {
            error!("Failed to shutdown metrics provider: {err}");
        }

        Ok(())
    })
}
