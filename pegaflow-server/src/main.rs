mod utils;

use clap::Parser;
use cudarc::driver::result as cuda_driver;
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService};
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

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use utils::parse_memory_size;

#[derive(Parser, Debug)]
#[command(
    name = "pega-engine-server",
    about = "PegaEngine gRPC server with CUDA IPC registry"
)]
struct Cli {
    /// Address to bind, e.g. 0.0.0.0:50055
    #[arg(long, default_value = "127.0.0.1:50055")]
    addr: SocketAddr,

    /// Default CUDA device to bind for server-managed contexts
    #[arg(long, default_value_t = 0)]
    device: i32,

    /// Pinned memory pool size (supports units: kb, mb, gb, tb)
    /// Examples: "10gb", "500mb", "1tb"
    #[arg(long, default_value = "30gb", value_parser = parse_memory_size)]
    pool_size: usize,

    /// Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317). Leave empty to disable.
    #[arg(long, default_value = "http://127.0.0.1:4321")]
    metrics_otel_endpoint: Option<String>,

    /// Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    #[arg(long, default_value_t = 5)]
    metrics_period_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
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
    cuda_driver::init().map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize CUDA driver: {err}"),
        )
    })
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
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "failed to initialize python/tensor CUDA runtime: {}",
                format_py_err(err)
            ),
        )
    })
}

fn init_metrics(
    endpoint: Option<String>,
    period_secs: u64,
) -> Result<Option<SdkMeterProvider>, Box<dyn Error>> {
    let Some(endpoint) = endpoint else {
        info!("OTLP metrics disabled (no endpoint configured)");
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    // Initialize CUDA in the main thread before starting Tokio runtime
    init_cuda_driver()?;
    init_python_cuda(cli.device)?;
    info!("CUDA runtime initialized for device {}", cli.device);

    let registry = CudaTensorRegistry::new().map_err(|err| {
        let msg = format_py_err(err);
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize torch CUDA context: {msg}"),
        )
    })?;
    let registry = Arc::new(Mutex::new(registry));

    info!(
        "Creating PegaEngine with pinned memory pool: {:.2} GiB ({} bytes)",
        cli.pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.pool_size
    );
    let engine = Arc::new(PegaEngine::new_with_pool_size(cli.pool_size));
    let shutdown = Arc::new(Notify::new());

    let service = GrpcEngineService::new(engine, Arc::clone(&registry), Arc::clone(&shutdown));

    // Now create and start Tokio runtime after CUDA initialization
    // Using multi-thread runtime since GPU operations now run in dedicated worker threads
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let metrics_endpoint = cli.metrics_otel_endpoint.clone();
    let metrics_period = cli.metrics_period_secs;

    runtime.block_on(async move {
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
