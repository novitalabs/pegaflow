use clap::Parser;
use cudarc::driver::result as cuda_driver;
use parking_lot::Mutex;
use pegaflow_core::PegaEngine;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService};
use pyo3::{types::PyAnyMethods, PyErr, Python};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Notify;
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
}

fn init_tracing() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        "info,pegaflow_server=info,pegaflow_core=info"
            .parse()
            .unwrap()
    });
    let fmt_layer = tracing_subscriber::fmt::layer();
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init();
}

fn format_py_err(err: PyErr) -> String {
    Python::with_gil(|py| err.value_bound(py).to_string())
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
    Python::with_gil(|py| -> pyo3::PyResult<()> {
        let torch = py.import_bound("torch")?;
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

fn main() -> Result<(), Box<dyn Error>> {
    init_tracing();
    let cli = Cli::parse();

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
    let engine = Arc::new(PegaEngine::new());
    let shutdown = Arc::new(Notify::new());

    let service = GrpcEngineService::new(engine, Arc::clone(&registry), Arc::clone(&shutdown));

    // Now create and start Tokio runtime after CUDA initialization
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
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
        Ok(())
    })
}
