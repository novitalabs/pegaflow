pub mod http_server;
pub mod metric;
pub mod proto;
pub mod service;
pub mod store;

pub use service::GrpcMetaService;
pub use store::BlockHashStore;

use clap::Parser;
use log::{error, info};
use opentelemetry::global;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use prometheus::Registry;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Notify;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(
    name = "pegaflow-metaserver",
    version,
    about = "PegaFlow MetaServer - manages block hash keys across multi-node instances"
)]
pub struct Cli {
    /// Address to bind, e.g. 0.0.0.0:50056
    #[arg(long, default_value = "127.0.0.1:50056")]
    pub addr: SocketAddr,

    /// HTTP server address for health check and Prometheus metrics.
    #[arg(long, default_value = "0.0.0.0:9092")]
    pub http_addr: SocketAddr,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Cache entry TTL in minutes
    #[arg(long, default_value = "120")]
    pub ttl_minutes: u64,
}

fn init_metrics() -> Result<(SdkMeterProvider, Registry), Box<dyn Error>> {
    let registry = Registry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()?;
    let meter_provider = SdkMeterProvider::builder().with_reader(exporter).build();
    global::set_meter_provider(meter_provider.clone());
    info!("Prometheus metrics exporter enabled");

    Ok((meter_provider, registry))
}

/// Wait for OS termination signal (Ctrl+C or SIGTERM), then notify dependents.
async fn shutdown_signal(notify: Arc<Notify>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, initiating shutdown");
        }
        _ = terminate => {
            info!("Received SIGTERM, initiating shutdown");
        }
    }

    notify.notify_waiters();
}

/// Run the MetaServer gRPC service
pub async fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    pegaflow_common::logging::init_stderr(&cli.log_level);
    info!(
        "Starting pegaflow-metaserver v{}",
        env!("CARGO_PKG_VERSION")
    );

    info!("Starting PegaFlow MetaServer");
    info!("Binding to address: {}", cli.addr);
    info!("Cache entry TTL: {} minutes", cli.ttl_minutes);

    // Initialize metrics
    let (meter_provider, prometheus_registry) = init_metrics()?;

    // Create the block hash store with TTL
    let store = Arc::new(BlockHashStore::with_ttl(cli.ttl_minutes));

    // Register store observable gauges
    metric::register_store_gauges(&store);

    // Spawn background TTL sweep task (every 10 minutes)
    {
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10 * 60));
            loop {
                interval.tick().await;
                let removed = store.sweep_expired();
                if removed > 0 {
                    metric::record_ttl_sweep(removed as u64);
                    info!(
                        "TTL sweep: removed {} stale block keys, {} remaining",
                        removed,
                        store.entry_count()
                    );
                }
            }
        });
    }

    // Create shutdown notifier
    let shutdown = Arc::new(Notify::new());

    // Start HTTP server for health check and metrics
    let _http_handle =
        http_server::start_http_server(cli.http_addr, prometheus_registry, Arc::clone(&shutdown))
            .await?;

    // Create the gRPC service
    let service = GrpcMetaService::new(store.clone());

    info!("MetaServer initialized successfully");
    info!("Listening on {}", cli.addr);

    // Start the gRPC server
    let server_future = Server::builder()
        .add_service(MetaServerServer::new(service))
        .serve_with_shutdown(cli.addr, shutdown_signal(shutdown.clone()));

    if let Err(e) = server_future.await {
        error!("Server error: {}", e);
        return Err(Box::new(e));
    }

    // Flush metrics before exit
    if let Err(err) = meter_provider.shutdown() {
        error!("Failed to shutdown metrics provider: {err}");
    }

    info!("MetaServer shut down gracefully");
    Ok(())
}
