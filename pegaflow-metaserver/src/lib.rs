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
use std::time::Duration;
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

    /// Seconds without heartbeat before a node is hidden from query results.
    #[arg(long, default_value_t = store::DEFAULT_NODE_STALE_SECS)]
    pub node_stale_secs: u64,

    /// Minutes before block ownership records are purged by the lifecycle sweep.
    #[arg(long, default_value_t = store::DEFAULT_TTL_MINUTES)]
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
    info!(
        "Node lifecycle: stale_after={}s ttl={}m",
        cli.node_stale_secs, cli.ttl_minutes
    );
    let ttl_secs = cli
        .ttl_minutes
        .checked_mul(60)
        .ok_or("ttl-minutes is too large")?;
    if cli.ttl_minutes == 0 {
        return Err("ttl-minutes must be greater than 0".into());
    }
    if ttl_secs < cli.node_stale_secs {
        return Err(format!(
            "ttl-minutes ({}) must be >= node-stale-secs ({})",
            cli.ttl_minutes, cli.node_stale_secs
        )
        .into());
    }

    // Initialize metrics
    let (meter_provider, prometheus_registry) = init_metrics()?;

    let store = Arc::new(BlockHashStore::with_config(store::StoreConfig {
        node_stale_after: Duration::from_secs(cli.node_stale_secs),
        ttl: Duration::from_secs(ttl_secs),
    }));

    // Register store observable gauges
    metric::register_store_gauges(&store);

    // Spawn background node lifecycle sweep task.
    {
        let store = Arc::clone(&store);
        let sweep_interval = Duration::from_secs(ttl_secs);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(sweep_interval);
            loop {
                interval.tick().await;
                let stats = store.sweep_expired();
                if !stats.is_empty() {
                    metric::record_sweep(stats);
                    info!(
                        "Node sweep: removed keys={} owners={} nodes={}, {} keys remaining",
                        stats.removed_keys,
                        stats.removed_owners,
                        stats.removed_nodes,
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
