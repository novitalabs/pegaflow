pub mod proto;
pub mod service;
pub mod store;

pub use service::GrpcMetaService;
pub use store::BlockHashStore;

use clap::Parser;
use log::{error, info};
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::Notify;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(
    name = "pegaflow-metaserver",
    about = "PegaFlow MetaServer - manages block hash keys across multi-node instances"
)]
pub struct Cli {
    /// Address to bind, e.g. 0.0.0.0:50056
    #[arg(long, default_value = "127.0.0.1:50056")]
    pub addr: SocketAddr,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

/// Initialize logging with the specified log level
fn init_logging(level: &str) {
    use logforth::append;
    use logforth::filter::EnvFilter;
    use logforth::layout::TextLayout;

    let filter = match level.to_lowercase().as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "info" => "info",
        "warn" => "warn",
        "error" => "error",
        _ => {
            eprintln!("Invalid log level: {}, defaulting to info", level);
            "info"
        }
    };

    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(EnvFilter::from(filter))
                .append(append::Stderr::default().with_layout(TextLayout::default().no_color()))
        })
        .apply();
}

/// Graceful shutdown signal handler
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
        _ = notify.notified() => {
            info!("Received shutdown notification from RPC");
        }
    }
}

/// Run the MetaServer gRPC service
pub async fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    init_logging(&cli.log_level);

    info!("Starting PegaFlow MetaServer");
    info!("Binding to address: {}", cli.addr);

    // Create the block hash store
    let store = Arc::new(BlockHashStore::new());

    // Create shutdown notifier
    let shutdown = Arc::new(Notify::new());

    // Create the gRPC service
    let service = GrpcMetaService::new(store.clone(), shutdown.clone());

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

    info!("MetaServer shut down gracefully");
    Ok(())
}
