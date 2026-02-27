use std::process;

#[tokio::main]
async fn main() {
    if let Err(e) = pegaflow_metaserver::run().await {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
