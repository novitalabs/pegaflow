// Binary wrapper for pegaflow-metaserver
// This delegates to the pegaflow-metaserver crate's run() function

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pegaflow_metaserver::run().await
}
