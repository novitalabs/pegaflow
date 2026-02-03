/// Example client demonstrating how to use the PegaFlow MetaServer gRPC APIs
///
/// Usage:
///   1. Start the metaserver:
///      cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056
///   2. Run this example:
///      cargo run -p pegaflow-metaserver --example basic_client
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use pegaflow_proto::proto::engine::{
    HealthRequest, InsertBlockHashesRequest, QueryBlockHashesRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to the metaserver
    let endpoint = "http://127.0.0.1:50056";
    println!("Connecting to MetaServer at {}", endpoint);
    let mut client = MetaServerClient::connect(endpoint).await?;

    // 1. Health check
    println!("\n1. Checking health...");
    let response = client.health(HealthRequest {}).await?;
    println!("   Health response: {:?}", response.into_inner().status);

    // 2. Insert block hashes
    println!("\n2. Inserting block hashes...");
    let namespace = "model-llama3";
    let block_hashes = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],         // Block hash 1
        vec![9, 10, 11, 12, 13, 14, 15, 16],  // Block hash 2
        vec![17, 18, 19, 20, 21, 22, 23, 24], // Block hash 3
    ];

    let request = InsertBlockHashesRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.clone(),
    };

    let response = client.insert_block_hashes(request).await?;
    let inner = response.into_inner();
    println!("   Status: {:?}", inner.status);
    println!("   Inserted count: {}", inner.inserted_count);

    // 3. Query existing hashes
    println!("\n3. Querying block hashes (all should exist)...");
    let request = QueryBlockHashesRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.clone(),
    };

    let response = client.query_block_hashes(request).await?;
    let inner = response.into_inner();
    println!("   Status: {:?}", inner.status);
    println!("   Found: {}/{}", inner.found_count, inner.total_queried);
    println!("   Existing hashes count: {}", inner.existing_hashes.len());

    // 4. Query mix of existing and non-existing hashes
    println!("\n4. Querying mix of existing and non-existing hashes...");
    let mixed_hashes = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],         // Exists
        vec![99, 99, 99, 99, 99, 99, 99, 99], // Doesn't exist
        vec![17, 18, 19, 20, 21, 22, 23, 24], // Exists
        vec![88, 88, 88, 88, 88, 88, 88, 88], // Doesn't exist
    ];

    let request = QueryBlockHashesRequest {
        namespace: namespace.to_string(),
        block_hashes: mixed_hashes,
    };

    let response = client.query_block_hashes(request).await?;
    let inner = response.into_inner();
    println!("   Status: {:?}", inner.status);
    println!("   Found: {}/{}", inner.found_count, inner.total_queried);
    println!("   Existing hashes: {:?}", inner.existing_hashes);

    // 5. Query with different namespace (should find nothing)
    println!("\n5. Querying with different namespace...");
    let request = QueryBlockHashesRequest {
        namespace: "different-model".to_string(),
        block_hashes: block_hashes,
    };

    let response = client.query_block_hashes(request).await?;
    let inner = response.into_inner();
    println!("   Status: {:?}", inner.status);
    println!("   Found: {}/{}", inner.found_count, inner.total_queried);

    println!("\n✓ All operations completed successfully!");
    Ok(())
}
