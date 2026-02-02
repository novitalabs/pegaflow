# PegaFlow MetaServer

A gRPC server for managing block hash keys across multi-node PegaFlow instances. The MetaServer provides a centralized registry for tracking which block hashes exist across distributed deployments.

## Overview

The MetaServer acts as a coordination service for distributed PegaFlow deployments. It maintains a global registry of block hash keys, allowing PegaFlow instances to:

- **Insert block hashes**: Register blocks that have been saved locally
- **Query block hashes**: Check which blocks exist in the distributed system
- **Namespace isolation**: Separate blocks by model/namespace

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PegaFlow       │     │  PegaFlow       │     │  PegaFlow       │
│  Instance 1     │     │  Instance 2     │     │  Instance 3     │
│  (Node A)       │     │  (Node B)       │     │  (Node C)       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                          gRPC   │
                                 ▼
                    ┌─────────────────────┐
                    │   MetaServer        │
                    │                     │
                    │  - Block Registry   │
                    │  - Hash Storage     │
                    │  - Query Service    │
                    └─────────────────────┘
```

## Building

```bash
# Build debug version
cargo build -p pegaflow-metaserver

# Build release version
cargo build -p pegaflow-metaserver --release

# Run tests
cargo test -p pegaflow-metaserver
```

## Running

### Start the server

```bash
# Default bind address (127.0.0.1:50056)
cargo run -p pegaflow-metaserver

# Custom bind address
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056

# With debug logging
cargo run -p pegaflow-metaserver -- --log-level debug

# Custom cache capacity (1GB) and TTL (60 minutes)
cargo run -p pegaflow-metaserver -- --max-capacity-mb 1024 --ttl-minutes 60

# All options combined
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056 --max-capacity-mb 2048 --ttl-minutes 180 --log-level info

# Show all options
cargo run -p pegaflow-metaserver -- --help
```

### Server Options

- `--addr <ADDR>`: Bind address (default: `127.0.0.1:50056`)
- `--log-level <LEVEL>`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--max-capacity-mb <MB>`: Maximum cache capacity in MB (default: `512`)
- `--ttl-minutes <MINUTES>`: Cache entry TTL in minutes (default: `120`)

### Cache Configuration

The MetaServer uses an in-memory cache with the following settings:

- **Default Max Capacity**: 512 MB (configurable via `--max-capacity-mb`)
- **Eviction Policy**: LRU (Least Recently Used)
- **TTL (Time-To-Live)**: 120 minutes (configurable via `--ttl-minutes`)
- **Size Calculation**: Automatic based on key allocated capacity (namespace + hash + overhead)
- **Async Operations**: All cache operations are non-blocking async

The cache automatically evicts least recently used entries when the memory limit is reached, and all entries expire after the configured TTL. This ensures the service operates within memory constraints even under high load while preventing stale entries from accumulating indefinitely.

You can customize the cache behavior using CLI flags:
```bash
# Example: 1GB capacity with 60-minute TTL
cargo run -p pegaflow-metaserver -- --max-capacity-mb 1024 --ttl-minutes 60
```

## gRPC APIs

The MetaServer provides the following gRPC endpoints:

### 1. InsertBlockHashes

Register a list of block hashes. Matches the `BlockKey` structure from pegaflow-core.

**Request:**
```protobuf
message InsertBlockHashesRequest {
  string namespace = 1;         // Model namespace (part of BlockKey)
  repeated bytes block_hashes = 2;  // List of block hashes to insert (part of BlockKey)
}
```

**Response:**
```protobuf
message InsertBlockHashesResponse {
  ResponseStatus status = 1;    // Success/error status
  uint64 inserted_count = 2;    // Number of hashes inserted
}
```

### 2. QueryBlockHashes

Query which block hashes exist in the system. Matches the `BlockKey` structure from pegaflow-core.

**Request:**
```protobuf
message QueryBlockHashesRequest {
  string namespace = 1;         // Model namespace (part of BlockKey)
  repeated bytes block_hashes = 2;  // List of hashes to query (part of BlockKey)
}
```

**Response:**
```protobuf
message QueryBlockHashesResponse {
  ResponseStatus status = 1;        // Success/error status
  repeated bytes existing_hashes = 2;  // Hashes that exist
  uint64 total_queried = 3;         // Total number queried
  uint64 found_count = 4;           // Number found
}
```

### 3. Health

Health check endpoint.

**Request:** `HealthRequest {}`
**Response:** `HealthResponse { status }`

### 4. Shutdown

Graceful shutdown trigger.

**Request:** `ShutdownRequest {}`
**Response:** `ShutdownResponse { status }`

## Example Usage

See [examples/basic_client.rs](examples/basic_client.rs) for a complete example.

```bash
# Terminal 1: Start the server
cargo run -p pegaflow-metaserver

# Terminal 2: Run the example client
cargo run -p pegaflow-metaserver --example basic_client
```

### Quick Example

```rust
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use pegaflow_proto::proto::engine::{InsertBlockHashesRequest, QueryBlockHashesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to metaserver
    let mut client = MetaServerClient::connect("http://127.0.0.1:50056").await?;

    // Insert block hashes (only needs namespace + hashes, matching BlockKey)
    let request = InsertBlockHashesRequest {
        namespace: "llama3-70b".to_string(),
        block_hashes: vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![9, 10, 11, 12, 13, 14, 15, 16],
        ],
    };

    let response = client.insert_block_hashes(request).await?;
    println!("Inserted: {}", response.into_inner().inserted_count);

    // Query block hashes (only needs namespace + hashes, matching BlockKey)
    let request = QueryBlockHashesRequest {
        namespace: "llama3-70b".to_string(),
        block_hashes: vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![99, 99, 99, 99, 99, 99, 99, 99],  // doesn't exist
        ],
    };

    let response = client.query_block_hashes(request).await?;
    let inner = response.into_inner();
    println!("Found: {}/{}", inner.found_count, inner.total_queried);

    Ok(())
}
```

## Storage Implementation

The MetaServer uses a high-performance async cache based on **Moka**:

- **Cache**: `moka::future::Cache<BlockKey, ()>` - async concurrent cache with LRU eviction and TTL
- **BlockKey**: `{ namespace: String, hash: Vec<u8> }` - matches pegaflow-core's BlockKey
- **Eviction Policy**: LRU (Least Recently Used)
- **TTL**: 120 minutes - entries automatically expire after 2 hours
- **Capacity Management**: Size-aware eviction with configurable max capacity (default: 512 MB)
- **Concurrency**: Fully async with high-performance concurrent operations optimized for high-throughput services
- **Memory Safety**: Automatic eviction when cache size exceeds max capacity or TTL expires
- **Persistence**: In-memory only (restart clears state)

### Future Enhancements

Potential improvements for production deployments:

- [ ] Persistent storage backend (Redis, RocksDB, etc.)
- [ ] Replication and high availability
- [x] TTL/expiration for stale entries (120 minutes)
- [ ] Metrics and monitoring (Prometheus)
- [ ] Authentication and authorization
- [ ] Batch operations for improved performance
- [ ] Compression for network efficiency

## Integration with PegaFlow Core

To integrate the MetaServer with PegaFlow instances:

1. **On block save**: Call `InsertBlockHashes` to register new blocks
2. **On block query**: Call `QueryBlockHashes` to check remote availability
3. **On block load**: Query metaserver, then fetch from remote instance if found

The metaserver enables cross-node block discovery without peer-to-peer coordination.

## Environment Variables

- `RUST_LOG`: Control logging (e.g., `RUST_LOG=debug`)

## License

Part of the PegaFlow project.
