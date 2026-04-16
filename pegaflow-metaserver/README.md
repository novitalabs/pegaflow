# PegaFlow MetaServer

A gRPC server for managing block hash keys across multi-node PegaFlow instances. The MetaServer provides a centralized registry for tracking which block hashes exist across distributed deployments.

## Overview

The MetaServer acts as a coordination service for distributed PegaFlow deployments. It maintains a global registry of block hash keys, allowing PegaFlow instances to:

- **Insert block hashes**: Register blocks that have been saved locally
- **Remove block hashes**: Deregister blocks on cache eviction (owner-conditional)
- **Query block hashes**: Check which blocks exist and on which nodes (multi-owner aware)
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

# Custom TTL (60 minutes)
cargo run -p pegaflow-metaserver -- --ttl-minutes 60

# All options combined
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056 --ttl-minutes 180 --log-level info

# Show all options
cargo run -p pegaflow-metaserver -- --help
```

### Server Options

- `--addr <ADDR>`: Bind address (default: `127.0.0.1:50056`)
- `--log-level <LEVEL>`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--ttl-minutes <MINUTES>`: Cache entry TTL in minutes (default: `120`)

### Storage Configuration

The MetaServer uses a DashMap-based in-memory store with the following characteristics:

- **Multi-owner**: A block hash can be registered by multiple nodes simultaneously
- **TTL (Time-To-Live)**: 120 minutes default (configurable via `--ttl-minutes`). A background task sweeps expired entries every 10 minutes to prevent leaks from crashed nodes.
- **Conditional removal**: `RemoveBlockHashes` only removes the requesting node's ownership; other nodes' entries are untouched.
- **Memory**: Scales with unique blocks across all nodes. No hard capacity cap — memory is naturally bounded by the total number of blocks in the cluster.

## gRPC APIs

The MetaServer provides the following gRPC endpoints:

### 1. InsertBlockHashes

Register a list of block hashes. Re-inserting refreshes the TTL timestamp.

**Request:**
```protobuf
message InsertBlockHashesRequest {
  string namespace = 1;         // Model namespace (part of BlockKey)
  repeated bytes block_hashes = 2;  // List of block hashes to insert (part of BlockKey)
  string node = 3;              // The pegaflow-server gRPC address that owns these blocks
}
```

**Response:**
```protobuf
message InsertBlockHashesResponse {
  ResponseStatus status = 1;    // Success/error status
  uint64 inserted_count = 2;    // Number of hashes inserted
}
```

### 2. RemoveBlockHashes

Remove block hashes owned by a specific node (conditional delete).

**Request:**
```protobuf
message RemoveBlockHashesRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;
  string node = 3;              // Only this node's ownership is removed
}
```

**Response:**
```protobuf
message RemoveBlockHashesResponse {
  ResponseStatus status = 1;
  uint64 removed_count = 2;
}
```

### 3. QueryPrefixBlocks

Query the longest contiguous prefix of block hashes that exist, with per-node prefix lengths.

**Request:**
```protobuf
message QueryPrefixBlocksRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;  // Ordered list of block hashes
}
```

**Response:**
```protobuf
message NodePrefixResult {
  string node = 1;
  uint32 prefix_len = 2;       // Consecutive hashes from h0 this node owns
}

message QueryPrefixBlocksResponse {
  repeated NodePrefixResult nodes = 1;
}
```

### 4. Health

Health check endpoint.

**Request:** `HealthRequest {}`
**Response:** `HealthResponse { status }`

### 5. Shutdown

Graceful shutdown trigger.

**Request:** `ShutdownRequest {}`
**Response:** `ShutdownResponse { status }`

## Storage Implementation

- **Data structure**: `DashMap<BlockKey, HashMap<Arc<str>, Instant>>` — each block key maps to a set of owning nodes with their registration timestamps
- **BlockKey**: `{ namespace: String, hash: Vec<u8> }` — matches pegaflow-core's BlockKey
- **Multi-owner**: Multiple nodes can register the same block hash (e.g., after replication or shared prefill)
- **TTL sweep**: Background task runs every 10 minutes, removing per-node registrations older than the configured TTL
- **Concurrency**: DashMap uses shard-level locking for high-throughput concurrent access
- **Persistence**: In-memory only (restart clears state)

## Integration with PegaFlow Core

1. **On block save**: Call `InsertBlockHashes` to register new blocks
2. **On cache eviction**: Call `RemoveBlockHashes` to deregister evicted blocks
3. **On block query**: Call `QueryPrefixBlocks` to discover which nodes hold a prefix
4. **On block load**: Query metaserver, then fetch from the best remote node via RDMA

## Environment Variables

- `RUST_LOG`: Control logging (e.g., `RUST_LOG=debug`)

## License

Part of the PegaFlow project.
