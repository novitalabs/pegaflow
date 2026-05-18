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

# Custom node lifecycle timings
cargo run -p pegaflow-metaserver -- --node-stale-secs 30 --ttl-minutes 120

# All options combined
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056 --node-stale-secs 30 --ttl-minutes 120 --log-level info

# Show all options
cargo run -p pegaflow-metaserver -- --help
```

### Server Options

- `--addr <ADDR>`: Bind address (default: `127.0.0.1:50056`)
- `--log-level <LEVEL>`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--node-stale-secs <SECONDS>`: Hide nodes from query after this many seconds without heartbeat (default: `30`)
- `--ttl-minutes <MINUTES>`: Purge ownership and node records after this many minutes; the sweep runs at this interval (default: `120`)

### Storage Configuration

The MetaServer uses a DashMap-based in-memory store with the following characteristics:

- **Multi-owner**: A block hash can be registered by multiple nodes simultaneously
- **Node lifecycle**: Servers generate a `node_id`, announce it with `HeartbeatNode`, heartbeat periodically, and include the same `node_id` in insert/remove RPCs.
- **Stale filtering**: Nodes stop appearing in query results after 30 seconds without heartbeat by default.
- **TTL sweep**: A background task removes expired owners and nodes after `--ttl-minutes`.
- **Conditional removal**: `RemoveBlockHashes` only removes the requesting node's ownership; other nodes' entries are untouched.
- **Memory**: Scales with unique blocks across all nodes. No hard capacity cap — memory is naturally bounded by the total number of blocks in the cluster.

## gRPC APIs

The MetaServer provides the following gRPC endpoints:

### 1. HeartbeatNode

Register or refresh node liveness for the current session. A different
`node_id` may take over the same node URL only after the current session is
stale.

```protobuf
message HeartbeatNodeRequest {
  string node = 1;
  string node_id = 2;
}
```

### 2. UnregisterNode

Gracefully remove a node and its matching ownership records.

```protobuf
message UnregisterNodeRequest {
  string node = 1;
  string node_id = 2;
}
```

### 3. InsertBlockHashes

Register a list of block hashes. The request must include the current `node_id`.

**Request:**
```protobuf
message InsertBlockHashesRequest {
  string namespace = 1;         // Model namespace (part of BlockKey)
  repeated bytes block_hashes = 2;  // List of block hashes to insert (part of BlockKey)
  string node = 3;              // The pegaflow-server gRPC address that owns these blocks
  string node_id = 4;           // Server-generated session id announced by HeartbeatNode
}
```

**Response:**
```protobuf
message InsertBlockHashesResponse {
  ResponseStatus status = 1;    // Success/error status
  uint64 inserted_count = 2;    // Number of hashes inserted
}
```

### 4. RemoveBlockHashes

Remove block hashes owned by a specific node (conditional delete).

**Request:**
```protobuf
message RemoveBlockHashesRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;
  string node = 3;              // Only this node's ownership is removed
  string node_id = 4;           // Only matching ownership is removed
}
```

**Response:**
```protobuf
message RemoveBlockHashesResponse {
  ResponseStatus status = 1;
  uint64 removed_count = 2;
}
```

### 5. QueryPrefixBlocks

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

### 6. Health

Health check endpoint.

**Request:** `HealthRequest {}`
**Response:** `HealthResponse { status }`

### 7. Shutdown

Graceful shutdown trigger.

**Request:** `ShutdownRequest {}`
**Response:** `ShutdownResponse { status }`

## Storage Implementation

- **Data structure**: `blocks: DashMap<BlockKey, HashMap<Arc<str>, OwnerRecord>>` and `nodes: DashMap<Arc<str>, NodeRecord>`
- **BlockKey**: `{ namespace: String, hash: Vec<u8> }` — matches pegaflow-core's BlockKey
- **Multi-owner**: Multiple nodes can register the same block hash (e.g., after replication or shared prefill)
- **Lifecycle sweep**: Background task removes owners whose node record is missing or whose ownership TTL expired; superseded sessions are hidden from queries by `node_id` matching and purged by TTL.
- **Concurrency**: DashMap uses shard-level locking for high-throughput concurrent access
- **Persistence**: In-memory only (restart clears state)

## Integration with PegaFlow Core

1. **On server startup**: Generate a `node_id` and announce it with `HeartbeatNode`
2. **During server lifetime**: Call `HeartbeatNode` periodically with the same `node_id`
3. **On block save**: Call `InsertBlockHashes` with `{ node, node_id }`
4. **On cache eviction**: Call `RemoveBlockHashes` with `{ node, node_id }`
5. **On block query**: Call `QueryPrefixBlocks` to discover which live nodes hold a prefix
6. **On block load**: Query metaserver, then fetch from the best remote node via RDMA

## Environment Variables

- `RUST_LOG`: Control logging (e.g., `RUST_LOG=debug`)

## License

Part of the PegaFlow project.
