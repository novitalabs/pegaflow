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
cargo run -p pegaflow-metaserver -- --node-stale-secs 30 --ttl-minutes 120 --sweep-interval-secs 600

# Configure public HTTP and localhost-only maintenance ports
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056 --http-addr 0.0.0.0:9092 --admin-http-addr 127.0.0.1:9093

# Show all options
cargo run -p pegaflow-metaserver -- --help
```

### Server Options

- `--addr <ADDR>`: gRPC bind address (default: `127.0.0.1:50056`)
- `--http-addr <ADDR>`: HTTP health and metrics bind address (default: `0.0.0.0:9092`)
- `--admin-http-addr <ADDR>`: HTTP maintenance bind address; must be loopback (default: `127.0.0.1:9093`)
- `--log-level <LEVEL>`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--node-stale-secs <SECONDS>`: Hide nodes from query after this many seconds without heartbeat (default: `30`)
- `--ttl-minutes <MINUTES>`: Delete nodes and their owners after this many minutes without node activity (default: `120`); does not expire blocks by registration age
- `--sweep-interval-secs <SECONDS>`: Run the lifecycle sweep at this interval (default: `600`)

### Storage Configuration

The MetaServer uses a DashMap-based in-memory store with the following characteristics:

- **Multi-owner**: A block hash can be registered by multiple nodes simultaneously
- **Node lifecycle**: Servers generate a `node_id`, announce it with `HeartbeatNode`, heartbeat periodically, and include the same `node_id` in insert/remove RPCs.
- **Stale filtering**: Nodes stop appearing in query results after 30 seconds without activity by default. Their records remain until node TTL cleanup; the same session can recover before cleanup without reinserting its blocks.
- **Lifecycle sweep**: A background task runs every `--sweep-interval-secs`. It scans the block directory only after deleting an inactive node past `--ttl-minutes`, or when a session takeover needs reconciliation. Otherwise it only inspects nodes. Block registration age never causes automatic deletion.
- **Conditional removal**: `RemoveBlockHashes` only removes the requesting node's ownership; other nodes' entries are untouched.
- **Memory**: Scales with retained block and owner records; there is no hard capacity cap. Use the manual cleanup endpoint below for explicit maintenance.

## HTTP APIs

Health and metrics use `--http-addr`; manual cleanup uses the separate
`--admin-http-addr` listener. The admin address must be loopback (for example,
`127.0.0.1:9093` or `[::1]:9093`); a non-loopback address makes startup fail.
Choose distinct available ports when running multiple MetaServers on one host.

| Default address | Method and path | Purpose |
| --- | --- | --- |
| `0.0.0.0:9092` | `GET /health` | Returns `ok` |
| `0.0.0.0:9092` | `GET /metrics` | Prometheus metrics |
| `127.0.0.1:9093` | `POST /admin/cleanup-expired-blocks` | Remove owner registrations older than one hour |

### Manual block cleanup

Run this on the MetaServer host, or inside its container when using container
networking. No request body or authentication is required:

```bash
curl --fail-with-body --silent --show-error --request POST \
  http://127.0.0.1:9093/admin/cleanup-expired-blocks
```

The threshold is fixed at one hour and independent of `--ttl-minutes`. The
cleanup pass scans all namespaces and removes owners whose last
`InsertBlockHashes` registration was **strictly more than 3,600 seconds ago**
when the pass started. Reinserting an owner refreshes its registration time;
heartbeats and queries do not.

Cleanup applies even to active nodes and blocks whose KV data still exists.
It removes MetaServer metadata, leaves node records and physical KV data in
place, and removes a block key only when its last owner is deleted. Fresh or
refreshed owners remain. A deleted owner becomes discoverable again after
`InsertBlockHashes` registers it; a heartbeat alone does not restore it.

A successful call waits for the scan to complete and returns HTTP 200, for
example:

```json
{"removed_owners":2,"removed_keys":1}
```

`removed_owners` counts deleted ownership records; `removed_keys` counts block
keys left without any owner. Both are zero if no registration is old enough.
The public HTTP listener returns 404 for this admin route; GET on the admin
route returns 405. A cleanup worker failure returns HTTP 500.

Manual cleanup scans the entire block directory on a blocking worker. It
holds a block shard write lock while processing that shard, so concurrent
RPCs accessing the same shard can experience increased latency. Use it for
explicit maintenance and monitor RPC latency during the call.

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
  repeated bytes reclaimable_hashes = 3; // New third-or-later owner hashes
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

Build an ordered fetch plan for the longest contiguous prefix that is available
from live remote nodes. The requester is excluded from the owner candidates.

**Request:**
```protobuf
message QueryPrefixBlocksRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;  // Ordered list of block hashes
  string exclude_node = 3;          // Requester address
}
```

**Response:**
```protobuf
message FetchSegment {
  string node = 1;
  uint32 block_count = 2;      // Consecutive hashes fetched from this node
}

message QueryPrefixBlocksResponse {
  reserved 1;
  reserved "nodes";
  repeated FetchSegment segments = 2;
}
```

Segments are ordered and contiguous. Their block counts are cumulative offsets
into the request's `block_hashes`; planning stops at the first hash with no live
remote owner.

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
- **Lifecycle sweep**: An inactive-node deletion or pending takeover triggers a block scan that removes owners with a missing node or a mismatched session. Superseded sessions are immediately hidden from queries and reconciled by the next sweep, even while the replacement session stays active.
- **Concurrency**: DashMap uses shard-level locking for high-throughput concurrent access
- **Persistence**: In-memory only (restart clears state)

## Integration with PegaFlow Core

1. **On server startup**: Generate a `node_id` and announce it with `HeartbeatNode`
2. **During server lifetime**: Call `HeartbeatNode` periodically with the same `node_id`
3. **On block save**: Call `InsertBlockHashes` with `{ node, node_id }`
4. **On cache eviction**: Call `RemoveBlockHashes` with `{ node, node_id }`
5. **On block query**: Call `QueryPrefixBlocks` to build an ordered remote fetch plan
6. **On block load**: Fetch each plan segment in order via RDMA, stopping on the first failure

## Environment Variables

- `RUST_LOG`: Control logging (e.g., `RUST_LOG=debug`)

## License

Part of the PegaFlow project.
