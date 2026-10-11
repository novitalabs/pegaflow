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

# Configure the HTTP listener
cargo run -p pegaflow-metaserver -- --addr 0.0.0.0:50056 --http-addr 0.0.0.0:9092

# Show all options
cargo run -p pegaflow-metaserver -- --help
```

### Server Options

- `--addr <ADDR>`: gRPC bind address (default: `127.0.0.1:50056`)
- `--http-addr <ADDR>`: HTTP health, metrics, and maintenance bind address (default: `0.0.0.0:9092`)
- `--log-level <LEVEL>`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--node-stale-secs <SECONDS>`: Hide nodes from query after this many seconds without heartbeat (default: `30`)
- `--ttl-minutes <MINUTES>`: Delete nodes and their owners after this many minutes without node activity (default: `120`); does not expire blocks by registration age
- `--sweep-interval-secs <SECONDS>`: Run the lifecycle sweep at this interval (default: `600`)
- `--min-reclaimable-owner-count <COUNT>`: Return a reclaim hint once this many live owners hold a block (default: `3`, minimum: `2`). `2` reclaims more aggressively than the default; values below `2` are rejected.
- `--enable-decode-backup`: Let idle nodes back up prefill nodes' retained LRU-tail blocks (default: off; env `PEGAFLOW_METASERVER_ENABLE_DECODE_BACKUP`). See [Decode backup](#decode-backup).
- `--backup-target-delay-secs <SECONDS>`: A live node with no inference instance for this long becomes a backup target (default: `180`)
- `--backup-inflight-ttl-secs <SECONDS>`: A dispatched backup block is not re-planned for this long unless a backup owner registers it (default: `180`)
- `--backup-max-bytes-per-sec <BYTES>`: Planned backup bytes per second per source and per target; `0` disables pacing (default: 2 GiB)

### Decode backup

With `--enable-decode-backup`, the MetaServer splits live nodes by role:
a node that reports an inference instance is a source immediately, and a node
without one becomes a target after `--backup-target-delay-secs`. Sources and
targets are paired by rendezvous hashing with bounded load, recomputed on
demand, so each source has one target and targets share sources evenly.

1. Once its cache is at least 90% full, a source sends its oldest retained
   cache-owned blocks with each heartbeat: enough bytes to bring the
   reclaimable class back to 10% of capacity. Each report replaces that
   source's previous snapshot. Eviction that drains the reclaimable class
   below this watermark sends an extra heartbeat, at most once per second.
2. Reported blocks that already have a live target owner come back in
   `backed_hashes`; the source moves them to its reclaimable class.
3. A target calls `PullBackupPlan` back to back while it gets plans, and
   once a second when idle. The MetaServer pops blocks from its paired
   sources oldest first, skips blocks that already have a target owner, whose
   source is gone, or that are in flight, and returns one namespace from one
   source within the byte budget. When pacing holds the pair back it returns
   an empty plan with `retry_after_ms`.
4. The target RDMA-reads the blocks with the regular P2P transfer path,
   inserts them as retained, and registers them. On a completed transfer the
   source demotes its copies to reclaimable, so pressure evicts them first
   and later prefix queries recall them from the target.

A target does not demote blocks it serves, and it only re-pulls blocks that
are still retained on a source, so evicted backups are not copied again.

### Storage Configuration

The MetaServer uses a DashMap-based in-memory store with the following characteristics:

- **Multi-owner**: A block hash can be registered by multiple nodes simultaneously
- **Node lifecycle**: Servers generate a `node_id`, announce it with `HeartbeatNode`, heartbeat periodically, and include the same `node_id` in insert/remove RPCs.
- **Stale filtering**: Nodes stop appearing in query results after 30 seconds without activity by default. Their records remain until node TTL cleanup; the same session can recover before cleanup without reinserting its blocks.
- **Lifecycle sweep**: A background task runs every `--sweep-interval-secs`. It scans the block directory only after deleting an inactive node past `--ttl-minutes`, or when a session takeover needs reconciliation. Otherwise it only inspects nodes. Block registration age never causes automatic deletion.
- **Conditional removal**: `RemoveBlockHashes` only removes the requesting node's ownership; other nodes' entries are untouched.
- **Memory**: Scales with retained block and owner records; there is no hard capacity cap. Use the manual cleanup endpoint below for explicit maintenance.

## HTTP APIs

Health, metrics, and manual cleanup use the `--http-addr` listener. These
endpoints are intended for the internal MetaServer network. Choose distinct
available ports when running multiple MetaServers on one host.

| Default address | Method and path | Purpose |
| --- | --- | --- |
| `0.0.0.0:9092` | `GET /health` | Returns `ok` |
| `0.0.0.0:9092` | `GET /metrics` | Prometheus metrics |
| `0.0.0.0:9092` | `POST /admin/cleanup-expired-blocks` | Remove owner registrations older than one hour |

### Manual block cleanup

Run this on the MetaServer host, or inside its container when using container
networking. No request body or authentication is required:

```bash
curl --fail-with-body --silent --show-error --request POST \
  http://127.0.0.1:9092/admin/cleanup-expired-blocks
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
  bool has_instance = 3;                            // Decides the decode backup role
  repeated BackupCandidates backup_candidates = 4;  // Retained LRU-tail blocks, oldest first
}

message HeartbeatNodeResponse {
  uint64 stale_after_secs = 1;
  bool backup_enabled = 2;
  bool backup_target = 3;                           // Pull plans; do not demote served blocks
  repeated BackupCandidates backed_hashes = 4;      // Candidates that already have a backup
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
  repeated bytes reclaimable_hashes = 3; // New threshold-or-later owner hashes
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

### 6. PullBackupPlan

Decode backup targets ask for the next batch of blocks to copy. Returns an
empty response when the node is not a target or nothing is queued, and an
empty response with `retry_after_ms` when byte pacing holds it back.

```protobuf
message PullBackupPlanRequest {
  string node = 1;
  string node_id = 2;
  uint64 max_bytes = 3;
}

message PullBackupPlanResponse {
  string source_node = 1;
  string namespace = 2;
  repeated bytes block_hashes = 3;
  uint64 retry_after_ms = 4;
}
```

### 7. Health

Health check endpoint.

**Request:** `HealthRequest {}`
**Response:** `HealthResponse { status }`

### 8. Shutdown

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
- `PEGAFLOW_METASERVER_MIN_RECLAIMABLE_OWNER_COUNT`: Override the minimum live-owner count for reclaim hints (default `3`, minimum `2`); `2` reclaims more aggressively than the default, and the command-line option takes precedence.

## License

Part of the PegaFlow project.
