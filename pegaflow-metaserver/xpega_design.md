# Cross-Node KV Cache Sharing — Design Document

> **Status:** Proposal
> **Authors:** PegaFlow Team
> **Last Updated:** 2026-03-17

## 1. Motivation

PegaFlow currently provides a single-node KV cache storage hierarchy:

```
GPU Memory → Pinned CPU Memory (→ optional SSD Cache)
```

When a request arrives, the engine checks the local pinned memory cache for prefix blocks. A block that is missing locally may already exist in another PegaFlow node's memory cache — today, that cache is unreachable and the GPU must recompute from scratch.

**Cross-node KV cache sharing** adds a remote tier to the lookup path:

```
GPU Memory → Pinned CPU Memory → Remote PegaFlow Node (RDMA)
```

By pulling cached KV blocks from a peer node over RDMA instead of recomputing them on GPU, we can significantly reduce prefill latency (TTFT) and GPU utilization for workloads with shared prefixes across a multi-node deployment.

> **Design note:** The cross-node fetch borrows the async prefetch pattern from PegaFlow's existing SSD tier (state machine, backpressure, `oneshot` completion channel), but operates independently — it does not depend on or interact with SSD storage.

### Target Use Cases

| Scenario | Benefit |
|----------|---------|
| **Multi-node prefix caching** | Requests with common system prompts hit the remote cache instead of recomputing |
| **P/D disaggregation** | Decode nodes fetch KV cache from prefill nodes via RDMA |
| **Elastic scaling** | Newly scaled-up nodes warm their cache from existing peers |
| **Node recovery** | Replacement nodes pull cached blocks from survivors |

## 2. Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PegaFlow    │     │  PegaFlow    │     │  PegaFlow    │
│  Node A      │     │  Node B      │     │  Node C      │
│              │     │              │     │              │
│  ┌────────┐  │     │  ┌────────┐  │     │  ┌────────┐  │
│  │GPU Mem │  │     │  │GPU Mem │  │     │  │GPU Mem │  │
│  ├────────┤  │     │  ├────────┤  │     │  ├────────┤  │
│  │Pinned  │  │     │  │Pinned  │  │     │  │Pinned  │  │
│  │CPU Mem │◄─┼─ RDMA READ ─────┤  │     │  │CPU Mem │  │
│  └────────┘  │     │  └────────┘  │     │  └────────┘  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       └────────────────────┼─────────────────────┘
                            │  gRPC
                            ▼
                  ┌───────────────────┐
                  │  MetaServer       │
                  │  (Block Registry) │
                  └───────────────────┘
```

Three components work together:

| Component | Role |
|-----------|------|
| **MetaServer** | Centralized block hash registry. Maps `(namespace, hash) → node_address`. |
| **PegaFlow Server** (each node) | Stores KV blocks in pinned CPU memory. Serves block metadata and RDMA memory regions to requesting peers. |
| **PegaFlow Transfer** (RDMA engine) | Handles one-sided RDMA READ/WRITE between nodes using RC queue pairs with automatic session management. |

## 3. Detailed Data Flow

### 3.1 Block Registration (Write Path)

When a PegaFlow node saves KV cache blocks to its local storage, it also registers the block hashes with the MetaServer so that other nodes can discover them.

```
vLLM/SGLang Worker                PegaFlow Server (Node A)              MetaServer
       │                                   │                               │
       │ ── SaveKVCacheBlocks ──►          │                               │
       │                                   │── seal blocks ──►             │
       │                                   │   pinned memory               │
       │                                   │                               │
       │                                   │── InsertBlockHashes ─────────►│
       │                                   │   (namespace, hashes, node)   │
       │                                   │                               │
       │                                   │◄── inserted_count ───────────│
       │◄── OK ───────────────────────────│                               │
```

### 3.2 Cross-Node Lookup and Fetch (Read Path)

When a node receives a request and some prefix blocks miss locally, it queries the MetaServer to find which remote node holds the missing blocks, then pulls them over RDMA.

```
Step 1: Local Memory Scan
─────────────────────────
PegaEngine.count_prefix_hit_blocks(instance_id, block_hashes)
  │
  ├── Check pinned CPU memory cache (ReadCache)
  │     hit: 80 blocks ✓
  │
  └── Missing: 30 blocks ✗
        → These are candidates for cross-node fetch


Step 2: MetaServer Query
────────────────────────
PegaflowClient → MetaServer.QueryBlockHashes(namespace, missing_hashes)
  │
  └── Response: {
        node_blocks: [
          { node: "10.0.0.2:50055", block_hashes: [h1, h2, ..., h25] },
          { node: "10.0.0.3:50055", block_hashes: [h26, h27, ..., h30] }
        ]
      }


Step 3: Remote Block Fetch (RDMA)
─────────────────────────────────
For each remote node with matching blocks:
  │
  ├── 3a. gRPC: Query remote node for block metadata
  │         → get RDMA memory region info (rkey, remote_ptr, size)
  │         → remote node locks the blocks (prevents eviction during transfer)
  │
  ├── 3b. Allocate local pinned memory for incoming blocks
  │         → reuse the existing PinnedPool allocator
  │
  ├── 3c. RDMA READ: pull block data from remote pinned memory
  │         → one-sided, does not involve remote CPU
  │         → batch_transfer_sync_read() for multiple blocks
  │
  ├── 3d. Insert received blocks into local ReadCache
  │         → blocks are now available for GPU load
  │
  └── 3e. Notify remote node to release block locks
```

### 3.3 Lookup Order

```
Request arrives with prefix block_hashes[0..N]
  │
  ▼
┌─────────────────────────────┐
│ 1. ReadCache (pinned memory)│  ◄── Fastest: ~μs
│    Prefix scan, stop at     │
│    first miss               │
└─────────────┬───────────────┘
              │ missing blocks
              ▼
┌─────────────────────────────┐
│ 2. Remote Node (RDMA)       │  ◄── Fast: ~low ms
│    MetaServer lookup        │
│    + RDMA READ from peer    │
└─────────────┬───────────────┘
              │ truly missing
              ▼
┌─────────────────────────────┐
│ 3. Recompute on GPU         │  ◄── Slowest: ~100s of ms
│    Full prefill pass        │
└─────────────────────────────┘
```

## 4. Component Design

### 4.1 MetaServer — Block Hash Registry

The MetaServer is an in-memory, LRU-evicting registry that tracks `(namespace, block_hash) → node_address` mappings across all PegaFlow nodes. It is the **discovery layer** — it tells you *where* a block lives, but does not store block data itself.

**Current implementation** (already built):
- Moka async cache with LRU eviction and configurable TTL (default: 120 min)
- Size-aware capacity management (default: 512 MB)
- gRPC API: `InsertBlockHashes`, `QueryBlockHashes`, `Health`, `Shutdown`
- Namespace isolation for multi-model deployments

**Design decisions:**
- **Single-writer semantics**: A block hash maps to exactly one node. If the same block is saved on multiple nodes, the latest insert wins. This is acceptable because any node that has the block can serve it.
- **Eventual consistency**: Block registration is asynchronous. A brief window exists where a block has been saved locally but not yet registered. During this window, the block won't be found via cross-node lookup, and the system falls back to recomputation — a safe degradation.
- **TTL-based expiration**: Entries auto-expire after the configured TTL. Nodes must periodically re-register their blocks, or the MetaServer will forget them. This prevents stale entries from accumulating after node failures.

### 4.2 Remote Node Server — Block Metadata and Lock API

Each PegaFlow server needs two new gRPC endpoints for cross-node fetch:

#### `QueryBlocksForTransfer` — Get RDMA metadata for specific blocks

```protobuf
message TransferBlockInfo {
  bytes block_hash = 1;
  uint64 remote_ptr = 2;    // RDMA-registered pinned memory address
  uint64 size = 3;           // Block data size in bytes
  uint32 rkey = 4;           // RDMA remote key for this memory region
}

message QueryBlocksForTransferRequest {
  string namespace = 1;
  repeated bytes block_hashes = 2;
  string requester_id = 3;   // For lock tracking
}

message QueryBlocksForTransferResponse {
  ResponseStatus status = 1;
  repeated TransferBlockInfo blocks = 2;
  string transfer_session_id = 3;  // Used for unlock
}
```

#### `ReleaseTransferLock` — Unlock blocks after RDMA transfer

```protobuf
message ReleaseTransferLockRequest {
  string transfer_session_id = 1;
}

message ReleaseTransferLockResponse {
  ResponseStatus status = 1;
}
```

**Block locking semantics:**

When a remote node serves block metadata for RDMA transfer, it must **lock** those blocks to prevent eviction while the RDMA READ is in progress. Without locking, the LRU cache could evict a block mid-transfer, causing the requesting node to read corrupted or freed memory.

| Aspect | Design |
|--------|--------|
| **Lock scope** | Per-transfer-session. All blocks in a single `QueryBlocksForTransfer` call share one session ID. |
| **Lock duration** | Bounded by a configurable timeout (default: 30s). If the requester crashes, the lock auto-expires. |
| **Lock effect** | Locked blocks are exempt from LRU eviction. They remain pinned in memory. |
| **Unlock trigger** | Either explicit `ReleaseTransferLock` call, or timeout expiration. |
| **Concurrency** | Multiple concurrent transfer sessions are supported. A block can be locked by multiple sessions simultaneously (shared read lock). |

### 4.3 RDMA Transfer — Data Movement

The `pegaflow-transfer` crate (already built) provides the RDMA transport layer. Cross-node fetch uses it as follows:

```rust
// On the requesting node:
let engine = MooncakeTransferEngine::new();
engine.initialize("mlx5_0", port)?;
engine.register_memory(local_pinned_buffer, buffer_len)?;

// After getting TransferBlockInfo from remote node:
engine.batch_transfer_sync_read(
    &remote_session_id,
    &local_ptrs,    // allocated pinned memory on this node
    &remote_ptrs,   // from TransferBlockInfo.remote_ptr
    &block_sizes,   // from TransferBlockInfo.size
)?;
```

**Reliability requirements:**

| Requirement | Implementation |
|-------------|----------------|
| **Reliable connection** | RDMA RC (Reliable Connection) queue pairs — the transport layer already provides this with retransmission, ordering, and completion notification. |
| **Read-side timeout** | The requesting node sets a deadline for the RDMA READ to complete. On timeout, it aborts the transfer and falls back to GPU recomputation. |
| **Transfer-side timeout** | The remote node auto-releases block locks after timeout, preventing indefinite memory pinning. |
| **Partial failure** | If RDMA READ succeeds for some blocks but fails for others, the requesting node uses whatever blocks it received and recomputes the rest. |

### 4.4 Integration with StorageEngine

The cross-node fetch follows the same async prefetch pattern as the existing SSD tier (state machine + `oneshot` completion channel + backpressure), but is implemented as an independent fetch source in `StorageEngine`.

```
StorageEngine.query_with_remote_fetch()
  │
  ├── Poll existing remote fetch (if any)
  │     StillLoading → return Loading status
  │     Completed → insert blocks into ReadCache, continue
  │
  └── full_prefix_scan()
        │
        ├── ReadCache.get_prefix_blocks() → hit N blocks
        │
        ├── remaining blocks:
        │     └── submit_remote_fetch() → loading K blocks
        │           ├── MetaServer.QueryBlockHashes()
        │           ├── Remote.QueryBlocksForTransfer()
        │           ├── RDMA READ (async, via oneshot channel)
        │           └── on completion → batch_insert into ReadCache
        │
        └── Return RemoteFetchStatus { hit: N, loading: K, missing: rest }
```

The remote fetch status tracks the async lifecycle:

```rust
pub enum RemoteFetchStatus {
    /// All blocks resolved — either hit locally or confirmed missing everywhere.
    Done { hit: usize, missing: usize },
    /// Some blocks are being fetched from a remote node via RDMA.
    Loading { hit: usize, loading: usize },
}
```

## 5. Configuration

### MetaServer

| Flag | Default | Description |
|------|---------|-------------|
| `--addr` | `127.0.0.1:50056` | MetaServer bind address |
| `--max-capacity-mb` | `512` | Maximum cache capacity |
| `--ttl-minutes` | `120` | Entry expiration time |

### PegaFlow Server (per-node)

| Flag | Default | Description |
|------|---------|-------------|
| `--metaserver-addr` | (none) | MetaServer endpoint. Enables cross-node block registration when set. |
| `--remote-fetch` | `false` | Enable cross-node block fetching on cache miss |
| `--transfer-lock-timeout-secs` | `30` | Auto-release lock timeout for remote block transfers |
| `--max-remote-fetch-blocks` | `200` | Backpressure limit on concurrent remote fetch blocks |

## 6. Error Handling and Failure Modes

| Failure | Behavior |
|---------|----------|
| **MetaServer unreachable** | Cross-node lookup is skipped. Falls back to GPU recomputation. No impact on local operations. |
| **Remote node unreachable** | gRPC connection fails. Falls back to recomputation for those blocks. Client pool evicts the stale connection. |
| **RDMA transfer failure** | Transfer times out or returns error. Local node recomputes the missed blocks. Remote node auto-releases locks. |
| **Remote node evicted blocks between query and fetch** | `QueryBlocksForTransfer` will return fewer blocks than requested. The requesting node handles partial results gracefully. |
| **MetaServer returns stale entry** | The block was evicted from the remote node after MetaServer registration. The remote query returns "not found" for those blocks. Cost: one wasted gRPC round-trip per stale lookup. Mitigated by TTL expiration. |
| **Lock timeout before transfer completes** | RDMA READ may read invalid memory. **Mitigation**: set lock timeout >> expected RDMA transfer time. Monitor and alert on lock timeout events. |

## 7. Observability

New metrics for cross-node operations:

| Metric | Type | Description |
|--------|------|-------------|
| `remote_fetch_requests_total` | Counter | Total cross-node fetch attempts |
| `remote_fetch_blocks_total` | Counter | Total blocks fetched from remote nodes |
| `remote_fetch_blocks_hit` | Counter | Blocks successfully fetched |
| `remote_fetch_blocks_missed` | Counter | Blocks not found on remote (stale MetaServer entry) |
| `remote_fetch_latency_seconds` | Histogram | End-to-end latency of cross-node fetch (query + RDMA) |
| `remote_fetch_rdma_bytes` | Counter | Bytes transferred via RDMA READ |
| `metaserver_insert_latency_seconds` | Histogram | Block registration latency |
| `metaserver_query_latency_seconds` | Histogram | MetaServer lookup latency |
| `transfer_lock_active` | Gauge | Currently held transfer locks |
| `transfer_lock_timeouts_total` | Counter | Locks released by timeout (potential issue indicator) |

## 8. Implementation Plan

### Phase 1: Foundation (MetaServer + Registration)

- [x] MetaServer with in-memory LRU store, gRPC API, TTL expiration
- [x] `InsertBlockHashes` / `QueryBlockHashes` APIs
- [x] Internode client with connection pooling (`PegaflowClientPool`)
- [x] Kubernetes service discovery for PegaFlow instances
- [x] Automatic block hash registration on the write path (call `InsertBlockHashes` after sealing blocks)

### Phase 2: Cross-Node Fetch (RDMA)

- [ ] `QueryBlocksForTransfer` gRPC endpoint on PegaFlow server
- [ ] Block locking mechanism with timeout-based auto-release
- [ ] `ReleaseTransferLock` gRPC endpoint
- [ ] Remote fetch integration in `StorageEngine` (async state machine, inspired by SSD prefetch pattern)
- [ ] Local pinned memory allocation for incoming remote blocks
- [ ] RDMA READ via `pegaflow-transfer` batch API
- [ ] Insert fetched blocks into `ReadCache`

### Phase 3: Production Hardening

- [ ] Observability: Prometheus metrics for all cross-node operations
- [ ] Backpressure: limit concurrent remote fetches (`--max-remote-fetch-blocks`)
- [ ] Graceful degradation: automatic fallback when MetaServer / remote nodes are unhealthy
- [ ] Lock timeout tuning and monitoring
- [ ] Performance benchmarking: RDMA fetch latency vs. GPU recomputation
- [ ] Integration tests with multi-node setup

### Phase 4: Optimizations (Future)

- [ ] Speculative prefetch: proactively pull blocks that are likely to be needed soon
- [ ] Multi-node fan-out: fetch blocks from multiple remote nodes in parallel
- [ ] MetaServer replication for high availability
- [ ] Persistent MetaServer backend (Redis / RocksDB) for faster restart
- [ ] Compression for gRPC metadata exchange (block hashes are small, but volume can be high)
- [ ] Topology-aware routing: prefer RDMA peers on the same switch / rack
