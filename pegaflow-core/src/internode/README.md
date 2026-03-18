# internode/ — Cross-Node Communication

This module handles all inter-node communication for PegaFlow's multi-node KV cache sharing.

## Module Overview

```
internode/
├── mod.rs                 Module root + re-exports
├── types.rs               Shared types: configs, errors, PegaflowInstance
├── registry.rs            InstanceRegistry — thread-safe store of discovered nodes
├── service_discovery.rs   K8s pod watcher (label: novita.ai/pegaflow=app)
├── client.rs              gRPC client → remote pegaflow-server (Engine service)
├── registrar.rs           Fire-and-forget registrar → pegaflow-metaserver (Meta service)
├── metaserver_query.rs    gRPC query client → pegaflow-metaserver (block location discovery)
└── remote_fetch_worker.rs Async RDMA fetch worker (MetaServer → gRPC → RDMA READ → SealedBlock)
```

## Data Flow

```
                        ┌──────────────┐
  service_discovery ──► │   registry   │  K8s watches pods, populates registry
                        └──────┬───────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                        ▼
┌─────────────┐    ┌───────────────────┐       ┌──────────────┐
│  client.rs  │    │ remote_fetch_     │       │ registrar.rs │  WRITE path
│             │    │ worker.rs         │       │              │  "I just sealed
│  Query /    │    │                   │       │  Insert      │   these blocks"
│  Transfer / │    │  Cross-node RDMA  │       │  BlockHashes │
│  Health     │    │  fetch pipeline   │       └──────┬───────┘
└──────┬──────┘    └────────┬──────────┘              │
       │                    │                         ▼
       │           ┌────────┼────────┐     Central pegaflow-metaserver
       │           ▼        ▼        ▼     (shared Meta gRPC)
       │    MetaServer   gRPC     RDMA
       │    query        lock     READ
       ▼
Remote pegaflow-server
(per-node Engine gRPC)
```

## Module Responsibilities

### client.rs — Remote Engine Client

|                  | Details |
|------------------|---------|
| Talks to         | Remote **pegaflow-server** |
| RPC service      | `Engine` (Query, QueryBlocksForTransfer, ReleaseTransferLock, Health) |
| Direction        | Read path — query remote cache, transfer block RDMA metadata |
| Call pattern     | Request-response (caller awaits) |
| Connection mgmt  | `PegaflowClientPool` — DashMap cache keyed by endpoint URL |
| Failure mode     | Returns error to caller |
| Used by          | Cross-node queries, remote fetch worker |

### registrar.rs — Block Hash Registration (Write Path)

|                  | Details |
|------------------|---------|
| Talks to         | Central **pegaflow-metaserver** |
| RPC service      | `MetaServer` (InsertBlockHashes) |
| Direction        | Write path — register sealed blocks for cross-node discovery |
| Call pattern     | Fire-and-forget (`try_send` to mpsc channel) |
| Connection mgmt  | Single lazy gRPC connection with auto-reconnect |
| Failure mode     | Log + metric, drop on queue full |
| Used by          | Write pipeline (after block seal) |

### metaserver_query.rs — Block Location Discovery (Read Path)

|                  | Details |
|------------------|---------|
| Talks to         | Central **pegaflow-metaserver** |
| RPC service      | `MetaServer` (QueryBlockHashes) |
| Direction        | Read path — discover which node holds specific blocks |
| Call pattern     | Request-response (caller awaits) |
| Connection mgmt  | Single gRPC connection created at startup |
| Failure mode     | Returns error, caller skips remote fetch |
| Used by          | `remote_fetch_worker.rs` |

### remote_fetch_worker.rs — Async RDMA Fetch Worker

Orchestrates the full cross-node block fetch pipeline as a tokio task:

```
1. Query MetaServer → which nodes hold the missing blocks
2. For each remote node:
   a. gRPC: QueryBlocksForTransfer → lock blocks + get RDMA metadata
   b. Allocate local pinned memory
   c. RDMA READ: pull block data from remote pinned memory
   d. Build SealedBlock from received data
   e. gRPC: ReleaseTransferLock
3. Send result via oneshot channel → ReadCache batch_insert
```

Uses closure-based RDMA abstraction (`RdmaBatchReadFn`, `RdmaRegisterMemoryFn`) to decouple
`pegaflow-core` from `pegaflow-transfer`.

## How registrar.rs Integrates

The registrar plugs into the **write path** via `InsertDeps`:

```
insert_worker_loop (sync thread)
  └─► process_insert_batch
        └─► send_backing_batches
              ├─► SsdBackingStore::ingest_batch()     (SSD write, fire-and-forget)
              └─► MetaServerRegistrar::try_register()  (metaserver registration, fire-and-forget)
```

Both use the same pattern: `tokio::sync::mpsc::try_send()` from the sync insert thread,
with an async tokio task draining the channel on the other end.

## How remote_fetch_worker.rs Integrates

The remote fetch plugs into the **read path** via `RemoteFetchScheduler`:

```
StorageEngine::check_prefix_and_remote_fetch()
  └─► RemoteFetchScheduler::check_and_fetch()
        ├─► Poll existing fetch (oneshot try_recv)
        ├─► Prefix scan ReadCache
        └─► Dispatch: RemoteFetchFn(missing_keys, oneshot_tx)
              └─► tokio::spawn(execute_remote_fetch(...))
                    ├─► MetaServerQueryClient::query_block_hashes()
                    ├─► PegaflowClient::query_blocks_for_transfer()
                    ├─► RDMA READ via RdmaBatchReadFn
                    └─► oneshot_tx.send(fetched_blocks)
```

The state machine mirrors `prefetch.rs` (SSD prefetch):
- Per-request tracking via `HashMap<req_id, RemoteFetchEntry>`
- Non-blocking polling with `oneshot::try_recv()`
- Backpressure via `max_remote_fetch_blocks` config

## Configuration

```bash
# Block registration + cross-node fetch + RDMA transfer
pegaflow-server \
  --metaserver-addr http://127.0.0.1:50056 \
  --advertise-addr 10.0.0.1:50055 \
  --rdma-nic mlx5_0 \
  --rdma-port 50057 \
  --transfer-lock-timeout-secs 30 \
  --max-remote-fetch-blocks 200
```

When `--metaserver-addr` is not set, both registration and remote fetch are disabled (`None`)
with zero overhead. When set, `--advertise-addr` and `--rdma-nic` are required.
