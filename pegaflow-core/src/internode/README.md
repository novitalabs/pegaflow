# internode/ — Cross-Node Communication

This module handles all inter-node communication for PegaFlow's multi-node KV cache sharing.

## Module Overview

```
internode/
├── mod.rs               Module root + re-exports
├── types.rs             Shared types: configs, errors, PegaflowInstance
├── registry.rs          InstanceRegistry — thread-safe store of discovered nodes
├── service_discovery.rs K8s pod watcher (label: novita.ai/pegaflow=app)
├── client.rs            gRPC client → remote pegaflow-server (Engine service)
└── registrar.rs         Fire-and-forget registrar → pegaflow-metaserver (Meta service)
```

## Data Flow

```
                        ┌──────────────┐
  service_discovery ──► │   registry   │  K8s watches pods, populates registry
                        └──────┬───────┘
                               │
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
    ┌─────────────┐                        ┌──────────────┐
    │  client.rs  │  READ path             │ registrar.rs │  WRITE path
    │             │  "do you have          │              │  "I just sealed
    │  Query /    │   these blocks?"       │  Insert      │   these blocks"
    │  Health     │                        │  BlockHashes │
    └──────┬──────┘                        └──────┬───────┘
           │                                      │
           ▼                                      ▼
    Remote pegaflow-server              Central pegaflow-metaserver
    (per-node Engine gRPC)              (shared Meta gRPC)
```

## client.rs vs registrar.rs

|                  | client.rs                          | registrar.rs                        |
|------------------|------------------------------------|-------------------------------------|
| Talks to         | Remote **pegaflow-server**         | Central **pegaflow-metaserver**     |
| RPC service      | `Engine` (Query, Health)           | `MetaServer` (InsertBlockHashes)    |
| Direction        | Read path — query remote cache     | Write path — register sealed blocks |
| Call pattern     | Request-response (caller awaits)   | Fire-and-forget (`try_send`)        |
| Connection mgmt  | Pool of connections (multi-node)   | Single lazy connection              |
| Failure mode     | Returns error to caller            | Log + metric, drop on queue full    |
| Used by          | P/D router, cross-node queries     | Write pipeline (after block seal)   |

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

## Configuration

The registrar is enabled via CLI flags on `pegaflow-server`:

```bash
pegaflow-server \
  --addr 10.0.0.1:50055 \
  --pool-size 30gb \
  --metaserver-addr http://127.0.0.1:50056
```

When `--metaserver-addr` is not set, registration is disabled (`None`) and the write path
skips the registration step with zero overhead.
