# internode/ — Cross-Node Communication

This module handles MetaServer coordination for PegaFlow's multi-node KV cache sharing.

## Module Overview

```
internode/
├── mod.rs                Module root + re-exports
└── metaserver_client.rs  MetaServer registration, removal, query, and node heartbeat
```

## Data Flow

```
  write path                       read path
      |                                |
      v                                v
  try_register_namespace          query_prefix
  try_unregister
      |                                |
      v                                v
  background MetaServer loop       lazy MetaServer gRPC client
      |                                |
      +--------------+-----------------+
                     |
                     v
            pegaflow-metaserver
```

## MetaServer Client Responsibilities

| Responsibility | Path | Behavior |
|----------------|------|----------|
| Block registration | Write path | Fire-and-forget `try_send` after block seal |
| Block removal | Eviction path | Best-effort remove messages after local cache eviction |
| Prefix query | Read path | Request-response query for per-node prefix lengths |
| Node session | Background task | Heartbeat, stale-session recovery, and graceful unregister |

## How Registration Integrates

The MetaServer client plugs into the **write path** via `InsertDeps`:

```
insert_worker_loop (sync thread)
  └─► process_insert_batch
        └─► send_backing_batches
              ├─► SsdBackingStore::ingest_batch()     (SSD write, fire-and-forget)
              └─► MetaServerClient::try_register_namespace() (MetaServer registration, fire-and-forget)
```

Both use the same pattern: `tokio::sync::mpsc::try_send()` from the sync insert thread,
with an async tokio task draining the channel on the other end. The background
task generates a local `node_id`, announces it with `HeartbeatNode`, and attaches
`{node, node_id}` to insert/remove RPCs. If the MetaServer restarts or rejects a
missing session, the task sends heartbeat again and continues with future queued
updates. It does not backfill resident cache keys that were registered before
the MetaServer lost state.

## Configuration

MetaServer registration is enabled via CLI flags on `pegaflow-server`:

```bash
pegaflow-server \
  --addr 10.0.0.1:50055 \
  --pool-size 30gb \
  --metaserver-addr http://127.0.0.1:50056
```

When `--metaserver-addr` is not set, registration is disabled (`None`) and the write path
skips the registration step with zero overhead.
