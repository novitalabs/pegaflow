# RPC V2: Byte-Addressed Objects

RPC V2 stores KV cache as objects. Each object has a key `(namespace, prefix)`, two 32-byte digests. The client describes where the object's bytes sit in HBM as `[offset, len]` spans. The server does not know the model: it never sees heads, K/V, MLA, or checkpoints. It reads the sizes in `NamespaceMeta` only to cut the client's bytes into per-layer, per-slice slots.

V2 sits next to the v1 KV block RPCs in the same `Engine` service. A deployment uses one path or the other. V2 objects go through the same insert worker, pinned pool, LRU, SSD cache, RDMA fetch and MetaServer as v1 blocks.

The full spec, with field-by-field rationale, is the trellis task `11-08-rpc-v2-refactor`.

## RPCs

| RPC | Caller | What it does |
|-----|--------|--------------|
| `RegisterContextV2` | Each worker rank, once | Registers the rank's namespaces (`NamespaceMeta`) and HBM regions (CUDA IPC handles of whole `cuMemAlloc` allocations) |
| `SaveObjects` | Saver ranks | Copies each object's spans of every layer into pinned memory; returns after the copy |
| `QueryObject` | Scheduler | Prefix or membership lookup of keys of one namespace; returns `Loading` while fetching, then `Ready` with a lease |
| `LoadObjects` | Every rank | Copies leased objects into the rank's HBM; completion lands in a `LoadState` |
| `Release`, `Session` | Scheduler | Same as v1. `SessionRequest` carries only `instance_id` |

Business errors come back in `ResponseStatus status` (`ok = false`). The gRPC status is only for transport errors.

## Layout

One rank's slice of one layer is `L = local_hidden × block_size` bytes. A layer has `S = bytes_per_block / L` slices, and rank `r` holds slice `r × S / world_size`. GQA over TP4 with 4 KV heads gives `S = 4`, one slice per rank. MLA gives `S = 1`, so every rank holds the same slice.

An object has one slot per (saver rank, layer). It seals, becoming visible to queries, SSD and the MetaServer, once every saver has written every layer. When two savers map to the same slice (MLA with `saver_rank = [0, 4]`), the server keeps both copies, each on its writer's NUMA node. A load reads the copy on the loading rank's NUMA node when there is one.

Spans of one object list every layer of its namespace, in `layer_names` order, and must add up to `len(layer_names) × L`. Each span element is 16 bytes, `offset` then `len` as little-endian u64, with offsets counted from the start of the region's allocation.

A lease is consumed once per `LoadObjects` call that carries it and is released after `world_size` calls, one per rank.

## Queries

- `prefix_search = true` works like v1 `QueryPrefetch` without `wait_for_full_prefix`. It counts consecutive hits from `keys[0]` and prefetches misses from SSD or peers.
- `prefix_search = false` is a membership query. Unlike v1, it fetches every local miss it can find on SSD or on peers (MetaServer `members` lookup, then RDMA). It answers `Ready` only after those fetches finish. `hit_positions` lists the hit indices into `keys`.

Poll a `Loading` answer with the returned `query_id` and the same keys.

## Known Issues and Non-goals

- An RDMA fetch moves an object slot by slot, so both MLA copies cross the network. The result is correct, but the traffic is doubled.
- Repeated queries may prefetch remote objects that are never loaded.
- No pipeline parallelism: `world_size` is the TP size.
- No cross-config sharing: objects saved under one TP layout cannot be loaded under another. Same-node deployments always share a config. Cross-node re-layout, if ever needed, belongs in the RDMA layer.
- DCP is not supported: `interleave_token` is stored but not read.
