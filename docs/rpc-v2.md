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
- `prefix_search = false` is a latest query over ascending keys. It answers only the last key held in memory, on SSD or at a peer, with its index in `hit_positions`. When keys follow the last one in memory, the first ask answers `Loading` while a task asks the MetaServer (`members` lookup) and SSD and fetches only the last one they hold. When nothing follows it, the answer is `Ready` at once. Checkpoint groups use it: a resume needs one state, and the latest one below the prefix bound is the best.

Poll a `Loading` answer with the returned `query_id` and the same keys.

## vLLM Connector

`python/pegaflow/connector_v2` (`kv_connector="PegaKVConnectorV2"`, `kv_connector_module_path="pegaflow.connector_v2"`) is the vLLM client of these RPCs. It handles TP only, with full attention, MLA, sliding-window and Mamba (`align`) cache groups.

- Each vLLM cache group is one namespace. The 32-byte namespace hashes the model identity and the group spec. The object prefix is the vLLM block hash of the group's last token block.
- `semantics.py` says what resuming at token `p` needs from each group: every block for full attention, the last `W - 1` tokens for a window, and the one state at `p` for a checkpoint. `resolve` picks the largest `p` that every group can back.
- Full groups take a prefix query from vLLM's local hit. Checkpoint groups take a latest query over every candidate point in the same pass, so their fetches overlap the prefix fetch; one above the prefix bound or the other checkpoints asks again below the lowest. Window groups take a prefix query over the blocks the largest point needs; one with a hole asks again at the highest lower point whose window ends below the hole.
- Every load is asynchronous. A failed load is reported as a failed receive, and vLLM recomputes the request, so run with `kv_load_failure_policy=recompute`.
- Saves follow `semantics.saveable`. Mamba states are saved only from vLLM's `boundary_state_offloads`. A saved block stays pinned in vLLM's block pool until every rank reports the save done.
- `policy.py` holds the per-model choices, as v1 defaults them. A pure-attention model (full and MLA groups only) saves one copy per slice. A hybrid model (any window or checkpoint group) saves one copy per slice on each CPU socket, so loads read a copy on their own socket. MLA models copy with the kernel and all others with direct copies; `pegaflow.transfer_backend` overrides.
- vLLM 0.32 keeps a rank's whole KV cache in one allocation, so each rank registers one region. The span of block `b` in layer `l` is `offset_l + b × block_stride_l` with length `page_size_bytes`.

Run the E2E suite with `PEGAFLOW_E2E_CONNECTOR=v2` to use it.

## Known Issues and Non-goals

- An RDMA fetch moves an object slot by slot, so both copies of a hybrid MLA model cross the network. The result is correct, but the traffic is doubled.
- Repeated queries may prefetch remote objects that are never loaded.
- No pipeline parallelism: `world_size` is the TP size.
- No cross-config sharing: objects saved under one TP layout cannot be loaded under another. Same-node deployments always share a config. Cross-node re-layout, if ever needed, belongs in the RDMA layer.
- Saver ranks and socket placement are not part of the namespace digest. Instances sharing a namespace must use compatible TP layouts, replica counts for each slice, and slot-to-slice ordering. E2E runs must verify and record these before testing cache sharing.
- DCP is not supported: `interleave_token` is stored but not read.
