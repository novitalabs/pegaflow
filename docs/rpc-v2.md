# RPC V2: Byte-Addressed Objects

RPC V2 stores KV cache as objects. Each object has a key `(namespace, prefix)`, two 32-byte digests. The client describes where the object's bytes sit in HBM as `[offset, len]` spans. The server does not know the model: it never sees heads, K/V, MLA, or checkpoints. It reads the sizes in `NamespaceMeta` only to cut the client's bytes into per-layer, per-slice slots.

V2 sits next to the v1 KV block RPCs in the same `Engine` service. A deployment uses one path or the other. V2 objects go through the same insert worker, pinned pool, LRU, SSD cache, RDMA fetch and MetaServer as v1 blocks.

## RPCs

| RPC | Caller | What it does |
|-----|--------|--------------|
| `RegisterContextV2` | Each worker rank, once | Registers the rank's namespaces (`NamespaceMeta`) and HBM regions (CUDA IPC handles of whole `cuMemAlloc` allocations) |
| `SaveObjects` | Saver ranks | Copies each object's spans of every layer into pinned memory; returns after the copy |
| `QueryObject` | Scheduler | Prefix or membership lookup of keys of one namespace; returns `Loading` while fetching, then `Ready` with a lease |
| `LoadObjects` | Every rank | Copies leased objects into the rank's HBM; completion lands in a `LoadState` |
| `Release`, `Session` | Scheduler | Same as v1. The server only logs `SessionRequest.namespace` |

Business errors come back in `ResponseStatus status` (`ok = false`). The gRPC status is only for transport errors.

## Layout

One rank's slice of one layer is `L = local_hidden × block_size` bytes. A layer has `S = bytes_per_block / L` slices, and rank `r` holds slice `r × S / world_size`. GQA over TP4 with 4 KV heads gives `S = 4`, one slice per rank. MLA without DCP gives `S = 1`, so every rank holds the same slice.

An object has one slot per (saver rank, layer). It seals, becoming visible to queries, SSD and the MetaServer, once every saver has written every layer. When two savers map to the same slice (MLA of a hybrid model over two sockets, `saver_rank = [0, 4]`), the server keeps both copies, each on its writer's NUMA node. A load reads the copy on the loading rank's NUMA node, else one on its socket, else the first.

Spans of one object list every layer of its namespace, in `layer_names` order, and must add up to `len(layer_names) × L`. Each span element is 16 bytes, `offset` then `len` as little-endian u64, with offsets counted from the start of the region's allocation.

A lease is consumed once per `LoadObjects` call that carries it and is released after `world_size` calls, one per rank.

## Queries

- `prefix_search = true` works like v1 `QueryPrefetch` without `wait_for_full_prefix`. It counts consecutive hits from `keys[0]` and prefetches misses from SSD or peers.
- `prefix_search = false` is a membership query. Unlike v1, it fetches every local miss it can find on SSD or on peers (MetaServer `members` lookup, then RDMA). It answers `Ready` only after those fetches finish; without a peer to ask, it checks SSD at once and answers `Ready` when SSD holds none. `hit_positions` lists the hit indices into `keys`. Checkpoint groups use it: the scheduler picks the highest point every checkpoint group holds and loads only that state of each.

Poll a `Loading` answer with the returned `query_id` and the same keys.

## vLLM Connector

`python/pegaflow/connector_v2` (`kv_connector="PegaKVConnectorV2"`, `kv_connector_module_path="pegaflow.connector_v2"`) is the vLLM client of these RPCs. It handles TP, PCP and DCP (no PP), with full attention, MLA, sliding-window and Mamba (`align`) cache groups.

- Each vLLM cache group is one namespace. The 32-byte namespace hashes the model identity, the group index, its spec without per-rank fields (block size, heads, state and page sizes) and the tokens one object covers, so every TP/PCP/DCP layout of a model gets the same namespaces. The object prefix is the vLLM block hash of the group's last token block.
- `semantics.py` says what resuming at token `p` needs from each group: every block for full attention, the last `W - 1` tokens for a window, and the one state at `p` for a checkpoint. `resolve` picks the largest `p` that every group can back.
- Full groups take a prefix query from vLLM's local hit. Once the prefix answers, checkpoint groups take a membership query over the candidate points up to the prefix bound, in the same step when the prefix was already in memory. The highest point every checkpoint group holds wins; only that state of each loads, and the rest of each lease is released. Window groups take a prefix query over the blocks that point needs; one with a hole asks again at the highest lower point whose window ends below the hole.
- Every load is asynchronous. As v1, a failed load of a single-group model reports the GPU blocks it was writing; vLLM cuts the hit at the first one and keeps the local prefix. vLLM takes block ids only with one cache group, so a hybrid model reports a failed receive and the request starts over. As v1, the connector runs under vLLM's default `kv_load_failure_policy=fail`, so a failed load fails the request. Under `recompute` the request recomputes instead and never loads again.
- Saves follow `semantics.saveable`. Mamba states are saved only from vLLM's `boundary_state_offloads`. A saved block stays pinned in vLLM's block pool until every rank reports the save done.
- `policy.py` holds the per-model choices, as v1 defaults them. A pure-attention model (full and MLA groups only) saves one copy per slice. A hybrid model (any window or checkpoint group) saves one copy per slice on each CPU socket, so loads read a copy on their own socket. MLA models copy with the kernel and all others with direct copies; `pegaflow.transfer_backend` overrides.
- vLLM 0.32 keeps a rank's whole KV cache in one allocation, so each rank registers one region. The span of block `b` in layer `l` is `offset_l + b × block_stride_l` with length `page_size_bytes`.
- PCP and DCP are handled on the client side; the server only sees slices. `world_size` is `tp × pcp`. PCP ranks hold the same bytes. Under DCP one vLLM block covers `block_size × dcp` tokens and each DCP rank holds its own share of them, so a DCP-sharded group has `dcp` times as many slices. Mamba states are not DCP-sharded. The connector numbers server ranks so the ranks holding one slice are consecutive, as `slice = rank × S / world_size` needs. vLLM puts TP innermost, so under PCP the holders of one slice are not always consecutive in world rank order: GQA with 2 KV heads over TP2 × PCP2 holds heads `(0, 1, 0, 1)` and registers server ranks `(0, 2, 1, 3)`.

Run the E2E suite with `PEGAFLOW_E2E_CONNECTOR=v2` to use it.

## Known Issues and Non-goals

- An RDMA fetch moves an object slot by slot, so both copies of a hybrid MLA model cross the network. The result is correct, but the traffic is doubled.
- Repeated queries may prefetch remote objects that are never loaded.
- Objects go to the saver's NUMA node only. With one copy of a replicated slice (MLA), the pinned pools of the other ranks' nodes stay unused. This matters on hosts with several NUMA nodes per socket; v1 stripes MLA saves over every rank instead.
- No pipeline parallelism.
- No cross-config loads yet. The namespace digest leaves out the parallel layout (TP/PCP/DCP sizes, slices, saver ranks, socket placement, DCP interleave size `cp_kv_cache_interleave_size`), so instances of one model share namespaces across layouts. Until cross-config loads exist, deployments keep instances that share a namespace on the same layout, replica counts for each slice and interleave size; E2E runs must verify and record it before testing cache sharing. A load across layouts usually fails the slot-count check; it reads wrong bytes when the slot counts match, as when only the interleave size differs. Same-node deployments always share a config; cross-node re-layout belongs in the RDMA layer.
- `interleave_token` in `NamespaceMeta` is stored but not read; the connector sends 0.
