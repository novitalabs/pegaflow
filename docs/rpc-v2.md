# RPC V2: Byte-Addressed Objects

RPC V2 stores KV cache as objects. Each object has a key `(namespace, prefix)`, two 32-byte digests. The client describes where the object's bytes sit in HBM as `[offset, len]` spans. The server does not know the model: it never sees heads, K/V, MLA, or checkpoints. It reads `NamespaceMeta` only to place the client's bytes in per-layer, per-group slots. It knows devices, not ranks.

V2 sits next to the v1 KV block RPCs in the same `Engine` service. A deployment uses one path or the other. V2 objects go through the same insert worker, pinned pool, LRU, SSD cache, RDMA fetch and MetaServer as v1 blocks.

## RPCs

| RPC | Caller | What it does |
|-----|--------|--------------|
| `RegisterContextV2` | Each worker device, once | Registers the device's role in each namespace (`NamespaceRole`: `NamespaceMeta`, `group`, `save`) and its HBM regions (CUDA IPC handles of whole `cuMemAlloc` allocations) |
| `SaveObjects` | Devices that save | Copies each object's spans of every layer into its group's slots in pinned memory; returns after the copy |
| `QueryObject` | Scheduler | Prefix or membership lookup of keys of one namespace; returns `Loading` while fetching, then `Ready` with a lease |
| `LoadObjects` | Every device | Copies its group's slots of leased objects into the device's HBM; completion lands in a `LoadState` |
| `Release`, `Session` | Scheduler | Same as v1. The server only logs `SessionRequest.namespace` |

Business errors come back in `ResponseStatus status` (`ok = false`). The gRPC status is only for transport errors.

## Layout

`NamespaceMeta` is byte-identical on every device of an instance: `namespace`, `layer_names`, `slice_bytes`, `num_groups` (G), `world_size` and `interleave_token`. An object of the namespace is `G × len(layer_names)` slots of `slice_bytes`; slot `layer × G + group` belongs to group `group`. A group is one copy of one part of a layer, e.g. one TP head shard, or the whole MLA latent on one CPU socket.

Each device registers, per namespace, the `group` it loads and whether it saves it (`save`). The server checks that `group < G`, that the meta matches what other devices registered, and that no group gets two savers; it needs no "all registered" step. Groups are per namespace, so one device can hold group 0 of one namespace and group 1 of another. Which group a device takes is the client's choice; the server never sees ranks. Examples:

| Deployment | G | `slice_bytes` | Device `(group, save)` |
|---|---|---|---|
| GQA, 8 KV heads over TP4 | 4 | one page of 2 heads | `(0,✓) (1,✓) (2,✓) (3,✓)` |
| GQA, 4 KV heads over TP8 (heads replicated on 2 ranks) | 4 | one page of 1 head | `(0,✓) (0) (1,✓) (1) (2,✓) (2) (3,✓) (3)` |
| MLA over TP8, one copy | 1 | one page | all group 0, world rank 0 saves |
| MLA over TP8, one copy per socket | 2 | one page | groups `0,0,0,0,1,1,1,1`, world ranks 0 and 4 save |

An object seals, becoming visible to queries, SSD and the MetaServer, once every group's saver has written every layer. Each slot sits on its saver's NUMA node. A group without a saver is a client bug: its objects never seal and expire with unsealed KV blocks.

Spans of one object list every layer of its namespace, in `layer_names` order, `slice_bytes` per layer; no span crosses a layer, and a layer may take several spans. Save and Load use the same spans and the same slot, `layer × G + group`. Each span element is 16 bytes, `offset` then `len` as little-endian u64, with offsets counted from the start of the region's allocation. A load checks that the object has `G × layers` slots of at least `slice_bytes` (slots are padded to the SSD alignment when SSD is on).

A lease is consumed once per `LoadObjects` call that carries it and is released after `world_size` calls, one per device.

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
- `policy.py` holds the per-model choices, as v1 defaults them. A pure-attention model (full and MLA groups only) saves one copy of each part. A hybrid model (any window or checkpoint group) saves one copy of each part on each CPU socket, so loads read a copy on their own socket. MLA models copy with the kernel and all others with direct copies; `pegaflow.transfer_backend` overrides.
- vLLM 0.32 keeps a rank's whole KV cache in one allocation, so each rank registers one region. The span of block `b` in layer `l` is `offset_l + b × block_stride_l` with length `page_size_bytes`.
- Groups: per namespace, each world rank holds one part of a layer: its TP head shard and, for a DCP-sharded group, its DCP rank. vLLM replicates a KV head on `tp / kv_heads` adjacent ranks when TP exceeds the model's KV heads, and those ranks hold the same part. The connector gives each distinct (part, CPU socket) one group, in sorted order, and the lowest world rank of each group saves it. `world_size` is `tp × pcp`. PCP ranks hold the same bytes. Under DCP one vLLM block covers `block_size × dcp` tokens and each DCP rank holds its own share of them; Mamba states are not DCP-sharded. So under TP2 × PCP2 × DCP2 an MLA namespace takes groups `(0, 0, 1, 1)` while a Mamba namespace takes `(0, 1, 0, 1)`.

Run the E2E suite with `PEGAFLOW_E2E_CONNECTOR=v2` to use it.

## Known Issues and Non-goals

- An RDMA fetch moves an object slot by slot, so both copies of a per-socket MLA object cross the network. The result is correct, but the traffic is doubled.
- Repeated queries may prefetch remote objects that are never loaded.
- Objects go to the saver's NUMA node only. With one copy of a replicated part (MLA), the pinned pools of the other ranks' nodes stay unused. This matters on hosts with several NUMA nodes per socket; v1 stripes MLA saves over every rank instead.
- No pipeline parallelism.
- No cross-config loads yet. The namespace digest leaves out the parallel layout (TP/PCP/DCP sizes, groups, socket placement, DCP interleave size `cp_kv_cache_interleave_size`), so instances of one model share namespaces across layouts. Until cross-config loads exist, deployments keep instances that share a namespace on the same layout, copies of each part and interleave size; E2E runs must verify and record it before testing cache sharing. A load across layouts usually fails the slot-count check of `G × layers`; it reads wrong bytes when the slot counts match, as when only the interleave size differs. Same-node deployments always share a config; cross-node re-layout belongs in the RDMA layer.
- `interleave_token` in `NamespaceMeta` is stored but not read; the connector sends 0.
