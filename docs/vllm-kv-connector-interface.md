# vLLM KV Connector Interface Notes

This note records the vLLM V1 KV connector contract that PegaFlow must fit before
designing P/D disaggregation. It intentionally avoids PegaFlow-specific protocol
choices.

## Entry Points

vLLM loads a connector through `KVTransferConfig`:

- `kv_connector`: connector class name.
- `kv_connector_module_path`: optional external Python module path.
- `engine_id`: per-engine transfer identity, generated if omitted.
- `kv_role`: `kv_producer`, `kv_consumer`, or `kv_both`.
- `kv_buffer_device`: transfer buffer device, commonly `cuda`, `cpu`, or `xpu`.
- `kv_connector_extra_config`: connector-owned options.
- `enable_permute_local_kv`: experimental layout conversion flag.
- `kv_load_failure_policy`: `fail` or `recompute`.

The factory creates separate connector instances for:

- scheduler side: policy, matching, metadata construction, request lifecycle.
- worker side: KV cache registration, actual load/save/transfer, completion
  reporting.

External connectors can be provided by setting both `kv_connector` and
`kv_connector_module_path`.

## Request-Level Parameters

P/D information enters vLLM as ordinary request data:

1. OpenAI / completion request accepts `kv_transfer_params`.
2. The serving layer places it under `sampling_params.extra_args`.
3. `Request` copies `extra_args["kv_transfer_params"]` into
   `request.kv_transfer_params`.

This means the router is responsible for carrying connector-specific metadata
from a P output into the corresponding D input. vLLM does not interpret the
schema beyond passing the dict to the connector.

## Scheduler-Side Contract

The scheduler connector implements these core methods:

- `get_num_new_matched_tokens(request, num_computed_tokens)`:
  returns `(tokens, async_load)`. `tokens=None` means the scheduler should skip
  the request and ask again later. If `async_load=True`, vLLM allocates KV blocks
  for external tokens but does not schedule model compute for the request yet.
- `update_state_after_alloc(request, blocks, num_external_tokens)`:
  called after KV block allocation. The connector records enough state to build
  worker metadata for the load/save.
- `build_connector_meta(scheduler_output)`:
  returns an opaque `KVConnectorMetadata` object consumed by worker connectors in
  the same engine step.
- `update_connector_output(connector_output)`:
  receives worker-side completion/stat/event metadata after execution.
- `request_finished(request, block_ids)` or, for HMA connectors,
  `request_finished_all_groups(request, block_ids)`:
  called before vLLM frees the request blocks. Returning `True` means the
  connector owns delayed release until worker `get_finished()` reports the
  request in `finished_sending`. The optional returned dict becomes
  `kv_transfer_params` in the final `RequestOutput`.

The scheduler creates `SchedulerOutput.kv_connector_metadata` after scheduling
and before worker execution. The connector should treat `build_connector_meta()`
as the point where per-step pending state is consumed.

## Worker-Side Contract

The worker connector implements:

- `register_kv_caches(kv_caches)` or `register_cross_layers_kv_cache(...)`:
  called after vLLM allocates KV cache tensors. Connectors that need RDMA or
  CUDA IPC registration should do it here.
- `set_host_xfer_buffer_ops(copy_operation)`:
  gives connector a vLLM-provided block copy helper for host transfer buffers.
- `handle_preemptions(kv_connector_metadata)`:
  called before overwritten blocks are reused. Async save connectors must use
  this to prevent data corruption.
- `start_load_kv(forward_context)`:
  begins loads described by the scheduler metadata. This can be asynchronous.
- `wait_for_layer_load(layer_name)`:
  optional layer-level synchronization point.
- `save_kv_layer(layer_name, kv_layer, attn_metadata)`:
  optional layer-level save hook.
- `wait_for_save()`:
  called after the model forward. A connector may submit saves here instead of
  in `save_kv_layer()` if it is metadata-driven.
- `get_finished(finished_req_ids)`:
  returns `(finished_sending, finished_recving)`.
- `get_block_ids_with_load_errors()`:
  reports externally computed blocks that failed to load, so vLLM can recompute
  or fail according to policy.
- `get_handshake_metadata()`:
  optional worker-to-scheduler startup metadata for out-of-band P/D handshakes.

`KVConnectorOutput` is aggregated across workers. By default, a request is
considered finished only after every worker reports it. A connector can override
the expected completion count through `get_finished_count()` or dynamically via
`KVConnectorOutput.expected_finished_count`.

## Runtime Sequence

Startup:

1. vLLM creates scheduler and worker connector instances.
2. Workers register KV cache tensors.
3. The engine collects worker `get_handshake_metadata()` output.
4. Scheduler connector receives the merged metadata through
   `set_xfer_handshake_metadata(...)`.

D-side external KV load:

1. Scheduler calls `get_num_new_matched_tokens()`.
2. If the connector returns `(N, True)`, vLLM allocates blocks for `N` external
   tokens and marks the request `WAITING_FOR_REMOTE_KVS`.
3. Scheduler calls `update_state_after_alloc()`.
4. Scheduler emits connector metadata with `build_connector_meta()`.
5. Worker calls `start_load_kv()`.
6. Worker eventually reports the request in `finished_recving`.
7. Scheduler moves the request back to `WAITING` and caches successfully loaded
   blocks. If the whole prompt was loaded, vLLM rewinds one token so the next
   forward can sample correctly.

P-side delayed release / transfer publication:

1. When a request finishes, scheduler calls connector `request_finished...()`.
2. If the connector returns `(True, params)`, vLLM keeps the KV blocks alive.
3. The `params` dict is attached to the final output as `kv_transfer_params`.
4. Router forwards those params to the D request.
5. Worker reports `finished_sending` after consumers no longer need the P-side
   blocks.
6. Scheduler frees the held blocks.

## NIXL Reference Shape

NIXL uses `kv_transfer_params` as the P-to-D control plane:

- P request input can carry `do_remote_decode=True`.
- P `request_finished()` returns params like:
  - `do_remote_prefill=True`
  - `remote_block_ids`
  - `remote_engine_id`
  - `remote_request_id`
  - `remote_host`
  - `remote_port`
  - `tp_size`
- D receives those params and treats the request as a remote prefill load.
- D performs an out-of-band handshake to fetch worker transfer metadata, then
  starts the actual KV transfer.

Important implication: vLLM's interface already supports the async P/D ownership
model, but the connector decides how synchronous or asynchronous the transport
itself is.

## Design Implications for PegaFlow

- PegaFlow should treat vLLM block IDs as local, engine-scoped physical block
  handles. Any cross-engine protocol must use its own stable identities, such as
  block hashes, remote request IDs, or PegaFlow-managed transfer handles.
- `kv_transfer_params` is the natural place for router-carried P/D metadata.
  The schema should stay connector-owned and versioned.
- The selected initial P/D path is CPU-staging push: P writes KV into
  PegaFlow-managed D-side CPU/pinned memory, then D installs the staged KV into
  vLLM GPU KV blocks during `start_load_kv()`.
- In that path, D should not allocate vLLM GPU KV blocks while the P->D RDMA
  WRITE is still in flight. D connector should poll the PegaFlow staging lease
  through the existing `QueryPrefetch` path from `get_num_new_matched_tokens()`
  and return `(None, False)` until the write-with-imm completion marks the
  staging lease ready.
- First-cut control plane adds idempotent `PreparePdReceive`,
  `GetPdReceiveDescriptor`, and a D-side receive-state query. It should not add
  separate `Complete` or hot-path `Release` RPCs. Ready is driven by RDMA WRITE
  with immediate, and cleanup is handled by D-side TTL/GC or internal consume
  transitions.
- `request_finished...()` is the key hook for P-side block lifetime extension.
  `finished_sending` is the corresponding release signal.
- `get_num_new_matched_tokens()` must be conservative: it should only return
  tokens whose KV can actually be loaded. Returning `None` is available for
  pending discovery or prefetch.
- `start_load_kv()` is the key D-side hook for scheduling RDMA work before model
  execution. It is the right place to integrate with a PegaFlow-managed transfer
  queue.
- `get_block_ids_with_load_errors()` provides a built-in recompute/fail path,
  which PegaFlow should use instead of inventing a separate scheduler recovery
  mechanism.
- HMA support matters for modern models. A production P/D connector should
  implement `SupportsHMA` and reason in `tuple[list[int], ...]` block groups.
- Cross-layer blocks and required KV layout are connector-level choices via
  `prefer_cross_layer_blocks` and `get_required_kvcache_layout()`.
- Direct GPU push is deferred. It can reuse the same transaction model later,
  but should require stricter admission and explicit GPU memory-ordering
  validation before it reports `finished_recving`.

## Open Questions

- Should PegaFlow expose P/D as a new connector mode of `PegaKVConnector`, or a
  separate connector class with a stricter `kv_transfer_params` schema?
- Should PegaFlow reuse vLLM's P/D `remote_block_ids` style for compatibility,
  or only use PegaFlow block hashes and CPU-staging transfer handles?
- How much of the P/D control plane should live in the router versus the
  PegaFlow server/metaserver?
- For heterogeneous TP/block-size/layout support, which conversions belong in
  vLLM connector code and which belong below the PegaFlow server API?
- Delta transfer: if D already has a prefix of the KV locally or through P2P
  cache discovery, can `PreparePdReceive` communicate those hits so P only
  pushes missing blocks? This is deferred; initial CPU-staging push can transfer
  the full external KV span.
- What is the desired failure policy for D load failure: immediate request
  failure, local recompute, or retry through another P replica?
