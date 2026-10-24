# Review: P2 connector skeleton (2026-10-23)

Reviewer: independent agent, fresh context, adversarial ("can a policy, adapter or vLLM path make the
connector report a hit whose bytes are wrong or missing?").
Scope: `python/pegaflow/connector_v2/**` (connector.py, core/{layout,scheduler,worker}.py, adapters/__init__.py,
policy.py, models/default.toml, README.md), `python/tests/test_connector_v2_core.py`, and the "P2 skeleton"
entry in `docs/rsi/PROGRESS.md`.
References read: design doc rev 62, `pegaflow-proto/proto/engine.proto`, `python/src/lib.rs`
(EngineRpcClient), `pegaflow-core/src/{lib.rs,lease.rs,gpu_worker.rs}`, `pegaflow-common/src/block.rs`, and
vLLM **0.32.0** source on .3 (`v1/core/sched/scheduler.py`, `sched/output.py`, `kv_cache_interface.py`,
`kv_connector/v1/base.py`, `multi_connector.py`, `kv_connector/utils.py`). Every vLLM claim below was checked
on the installed 0.32.0; nothing is marked [unverified-0.32].

## Verdict: APPROVE-WITH-FIXES

The skeleton is sound and matches the 0.32 contract:
- Loads are always async, and failures go through `failed_recving` (also in `finished_recving`).
- `skip_zeroing` comes from `get_loaded_kv_cache_group_ids`.
- gid and sid are threaded consistently through queries, targets, registration and saves.

The README claims "a bad policy costs hit rate, never correctness". That is false today. A policy can make
the core report a hit with unloaded blocks (M1), or save a wrong (block, hash) pair (M2). The core also
handles Checkpoint groups positionally, which vLLM 0.32 says is wrong for Mamba align states (M3). The tests
that should guard these contracts are skipped in the source-only gate and do not reach the save path (M4).
Fix M1-M4 before P3 starts. S3 must be fixed before P4, and S4/M3 before P5.

## Must-fix

**M1. A policy can widen a hit by mutating `present`. The core then reports tokens it never loads.**
`core/scheduler.py:163-187`, `policy.py:44`.
`policy.accept(p, lookup.present)` receives the live `list[set[int]]`. The recheck at line 173
(`need <= have`) reads the same sets. A policy that does `present[1].add(b)` (for example, "assume the
window block is there") passes the recheck. Then line 183 `lease.wanted = wanted[sid] & set(lease.positions)`
silently drops `b`, because no lease covers it. `_Hit.p` is still returned, and vLLM marks `[num_computed, p)`
computed. Block `b` keeps whatever was in the freshly allocated GPU block, and the result is wrong output.
If every wanted block is dropped this way, the `LoadPlan` has empty `loads`. `Worker.start_loads` puts it in
`_done` (`worker.py:124-125`), so the hit completes with nothing loaded.
Fix:
- Pass `tuple(frozenset(s) for s in present)` to the policy.
- Recheck against sets rebuilt from the server answers (the lease positions), not against the object the
  policy saw.
- Assert `wanted[sid] <= set(lease.positions)` for every group with a non-empty `wanted`, and require a lease
  for each such group. Otherwise return None (a miss).
- Add a table case with a mutating policy.

**M2. `choose_saves` can inject items it was not offered.** `core/scheduler.py:279-285`.
`offered = set(items)` is computed after `self._policy.choose_saves(items)`. A policy that appends to or
replaces entries in `items` in place gets its own items into `offered`, so the ValueError guard never fires.
The scheduler then saves an arbitrary `(block_id, block_hash)` pair, for example the hash of block b on the
GPU block of b+1. That pair poisons the store for every later reader of that hash: a wrong hit that no recheck
can catch.
Fix: `offered = frozenset(items)` before the call, and pass `tuple(items)`. Add a test case with a
misbehaving policy.

**M3. Checkpoint groups are saved and loaded by block position. vLLM 0.32 says that is wrong for Mamba align
state.**
`core/scheduler.py:270-278, 292-301` (save) and `226-232` (load target).
`KVConnectorBlockState.boundary_state_offloads` exists because, in 0.32 (`kv_cache_manager.py:871-883`,
`scheduler.py:1454-1471`), "Mamba 'align' boundary states must be handed off with exact block ids; they cannot
be reconstructed from a connector's append-only block table … never resolving them positionally". The core
ignores that field. It saves `block_ids[gid][b]` for any group `saveable` returns, including Checkpoint. It
also loads into `block_ids[gid][b]`.
Capping has the same effect: `after = min(computed + n_sched, num_tokens)` (line 265). A checkpoint block on a
boundary at `after` holds the state at `computed + n_sched`, not at `after`, whenever spec decode or async
placeholders make the two differ.
P2 has no adapters, so nothing triggers this yet. But P5 is defined as "add an adapter only". Its first
Checkpoint adapter would make the core store wrong state under a correct-looking key. Only V3 can catch that,
and only if the digest maps Mamba blocks correctly (see P0 S5).
Fix (cheap, fail-closed): `build_layout` raises `Unsupported("checkpoint groups need core support for
boundary_state_offloads")` for `kind == "checkpoint"` until the core:
- reads `boundary_state_offloads` for saves,
- has a verified load placement and `skip_zeroing` story for align mode,
- has a test for both.
Log this as a P5 core task, not an adapter task.

**M4. Core contract tests are skipped in the source-only gate and do not cover saves or policy safety.**
`tests/test_connector_v2_core.py:13`.
`pytest.importorskip("pegaflow.pegaflow")` skips the whole file without the native extension. The
source-only gate (CLAUDE.md) and the design ("Python 默认单测只测 connector 的状态机，在 semantics 的边界上用
stub，不需要编译扩展") both require these state-machine tests to run without the extension. README:62-63 says
the file is part of the default unit gate, but in CI it proves nothing.
The 7 cases pass on .3 (re-run by this reviewer: `7 passed in 0.03s`, remote files sha256-identical to the
local ones). They cover only matched-tokens, the hit lifecycle and Unsupported. Nothing exercises:
- `plan_saves`, `_saved_upto` or `choose_saves`;
- the `update_state_after_alloc` fail paths (mismatch, null block);
- `Worker.start_loads` / `poll` (the failed ⊂ finished invariant);
- a misbehaving policy (M1/M2).

Those are the contracts every RSI iteration touches.
Fix:
- Make semantics injectable into `Scheduler`, for example a `semantics` constructor argument that defaults to
  `pegaflow.semantics`.
- Drive the default-gate tests with a small table stub (fixed `resolve`/`needed`/`saveable` answers), so they
  run without the extension.
- Keep one native-backed case behind the existing importorskip. It is a cheap check that the real
  `pegaflow.semantics` matches the stub's signatures.
- Add table cases:
  - saves: full, window, a null id, async-capped `after`;
  - fail paths;
  - `Worker.poll` with a FakeLoadState;
  - the two policy-misbehavior cases.

## Should-fix

**S1. Under the default `kv_load_failure_policy="fail"`, any load problem becomes a user-visible request
error. Batching widens the blast radius.**
`core/scheduler.py:208-220`, `core/worker.py:128-143`.
- In 0.32, `_handle_failed_recving` (`scheduler.py:3341-3354`) returns the affected ids when not recomputing,
  and those requests finish with an error.
- A `num_external_tokens != planned` mismatch (for example the DSV4.1 `prefix_replay` trim at
  `scheduler.py:992-1003`) or a null target block becomes `plan.fail`. That is an error, not a fallback.
- All requests of a step share one `rpc.load`. One expired or invalid lease ("query lease is unknown or
  expired", `lease.rs:118`) fails every request in the step.

Fix:
- Isolate faults per request: one `load` per `LoadPlan`, or retry the batch per request on error.
- Require or document `kv_load_failure_policy=recompute` in README and in `run_gate`'s connector config.
- On a mismatch where `num_external_tokens < planned` and the plan can be cut, load the shorter prefix instead
  of failing.

**S2. Adapter declarations are barely validated, and the design's V0 check is missing.**
`core/worker.py:71`, `core/layout.py:102-117`.
- `register` only rejects `max(decl.segments) > view.bytes_per_block`.
- Only `len(decl.segments)` reaches the server.
- `decl.shard` is hashed into the namespace and never checked.

So an adapter that declares segments `(64, 64)` for a view with a 256-byte `bytes_per_block` passes. The
design (line 261) says to run `seal_plan` locally over every rank's declaration before registering.
Fix:
- In `build_layout`, build `RankDecl`s for all TP ranks (you already loop over them in `_layer_fingerprint`)
  and call `semantics.seal_plan`. Raise `Unsupported` on error.
- In `register`, require each declared segment size to equal what the server will copy per segment, for
  example `view.bytes_per_block`. State the meaning of `bytes_per_block` in `LayerView`'s docstring.

**S3. `transfer_backend="direct"` and `page_first=False` are hard-coded in core (P4 blocker).**
`core/worker.py:100-103`.
The proto (`engine.proto:39-48`) says MLA uses the `kernel` backend and page-first storage. The DeepSeek-V2
onboarding agent would have to edit core.
Fix: add `transfer_backend` and `page_first` to the adapter surface, for example on a per-group `GroupDecl`,
or derive them from `LayerView`. Validate that all layers agree, and include them in the namespace
fingerprint.

**S4. Groups other than sid 0 never fetch from SSD or remote nodes, and sid 0 may not be a full group.**
`core/scheduler.py:141-143`, `core/layout.py:85-87`.
`query_prefetch(..., wait_for_full_prefix=False, sid)` for sid>0 goes to the local-only membership query
(`lib.rs:604-622`). With `True` it goes to `query_group_membership_with_fetch` (`lib.rs:559-592`), which is
all-or-nothing with no lease when the answer is short.
Group 0's prefetch only fetches group-0 keys (`group_hash(h, 0)`), so window and checkpoint blocks evicted to
SSD are never hits. With no full group (all-window), the sort leaves a window group at sid 0 under prefix
semantics. Both cost hit rate only, so this is G3, not correctness.
Fix:
- For sid>0, when sid 0 answered `Loading` or the local membership is short, issue the fetch variant on the
  following retry for exactly the `needed(g, p)` hashes.
- Raise `Unsupported` when no full group exists, until a membership-only mode exists.

**S5. Saves block the step.** `connector.py:198-202`, `core/worker.py:153-158`.
`wait_for_save` does `current_stream().synchronize()` and then a blocking `save` RPC. The server awaits the
GPU→CPU copy (`gpu_worker.rs` batch_save). This is correct, and needed because the server copies on its own
stream, but every step that saves pays copy time on the critical path, and it defeats async scheduling
overlap. This is the first G4 lever.
Fix (later, policy-safe): record a CUDA event instead of synchronizing, issue the save asynchronously,
complete it through `finished_sending`, and have `request_finished_all_groups` return True while a save of
that request is in flight.

**S6. Exceptions after the query leak leases into a vLLM scheduler crash.** `core/scheduler.py:120`.
`_lookup` releases on error, but `_plan_hit` does not. Leases leak until the 600 s TTL in either of these
cases:
- `semantics.resolve`/`needed` raise (bad group, overflow), or
- a policy raises.

The exception also escapes into `Scheduler.schedule()` and kills the engine.
Fix: wrap `_plan_hit` the same way as `_lookup` (release, log, return `(0, False)`). Keep the raise in M2's
guard on purpose: a policy that lies about saves should fail loudly.

## Nits

- n1 `core/scheduler.py:177`: a block straddling `num_computed` turns the whole hit into a miss. vLLM passes
  `block_aligned_local` aligned to the scheduler block size, which is not the group block size. When a group's
  `bt` is larger than the scheduler block size, every request whose local hit is not a multiple of `bt` gets
  no external hit at all. Prefer lowering `p` (or document it as a G3 risk for P6).
- n2 `core/scheduler.py:261-268`: the comment says skipped positions "are picked up by a later step".
  But `_saved_upto` is advanced to `after` even when `_save_items` skips a block (missing hash, null id) or
  `block_ids_of` returns None, so those blocks are never offered again. In 0.32 this is practically
  unreachable for full and window groups. Fix the comment, or advance only past blocks that were offered.
- n3 Lease leak paths that remain (bounded by TTL 600 s; `release_instance` on unregister clears all):
  - one rank's `load` RPC fails before consuming while another rank consumed, so
    `remaining_consumers` never reaches 0;
  - a `LoadPlan` built but never executed (engine shutdown mid-step);
  - a `release` RPC that failed.

  `Scheduler.shutdown` releases only un-allocated hits, not leases already in `self._plan`.
- n4 `core/layout.py:93-97`, `connector.py:69-73`: the identity is `model` (a path), `dtype` and
  `kv_cache_dtype`. It lacks `revision`, `quantization` and the vLLM hash algorithm (`prefix_caching_hash_algo`).
  Two configs that differ only in those would share a namespace.
- n5 `policy.py:54`: `model_profile` from the extra config is used as a path component without checks
  (`"../x"`). Restrict it to `[A-Za-z0-9_.-]+`.
- n6 The `Policy()` default `loading_wait_ms=0` differs from the shipped `default.toml` value of 1000. Tests
  build `Policy()` directly, so the shipped defaults are never loaded in a test. Make the dataclass defaults
  equal to the TOML, or add one `load_policy(None)` assertion.
- n7 `connector.py:234-245`: `_server_device_id` falls back to the local ordinal for UUID or MIG entries in
  `CUDA_VISIBLE_DEVICES`. That is wrong whenever the server sees all GPUs. Raise, or map by UUID.
- n8 `docs/rsi/PROGRESS.md:68`: the phase table still says P2 "todo". Record "7 passed" with the caveat that
  the gate needs the native extension (M4).

## Answers to the author's open server questions

Sources: proto, `pegaflow-core/src/lib.rs`, `lease.rs` and `gpu_worker.rs`. The server query branches of
`service.rs` were not read.
1. **Save keying per group: yes.** The server maps each saved layer to its registered `layer_group_ids`
   entry, which is the sid. Keys are `group_hash(hash, sid)` and delegate to `semantics::key`
   (`pegaflow-common/src/block.rs:36`). Raw hashes in `SaveLayer` are correct. Because keys use the sid, the
   sid order is part of the storage layout. It is deterministic (it comes from the kinds), and the gid-ordered
   fingerprint covers it.
2. **Membership for group>0: local only** with `wait_for_full_prefix=False`. With `True` it does an SSD/RDMA
   set fetch, all-or-nothing, with no lease on a short answer. Unknown group ids are an error, not a miss.
   See S4.
3. **No partial lease release: correct.** The group-0 lease pins `[0, num_hit)`. `None` targets skip the
   copy, but the blocks stay pinned until each worker consumes the lease (`consumers = world_size`) or it is
   released. This costs memory, not correctness.
4. **`device_id`:** the mapping is right only if the server process sees every GPU in the default order. See
   n7.
5. **Namespace and `seal_plan`:** the namespace only isolates keys (`BlockKey(namespace, hash)`). The server
   seals by slot count, not by `seal_plan`, which is equivalent at TP1. The core never calls `seal_plan`; see
   S2 for the local V0 check.
6. **Failed loads: correct channel.** In 0.32, `invalid_block_ids` raises when there is more than one KV
   group (`scheduler.py:3365-3374`). `failed_recving` must be a subset of `finished_recving` (`base.py`,
   enforced by the aggregator in `utils.py:169-172`), and `poll` returns `finished | failed`. Under the
   default `"fail"` policy, though, the request errors. See S1.

## vLLM 0.32 contract check (verified, no action)

- **`get_num_new_matched_tokens`.** It is called only when `request.num_computed_tokens == 0`, with
  `block_aligned_local` (`scheduler.py:960-1027`). It always returns `load_kv_async=True`, and the
  partial-tail rule is handled by vLLM. A hit is never reported before its bytes are loaded:
  - the request stays in `WAITING_FOR_REMOTE_KVS` until every worker reports it in `finished_recving`;
  - the worker reports it only after `PyLoadState.is_ready()`, and reports it as failed when the state is not
    1;
  - the only bypass is M1.
- **`allocate_slots`.** It receives `skip_zeroing_group_ids=get_loaded_kv_cache_group_ids(request)`
  (`scheduler.py:1263-1267`), and those are the gids of kept leases. An allocation failure leaves the hit
  stored; the next call drops it (`_drop_hit`). The second `update_state_after_alloc` call (allowed by
  `base.py`) finds no hit and returns.
- **Abort while loading.** vLLM delays the free until `finished_recving` (`scheduler.py:2702-2707`).
  `request_finished` only drops scheduler bookkeeping, and the in-flight worker batch still reports the
  request. Returning `(False, None)` is correct because saves finish inside the step.
- **Preemption and resume.** A resumed request re-enters with `num_computed_tokens == 0` and re-queries.
  It goes to `scheduled_cached_reqs` (v1 runner, with `resumed_req_ids`) or `scheduled_new_reqs` (v2 runner,
  `scheduler.py:1418-1420`). Both are read, and `CachedRequestData.num_computed_tokens` exists. Re-saves are
  idempotent.
- **Async scheduling and chunked prefill.** `build_connector_meta` runs before `_update_after_schedule`
  (`scheduler.py:1527-1560`), so `num_computed_tokens` is the pre-step value. `after` is capped at
  `num_tokens`, and the next step resumes from `_saved_upto`. This is correct for full and window groups; for
  checkpoint groups see M3.
- **`block_state.get_block_ids`.** It returns None only for ids outside `req_ids`, and every scheduled
  request is inside it (`output.py:225-228`).
- **`MultiConnector`** forwards `get_loaded_kv_cache_group_ids` to the connector that owns the request
  (`multi_connector.py:430-432`).
- **APIs used exist in 0.32:** `KVCacheConfig.transfer_group_ids` (`kv_cache_interface.py:1496`),
  `hash_block_size` (1479), `ModelConfig.architecture` (`config/model.py:1187`),
  `get_dcp_world_size_and_rank`, `cp_kv_cache_interleave_size`.

## Semantics use

- **Hit.** `resolve`, then `min(policy.accept, p)`, then the step check, then `needed ⊆ present` per group.
  This is the right shape; M1 is the hole in it.
- **Save.** `saveable(g, before, after)`, then `choose_saves`. This is the right shape; M2 and M3 are the
  holes.
- **Model-specific logic in core.** The only model-specific rule in core is "full group first" (driven by the
  server's prefix query; see S4), plus the hard-coded backend (S3).
- **Sorting the full group to sid 0** does not break gid↔sid. Every place that touches vLLM uses gid:
  `block_ids[g.gid]`, `loaded_group_ids`, `SaveItem.group`. Every place that touches the server uses sid:
  query `group_id`, `targets[sid]`, `layer_group_ids`, `GroupSave.sid`. `test_hit_lifecycle` pins the `[1, 0]`
  order case.

## Isolation check (name-level only; old connector bodies not read)

I grepped `python/pegaflow/connector/` for 26 connector_v2 identifiers.
- **Zero hits (20):** `_drop_hit`, `_fill_targets`, `plan_saves`, `_saved_upto`, `_plan_hit`,
  `loaded_group_ids`, `_server_device_id`, `_ipc_bytes`, `endpoint_of`, `pegaflow_host`, `pegaflow_port`,
  `model_profile`, `StepPlan`, `LoadPlan`, `GroupSave`, `_with_worker_ranks`, `loading_wait_ms`,
  `min_hit_tokens`, `choose_saves`, `_loading_since`.
- **Shared names:**
  - `PegaConnectorMetadata` (18): also in the allowed `tests/test_connector_fault_tolerance.py` and
    `test_combine_hashes.py`; a natural name.
  - `_lookup` (9): generic; vLLM's mooncake store uses it too.
  - `"direct"` (3): the documented `register_context_batch` value in `python/src/lib.rs:314`.
  - `PEGAFLOW_HOST` (1): also in the allowed `tests/vllm_helpers.py`.
  - `group_hashes` (2).
  - `hash_scale` (5).

`hash_scale` is the only shared identifier with no other source in the repo (not in tests, docs, the design
doc or vLLM 0.32). It is a natural name for `block_size // hash_block_size`, and `GroupInfo.hash_scale` plus
the one-line `group_hashes` slice are trivial. This is weak evidence, not proof of copying. The overall
structure (Layout/GroupInfo, the Scheduler/Worker split by role, StepPlan/LoadPlan, choose_saves) shares no
other names. No action beyond noting it; if the orchestrator wants certainty, it can diff the two
`group_hashes` bodies without showing them to onboarding agents.

## P3 readiness

Qwen3-4B (FullAttentionSpec, TP1) can be onboarded by adding only `adapters/<name>.py` plus one `_REGISTRY`
line:
- `group`: `Group.full(block_size)`;
- `layer`: segments `(K, V)`, `Shard.heads`;
- `view`: a split-KV view with `kv_stride_bytes` = the dim-0 stride.

Fix M1-M4 first, and S2's segment-size check, so that a wrong `view` fails at registration instead of in V3.
P4 (MLA) needs S3, P5 (Mamba) needs M3 lifted properly, and P6 (Gemma4 window and full, some per-layer specs)
is expressible today (`layer_specs` handles `UniformTypeKVCacheSpecs`). The P4 and P5 items are core work and
should be scheduled for the orchestrator, not left to the onboarding agent.

## Resolution (2026-10-23, fix agent, isolated)

Scope: only `python/pegaflow/connector_v2/**`, `python/tests/test_connector_v2_core.py` and the docs.
I did not read the old connector, `instance.rs`, `offload.rs` or the `service.rs` query branches. The
evidence for this resolution was produced on .3:
- **Shared venv:** `77 passed`.
- **Source-only, real source-only proof:** a copy of the tree with no `*.so`, run under
  `uv run --isolated --no-project ...`. Result: `51 passed, 26 skipped`. The skips are the native-only
  parametrizations. In the remote checkout the in-tree `.so` loads, so the same command reports
  `77 passed`.
- **Ruff:** `ruff format --check` reports 10 files already formatted, and `ruff check` passes.

**M1. Fixed.**
- `_lookup` builds `present` as `tuple[frozenset[int], ...]` from the server answers only. The policy
  gets that immutable object, so `present[1].add(b)` raises.
- `_plan_hit` caps the answer: `p = max(0, min(operator.index(accept(...)), p))`. It re-checks the step.
- A new `_assign` re-derives `needed` per group against the server sets. For every group with non-empty
  `wanted`, it requires a lease from that group, and it requires `wanted ⊆ lease.positions`. If a group
  fails either check, `_assign` returns None (a miss). It never returns an empty lease list, so no
  `LoadPlan` can complete with nothing loaded.
- `_Lease` is now frozen and nothing is mutated in place.
- New cases: `policy-mutates-present`, `policy-widens`, `policy-lowers-into-gap`, `policy-lowers` and
  `policy-off-step`.

**M2. Fixed.**
- `offered` is built before the call, and the policy gets `tuple(items)`.
- Each returned item is looked up by equality in `offered`. `SaveItem` is frozen, so a forged
  `(block_id, hash)` pair does not match and raises ValueError. The loud failure is intentional.
- Duplicates are saved once.
- New cases: `forged-item` (ValueError), `mutates-items` (in-place append on the tuple raises), `subset`,
  `duplicate-once` and `save-none`.

**M3. Fixed (fail closed). Real support is deferred to P5 as a core task.**
- `build_layout` raises `Unsupported("... checkpoint groups need core support for vLLM
  boundary_state_offloads")` for any `kind == "checkpoint"` group, on both roles.
- Lifting it needs three things in core: saves driven by `boundary_state_offloads`, a verified load
  placement and `skip_zeroing` story for align mode, and tests for both. It is not an adapter task.
- Case: `test_layout_rejects[checkpoint]`.

**M4. Fixed.**
- `Scheduler(..., semantics=None)` and `build_layout(..., semantics=None)` take an injectable semantics
  module. The default comes from `core.layout.native_semantics()`, a lazy `from pegaflow import
  semantics`, so `core` imports without the extension.
- The test file has no module-level importorskip. It defines `StubSemantics`, a pure-Python model of
  `Group`, `Shard`, `Region`, `needed`, `resolve`, `saveable` and `seal_plan`.
- Cases marked `BOTH` also run against the real `pegaflow.semantics`: hit table, lifecycle, saves, layout
  rejects and seal_plan. `test_stub_matches_native` cross-checks the stub against native on 300 random
  cases (`resolve`, `needed`, `saveable` and the group attributes). Native cases skip without the
  extension.
- New coverage:
  - saves: prefill, chunked, null id, async-capped `after`, missing block ids (re-offered), unknown
    request;
  - `update_state_after_alloc` fail paths: more than planned, trimmed to a legal p (re-plans), trimmed
    off-step, null block, short block table, second call;
  - `Worker.start_loads` / `poll` with `FakeLoadState` (failed ⊂ finished, pending batch, per-request
    isolation);
  - registration checks, the save fan-out by sid, `shutdown`, and the policy-defaults and profile-name
    checks.

**S1. Partly fixed. The `run_gate` config is deferred.**
- `Worker.start_loads` issues one `load` RPC per `LoadPlan`, so an expired lease fails one request
  (case `bad-lease` in `test_worker_loads_and_poll`).
- On `num_external_tokens < planned`, if the shorter `p` is still a step-aligned legal resume point,
  `_recut` re-plans it from the held leases and releases the ones it no longer needs (`trimmed-replans`).
  Otherwise the load fails as before.
  - The skip-zeroing gids were already reported for the longer hit, so they can be a superset. That is
    harmless for full and window groups, because the blocks are written before they are read.
    Checkpoint groups are rejected (M3).
- README requires `kv_load_failure_policy=recompute`, and the scheduler logs a warning at init when it
  sees `fail`.
- Not done: `run_gate`'s connector config. `rsi/` is owned by another session; whoever owns `rsi/`
  should set `"kv_load_failure_policy": "recompute"` there before P3's gate run.

**S2. Fixed.**
- `register` requires every declared segment to equal `view.bytes_per_block`. That value is the
  server's per-segment copy size (`KVCacheLayout::new` takes it as `segment_bytes`).
- For multi-segment layers it also requires `kv_stride_bytes >= bytes_per_block`.
- The `LayerView` docstring defines both fields.
- `build_layout` builds every TP rank's `Region`s (sid, segments, shard) and runs `semantics.seal_plan`
  (V0). Errors become `Unsupported("adapter declarations fail seal_plan: ...")`.
- Cases: `test_worker_register[short-segment|whole-block]` and `test_layout_runs_seal_plan`, a rank-skewed
  TP2 declaration run against the stub and the native semantics.

**S3. Fixed.**
- `LayerDecl` gains `transfer_backend` (default "direct") and `page_first` (default False).
- `build_layout` requires every layer on every rank to agree, rejects unknown backends, and rejects
  `page_first` with more than one storage group (the proto says page-first stays single-group).
- Both fields are stored on `Layout`, hashed into the namespace and passed to `register_context_batch`.
- Cases: `mixed-backend`, `bad-backend`, `kernel-backend` and `test_namespace_covers_transfer_mode`.

**S4. Partly fixed. The fetch is deferred.**
- `build_layout` raises `Unsupported("no full-attention group ...")` when sorting cannot put a full
  group at sid 0 (case `window-only`).
- Deferred: issuing the fetching membership query for sid>0. It costs only hit rate (G3) when window or
  state blocks were evicted to SSD or remote. Schedule it as a core task before any gate that enables
  SSD or remote tiers, and by P6 at the latest.

**S5. Deferred (G4).** Async saves (a CUDA event instead of a synchronize, completion through
`finished_sending`, and `request_finished_all_groups` returning True while a save is in flight) change
the save lifecycle and need their own e2e evidence. They are the first perf lever after P3 passes
correctness.

**S6. Fixed.**
- `_plan_hit` runs inside `try`. On any exception (semantics or policy), the core logs, releases every
  lease of the lookup and returns `(0, False)`.
- `update_state_after_alloc` does the same around re-planning and target filling, and reports a failed
  load instead of raising into vLLM.
- The M2 guard still raises on purpose.
- Cases: `policy-raises`, `policy-mutates-present` and `test_semantics_error_releases_leases`.

**Nits.**
- n1: no code change; documented. The `num_computed` value passed to the connector is
  `block_aligned_local`, aligned to the scheduler block size (`scheduler.py:973-976`). With several
  groups, the scheduler block size is the lcm of the group block sizes (`kv_cache_utils.py:851`). So a
  block that straddles the local hit is unreachable in 0.32 without DCP, and the check stays as a
  fail-safe with a comment.
- n2: fixed.
  - When the block ids are missing, `_saved_upto` keeps `before`, so the range is offered again
    (`no-block-ids`, `test_saves_reoffer_after_missing_block_ids`).
  - The comment now says that blocks skipped for a null id or a missing hash are not offered again, and
    why that cannot happen for full and window groups in 0.32.
- n3: partly fixed. `Scheduler.shutdown` also releases leases in load plans that were never taken. The
  remaining paths are a partial consume across ranks and a failed `release` RPC. They are bounded by the
  600 s TTL and by `release_instance`, and a fix needs server-side per-consumer accounting; deferred to
  the server phase.
- n4: fixed. The identity adds `revision`, `quantization` and `prefix_caching_hash_algo`.
- n5: fixed. `model_profile` must match `[A-Za-z0-9_.-]+`, otherwise ValueError
  (`test_policy_profile_name_is_checked`).
- n6: fixed. The `Policy.loading_wait_ms` default is now 1000, matching `default.toml`, and
  `test_policy_defaults_match_shipped_toml` asserts `load_policy(None) == Policy()`.
- n7: fixed. `_server_device_id` raises when `CUDA_VISIBLE_DEVICES` is set and the entry is not a plain
  index (UUID or MIG). When the variable is unset, it uses the local ordinal.
- n8: fixed. The phase table and the log in PROGRESS.md are updated.

**Isolation note (`hash_scale`).** No action.
