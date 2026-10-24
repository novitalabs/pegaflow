# Review: P3 Qwen3-4B it2d (fail closed after a failed sync load)

Reviewer: independent review agent (fresh context), 2026-10-24.
Scope: `8af8fe4..8a05b7c` (python/pegaflow/connector_v2 + tests), gate
backport `df01403`, docs in `a49bd21`, read against the whole connector_v2
package and vLLM v0.32.0 sources (`v1/core/sched/scheduler.py`,
`v1/engine/core.py`, `config/vllm.py`, `kv_connector/v1/{base,multi_connector}.py`,
`kv_connector/utils.py`, `v1/worker/kv_connector_model_runner_mixin.py`,
`v1/worker/gpu_model_runner.py`, `v1/core/kv_cache_manager.py`).

Verdict: **ACCEPT-WITH-FIXES**. No must-fix. M1' is fixed: once vLLM reports
a failed sync load, no save can reach the server. The should-fix items are a
residual vLLM-side exposure, test gaps, and evidence/process wording.

## Must-fix

None.

## Should-fix

- **S1** Sync loads keep running after saves are disabled
  (`core/scheduler.py:192`). Fail-closed protects the store only. Under the
  recompute policy vLLM keeps the bad blocks prefix-cached (`scheduler.py`
  `_handle_invalid_blocks`: no eviction under recompute). Requests that hit
  them before the first holder recomputes can produce wrong outputs in this
  process: a sharer not rewound (`marked_invalid_block_ids` case), or a hit on
  a preempted/aborted loader's freed blocks. Every further sync failure (e.g.
  a server restart, or lease expiry) adds more such blocks. Failure scenario:
  a server hiccup fails sync loads repeatedly for minutes, and each one
  plants more bad KV in vLLM's GPU cache. Fix: also gate
  `hit.sync` on `not self._saves_disabled`, which makes every load async after
  the first failure (async failures are clean: blocks stay uncached until
  `finished_recving`, and a failed request restarts from 0). Also record the
  vLLM-side output risk in ITERATIONS "Open risks". Today the doc only
  says the store is protected.
- **S2** The fail-closed test does not pin the lifecycle that must survive
  disabling (`tests/test_connector_v2_core.py:545-616`). I ran two mutations
  against the stub suite, and both still passed (88 passed):
  (a) moving `if self._saves_disabled: return` above the preemption loop in
  `plan_saves` (`core/scheduler.py:422-427`), which drops `plan.flush` while
  pre-failure jobs are still in flight. Scenario: a preempted request's
  freed blocks are reused while an older save job still reads them.
  (b) making `request_finished` return False once disabled
  (`core/scheduler.py:516`), which frees blocks under an in-flight save.
  The current code is correct on both counts. Add one table row: a request
  that saved before the failure is preempted or finished after it, and
  `plan.flush` / `plan.finished` are still emitted. Removing the early
  return does fail 2 cases, and an off-by-one in `until` fails 3, so the
  core contract is covered.
- **S3** The provenance claim does not match its artifact. ITERATIONS
  (`docs/rsi/models/Qwen3-4B/ITERATIONS.md:647`) says "import path checked in
  vllm.log", but no file in the run dir contains `pegaflow-rsi-qwen3-it2d`.
  vllm.log has no path for the `pegaflow` package. `run_gate.vllm_server`
  puts only REPO, not `REPO/python`, on PYTHONPATH (`rsi/gates/run_gate.py:244`).
  The shared venv's `.pth` points at `~/kexi/pegaflow-semantics-rsi/python`,
  which is still at 056eb4b and has no `_saves_disabled`. So the run depended
  on a PYTHONPATH that was set outside the gate and was never recorded. I
  confirmed the it2d code ran another way: in the it2d tree,
  `connector_v2/__pycache__/connector.cpython-312.pyc` has mtime 17:42:05.4,
  the same second the v2 gsm8k vLLM started. The it2d tree's connector_v2
  `.py` md5s equal 8a05b7c. Fix the sentence to cite that evidence. For
  later runs, have the gate (gate owner) or the connector log
  `pegaflow.__file__` at init.
- **S4** Acceptance relies on exit code 1. The branch gate counts `old`'s
  verdicts in the exit code (`rsi/gates/run_gate.py:813`). GATES.md:151 says
  exit 0 is the bar, and GATES.md:56-57/84 call `old` reference-only. This is
  documented, and both prior reviews accepted it. But "exit 0" is still never
  shown for this model. Either run the official profile without `old`
  (it is not in `official_problems`' required kinds), or land the
  reference-arm exclusion through the gate owner.
- **S5** (pre-existing, from P2) Policy rule "every knob needs gate evidence;
  a knob no accepted model uses is deleted"
  (`.claude/skills/pegaflow-new-model/SKILL.md:87-88`):
  - `min_hit_tokens = 0` is a no-op that no accepted model overrides.
  - `loading_wait_ms` has no gate evidence on this setup (no SSD/RDMA tier,
    so no QueryLoading answers are expected).
  - `sync_load_tokens` has evidence (it2).

  Delete `min_hit_tokens`, or record why it stays. The rest of the rules hold:
  `policy.py` names no model or vLLM spec class (grep: no hits), it has 111
  lines and 3 knobs, `default.toml` has 3 keys, and no `models/<model>.toml`
  exists.

## Nits

- `core/worker.py:229-231`: a sync `LoadPlan` with no loads and `fail=False`
  reports nothing, so vLLM would treat unwritten blocks as computed.
  Unreachable today (`_assign` never returns an empty list, and
  `_fill_targets` appends one entry per lease). Fail-safe: report
  `lp.covered` invalid.
- `rsi/gates/run_gate.py:119`: `stop()` raising inside `finally` replaces an
  exception already propagating from the stage body. It only affects
  diagnostics.
- `rsi/gates/test_teardown.py` cannot run in the source-only env: importing
  `run_gate` pulls in `oracle` -> `pegaflow.semantics` -> the native extension.
  I could not run it locally, so the ".44 passes" claim is unverified by me.
- `connector.py:153-155`: the `getattr` fallback is dead on 0.32
  (`max_concurrent_batches` is a `VllmConfig` property). Harmless, and
  conservative (pp + 1 is the largest value).
- G4 vs the last accepted run (it2 official): replay warm2 TTFT 0.563 vs
  0.541 s (+4.1%), inside the 5% bound but close. warm3 is 0.538 vs 0.552.

## Verified

Fail-closed (question 1):
- Raising paths. Every load failure path raises `invalid_block_ids` for sync
  loads: `lp.fail`, RPC error/exception in `_start_load`, final state != 1
  (`worker.py:224-244, 286-292`). There is no deadline, so no timeout path.
  An exception from `is_ready` propagates and fails loud.
- Aggregation. TP ranks are unioned by `KVOutputAggregator`
  (`utils.py:168`). MultiConnector unions children's
  `get_block_ids_with_load_errors` and forwards `update_connector_output` to
  every child (`multi_connector.py:378, 496-515`). Truthiness: the dataclass
  is always truthy, and `is_empty()` counts `invalid_block_ids`.
  `update_from_output` has no early return before
  `_update_from_kv_xfer_finished`.
- Scope. The scheduler plans saves for all workers, so one rank's failure
  stops saves on every rank of the engine. After the flag is set,
  `plan_saves` plans no save (`scheduler.py:426`). Preemption flushes and
  `finished` still ship (`request_finished` appends to `self._plan`
  directly), so every held request still reaches `finished_sending`: no
  block leak.
- In-flight jobs. Saves already submitted before the flag cover steps
  <= N+1, and none of them touches a sync-loaded block (held).
- Async failures stay safe. `_handle_failed_recving` sets computed to 0, and
  `_update_waiting_for_remote_kv` frees the blocks, which were never cached
  (`delay_cache_blocks`).

Hold window (question 2):
- `until = _steps + max_concurrent_batches` is set in
  `update_state_after_alloc`, inside the same `schedule()` call and before
  that step's `build_connector_meta`. Plans N..N+k-1 hold, plan N+k
  releases.
- `step_with_batch_queue` asserts `len(queue) < size` before every schedule,
  so `update_from_output(N)` runs before `schedule(N+size)` (`core.py:720-783`).
  `size = max_concurrent_batches` (2 with V1 async + PP1; pp+1 on V2 async;
  1 sync). Deferred structured-output batches keep the same order.
- Every `schedule()` calls `build_connector_meta`, so `_steps` counts
  scheduled batches.
- Prefix sharing keeps a block at the same index, so `_touches_unconfirmed`
  catches sharers.
- Finish drops `_saved_upto`. Preempt keeps `before = min(computed, ...)`.
  Expired keys are filtered before the early return.

Taint path (question 3):
- `_handle_invalid_blocks` raises for more than one KV group.
- `sync_loads` requires one group and `not needs_kv_cache_zeroing`, a
  property in 0.32 (`kv_cache_interface.py:1586`).
- `covered` = every non-null block holding external tokens of the hit
  vLLM allocated.

Gate integrity (question 4):
- `git diff 8af8fe4..a49bd21 -- rsi docs/rsi/GATES.md` touches only
  `stop`/`_group_gone` + the test.
- The change waits longer (TERM 60->120 s, KILL 30->300 s) and waits for the
  whole process group rather than the leader alone, which is stricter. It
  still raises if the group survives.
- No verdict, constant or official rule changed.
- The official run itself used the old teardown: no "gone after" lines; the
  `aborted1` traceback is the old `proc.wait(timeout=30)`.

Policy (question 5): see S5.

Unit tests (question 6):
- `tests/test_connector_v2_core.py` locally: 88 passed, 43 skipped (native).
- Mutation results: removing the early return fails `sync-failed-sharer-next-turn`
  and `sync-failed-loader-preempted`; `until - 1` fails 3 cases. See S2 for
  the surviving mutants.

Isolation: the diff touches only `connector_v2`, its tests and docs. I read
none of the listed isolated paths. S3 needed no isolated path.

Official run `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2d` (read-only,
checked against `summary.json` / `result.json`):
- Run identity: `official: true`, `not_official_because: []`, rev 8a05b7c,
  arms plain,baseline,old,v2, official args.
- v2 verdicts all pass:
  - G0.
  - G2 digests: gsm8k external 6643/6643, replay 18545/18545, 0 mismatched.
  - G2 accuracy: 0.8582 = baseline.
  - G3: gsm8k_warm, replay_warm, no_eviction (all 0).
  - G4: values match the doc to 3 digits.
- 0 load errors in `result.json`.
- 0 `sync load failed` / `saves disabled` lines.
- 28 `save stats` lines, all `failed=0`.

| G4 check | v2 | old | baseline |
|---|---|---|---|
| replay warm TTFT p50 (warm2/3) | 0.563 / 0.538 s | 0.677 / 0.601 s | 2.544 / 2.547 s |
| replay warm req/s (warm2/3) | 10.12 / 9.94 | 9.99 / 10.03 | 3.42 / 3.42 |
| replay cold TTFT p50 / req/s | 1.287 s / 6.70 | 1.334 s / 6.75 | 2.804 s / 3.26 |
| gsm8k cold / warm seconds | 165.96 / 164.29 | 166.7 / 169.9 | 164.45 / 164.28 |

| v2 pass | local | external | total | H* (oracle) |
|---|---|---|---|---|
| gsm8k cold | 0.933 | 0 | 0.933 | 0.933 |
| gsm8k warm | 0.920 | 0.072 | 0.992 | 0.992 |
| replay cold | 0.381 | 0.484 | 0.864 | 0.864 |
| replay warm / warm3 | 0.348 / 0.347 | 0.650 / 0.651 | 0.998 | 0.998 |

Not verified: runtime behaviour of a failed sync load (none occurred), the
teardown test (S3/nit), and E_ext / excess numbers beyond the passing G3
verdicts.

## Resolution

Resolved in it2e, one commit on `feat/semantics-rsi`; see `models/Qwen3-4B/ITERATIONS.md` it2e. None of the changes touches the Qwen3-4B success path, so G0-G4 stay accepted at 8a05b7c. The next official run (it3 or the G5 rerun) covers the final code.

| ID | Resolution |
|---|---|
| S1 | Fixed. `hit.sync` also requires `not _saves_disabled`, so every load is async after the first reported failure (a failed async load is never cached). The README records the residual vLLM-side output risk until the rewound holder recomputes. Test: `test_lifecycle_after_saves_disabled[later-loads-async]`. |
| S2 | Fixed. `test_lifecycle_after_saves_disabled[preempted-still-flushes\|finished-still-held]`: a request that saved before the failure still gets `plan.flush` on preemption and `plan.finished` + hold on finish. Mutants run on .44: (a) early return above the preemption loop, 1 failed; (b) `request_finished` False once disabled, 1 failed; dropping the S1 condition, 1 failed. |
| S3 | Fixed. The ITERATIONS it2d sentence now cites the `.pyc` mtime + md5 evidence. The connector logs `from=<import dir>` at init, the same hunk as P4's branch. |
| S4 | No action. The exit code is 1 only because of the reference `old` arm (G4.gsm8k_warm +3.4%). Every baseline and v2 verdict passes. |
| S5 | `min_hit_tokens` deleted (policy, default.toml, test row). `loading_wait_ms` kept, with the evidence gap recorded: no SSD/RDMA tier, so no `QueryLoading` and 0 `prefetch wait expired` lines in it2d. Deleting it would turn a Loading answer into an immediate miss once a tier exists. |
| nits | Not changed. The empty sync `LoadPlan` is unreachable today; the `run_gate` items are gate-owned; the `getattr` fallback is kept, being conservative and the same as P4. The teardown test passed on .44 (`PYTHONPATH=. pytest rsi/gates/test_teardown.py`: 1 passed at df01403). |
