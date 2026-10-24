# G5 gate re-review 4

Scope: `d6a5da0..8d529fd -- rsi/gates`, plus the GATES.md diff (HEAD
8d529fd). I reviewed the code diff, not the commit messages. Line numbers
are at 8d529fd. Host evidence was read-only, under `~/kexi/`.

Verdict: **merge after must-fix** (M8, M9, M10).

## Item table

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | M4-M7, S7-S9, N5 closure | closed | All code is in the HEAD tree, including the S8/S9 code lost in d90ac5d / 1b640f7. The M7 compensating controls are overclaimed (M9, M10). |
| 2 | Official-run integrity | open | An `rsi.*` connector for the arm under test escapes the snapshot check (M8). Provenance is checked only before stage 1 (M10). |
| 3 | Gate rules not relaxed | pass | Only tightenings. `modules` moved from G0.spec to G0.provenance. |
| 4 | Isolation residual | open | The residual is stated honestly, but two of the named controls do not hold (M9, M10). The allowlist is empty. The scan misses common patching routes (S12). |
| 5 | Teardown / ports | pass, with S11 | Leases are released on every path. N5 tests pass. PortRace cannot change an outcome. `VLLM_PORT` is behaviour-neutral. A root arm can leave its process group (S11). |
| 6 | Evidence | pass | GPU3: G0 pass, isolation prefix recorded, 4/4 leases released, clean teardown. Re-judge: only M4's G0.spec flipped; no arm `pass` changed. The KeyError crash should fail closed (S13). |
| 7 | Tests | pass, with N10 | 112 passed, 2 skipped (need Linux and root), 21 failed only on a stubbed `pegaflow.semantics`. The 9 signal/window tests pass (302 s). Nothing in the repo deselects them. |

## 1. Closure of re-review 3 items

| item | status | evidence |
|---|---|---|
| M4 | closed | `run_gate.py:1027`: `_VOLATILE` includes `modules`. Test `test_g0_spec_ignores_module_paths` (`test_oracle_capacity.py:1416`). The GPU3 run has G0.spec true (item 6). |
| M5 | closed | `provenance.py:116-134` enters every dir and lists symlinks without following them. `check` (`:152-194`) rejects symlinks, untracked files, any other `.so`, and `.pyc` unless `pycache_ok`. `stage` (`:269-315`) verifies the source with `pycache_ok=False` (`:282`), copies only the rev's blobs plus the native module, re-verifies the copy, and compares the native sha256 (`:311`). Stage processes get `PYTHONPYCACHEPREFIX` (`run_gate.py:446`, `:520`, `:661`). The gate re-execs with it (`:2312-2321`). The native module is still not tied to a rev (N7). |
| M6 | closed | `provenance_problems` (`run_gate.py:2162-2176`) covers the gate and venv trees, and a missing record counts. It is wired at `:2423`. Checked once only (M10). |
| M7a | closed as deterrence | `isolate.py`: mount namespace, tmpfs over the runs root, the arm dir rw, the snapshot ro, caps dropped. `isolated()` wraps both the server and vLLM of the arm under test (`run_gate.py:542`, `:739`). A non-isolated arm makes the run non-official (`:2436`). |
| M7b | closed | `provenance.scan` and an empty `SCAN_ALLOWLIST` (`provenance.py:324-394`). Flagged hits make the run non-official (`run_gate.py:2426`). Pattern coverage: S12. |
| M7c | closed, overclaimed | GATES.md "Boundary (M7)". Skill checklist: `.claude/skills/pegaflow-new-model/SKILL.md:87-89`. See M9, M10. |
| M7d | closed | `VLLM_PLUGINS=""` for every vLLM (`run_gate.py:671`). |
| S7 | closed | A missing token record is not aligned (`run_gate.py:1201`). `compare` counts `unjudged_nonaligned`, and a missing `prompt_len` counts (`digest_compare.py:203-205`). Warm requires 0 (`run_gate.py:1314`). `_hit_alignment` defaults to 16 (`:1163-1171`), so the count is always on. |
| S8 | closed | Baseline p99 is checked against the arms' median (`run_gate.py:1956-1961`). |
| S9 | closed | `G4.warmup_seconds` is reported per workload with no `pass` key, so it is not gated (`run_gate.py:1575-1582`). PROGRESS open question: `docs/rsi/PROGRESS.md:584`. |
| N5 | closed | The Popen-to-`_LIVE` window is deferred (`run_gate.py:342-353`) and honored by `check_stop()` inside the `try`. The remaining windows leave the group in `_LIVE`, which `stop_live` stops (`:300-306`, atexit at `:2090`). `stop` tolerates `fh=None` (AttributeError is suppressed). Both window tests pass locally. |

## 2. Official-run integrity

| condition | can `official: true` happen? | evidence |
|---|---|---|
| under-test tree unverified | no | `stage` returns `verified: False` on any source or copy mismatch. A missing package raises, so no summary is written. `provenance_problems` makes the run non-official. |
| gate / venv trees unverified | no at start; yes if they change mid-run | Checked once, before stage 1 (`run_gate.py:2333`), and never re-checked (M10). |
| connector `__file__` outside the snapshot | **yes, for an `rsi.*` module** | M8 |
| CUDA env mismatch | no, for the gate's env | `cuda_problems` (`:2043-2060`) is wired at `:2424`. An unknown torch CUDA version counts as a problem. The stage PATH differs from the PATH that is checked (N8). |
| `replay_skip > 0` | no | `OFFICIAL["replay_skip"] = 0` (`:72`); `official_problems` checks equality. |
| a PortRace | no for a required arm | A required arm re-raises (`:2210`), so the run aborts with no summary. A reference arm's PortRace is recorded like any other reference-arm error, by design (reference arms are not required). |

No swallowed exception turns a failed check into a pass:
- `Trees._fetched` returns a reason string, which means unverified.
- `listing` returns `None`, which also means unverified.
- `record_provenance` has no `except`.
- `cuda_env` suppresses a failed torch subprocess. That leaves `torch_cuda=None`, which is itself a problem.

Every check applies to the under-test arm, or to the whole run. The one exception is the connector-root choice in M8.

## 3. Gate rules not relaxed

- `OFFICIAL` at d6a5da0 vs 8d529fd: one key added, `replay_skip: 0`. No value changed.
- No `MIN_*`, `MAX_*`, `CONTAMINATION_P99` or `CAPACITY_*` constant changed in the diff.
- GATES.md: every removed line that holds a number comes back reworded with the same number (`> 10x`, coverage `0.9`). The added rules only tighten:
  - contamination is two-sided;
  - zero unjudged warm blocks on non-aligned prompts;
  - gate and venv tree provenance;
  - the connector's `__file__`;
  - `--under-test-path` is required;
  - isolation is required;
  - CUDA preflight;
  - the scan.
- `_VOLATILE += modules` drops a G0.spec comparison. G0.provenance judges the same paths: vLLM, torch and the prefix under the venv; the probe under the gate tree; `pegaflow` under the expected dir. The check moved; it was not relaxed.

## 4. Isolation residual

- GATES.md "Boundary" says "deterrence, not a boundary". It names the uid 0 + `CAP_DAC_OVERRIDE` residual (host files outside the runs root are writable). That part is accurate.
- It then names "the static scan, provenance and the review" as the controls, and calls completions an independent signal. Two of these do not hold as implemented (M9, M10).
- `SCAN_ALLOWLIST` is empty (`provenance.py:381`), so nothing gets through today. An entry matches on path glob + pattern name only, not on the hit's text (N11).
- On the `rsi` regex (`provenance.py:325`): it is adequate for literal references (`import rsi`, `"rsi.`, `rsi/gates`). Any built string (`"r" "si"`, `"rs" + "i"`) evades it, which is expected for deterrence. The material gap is the patching patterns, not the `rsi` regex (S12).

## 5. Teardown / ports

- Leases are released on every path:
  - `launched` releases in its outer `finally` (`run_gate.py:367-369`);
  - `run_stage` releases in its `finally` (`:2214-2215`), which covers a lease made before `launched` was entered;
  - `PortLease.__init__` releases on a partial failure (`:218-220`);
  - the summary records them (`:2457`).
  - Lock fds are `O_CLOEXEC`, so no child inherits them.
  - GPU3: 4 leases, all released.
- N5 windows: closed (item 1). `test_stop_signal_in_a_launch_window_still_stops_the_group` passes for `popen` and `finally`.
- PortRace cannot mask an outcome:
  - A required arm aborts the run whatever the exception class is. A reference arm records an error either way.
  - `_teardown_failed` still finds a TeardownError raised in `stop` while a PortRace is in flight, because it walks the context chain.
  - Only the *diagnosis* can be wrong (N9).
- `VLLM_PORT`: in the host venv's vLLM, it is read only by `get_open_port` / `get_open_ports_list` (`vllm/utils/network_utils.py:192-237`).
  - It is listed next to `VLLM_CACHE_ROOT` in `envs.py:2363`, in what reads as the compile-hash ignore list. I did not read the whole list.
  - On CUDA with TP=1, the uniproc executor uses a file store, not TCP (`v1/executor/uniproc_executor.py:78-82`). The DP port list is only used with DP.
  - So it changes no model behaviour.
  - Caveat: `_get_open_port` scans up to 1000 ports past `VLLM_PORT` without the lock protocol. If the 8-port block is taken, vLLM can walk into another run's leased ports (N9).

## 6. Evidence

GPU3 run, `~/kexi/rsi-runs/2026-10-24-qwen3-4b-g5-gpu3-7d477c5-b/summary.json`:
- PegaKVConnectorV2: `G0.spec` true and `G0.provenance` true. Baseline: `G0.provenance` true.
- Provenance verified for `gate`, `venv_pegaflow` and `under_test`.
- `isolation.available` true. The prefix is `python -I -S -B .../isolate.py --hide /root/kexi/rsi-runs --rw <run>/PegaKVConnectorV2 --ro <run>/under-test-overlay --`, and the residual text is recorded.
- `scan.flagged` and `scan.allowlisted` are both empty.
- 4 port leases, all with `released_at`.
- Teardown: 4 groups gone after SIGTERM in 1-7 s.
- Non-official only for `gsm8k_limit=64` and gsm8k-only workloads, as intended. The failing replay/cap verdicts are "not evaluated" for that reason.
- No `cuda_env`, `held_out` or `replay_skip` in the record: the run is at 7d477c5, before 23e618b and b3d43b4. No host run evidences the CUDA preflight or `--replay-skip` yet.

Re-judge, `~/kexi/rsi-runs/scratch/rejudge-qwen3/`: 49 runs per judge (`per/` and `g3/`). I diffed every verdict key, 39cd826 vs b3d43b4:

| key | change | count |
|---|---|---|
| `G0.spec` | False -> True (M4) | 1 |
| `G2.digest_groups` | absent -> True | 56 |
| `G2.digest_groups` | absent -> False | 26 |
| any arm `pass` | changed | 0 |

- `G2.digest_groups` is new in b3d43b4. Its 26 False values land only on arms that already failed, so no accepted verdict changed.
- 16 records crash identically under both judges with `KeyError: 'efficiency_c_own_local'` (`run_gate.py:1489`). These are records written before 20e6b58 added the key to `capacity.warm_timed`. Fail-closed in effect (no pass), but see S13.

## 7. Tests

- The given command does not collect on a clean venv. `run_gate` imports `requests` (not installed by it), and `oracle` imports `pegaflow.semantics`, which needs the built native module (`python/pegaflow/semantics.py:9`). The module docstring says so (`test_oracle_capacity.py:1-4`), but the command does not.
- A pre-existing `/tmp/rsi-venv` (from an earlier session) had a `.pth` pointing at a stub `/tmp/rsi-stub/pegaflow/semantics.py`. I recreated the venv from scratch and added `requests`.
- With the stub on `PYTHONPATH`: 112 passed, 2 skipped (`isolate.py needs Linux and root`), 30 failed.
  - 21 failures are oracle tests that need real semantics (stub ImportError).
  - 9 are the signal/window subprocess tests. They replace `PYTHONPATH` (`env = os.environ | {"PYTHONPATH": str(root)}`, `test_oracle_capacity.py:1388` and `:770`), so the child cannot see a `PYTHONPATH`-provided `pegaflow`.
- With the stub on a `.pth`, those 9 pass (`test_stop_signal_aborts_the_run` x6, `..._launch_window_...` x2, `test_sigterm_tears_down_launched_groups`) in 302 s.
- The 8 "deselected" tests: no conftest, pytest.ini, marker or doc in the repo deselects them. The documented command, `python -m pytest rsi/gates/test_oracle_capacity.py` (GATES.md:193), runs them. The deselection must have come from an ad-hoc `-k`. See N10.
- The isolation tests (Linux + root) were not run here.

## Must-fix

### M8. The arm under test can name an `rsi.*` module and skip the snapshot check

`g0_provenance` picks the connector's root by module name, not by arm kind (`run_gate.py:1128`):

```python
root = REPO.resolve() if connector.startswith("rsi.") else pegaflow_dir
```

`arm_kind` returns `under_test` for any `<module>:<Class>` that does not start with `selftest:` (`:581-584`).

Scenario:
1. An agent commits a connector under `rsi/` in the gate tree. For a git toplevel, a local commit verifies. For a `.rsi-rev` tree, it pushes the rev.
2. It runs `--arms plain,baseline,rsi.gates.myconn:MyConn --under-test-path <clean tree>`.
3. The snapshot is clean and scanned. The connector that actually runs lives in the gate tree and is never scanned: `scan_record` scans only the snapshot (`:2141-2159`).
4. G0.provenance accepts the connector because its file is under `REPO`.

The run is official, but the connector measured is not the one in the snapshot.

Fix:
- Choose the root by kind. Only `selftest` maps to the gate tree. For `under_test`, the connector must lie in the snapshot.
- Make `official_problems` reject an under-test module outside the `pegaflow` package.
- Add an `rsi.x:Y` under-test case to `test_connector_module_must_come_from_the_snapshot` (`test_oracle_capacity.py:1191`).

### M9. The "independent signal" is stored where the arm can rewrite it

The gate writes `<name>-outputs.json` and `<name>-tokens.json` into `run_dir/<label>/<wl>` (`run_gate.py:939-941`). `verdicts` reads them back from disk at the end (`_outputs` / `_tokens`, `:1147-1160`). That dir is inside `<run>/<arm>`, which `isolation()` binds **rw** for every later stage of the arm (`:485-496`).

Scenario:
1. The connector computes the `cold` pass honestly.
2. In `warm`, it claims hits and loads garbage, which is fast.
3. From the same process, or from its replay/cap stage processes, it copies `cold-outputs.json` over `warm-outputs.json`. Under batch invariance, warm must equal cold. It also rewrites its own `digest-*` files.
4. G2.gsm8k_outputs and G3 pass.

The scan patterns `-outputs\.json` and `digest-` are evaded by `"-outputs" + ".json"`. GATES.md "Independent signal" ("Completions are read by the gate *client-side* ... and judged in the gate process") is therefore not true of what is judged.

Fix (cheap, either one):
- Write gate-produced artifacts (outputs, tokens, `result.json`) to a dir the arm never gets, e.g. `<run>/.gate/<arm>/<wl>`.
- Or keep their sha256 in gate memory and verify it before `verdicts` reads them.

Also bind rw only the current stage's `probe/` and `.pycache` dirs, not the whole `<run>/<arm>`. Earlier stages' probe records would then be read-only for later stages.

### M10. Provenance is named as a control for mid-run writes, but it runs once

GATES.md "Boundary" says the arm can write "code that later stages ... run (gate tree, venv, system files)", and that "the controls for that are the static scan, provenance and the review". Provenance runs once, before stage 1 (`run_gate.py:2333`), and is never re-checked.

Scenario:
1. The arm's gsm8k vLLM (uid 0, `CAP_DAC_OVERRIDE`) rewrites `<gate tree>/rsi/gates/rsi_probe.py`. The gate tree is not under the hidden runs root.
2. Its replay and cap stages load the patched probe, which fakes `requests.jsonl` and the digests that G2.digest_self.cap and G5 read.
3. The patched module can restore the original file in a SIGTERM/atexit handler during teardown.
4. The record says "gate tree verified".

A variant needs no write outside the runs root. `PYTHONPYCACHEPREFIX` for
the arm's stages is `<run>/<arm>/.pycache` (`run_gate.py:446-453`), inside the
rw bind, and `fresh_pycache` clears it once per run, not per stage
(`:456-461`). A gsm8k-stage process can plant a timestamp-validated `.pyc`
for `rsi/gates/rsi_probe.py` there, with a header that matches the source's
mtime and size. The replay and cap stages then run it. Tree verification never
looks at this dir, and G0.provenance records the `.py` path.

Fix:
- Give each stage a fresh pycache dir outside the arm's rw bind, or bind it
  per stage and wipe it before the next stage.
- Re-verify the gate and venv trees before each stage, and after the last one. Any change makes the run non-official. The fetch is cached in `Trees`, so this costs only a hash walk. Checking at launch time catches the patch, because the patched file must be present when the next stage imports it.
- Pair this with S11: an escaped process could otherwise patch after the check.
- Or, at minimum, drop "provenance" from that sentence in GATES.md and `ISOLATION_RESIDUAL` (`run_gate.py:466-478`).

## Should-fix

- S11. A root arm can leave its process group.
  - The arm can `setsid()` a child. `stop` / `_group_gone` only wait for the group (`run_gate.py:388-443`).
  - The survivor outlives teardown. It keeps GPU or pinned memory, so the next stage is measured on a busy GPU. It can also undo M10's evidence after a final check.
  - It cannot leave the arm's private mount namespace: `unshare` needs `CAP_SYS_ADMIN`, which is dropped.
  - Fix: record `/proc/<leader>/ns/mnt` at launch. After `stop`, scan `/proc/*/ns/mnt` for that inode. Any match is a `TeardownError`. Alternatively, use a per-stage cgroup.
- S12. The scan misses the usual ways to reach the probe object.
  - Not covered: `inspect.currentframe` / `inspect.stack` / `sys._current_frames` (frames up to `MultiConnector._connectors`), `KVConnectorBase_V1.__subclasses__()` / `object.__subclasses__()` (finds the `RsiProbe` class directly, then `setattr`), `importlib.import_module` with a computed name, and `ctypes`.
  - Add these patterns. They are cheap, and a hit only means a review is needed.
- S13. `_capacity_verdict` indexes required keys directly (`run_gate.py:1489`, and `w["excess_tokens"]`, `w["unmatched_requests"]`, `r["capacity_bytes"]`, `r["load_errors"]`).
  - A record without them crashes `verdicts`. That is fail-closed, but a live run then writes no `summary.json` at all, and a re-judge loses every other verdict of the run (16 records here).
  - Fix: return `{"pass": False, "reason": "capacity record lacks <key> (re-run the cap stage)"}` for any missing key, matching GATES.md "a required verdict that cannot be evaluated is a fail".

## Nit

- N7. The one allowed native module (`pegaflow<EXT_SUFFIX>`) is checked only for equality between source and snapshot (`provenance.py:311`). Nothing ties it to the rev, and it can hold arbitrary code. Its strings are scanned. Say in GATES.md that it is a claim, as the server binary already is.
- N8. `cuda_env()` checks the gate's `PATH` (`run_gate.py:2020-2041`), but vLLM runs with `<venv>/bin` prepended (`:674`). Compute it from the stage env.
- N9. PortRace diagnosis:
  - It triggers on any EADDRINUSE line in the last 50 kB of the log (`:279-284`). A benign, retried bind followed by an unrelated crash (OOM, CUDA error) or a hang (TimeoutError) is reported as a PortRace.
  - `stage-error.json` (`:2206`) keeps only `str(e)`, so the real cause in `__cause__` is lost from the record.
  - Fix: include the first line of the original error in the PortRace message.
  - Also: vLLM's upward scan from `VLLM_PORT` ignores the lock protocol (section 5).
- N10. Tests:
  - Document a runnable local command. It needs `requests` and a built `pegaflow.semantics`, or a `PYTHONPATH` to a built `python/`.
  - The subprocess tests should prepend `root` to `PYTHONPATH` instead of replacing it.
  - Mark the 9 signal tests (about 5 min) so a fast default can deselect them on purpose.
- N11. `allowlisted` matches on path glob + pattern name (`provenance.py:384-394`). An entry for one file and `sys.modules write` covers every future write in that file. Require exact paths, and the hit text or a count, in each entry.

## Verdict

**merge after must-fix**: M8, M9, M10. All three are small changes. M9 and M10 can alternatively be closed by correcting GATES.md "Boundary" to drop the claims, but that leaves G2/G3 outputs and G5 entirely self-reported against gaming.
