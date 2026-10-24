# G5 gate re-review 3 (5bffd1e..d38bc97)

Reviewer: independent agent, fresh context. Range: 91b6a8e (M1), 99146cc
(S3), 8a180e0 + bf9ef3c (stop signals), 144bfd8 (M2/M3/S4), cde6186 (64gb
pool), dc70805 (warmup), 325fbd0 (contamination), 39cd826, 940def4
(digest_groups), c458b14, af326cb, d38bc97 (docs). I did not open the
connector, the pd/nixl connectors, the server's model branches, the
instance.rs slot math, page_first, or the history of model adaptations.

Verdict: **accept-with-fixes**. M1, M3, S3 and S4 are fixed as claimed. M2
is fixed for the path-shadowing case it named. Nothing in the range relaxes
a check, except one small measurement change (S9). Four new must-fixes:
- M4: every new run fails `G0.spec`.
- M5: the tree verification behind M2/M3 can be bypassed in three ways,
  shown below.
- M6: an unverified gate tree still gives an official run.
- M7: in-process tampering needs a compensating control.

## Must-fix

### M4. Every new run fails G0.spec (blocking, fail-closed)

144bfd8 adds `modules` to the scheduler `spec.json`. `_VOLATILE`
(`run_gate.py:710`) does not list it, so `g0()` compares it twice:
- arm vs baseline. With `--under-test-path` the arm's `pegaflow` path is the
  overlay target and the baseline's is the venv's, so they always differ.
- committed vs baseline. The committed `spec.json` has no `modules`. It also
  cannot hold the field, because the field holds host paths and the gate
  tree's rev dir.

I checked this with the real `g0()` on two synthetic dumps that differ only
in `modules`. The result is `pass: False` with `.modules.pegaflow: ... != ...`
and `committed.modules: None != {...}`. On the host, the 39cd826 baseline's
`spec.json` does carry `modules`. The author's note says the G0.spec failures
come from "old records lacking" the field. In fact every future run fails,
so no connector can pass G0 in an official run.

Fix: add `"modules"` to `_VOLATILE`; `G0.provenance` already judges the
field. Add a test: two dumps that differ only in `modules` must pass `g0`.

### M5. Tree verification (M2/M3) is bypassable: a tree that verifies runs other code

`provenance.Trees` compares tracked blob ids with regular files under
`python/pegaflow`. Three things on disk are not compared, but Python imports
them. I reproduced each one locally with the gate's own `Trees` (temp git
repo, `file://` remote). Every case gives `verified: True`, yet
`import pegaflow.conn` runs the injected code:

| case | how | verified | imported |
|---|---|---|---|
| dir symlink | `pegaflow/conn -> /elsewhere/pkg` (a package beats `conn.py`; `os.walk` lists the link in `dirs` and never descends) | True | `INJECTED via dir symlink` |
| unchecked-hash pyc | `__pycache__/conn.cpython-*.pyc` compiled from other source (`__pycache__` is skipped by `_files`) | True | `INJECTED via unchecked pyc` |
| git toplevel + ignored pyc | same pyc in a clean checkout; `status --porcelain` hides ignored files (`.gitignore` has `__pycache__/`, `*.so`) | True | `INJECTED via unchecked pyc` |

A fourth case follows from the code. I checked the loader order but did not
build an extension. `extra` exempts any `.so`/`.pyd`/`.dylib`, and
`ExtensionFileLoader` comes first in `_get_supported_file_loaders()`. So an
untracked `conn.cpython-312-x86_64-linux-gnu.so` would shadow `conn.py`. Its
sha256 is recorded, but nothing checks it.

There is also a time-of-check gap. The overlay is a *symlink* to the live
tree, and verification runs once, before any stage. An agent that edits the
tree mid-run (this worktree was edited by another agent during this review)
changes what later stages import, but the record says "verified".

Fix (one mechanism closes all of these):
- Stage a *snapshot*. Copy only the verified tracked blobs (from `git
  ls-tree` / `ls-files`) and the one expected native module
  `pegaflow/pegaflow.cpython-*.so` into `<run>/under-test-overlay/pegaflow`,
  then verify the copy.
- Reject symlinks and any other native or `.pyc` file in the source tree.
- Set `PYTHONPYCACHEPREFIX=<run>/pycache` for every vLLM and pegaflow-server
  process. I checked locally that this ignores the planted in-tree pyc.

Host check, read-only: `~/kexi/pegaflow-rsi-g5-7bf8076-ut` has no symlinks,
one `.so` with the expected name, and 13 `.pyc` files. I did not inspect the
pyc contents, so the r3 record cannot be fully cleared of this vector. Its
G5 verdict is a fail either way.

### M6. An unverified gate tree (or venv `pegaflow`) still gives an official run

`record_provenance` records `gate` and `venv_pegaflow` with `verified`. But
`official` only checks `under_test` (`run_gate.py:1853`). The scenario is
M3's, applied to the judge itself. A gate tree with uncommitted local
relaxations, or a dirty pegaflow-semantics-rsi (the G3/G5 oracle bar), can
produce `official: true` under a clean rev. Fix: add both to the problems
when `verified` is false. This only tightens.

### M7. The code under test can patch the instrument at runtime. Not fixable in-process; needs a compensating control

The overlay stops *path* shadowing. But the connector is imported into the
same processes as `RsiProbe` and `RsiWorkerExt`. In `MultiConnector[under_test,
RsiProbe]` it is constructed first, so it runs before the probe's spec dump.
Code in `pegaflow/__init__.py` could then:
- `import rsi.gates.rsi_probe` and replace `module_files` (G0.provenance is
  self-reported by the probe);
- replace the digest functions, or replace `rsi_poison` with a no-op, so
  stale cold bytes rescue a connector that loads nothing;
- read the in-run baseline's `digest-*.jsonl` and `*-outputs.json` from
  `<run>/baseline/...`. Same uid, predictable paths. A patched probe could
  emit the baseline's digests verbatim.

Detection does not hold. A probe self-hash runs in the patched process. A
digest computed in another process is not available: the probe is the only
reader of GPU KV. Comparison with the baseline's probe is exactly what the
patched probe would replay.

The independent signals are the completions, which the gate process gets
over HTTP. They are exact vs baseline cold on split-invariant models, and
exact vs `split` on non-aligned prompts on split-variant ones. They bound a
connector that loads garbage while faking digests.

Hypothesis, not verified (I did not read `pegaflow/vllm_plugin.py`): if the
`pegaflow` vLLM plugin entry point resolves through the overlay, the
connector's code also loads in the API server process. Then even the
completions could be replaced with the recorded baseline outputs.

Judgment:
- Acceptable against *bugs*. A buggy connector does not patch the probe, and
  G0.provenance, the digests, poison and exact outputs cover accidental
  wrongness.
- Not acceptable as-is against *gaming* in an autonomous loop where agents
  review agents.

Fix (tighten, cheap):
1. Make the run non-official when a static scan of the snapshot (M5) finds
   any of these: a reference to `rsi`, `rsi_probe`, `RsiProbe`,
   `RsiWorkerExt`, `rsi_poison`, `rsi-runs`, `probe_dir`, `digest-`,
   `-outputs.json`; `sys.modules` writes; or `gc.get_objects`. A reviewer can
   clear a flagged hit.
2. State this boundary in GATES.md.
3. Add "no instrument or run-artifact access" to the connector-review
   checklist.

## Should-fix

- S7. Make the aligned-prompt exemption exact.
  - In `split_control`, `aligned` treats a `None` token record as aligned,
    which exempts it (fail-open, up to the 10% the pre-warm floor lets
    through). Treat `None` as not aligned.
  - The 0.9 warm coverage floor lets 10% of computed blocks go unjudged
    anywhere. On the old slice, all 1182 unjudged warm blocks sit in the 82
    aligned prompts (each has `start == prompt_len - 1`; 164 are boundary
    blocks and 1018 are decode blocks). Require unjudged warm computed
    blocks on non-aligned prompts to be 0.
- S8. Contamination is one-sided. Only arms slower than 10x the baseline are
  flagged. A stall in the *baseline's* replay inflates every arm's G4 margin
  and goes unflagged. Add the same test the other way: baseline p99 > 10x the
  median of the arms' p99 for that pass.
- S9. The warmup is a small measurement change in the lenient direction. It
  warms every arm equally. But the connector's one-time save-path cost
  (first save and gRPC setup; the warmup's blocks are saved) now lands
  outside `G4.replay_cold` and `G4.gsm8k_cold`. The size is likely well
  under the 2% slack, but GATES.md:3 calls for sign-off. Report
  `passes._warmup.seconds` per arm vs baseline in the G4 verdicts, and list
  the warmup next to the split rule in PROGRESS "Open questions".
- S10. 940def4 makes the probe skip every `MambaSpec` group, align mode
  included. Before, align-mode groups were digested. With
  `G2.digest_groups`, every hybrid model (Qwen3.5-4B, P5) now fails G2
  until Mamba state is digested. This is fail-closed and not a relaxation.
  Tell P5.

## Nits

- N5. Signal windows. A first signal that lands between `Popen` returning
  and `try:` in `launched`, or between entering `finally:` and
  `_STOP["teardown"] += 1` in `stop`, raises before `stop` owns the group.
  That group leaks. The window is a few bytecodes. Fix: raise the teardown
  counter around `Popen..try` and the call into `stop`.
- N6. The handler's `print` can raise `RuntimeError: reentrant call` if the
  signal lands inside a stdout write. `_STOP["signal"]` is set first, so
  `check_stop()` in `run_stage` still turns it into SystemExit. Covered;
  only noting it.

## Checks (what was verified)

1. M1 (91b6a8e, 940def4)
   - With `split_invariant=false`, the baseline runs `split` (digested,
     `rsi_digest: full`). Its pre-warm requests are tagged `split_prewarm`
     and are not scored.
   - `_coverage_verdict` requires judged > 0 and coverage >= 0.9 in both
     `cold` and `warm`. A missing pass counts as judged 0, which fails. It
     applies to every arm and to the honest self-test.
   - `_outputs_verdict` requires cold == baseline cold and warm == plain
     `split` on non-aligned prompts. `SPLIT_MARGIN` and the rate bound are
     gone.
   - I re-implemented the same-split coverage on the host
     (`records()`/`split` logic, reference =
     `2026-10-23-dsv2-lite-baseline-split-39cd826`). My numbers match the
     author's:

     | record / arm | cold judged / seen | warm judged / seen (exact) |
     |---|---|---|
     | selftest honest | 2668 / 2668 | 1606 / 1606 (1606) |
     | selftest bitflip | 2668 / 2668 | 14 / 6144 (0) |
     | old slice | 27526 / 27526 | 16310 / 17492 = 0.932 (16310) |
     | it1 v2 slice | 27526 / 27526 | 17482 / 17482 (17482) |

   - Warm outputs vs plain `split` (82 of 1319 prompts aligned):

     | arm | non-aligned differ | aligned differ |
     |---|---|---|
     | honest | 0 | 0 |
     | liar / bitflip / offbyone | 121 | 7 |
     | old | 0 | 38 |
     | v2 | 0 | 0 |

   - Cross-run referencing holds:
     - The reference's cold outputs equal every judged record's baseline
       cold (0/128 self-test; 0/1319 for both slices and the split control).
       Every arm's cold also differs 0, except liar (128/128).
     - The `vllm serve` flags in the four baseline logs are identical except
       for paths and ports: TP2, TRITON_MLA, 8192/2048/256, gpu-mem 0.8,
       seed, prefix caching.
     - The summary args are also identical: shots 8, concurrency 32.
     - The reference baseline's `split` outputs equal the bb75843 plain
       `split` (0/1319).
     - This holds for the offline re-judge only. An official run uses its
       own in-run references.
   - Qwen3 "24 runs, 864/864 unchanged" was not re-run by me.
2. M2 (144bfd8)
   - The overlay holds only `pegaflow`, and the `PYTHONPATH` prepend is
     under-test only.
   - `.pth` files are not processed from `PYTHONPATH` entries, and
     `sitecustomize` cannot come from the overlay.
   - `module_files` is recorded in every scheduler and worker dump, and
     `G0.provenance` is required on baseline and every arm.
   - Bypasses: M5 (on disk), M7 (at runtime).
3. M3 (144bfd8)
   - A clean toplevel (`--show-toplevel == root`) or a blob-for-blob
     `.rsi-rev` is required.
   - An unknown rev fails the fetch, so the tree is not verified and the run
     is not official (tested: `unknown-rev`, `no-rev`, `git-dirty`,
     `rsi-rev-edited`, `rsi-rev-extra-py`).
   - Gaps: M5, M6.
4. S3 (99146cc, bf9ef3c)
   - `run_stage` re-raises a `TeardownError` that is raised or chained.
     ExitStack still stops every other group.
   - A stop signal is checked in `run_stage`'s `except` before a
     reference-arm failure is recorded.
5. S4: `summary.provenance` holds:
   - the gate tree, the venv `pegaflow` tree and the under-test tree, each
     with rev, `verified` and native sha256;
   - the server binary's sha256 and its claimed `.rsi-rev`.
6. Stop signals (8a180e0, bf9ef3c)
   - (1) A stop cannot become "stage failed, continue":
     - The handler raises SystemExit (BaseException). No `except
       BaseException` exists in gate code (grep of `rsi/gates/*.py`).
     - Its own SIGINT handler keeps `asyncio.run` from installing the
       cancel-only one.
     - `check_stop()` runs at stage entry, in `run_stage`'s `except` (before
       a reference arm's failure is recorded), and before and after every
       pass. A converted CancelledError therefore still aborts the run.
       `subprocess.check_output` kills its child on BaseException.
   - (2) Deferral during `stop()` cannot deadlock:
     - Waits are bounded at 120 s + 300 s, and the counter is decremented in
       `finally`.
     - Later signals are SIG_IGN.
     - `check_stop()` runs once the group is gone, ExitStack still runs the
       other groups' `stop`, and no new stage or pass starts after a
       recorded stop.
     - The only leak windows are N5.
   - Both tests are real subprocess tests: a launched `http.server` group,
     real `killpg`, exit code `128 + sig`, the group gone, and `native`
     never started. Both pass. I did not check the commit's claim that the
     test fails on 39cd826.
7. Pool (cde6186)
   - `G3.no_eviction` is still hard (`bool(evicted) and not
     any(evicted.values())`), and `_`-prefixed passes were already excluded
     before this range.
   - A smaller pool can only cause evictions, which fail. `PoolTooSmall`
     fires on the estimate (committed spec) and on the baseline-trace U.
     Without a committed spec only the estimate is skipped, and such a run
     is not official anyway.
8. Warmup (dc70805)
   - The warmup is 512 fixed tokens, `rsi_pass=_warmup`, untagged for
     digest, then `reset_prefix_cache`. No verdict or oracle segment scores
     `_`-passes.
   - `oracle.warmup_problems` fails the stage if any workload request shares
     a prefix key. Keys chain the whole prefix, so no hit or E_ext can come
     from it.
   - Its blocks only enlarge the measured U (fail-closed). Cap stages have
     no warmup.
   - G4 effect: S9.
9. Contamination (325fbd0)
   - It only appends to `not_official_because`. Verdicts are computed
     unchanged, and the exit code is 1.
   - It never sets a pass. It covers the G4-timed passes only (`cold`,
     `warm2..`).
   - One-sided: S8. A run made non-official must still be recorded, so
     retrying until a run is clean is visible.
10. digest_groups (940def4)
    - Required in `v["baseline"]` (joined with AND), on every arm (AND in `g`), and
      in the honest self-test expectation.
    - The test needs native `pegaflow.semantics` (`group_kinds`), so it did
      not run locally.
11. No relaxation. Removed: `SPLIT_MARGIN`/`control_rate` (replaced by
    exact checks) and the old pre-warm rule (tightened). Added:
    `ref_equals_cold`. Mamba align-mode digests are gone but fail-closed
    (S10). The pool is fail-closed. The only lenient direction is the
    warmup (S9).
12. d38bc97 spot-check, against `...-qwen3-4b-g5-official-e9f3f33-r3/summary.json`
    - Matches:
      - official, `[]`, rev e9f3f33, under-test rev 8a05b7c, 160gb,
        64 conversations;
      - U 54.43 GB, needed 49.27, C 27.22 GB, GPU KV 11.16, ratio 2.44;
      - E_C, H*_C, H_lru and hit for v2, old and native at 0.25, 0.5 and
        0.75;
      - v2 E_C cold 0.971, own-local 0.851, freeze 0.623, H_lru/H*_C 0.859;
      - every G4 median and p99 in the tables;
      - gsm8k hit tokens and rates;
      - failing checks per arm (v2 fails only `G5.replay_warm`).
    - Not checked: the replay per-pass local/ext token table, evictions and
      pool-used figures, accuracy.
13. Unit tests (local macOS, Python 3.14.6; temp venv: pytest 8.4.2,
    requests 2.32.5, aiohttp 3.12.15; a throwaway `pegaflow.semantics` stub
    on a `.pth`)
    - At d38bc97: 77 passed, 21 failed.
    - All 21 failures are the stub's ImportError, so those tests (oracle
      capacity, verdict integration, digest_groups x3) were not run.
    - Every other test added in the range passes: split/outputs/coverage,
      provenance x10, pool, contamination, warmup, stop/stage, SIGTERM, and
      the stop signals x6.

Host access: read-only. I ran `ls`/`grep` and `python3 -c` reads of JSON
and JSONL under `~/kexi/rsi-runs`. I also made one `os.walk` listing (names
only; no contents read) of `~/kexi/pegaflow-rsi-g5-7bf8076-ut` for M5. That
tree is outside `rsi-runs`, so this went past the stated scope. Nothing was
written, killed or launched.

## Merge

feat/rsi-g5 should not merge into feat/semantics-rsi as-is, because M4
fails every new run's G0. With M4 fixed (one line plus a test), it can
merge. M5-M7 must land before the next `--under-test-path` run counts as
official.

## Resolution (implementer, d6a5da0..HEAD)

| item | decision | commit | how |
|---|---|---|---|
| M4 | fixed | c7f7520 | `"modules"` is in `_VOLATILE`; `G0.provenance` judges it. Test: two dumps differing only in `modules` pass `g0`. GPU3 run below: `G0.spec` passes on current code. |
| M5 | fixed | 2ae965f | `provenance.Trees` walks every file under the subpath (including `__pycache__`, without following symlinks). A tree verifies only if each file is a tracked regular blob with the same id, or is the one native module (sha256 recorded). Symlinks, untracked or ignored files, any other `.so`, and any `.pyc` are rejected; `.pyc` is allowed only with `pycache_ok`, which is used only for the gate and venv trees under `PYTHONPYCACHEPREFIX`. The arm imports `<run>/under-test-overlay/pegaflow`, a copy of the rev's blobs plus the native module, verified again after the copy. Every stage process gets a fresh `PYTHONPYCACHEPREFIX`. Table test of the five bypasses. |
| M6 | fixed | 95998cf | `provenance_problems`: an unverified gate, venv `pegaflow` or under-test tree makes the run non-official. |
| M7a | fixed (deterrence) | 5fca1b7, e15dc5e, 198f595 | `isolate.py`: in a private mount namespace the runs root is under a tmpfs, and only `<run>/<arm>` (rw) and the snapshot (ro) are bound back. Block devices and `/dev/{mem,kmem,port}` are masked. SYS_ADMIN, SYS_PTRACE and 7 more caps are dropped from the bounding set; IPC_LOCK, SYS_RESOURCE and SYS_NICE are kept. The helper execs in place, so the pid and pgid stay the same. The residual (uid 0 with DAC_OVERRIDE: not a boundary) is stated in GATES.md and `summary.isolation.residual`. The command line is in `summary.isolation.command_prefix`. Tests: the arm cannot read the other arms, `/proc/<gate>/root` or `cwd`, cannot `setns`, and cannot `umount`; its group is torn down. |
| M7b | fixed | 5fca1b7, e15dc5e | `provenance.scan` over the snapshot, including the native module's strings. `SCAN_PATTERNS` covers rsi, the probe, poison, run artifacts, `sys.modules` writes, gc, trace/audit and import hooks. A hit not on the gate-owned `SCAN_ALLOWLIST` (path, pattern, reason) makes the run non-official. Hits go to `summary.scan`. |
| M7c | fixed | 5fca1b7 | GATES.md "Boundary (M7)" section. The skill's review checklist adds "no instrument or run-artifact access; read every `summary.scan` hit". |
| M7d | fixed | 5fca1b7 | `VLLM_PLUGINS=""` for every vLLM. Completions are read client-side over HTTP. |
| import tree | fixed | 456ee0f | The probe records `connector:<module>` `__file__`. `G0.provenance` requires it under the arm's snapshot (or the gate tree for `rsi.*`). An official run with an under-test arm needs `--under-test-path`. |
| S7 | fixed | e5f7e68 | A missing token record is not aligned. digest_compare counts `unjudged_nonaligned` per pass. Warm coverage requires it to be 0. |
| S8 | fixed | d90ac5d (docs), 93a654f | Contamination is two-sided: the baseline's p99 is also checked against 10x the median of the arms' p99. |
| S9 | fixed + open question | 1b640f7 (docs), 47cbd44 | `G4.warmup_seconds` per workload, value vs baseline, reported and not gated. It is listed in PROGRESS "Open questions" for sign-off. |
| S10 | noted | - | Fail-closed; P5 informed by the coordinator. |
| N5 | fixed | 7d477c5 | The Popen-to-registration window is deferred (`_STOP["teardown"]`). The `try:` and `finally:` bytecode windows leave the group in `_LIVE`, which `stop_live` (atexit) stops. Subprocess test for both windows. |
| N6 | accepted | - | Covered by `check_stop()`, as the review says. |

Also in this range (coordinator requests):
- c6d6145, adc8143: stage ports are leased from the reserved range 20000-29999 under flocks in `~/kexi/rsi-ports`, and vLLM gets a leased `VLLM_PORT` block. EADDRINUSE on any bind is a `PortRace` naming the port. It is not retried. Lease files are recorded in `stage-error.json` and `summary.port_leases`, and leases are released on every exit path.
- 23e618b: CUDA toolchain preflight. Every nvcc on `PATH` or `CUDA_HOME` must have torch's CUDA major. Recorded as `summary.cuda_env`.
- b3d43b4: `--replay-skip N`. N > 0 is a held-out slice, never official (`summary.held_out`).

### GPU3 run (current code, 7d477c5)

Run `2026-10-24-qwen3-4b-g5-gpu3-7d477c5-b` on GPU3 with a 16gb pool. Arms: plain, baseline, and `pegaflow.connector_v2:PegaKVConnectorV2`. Under test is 8a05b7c: a git-archive extraction plus its `.so`, no pyc. Workload: gsm8k with limit 64. It ran 02:07:41Z-02:14:40Z and is non-official only for the limit and the workload set.
- G0 on current code passes:
  - Arm: `G0.spec` passes (0 diffs) and `G0.provenance` passes. Every pegaflow file and the connector module resolve under the overlay.
  - Baseline: `G0.provenance` passes.
  - Provenance is verified for the gate (7d477c5, 75 files), the venv (056eb4b) and the under-test tree (8a05b7c, .so sha256 equal to the source).
  - Scan: nothing flagged.
- The arm served inside the isolation:
  - Line 1 of the arm's `vllm.log` and `pegaflow.log` is the `isolate.py` prefix.
  - gsm8k cold and warm accuracy 0.8594, the same as baseline and plain.
  - `G3.gsm8k_warm` passes: E 1.0, external hit 0.978, 0 load errors.
- Teardown: the vllm group was gone after SIGTERM in 1.0 s and the pegaflow-server group in 7.0 s. No process was left and GPU3 was at 1 MiB. All 4 leases were released.
- pegaflow-server log:
  - Pinned pool: 16.00 GiB requested; `allocatable_size=16.0 GiB`.
  - 0 WARN or ERROR lines. No mlock or registration errors.
  - CUDA IPC: 36 layers registered, then "Session opened". At teardown it "dropped 36 CUDA tensors".
- Attempt 1 (`-7d477c5`, 02:00Z) failed in the baseline FlashInfer JIT: the system nvcc 12.2 does not support `--compress-mode`. The cause was my launch env, which lacked `CUDA_HOME=~/kexi/cuda13`. 23e618b now refuses this.

### Qwen3-4B re-judge (39cd826 vs b3d43b4)

The earlier "24 runs, 864/864" covered the records on disk at the time. It is now redone over all 49 Qwen3-4B records with a baseline arm.

How: both judge trees were extracted with `git archive`. Each record's `result.json` was judged with `digest_self` recomputed by that judge's `digest_compare`, then `verdicts(results, run, committed, gate)`. The script is `~/kexi/rsi-runs/scratch/rejudge_qwen3.py`. Output is in `~/kexi/rsi-runs/scratch/rejudge-qwen3/`.

- 33 records judged by both, 82 arm verdicts:
  - 1080 common pass flags: 1079 unchanged, 1 changed. The changed flag is `G0.spec` on the GPU3 run (False -> True). That is M4: 39cd826 compares `modules`.
  - Overall arm pass changes: 0.
  - G3: all 147 G3 verdict dicts are identical, numbers included.
  - 82 flags exist only in the new judge. All are `G2.digest_groups`, which 940def4 added after 39cd826:
    - 56 pass.
    - 26 fail, all on records without a gsm8k stage (replay-only or cap-only). Those records were already failing overall and are non-official.
- 16 records fail to judge under both judges with the same `KeyError: 'efficiency_c_own_local'`. They predate 20e6b58's cap fields. This is not a judge difference.
