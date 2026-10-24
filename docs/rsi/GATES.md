# RSI gates (human-owned)

Agents may add checks or tighten thresholds here; they may not remove or relax
anything (only the human gate owner can, as on 2026-10-24 for hit-rate levels). Changes to this file need the review agent's sign-off and a log line in
`PROGRESS.md`. Status: DRAFT until the user confirms the metric set.

A model is "supported" only when every hard gate passes on that model and no
earlier model regresses.

## Hit rate: reported, judged by the agent (user, 2026-10-24)

User, 2026-10-24: "命中率那块，不要设置硬性门槛，只要agent自己判断命中率足够就行了，不要差太多"
("for hit rate, set no hard threshold; the agent judges whether the hit rate
is enough, it must not be far off"). The human gate owner's decision; it
replaces the hit-rate bars below (G2 "> 0.9 of oracle", G3 `E_ext >= 0.95`
and `oracle_hit_rate >= 0.5`, G5 `E_C >= 0.9`, and the "no `H` / `E_C`
regression" clauses).

- Reported, never gated: `H`, `H*`, `E_ext`, `H*_C`, `H_lru`, `H_freeze`,
  `E_C`, the oracle hit rate and per-group attribution. Every number is
  still computed. Each verdict carries a `reported` block (value, former
  bar as `reference`, `below`). `summary.hit_report` (also printed as the
  `[gate] hit rates (reported, not gated)` line) lists them per arm with
  `status: reported`.
- Still hard, unchanged: load errors, mismatched / unreferenced blocks,
  digest coverage (`MIN_JUDGED_*`, `MIN_COMPUTED_COVERAGE`,
  `MIN_PREWARMED`), `excess` within one block per request, `G3.no_eviction`
  on the unlimited runs, the cap pool and load proofs (`pool_ok`,
  `loads_ok`, `unmatched_requests == 0`), the G5 capacity floor and freeze
  separation (run validity, `not_official_because`), and every G0 / G2 / G4
  check.
- Hit-rate judgment: the onboarding agent writes a "hit-rate judgment" in
  the model's `ITERATIONS.md` for each accepted run. It compares the
  measured hit rate with `H*` / `H*_C` and the reference arms (`H_lru`,
  `old`, `native`), explains every gap (Kind limits, retention, eviction,
  per-group misses), and says whether it is close enough. The review agent
  checks that judgment (numbers against `summary.hit_report`, the
  explanation against the attribution) and can reject the run on it.

## G0 Layout (hard)

- Spec dump of the model under vLLM 0.32 (`kv_cache_config`, per-group spec
  class, block_size, page bytes, layer→group map, TP rank shard info) is
  committed to `docs/rsi/models/<model>/spec.json`.
- Adapter's declared layout (Kind, Shard, unit size, needed set) is checked
  against the dump: byte sizes add up, every layer belongs to exactly one group.

## G1 Semantics vectors (hard)

- `pegaflow-semantics` V1 test vectors pass (`cargo test -p pegaflow-semantics`).
- Every new Kind/Shard ships with vectors for resolve/saveable/needed/seal_plan.

## G2 Correctness (hard)

1. Per-block digest (V3): for every block loaded from PegaFlow, its bytes equal
   the bytes vLLM computes for the same prefix without PegaFlow. Zero mismatches.
2. Greedy output identity: the e2e prompt set
   (`python/tests/test_vllm_e2e_correctness.py`) gives the same tokens as
   baseline vLLM, cold and warm.
3. gsm8k accuracy (8-shot CoT, full test split 1319, greedy, temp 0):
   - runs: baseline (no PegaFlow), cold (PegaFlow saves), warm (GPU prefix cache
     reset via `/reset_prefix_cache` before the pass, so reuse must come from
     PegaFlow)
   - gate: `|warm - baseline| <= 1.0pp` and `|cold - baseline| <= 1.0pp`
   - warm external hit rate vs oracle: reported. A warm pass that loads little
     says little about loaded KV, and the hit-rate judgment has to say so.

## G3 Hit rate (reported, primary RSI objective; see "Hit rate" above)

Definitions, per run, prompt tokens only:

- `local_hit` = tokens served by vLLM's own GPU prefix cache
- `external_hit` = tokens loaded through the connector
- `computed` = prompt tokens actually prefilled
- external hit rate `H = external_hit / (prompt_tokens - local_hit)`
- oracle `H*` = same trace replayed offline through `semantics.resolve` with
  unlimited CPU capacity (upper bound given the model's Kinds; e.g. a
  Checkpoint group can only hit at checkpoint boundaries)
- hit efficiency `E = H / H*`

Also reported, per group: hit tokens, resolve fallback count and tokens lost to
fallback (P shrunk because some group lacked data), lease/prefetch failures,
loaded-but-unused bytes, saved-but-never-loaded bytes (write amplification).

Gate: none on the level since 2026-10-24. `E` (reference 0.95) and `H` against
the model's last accepted iteration go into the hit-rate judgment. Old
connector's `H` on the same trace is recorded as a black-box reference number
only.

## G4 Performance (hard gate, RSI objective)

Workload: multi-turn agent traces (user-provided "agentx" — TBD; fallback
`/ppio/data_sets/R2EGym-SFT-Trajectories` replayed as multi-turn chat with
growing context), plus gsm8k few-shot as a short-prefix workload.

Metrics: TTFT p50/p99, end-to-end throughput (req/s, output tok/s), load and
save bandwidth (GB/s), connector scheduler-side CPU time per step, GPU memory
overhead. Compared against baseline vLLM (no PegaFlow) and the old connector.

Gate (hard): warm replay TTFT p50 and throughput must beat baseline vLLM;
cold replay and gsm8k may be at most 2% worse than baseline (see G4 under
precise definitions); no >5% regression vs. last accepted iteration of the
same model.

## G5 Capacity hit rate (RSI objective; added 2026-10-23; `E_C` reported since 2026-10-24, correctness and validity hard; see G5 under precise definitions)

User: actual hit rate has to approach the theoretical one at a fixed capacity.
G3 runs with an unlimited pool, so it cannot see eviction policy, write
amplification or cross-group coherence. Example: a checkpoint group's state at
token P is useless once the full-attention blocks before P are evicted, and
full blocks past the last resident checkpoint are useless too. The server's
LRU (`pegaflow-core/src/cache.rs`) evicts each (hash, group) key on its own,
so such stranded bytes are possible today.

- Workload `cap`: its own multi-turn replay, larger than `replay` so the
  store matters: `--cap-convs auto x --cap-turns 12` conversations from the
  same r2egym file (official; `auto` = the fewest conversations, at least
  64, whose `C(0.5)` clears the capacity floor below, chosen before the
  baseline cap stage and recorded as `summary.cap_plan`; Qwen3-4B 64,
  DeepSeek-V2-Lite TP2 76). The baseline arm runs it first (`cap_cold`,
  `cap_warm`, no store) to fix `U` and the per-request reference local hits.
  Each store arm (under test, `old`, `native`, `lmcache`) then runs `cap_cold`,
  `cap_warm`, `cap_warm2..N` once per capacity fraction, each on a fresh vLLM
  and a fresh store of `C` bytes. Warm pass `j` starts the conversations
  rotated by `j/4` of their count, so the passes differ in arrival order.
- Capacity: `C = f x U`, `f` in `{0.25, 0.5, 0.75}` (official). Only
  `f = 0.5` is gated; the others are a reported curve. `U` = bytes of the
  distinct (group, key) blocks the unlimited oracle stores over the baseline
  `cap_cold` pass. A run is official only if `C(0.5) >= 2 x` vLLM's GPU KV
  bytes, so G5 measures the external store and not vLLM's own cache. Server
  fragmentation and headroom count against the connector/server.
- All external KV must be loaded from the PegaFlow server. A connector-side
  cache (holding KV in vLLM-process memory or anywhere outside the C-byte
  server) is a gate violation; G5 checks it through the server's load bytes.
- Oracle at capacity `H*_C`: the G3 store events, with a store of `C` bytes and
  clairvoyant eviction: an item (group, key, bytes) whose next use is furthest
  in the future goes first; next use = the next request (by first schedule)
  whose unlimited-oracle resolved prefix covers the item; an item never used
  again is not stored. Hits are `semantics.resolve` over present blocks, as in
  G3. This is a strong reference, not a proven optimum; `E_C > 1` is allowed.
- References `H_lru` (group-oblivious LRU, everything stored) and `H_freeze`
  (store until full, never evict): reported only, to show how far naive
  policies are from the bar.
- Score over the timed cap warm passes, against the baseline's local hit for
  the same request (`ref`), so an arm cannot raise its own score by keeping
  less in vLLM's local cache, nor gain from keeping more:
  `ext = max(0, min(actual, oracle) - max(ref, own local))` (capped by the
  unlimited oracle), `ext*_C = max(0, oracle_C - ref)`, `E_C = Σext / Σext*_C`.
- Diagnostics, per group (not gated): resident bytes and stranded bytes at the
  end of each pass (resident blocks that lie on no resolvable prefix of any
  request in the trace), evictions, saved / loaded bytes. Implemented for the
  simulated stores; the server side reports eviction and save/load byte
  counters and its resident / used bytes at each pass end. TODO: actual
  stranded bytes need a server-side resident-key dump (server change, own PR).
- Gate at `f = 0.5`: no load errors; the cap passes' digests pass
  `G2.digest_self` (eviction must never produce a wrong load); server load
  bytes cover every external hit; `excess` bounded as in G3 against the
  unlimited oracle. `E_C` (reference 0.9) and its change vs the last
  accepted iteration are reported and go into the hit-rate judgment. A slice-size G5 (below the capacity floor) is a smoke test of
  flow and digests only; its `E_C` supports no conclusion.
- Reference arms: `old`, `native` (vLLM `OffloadingConnector` with exactly
  `C` CPU bytes) and `lmcache` (LMCache local CPU backend of exactly `C`
  bytes, 256-token chunks) record their G5 numbers and are never required;
  phase 2 will gate warm replay and `E_C` against `native`.
- lmcache: not comparable on vLLM 0.32 (wrong KV), 2026-10-23. lmcache 0.5.6
  (wheel built against torch 2.13.0; ops load under the shared venv's
  2.13.0+cu130), vLLM 0.32.0, Qwen3-4B, FLASH_ATTN, layout `LBNHC`, LMCache
  detects `NL_X_NB_BS_NH_CS`. Default adapter (LMCache's "latest dev"):
  gsm8k cold accuracy 0.859 (= baseline), warm 0 (outputs empty / one
  token); under batch invariance external digests 64/64 mismatched vs
  baseline (gsm8k) and 468/468 (replay, self), every one of 36 layers.
  The wrong blocks are not the poison pattern (per-layer L1 is never the
  all-0x3F value 24448) and equal no computed copy of any key (so no
  block / chunk mix-up and not a load after poison); per-layer L1 differs
  from the right block by 6-46%: wrong bytes, not a reordering of the right
  ones. vLLM's in-tree adapter (`use_native: true`) does not start with
  0.5.6 (`ImportError: CudaIPCWrapper` from
  `lmcache.v1.multiprocess.custom_types`). The arm stays wired
  (`--arms ...,lmcache`) for a later lmcache; it is not in official runs.
- Known server limit (slice 2026-10-23, `old` and connector_v2 on the same
  build): reclaim evicts `RECLAIM_BATCH_SIZE = 512` blocks per round
  (`pegaflow-core/src/storage/mod.rs`), so on a small pool each eviction
  burst drops about half of `C` (Qwen3-4B: 512 x 2.25 MiB = 1.13 GiB of a
  2.24 GiB pool) and only ~0.5 C stays resident. LFU admission is off
  (all saves inserted). The server LRU is also not prefix-aware: a chain's
  head blocks, served from vLLM's local cache and never re-read from the
  server, age out first and strand the chain's tail for any lookup that
  starts at block 0 (`old` slice: 0 server hits in every cap pass, pool =
  C, no alloc failure or admission reject). Fixing either is a server
  change (own PR).
- Pre-fix reference (`2026-10-23-qwen3-4b-g5-official`, gate rev 369b253,
  server from `feat/semantics-rsi`; before the review fixes: cap = replay
  trace 32 x 12, `C = 0.5 x U` from the baseline replay, per-arm local, no
  rotation, so not comparable to later runs). U 25.81 GiB, C 12.91 GiB
  (1.24x GPU KV, below the current floor). `warm_timed`, prompt 2.55 M tokens:

  | arm | E_C | hit | H*_C | H_lru | evictions / pass | load bytes / warm pass |
  |---|---|---|---|---|---|---|
  | connector_v2 | 0.216 | 0.440 | 0.785 | 0.517 | 42.5k-45.6k | 16.6-19.2 GB |
  | old | 0.124 | 0.459 | 0.814 | 0.523 | 35.3k-40.4k | 8.8-9.6 GB |
  | native | 0.558 | 0.649 | 0.843 | 0.561 | n/a | n/a |

  `H*` (unlimited) 0.998 for all. All three are far below 0.9 and below
  `H_lru`; native (vLLM's CPU offload, same C) beats both PegaFlow arms.
  Warm TTFT p50: native 1.38-1.51 s, connector_v2 2.01-2.08 s, old
  2.09-2.14 s, baseline (replay, no store) 2.54 s.

## Run record

Every gate run writes `~/kexi/rsi-runs/<date>-<model>-<iter>/` with
`summary.json` (all numbers above), raw logs, the commit hash, and the vLLM /
PegaFlow command lines. `PROGRESS.md` links the summary.

## Implementation (`rsi/gates/`, P0)

Tooling is gate-owned like this file: onboarding agents run it and read its
output, they do not edit it.

| File | Role |
|---|---|
| `rsi_probe.py` | `RsiProbe` connector, never loads/saves. Stacked after the connector under test in `MultiConnector`; dumps G0 spec, per-request trace, V3 digests |
| `digest_compare.py` | G2.1 byte comparison of digested blocks |
| `oracle.py` | G3 `H*` replay through `semantics.resolve`; G5 `U`, `H*_C` (clairvoyant) and `H_lru` capacity simulation |
| `test_oracle_capacity.py` | G5 oracle and gate contract on synthetic traces, no GPU (command below) |
| `workloads.py` | gsm8k, multi-turn replay, Prometheus scrape |
| `run_gate.py` | orchestrator: arms x workloads, cold/warm passes, verdicts |
| `provenance.py` | tree verification, under-test snapshot, static scan + allowlist |
| `isolate.py` | runs the arm under test with the runs root hidden (mount namespace) |
| `selftest/connector.py` | gate self-test connector: `honest` / `liar` / `bitflip` / `offbyone` |

Tests. They need `pytest`, `requests` and a built `pegaflow.semantics` (the
native module), so they run in a venv where `python/` is built. Everything
else they import is stdlib. From a checkout:

```bash
cd python && uv sync --extra test                 # pytest, requests
uv pip install 'maturin>=1.0,<2.0'
uv run maturin develop -r                         # CUDA 13: add --no-default-features --features cuda-13
cd .. && PYTHONPATH=. python/.venv/bin/python -m pytest -q -p no:cacheprovider rsi/gates
```

The full suite is the gate's test run, signal and teardown tests (real
subprocesses) included: 164 tests in about 45 s on the 4090 host. Nothing is
deselected. The isolation tests (`isolate.py`) need Linux and root and skip
elsewhere. The shared venv on the GPU hosts already has the built
`pegaflow`, so there only the last line is needed.

Trees on the host carry no bytecode by construction (M5 rejects any `.pyc`
in the tree under test): extract a rev with
`git archive <rev> | tar -x -C <dir>` plus `echo <rev> > <dir>/.rsi-rev`,
or sync a checkout with `tar --exclude=__pycache__ --exclude='*.pyc'`. Then
build the native module in place (`maturin develop` / copy
`pegaflow<EXT_SUFFIX>` into `python/pegaflow/`).

Usage (on the 4090 host 10.1.0.44, from the feature dir, shared venv):

```bash
CUDA_HOME=~/kexi/cuda13 PATH=~/kexi/pegaflow/python/.venv/bin:~/kexi/cuda13/bin:$PATH \
LD_LIBRARY_PATH=~/kexi/cuda13-lib \
~/kexi/pegaflow/python/.venv/bin/python -m rsi.gates.run_gate \
  --model ~/kexi/models/<model> --tag <model>-it<N> \
  --arms plain,baseline,old,native,lmcache,<module>:<Class> --workloads gsm8k,replay,cap --gpu 0
# G5 smoke on a slice (below the capacity floor: not official, no E_C conclusion):
#   ... --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --cap-convs 8 --cap-turns 6
```

Self-test at onboarding (gsm8k only, 128 questions per self-test arm, no PegaFlow):

```bash
... run_gate --model ... --tag <model>-selftest \
  --arms plain,baseline,selftest:honest,selftest:liar,selftest:bitflip,selftest:offbyone \
  --workloads gsm8k --gsm8k-limit 128
```

Exit code 0 only for an **official** run in which every required verdict passes.
A run is official only with the constants of `run_gate.OFFICIAL` (full gsm8k, 8
shots, 32 x 12 replay, 3 replay passes, pinned scheduler flags), the arms
`plain`, `baseline` and one connector under test, all three workloads
(`gsm8k`, `replay`, `cap`; cap `auto` x 12 at fractions 0.25,0.5,0.75), the G5
capacity floor (`C(0.5) >= 2 x` GPU KV bytes), no
`--num-gpu-blocks`/`--vllm-arg` override, and committed
`docs/rsi/models/<model>/{gate.json,spec.json}`. A run is also not official
when it is contaminated: any arm's G4-timed replay pass (`cold`, `warm2..`)
has p99 TTFT `> 10x` the in-run baseline's p99 for the same pass, or the
baseline's p99 is `> 10x` the median of the arms' p99 for that pass (a
baseline stall inflates every arm's margin) (`summary.contamination`; the pass is kept and still judged, never
dropped). `summary.json` lists `not_official_because`. A required verdict that cannot be evaluated is a fail.
Required verdicts: `baseline` and the connector under test, including
`G2.digest_self.cap` and `G5.replay_warm`. Not required (`required: false`,
every verdict recorded): `old` (black-box reference), `native` and `lmcache`. A crash of
a reference arm's stage is recorded under `errors` in its verdicts and the
run continues; a crash of baseline or the connector under test aborts it.
Every stage's vLLM / pegaflow-server runs in its own process group; teardown
SIGTERMs the group, waits up to 120 s, SIGKILLs it, waits up to 300 s, and a
group that survives that aborts the run (`TeardownError`, also for reference
arms: the next stage would start on a live GPU). For the arm under test the
same signals and waits also cover every process in its private mount
namespace (S11), so a process that `setsid()`s out of the group is stopped
too. `isolate.py` blocks right after its `unshare` until the gate has opened
`/proc/<pid>/ns/mnt` and is holding it, so the namespace is recorded before any arm
code runs. Processes are matched by namespace inode only and signaled through
pidfds; the arm cannot leave the namespace (no `CAP_SYS_ADMIN`).
Stage ports (vLLM, pegaflow-server gRPC / HTTP) are leased from the
host-reserved range 20000-29999 (`ip_local_reserved_ports`: never an
ephemeral port) under per-port `flock` files in `~/kexi/rsi-ports`, each
bind-checked at lease time. vLLM also gets a leased block of 8 contiguous
ports as `VLLM_PORT`: its own TCP ports (`get_open_port`) scan up from
there. On CUDA, TP>1 rendezvous goes through a file store and the engine
core's ZMQ sockets are `ipc://` paths, so the block is normally unused.
A stage process that fails before it is healthy, with EADDRINUSE on any bind
in its log (a leased port, or one vLLM / torch / ZMQ / the server picked
itself), fails the stage with `PortRace`. The error names each port parsed
from the failing lines, plus any leased port another process holds once
the stage process is dead. If no port is found, it names "unknown". It also
records whether the gate leased each port, the lease files and the failing
line. A port race is never retried. A stage's leases are released when the
stage ends, on every path: success, exception, stop signal, teardown. Every
failed stage writes `stage-error.json` (arm, workload, exception type,
error, `port_leases`) in its dir. `summary.port_leases` lists every lease of
the run (log, ports, block, lock files, leased/released times).
CUDA toolchain preflight: vLLM / FlashInfer JIT-build kernels with the
`nvcc` of `CUDA_HOME` or `PATH`, so a run is official only when every nvcc
found there has the venv torch's CUDA major. A missing nvcc, or a
`CUDA_HOME` without one, makes the run non-official.
`summary.cuda_env` records torch's CUDA version, `CUDA_HOME`,
`LD_LIBRARY_PATH`, `PATH`, and the path and version of each nvcc.

Stopping a run: `kill -TERM -<run_gate pgid>` (or `-INT`, `-HUP`; launched
with `setsid`, the gate's pgid is its pid). Each aborts the whole run: the
gate turns the signal into `SystemExit` (its own SIGINT handler replaces an
inherited SIG_IGN and asyncio's cancel-only handler), records it so no later
pass or stage starts even if a pass converts the exception, stops the
current stage's process groups as above (a signal during teardown or a
launch waits for the group; one landing in the few bytecodes before
`stop` owns a group leaves it in a live-group registry that an `atexit`
hook stops) and exits 128 + signal, with no `summary.json`. Never SIGKILL the
gate: its children, in their own sessions, would keep the GPU.

Exit 0 does **not** cover G1 (run `cargo test -p pegaflow-semantics`) or the
"no regression vs last accepted iteration" clauses of G3/G4: compare against
the last accepted run record by hand until `--accepted` exists.

### Precise definitions (these override the prose above where narrower)

- Arms: `plain` (no kv-transfer-config, gsm8k only), `baseline` (probe only),
  `old` (black-box reference), `native` (reference arm, record only: vLLM's
  `OffloadingConnector` as `--kv-offloading-size` sets it up,
  `kv_connector_extra_config = {"cpu_bytes_to_use": B}`, no PegaFlow server,
  `B` = `--pool-size` bytes for gsm8k/replay and exactly `C` for `cap`;
  phase 2 will gate warm replay and `E_C` against it; OffloadingConnector
  switches `kv_cache_layout` to `LBHNC`, so its `G0.spec` and
  `G2.digest_vs_baseline.gsm8k` fail by construction: layout, not data;
  `G2.digest_self.*` and `G2.gsm8k_outputs` are its correctness evidence),
  `lmcache` (reference arm, record only: LMCache's vLLM connector
  `LMCacheConnectorV1`, lmcache 0.5.6 from the overlay `--lmcache-overlay`
  (default `~/kexi/overlays/lmcache`, on `PYTHONPATH` of that arm's vLLM
  only: its pins would downgrade numpy / protobuf / otel in the shared venv),
  no PegaFlow server, configured by env only (inherited `LMCACHE_*` dropped):
  `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE` = `B / 2^30` GiB
  (exact: LMCache allocates `int(GiB x 2^30)` bytes = `B`, same `B` as
  `native`), `LMCACHE_MAX_LOCAL_DISK_SIZE=0`, no disk / remote URL, chunk
  size LMCache's default 256 tokens, telemetry off (`LMCACHE_TRACK_USAGE=false`,
  `DO_NOT_TRACK=1`); `result.lmcache` records the env and LMCache's logged
  configuration. LMCache stores and looks up 256-token chunks (a trailing
  partial chunk is not saved by default), so its hit lengths are multiples of
  256 where the oracle counts 16-token blocks: its `E` / `E_C` include that
  granularity loss and are not a pure eviction-policy comparison),
  `selftest:<mode>` (gsm8k only), `<module>:<Class>`. `old`, `native`,
  `lmcache` and the connector under test run as `MultiConnector[arm, RsiProbe]`.
  `--under-test-path DIR` (another tree's `python/`) stages a snapshot,
  `<run>/under-test-overlay/pegaflow`, and puts only that dir first on the
  `PYTHONPATH` of the connector-under-test arm (`provenance.stage`). The
  snapshot holds the verified rev's tracked blobs of `python/pegaflow` plus
  the one expected native module `pegaflow<EXT_SUFFIX>`; it is verified again
  after the copy (blob ids, sha256 equal to the source's). The source tree
  must hold nothing else under `python/pegaflow`: a symlink, an untracked or
  ignored file, another `.so`, or any `.pyc` makes it unverified. Editing the
  tree mid-run changes nothing the arm imports, and the tree cannot replace
  the probe, its worker extension, vLLM or `sitecustomize`. Every vLLM and
  pegaflow-server process runs with
  `PYTHONPYCACHEPREFIX=<run>/.pycache-stages/<arm>/<workload>`. That cache is
  made fresh before the stage, is outside every arm dir, and no later stage
  reads it (M10), so a stage cannot plant bytecode for a later one. The
  gate re-execs itself with
  `PYTHONPYCACHEPREFIX=<run>/.pycache`. So no process reads an in-tree
  `__pycache__`, and the gate and venv trees may carry `__pycache__/*.pyc`
  (any other untracked file still unverifies them). Gate code (`rsi/`)
  always comes from the gate tree. `pegaflow` does not: for the gate process
  (the oracle's `pegaflow.semantics`), the reference arms (`old`) and
  pegaflow-server's embedded Python it resolves through the shared venv's
  `pegaflow_llm.pth` (on the 4090 host:
  `/root/kexi/pegaflow-semantics-rsi/python`), not the gate tree's
  `python/`. `summary.provenance` records, before any stage: the gate
  tree's, the venv `pegaflow` tree's and the under-test tree's rev with
  `verified` (a clean git toplevel, or `.rsi-rev` whose `git ls-tree -r
  <rev> python/pegaflow` blob ids equal the files, fetched blob-less from
  `provenance.REMOTE`), each tree's native-extension sha256, and the
  pegaflow-server binary's sha256 with its tree's `.rsi-rev` (a claim: a
  binary is not tied to a rev). A run with `--under-test-path` is official
  only if the under-test snapshot is verified, and any run only if the gate
  tree and the venv `pegaflow` tree are verified too (an unverified judge or
  oracle is not official; `run_gate.provenance_problems`).
  Re-verification (M10): right after that, the gate fingerprints by blob
  id:
  - the gate tree (`rsi/`, `python/pegaflow`, the model's
    `docs/rsi/models/<model>`);
  - the venv `pegaflow` tree;
  - the snapshot the arm under test imports;
  - the pegaflow-server binary.
  It compares them again before every stage and after the last one. Any
  change (a file edited, added, removed or turned into a symlink) makes the
  run non-official, with the role, the path and the stage it was found
  before named in `not_official_because`. Every check is in
  `summary.provenance_watch`. The run goes on. Not covered: other venv or
  system files, and a change made and undone within one stage.
  G0.provenance (every arm, and
  baseline): every spec dump (scheduler and each worker) records the module
  files of `rsi.gates.rsi_probe`, the worker extension, vLLM, torch,
  `pegaflow` (+ its loaded native module, `site/usercustomize` if loaded)
  and the interpreter prefix; probe and extension must resolve under the gate
  tree, vLLM / torch / prefix to the shared venv, and `pegaflow` to the arm's
  expected dir (the snapshot for the arm under test, the venv's
  otherwise). The probe also records `__file__` of the arm's connector
  module (`connector:<module>`, named by the gate in the probe's
  `connector_modules`). The root it must lie under is chosen by arm kind,
  never by module name. Only the declared self-test arms (`selftest:*`)
  resolve under the gate tree. The arm under test, whatever its module
  name, must resolve under its snapshot, and `old` under the venv
  `pegaflow`. The arm under test's module must also be in the `pegaflow`
  package, the only one the snapshot holds. An `rsi.*` module for the arm
  under test makes the run non-official. An official run needs
  `--under-test-path`; without it the arm under test imports the venv's
  tree, which nothing snapshots.
  The connector under test runs as `MultiConnector[under_test, RsiProbe]` with
  `kv_load_failure_policy=recompute`, reads its server port from
  `PEGAFLOW_PORT`, and gets an empty `kv_connector_extra_config`. The probe
  keeps `prefer_cross_layer_blocks=False`, so cross-layer KV layouts are out of
  scope for now.
- Boundary (M7). The connector under test runs in the same processes as
  `RsiProbe` / `RsiWorkerExt`, and `MultiConnector` imports its module in
  vLLM's API server too (`requires_piecewise_for_cudagraph`). So it *can*
  patch the instrument at runtime, and nothing in-process can detect that:
  a probe self-check would run in the patched process. The gate bounds this;
  it does not prevent it:
  - Isolation (a). The arm under test's vLLM and pegaflow-server are exec'd
    through `rsi/gates/isolate.py`, which uses a private mount namespace. The
    runs root is under an empty tmpfs. Only the current stage's dir
    `<run>/<arm>/<workload>` (rw) and the snapshot (ro) are bound back, along
    with the arm's bytecode cache. Its earlier stages and the gate dir stay
    hidden. Block devices and `/dev/{mem,kmem,port}`
    are masked. `CAP_SYS_ADMIN` / `CAP_SYS_PTRACE` / ... leave the bounding
    set, so the arm cannot unmount the cover or reach the host tree through
    `/proc/<gate pid>/root`. The baseline's outputs, digests and requests,
    and every other arm's, are not visible to it. The pid, IPC and network
    namespaces are the host's (CUDA IPC, loopback ports): the exec is in
    place, so `stop` signals and waits for the same process group as before,
    plus everything in the arm's mount namespace (S11).
    The command prefix is in `summary.isolation`. A run whose arm under test
    could not be isolated (not Linux, not root) is not official.
    Residual: this is deterrence, not a boundary. What holds (tested in
    `test_isolated_arm_cannot_read_other_arms`): with those capabilities out
    of the bounding set, opening `/proc/<pid>/ns/*` or `/proc/<pid>/root` of
    a full-capability process fails the kernel's ptrace access check
    (EACCES), so `setns` / `nsenter` into the host mount namespace and reads
    through `/proc/<gate pid>/root` fail, and the tmpfs cannot be unmounted.
    What does not hold: the arm stays uid 0 with `CAP_DAC_OVERRIDE` in the
    host pid / IPC / network namespaces, so it can write any host file outside
    the runs root. That includes code that later stages or other root
    processes run with full capabilities (gate tree, venv, system files).
    The controls for that:
    - re-verification before every stage and after the last one (M10). It
      covers the gate tree, the venv `pegaflow` tree, the snapshot and the
      server binary;
    - a fresh per-stage bytecode cache that no later stage reads;
    - the static scan and the review.
    Not covered: other venv or system files, and a change made and undone
    within one stage (`summary.isolation.residual` says the same).
  - Static scan (b). `provenance.scan` scans the snapshot, every text line
    and the strings of the native module, for `provenance.SCAN_PATTERNS`:
    names of the instrument (`rsi`, `rsi_probe`, `RsiProbe`, `RsiWorkerExt`,
    `rsi_poison`) and of run artifacts (`rsi-runs`, `probe_dir`, `digest-`,
    `-outputs.json`, `-tokens.json`, `requests.jsonl`), plus runtime
    patching (`sys.modules` writes, `gc.get_objects`/referrers, trace /
    audit hooks, `sys.meta_path` / `path_hooks`, `importlib.reload`). `rsi`
    matches the gate package as code (`import rsi`, `from rsi`, a `"rsi.`
    string, `rsi.gates` / `rsi/gates`), not a path containing `rsi`. A hit
    makes the run non-official unless it is on `provenance.SCAN_ALLOWLIST`
    (path glob + pattern + reason; adding an entry is a gate change, with
    review sign-off). Allowlisted hits are recorded with their reason in
    `summary.scan.allowlisted`, the others in `summary.scan.flagged`.
    Residual: the scan matches literal text, so it is deterrence too. It
    does not chase every route by pattern. These known gaps are covered by
    review of the snapshot, not by the scan (S12):
    - frame walking (`inspect.currentframe` / `inspect.stack`,
      `sys._getframe`, `sys._current_frames`, a traceback's frames) up to
      `MultiConnector._connectors` and the probe object;
    - class discovery through `__subclasses__()` (of `KVConnectorBase_V1`,
      `object`, ...), then `setattr` on the probe class;
    - `importlib.import_module` / `__import__` with a computed name, and any
      built string (`"r" "si"`), which the `rsi` pattern cannot see;
    - `ctypes` (raw memory, or calls into the interpreter's C API), and
      the native module, whose strings are scanned but whose code is not.
  - No plugins (d). Every vLLM process runs with `VLLM_PLUGINS=""`: no
    `vllm.general_plugins` entry point runs (the venv's `pegaflow` plugin
    would run a tree's code in every vLLM process, the API server included).
    Arms name their connector by module path. The connector module is still
    imported in the API server, by `MultiConnector`'s class lookup.
    Exception: the `old` reference arm runs with `VLLM_PLUGINS=pegaflow` (that
    entry point only). vLLM 0.32's API server rebuilds each MultiConnector
    child's stats by registered name
    (`MultiConnector.build_kv_connector_stats` ->
    `KVConnectorFactory.get_connector_class_by_name`). `PegaKVConnector`
    reports stats and only that plugin registers its name, so without it
    every old-arm request fails with HTTP 500. The old arm is a reference arm
    whose connector is imported from the venv's `pegaflow`, and the plugin
    module comes from that same package, never from the tree under test. It
    only registers connector names (lazy imports). The other arms' children
    report no stats (probe, selftest, `connector_v2` today), or are vLLM
    builtins registered under their class names (native
    `OffloadingConnector`, lmcache `LMCacheConnectorV1`). If the connector
    under test starts reporting stats, it hits the same lookup and fails
    loudly (HTTP 500 on every request), never silently. The fix is in that
    connector: its own module registers its class under its class name when
    it is imported (`KVConnectorFactory.register_connector(name, __name__,
    name)`). vLLM imports that module in the API server by module path. The
    arm under test never gets the plugin, which would run the venv's
    `pegaflow` code in the arm's processes. `summary.vllm_plugins` records
    each arm's value.
  - Independent signal. Completions are read by the gate *client-side*, from
    the `/v1/completions` HTTP response (`workloads.run_gsm8k`), and judged
    in the gate process. What the gate judges from and writes itself (pass
    outputs and tokens, `result.json`, `stage-error.json`) is in
    `<run>/.gate/<arm>/<workload>` (M9). No arm process ever gets that dir.
    Only what the arm's processes must write (the probe's records, logs) is
    in its stage dir, and it is untrusted. A symlink or non-regular file left
    there fails the stage. A later stage of the arm cannot see an earlier
    one. A connector that fakes digests while loading
    garbage still changes completions (exact vs baseline cold, or vs `split`
    on non-aligned prompts on split-variant models). A connector patching
    the API server could in principle forge completions too. That is what
    the scan and the review checklist below are for.
  - Review checklist (skill step 6): the connector does not touch the probe,
    the worker extension, `sys.modules`, the poison RPC or any run artifact.
    No instrument or run-artifact access, by code or by string.
- Per-model settings: `docs/rsi/models/<model>/gate.json` (gate-owned):
  `batch_invariant`, `split_invariant` (default true, established by
  `G2.split_control`, see below), `invariant_attention_backend`,
  `attention_backend`, `vllm_args` (e.g. `mamba_cache_mode`). Pinned for every arm:
  `--max-num-batched-tokens 2048 --max-num-seqs 256 --enable-prefix-caching
  --seed 42 --generation-config vllm`, `PYTHONHASHSEED=0`.
- Passes: each arm x workload gets fresh vLLM (+ fresh PegaFlow for `old` and
  under-test arms). gsm8k and replay first send one untimed warmup request
  (`workloads.warmup_prompt()`: 512 fixed pseudo-random token ids, 16 output
  tokens, `rsi_pass=_warmup`, never scored) and `POST /reset_prefix_cache`;
  the stage fails unless no workload request shares a prefix block with it
  (`oracle.warmup_problems` on the probe trace). Its seconds are reported
  per arm vs baseline as `G4.warmup_seconds` (not gated; the one-time
  save-path cost it moves out of the cold passes, PROGRESS "Open
  questions"). Cap stages have no warmup
  (untimed; its blocks would take capped store space). `cold`, then per warm pass: sleep `--settle` s,
  `POST /reset_prefix_cache`, then `POST /collective_rpc {"method": "rsi_poison"}`
  (`RsiWorkerExt`, fills every byte of every KV allocation with 0x3F, a finite
  bf16/fp16 value), then the pass. Warm reuse can only come from the
  connector; stale cold-pass bytes cannot rescue a connector that loads nothing.
  gsm8k: `cold`, `warm` (plain arm also `split_ref`, `split`: the split
  control, below; with `split_invariant=false` the baseline arm also runs
  `split`, the same-split references for warm computed KV; its pre-warm
  requests are tagged `<pass>_prewarm` and never scored). Replay: `cold`, `warm` (digested), `warm2..warmN`
  (timed, N = `--replay-repeats`). Each request carries
  `vllm_xargs.rsi_pass = <pass>`; the oracle scores by that tag.
- Probe trace per request: `episodes = [step, local_hit, start]` per schedule
  episode (a resume after preemption opens a new one), `progress = [step,
  computed_after]` whenever a step crosses a key-token boundary. `local_hit` is
  the value passed to `get_num_new_matched_tokens` at the last attempt before
  the episode was scheduled; `first_computed` = `start` of the first episode.
  Load failures (`invalid_block_ids`, `failed_recving`) go to
  `load_errors.jsonl`.
- Digest (V3): requests tagged `vllm_xargs.rsi_digest = full` (gsm8k: all;
  replay `warm`/`cold`: every 4th conversation, all of its turns) are digested
  at finish: every fully computed block of every transferable group, per layer,
  on the raw allocation with `KVCacheTensor` geometry, unpadded page bytes (two
  64-bit position-weighted sums + float L1). `register_kv_caches` checks that
  geometry against every layer view and raises on mismatch. Tags, from the
  request's last episode: `local` (`covered <= local_hit`), `external`
  (`<= start`), `computed`.
- Digest comparison (`digest_compare.compare(REF, CAND, invariant)`): a loaded
  block (local or external) is `matched` if all its layers equal some computed
  copy of the same (rank, group, prefix key) in REF; `mismatched` if it equals
  none and the reference is exhaustive (batch-invariant, or every computation
  of that key in REF was digested); `uncertain` otherwise; `unreferenced` if REF
  never computed the key. A computed block is `exact` if it equals another
  computed copy; with `split_invariant=false` only copies with the same
  *split* count as references: split = (start of the computing episode, the
  `progress` chunk ends of that episode below the block's end), i.e. where the
  cached context ended and how prefill was chunked up to the block. A computed
  block with copies only at other splits is `computed_unmatched_split`, one
  REF never computed is `computed_unreferenced` (REF != CAND); both are not
  judged. `computed_by_pass` breaks these counts down by the candidate's
  `rsi_pass`. The split key misses a prefill chunk that ends without
  crossing a key-token boundary; that can only give two different MLA splits
  the same key, i.e. make an honest block inexact (fail-closed), never a
  wrong block exact.
- G0.spec: the arm's scheduler spec dump and per-layer view layouts (shape
  without the block dim, stride, dtype) equal the baseline's, and the
  baseline's equals the committed `spec.json`. Volatile fields (`num_blocks`,
  tensor `size`, `layer_stride`, model path, `max_model_len`) are excluded.
- G2 controls (baseline arm): `plain_equals_baseline` (gsm8k cold completions
  identical: the probe changes nothing), `warm_equals_cold` (under invariance:
  reset + poison change nothing), `digest_self.gsm8k` (no local mismatch).
- G2.digest_self.gsm8k / digest_vs_baseline.gsm8k: local and external
  `mismatched == 0`, `unreferenced == 0`, `external_matched > 0`, judged
  (matched + mismatched) >= 0.9 of all external blocks, and under invariance
  `computed_exact == computed_compared`. With `split_invariant=false`,
  digest_vs_baseline.gsm8k also needs, in each of `cold` and `warm`, judged
  computed blocks > 0 and `compared / (compared + unmatched_split +
  unreferenced) >= 0.9` (`MIN_COMPUTED_COVERAGE`; warm is judged against the
  baseline's `split` pass), and in `warm` zero unjudged computed blocks on
  prompts whose length is not a multiple of the hit alignment (a prompt
  without a recorded length counts as non-aligned; only aligned prompts,
  where the plain split stops one block short of a connector's hit, may
  leave warm blocks unjudged). The split control's `aligned` exemption
  likewise never applies to a request without a token record.
  G2.digest_self.replay: same, with no
  judged-fraction floor (untagged conversations share prefixes).
- G2.digest_groups (baseline and every arm, gsm8k probe): every group the
  oracle scores (not skipped, not scratch) has digest records and is not in
  `probe.digest_skipped_groups`. The probe lists every `MambaSpec` group
  there (outside `mamba_cache_mode=align` a state block is the running
  state; in align mode the block-end states are freed before finish, where
  the probe digests), so a hybrid model fails G2 until Mamba state is
  digested (the honest self-test needs it too).
- G2.gsm8k: `|acc(pass) - acc(baseline cold)| <= 0.01` for cold and warm.
  G2.gsm8k_outputs (when gate.json declares invariance): every completion equals
  baseline cold, cold and warm; with `split_invariant=false`: every cold
  completion equals baseline cold (same split), and every warm completion
  equals the plain arm's `split` completion of the same prompt, except on
  prompts with `prompt_len % cache_hit_alignment_tokens == 0`
  (`warm_differ_vs_split_aligned`, reported only: vLLM's own hit stops one
  block short there, a connector's need not). The warm differ count vs
  baseline cold is reported only.
- Split invariance (`split_invariant`, `G2.split_control`, baseline verdicts).
  Batch invariance makes a token's KV independent of batch composition, not
  of *where the request's computation starts*. vLLM 0.32's MLA prefill
  (`vllm/model_executor/layers/attention/mla_attention.py`: `forward_mha`
  attends to the new tokens, then, when `prefill_metadata.chunked_context`
  is set, i.e. `num_computed_tokens > 0`, runs `_compute_prefill_context` over
  the cached context and combines the two with `merge_attn_states`) splits the
  softmax at `num_computed_tokens`, so a prompt hit that ends at block 68
  computes the remaining KV and logits with different rounding than a hit
  ending at 72. Dense FLASH_ATTN attends over the whole context in one kernel
  and is split-invariant. Evidence on DeepSeek-V2-Lite TP2 TRITON_MLA, plain
  vLLM, `VLLM_BATCH_INVARIANT=1`: one 900-token prompt, prefix cache warmed to
  0 / 256 / 512 tokens: 256 changes logprobs, 512 diverges at output token 3.
  An honest connector's warm hit ends at the last full prompt block, the
  baseline's at the few-shot prefix, so on such models warm outputs differ from
  baseline cold and computed copies differ across runs although every loaded
  block is byte-exact. Onboarding writes the flag in its model's gate.json
  (on its own branch), but the run never trusts it: the plain arm
  (no connector, no probe) runs `split_ref` (the gsm8k prompts, greedy, with
  logprobs, after a reset + poison; hits end at the shared few-shot prefix)
  and `split` (same, but each request is preceded by the same prompt with
  `max_tokens=1`, so its hit ends at the last full prompt block, where a
  connector's warm load ends). `--enable-prompt-tokens-details` (plain arm
  only) reports cached prompt tokens per request. `G2.split_control` passes
  iff measured invariance (every completion and every sampled-token logprob
  equal between `split` and `split_ref`) equals gate.json's
  `split_invariant`, `split_ref` equals plain cold, and the pre-warm took
  (`split` cached `>= (prompt_tokens - 1) // align * align` tokens, i.e. up to
  the last full prompt block) for >= 0.9 of requests. So `false` with an
  invariant control, or `true` with any divergence, fails the run. With
  `split_invariant=false` the loaded-block checks (`mismatched == 0`,
  `unreferenced == 0`, judged >= 0.9, `external_matched > 0`) and G2.gsm8k
  stay as they are; only `computed_exact` (same-split references, with the
  coverage floor above) and `G2.gsm8k_outputs` (warm vs the plain `split`
  completions) change. A correct connector's warm pass is the control's
  `split` pass with the prefix served from the store instead of vLLM's
  prefix cache: same split, same bytes (byte-exact loads), so under batch
  invariance its completions equal `split`'s exactly; only the aligned
  prompts end their hit elsewhere (the connector loads to `prompt_len - 1`).
  Measured on DeepSeek-V2-Lite TP2: honest self-test 0/128 differ from
  `split`, liar/bitflip/offbyone 128/128; the old connector 38/1319, all 38
  aligned (82 aligned prompts).
- G3 (`oracle.py`): replays `requests.jsonl` with unlimited capacity. A block
  ending at token e is stored at the first step whose `progress` reaches e
  (checkpoint groups: only a step ending exactly on e) and is visible to
  requests first scheduled after that step. Kinds come from the spec dump
  (no window -> Full, sliding window sw -> Window{sw-1}, sw == 1 -> Scratch,
  Mamba -> Checkpoint, `UniformTypeKVCacheSpecs` -> its inner Kind, unknown ->
  error). Per request: `ext = max(0, min(actual, oracle) - local)`,
  `ext* = max(0, oracle - local)`; `E_ext = Σext / Σext*`;
  `excess = Σ max(0, actual - max(oracle, local))`. G3.gsm8k_warm and
  G3.replay_warm: `excess <= one block per request`, no load errors; replay
  G3 also requires G2.digest_self.replay. Reported (`reported`, not in
  `pass`): `E_ext` (reference 0.95, `None` is `below`) and
  `oracle_hit_rate` (reference 0.5), with hit, local and external rates.
- G3.no_eviction: `pegaflow_cache_block_evictions_total` delta = 0 in every
  gsm8k and replay pass (the oracle has unlimited capacity; `cap*` passes are
  meant to evict and are excluded; `native` / `lmcache` have no PegaFlow server: n/a, pass).
  The server runs with
  `--pool-size 64gb --disable-numa-affinity` (per-NUMA pools give one GPU only
  `pool_size / numa_nodes`; 64gb since 2026-10-24: a 160 GiB pinned pool took
  2-3 min to allocate, 60-95 s to free, and stalled other processes' JIT;
  measured U: Qwen3-4B gsm8k 30.3 GiB, replay ~26 GiB, DeepSeek-V2-Lite TP2
  <= 7.8 GiB). The run fails before any store stage (`PoolTooSmall`) when the
  uncapped workload's `UniqueEstimate` from the committed spec (replay:
  prompts + output tokens; gsm8k: prompts, a lower bound), or later the U
  measured on the baseline's trace, exceeds `--pool-size`;
  `summary.pool_checks` records each check.
- G4 (hard since 2026-10-23, user: "性能需要达标才行，需要比基准快"; every
  check is against the baseline arm on the same run, medians over the listed
  passes, `MAX_OVERHEAD = 0.02`):
  - `G4.replay_warm.ttft_p50` / `.req_per_s`: timed warm passes (warm2..N)
    strictly better than baseline's.
  - `G4.replay_cold.ttft_p50` / `.req_per_s`: cold pass at most 2% worse than
    baseline's cold (nothing to load yet; vLLM's own prefix cache serves the
    earlier turns, so cold can only tie; this bounds the save cost).
  - `G4.gsm8k_cold.seconds` / `G4.gsm8k_warm.seconds`: wall time at most 2%
    worse than baseline (the shared few-shot prefix stays in vLLM's local
    cache, so gsm8k bounds per-step connector overhead).
  - Replay G4 checks are void (fail) if G2.digest_self.replay fails. p99 values
    are recorded in pass results but not gated.
- G5 (`oracle.simulate` / `summarize_capacity`, `run_gate` workload `cap`):
  - Store events, visibility and scoring are G3's; per (group, key) block
    bytes = `real_page_size_bytes` (unpadded, as digested; `page_size_bytes`
    if absent) x the group's `num_layers` (`UniformTypeKVCacheSpecs` already
    sums its layers: x 1) x distinct TP copies `tp / kv_tp_replicas`
    (`KVCacheConfig.kv_tp_replicas`: ranks holding identical bytes; MLA's one
    latent head at TP2 = 2, so one copy). Stores keep one copy of replicated
    bytes: PegaFlow on DeepSeek-V2-Lite TP2 (2026-10-23) reported
    `pegaflow_save_bytes_total` 5,799,278,592 = 11,653 insertions x 497,664 B
    (18,432 page x 27 layers, one copy); vLLM's offloading CPU spec stores
    one copy under `replicated_layout` and `cpu_bytes_to_use` is the total
    over workers (`v1/kv_offload/cpu/spec.py`), so `native` gets `C` as is.
    Skipped groups (no kv transfer, eagle) hold no bytes. GPU KV bytes = Σ
    `kv_cache_config.tensors[].size` of one rank (not host-resident; groups
    sharing a tensor count once) x the same TP copies (Qwen3-4B:
    11,161,829,376 = `num_blocks x` block bytes; DeepSeek-V2-Lite TP2:
    3,096,465,408).
  - Trace: `--cap-convs` x `--cap-turns` (official `auto` x 12) conversations
    of `--replay-file`, `--replay-out-tokens` outputs, `--concurrency` workers.
    `--replay-skip N` (replay and cap) starts at conversation N of the file.
    The official value is 0. Any N > 0 is a held-out slice
    (`summary.held_out`, `summary.replay_skip`), never official.
    `auto`: before the baseline cap stage, from the baseline's spec dump,
    conversations are added from the file start (at least 64) until the
    estimated `U` (distinct prompt-prefix blocks per group plus each
    request's generated complete blocks, at block bytes) reaches
    `1.1 x 2 x GPU KV / 0.5`; the estimate equals the measured `U` exactly
    on the recorded Qwen3-4B 32 x 12 and 8 x 6 traces. If the whole file is
    not enough every conversation is used and the measured floor fails the
    run closed (`capacity below 2x GPU KV (whole replay file used)`).
    Every replay request carries `vllm_xargs.rsi_conv` / `rsi_turn`; the probe
    records them as `conv` / `turn`. Pass `cap_cold` starts conversations in
    order 0..n-1; warm pass `j` (`cap_warm` = 1, `cap_warmK` = K) starts them
    rotated by `j * n // 4`.
  - Baseline cap stage (runs after gsm8k and replay, before any store arm's
    cap): passes `cap_cold`, `cap_warm` (reset + poison before it, digested
    like replay `warm`), no connector. `U` = bytes of the distinct (group,
    key) blocks the unlimited oracle stores over its `cap_cold` pass (every
    block of the trace appears there; a rotated warm pass adds only differing
    generated tails, which `U` does not count). Reference local hit
    `ref(pass, conv, turn)` = the baseline's `local_hit` at that position, from
    `cap_cold` for `cap_cold` and from `cap_warm` for every warm pass.
    `G5.capacity_floor` (in `summary.json`, not a verdict): `C(0.5) >= 2 x`
    the baseline's GPU KV bytes, else `not_official_because: capacity below
    2x GPU KV`.
  - Store-arm cap stages, per fraction `f` in `--cap-fractions` (must include
    0.5; official `0.25,0.5,0.75`), dir `cap` for 0.5 and `capNNN` (percent)
    otherwise: fresh vLLM and (except `native` / `lmcache`, CPU bytes =
    `C`) a fresh PegaFlow server with `--pool-size C`
    `--disable-numa-affinity`, `C = int(f x U)`; passes `cap_cold`,
    `cap_warm`, `cap_warm2..N` (timed), all digested (every request of
    every pass: the timed passes are where `E_C` is scored, so their loads
    must be judged too, and with nothing undigested the reference is
    exhaustive; cap timing is not gated, the digest cost is accepted), the
    replay
    settle / reset / poison protocol. `pool_capacity_bytes` = the server's
    `pegaflow_pool_capacity_bytes` at start; the server allocates the whole
    pool up front and rounds allocations to 512-byte units and
    power-of-two-mantissa size classes (at most 1/8 waste): server headroom,
    counts against the server.
  - Simulated stores over the arm's own cap trace (all its `cap_*` passes, in
    first-schedule order), `C` bytes each, with `ref` as every request's local
    hit:
    - `H*_C` (clairvoyant): the next use of an item is the next request whose
      unlimited-oracle resolved prefix `P` needs it (`semantics.needed` at
      `P`, minus what `needed` at `ref` already gives); an arriving item with
      no next use is not stored; otherwise items whose next use is furthest
      are evicted first, and an arriving item whose own next use is the
      furthest is not stored. Greedy under variable item sizes and blind to
      partial-prefix coherence: a strong reference, not a proven optimum
      (`E_C > 1` is allowed).
    - `H_lru`: group-oblivious LRU, every item stored, a re-store of a
      resident item does not refresh it, a hit refreshes the blocks loaded.
    - `H_freeze`: items stored in arrival order until the next one does not
      fit, never evicted.
    - Hits: `resolve` over resident blocks plus each group's `needed` range
      at `ref`, capped at `max(unlimited oracle, ref)`.
  - Scores per segment (`warm_timed` = `cap_warm2..N`, and per pass):
    `ext = max(0, min(actual, oracle) - max(ref, own))` (unlimited oracle;
    `own` = the arm's own local hit: tokens vLLM held locally are never store
    credit, also when the arm's vLLM kept more than the baseline did),
    `ext*_C = max(0, oracle_C - ref)`, `E_C = Σ ext / Σ ext*_C` (`None` when
    `Σ ext*_C = 0`: fails). Reported: `lru_vs_oracle_c` and
    `freeze_vs_oracle_c` (`Σ max(0, H - ref) / Σ ext*_C`),
    `efficiency_c_own_local` (the same `E_C` with the arm's own local hit in
    place of `ref` for `ext`, `ext*_C` and the Belady run; diagnostic only),
    `efficiency_c_cold` / `freeze_vs_oracle_c_cold` (`cap_cold`: within-pass
    multi-turn reuse after vLLM evicted the earlier turn; diagnostic only).
    `unmatched_requests` = requests without a `ref` (no position tag, or the
    baseline has no request at that position).
  - Load proof: `external_hit_bytes(pass)` = Σ over the pass's requests of the
    bytes of the blocks `needed` at `actual` that `needed` at the request's
    own local hit does not hold; `loaded_bytes(pass)` = the
    `pegaflow_load_bytes_total` delta of the pass.
  - `G5.replay_warm` (gated fraction 0.5): `E_C` on `warm_timed` reported
    (reference 0.9, not in `pass`); gated: `excess <= one block per request`
    (unlimited oracle, own local, as G3),
    no load errors, `unmatched_requests == 0`, `G2.digest_self.cap` passes,
    and for a
    PegaFlow arm `pool_capacity_bytes` present and `<= C` and Σ
    `loaded_bytes >= Σ external_hit_bytes` over `warm_timed` (`native` / `lmcache`: no
    server, pool and load proof not applicable). Reported with it: `U`, `C`,
    hit / oracle / oracle_C / LRU / freeze hit rates, per-pass eviction
    counter deltas, per-pass server I/O (`saved_bytes`, `loaded_bytes`,
    `external_hit_bytes`) and end-of-pass resident / used bytes, per-group
    stored / not-admitted / evicted counts and bytes and per-pass resident /
    stranded bytes for `H*_C` and `H_lru`.
  - `G2.digest_self.cap` (cap probe dir against itself, every request
    digested): 0 mismatched and 0 unreferenced (local and external), and
    exhaustive: every loaded block judged. The only allowance is a block
    whose undigested computations are all preempted (non-last) episodes of
    digested requests (`*_uncertain_preempted`), at most 1% of the loaded
    blocks, reported as `preempted_uncertain`; any other `uncertain` fails.
    The connector under test must have external blocks (`judged > 0`: a
    passing G5 needs loads, and they must be judged); a reference arm
    (`old`, `native`, `lmcache`) with none passes flagged `vacuous`. Before
    2026-10-24 cap digested only every 4th conversation of `cap_warm` and no
    timed pass: undigested computations in `cap_warm2..N` made every key
    non-exhaustive, so wrong external bytes counted as `uncertain` and one
    match passed (re-review: 79 of 158 blocks corrupted, verdict passed).
    Baseline: the same, with no external block required.
  - `G5.curve`: the same evaluation at each other fraction (`E_C`, oracle_C,
    LRU, freeze, evictions, load proof, digests); never gated.
  - Why rotate: with identical warm passes a store that fills once and never
    evicts already holds what the next pass needs. Recorded traces with
    identical order: `freeze / H*_C` on the warm passes 0.96 (slice,
    connector_v2) and 0.92-0.99 (32 x 12 `old` arm), so a freeze policy could
    pass. A step-level vLLM simulation of the 64 x 12 trace (GPU KV 4745
    blocks, `C = 0.5 U`) gives 0.76-0.77 identical and 0.65-0.72 rotated
    (`H_lru / H*_C` 0.87-0.89 either way; `cap_cold` alone 0.71); it
    under-estimates freeze against real traces, so rotation is the gate's
    mechanism and the self-check below is its guard.
  - Separation self-check: if `freeze_vs_oracle_c` of the connector under
    test at the gated fraction is `>= 0.9` (`FREEZE_SEPARATION`), the trace
    does not separate a frozen store from the bar and the run is not official
    (`not_official_because: G5 trace does not separate freeze from H*_C`).
    This is workload validity, not a bar on the arm, so it stays hard.
- Self-test arms (`rsi/gates/selftest/connector.py`, full-attention groups
  only): `honest` saves at finish and loads saved prefixes (must pass every G2
  check); `liar` claims every complete unit and loads nothing; `bitflip` flips
  one byte of the first loaded block; `offbyone` loads block i+1 into slot i.
  Each negative arm must fail both G2 digest checks; output deltas are reported
  only (one flipped byte need not change a greedy completion).
