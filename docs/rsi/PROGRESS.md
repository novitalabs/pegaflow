# RSI: vLLM connector + PegaFlow — progress log

This file is the source of truth for the RSI effort. Any session (human or agent)
that resumes this work MUST read this file first, then `docs/rsi/GATES.md`, then
the skill `.claude/skills/pegaflow-new-model/SKILL.md`. Append, do not rewrite history.

Design doc: Feishu "Offloading rsi"
https://ppio-cloud.feishu.cn/wiki/XRmWwEfWjiIUHok6coIcz34fnpf (rev 62 at start).

## Scope (fixed by the user, 2026-10-23)

- Build the vLLM connector + PegaFlow path of the design; onboard models one by
  one in RSI mode.
- Models, in order: Qwen3-4B, DeepSeek-V2-Lite-Chat, Qwen3.5-4B, gemma-4-E4B-it.
- vLLM: latest release (0.32.0). PD disaggregation: out of scope.
- New semantics (Kind/Shard/...) may be added without asking; every addition is
  logged in "Semantics changelog" below.
- An independent review agent checks every iteration.
- Onboarding agents must NOT read or copy the old connector implementation
  (see "Isolation"). Old connector is a black-box baseline only.
- Every step is recorded here so a lost context never derails the RSI loop.

## Layering (from the design doc)

| Layer | Language | Who changes it |
|---|---|---|
| adapter (per KVCacheSpec type) | Python | onboarding agent |
| policy (what to save/load/prefetch) | Python | onboarding agent (main RSI surface) |
| semantics (`pegaflow-semantics`: needed/resolve/saveable/key/seal_plan) | Rust, PyO3 | agent proposes, review agent + human audit |
| server (storage only) | Rust | not changed by agents (user, 2026-10-24): gates run on the base server; a server limit is attributed and listed in "Server limitations" below for a human to decide |
| gates (`docs/rsi/GATES.md`, `rsi/gates/`) | — | human only; agents may add checks, never relax |

## PR policy (user, 2026-10-23)

- Connector code is NOT per model: one `connector_v2`, adapters per vLLM
  KVCacheSpec class, policy shared, per-model knobs only in
  `connector_v2/models/<model>.toml`. A model iteration may touch adapters /
  policy as long as every earlier accepted model still passes its gates.
- Review unit = one model: each model's onboarding (all its iterations, docs,
  accepted run) is one branch `feat/rsi-<model>` and one PR, stacked on the
  previous one. #473 is the observation PR for the foundation (semantics,
  gates, connector core) + Qwen3-4B.
- Server stays at the base version (user, 2026-10-24: "server还是保持原始版本，
  把server的局限问题汇报给人就行，gate不要被server受限，命中率不用过度追求最高，
  因为淘汰策略不一定是全局最优的"). Every gate run uses the pegaflow-server
  built from `feat/semantics-rsi`. Agents do not write or tune server code;
  they attribute a server-caused gap and list it under "Server limitations".
  A model is never blocked or held "pending server" because of one. The
  earlier server work (H1-H5, PRs GentleCold/pegaflow#3 and #4) is parked as
  a proposal for the human; nothing depends on it.
- Gate changes (`rsi/gates/**`, GATES.md) stay human-owned: agent-written gate
  code lands on its own branch (e.g. `feat/rsi-g5`) and is reviewed before it
  is used to accept anything.
- Review flow (user, 2026-10-23: "这些评审都不要等我"): nothing blocks on the
  user. Every review (iteration, gate branch, server branch) is done by an
  independent review agent; the orchestrator merges gate branches and moves on
  once that review passes. The user does one acceptance pass after all four
  models are supported, covering the gate PRs, model PRs and the "Server
  limitations" list (with the parked server proposals) together.
- Every progress report and ITERATIONS entry includes the hit-rate table
  (local / external / total hit, oracle H*, E_ext, and at capacity H*_C, H_lru,
  E_C), not only the perf numbers.

### Server limitations (report to the human, do not fix)

Measured on Qwen3-4B G5 (single Full group); details and numbers in
`feat/rsi-server-reclaim:docs/rsi/server/reclaim-eviction.md` and
`feat/rsi-server-evict-tail:docs/rsi/server/evict-tail.md`. Current plan
(r3, C(0.5) = 11536 blocks) at C/U 0.5: base server E_C 0.664-0.684,
H1+H3+H4 0.699-0.719, + H5 0.759-0.769 (hit 0.86 -> 0.89). H1+H3+H4 add
+0.10 at 0.25 and +0.06 at 0.75; H5 costs 0.01-0.08 E_C at 0.25.

- Reclaim drops a fixed `RECLAIM_BATCH_SIZE = 512` blocks per round. On a
  small pool one round frees about half of the cache (1019-block slice: 502
  resident after cap_cold). Proposal H1: `clamp(resident / 64, 1, 512)`.
- Pinned-region fragmentation: a layer-first save call takes one region per
  layer, freed only when the call's last block is evicted, so dead bytes
  stay pinned. Large on the older 32-conv plan (C = 5873 blocks: residency
  after cap_cold 5473 v2 / 4750 old, E_C 0.1-0.25 before H3), small on r3
  (see the E_C line above). Proposal H3: one region per saved block.
- H3's many small copies cost warm TTFT with per-range `cuMemcpyAsync`.
  Proposal H4: one `cuMemcpyBatchAsync` per transfer.
- Eviction is plain LRU per (hash, group) key. On r3 most of the gap to
  tail-first is fully evicted ranges (0.28 -> 0.23 of ext*_C), not
  stranding (1.7% v2 / 2.5% old). Heads aging out mostly hits `old`, which
  queries from its local hit and never refreshes them; v2 queries from
  block 0. Proposal H5: opt-in tail-first eviction in the retained class.
- Not group-coherent: a resume point needs every group, but each group key
  is evicted on its own (matters for hybrid models; not measured yet).

### Gate changes

- 2026-10-24, user (human gate owner): "命中率那块，不要设置硬性门槛，只要agent自己判断命中率足够就行了，不要差太多".
  Hit-rate levels are reported, not gated: G3 `E_ext >= 0.95` and
  `oracle_hit_rate >= 0.5`, G5 `E_C >= 0.9`, the G2 "> 0.9 of oracle" item
  and the no-`H`/`E_C`-regression clauses. Every number is still computed
  (`reported` per verdict, `summary.hit_report`). Correctness and run
  validity stay hard and unchanged. The onboarding agent writes a hit-rate
  judgment in ITERATIONS.md; the review agent checks it (GATES.md "Hit
  rate"). Branch `fix/rsi-gate-old-plugin`.
- 2026-10-24: the `old` reference arm runs with `VLLM_PLUGINS=pegaflow` (vLLM
  0.32 looks MultiConnector children up by registered name to rebuild stats;
  P4 MF1 got HTTP 500 on every old-arm request). Every other arm keeps `""`.
  Branch `fix/rsi-gate-old-plugin` 3fe119c.

## Isolation (no reference to existing implementation)

Forbidden for onboarding agents (read or grep):
- `python/pegaflow/connector/**` (old connector)
- model-specific branches in server: `pegaflow-server/src/service.rs` query
  branches, `pegaflow-core/src/instance.rs` slot math, `offload.rs` page_first
- git history/PRs of model adaptations (e.g. Gemma4 #455, K3)

Allowed: `pegaflow-proto`, `pegaflow-semantics`, the new connector package,
vLLM 0.32 source, `docs/rsi/**`, gate outputs, the Feishu design doc.

The main session has already read the old connector, so it only orchestrates;
model onboarding and optimization are done by fresh sub-agents with the
forbidden list in their prompt. The review agent checks for copying (structure,
names, magic constants that only exist in the old code).

## Environment

- Remote: 10.1.0.44 since 2026-10-23 (8x4090, driver 595, `LD_LIBRARY_PATH=~/kexi/cuda13-lib`);
  before that 10.121.196.3 (CUDA 13.0, driver 580). Everything under `~/kexi/`.
- Shared venv `~/kexi/pegaflow/python/.venv` (py3.12, vLLM 0.32.0, torch 2.13.0,
  flashinfer-python 0.7.0.post1). No per-branch venv.
- Feature dir `~/kexi/pegaflow-semantics-rsi`, sync via
  `.local/bin/pegaflow-remote-sync.sh --ip 10.1.0.44 --feature semantics-rsi`.
- Build: maturin with `cuda-13,rdma`, `PYO3_PYTHON` = shared venv python.
- Models: `/root/kexi/models/{Qwen3-4B,DeepSeek-V2-Lite-Chat,Qwen3.5-4B,gemma-4-E4B-it}`.
- Logs `~/kexi/logs`, run records `~/kexi/rsi-runs`.
- Local worktree `.worktrees/semantics-rsi`, branch `feat/semantics-rsi`
  (base upstream/master 1749353), trellis task `.trellis/tasks/10-23-semantics-rsi`.

## Phases

| Phase | Content | Status |
|---|---|---|
| P0 | gates: spec dump, V3 digest, gsm8k, perf workload, hit-rate metrics, old-connector baseline | done (p0-final 2026-10-23); G5 capacity gate in progress |
| P1 | `pegaflow-semantics` crate + PyO3 + V1 vectors | done (review: `reviews/P1-semantics.md`) |
| P2 | new connector skeleton (core glue + adapters + policy + models), coexists with old | done, review fixes applied (`reviews/P2-connector-skeleton.md` Resolution) |
| P3 | onboard Qwen3-4B | G0-G4 accepted at 8a05b7c (it2d official `2026-10-23-qwen3-4b-p3-it2d`, review ACCEPT-WITH-FIXES, fixes in it2e); G5 E_C reported (not gated since 2026-10-24): 0.684 on the base server (official r3, `models/Qwen3-4B/ITERATIONS.md` "G5 official (r3)"), hit-rate judgment pending on the base server (server proposals H1-H5 parked, see "Server limitations") |
| P4 | onboard DeepSeek-V2-Lite-Chat | in progress (`feat/rsi-deepseek-v2-lite`, TP2); review accept-after-must-fix, official run pending this merge |
| P5 | onboard Qwen3.5-4B | todo |
| P6 | onboard gemma-4-E4B-it | todo |

## Log

### 2026-10-23

- Worktree + remote env set up; vLLM upgraded to 0.32.0 in shared venv.
- Incident: `rm -rf /ppio/kexi` deleted an old `gds` dir and in-flight model
  downloads (outside `~/kexi`). Rule: `ls` before delete; only write under `~/kexi`.
- Models: Qwen3-4B, Qwen3.5-4B done; gemma-4-E4B-it was still downloading.
- Old-connector baseline (Qwen3-4B e2e) failed at vLLM startup:
  `flashinfer-cubin (0.6.12) does not match flashinfer (0.7.0.post1)`.
  No 0.7.x cubin on PyPI and vLLM 0.32 does not require it -> uninstalled the
  stale `flashinfer-cubin`. Rerun log: `~/kexi/rsi-runs/baseline-old-qwen3.log`.
- Second failure: pegaflow-server (cuda-13 build) picked `libcudart.so` 12.5 via
  ldconfig. Fix: `LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib`
  for every gate run on .3.
- All 4 models downloaded (DeepSeek-V2-Lite-Chat 30G, gemma-4-E4B-it 15G).
- Old-connector black-box baseline, Qwen3-4B e2e (`~/kexi/rsi-runs/baseline-old-qwen3-full.log`):
  4 passed / 1 failed. Outputs match baseline; saves 74 blocks (174.6MB), hits
  40 blocks (94.4MB) in the restart pass (vLLM "External prefix cache hit rate"
  71.1%). FAILED `test_same_process_hma_load_uses_pegaflow`: the warm repeat in
  the same vLLM process (prefix caching off) got hit_delta=0, load_delta=0.
  Recorded as a known baseline gap; the new connector must pass it (G2).
- Skill `.claude/skills/pegaflow-new-model/SKILL.md` created.
- Gate metric plan drafted in `docs/rsi/GATES.md`.
- User: "继续任务" without answering what agentx is -> keep the R2EGym fallback
  for G4 until told otherwise.
- P1 code: `pegaflow-semantics` crate (groups/keys/sharding), shared JSON vectors
  (`pegaflow-semantics/vectors/`) replayed by `tests/vectors.rs` and
  `python/tests/test_semantics_bindings.py`; PyO3 submodule
  `pegaflow.semantics` (`python/src/semantics.rs`, `semantics.py`, `.pyi`).
  Remote: 12 Rust tests pass, clippy clean (fixed one `is_multiple_of` lint).
  Independent semantics review agent launched.
- P1 review: APPROVE-WITH-FIXES, 6 should-fix + nits, all resolved or deferred
  with reason in `docs/rsi/reviews/P1-semantics.md`. Notable: `Group.window`
  arg order now `(block_tokens, tokens)` per design doc; `Group::new` fallible;
  scratch excluded from lcm step; seal_plan requires equal segments across all
  parts; `pegaflow-common::group_hash` now delegates to `semantics::key`.
  Evidence: semantics 16 tests, common 48, bindings 35, workspace clippy clean.
- Next: P0 gate tooling (spec dump, V3 digest, gsm8k runner, hit-rate/oracle),
  then P2 connector skeleton.
- P0 tooling written under `rsi/gates/` (gate-owned, agents read-only):
  - `rsi_probe.py`: `RsiProbe(KVConnectorBase_V1, SupportsHMA)`, never loads
    or saves. Scheduler: `spec.json` (kv_cache_config, spec fields + derived
    page sizes, tensors), `requests.jsonl` (prompt_len, local_hit from
    `get_num_new_matched_tokens`, first_computed = local + external at first
    schedule, first/finish scheduler step, prefix-key chain at gcd(block_size)).
    Worker: `spec-pp*-tp*.json` (per-layer view shape/stride/dtype) and V3
    digests of every fully computed block of finished requests (sampled by
    prompt hash), addressed via `KVCacheTensor` geometry on the raw backing
    buffer, so Mamba/raw specs are covered without view assumptions.
    Digest = two 64-bit weighted byte sums + float L1, computed on GPU.
  - `digest_compare.py`: tags blocks `hit` (covered <= first_computed), refs
    are the first computed occurrence; hit blocks must be byte-identical.
  - `oracle.py`: replays `requests.jsonl` through `semantics.resolve` with an
    unlimited store (saved at finish, visible from the next step); Kinds come
    from the spec dump (Full / Window{sw-1} / Checkpoint for Mamba), not from
    the connector under test.
  - `workloads.py`: gsm8k 8-shot (fixed train[:8] prefix), R2EGym multi-turn
    replay (token-id prompts, streaming TTFT), Prometheus scrape + diff.
  - `run_gate.py`: arms baseline / old / `<module>:<Class>`; per arm+workload
    fresh PegaFlow + vLLM (probe alone or MultiConnector[under_test, probe]),
    cold pass, local `reset_prefix_cache`, warm pass; writes
    `~/kexi/rsi-runs/<date>-<tag>/summary.json` with G2/G3/G4 verdicts.
    gsm8k runs with `VLLM_BATCH_INVARIANT=1` + FLASH_ATTN for byte comparison.
- Probe smoke (Qwen3-4B, .3 GPU0): 601-token prompt twice with prefix cache:
  2nd request first_computed=592, 38 blocks x 36 layers digested, hit blocks
  identical. Baseline without prefix cache (no batch invariance): 1333/1368
  layer-blocks byte-equal, max rel L1 1.9e-3, so cross-run byte equality
  needs batch-invariant mode (now default for gsm8k).
- Gate smoke (Qwen3-4B, 64 gsm8k q + 8x6 replay, baseline + old arms, run dirs
  `~/kexi/rsi-runs/2026-10-23-qwen3-4b-smoke{,-replay}`): all verdicts pass for
  the old connector. gsm8k 0.859 everywhere; old cold/warm completions identical
  to baseline (64/64); digest_vs_baseline 306432 hit layer-blocks, 0 mismatched,
  37296/37296 computed exact under batch invariance; old warm E=1.0008
  (external hit rate 0.946); replay old warm E=1.0004, TTFT p50 0.132s vs 0.209s.
- Oracle fix found by the smoke: the first version made blocks visible only at
  finish, so baseline cold scored E=1.97 (vLLM reuses the 8-shot prefix across
  concurrent requests). Now prompt blocks are visible after the first schedule
  step, generated blocks after finish; baseline cold E=1.000.
- Added G2.gsm8k_outputs (completion texts must equal baseline cold under batch
  invariance), GATES.md "Implementation" section with the precise definitions.
- Full Qwen3-4B P0 run started: `--tag qwen3-4b-p0-baseline` (gsm8k 1319,
  replay 32 convs x 12 turns). Baseline gsm8k 0.8582 cold/warm; replay warm
  TTFT p50 2.54s at concurrency 32.
- P0 review agent launched -> `docs/rsi/reviews/P0-gates.md`.
- P2 plan (started while the P0 run finishes):
  - Package `python/pegaflow/connector_v2/` (`PegaKVConnectorV2`), coexists with
    `pegaflow.connector`; layout per design doc: `core/` glue, `adapters/` (one
    file per KVCacheSpec class), `policy.py`, `models/*.toml`.
  - P2 ships core + adapter registry with NO adapters: startup on any model
    fails with a clear `Unsupported` naming the spec class. Default-gate unit
    tests drive the core state machine through a stub adapter. Adapters are
    added by the onboarding agent of each model (P3 adds the first one).
  - Server stays as is for P2-P6: all 4 models run TP1, where the current
    seal logic equals seal_plan. Server-side seal_plan + layout fingerprint
    namespace go to a later server phase (open question below).
  - Written by an isolated sub-agent (this session read the old connector).
- Full P0 run `2026-10-23-qwen3-4b-p0-baseline` finished. Baseline: gsm8k 0.8582,
  replay TTFT p50 cold/warm 2.525/2.541s, replay local hit rate 0.377 (GPU cache
  small at 8192 x 32), oracle hit rate warm 0.998. Old arm: gsm8k 0.8582 with
  outputs identical, but warm external hit rate 0 (gsm8k) / 0.004 (replay);
  G2.digest_self (0 hits compared), G3 (E 0.94 / 0.40) and G4 fail.
  Root cause is the gate, not the connector: the server ran with NUMA-split
  pools (60 GB / 8 nodes = 7.5 GiB on GPU0's node), 10752 evictions in gsm8k
  cold. Fix in `run_gate`: `--pool-size 160gb --disable-numa-affinity`, plus a
  new verdict G3.no_eviction (oracle assumes unlimited capacity, so evictions
  invalidate E). Rerun: `--tag qwen3-4b-p0-baseline2`, log
  `~/kexi/rsi-runs/p0-baseline2.log`.
- P2 skeleton: `python/pegaflow/connector_v2/` (`PegaKVConnectorV2`, lazy
  export). The vLLM-free `core/` does three things:
  - Hits: queries each group (storage group 0 = a full group via prefix query;
    the others via membership), then `resolve` -> `policy.accept`.
  - Leases: loads only the `needed` blocks past the local hit; leases with
    nothing wanted are released, unwanted blocks get `None` targets.
  - Saves: per step via `saveable` + `choose_saves`, synchronous in
    `wait_for_save`; failed loads are reported through `failed_recving`.
  - The namespace is a sha256 of the adapter declarations, the parallel shape
    and the model/dtypes.
  - No concrete adapters yet: unknown specs raise `Unsupported` at init on
    both roles.
  - Policy knobs come from `models/default.toml`.
  - `tests/test_connector_v2_core.py`: 7 passed; ruff clean; import proof OK.
  - Written in isolation: the old connector, `instance.rs` and `offload.rs`
    were not read.
- P2 review fixes (`reviews/P2-connector-skeleton.md`, "Resolution"):
  - M1/M2: the policy only sees immutable copies (frozensets, a tuple of
    SaveItems).
    - `accept` is capped at the legal `p`, then re-checked against the server
      answers.
    - Every wanted block must be covered by a lease of its group, so an empty
      hit is impossible.
    - `choose_saves` may only return offered items; anything else raises.
  - M3: checkpoint groups are `Unsupported` until the core handles
    `boundary_state_offloads` (P5 core task).
  - M4: `semantics` is injectable and imported lazily. The tests use a
    pure-Python stub plus native cross-checks.
  - S2: segment size == `bytes_per_block`, and a local `seal_plan` (V0) over
    all TP ranks.
  - S3: `LayerDecl.transfer_backend` / `page_first`, included in the namespace.
  - S6: hit and alloc planning never leak leases or raise into vLLM.
  - S1: one load RPC per request, a vLLM-trimmed hit is re-planned, and the
    README requires `kv_load_failure_policy=recompute`.
  - Also fixed: S4 (`Unsupported` without a full group), n2, n4-n8, part of n3.
  - Deferred:
    - S4 fetch for sid>0 (G3; before SSD/remote gates, P6 at latest);
    - S5 async saves (G4, after P3);
    - n3 remainder (server phase);
    - `run_gate` `kv_load_failure_policy=recompute` (owner of `rsi/`,
      before P3's gate run).
  - Test results on .3:
    - `test_connector_v2_core.py`: 77 passed (shared venv).
    - Source-only uv with no `.so` in the tree: 51 passed, 26 native-only
      cases skipped.
    - ruff format/check clean.
- P0 review fixes (`reviews/P0-gates.md` "Resolution", all M1-M9, S1-S12 and
  nits addressed; S5 mapping check stays a P5 task). Main changes:
  - Probe: schedule episodes + per-step progress, `load_errors.jsonl`,
    geometry self-check, digest tags via `vllm_xargs.rsi_digest`,
    `RsiWorkerExt.rsi_poison` (KV poisoned after every reset).
  - Digest: local / external / computed tags, any-copy reference with an
    exhaustiveness rule (`uncertain` otherwise).
  - Oracle: connector-only `E_ext`, `excess`, step-accurate store visibility,
    checkpoint-only-at-step-end.
  - run_gate: `plain` and `selftest:<mode>` arms, `kv_load_failure_policy=recompute`,
    `gate.json`, pinned scheduler flags, 3 replay passes, official profile.
  - Committed `docs/rsi/models/Qwen3-4B/spec.json` refreshed from the new probe
    (old dump lacked `real_page_size_bytes`, max_model_len 4096).
- Gate self-test, Qwen3-4B, gsm8k 128 q (`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p0-selftest`):
  - baseline controls pass: plain == baseline outputs (0/128 differ), poisoned
    warm == cold (0 differ), 16510 local blocks matched, 2576/2576 computed exact.
  - honest: 579 external blocks matched, 0 mismatched, outputs identical, E_ext 1.0.
  - liar: 1158 external mismatched vs baseline (unreferenced in self), acc 0.
  - bitflip: 128 external mismatched (exactly one per request), 28/128
    warm outputs differ, acc 0.844 (an accuracy-only gate would have passed it).
  - offbyone: 451 external mismatched, acc 0.18.
  - All four arms behave as expected -> the digest gates are not blind.
  - Found by the liar arm: gsm8k answer parsing crashed on garbage digit runs
    (`float('1e999')`); `_norm` keeps non-finite numbers as strings.
- Commits on feat/semantics-rsi: `feat(rsi): add P0 gate tooling`,
  `feat(connector): add connector v2 skeleton`.
- Full official run started: `--tag qwen3-4b-p0-final --arms plain,baseline,old`
  (log `~/kexi/rsi-runs/p0-final.log`, dir `2026-10-23-qwen3-4b-p0-final`).
  Baseline gsm8k 0.8582 cold/warm, replay cold TTFT p50 2.84s. Then .3 became
  unreachable from the jump host ("No route to host", .2 too). NEXT on resume:
  check whether the run finished (`summary.json`), else rerun the same command;
  read the old-arm verdicts, record them here, mark P0 done, start P3.
- P3 plan (isolated onboarding agent, Qwen3-4B, record `docs/rsi/models/Qwen3-4B/ITERATIONS.md`):
  - Spec dump: one `FullAttentionSpec` group, layout LBNHC, `block_stride == page`
    (65536 B). In LBNHC a block is `[N tokens][H heads][K|V]`, so K and V are
    interleaved and cannot be stored as two segments: declare ONE segment = the
    whole page.
  - New `adapters/full_attention.py`: `Group.full(block_size)`, `Heads` shard
    (replicas beyond `max_tp_shards`), view checked to be dense pages at the
    server's default block stride (the gRPC register path has no block stride),
    else `Unsupported`. No policy profile needed (defaults apply).
  - The core already matches vLLM 0.32 (block_state, transfer results,
    MultiConnector, empty extra, `PEGAFLOW_PORT`). No semantics change needed.
- P3 it1 written (`ITERATIONS.md` it1): `adapters/full_attention.py` + registry
  entry, adapter tests in `test_connector_v2_core.py` (local stub run 66 passed,
  41 native skipped; ruff clean), README adapter table, SKILL lessons. No core,
  semantics or policy change. .3 still "No route to host" at the end. NEXT on
  resume: sync, native unit run, then the it1 slice gate (command in
  ITERATIONS.md), then the official run only if the slice passes.
- P3 it1b: review `docs/rsi/reviews/P3-qwen3-4b-it1.md` (APPROVE-WITH-FIXES) addressed;
  Resolution table appended there. M1: `kv_cache_layout` in the namespace, plus
  kernel-split HNC pages rejected. S1: `non_causal` fails closed. S2: worker
  `save stats` log line every 500 steps and at shutdown. n1-n6 fixed. Local stub
  run 69 passed, 43 skipped; ruff clean. .3 still "No route to host". NEXT: the
  same as it1 (sync, native units, slice gate on GPU 1 after checking GPU 0 is
  not running the P0 final run), then read `save stats` for the S5 decision.

- Host move (user, 2026-10-23): .3/.2 stayed "No route to host"; the 4090
  host is now `10.1.0.44` (8x4090, driver 595 / CUDA 13.2, 1 TiB RAM,
  8 NUMA nodes, fresh box). Rebuilt from scratch under `~/kexi/`:
  - apt build deps (protobuf, ibverbs/rdmacm, numa, clang); rustup with the
    default dist server (the tuna rustup mirror 404s on 1.97.0), crates via
    the tuna sparse index in `~/.cargo/config.toml`.
  - Shared venv `~/kexi/pegaflow/python/.venv`: py3.12, vLLM 0.32.0,
    torch 2.13.0+cu130 (tuna). `maturin develop --uv` (the uv venv has no pip).
  - pegaflow-server must be built with `PYO3_PYTHON` = venv python (otherwise
    it links the system libpython3.10). The system toolkit is CUDA 12.2 and
    cudarc dlopens `libcudart.so` unversioned, so the server picked 12.2 and
    failed the preflight. Fix: `~/kexi/cuda13-lib` symlinks the venv's
    `nvidia/cu13/lib` with unversioned names; every gate run on .44 uses
    `LD_LIBRARY_PATH=~/kexi/cuda13-lib` (replaces the .3 cuda-13.0 path).
  - Data: gsm8k from the same OSS mirror (7473/1319); R2EGym parquet from
    hf-mirror, `r2egym-256.jsonl` regenerated with `prep-r2egym --n 256`.
    Models via modelscope into `~/kexi/models/`.
  - Remote checks: `test_connector_v2_core.py` + `test_semantics_bindings.py`
    147 passed (native included); server smoke healthy.
  - First launch died in vLLM startup: flashinfer JIT (sampling) used the
    system nvcc 12.2 (`--compress-mode` unknown). Fix: pin venv
    `nvidia-cuda-nvcc`/`nvidia-cuda-crt`/`nvidia-nvvm` to 13.0.x (13.4 nvcc
    rejects the 13.0 headers; 13.4 cicc emits PTX 9.4 that ptxas 13.0
    refuses), and `~/kexi/cuda13` = CUDA_HOME (bin/include/nvvm from the
    venv, lib -> cuda13-lib). Gate env on .44:
    `CUDA_HOME=~/kexi/cuda13 PATH=$VENV/bin:~/kexi/cuda13/bin:$PATH LD_LIBRARY_PATH=~/kexi/cuda13-lib`.
  - The .3 p0-final record is lost with the host, so p0-final is rerun from
    scratch on .44 GPU0 (`p0-final.log`). P3 it1 slice on GPU1
    (`p3-it1-slice.log`), rev 41fb158.

- P3 it1b slice gate (.44 GPU1, rev 41fb158, 64 gsm8k / 8x6 replay, not
  official): every verdict passes. G0.spec 0 diffs. G2 digest: external
  1492/1492 (gsm8k) and 617/617 (replay) matched, 0 mismatched; gsm8k 0.8594
  cold = warm = baseline, outputs identical. G3: gsm8k warm H=0.978 vs oracle
  0.992, E_ext=1.0; replay warm H=0.991 vs oracle 0.998, E_ext=1.0; excess 0,
  no evictions, no load errors. G4: warm TTFT p50 0.101/0.104 s vs baseline
  0.209 s (2.0x); cold 0.266 s vs 0.211 s (save cost). Save stats (gsm8k):
  ~21 ms per save step, 30% of steps save, sync 3.2 s of a 13.8 s window, rpc
  77 ms, 0 failed. Sync dominates, so S5 async saves are the it2 candidate
  (cold TTFT). Official run (full gsm8k, 32x12 replay) launched on GPU1:
  `~/kexi/rsi-runs/p3-it1.log`. p0-final on GPU0 still running (old arm).

- G4 made hard (user: "性能需要达标才行，需要比基准快"). New checks in
  `run_gate._perf_verdicts`, defined in GATES.md: warm replay TTFT p50 and
  req/s strictly better than baseline; cold replay and gsm8k cold/warm wall
  time at most 2% worse (`MAX_OVERHEAD`). Re-judged the it1b slice offline:
  warm wins (TTFT 0.10 vs 0.21 s, 6.6 vs 5.5 req/s) but cold replay TTFT
  0.266 vs 0.211 s (+26%), cold req/s 4.68 vs 5.11 (-8%), gsm8k +4.1% cold /
  +3.2% warm -> it1b NOT accepted. The official it1 run keeps going on GPU1 as
  full-scale data (judged with the old G4; re-judge its result.json offline).
  NEXT: it2 = remove the synchronous save wait (S5), slices on GPU2, by an
  isolated onboarding agent.

- Official it1 (`2026-10-23-qwen3-4b-p3-it1`) finished, all verdicts passed
  under the old G4. Re-judged with the hard G4: replay warm 4.9x TTFT / 2.8x
  req/s and replay cold 2.1x TTFT / 1.9x req/s vs baseline all pass; gsm8k
  cold +8.3% / warm +7.4% wall time FAIL. Not accepted; it2 (async save)
  targets exactly the gsm8k overhead. Details in Qwen3-4B/ITERATIONS.md.
  Save stats: device sync ~50% of each 500-step window (e.g. 10.5 s of 19.0 s
  on gsm8k), save RPC ~1.5%, 0 failed saves. Decision (timer check): stay on
  P3, it2 (async save) continues; P4 starts only after it2 is accepted.

- p0-final (`2026-10-23-qwen3-4b-p0-final`, arms plain/baseline/old) done:
  baseline controls pass; the `old` reference arm passes every G0-G3 check
  (E_ext 1.0 gsm8k and replay). Reference numbers (black box): gsm8k cold
  170.1 s vs baseline 169.5 s (+0.4%), replay warm2 req/s 10.2 vs 2.59,
  TTFT p50 0.62 vs 3.57 s. So a <2% gsm8k overhead is reachable. Note
  `plain` gsm8k is 148 s vs baseline 169.5 s: the probe (digesting every
  gsm8k request) costs ~13%; G4 compares arms that both carry the probe.
  P0 done.

- User: actual hit rate must approach the theoretical hit rate at a FIXED
  capacity; e.g. Kimi K3-style MLA + checkpoint groups whose boundaries differ:
  if the MLA blocks are evicted, later checkpoints are useless, which is a
  PegaFlow server limit (`cache.rs` LRU evicts each (hash, group) key on its
  own). Decisions: (1) new hard gate G5 (GATES.md): pool = 0.5 x unique bytes
  of the trace, bar = clairvoyant oracle at that capacity, `E_C >= 0.9`, plus
  per-group stranded/evicted/write-amp diagnostics; (2) the server is now in
  the RSI surface (Layering table): an onboarding agent may change eviction /
  admission / pools when a gate points there, Kind-aware and model-agnostic,
  with `cargo test`, an `old`-arm non-regression run and a separate review;
  (3) skill step 5 gets a hypothesis checklist (window tail, checkpoint
  boundaries, never-resumable state, group-coherent eviction, scan
  resistance, async save). Qwen3.5-4B (P5, linear attention + full) is the
  first model where cross-group coherence matters on the 4090s. G5 tooling is
  being implemented by a gate agent on branch `feat/rsi-g5`.

- User asked whether baseline should include vLLM's official offload. Yes:
  vLLM 0.32 ships `OffloadingConnector` (`--kv-offloading-size <GiB>`,
  backend `native`; `SimpleCPUOffloadConnector` with
  `VLLM_USE_SIMPLE_KV_OFFLOAD`), both `SupportsHMA`. New arm `native`
  (OffloadingConnector + RsiProbe, CPU bytes = PegaFlow pool size; at G5 the
  same capacity C). Phase 1: recorded next to `baseline` in G3/G4/G5 tables.
  Phase 2, after the first numbers: G4 warm replay and G5 `E_C` must also be
  at least as good as `native` (hard). Added to the G5 agent's scope.
  PR policy section added (per-model PRs, server PRs separate and
  human-reviewed).

- User worried policy would grow too complex. Agreed rules (SKILL.md step 5,
  checked by the review agent in step 6): no model / spec class names in
  `policy.py`; every knob needs gate evidence, toml only overrides defaults,
  unused knobs are deleted; budget ~150 lines per Kind and <= 12 knobs, refactor
  first when exceeded; simplification is a valid iteration; held-out replay
  slice against tuning to the trace; growing policy logic means it belongs in
  semantics (new Kind) or the server (separate PR).

- P3 it2 (isolated onboarding agent, `ITERATIONS.md` it2), two measured
  connector-core changes, no semantics or server change:
  - it2a (0bb44bf) async saves: `wait_for_save` records a CUDA event and a
    saver thread issues the save RPC after it. Finished requests with saves
    return `request_finished=True` and are released via `finished_sending`
    once their jobs land; preemption (and any window-group layout) flushes
    pending saves in `handle_preemptions`. Slice1 (full gsm8k + 8x6 replay,
    GPU2): all pass except gsm8k warm +2.9%; cold +1.1% (was +8.3%), replay
    cold TTFT within 0.5%. Save step-thread cost 10.4 s -> 15 ms per 500 steps.
  - it2b (2ff447e) sync short loads: warm gsm8k paid the async load round
    trip (7.6 steps admission -> first compute; TTFT sum 282 s vs 179 s).
    Hits <= `sync_load_tokens` (new knob, 256) load before the forward;
    failures go through `invalid_block_ids`. Enabled only for one KV group
    without block zeroing.
  - Slice2 `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2-slice2` (rev 2ff447e):
    every verdict passes. G4: replay warm TTFT 0.091 vs 0.210 s, req/s 6.85
    vs 5.51; replay cold TTFT 0.208 vs 0.210 s, req/s 5.07 vs 5.08; gsm8k
    164.6 vs 163.1 s cold (+0.9%), 163.5 vs 163.1 s warm (+0.3%). Hits: gsm8k
    warm local 0.916 / external 0.077 / total 0.992 = H* 0.992, E_ext 1.0;
    replay warm 0.730 / 0.268 / 0.998 = H* 0.998, E_ext 1.0. G2 digests 0
    mismatched. NEXT (orchestrator): review agent + official run.

- G5 branch review (independent agent): APPROVE-WITH-FIXES. Must-fix: pool
  check fails open when the metric is missing; nothing proves external hits
  came from the C-byte store (add load-bytes >= ext blocks x block bytes).
  Should-fix: E_C rewards lowering vLLM's own local hit (score against the
  baseline arm's per-request local instead); crashing reference arms abort the
  run; `old` still counted as required. Also: a "fill once, never evict" policy
  scores E_C ~0.95 because warm passes replay the same trace. User flagged the
  slice capacity as too small (C = 2.24 GiB = 0.22x GPU KV; official 12.9 GiB =
  1.24x). Decision: G5 gets its own larger workload (64 convs), C/U in
  {0.25, 0.5, 0.75} (gate at 0.5), hard floor C >= 2x vLLM GPU KV for an
  official run; slice G5 = smoke test only. Fixes + capacity change assigned to
  the gate agent on `feat/rsi-g5`; merge after a re-review.
- Server iteration started on `feat/rsi-server-reclaim` (separate PR):
  reclaim frees a fixed 512 blocks per round (~0.5 C on the slice) and the LRU
  is prefix-oblivious (chain heads served from vLLM's GPU cache age out first,
  breaking every lookup that starts at block 0). GPU6/7.
- P3 it2 official run `qwen3-4b-p3-it2` started on GPU1; independent review of
  it2 running.

- Official it2 (`2026-10-23-qwen3-4b-p3-it2`, rev 2ff447e): official, exit 0,
  all G0-G4 pass. replay warm TTFT 0.54 vs 2.55 s (4.7x), req/s 10.0 vs 3.4;
  cold TTFT 1.24 vs 2.79 s; gsm8k +0.9% cold / +0.0% warm. Hits: replay warm
  total 0.998 = H*, E_ext 1.0 (local 0.35 + external 0.65); gsm8k warm 0.992 =
  H*. Digests 0 mismatched. Table in Qwen3-4B/ITERATIONS.md.

- G5 official run, PRE-FIX gate code (`2026-10-23-qwen3-4b-g5-official`, gate
  rev 369b253, GPU4, same run for all arms; v2 numbers match it2). Reference
  only: the gate fixes change E_C (per-arm local subtraction) and capacity.

  | arm | replay warm2 TTFT | warm2 req/s | cold TTFT | gsm8k cold/warm s | replay warm hit (E_ext) |
  |---|---|---|---|---|---|
  | baseline | 2.538 | 3.41 | 2.81 | 164.6 / 164.8 | 0.377 (0) |
  | old | 0.630 | 10.24 | 1.356 | 166.7 / 169.5 | 0.998 (1.0) |
  | native (vLLM OffloadingConnector) | 0.500 | 9.83 | 1.308 | 169.6 / 169.4 | 0.998 (1.0) |
  | connector_v2 | 0.522 | 10.02 | 1.245 | 166.0 / 165.1 | 0.998 (1.0) |

  At capacity C = 12.9 GiB (0.5 x U = 25.8 GiB, ~1.24x GPU KV):

  | arm | cap hit | H*_C | H_lru | E_C | cap_warm2 TTFT | req/s |
  |---|---|---|---|---|---|---|
  | old | 0.459 | 0.814 | 0.523 | 0.12 | 2.087 | 3.73 |
  | native | 0.649 | 0.843 | 0.561 | 0.56 | 1.378 | 4.70 |
  | connector_v2 | 0.440 | 0.785 | 0.517 | 0.22 | 2.007 | 3.67 |

  Findings: unlimited, v2 beats native on cold TTFT, throughput and gsm8k but
  NOT on warm TTFT (0.522 vs 0.500 s, +4%) -> would fail the planned phase-2
  "at least as good as native" G4 check. At capacity native is clearly ahead
  (hit 0.65 vs 0.44, TTFT 1.38 vs 2.01 s): PegaFlow server reclaim/eviction is
  the gap (`feat/rsi-server-reclaim`). H*_C differs per arm because of the
  per-arm local subtraction (review S1, being fixed).
- User: add LMCache with the same config as another reference arm. Assigned to
  the gate agent after the G5 fixes: `LMCacheConnectorV1` + RsiProbe, local CPU
  only, capacity = pool size / C, default chunk size recorded. lmcache is not
  installed; installing into the shared venv would downgrade numpy/protobuf/otel,
  so it goes into an overlay dir on PYTHONPATH for that arm only.
- Reporting rule: every comparison table carries baseline, old, native,
  lmcache (when available) and connector_v2.
- P3 it2c (review fixes, 50e186b + 056eb4b): M1 saves held after sync loads
  until vLLM saw the outcome, no saves from un-rewound requests on failed
  blocks; S1 sync saves for zeroing/CoW layouts; S2 no sync-load deadline;
  S4 saver death releases holds (hung RPC still waits, no client deadline).
  Slice `2026-10-23-qwen3-4b-p3-it2c-slice1` (GPU1): every verdict passes;
  gsm8k v2 163.8 s cold / 162.3 s warm (+0.2%), replay warm TTFT 0.094 vs
  0.210 s; hits unchanged (replay warm 0.998 = H*). Held-out replay
  `2026-10-23-qwen3-4b-p3-it2c-heldout` (convs 33..): all replay verdicts
  pass. (Corrected by the re-review: replay warm TTFT moved +3.2% / +6.0% vs
  slice2, +1.4 / +2.6 pp vs baseline, and warm3 req/s -1.5 pp, so G4 was NOT
  within 0.5 pp of slice2; official rerun `qwen3-4b-p3-it2d` instead.)

- it2c done (50e186b, 056eb4b, 8af8fe4): all it2 review findings resolved
  (S4 partly: a hung save RPC still blocks, needs a server-side deadline).
  Slice and held-out replay slice pass. (The "G4 within 0.5 pp of slice2, the
  official it2 run stands" claim was wrong for replay warm, see the it2c
  re-review entry below.) Independent re-review of it2c running; Qwen3-4B is
  accepted for G0-G4 when it passes (G5 pending).
- P4 DeepSeek-V2-Lite-Chat started on branch `feat/rsi-deepseek-v2-lite`
  (model PR, stacked on #473), TP=2 (31 GB bf16 does not fit one 4090),
  GPUs 0-3, isolated onboarding agent. Its runs import connector_v2 from its
  own remote tree via PYTHONPATH (the shared venv's .pth points at
  `~/kexi/pegaflow-semantics-rsi/python`).

- P4 gate finding (DeepSeek-V2-Lite-Chat, TP2, TRITON_MLA, batch invariant):
  vLLM 0.32 MLA prefill attends to the cached context separately
  (`_compute_prefill_context` + `merge_attn_states` at the computed-token
  boundary), so outputs after a hit depend on where the hit ends; plain vLLM,
  no connector: prefix warmed to 256 changes logprobs, 512 diverges at output
  token 3. The honest selftest therefore fails `G2.gsm8k_outputs` (60/128) and
  `computed_exact`, while every loaded block is byte-exact and the negative
  selftests still fail. Gate fix assigned to the gate owner on `feat/rsi-g5`:
  `split_invariant` established by a gate-run split control (not declared by
  the agent); loaded-block byte checks and accuracy stay hard; computed_exact
  only against same-split references; outputs bounded by the control's own
  divergence rate. P4 official run waits for that merge.

- it2c re-review: APPROVE-WITH-FIXES. M1': a later turn of a tainted request
  can prefix-hit blocks computed on the bad bytes (recompute policy evicts
  nothing) and save on top of them; also a rewound loader leaving before it
  recomputes. Fix direction: fail closed (stop saves for the process after a
  taint). Should-fix: the "within 0.5 pp" no-rerun rule was misapplied
  (replay warm TTFT +1.4/+2.6 pp vs baseline), so the it2 official run does
  NOT stand for the final code: official rerun `qwen3-4b-p3-it2d` (GPU4, with
  old + native arms) after the fix. Qwen3-4B acceptance waits for it.

- G5 re-review: APPROVE-WITH-FIXES, merge after: (1) `G2.digest_self.cap`
  could not catch wrong loads (reference built from undigested timed passes is
  never exhaustive; 79 corrupted blocks passed as `uncertain`), (2) a vacuous
  cap digest must fail for the arm under test. Same change: E_C must not credit
  the arm's own local-cache variance; replicated MLA bytes were doubled at TP2;
  `cap_convs` auto-scales until C(0.5) >= 2x GPU KV (fixed 64x12 gives
  DSV2-Lite only 1.87x); GPU KV bytes from distinct tensors, not per group.
  Verified: G3 identical on 125 probe dirs, M1/M2/S1/S2/S3 fixed in code.

- P3 it2d (8a05b7c): M1' fixed by failing closed. The first sync-load failure
  vLLM reports (`invalid_block_ids`) disables all saves for the process: under
  the recompute policy vLLM evicts nothing, so blocks computed on bad bytes
  stay prefix-cached and any later request can extend them. Tests
  `sync-failed-sharer-next-turn`, `sync-failed-loader-preempted`; 166 passed.
  Teardown backport df01403 fast-forwarded (first official attempt
  `...-it2d-aborted1` died in gate teardown; unchanged relaunch is official).
  Official `2026-10-23-qwen3-4b-p3-it2d` (GPU4, `official: true`): every v2
  verdict passes; `old` (reference) fails G4.gsm8k_warm +3.4%, recorded only
  (this branch's gate counts it, so exit 1).

  | G4 check | v2 | old | baseline |
  |---|---|---|---|
  | replay warm TTFT p50 (warm2/3) | 0.563 / 0.538 s | 0.677 / 0.601 s | 2.544 / 2.547 s |
  | replay warm req/s (warm2/3) | 10.12 / 9.94 | 9.99 / 10.03 | 3.42 / 3.42 |
  | replay cold TTFT p50 | 1.287 s | 1.334 s | 2.804 s |
  | replay cold req/s | 6.70 | 6.75 | 3.26 |
  | gsm8k cold s | 165.96 (+0.9%) | 166.73 (+1.4%) | 164.45 |
  | gsm8k warm s | 164.29 (+0.0%) | 169.93 (+3.4%) | 164.28 |

  | hits | local | ext | total | H* | E_ext | excess |
  |---|---|---|---|---|---|---|
  | baseline gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0 | 0 |
  | old gsm8k warm | 0.928 | 0.065 | 0.993 | 0.992 | 1.0 | 1170 |
  | v2 gsm8k warm | 0.920 | 0.072 | 0.992 | 0.992 | 1.0 | 0 |
  | baseline replay warm | 0.378 | 0 | 0.378 | 0.998 | 0 | 0 |
  | old replay cold / warm | 0.386 / ~0.36 | 0.478 / ~0.64 | 0.864 / 0.998 | 0.864 / 0.998 | 1.0 | 0 / 195 |
  | v2 replay cold / warm | 0.381 / 0.348 | 0.484 / 0.650 | 0.864 / 0.998 | 0.864 / 0.998 | 1.0 | 0 |

  Save stats: 0 failed, 0 flushes, max 3 pending, step-thread submit 13-16 ms.
  Qwen3-4B: G0-G4 accepted pending the it2d review; G5 waits for the gate
  rerun.

- it2d review: ACCEPT-WITH-FIXES, no must-fix. Qwen3-4B G0-G4 accepted at
  8a05b7c; G5 pending r3. it2e (review fixes, no success-path change):
  - S1: every load is async once saves are disabled;
  - S2: lifecycle tests pin the two surviving mutants;
  - S3: provenance sentence fixed, and the connector logs its import dir;
  - S5: `min_hit_tokens` deleted, `loading_wait_ms` kept with the evidence
    gap recorded;
  - S4: exit 1 is the reference `old` arm only.
- Native reference (vLLM OffloadingConnector), record only and noisy (four
  gate runs on the host at once), `2026-10-23-qwen3-4b-p3-it2d-native`:
  - gsm8k cold/warm +7.5% / +4.0% vs the same run's baseline (G4 fail);
  - replay warm2/3 TTFT 0.505 / 0.459 s, req/s 9.63 / 9.76;
  - E_ext 1.0, excess 1170 / 195 (same as `old`);
  - G0/G2 digest fail because it runs LBHNC, not LBNHC (accuracy and
    outputs match).

  The first attempt died to the gate's aiohttp keep-alive race (fixed by
  8462ed9). The same-run v2 vs native comparison waits for G5 r3. ITERATIONS
  has an "it3 candidate" on load-path cost: native submits one batched
  `cuMemcpyBatchAsync` per load, v2 one `cuMemcpyAsync` per merged run;
  14.2 vs 21.9 GB/s effective H2D on replay warm2. Both copy 64 KiB
  per-layer pages.

- P3 it2e non-regression (fc06c9b):
  - Slice `2026-10-23-qwen3-4b-p3-it2e-slice`: all correctness and replay
    verdicts pass. gsm8k cold failed at +9.9%. A 10 s zero-throughput stall
    matches a P4 160 GiB pool teardown plus an allocation on the host.
  - gsm8k-only rerun `...-it2e-gsm8k-rerun`: cold +0.5%, warm -0.2%, pass.
  - Second native record-only sample (`...-it2d-native-e9f`, gate e9f3f33):
    the same hits, noisy timings.
- P3 it3 (measurement only, `2026-10-23-qwen3-4b-p3-it3-backend`):
  - v2 loads are blocks x 36 layers x 64 KiB copies (p50 7704 per load).
  - Direct backend: 38.2 ms per load at 15.0 GB/s.
  - Kernel backend: 48.3 ms at 12.3 GB/s (+26%); warm2 TTFT 0.709 vs
    0.518 s. Kernel rejected.
  - The batched-submit lever (native: 21.9 GB/s) is server-owned, so the
    numbers went to the server agent. P3 it3 stops.

- G5 implemented on `feat/rsi-g5` (gate agent): `oracle.py` capacity
  simulation (`U`, clairvoyant `H*_C`, group-oblivious `H_lru`, per-group
  stored / evicted / stranded diagnostics), `run_gate` workload `cap` (fresh
  PegaFlow `--pool-size C`, `C = 0.5 x U` from the baseline replay cold pass,
  passes `cap_cold`, `cap_warm`, `cap_warm2..N`), verdicts `G2.digest_self.cap`
  and `G5.replay_warm` (`E_C >= 0.9`) required for the connector under test,
  `cap` now part of the official profile, `native` reference arm
  (OffloadingConnector + RsiProbe, record only). G3 numbers re-derived
  identically on all p0-final / p3-it1 probe dirs. Slice
  `2026-10-23-qwen3-4b-g5-slice` (8 x 6, U 4.48 GiB, C 2.24 GiB): E_C
  connector_v2 0.40, native 0.20, old 0.0 (0 server hits); oracle_C hit rate
  0.903, LRU 0.827, local 0.79. Server limit found: reclaim drops 512 blocks
  per round, so a small pool keeps only ~0.5 C resident (GATES.md G5).

- G5 review (APPROVE-WITH-FIXES) applied on `feat/rsi-g5`: pool size must be
  proven for PegaFlow arms (M1); server `pegaflow_load_bytes_total` must
  cover every external-hit block, connector-side caches are a violation
  (M2); `E_C` subtracts the baseline's local hit for the same
  pass/conversation/turn, per-arm local kept as a diagnostic (S1); a
  reference arm's crash is recorded, not fatal (S2); `old` and `native` are
  `required: false` throughout (S3). User capacity concern (slice C 2.24 GiB
  = 0.22x GPU KV): cap is now its own 64 x 12 trace, baseline runs it first
  (U + reference locals), fractions 0.25/0.5/0.75 per arm (0.5 gated, rest
  a curve), official only if C(0.5) >= 2x GPU KV. Freeze policy: warm passes
  start rotated by j/4; a run whose under-test freeze/H*_C >= 0.9 is not
  official. G3 re-derived identically on all 45 recorded probe dirs.

- G5 pre-fix reference `2026-10-23-qwen3-4b-g5-official` (rev 369b253,
  32 x 12, U 25.81 GiB, C 12.91 GiB = 1.24x GPU KV): `warm_timed` E_C
  connector_v2 0.216, old 0.124, native 0.558; H*_C 0.79-0.84, H_lru
  0.52-0.56. Both PegaFlow arms are below H_lru, consistent with the server
  reclaim-batch / prefix-oblivious LRU limit (GATES.md G5). Not comparable to
  post-fix runs (different trace, per-arm local, no rotation).

- `lmcache` reference arm (LMCacheConnectorV1, local CPU = pool / C bytes
  exactly, chunk 256) wired on `feat/rsi-g5`, lmcache 0.5.6 in the overlay
  `~/kexi/overlays/lmcache` (shared venv untouched). lmcache: not comparable
  on vLLM 0.32 (wrong KV): smoke `2026-10-23-qwen3-4b-g5-lmcache-smoke` gsm8k
  warm accuracy 0, external digests 64/64 (gsm8k) and 468/468 (replay)
  mismatched, not poison and not another key's bytes; the in-tree adapter
  fails to import (GATES.md G5). Left out of the official run.
- `G2.digest_self.cap` fail for `old` / `native` on the fix slice was a gate
  artifact: cap sampled every 4th conversation and their few external loads
  (old: none; native: 6 requests in convs 2/5/6/7) fell outside it, so
  `judged > 0` failed with 0 mismatches (connector_v2: 2 sampled). Cap now
  digests every conversation and an empty sample passes flagged `vacuous`.
  Under the old rule the pre-fix official run had all three passing (old
  541/541, native 8567/8585 + 18 uncertain, v2 1375/1375 matched).

- G5 re-review (APPROVE-WITH-FIXES on 20e6b58..6510116) applied:
  `G2.digest_self.cap` accepted wrong loads (undigested timed passes made
  the reference non-exhaustive; the reviewer's 79/158 corruption passed):
  cap now digests every request of every pass and requires every loaded
  block judged (only preempted-episode uncertainty, <= 1%, flagged);
  vacuous passes only for reference arms. `ext` subtracts
  `max(ref, own local)` (slice: connector_v2 / old E_C 0.0775 -> 0 with 0
  server load bytes). Block / GPU KV bytes count `tp / kv_tp_replicas`
  copies (PegaFlow saves one MLA copy at TP2: 5,799,278,592 B = 11,653 x
  497,664) and GPU KV = Σ KV tensor sizes. `--cap-convs auto` (Qwen3-4B 64,
  est. C/GPU-KV 2.44; DeepSeek-V2-Lite TP2 76, 2.22; a fixed 64 x 12 gives
  it ~1.87). Re-scored offline: pre-fix official E_C unchanged (v2 0.216,
  old 0.124, native 0.558) but its `G2.digest_self.cap` now fails for
  native (18 external + 1 local unjudged, all from undigested requests):
  old-sampling records cannot pass the exhaustive rule; a rerun is needed.
  G3 `summarize` identical on all 153 recorded probe dirs.

- Gate teardown (d646ac3, backport df01403 on fix/rsi-teardown-backport):
  pegaflow-server with a 160 GB pinned pool took > 30 s to exit after
  SIGKILL and `proc.wait(timeout=30)` aborted it2d's official run; the gate
  now waits on the whole process group (120 s after SIGTERM, 300 s after
  SIGKILL) and logs the exit time.
- `split_invariant` (P4 finding): vLLM 0.32 MLA prefill attends to the cached
  context separately (`_compute_prefill_context` + `merge_attn_states` at
  `num_computed_tokens`), so on DeepSeek-V2-Lite the computed KV and greedy
  outputs depend on where a hit ends; an honest connector failed
  `G2.gsm8k_outputs` (warm 60/128) and `computed_exact` with every loaded
  block byte-exact. New gate.json field (default true), established by a
  plain-arm split control (`split_ref` vs `split`, each prompt pre-warmed);
  with false, `computed_exact` judges only same-split copies and
  `G2.gsm8k_outputs` is bounded by the control's rate + 0.02. Controls
  measured (full gsm8k 1319, pre-warm took on 1319/1319):
  Qwen3-4B FLASH_ATTN 0 differ / 0 logprob differ (split-invariant);
  DeepSeek-V2-Lite TP2 TRITON_MLA 647 differ (0.4905; first 128: 0.4688),
  1319/1319 logprobs differ (split-variant). Re-judged offline with
  `split_invariant=false` + that control: P4 self-test honest passes every
  G2 check (ext 1084/1084, warm 60/128 = 0.469 <= 0.489), liar / bitflip /
  offbyone still fail both digest checks (2168 / 256 / 828 ext mismatched);
  P4 old slice: old loaded blocks 10390/10390 matched, computed_exact
  27526/27526 on same-split copies (coverage 0.70 vs baseline),
  `G2.gsm8k_outputs` 642/1319 = 0.487 <= 0.511. Qwen3-4B: every verdict
  unchanged on the 24 recorded runs (5580 verdict leaves) apart from the
  new `G2.split_control`, which old records lack. DeepSeek-V2-Lite's
  gate.json must set `"split_invariant": false` (P4's file).
- G5 official r3 (2026-10-23, .44 GPU5): gate e9f3f33, connector_v2 at
  8a05b7c (`--under-test-path`, tree verified blob-for-blob afterwards),
  official, uncontaminated (max timed p99 ratio 3.2x vs the 10x limit).
  Qwen3-4B connector_v2 passes G0-G4. It fails G5: E_C(0.5) is 0.684 vs
  0.9, with hit 0.842 vs H*_C 0.929, about LRU level (H_lru 0.857). At
  C/U 0.25 / 0.75, E_C is 0.084 / 0.669. Tables are in
  `models/Qwen3-4B/ITERATIONS.md` "G5 official (r3)".
- G5 re-review 2 (accept-with-fixes on 4797a05..e9f3f33) applied in
  91b6a8e..940def4; resolution table in `reviews/G5-gate-rereview2.md`.
  Split-variant G2 is now exact: warm outputs == plain `split` (aligned
  prompts reported only), cold == baseline cold, warm computed KV judged
  against the baseline's `split` pass with coverage >= 0.9 per pass (DSV2
  honest self-test warm coverage 0 -> 1.0; liar/bitflip/offbyone fail;
  P4 old and it1 slices pass G2). Provenance: overlay with only `pegaflow`
  for `--under-test-path`, `G0.provenance` on module files, verified tree
  revs (blob-for-blob vs GitHub) and binary sha256 in `summary.provenance`.
  Teardown failures abort the run; SIGTERM/SIGHUP tear stages down;
  `--pool-size` 64gb with an early U check; untimed `_warmup` request per
  stage; contaminated timed passes (p99 > 10x baseline) make a run
  non-official; `G2.digest_groups` fails a transferred group without
  digests (hybrid / Mamba models fail G2 until Mamba state is digested).

### 2026-10-24

- G5 re-review 3 (M4-M7, S7-S9, N5) applied in c7f7520..b3d43b4. The
  resolution table is in `reviews/G5-gate-rereview3.md`.
  - Every new run passes `G0.spec` again (M4).
  - The arm under test imports a verified snapshot of the rev's blobs.
    Symlinks, untracked files, extra `.so` and pyc all reject the tree.
    Trees for `--under-test-path` must be pyc-free (`git archive`).
  - Unverified gate and venv trees make a run non-official.
  - The arm runs in a private mount namespace that hides the other arms,
    with SYS_ADMIN and SYS_PTRACE dropped. This is deterrence, not a
    boundary (uid 0).
  - A static scan of the snapshot flags instrument access.
  - `VLLM_PLUGINS=""` for every vLLM.
  - Gate fixes and run hygiene:
    - The connector module's import path is checked.
    - The aligned-prompt exemption is exact.
    - Contamination is two-sided.
    - `G4.warmup_seconds` is reported.
    - Stage ports are leased (`PortRace`, never retried).
    - CUDA toolchain preflight.
    - `--replay-skip` for held-out slices (never official).
- GPU3 run on 7d477c5 (02:07:41Z-02:14:40Z, gsm8k 64, 16gb pool):
  - G0.spec and G0.provenance pass.
  - connector_v2 at 8a05b7c served inside the isolation: accuracy equal to
    baseline, G3 E 1.0.
  - The namespaced groups were torn down.
  - The server log shows the full 16 GiB pool and 36 CUDA IPC layers
    registered, with no errors.
- The Qwen3-4B re-judge (39cd826 vs b3d43b4, 49 records) left the G3
  numbers unchanged (147/147) and overall passes unchanged. The only flag
  change is M4's `G0.spec` fix.

## Semantics changelog

Each entry: what was added, why, which vectors cover it.

- S1 (P1, initial): `Kind::{Full, Window{tokens}, Checkpoint, Scratch}`,
  `Group{kind, block_tokens}`; `needed`, `resolve` (lcm step, cap prompt_len-1,
  all-scratch -> 0), `saveable` (checkpoint only on block boundary).
  Vectors: `vectors/resolve.json` (11 cases incl. design-doc example P=12),
  randomized brute-force cross-check in `groups.rs`.
- S2 (P1, initial): `key(hash, group)` = raw hash for group 0, else hash ++ BE u32.
  Vectors: `vectors/key.json`.
- S3 (P1, initial): `Shard::{Replicated, Heads{index,count}, Tokens{index,count,interleave}}`,
  `Region`, `RankDecl`, `seal_plan` -> per-group fragment list (slot = position,
  ordered by layer name then part; identical parts merge, lowest rank = writer).
  Errors: duplicate rank/layer, invalid shard, group/scheme/segment mismatch,
  missing part. Unit tests in `sharding.rs`.
- S4 (P1 review): `Group::new` rejects `block_tokens == 0` and `Window{tokens: 0}`;
  resolve step = lcm over non-scratch groups only (scratch loads nothing);
  `Window.tokens` documented as "trailing cached tokens" (vLLM `sliding_window - 1`
  exact). seal_plan: all parts of a layer must share segment sizes; zero-byte
  segments rejected; writer = de-dup hint, store accepts any holder.
  Vectors: `resolve.json` (18), `seal_plan.json` (12), `key.json` (3).

## Open questions

- "agentx" perf workload: not available. User (2026-10-23): use the R2EGym
  replay for now. R2EGym (`~/kexi/datasets/r2egym-256.jsonl`, 32 x 12 official)
  is the G4/G3 replay workload until agentx is provided; results only claim
  this workload.
- For final acceptance (user's final pass; agent review counts as acceptance
  until then, GATES.md stays DRAFT): the `split_invariant=false` rule replaces
  cross-split `computed_exact` and exact outputs-vs-baseline-cold on
  split-variant models (MLA prefill, e.g. DeepSeek-V2-Lite). Instead: warm
  outputs == plain `split` exactly (aligned prompts reported only), cold ==
  baseline cold, `computed_exact` only against same-split copies with
  per-pass coverage >= 0.9; the flag is checked by `G2.split_control` in
  every run. Needs the user's confirmation.
- For final acceptance, next to the split rule: the untimed warmup request
  (dc70805, `passes._warmup`) is a measurement change in the lenient
  direction. It warms every arm equally, but the connector's one-time
  save-path cost (first save, gRPC setup) now lands outside
  `G4.replay_cold` and `G4.gsm8k_cold`. It is reported per arm vs baseline
  as `G4.warmup_seconds` (not gated). Needs the user's confirmation.
- Capacity-blind gates (raised 2026-10-23, user asked how the loop would
  find "manage SWA and full blocks separately" for DeepSeek-V4.1-Flash):
  every gate runs with an unlimited pool (`G3.no_eviction`), and write
  amplification / saved-but-never-loaded bytes are listed in GATES.md but not
  computed. So a policy that saves only the trailing window of SWA groups, or
  evicts them first, scores the same as saving everything. Proposal: G5
  capacity gate (fixed pool sized to a fraction of the trace's unique KV bytes,
  oracle-at-capacity as the bar), per-group write-amp / unused-bytes metrics in
  `oracle.py`, and an "optimization hypotheses" checklist in the skill.
  DSV4.1 under vLLM 0.32: kv-source layers `MLAAttentionSpec`
  (`tokens_per_state = compress_ratio`), SWA part `SlidingWindowMLASpec`
  (`DeepseekV4SWACache`, block 64), compressor `CircularBufferSpec`
  (one ring block per request, `prefix_cacheable=False`); the oracle has no
  Kind for `CircularBufferSpec` yet (fails as unknown). Not in the four-model
  scope and no weights downloaded.
- Server-side seal_plan (replace slot counting) and layout-fingerprint
  namespace in the server: not needed for TP1 onboarding; schedule after P6
  or when a TP>1 gate is added.
