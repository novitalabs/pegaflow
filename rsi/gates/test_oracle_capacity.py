"""G5 capacity oracle and verdict contract (no GPU). Needs ``pytest``,
``requests`` and a built ``pegaflow.semantics``; the command is in
docs/rsi/GATES.md ("Tests"). The isolation tests run only on Linux as root.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.gates import oracle, run_gate

BT = 4  # tokens per block, every group
PAGE = 500  # bytes per layer page
LAYERS = 2  # -> 1000 bytes per (group, key) block


def _group(gid: int, cls: str, mro: list[str], bt: int = BT, page: int = PAGE) -> dict:
    return {
        "id": gid,
        "num_layers": LAYERS,
        "enable_kv_transfer": True,
        "is_eagle_group": False,
        "spec": {
            "class": cls,
            "mro": mro,
            "block_size": bt,
            "page_size_bytes": page,
            "real_page_size_bytes": page,
            "sliding_window": None,
        },
    }


FULL = _group(0, "FullAttentionSpec", ["FullAttentionSpec", "AttentionSpec", "KVCacheSpec"])
CKPT = _group(1, "MambaSpec", ["MambaSpec", "KVCacheSpec"])


def _spec(*groups: dict) -> dict:
    return {"parallel": {"tp": 1}, "kv_cache_config": {"groups": list(groups), "num_blocks": 4}}


def _req(prompt: str, n_blocks: int, step: int, label: str, conv: int = 0) -> dict:
    """One request whose prompt is ``n_blocks`` key units (``BT`` tokens)
    plus one token, computed in one step per key unit (so checkpoint groups
    store every block end too)."""
    plen = n_blocks * BT + 1
    return {
        "pass": label,
        "conv": conv,
        "turn": 0,
        "prompt_len": plen,
        "computed": plen,
        "first_step": step,
        "local_hit": 0,
        "first_computed": 0,
        "progress": [[step, e] for e in range(BT, plen + 1, BT)],
        "key_tokens": BT,
        "keys": [f"{prompt}:{j}" for j in range(n_blocks)],
    }


def _trace(prompts: list[str], n_blocks: int, passes: list[str]) -> list[dict]:
    reqs, step = [], 1
    for label in passes:
        for c, p in enumerate(prompts):
            reqs.append(_req(p, n_blocks, step, label, c))
            step += 2  # visible to the next request
    return reqs


def _by_pass(reqs: list[dict], hits: list[int], label: str) -> int:
    return sum(h for r, h in zip(reqs, hits, strict=True) if r["pass"] == label)


# Cyclic scan over 3 prompts x 4 blocks (12 kB) with room for 8 blocks.
CYCLIC = (_spec(FULL), _trace(["p0", "p1", "p2"], 4, ["a", "b", "c"]))
# Full + checkpoint: 2 prompts x (2 full + 2 checkpoint) blocks.
HYBRID = (_spec(FULL, CKPT), _trace(["p0", "p1"], 2, ["a", "b"]))
# Mixed block sizes: full 4 tokens / 1000 B, checkpoint 8 tokens / 3000 B.
CKPT8 = _group(1, "MambaSpec", ["MambaSpec", "KVCacheSpec"], bt=2 * BT, page=3 * PAGE)
MIXED = (_spec(FULL, CKPT8), _trace(["p0", "p1"], 4, ["a", "b"]))


@pytest.mark.parametrize(
    "case", [CYCLIC, HYBRID, MIXED], ids=["cyclic-full", "full+checkpoint", "mixed-block-sizes"]
)
@pytest.mark.parametrize("policy", ["belady", "lru"])
def test_capacity_at_least_u_equals_unlimited_oracle(case, policy):
    spec, reqs = case
    u = oracle.unique_bytes(spec, reqs)
    hits, diag = oracle.simulate(spec, reqs, u, policy)
    assert hits == oracle.replay(spec, reqs)
    assert all(g["evicted"] == 0 for g in diag["groups"])


@pytest.mark.parametrize(
    "case,sizes,unique",
    [
        # 2 prompts x (2 full + 2 checkpoint) blocks of 1000 B.
        (HYBRID, [1000, 1000], 8 * 1000),
        # 2 prompts x (4 full x 1000 B + 2 checkpoint x 3000 B).
        (MIXED, [1000, 3000], 2 * (4 * 1000 + 2 * 3000)),
    ],
    ids=["same-size", "mixed-block-sizes"],
)
def test_unique_bytes_counts_distinct_group_keys(case, sizes, unique):
    spec, reqs = case
    assert oracle.block_bytes(spec) == sizes
    assert oracle.unique_bytes(spec, reqs) == unique
    assert oracle.unique_bytes(spec, reqs, {"a"}) == unique  # pass b adds no new keys


def test_belady_beats_lru_on_cyclic_scan():
    spec, reqs = CYCLIC
    cap = 8 * PAGE * LAYERS
    unlimited = oracle.replay(spec, reqs)
    belady, _ = oracle.simulate(spec, reqs, cap, "belady", unlimited)
    lru, _ = oracle.simulate(spec, reqs, cap, "lru", unlimited)
    for label in ("b", "c"):
        assert _by_pass(reqs, lru, label) == 0  # LRU thrashes a scan larger than C
        assert _by_pass(reqs, lru, label) < _by_pass(reqs, belady, label)
        assert _by_pass(reqs, belady, label) <= _by_pass(reqs, unlimited, label)
    assert all(b <= o for b, o in zip(belady, unlimited, strict=True))


def test_lru_strands_checkpoints_belady_keeps_coherent_prefixes():
    """C = 6 blocks. LRU stores all 8 and evicts p0's full blocks first (they
    were saved before its checkpoints), stranding p0's checkpoints. Belady
    skips checkpoint block 0 (no resume point needs it) and keeps both
    prompts' resumable sets (2 full + last checkpoint each)."""
    spec, reqs = HYBRID
    cap = 6 * PAGE * LAYERS
    unlimited = oracle.replay(spec, reqs)
    belady, bdiag = oracle.simulate(spec, reqs, cap, "belady", unlimited)
    lru, ldiag = oracle.simulate(spec, reqs, cap, "lru", unlimited)
    assert ldiag["passes"]["a"]["stranded_bytes"][1] > 0
    assert bdiag["passes"]["a"]["stranded_bytes"] == [0, 0]
    assert _by_pass(reqs, belady, "b") == _by_pass(reqs, unlimited, "b") > 0
    assert _by_pass(reqs, lru, "b") < _by_pass(reqs, belady, "b")


def test_summarize_capacity_caps_ext_by_unlimited_oracle():
    """A connector claiming more than the unlimited store could serve gets no
    E_C credit for it, and is counted as excess."""
    spec, reqs = CYCLIC
    reqs = [dict(r) for r in reqs]
    for r in reqs:
        r["first_computed"] = r["prompt_len"] - 1  # claims everything, even in pass a
    out = oracle.summarize_capacity(spec, reqs, 8 * PAGE * LAYERS, {"a": {"a"}, "b": {"b"}})
    assert out["a"]["ext_tokens"] == 0 and out["a"]["excess_tokens"] > 0
    b = out["b"]
    assert b["ext_tokens"] == b["oracle_tokens"]
    assert b["efficiency_c"] >= 1.0
    assert b["lru_hit_rate"] == 0.0 < b["oracle_c_hit_rate"] <= b["oracle_hit_rate"]


def test_local_hit_blocks_count_as_present_and_are_not_external():
    """vLLM's local hit holds every group's ``needed`` range at that length:
    resolve may use it, and only the rest is an external load."""
    spec, reqs = HYBRID
    kinds = oracle._active(spec)
    r = reqs[0]  # 2 key units, prompt 9 tokens
    assert oracle._resolve(kinds, r, {}, 0) == 0
    assert oracle._resolve(kinds, r, {}, BT) == BT
    # Resuming at 2 blocks with 1 local: full block 1 and checkpoint block 1.
    ext = oracle._external_keys(kinds, r, 2 * BT, BT)
    assert ext == [(0, "p0:1"), (1, "p0:1")]
    resident = dict.fromkeys(ext, 0)
    assert oracle._resolve(kinds, r, resident, BT) == 2 * BT


def test_reference_local_hit_fixes_the_bar():
    """E_C subtracts the baseline's local hit: an arm that loses its own
    local hits (and reloads them externally) gains nothing."""
    spec, reqs = CYCLIC
    cap = 8 * PAGE * LAYERS
    honest, lazy = [dict(r) for r in reqs], [dict(r) for r in reqs]
    for h, z in zip(honest, lazy, strict=True):
        if h["pass"] != "a":
            h["local_hit"], h["first_computed"] = BT, 3 * BT  # vLLM kept block 0
            z["local_hit"], z["first_computed"] = 0, 3 * BT  # dropped it, loads it
    ref = [r["local_hit"] for r in honest]
    seg = {"b": {"b"}}
    a = oracle.summarize_capacity(spec, honest, cap, seg, ref)["b"]
    b = oracle.summarize_capacity(spec, lazy, cap, seg, ref)["b"]
    assert a["efficiency_c"] == b["efficiency_c"]
    assert a["unmatched_requests"] == b["unmatched_requests"] == 0
    own = oracle.summarize_capacity(spec, lazy, cap, seg)["b"]
    assert own["unmatched_requests"] == own["requests"]  # no reference given
    # Per-arm local (diagnostic) would reward dropping local hits.
    assert own["efficiency_c_own_local"] >= a["efficiency_c"]


def test_cap_order_rotates_deterministically():
    n = 8
    orders = [run_gate.cap_order(x, n) for x in run_gate.cap_names(4)]
    assert orders[0] == list(range(n))
    assert [o[0] for o in orders] == [0, 2, 4, 6, 0]
    assert all(sorted(o) == list(range(n)) for o in orders)


def _digest_ok() -> dict:
    keys = ("local_mismatched", "external_mismatched", "local_unreferenced",
            "external_unreferenced", "external_compared", "external_uncertain",
            "computed_exact", "computed_compared", "local_compared", "local_matched",
            "local_uncertain", "local_uncertain_preempted",
            "external_uncertain_preempted")  # fmt: skip
    return {**dict.fromkeys(keys, 0), "external_matched": 1, "invariant": False, "groups": {}}


def _cap_result(pool: int | None, loaded: int) -> dict:
    """A cap stage result built from the real oracle on CYCLIC, relabelled."""
    spec, reqs = CYCLIC
    names = {"a": "cap_cold", "b": "cap_warm", "c": "cap_warm2"}
    reqs = [r | {"pass": names[r["pass"]]} for r in reqs]
    for r in reqs:
        if r["pass"] != "cap_cold":
            r["first_computed"] = r["prompt_len"] - 1
    cap = 8 * PAGE * LAYERS
    segs = {"warm_timed": {"cap_warm2"}} | {n: {n} for n in names.values()}
    loads = {n: oracle.external_load_bytes(spec, reqs, {n}) for n in names.values()}
    metrics = {'pegaflow_load_bytes_total{otel_scope_name="x"}': loaded}
    return {
        "capacity_bytes": cap,
        "capacity_fraction": 0.5,
        "unique_bytes": 2 * cap,
        "pool_capacity_bytes": pool,
        "capacity": oracle.summarize_capacity(spec, reqs, cap, segs, [0] * len(reqs)),
        "external_load_bytes": loads,
        "load_errors": [],
        "digest_self": _digest_ok(),
        "passes": {
            "cap_cold": {},
            "cap_warm": {},
            "cap_warm2": {
                "pegaflow_metrics": metrics,
                "pegaflow_end": {"pegaflow_cache_resident_bytes": 1.0},
            },
        },
    }


def test_capacity_verdict_requires_pool_and_load_proof():
    good = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    need = good["external_load_bytes"]["cap_warm2"]
    assert need > 0
    v = run_gate._capacity_verdict(good, pegaflow=True, gated=True)
    assert v["pass"] and v["loads_ok"] and v["external_hit_bytes"] == need
    assert v["server_io_bytes"]["cap_warm2"]["loaded_bytes"] == 10**9
    assert v["server_end_bytes"] == {"cap_warm2": {"pegaflow_cache_resident_bytes": 1.0}}
    no_pool = _cap_result(None, loaded=10**9)
    assert not run_gate._capacity_verdict(no_pool, pegaflow=True, gated=True)["pass"]
    assert run_gate._capacity_verdict(no_pool, pegaflow=False, gated=True)["pass"]
    too_big = _cap_result(8 * PAGE * LAYERS + 1, loaded=10**9)
    assert not run_gate._capacity_verdict(too_big, pegaflow=True, gated=True)["pass"]
    short = _cap_result(8 * PAGE * LAYERS, loaded=need - 1)  # hits not from the store
    assert not run_gate._capacity_verdict(short, pegaflow=True, gated=True)["loads_ok"]


@pytest.mark.parametrize(
    "drop",
    [
        "capacity.warm_timed.efficiency_c_own_local",  # pre-20e6b58 records
        "capacity.warm_timed.excess_tokens",
        "capacity.diagnostics.lru",
        "capacity_bytes",
        "load_errors",
    ],
)
def test_capacity_record_without_a_field_fails_not_crashes(drop):
    """S13: a cap record lacking a field the verdict reads fails with
    "cannot evaluate: missing <field>" instead of raising (which would lose
    every other verdict of the run, and its summary.json)."""
    r = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    *path, key = drop.split(".")
    d = r
    for k in path:
        d = d[k]
    del d[key]
    v = run_gate._capacity_verdict(r, pegaflow=True, gated=True)
    assert v["pass"] is False and v["reason"].startswith("cannot evaluate: missing "), v
    assert drop in v["reason"]


def _g3_result(e: float | None, oracle_rate: float, excess: int, errors: int) -> dict:
    warm = {"efficiency": e, "oracle_hit_rate": oracle_rate, "excess_tokens": excess,
            "excess_limit": 16, "external_hit_rate": 0.5, "hit_rate": 0.6,
            "local_hit_tokens": 10, "prompt_tokens": 100}  # fmt: skip
    return {"hit": {"warm": warm}, "load_errors": [{}] * errors}


@pytest.mark.parametrize(
    "e,oracle_rate,excess,errors,ok,below",
    [
        (1.0, 0.99, 0, 0, True, []),
        (0.5, 0.99, 0, 0, True, ["efficiency"]),  # low E_ext: reported, not gated
        (None, 0.2, 0, 0, True, ["efficiency", "oracle_hit_rate"]),
        (1.0, 0.99, 17, 0, False, []),  # excess beyond one block per request
        (1.0, 0.99, 0, 1, False, []),  # load errors
    ],
    ids=["at-bar", "low-e", "no-e-low-oracle", "excess", "load-errors"],
)
def test_g3_hit_level_is_reported_correctness_is_gated(e, oracle_rate, excess, errors, ok, below):
    """User, 2026-10-24: no hard hit-rate threshold. E_ext and the oracle hit
    rate are reported with their reference; excess and load errors still fail."""
    v = run_gate._hit_verdict(_g3_result(e, oracle_rate, excess, errors), "replay")
    assert v["pass"] is ok
    assert [k for k, x in v["reported"].items() if x["below"]] == below
    assert v["efficiency"] == e and v["oracle_hit_rate"] == oracle_rate


def test_g5_low_e_c_is_reported_not_gated():
    """A capacity record whose E_C is far below 0.9 passes G5 on correctness
    and validity; the number and its reference stay in the verdict and in the
    run's hit report."""
    r = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    r["capacity"]["warm_timed"]["efficiency_c"] = 0.3
    v = run_gate._capacity_verdict(r, pegaflow=True, gated=True)
    assert v["pass"] and v["efficiency_c"] == 0.3
    assert v["reported"]["efficiency_c"] == {"value": 0.3, "reference": 0.9, "below": True}
    r["load_errors"] = [{}]
    assert not run_gate._capacity_verdict(r, pegaflow=True, gated=True)["pass"]
    rep = run_gate.hit_report({"baseline": {}, "X": {"G5.replay_warm": v}})
    assert rep["X"]["status"] == "reported"
    assert rep["X"]["G5.replay_warm"]["efficiency_c"] == 0.3
    assert rep["X"]["G5.replay_warm"]["below_reference"] == ["efficiency_c"]


def test_verdicts_reference_arms_not_required_and_errors_isolated(tmp_path):
    good = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    good["digest_self"] |= {"external_compared": 1}
    results = {
        "baseline": {},
        "old": {"gsm8k": {"arm": "old", "workload": "gsm8k", "error": "RuntimeError: boom"}},
        "native": {"cap": good},
        "lmcache": {"cap": good},
        "UnderTest": {"cap": good, "cap025": good},
    }
    v = run_gate.verdicts(results, tmp_path, tmp_path / "spec.json")
    assert v["old"]["required"] is False and v["old"]["errors"]
    for ref in ("native", "lmcache"):  # in-vLLM CPU stores: no server to prove
        assert v[ref]["required"] is False
        assert v[ref]["G3.no_eviction"]["na"]
        assert v[ref]["G5.replay_warm"]["pool_ok"]
    assert v["UnderTest"].get("required", True)
    assert v["UnderTest"]["G5.replay_warm"]["pass"]
    assert "cap025" in v["UnderTest"]["G5.curve"]


@pytest.mark.parametrize("cpu_bytes", [12_919_701_504, 2_404_122_624, 171_798_691_840, 1])
def test_lmcache_cpu_size_is_exact_bytes(cpu_bytes):
    """LMCache takes GiB and allocates ``int(GiB * 1024**3)``: the arm must get
    exactly the PegaFlow pool / C bytes, local CPU only."""
    env = run_gate.lmcache_env(cpu_bytes)
    assert int(float(env["LMCACHE_MAX_LOCAL_CPU_SIZE"]) * 1024**3) == cpu_bytes
    assert env["LMCACHE_LOCAL_CPU"] == "True" and env["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "0"
    assert not any(k in env for k in ("LMCACHE_LOCAL_DISK", "LMCACHE_REMOTE_URL"))


@pytest.mark.parametrize(
    "external,vacuous_ok,ok",
    [
        ({}, True, True),  # cap: nothing loaded in the sample is not a KV error
        ({}, False, False),  # replay: some external load must be judged
        ({"external_compared": 1, "external_mismatched": 1}, True, False),
        ({"external_uncertain": 1}, True, False),  # loaded but never judged
    ],
    ids=["cap-vacuous", "replay-vacuous", "cap-mismatch", "cap-unjudged"],
)
def test_digest_verdict_vacuous_sample(external, vacuous_ok, ok):
    d = _digest_ok() | {"external_matched": 0} | external
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=vacuous_ok)
    assert v["pass"] is ok
    assert ("vacuous" in v) is (ok and vacuous_ok)


# --------------------------------------------- G2.digest_self.cap (exhaustive)


def _probe(tmp_path, reqs: list[dict], digests: list[dict]):
    import json

    d = tmp_path / "probe"
    d.mkdir()
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (d / "digest-pp0-tp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in digests))
    return d


def _cap_probe(tmp_path, corrupt: int, timed_untagged: bool):
    """Cap self-trace: cold computes 4 convs x 4 blocks, warm loads them all
    (``corrupt`` of the 16 external blocks carry wrong bytes); with
    ``timed_untagged`` a third pass recomputes every key undigested, the
    pre-fix cap sampling that made the reference non-exhaustive."""
    reqs, digests, step = [], [], 1

    def block(c: int, j: int, bad: bool = False) -> dict:
        return {"key": f"c{c}:{j}", "covered": (j + 1) * BT, "exact": ["bad" if bad else f"d{c}{j}"], "l1": [1.0]}  # fmt: skip

    passes = [("cap_cold", "full"), ("cap_warm", "full")]
    if timed_untagged:
        passes.append(("cap_warm2", None))
    n = 0
    for name, tag in passes:
        for c in range(4):
            rid = f"{name}-{c}"
            r = _req(f"c{c}", 4, step, name, c) | {"req_id": rid, "digest": tag}
            if name == "cap_warm":
                r["episodes"] = [[step, 0, 4 * BT]]  # all 4 blocks loaded
                r["progress"] = [[step, 4 * BT + 1]]
            else:
                r["episodes"] = [[step, 0, 0]]
            r["keys"] = [f"c{c}:{j}" for j in range(4)]
            reqs.append(r)
            if tag:
                blocks = []
                for j in range(4):
                    bad = name == "cap_warm" and n < corrupt
                    n += name == "cap_warm"
                    blocks.append(block(c, j, bad))
                digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": blocks})  # fmt: skip
            step += 2
    return _probe(tmp_path, reqs, digests)


@pytest.mark.parametrize(
    "corrupt,timed_untagged,ok",
    [(0, False, True), (8, False, False), (1, False, False), (8, True, False)],
    ids=["clean", "half-corrupt", "one-corrupt", "half-corrupt-untagged-later-pass"],
)
def test_digest_self_cap_catches_partially_corrupt_loads(tmp_path, corrupt, timed_untagged, ok):
    """Some external blocks match, others are wrong: the cap verdict must fail
    (the reviewer's 79/158 corruption passed when undigested later passes
    made the reference non-exhaustive and wrong blocks went ``uncertain``)."""
    from rsi.gates import digest_compare

    probe = _cap_probe(tmp_path, corrupt, timed_untagged)
    d = digest_compare.compare(probe, probe, invariant=False)
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=False, exhaustive=True)
    assert v["pass"] is ok, d
    if corrupt and not timed_untagged:
        assert d["external_mismatched"] == corrupt


@pytest.mark.parametrize(
    "uncertain,preempted,loaded,ok",
    [(0, 0, 100, True), (1, 1, 100, True), (2, 2, 100, False), (1, 0, 100, False)],
    ids=["judged", "one-preempted", "over-allowance", "untagged-uncertain"],
)
def test_digest_self_cap_exhaustive_allowance(uncertain, preempted, loaded, ok):
    d = _digest_ok() | {
        "external_compared": loaded - uncertain,
        "external_matched": loaded - uncertain,
        "external_uncertain": uncertain,
        "external_uncertain_preempted": preempted,
    }
    v = run_gate._digest_verdict(d, 0.0, vacuous_ok=False, exhaustive=True)
    assert v["pass"] is ok
    assert v.get("preempted_uncertain", 0) == preempted


def test_vacuous_cap_digest_only_for_reference_arms(tmp_path):
    """No external block judged: a reference arm passes flagged vacuous, the
    arm under test fails (a passing G5 needs loads, and they must be judged)."""
    empty = _cap_result(8 * PAGE * LAYERS, loaded=10**9)
    empty["digest_self"] |= {"external_matched": 0}
    v = run_gate.verdicts({"baseline": {}, "old": {"cap": empty}, "UnderTest": {"cap": empty}}, tmp_path, tmp_path / "spec.json")  # fmt: skip
    assert v["old"]["G2.digest_self.cap"]["pass"] and v["old"]["G2.digest_self.cap"]["vacuous"]
    assert not v["UnderTest"]["G2.digest_self.cap"]["pass"]
    assert not v["UnderTest"]["G5.replay_warm"]["pass"]


# ------------------------------------------------------------ S1 / bytes / floor


def test_own_local_above_reference_earns_no_ext():
    """An arm whose vLLM happens to keep more locally than the baseline did
    (variance) gets no store credit for those tokens."""
    spec, reqs = CYCLIC
    reqs = [dict(r) for r in reqs]
    for r in reqs:
        if r["pass"] != "a":
            r["local_hit"] = r["first_computed"] = 2 * BT  # all local, no store load
    out = oracle.summarize_capacity(spec, reqs, 8 * PAGE * LAYERS, {"b": {"b"}}, [0] * len(reqs))
    assert out["b"]["ext_tokens"] == 0 and out["b"]["efficiency_c"] == 0.0


@pytest.mark.parametrize(
    "tp,replicas,copies", [(1, 1, 1), (2, 1, 2), (2, 2, 1), (4, 2, 2)], ids=str
)
def test_block_and_gpu_kv_bytes_count_distinct_tp_copies(tp, replicas, copies):
    spec = _spec(FULL) | {"parallel": {"tp": tp}}
    spec["kv_cache_config"] |= {"kv_tp_replicas": replicas, "tensors": [{"size": 7000}, {"size": 3000, "host_resident": True}]}  # fmt: skip
    assert oracle.block_bytes(spec) == [PAGE * LAYERS * copies]
    assert run_gate.gpu_kv_bytes(spec) == 7000 * copies


def test_gpu_kv_bytes_counts_shared_tensors_once():
    """Hybrid groups sharing one tensor: the allocation counts once (the old
    num_blocks x Σ group block bytes formula counted it per group)."""
    spec = _spec(FULL, CKPT)
    spec["kv_cache_config"]["tensors"] = [{"size": 4 * PAGE * LAYERS}]
    assert run_gate.gpu_kv_bytes(spec) == 4 * PAGE * LAYERS
    assert spec["kv_cache_config"]["num_blocks"] * sum(oracle.block_bytes(spec)) == 2 * run_gate.gpu_kv_bytes(spec)  # fmt: skip


def test_capacity_floor_fails_closed_and_g5_problems():
    assert not run_gate.capacity_floor(None)["pass"]
    assert not run_gate.capacity_floor({"gpu_kv_bytes": 1})["pass"]
    low = run_gate.capacity_floor({"unique_bytes": 3, "gpu_kv_bytes": 1})
    high = run_gate.capacity_floor({"unique_bytes": 4, "gpu_kv_bytes": 1})
    assert not low["pass"] and high["pass"]
    sep = {"G5.replay_warm": {"freeze_vs_oracle_c": 0.7}}
    flat = {"G5.replay_warm": {"freeze_vs_oracle_c": 0.95}}
    assert run_gate.g5_problems({"X": sep, "old": flat}, high, None) == []
    assert run_gate.g5_problems({"X": flat}, high, None) == ["G5 trace does not separate freeze from H*_C"]  # fmt: skip
    assert run_gate.g5_problems({"X": {}}, high, None)  # freeze not evaluated: not official
    assert run_gate.g5_problems({"X": sep}, low, {"enough": False}) == ["capacity below 2x GPU KV (whole replay file used)"]  # fmt: skip


def test_reference_locals_unmatched_path():
    ref = run_gate.reference_locals(
        [{"pass": "cap_cold", "conv": 0, "turn": 1, "local_hit": 32},
         {"pass": "cap_warm", "conv": 0, "turn": 1, "local_hit": None},
         {"pass": "cap_warm", "conv": None, "turn": 0, "local_hit": 5}]
    )  # fmt: skip
    assert ref == {"cap_cold/0/1": 32, "cap_warm/0/1": 0}
    reqs = [{"pass": "cap_cold", "conv": 0, "turn": 1}, {"pass": "cap_warm3", "conv": 0, "turn": 1},
            {"pass": "cap_warm2", "conv": 9, "turn": 0}, {"pass": "cap_warm2", "turn": 0}]  # fmt: skip
    assert run_gate.match_locals(reqs, ref) == [32, 0, None, None]
    assert run_gate.match_locals(reqs, None) == [None] * 4


def test_unique_estimate_counts_shared_prefix_once():
    spec = _spec(FULL)
    # vLLM computes prompt + out - 1 tokens: out = BT + 1 completes one block.
    est = run_gate.UniqueEstimate(spec, out_tokens=BT + 1)
    a = list(range(3 * BT))
    est.add([a[: 2 * BT], a])  # turn 2 extends turn 1: 3 distinct blocks + 2 generated
    assert est.bytes == (3 + 2) * PAGE * LAYERS


@pytest.mark.parametrize(
    "over,key,ok",
    [
        pytest.param(
            {"cap_fractions": "0.75,0.5,0.25"}, "cap_fractions", True, id="fractions-reordered"
        ),  # fmt: skip
        pytest.param(
            {"cap_fractions": "0.5,0.25,0.75,0.5"}, "cap_fractions", True, id="fractions-dup"
        ),  # fmt: skip
        pytest.param(
            {"cap_fractions": "0.25,0.5"}, "cap_fractions", False, id="fractions-missing"
        ),  # fmt: skip
        pytest.param({"replay_skip": 0}, "replay_skip", True, id="replay-skip-0"),
        pytest.param({"replay_skip": 32}, "replay_skip=32 (held-out", False, id="held-out"),
        pytest.param(
            {"arm": "pegaflow.connector_v2:C"}, "arm under test", True, id="pegaflow-module"
        ),  # fmt: skip
        pytest.param(
            {"arm": "rsi.gates.myconn:C"},
            "arm under test rsi.gates",
            False,
            id="rsi-module-under-test",
        ),  # fmt: skip
    ],
)
def test_official_args(tmp_path, over, key, ok):
    """cap_fractions compare as sets; ``--replay-skip`` > 0 is a held-out
    slice of the trace, never official; the arm under test's module must be
    in the ``pegaflow`` package (M8: the only one the snapshot holds)."""
    import argparse

    arm = over.pop("arm", "pegaflow.m:C")
    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": "/t/python"} | over))  # fmt: skip
    probs = run_gate.official_problems(args, ["plain", "baseline", arm], list(run_gate.WORKLOADS), tmp_path, tmp_path)  # fmt: skip
    assert any(p.startswith(key) for p in probs) is not ok, probs


def test_stop_waits_for_the_whole_process_group(tmp_path, monkeypatch):
    """The leader exits at once on SIGTERM, a child keeps tearing down for
    2 s (vLLM's EngineCore, a pinned pool after SIGKILL): ``stop`` must
    return only once the group is gone, then log the exit time."""
    import os
    import subprocess
    import time

    monkeypatch.setattr(run_gate, "TERM_GRACE", 30)
    script = "sh -c 'trap \"sleep 2; exit 0\" TERM; while :; do sleep 0.1; done' & wait"
    log = tmp_path / "log"
    with log.open("w") as fh:
        proc = subprocess.Popen(["sh", "-c", script], stdout=fh, start_new_session=True)
        time.sleep(0.5)
        t0 = time.monotonic()
        run_gate.stop(proc, "sh", fh)
        took = time.monotonic() - t0
    assert took >= 1.5
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)
    assert "gone after SIGTERM" in log.read_text()


@pytest.mark.parametrize(
    "arm,exc,fatal",
    [
        ("old", ValueError("crash"), False),
        ("old", run_gate.TeardownError("group alive"), True),
        ("native", "teardown-during-crash", True),
        ("UnderTest", ValueError("crash"), True),
    ],
    ids=["ref-crash-recorded", "ref-teardown-aborts", "ref-teardown-chained", "under-test-crash"],
)
def test_stage_never_swallows_a_teardown_failure(monkeypatch, tmp_path, arm, exc, fatal):
    """S3: a reference arm's crash is recorded, but a surviving process group
    aborts the run (the next stage would start on a live GPU)."""

    def boom(*_a, **_k):
        if exc == "teardown-during-crash":
            try:
                raise ValueError("crash")
            finally:
                raise run_gate.TeardownError("group alive")
        raise exc

    monkeypatch.setattr(run_gate, "run_arm_workload", boom)
    if fatal:
        with pytest.raises((ValueError, run_gate.TeardownError)):
            run_gate.run_stage(None, {}, arm, "gsm8k", tmp_path)
    else:
        assert "error" in run_gate.run_stage(None, {}, arm, "gsm8k", tmp_path)


_GATE_LIKE = """
import sys, time
from pathlib import Path
from rsi.gates import run_gate
run_gate.install_stop_handlers()
port, out = sys.argv[1], Path(sys.argv[2])
cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
with run_gate.launched(cmd, {}, out / "sleeper.log", f"http://127.0.0.1:{port}/", 60) as p:
    (out / "pid").write_text(str(p.pid))
    time.sleep(600)
"""


def test_port_leases_are_exclusive_and_released(tmp_path):
    """Stage ports: leased under per-port flocks, so two gate runs never get
    the same port (single or vLLM's contiguous ``VLLM_PORT`` block); released
    when the stage ends, with the lease files in the record."""
    p = run_gate.free_port()
    a = run_gate.PortLease(1, tmp_path, (p, p))
    with pytest.raises(RuntimeError, match="no 1 free ports"):
        run_gate.PortLease(1, tmp_path, (p, p))
    assert a.record["files"] == [str(tmp_path / str(p))] and a.record["released_at"] is None
    a.release()
    assert a.record["released_at"] is not None
    b = run_gate.PortLease(1, tmp_path, (p, p))
    assert b.ports == a.ports == [p]
    b.release()
    lo = 20000 + p % 5000
    c = run_gate.PortLease(1, tmp_path, (lo, lo + 9), block=4)
    assert len(set(c.block)) == 4 and c.block == list(range(c.block[0], c.block[0] + 4))
    assert c.ports[0] not in c.block
    with pytest.raises(RuntimeError, match="contiguous"):
        run_gate.PortLease(1, tmp_path, (c.block[0], c.block[-1]), block=4)
    c.release()


class _Dead:
    pid = 4242

    def poll(self):
        return 1


@pytest.mark.parametrize(
    "line,ports",
    [
        pytest.param("OSError: [Errno 98] Address already in use", [], id="no-port"),
        pytest.param(
            "OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 33641): address already in use",
            [33641],
            id="uvicorn",
        ),  # fmt: skip
        pytest.param(
            "torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use",
            [29500],
            id="tcpstore",
        ),  # fmt: skip
        pytest.param(
            "zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:20017')",
            [20017],
            id="zmq",
        ),  # fmt: skip
        pytest.param(
            'Error: Os { code: 98, kind: AddrInUse, message: "Address in use" } (os error 98) binding 127.0.0.1:21000',
            [21000],
            id="rust",
        ),  # fmt: skip
    ],
)
def test_any_bind_failure_is_a_port_race(tmp_path, line, ports):
    """EADDRINUSE on any bind in a stage log (the gate's leased port or one
    vLLM / torch / ZMQ / the server picked itself) is a PortRace naming the
    parsed port, whether the gate leased it, and the lease files."""
    p = ports[0] if ports else run_gate.free_port()
    lease = run_gate.PortLease(1, tmp_path / "locks", (p, p))
    log = tmp_path / "vllm.log"
    log.write_text(f"INFO starting\n{line}\n")
    race = run_gate.port_race(log, [lease], _Dead())
    assert isinstance(race, run_gate.PortRace)
    msg = str(race)
    assert (f"{p} (leased)" in msg) if ports else ("unknown" in msg), msg
    assert str(tmp_path / "locks" / str(p)) in msg and "not retried" in msg
    log.write_text("RuntimeError: CUDA out of memory\n")
    assert run_gate.port_race(log, [lease], _Dead()) is None
    lease.release()


def test_port_race_fails_the_stage_naming_the_port(tmp_path, monkeypatch):
    """Another process binds the leased port after the lease, before the
    stage does: the stage fails with ``PortRace`` naming the port (not
    retried), its leases are released, and run_stage writes the cause and
    the lease files to ``stage-error.json``."""
    import json
    import socket

    p = run_gate.free_port()
    locks, out = tmp_path / "locks", tmp_path / "stage"
    cmd = [sys.executable, "-c", f"import socket; socket.socket().bind(('127.0.0.1', {p}))"]
    squatter = socket.socket()
    got: list[run_gate.PortLease] = []

    def stage(args, gate, arm, wl, out, **kw):
        got.append(run_gate.PortLease(1, locks, (p, p)))
        squatter.bind(("127.0.0.1", p))  # the race: taken after the lease
        squatter.listen()
        out.mkdir(parents=True)
        with run_gate.launched(cmd, {}, out / "vllm.log", f"http://127.0.0.1:{p}/x", 30, got[0]):
            pass

    monkeypatch.setattr(run_gate, "run_arm_workload", stage)
    try:
        with pytest.raises(run_gate.PortRace, match=f"{p} \\(leased\\)"):
            run_gate.run_stage(None, {}, "baseline", "gsm8k", out)
    finally:
        squatter.close()
    assert got[0]._fds == [] and got[0].record["released_at"] is not None
    err = json.loads((run_gate.gate_dir(out) / "stage-error.json").read_text())
    assert err["type"] == "PortRace" and f"{p} (leased)" in err["error"]
    assert err["port_leases"][-1]["files"] == [str(locks / str(p))]


def test_leases_are_released_on_every_exit_path(tmp_path, monkeypatch):
    """A stage that raises before it launches, or is stopped by a signal
    (SystemExit), releases every lease it took."""
    p = run_gate.free_port()

    def take_then(exc):
        def stage(args, gate, arm, wl, out, **kw):
            run_gate.PortLease(1, tmp_path, (p, p))
            raise exc

        return stage

    for exc in (ValueError("before launch"), SystemExit(143)):
        monkeypatch.setattr(run_gate, "run_arm_workload", take_then(exc))
        with pytest.raises(type(exc)):
            run_gate.run_stage(None, {}, "baseline", "gsm8k", tmp_path / "s")
        assert run_gate._LEASES[-1]._fds == []
        run_gate.PortLease(1, tmp_path, (p, p)).release()  # free again


def test_sigterm_tears_down_launched_groups(tmp_path):
    """``kill -TERM <gate>`` must stop the stages' process groups (own
    sessions, so the signal never reaches them) before the gate exits."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.Popen([sys.executable, "-c", _GATE_LIKE, port, str(tmp_path)], env=env)
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    sleeper = int(pid_file.read_text())
    os.killpg(sleeper, 0)  # alive, own group
    gate.send_signal(signal.SIGTERM)
    assert gate.wait(timeout=60) == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(sleeper, 0)


_WINDOW_LIKE = """
import os, signal, subprocess, sys, time
from pathlib import Path
from rsi.gates import run_gate
port, out, where = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
real_popen, real_stop = subprocess.Popen, run_gate.stop


def popen(*a, **kw):  # the signal lands right after Popen returns
    p = real_popen(*a, **kw)
    (out / "pid").write_text(str(p.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return p


def stop(proc, name, fh, _first=[True]):  # ... or on entering finally:
    if _first.pop() if _first else False:
        os.kill(os.getpid(), signal.SIGTERM)
    return real_stop(proc, name, fh)


if where == "popen":
    run_gate.subprocess.Popen = popen
else:
    run_gate.stop = stop
run_gate.install_stop_handlers()
cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
with run_gate.launched(cmd, {}, out / "server.log", f"http://127.0.0.1:{port}/", 60) as p:
    (out / "pid").write_text(str(p.pid))
(out / "after").write_text("")
"""


@pytest.mark.parametrize("where", ["popen", "finally"])
def test_stop_signal_in_a_launch_window_still_stops_the_group(tmp_path, where):
    """N5: a stop signal right after ``Popen`` returns, or on entering the
    ``finally:`` before ``stop`` owns the group, still stops the group
    (deferred, or by ``stop_live`` at exit) and the gate exits 128 + sig."""
    import os
    import signal
    import subprocess
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.run(
        [sys.executable, "-c", _WINDOW_LIKE, port, str(tmp_path), where],
        env=env,
        timeout=120,
        capture_output=True,
        text=True,
    )
    assert gate.returncode == 128 + signal.SIGTERM, gate.stdout + gate.stderr
    assert not (tmp_path / "after").exists()
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    if where == "finally":
        assert "left by a signal window" in gate.stdout, gate.stdout


@pytest.mark.parametrize(
    "path_ver,home_ver,problem",
    [
        pytest.param("13.0", "13.0", None, id="both-match"),
        pytest.param(None, "13.0", None, id="home-only"),
        pytest.param("12.2", None, "is CUDA 12.2, torch is CUDA 13.0", id="path-12-attempt-1"),
        pytest.param("13.0", "12.2", "is CUDA 12.2", id="home-mismatch"),
        pytest.param(None, None, "no nvcc", id="none"),
    ],
)
def test_cuda_env_must_match_torch(tmp_path, path_ver, home_ver, problem):
    """GPU3 attempt 1: the system nvcc 12.2 on PATH, torch on CUDA 13, and
    FlashInfer's JIT failed mid-run. Every nvcc a stage can pick must have
    torch's CUDA major; paths and versions are recorded."""

    def nvcc(d, ver):
        b = d / "bin"
        b.mkdir(parents=True)
        f = b / "nvcc"
        f.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {ver}, V{ver}.1'\n")
        f.chmod(0o755)
        return d

    env = {"PATH": "/nonexistent", "LD_LIBRARY_PATH": "/x/lib"}
    if path_ver:
        env["PATH"] = str(nvcc(tmp_path / "p", path_ver) / "bin")
    if home_ver:
        env["CUDA_HOME"] = str(nvcc(tmp_path / "h", home_ver))
    rec = run_gate.cuda_env(env, torch_cuda="13.0")
    assert rec["nvcc_path_version"] == path_ver and rec["nvcc_cuda_home_version"] == home_ver
    assert rec["LD_LIBRARY_PATH"] == "/x/lib"
    got = run_gate.cuda_problems(rec)
    assert (problem in " ".join(got)) if problem else not got, got


# ------------------------------------------------------------ provenance


def _git_tree(tmp_path):
    """A repo with python/pegaflow/{__init__,a,conn}.py committed; returns (repo, rev)."""
    import subprocess

    repo = tmp_path / "src"
    pkg = repo / "python" / "pegaflow"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "conn.py").write_text("print('original')\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "config", "uploadpack.allowFilter", "true"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    rev = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    return repo, rev


def _copy(repo, tmp_path, rev, edit=None, extra=None):
    import shutil

    dst = tmp_path / "copy"
    shutil.copytree(repo / "python", dst / "python")
    (dst / ".rsi-rev").write_text(rev + "\n")
    if edit:
        (dst / "python" / "pegaflow" / "a.py").write_text(edit)
    for name, body in (extra or {}).items():
        (dst / "python" / "pegaflow" / name).write_bytes(body)
    return dst


def _native(pkg):
    from rsi.gates import provenance

    return pkg / Path(provenance.native_module()).name


@pytest.mark.parametrize(
    "case,verified",
    [("git-clean", True), ("git-dirty", False), ("rsi-rev", True), ("rsi-rev-edited", False),
     ("rsi-rev-extra-py", False), ("rsi-rev-native", True), ("unknown-rev", False), ("no-rev", False)],
)  # fmt: skip
def test_tree_rev_is_verified_blob_for_blob(tmp_path, case, verified):
    """M3: a run is official with a tree under test only if its rev is
    verified: a clean git toplevel, or ``.rsi-rev`` whose tracked files match
    on disk (the expected native module is allowed and hashed)."""
    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    if case.startswith("git"):
        root = repo
        if case == "git-dirty":
            (repo / "python" / "pegaflow" / "a.py").write_text("x = 2\n")
    elif case == "no-rev":
        root = _copy(repo, tmp_path, rev)
        (root / ".rsi-rev").unlink()
    else:
        root = _copy(
            repo,
            tmp_path,
            "0" * 40 if case == "unknown-rev" else rev,
            edit="x = 2\n" if case == "rsi-rev-edited" else None,
            extra={"b.py": b""} if case == "rsi-rev-extra-py" else None,
        )
        if case == "rsi-rev-native":
            _native(root / "python" / "pegaflow").write_bytes(b"elf")
    rec = trees.tree(root, ("python/pegaflow",), provenance.native_module())
    assert rec["verified"] is verified, rec
    if case == "rsi-rev-native":
        assert list(rec["native_sha256"]) == [provenance.native_module()]
    if verified:
        assert rec["rev"] == rev


def _plant(case, pkg, tmp_path):
    """M5 bypasses: each makes ``import pegaflow.conn`` run injected code
    from a tree whose tracked blobs are all intact."""
    import importlib.util
    import marshal

    evil = tmp_path / "elsewhere"
    if case == "dir-symlink":
        (evil / "conn").mkdir(parents=True)
        (evil / "conn" / "__init__.py").write_text("print('INJECTED')\n")
        (pkg / "conn").symlink_to(evil / "conn", target_is_directory=True)
    elif case in ("unchecked-pyc", "git-ignored-pyc"):
        code = compile("print('INJECTED')\n", str(pkg / "conn.py"), "exec")
        # PEP 552 unchecked-hash pyc: the importer never compares it with conn.py.
        head = importlib.util.MAGIC_NUMBER + (1).to_bytes(4, "little") + b"\0" * 8
        cache = pkg / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / f"conn.{sys.implementation.cache_tag}.pyc").write_bytes(head + marshal.dumps(code))
    elif case == "extra-so":
        ext = Path(_native(pkg)).name.removeprefix("pegaflow")
        (pkg / f"conn{ext}").write_bytes(b"elf")  # beats conn.py in the loader order
    elif case == "file-symlink":
        evil.mkdir(parents=True)
        (evil / "conn.py").write_text("print('INJECTED')\n")
        (pkg / "conn.py").unlink()
        (pkg / "conn.py").symlink_to(evil / "conn.py")


@pytest.mark.parametrize("case", ["dir-symlink", "unchecked-pyc", "git-ignored-pyc", "extra-so", "file-symlink"])  # fmt: skip
def test_tree_bypasses_are_rejected(tmp_path, case):
    """M5: every file Python could import from a tree is accounted for. A
    symlink, an untracked or ignored ``.pyc`` (even in ``__pycache__``, even
    in a clean git toplevel whose ``status`` hides it) and any native module
    but the expected one reject the tree."""
    import subprocess

    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    if case == "git-ignored-pyc":
        (repo / ".gitignore").write_text("__pycache__/\n*.so\n")
        subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ignore"], check=True)  # fmt: skip
        root = repo
        _plant(case, repo / "python/pegaflow", tmp_path)
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        )
        assert status == ""  # what the old check looked at
    else:
        root = _copy(repo, tmp_path, rev)
        _plant(case, root / "python/pegaflow", tmp_path)
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    rec = trees.tree(root, ("python/pegaflow",), provenance.native_module())
    assert rec["verified"] is False, rec
    staged = provenance.stage(root / "python", tmp_path / "ov", trees)
    assert staged["verified"] is False, staged
    # The snapshot never carries the planted file: importing it runs the
    # tracked code only, with no cache dir in the snapshot.
    snap = Path(staged["pegaflow_dir"])
    assert not any(p.suffix in (".pyc", ".so") or p.is_symlink() for p in snap.rglob("*"))
    if (snap / "conn.py").exists():
        env = {"PYTHONPATH": str(tmp_path / "ov"), "PYTHONPYCACHEPREFIX": str(tmp_path / "pyc")}
        out = subprocess.check_output([sys.executable, "-c", "import pegaflow.conn"], env=env, text=True)  # fmt: skip
        assert out == "original\n"


def test_stage_copies_only_the_verified_rev(tmp_path):
    """M5: the arm imports a snapshot (tracked blobs + the expected native
    module, sha256 equal to the source's), so editing the tree mid-run
    changes nothing it imports, and the snapshot holds only ``pegaflow``."""
    from rsi.gates import provenance

    repo, rev = _git_tree(tmp_path)
    root = _copy(repo, tmp_path, rev)
    for d in ("python/rsi/gates", "python/vllm"):
        (root / d).mkdir(parents=True)
    (root / "python/sitecustomize.py").write_text("")
    _native(root / "python/pegaflow").write_bytes(b"elf")
    trees = provenance.Trees(tmp_path / "work", remote=f"file://{repo}")
    rec = provenance.stage(root / "python", tmp_path / "ov", trees)
    assert rec["verified"], rec
    assert [p.name for p in (tmp_path / "ov").iterdir()] == ["pegaflow"]
    snap = Path(rec["pegaflow_dir"])
    assert sorted(p.name for p in snap.iterdir()) == sorted(["__init__.py", "a.py", "conn.py", _native(snap).name])  # fmt: skip
    assert rec["native_sha256"] == provenance.sha256(_native(root / "python/pegaflow"))
    (root / "python/pegaflow/a.py").write_text("x = 'edited mid-run'\n")
    assert (snap / "a.py").read_text() == "x = 1\n"


@pytest.mark.parametrize(
    "role,edit",
    [
        pytest.param(None, None, id="unchanged"),
        pytest.param("gate", "rsi/gates/rsi_probe.py", id="gate-probe-edited"),
        pytest.param("gate", "rsi/gates/new.py", id="gate-file-added"),
        pytest.param("under_test", "pegaflow/a.py", id="snapshot-edited"),
        pytest.param("pegaflow_server", "pegaflow-server", id="server-binary-replaced"),
    ],
)
def test_a_tree_changed_between_stages_is_named(tmp_path, monkeypatch, role, edit):
    """M10: what was verified before stage 1 is fingerprinted and checked
    again before every stage (and after the last one); a change makes the
    run non-official with the role and the path named. In-tree pyc of the
    gate tree is not part of it (never read)."""
    import argparse
    import sys

    gate = tmp_path / "gate"
    (gate / "rsi/gates").mkdir(parents=True)
    (gate / "rsi/gates/rsi_probe.py").write_text("x = 1\n")
    (gate / "python/pegaflow").mkdir(parents=True)
    (gate / "python/pegaflow/__init__.py").write_text("")
    ov = tmp_path / "ov"
    (ov / "pegaflow").mkdir(parents=True)
    (ov / "pegaflow/a.py").write_text("y = 1\n")
    srv = tmp_path / "bin"
    srv.mkdir()
    (srv / "pegaflow-server").write_bytes(b"elf")
    monkeypatch.setattr(run_gate, "REPO", gate)
    monkeypatch.setattr(run_gate.provenance, "gate_pegaflow_dir", lambda: None)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "pyc"))
    args = argparse.Namespace(under_test_path="/t", under_test_overlay=str(ov), pegaflow_server=str(srv / "pegaflow-server"))  # fmt: skip
    watch = run_gate.ProvenanceWatch(args, gate / "docs/rsi/models/M")
    (gate / "rsi/gates/__pycache__").mkdir()
    (gate / "rsi/gates/__pycache__/rsi_probe.cpython-312.pyc").write_bytes(b"pyc")
    assert watch.check("baseline/gsm8k") == {}
    if edit:
        base = {"gate": gate, "under_test": ov, "pegaflow_server": srv}[role]
        (base / edit).write_text("tampered\n")
    changes = watch.check("ut/replay")
    probs = watch.problems()
    if role is None:
        assert changes == {} and probs == []
    else:
        assert list(changes) == [role], changes
        assert len(probs) == 1 and "before ut/replay" in probs[0] and Path(edit).name in probs[0], probs  # fmt: skip
    assert [c["before"] for c in watch.record()["checks"]] == ["baseline/gsm8k", "ut/replay"]


def test_each_stage_gets_a_fresh_bytecode_cache_outside_the_arm(tmp_path):
    """M10: a stage's PYTHONPYCACHEPREFIX is its own dir, outside every arm
    dir; the isolated arm gets only its current stage's cache bound rw."""
    import argparse

    run = tmp_path / "runs" / "r"
    env: dict[str, str] = {}
    run_gate.pycache_env(env, run / "ut" / "replay" / "vllm.log")
    assert env["PYTHONPYCACHEPREFIX"] == str(run / ".pycache-stages" / "ut" / "replay")
    assert not Path(env["PYTHONPYCACHEPREFIX"]).is_relative_to(run / "ut")
    args = argparse.Namespace(
        runs_root=str(tmp_path / "runs"), under_test_path="", under_test_overlay=""
    )
    iso = run_gate.isolation(args, run / "ut" / "replay")
    assert iso["rw"] == [str((run / "ut" / "replay").resolve()), str((run / ".pycache-stages" / "ut" / "replay").resolve())]  # fmt: skip


@pytest.mark.parametrize("bad", [None, "gate", "venv_pegaflow", "under_test"])
def test_every_imported_tree_must_be_verified(bad):
    """M6: an unverified gate tree (the judge) or venv ``pegaflow`` (the
    oracle's semantics) makes the run non-official, as the tree under test
    does."""
    prov = {
        "gate": {"root": "/g", "verified": True},
        "venv_pegaflow": {"root": "/v", "verified": True},
        "under_test": {"source": {"root": "/u"}, "verified": True},
    }
    if bad:
        prov[bad] = prov[bad] | {"verified": False, "reason": "tree differs from rev"}
    problems = run_gate.provenance_problems(prov)
    assert len(problems) == (bad is not None), problems
    assert run_gate.provenance_problems({"gate": {"verified": True}}) == [
        "venv pegaflow (oracle semantics, reference arms): provenance not recorded"
    ]


_needs_isolation = pytest.mark.skipif(
    not run_gate.isolation_available(), reason="isolate.py needs Linux and root"
)

_PROBE_ESCAPE = """
import os, sys
base, arm, gate_pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = {}
seen["baseline_dir"] = os.path.exists(f"{base}/baseline/gsm8k/warm-outputs.json")
seen["own_dir"] = os.path.exists(f"{arm}/gsm8k/mine.txt")
seen["earlier_stage"] = os.path.exists(f"{arm}/replay/digest.jsonl")
seen["gate_dir"] = os.path.exists(f"{base}/.gate/ut/gsm8k/warm-outputs.json")
open(f"{arm}/gsm8k/written.txt", "w").write("ok")
for how in ("root", "cwd"):
    try:
        seen[how] = os.path.exists(f"/proc/{gate_pid}/{how}{base}/baseline/gsm8k/warm-outputs.json")
    except OSError:
        seen[how] = False
try:
    fd = os.open(f"/proc/{gate_pid}/ns/mnt", os.O_RDONLY)
    os.setns(fd, 0)
    seen["setns"] = True
except OSError:
    seen["setns"] = False
try:
    import ctypes
    seen["umount"] = ctypes.CDLL(None, use_errno=True).umount2(base.encode(), 2) == 0
except Exception:
    seen["umount"] = False
print(repr(seen))
"""


@_needs_isolation
def test_isolated_arm_cannot_read_other_arms(tmp_path):
    """M7(a), M9: the arm under test sees its current stage dir (rw) and
    not the baseline's outputs, its own earlier stages or the gate dir,
    neither by path nor through ``/proc/<gate>/root`` or ``cwd``, and cannot
    unmount the cover."""
    import argparse
    import os
    import subprocess

    runs = tmp_path / "runs"
    base = runs / "2026-10-24-t"
    (base / "baseline" / "gsm8k").mkdir(parents=True)
    (base / "baseline" / "gsm8k" / "warm-outputs.json").write_text("[]")
    arm = base / "ut"
    (arm / "gsm8k").mkdir(parents=True)
    (arm / "gsm8k" / "mine.txt").write_text("")
    (arm / "replay").mkdir()
    (arm / "replay" / "digest.jsonl").write_text("")
    (base / ".gate" / "ut" / "gsm8k").mkdir(parents=True)
    (base / ".gate" / "ut" / "gsm8k" / "warm-outputs.json").write_text("[]")
    args = argparse.Namespace(runs_root=str(runs), under_test_path="", under_test_overlay="")
    cmd = run_gate.isolated(args, "m:C", arm / "gsm8k" / "vllm.log", [sys.executable, "-c", _PROBE_ESCAPE, str(base), str(arm), str(os.getpid())])  # fmt: skip
    assert cmd[0] == sys.executable and "--hide" in cmd
    out = subprocess.check_output(cmd, text=True, timeout=60)
    seen = eval(out)  # noqa: S307 - our own child's repr
    assert seen == {"baseline_dir": False, "own_dir": True, "earlier_stage": False, "gate_dir": False, "root": False, "cwd": False, "setns": False, "umount": False}, seen  # fmt: skip
    assert (arm / "gsm8k" / "written.txt").read_text() == "ok"  # rw bind reaches the host dir
    assert os.path.exists(base / "baseline" / "gsm8k" / "warm-outputs.json")  # host unchanged


_ISOLATED_GATE = """
import sys, time, argparse
from pathlib import Path
from rsi.gates import run_gate
run_gate.install_stop_handlers()
port, runs = sys.argv[1], Path(sys.argv[2])
arm = runs / "r" / "ut" / "gsm8k"
arm.mkdir(parents=True, exist_ok=True)
args = argparse.Namespace(runs_root=str(runs), under_test_path="", under_test_overlay="")
# A child that forks a grandchild, as vLLM forks EngineCore / workers, and
# one that setsid()s out of the group (S11); its pid goes to the arm's dir.
inner = "import subprocess, sys, http.server; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); s = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], start_new_session=True); open(sys.argv[2], 'w').write(str(s.pid)); http.server.test(http.server.SimpleHTTPRequestHandler, port=int(sys.argv[1]), bind='127.0.0.1')"
cmd = run_gate.isolated(args, "m:C", arm / "vllm.log", [sys.executable, "-c", inner, port, str(arm / "setsid.pid")])
with run_gate.launched(cmd, {}, arm / "vllm.log", f"http://127.0.0.1:{port}/", 60) as p:
    (runs / "pid").write_text(str(p.pid))
    time.sleep(600)
"""


@_needs_isolation
def test_isolated_arm_is_torn_down_with_its_group(tmp_path):
    """M7(a): the isolated arm is exec'd in place, so it is the group
    leader ``launched`` started; its health endpoint is reachable on
    loopback, and SIGTERM to the gate stops every member of the group (the
    grandchild included) before the gate exits. S11: a child that setsid()'d
    out of the group is stopped too (it is in the arm's mount namespace)."""
    import os
    import signal
    import subprocess
    import time

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())
    gate = subprocess.Popen([sys.executable, "-c", _ISOLATED_GATE, port, str(tmp_path)], env=env)
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    leader = int(pid_file.read_text())
    assert os.readlink(f"/proc/{leader}/ns/mnt") != os.readlink("/proc/self/ns/mnt")
    assert os.readlink(f"/proc/{leader}/ns/pid") == os.readlink("/proc/self/ns/pid")
    members = [p for p in os.listdir("/proc") if p.isdigit() and _pgid(int(p)) == leader]
    assert len(members) >= 2, members  # leader + grandchild
    escaped = int((tmp_path / "r" / "ut" / "gsm8k" / "setsid.pid").read_text())
    assert _pgid(escaped) == escaped != leader  # out of the group
    assert os.readlink(f"/proc/{escaped}/ns/mnt") == os.readlink(f"/proc/{leader}/ns/mnt")
    gate.send_signal(signal.SIGTERM)
    assert gate.wait(timeout=120) == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.killpg(leader, 0)
    assert _dead(escaped), "setsid'd child survived teardown"


def _env_with_path(root: Path) -> dict[str, str]:
    """The test's env with the repo root prepended to ``PYTHONPATH`` (kept,
    not replaced: it may be what finds the built ``pegaflow``)."""
    import os

    path = [str(root), *filter(None, [os.environ.get("PYTHONPATH")])]
    return os.environ | {"PYTHONPATH": os.pathsep.join(path)}


def _dead(pid: int) -> bool:
    """Gone, or a zombie its new parent (init) has not reaped yet."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


def _pgid(pid: int) -> int | None:
    import os

    try:
        return os.getpgid(pid)
    except OSError:
        return None


@pytest.mark.parametrize(
    "body,hit",
    [
        ("x = 1\n", None),
        ("from rsi.gates import rsi_probe\n", "rsi"),
        ("see docs/rsi/models/Qwen3-4B/ITERATIONS.md, built in pegaflow-semantics-rsi/\n", None),
        ("m = importlib.import_module('rsi.gates.x')\n", "rsi"),
        ("import sys\nsys.modules['rsi.gates.rsi_probe'] = None\n", "sys.modules write"),
        ("import gc\nobjs = gc.get_objects()\n", "gc introspection"),
        ("p = run_dir / 'baseline' / 'gsm8k' / 'warm-outputs.json'\n", "-outputs.json"),
        ("glob('digest-*.jsonl')\n", "digest-"),
        ("sys.meta_path.insert(0, Finder())\n", "import hook"),
    ],
    ids=[
        "clean",
        "rsi-import",
        "rsi-in-paths",
        "rsi-module-string",
        "sys-modules",
        "gc",
        "outputs",
        "digests",
        "meta-path",
    ],
)
def test_static_scan_flags_instrument_access(tmp_path, body, hit):
    """M7(b): a snapshot that names the instrument or run artifacts, or
    patches modules at runtime, is flagged (run non-official until a
    reviewer clears the id); strings of a native module are scanned too."""
    import argparse

    from rsi.gates import provenance

    pkg = tmp_path / "ov" / "pegaflow"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "conn.py").write_text(body)
    hits = provenance.scan(pkg)
    names = {h["id"].rsplit(":", 1)[1] for h in hits}
    assert (hit in names) if hit else not names, hits
    args = argparse.Namespace(under_test_path="x", under_test_overlay=str(tmp_path / "ov"))
    assert bool(run_gate.scan_record(args)["flagged"]) is bool(hit)
    allow = [{"path": "pegaflow/conn.py", "pattern": n, "reason": "test"} for n in names]
    rec = run_gate.scan_record(args, allow)
    assert rec["flagged"] == [] and len(rec["allowlisted"]) == len(hits)
    assert all(h["reason"] == "test" for h in rec["allowlisted"])
    (pkg / "conn.py").write_text("x = 1\n")
    (pkg / "pegaflow.so").write_bytes(b"\x7fELF\x00\x01rsi_poison\x00")
    assert [h["id"] for h in provenance.scan(pkg)] == ["pegaflow/pegaflow.so:strings:rsi_poison"]


def test_official_needs_under_test_path(tmp_path):
    """An arm under test without ``--under-test-path`` imports the venv's
    pegaflow (no snapshot, no verification): never official."""
    import argparse

    args = argparse.Namespace(**(run_gate.OFFICIAL | {"num_gpu_blocks": 0, "vllm_arg": [], "under_test_path": ""}))  # fmt: skip
    probs = run_gate.official_problems(args, ["plain", "baseline", "m:C"], list(run_gate.WORKLOADS), tmp_path, tmp_path)  # fmt: skip
    assert any(p.startswith("--under-test-path missing") for p in probs), probs


@pytest.mark.parametrize(
    "arm,recorded,ok",
    [
        pytest.param("pegaflow.connector_v2:C", "snap", True, id="under-test-from-snapshot"),
        pytest.param("pegaflow.connector_v2:C", None, False, id="under-test-not-recorded"),
        pytest.param("pegaflow.connector_v2:C", "venv", False, id="under-test-from-venv"),
        pytest.param("pegaflow.connector_v2:C", "gate", False, id="under-test-from-gate"),
        pytest.param("rsi.gates.myconn:C", "gate", False, id="rsi-module-under-test-from-gate"),
        pytest.param("rsi.gates.myconn:C", "snap", True, id="rsi-module-under-test-from-snapshot"),
        pytest.param("selftest:honest", "gate", True, id="selftest-from-gate"),
        pytest.param("selftest:honest", "snap", False, id="selftest-from-snapshot"),
    ],
)
def test_connector_module_must_come_from_the_snapshot(tmp_path, arm, recorded, ok):
    """Import-tree provenance: the probe records ``__file__`` of the arm's
    connector module. Its root is chosen by arm kind (M8): only the declared
    self-test arms resolve under the gate tree; an arm under test must
    resolve under its snapshot whatever its module name (an ``rsi.*`` one
    in the gate tree is never staged or scanned)."""
    import json
    import sys

    from rsi.gates import provenance

    snap = tmp_path / "ov" / "pegaflow"
    purelib = provenance.venv_purelib()
    files = {
        "rsi.gates.rsi_probe": str(run_gate.REPO / "rsi/gates/rsi_probe.py"),
        "worker_extension": str(run_gate.REPO / "rsi/gates/rsi_probe.py"),
        "vllm": str(purelib / "vllm/__init__.py"),
        "torch": str(purelib / "torch/__init__.py"),
        "pegaflow": str(snap / "__init__.py"),
        "prefix": sys.prefix,
    }
    mod = run_gate.expected_connector(arm)
    where = {"snap": snap, "venv": tmp_path / "venv" / "pegaflow", "gate": run_gate.REPO / "rsi"}
    if recorded:
        files[f"connector:{mod}"] = str(where[recorded] / "conn" / "__init__.py")
    probe = tmp_path / "ut" / "gsm8k" / "probe"
    probe.mkdir(parents=True)
    (probe / "spec.json").write_text(json.dumps({"modules": files}))
    root = run_gate.connector_root(arm, snap.resolve())
    v = run_gate.g0_provenance(tmp_path, "ut", snap.resolve(), mod, root)
    assert v["pass"] is ok, v
    assert run_gate.expected_connector("selftest:honest") == "rsi.gates.selftest.connector"
    assert run_gate.expected_connector("native") is None


def test_connector_under_test_is_wired_to_the_probe(tmp_path):
    mod = "pegaflow.connector_v2"
    cfg = run_gate.kv_transfer_config(f"{mod}:PegaKVConnectorV2", tmp_path, "tagged")
    probe_cfg = cfg["kv_connector_extra_config"]["connectors"][1]
    assert probe_cfg["kv_connector_extra_config"]["connector_modules"] == [mod]


@pytest.mark.parametrize(
    "arm,plugins",
    [
        ("old", run_gate.PEGAFLOW_PLUGIN),
        ("plain", ""),
        ("baseline", ""),
        ("native", ""),
        ("lmcache", ""),
        ("selftest:honest", ""),
        ("pegaflow.connector_v2:PegaKVConnectorV2", ""),
    ],
)
def test_only_the_old_arm_loads_the_pegaflow_plugin(tmp_path, monkeypatch, arm, plugins):
    """vLLM's API server looks MultiConnector children up by registered
    name to rebuild their stats; ``PegaKVConnector`` is registered only by
    the pegaflow plugin. Every other arm runs with no plugin (M7d)."""
    # The name is the venv package's entry point (python/pyproject.toml).
    eps = (
        (run_gate.REPO / "python/pyproject.toml")
        .read_text()
        .split('[project.entry-points."vllm.general_plugins"]\n', 1)[1]
    )
    assert eps.startswith(f'{run_gate.PEGAFLOW_PLUGIN} = "pegaflow.vllm_plugin:register"')
    seen: dict[str, str] = {}

    class _Lease:
        ports, block = [20001], [20010]

    def launched(cmd, env, log, url, timeout, lease):
        seen["plugins"] = env["VLLM_PLUGINS"]
        return None

    monkeypatch.setattr(run_gate, "PortLease", lambda *a, **kw: _Lease())
    monkeypatch.setattr(run_gate, "launched", launched)
    monkeypatch.setattr(run_gate, "isolated", lambda args, arm, log, cmd: cmd)
    args = SimpleNamespace(
        gpu="0", model="m", tp=1, max_model_len=8, max_num_batched_tokens=8, max_num_seqs=1,
        gpu_mem=0.5, vllm_arg=[], num_gpu_blocks=0, under_test_path="",
        lmcache_overlay=str(tmp_path),
    )  # fmt: skip
    run_gate.vllm_server(args, {}, arm, tmp_path / "probe", False, None, tmp_path / "log", 1 << 30)
    assert seen["plugins"] == plugins == run_gate.vllm_plugins(arm)


def test_module_files_must_resolve_under_gate_and_venv(tmp_path):
    import sys

    from rsi.gates import provenance

    repo = run_gate.REPO
    purelib = provenance.venv_purelib()
    pf = tmp_path / "pf"
    good = {
        "rsi.gates.rsi_probe": str(repo / "rsi/gates/rsi_probe.py"),
        "worker_extension": str(repo / "rsi/gates/rsi_probe.py"),
        "vllm": str(purelib / "vllm/__init__.py"),
        "torch": str(purelib / "torch/__init__.py"),
        "pegaflow": str(pf / "__init__.py"),
        "prefix": sys.prefix,
    }
    assert provenance.module_problems(good, repo, pf, "x") == []
    for k, bad in (
        ("rsi.gates.rsi_probe", str(tmp_path / "rsi/gates/rsi_probe.py")),
        ("vllm", str(tmp_path / "vllm/__init__.py")),
        ("pegaflow", str(tmp_path / "other/__init__.py")),
        ("sitecustomize", str(tmp_path / "sitecustomize.py")),
    ):
        assert provenance.module_problems(good | {k: bad}, repo, pf, "x"), k


@pytest.mark.parametrize("pool,fits", [("64gb", True), ("1kb", False)])
def test_pool_check_fails_before_an_evicting_store(pool, fits):
    import argparse

    args = argparse.Namespace(pool_size=pool)
    u = {"workload": "replay", "unique_bytes": 2048, "how": "estimate"}
    if fits:
        assert run_gate.check_pool(args, u)["fits"]
    else:
        with pytest.raises(run_gate.PoolTooSmall):
            run_gate.check_pool(args, u)


def test_contamination_marks_not_drops():
    rep = lambda **p: {"replay": {"passes": {k: {"ttft_p99": v} for k, v in p.items()}}}  # noqa: E731
    results = {
        "baseline": rep(cold=1.0, warm2=0.5, warm3=0.5),
        "old": rep(cold=9.0, warm2=6.0, warm3=0.4),
        "X": rep(cold=1.0, warm=50.0, warm2=0.5, warm3=5.0, _warmup=99.0),
    }
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    # 9x and exactly 10x pass; 12x is flagged; untimed warm / _warmup ignored.
    assert got == {("old", "warm2")}
    # S8: a baseline stall (p99 > 10x the arms' median for that pass).
    results["baseline"] = rep(cold=1.0, warm2=0.5, warm3=40.0)
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    assert ("baseline", "warm3") in got  # median(0.4, 5.0) = 2.7; 40 > 27
    results["baseline"] = rep(cold=1.0, warm2=0.5, warm3=27.0)
    got = {(c["arm"], c["pass"]) for c in run_gate.contamination(results)}
    assert ("baseline", "warm3") not in got


def test_warmup_seconds_reported_not_gated():
    """S9: ``passes._warmup.seconds`` per arm vs baseline, in the G4 verdicts
    without a ``pass`` key (reported, not gated)."""

    def rep(w, c):
        return {"passes": {"_warmup": {"seconds": w}, "cold": {"ttft_p50": c, "req_per_s": 1.0}}}

    base_gsm = {"passes": {"_warmup": {"seconds": 0.5}}}
    g = run_gate._perf_verdicts(rep(3.0, 1.0), rep(1.0, 1.0), None, base_gsm, True)
    assert g["G4.warmup_seconds"] == {
        "replay": {"value": 3.0, "baseline": 1.0},
        "gsm8k": {"value": None, "baseline": 0.5},
    }
    assert "pass" not in g["G4.warmup_seconds"]


def test_warmup_shares_no_prefix_with_workload():
    from rsi.gates import workloads

    w = workloads.warmup_prompt()
    assert w == workloads.warmup_prompt() and len(w) == workloads.WARMUP_TOKENS
    reqs = [{"pass": "_warmup", "keys": ["w0", "w1"]}, {"pass": "cold", "keys": ["a0", "a1"]}]
    assert oracle.warmup_problems(reqs) == []
    reqs.append({"pass": "warm", "keys": ["w0", "b1"]})
    assert oracle.warmup_problems(reqs) == [
        "1 workload requests share a prefix block with the warmup"
    ]


@pytest.mark.parametrize(
    "digested,skipped,ok",
    [((0, 1), {}, True), ((1,), {}, False), ((0, 1), {"0": "MambaSpec: not digested"}, False)],
    ids=["all-groups", "group-without-records", "group-skipped-by-probe"],
)
def test_every_transferred_group_must_be_digested(tmp_path, digested, skipped, ok):
    """A hybrid model whose state group (e.g. Mamba) the probe cannot digest
    fails G2: the digest checks would judge only the attention bytes."""
    import json

    probe = tmp_path / "probe"
    probe.mkdir()
    spec = _spec(CKPT, FULL) | {"probe": {"digest_skipped_groups": skipped}}
    (probe / "spec.json").write_text(json.dumps(spec))
    (probe / "digest-pp0-tp0.jsonl").write_text("".join(json.dumps({"group": g}) + "\n" for g in digested))  # fmt: skip
    v = run_gate.digest_groups(probe)
    assert v["pass"] is ok, v
    assert v["groups"] == [0, 1]


_RUN_LIKE = """
import asyncio, sys, time
from pathlib import Path
from rsi.gates import run_gate
port, out, where = sys.argv[1], Path(sys.argv[2]), sys.argv[3]


async def swallowing_pass():
    # A pass that converts its cancellation into an ordinary error, which a
    # reference arm's stage would record and move past.
    try:
        await asyncio.sleep(600)
    except asyncio.CancelledError:
        raise RuntimeError("pass failed")


def fake_stage(args, gate, arm, wl, out_dir, **kw):
    (out / f"started-{arm}").write_text("")
    cmd = [sys.executable, "-m", "http.server", port, "--bind", "127.0.0.1"]
    with run_gate.launched(cmd, {}, out / f"{arm}.log", f"http://127.0.0.1:{port}/", 60) as p:
        (out / "pid").write_text(str(p.pid))
        try:
            if where == "asyncio":
                asyncio.run(swallowing_pass())
            else:
                time.sleep(600)
        except Exception as e:  # what a pass loop could do with it
            raise RuntimeError("stage failed") from e
    return {}


run_gate.run_arm_workload = fake_stage
run_gate.install_stop_handlers()
for arm in ("old", "native"):  # reference arms: an ordinary crash is recorded
    run_gate.run_stage(None, {}, arm, "replay", out)
(out / "finished").write_text("")
"""


@pytest.mark.parametrize("where", ["asyncio", "sleep"])
@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_stop_signal_aborts_the_run(tmp_path, sig, where):
    """``kill -<SIG> -<run_gate pgid>`` while a stage is inside asyncio.run or
    a sleep: the stage's process group is stopped, no later stage starts, and
    the gate exits 128 + signal. The gate is started with SIGINT ignored, as
    a background job of a non-interactive shell (``setsid nohup ... &``) is."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(run_gate.__file__).resolve().parents[2]
    env = _env_with_path(root)
    port = str(run_gate.free_port())

    def ignore_sigint():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    gate = subprocess.Popen(
        [sys.executable, "-c", _RUN_LIKE, port, str(tmp_path), where],
        env=env,
        start_new_session=True,
        preexec_fn=ignore_sigint,
    )
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    while not pid_file.exists():
        assert gate.poll() is None and time.monotonic() - t0 < 60
        time.sleep(0.2)
    time.sleep(1)  # inside asyncio.run / sleep
    sleeper = int(pid_file.read_text())
    signum = getattr(signal, sig)
    os.killpg(gate.pid, signum)
    assert gate.wait(timeout=120) == 128 + signum
    with pytest.raises(ProcessLookupError):
        os.killpg(sleeper, 0)
    assert not (tmp_path / "started-native").exists()
    assert not (tmp_path / "finished").exists()


def test_g0_spec_ignores_module_paths(tmp_path):
    """M4: a fresh probe dump carries ``modules`` (host paths: the arm under
    test imports pegaflow from its overlay, the baseline from the venv; the
    committed dump has none). G0.spec must pass on the committed Qwen3-4B
    layout with differing ``modules``; G0.provenance judges them."""
    import json

    committed = run_gate.REPO / "docs/rsi/models/Qwen3-4B/spec.json"
    spec = json.loads(committed.read_text())
    for label, pf in (("baseline", "/venv/pegaflow/__init__.py"), ("arm", "/run/overlay/pegaflow/__init__.py")):  # fmt: skip
        probe = tmp_path / label / "gsm8k" / "probe"
        probe.mkdir(parents=True)
        dump = spec | {"modules": {"pegaflow": pf, "rsi.gates.rsi_probe": f"/{label}/rsi_probe.py"}}
        (probe / "spec.json").write_text(json.dumps(dump))
    v = run_gate.g0(tmp_path, "arm", committed)
    assert v["pass"], v
    # A real layout difference still fails.
    bad = json.loads((tmp_path / "arm/gsm8k/probe/spec.json").read_text())
    bad["kv_cache_config"]["kv_cache_layout"] = "LBHNC"
    (tmp_path / "arm/gsm8k/probe/spec.json").write_text(json.dumps(bad))
    assert not run_gate.g0(tmp_path, "arm", committed)["pass"]


# ------------------------------------------------------------ split invariance


def _split_probe(tmp_path):
    """Key ``p:2`` (block 3) computed four times: twice from start 0 (bytes
    A, A), once from start 2 blocks (bytes B: another split, e.g. MLA's
    separately attended cached context), once from start 2 blocks with bytes
    C (same split as B, different bytes: a real fault)."""
    reqs, digests = [], []
    for i, (start, byte) in enumerate([(0, "A"), (0, "A"), (2 * BT, "B"), (2 * BT, "C")]):
        rid = f"r{i}"
        r = _req("p", 3, 10 * i + 1, "cold") | {"req_id": rid, "digest": "full"}
        r["episodes"] = [[10 * i + 1, start, start]]
        r["progress"] = [[10 * i + 1, 3 * BT + 1]]
        reqs.append(r)
        blk = {"key": "p:2", "covered": 3 * BT, "exact": [byte], "l1": [1.0]}
        digests.append({"req_id": rid, "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]})  # fmt: skip
    return _probe(tmp_path, reqs, digests)


@pytest.mark.parametrize(
    "split_inv,compared,exact,unmatched",
    [(True, 4, 2, 0), (False, 4, 2, 0)],
    ids=["split-invariant", "split-variant"],
)
def test_computed_exact_judges_same_split_only(tmp_path, split_inv, compared, exact, unmatched):
    """Split-invariant: every copy is a reference (B, C inexact). Split-variant:
    A/A judged against each other (exact), B/C against each other (both
    inexact: same split, different bytes still fails)."""
    from rsi.gates import digest_compare

    probe = _split_probe(tmp_path)
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=split_inv)
    assert (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"]) == (
        compared,
        exact,
        unmatched,
    )
    assert not run_gate._digest_verdict(d, None)["pass"]


def test_computed_other_split_only_is_reported_not_judged(tmp_path):
    """Drop C: B has only copies at another split. Split-variant: B is
    unmatched (reported, coverage < 1) and the verdict passes; split-invariant
    keeps the strict rule and fails."""
    import json

    from rsi.gates import digest_compare

    probe = _split_probe(tmp_path)
    for name in ("requests.jsonl", "digest-pp0-tp0.jsonl"):
        f = probe / name
        f.write_text("".join(x + "\n" for x in f.read_text().splitlines() if json.loads(x)["req_id"] != "r3"))  # fmt: skip
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=False)
    assert (d["computed_compared"], d["computed_exact"], d["computed_unmatched_split"]) == (2, 2, 1)
    v = run_gate._digest_verdict(d, None)
    assert v["pass"] and v["computed_coverage"] == pytest.approx(2 / 3)
    strict = digest_compare.compare(probe, probe, invariant=True, split_invariant=True)
    assert not run_gate._digest_verdict(strict, None)["pass"]
    # S7: the unmatched block counts as non-aligned unless its prompt length
    # is a multiple of the alignment (no recorded length: non-aligned).
    by = digest_compare.compare(probe, probe, True, False, align=16)["computed_by_pass"]
    assert by["cold"]["unjudged_nonaligned"] == 1
    f = probe / "requests.jsonl"
    f.write_text("".join(json.dumps(json.loads(x) | {"prompt_len": 64}) + "\n" for x in f.read_text().splitlines()))  # fmt: skip
    by = digest_compare.compare(probe, probe, True, False, align=16)["computed_by_pass"]
    assert by["cold"]["unjudged_nonaligned"] == 0  # reported, not absent


def _ctrl(differ: int, lp: int, n: int = 10, prewarmed: int = 10) -> dict:
    return {
        "n": n,
        "differ": [i < differ for i in range(n)],
        "logprob_differ": [i < lp for i in range(n)],
        "prewarmed": prewarmed,
        "ref_equals_cold": True,
    }


@pytest.mark.parametrize(
    "claim,ctrl,ok",
    [
        (True, _ctrl(0, 0), True),
        (False, _ctrl(0, 0), False),  # declared variant, measured invariant
        (True, _ctrl(0, 3), False),  # logprobs moved: not split-invariant
        (False, _ctrl(2, 5), True),
        (False, _ctrl(2, 5, prewarmed=5), False),  # pre-warm did not take
        (True, None, False),
        (True, _ctrl(0, 0) | {"ref_equals_cold": False}, False),  # N1: no stable basis
    ],
    ids=[
        "inv-ok",
        "false-claim",
        "logprob-only",
        "variant-ok",
        "vacuous",
        "missing",
        "ref-not-cold",
    ],  # fmt: skip
)
def test_split_control_establishes_gate_claim(claim, ctrl, ok):
    assert run_gate._split_control_verdict(ctrl, claim)["pass"] is ok


@pytest.mark.parametrize(
    "split_inv,cold,warm,vs_split,ok",
    [
        (True, 0, 0, (None, None), True),
        (True, 0, 1, (None, None), False),
        (False, 0, 60, (0, 3), True),  # differs only where the plain split stops a block early
        (False, 0, 60, (1, 0), False),  # one wrong completion fails
        (False, 1, 60, (0, 0), False),  # cold has the baseline's split: no slack
        (False, 0, 60, (None, None), False),  # no control: not judged, fails
    ],
    ids=[
        "inv-equal",
        "inv-differ",
        "variant-equal-split",
        "variant-one-wrong",
        "variant-cold",
        "variant-missing",
    ],  # fmt: skip
)
def test_outputs_exact_against_same_split(split_inv, cold, warm, vs_split, ok):
    v = run_gate._outputs_verdict({"cold": cold, "warm": warm}, True, split_inv, vs_split)
    assert v["pass"] is ok, v


def test_split_differ_separates_aligned_prompts():
    ctrl = {"n": 4, "split_outputs": ["a", "b", "c", "d"], "aligned": [False, True, False, True]}
    assert run_gate.split_differ(ctrl, ["a", "x", "c", "d"]) == (0, 1)
    assert run_gate.split_differ(ctrl, ["x", "b", "c"]) == (1, 0)
    assert run_gate.split_differ(None, ["a"]) == (None, None)


def test_arm_writes_to_its_dir_cannot_change_a_verdict(tmp_path):
    """M9: the gate judges outputs from ``<run>/.gate/<label>/<wl>`` (never
    bound into an arm). A forged ``warm-outputs.json`` in the arm's own
    stage dir changes nothing; a symlink the arm leaves there (e.g. its
    digest file -> the baseline's) fails the stage."""
    import json

    for label, warm in (("baseline", ["a", "b"]), ("ut", ["a", "x"])):
        g = tmp_path / run_gate.GATE_DIR / label / "gsm8k"
        g.mkdir(parents=True)
        (g / "cold-outputs.json").write_text(json.dumps(["a", "b"]))
        (g / "warm-outputs.json").write_text(json.dumps(warm))
    own = tmp_path / "ut" / "gsm8k"
    own.mkdir(parents=True)
    (own / "warm-outputs.json").write_text(json.dumps(["a", "b"]))  # the forgery
    ref = run_gate._outputs(tmp_path, "baseline", "cold")
    assert run_gate._differ(ref, run_gate._outputs(tmp_path, "ut", "warm")) == 1
    assert run_gate.gate_dir(own) == tmp_path / ".gate" / "ut" / "gsm8k"
    assert run_gate.untrusted_problems(own) == []
    (own / "probe").mkdir()
    (own / "probe" / "digest-pp0-tp0.jsonl").symlink_to(tmp_path / "baseline-digest.jsonl")
    assert run_gate.untrusted_problems(own) == ["symlink probe/digest-pp0-tp0.jsonl"]


def test_missing_token_record_is_not_aligned(tmp_path):
    """S7: a request without a token record is never exempted from the
    exact warm-vs-split comparison."""
    import json

    d = tmp_path / run_gate.GATE_DIR / "plain" / "gsm8k"
    d.mkdir(parents=True)
    for name in ("cold", "split_ref", "split"):
        (d / f"{name}-outputs.json").write_text(json.dumps(["a", "b"]))
    (d / "split_ref-tokens.json").write_text(json.dumps([None, None]))
    (d / "split-tokens.json").write_text(json.dumps([None, {"prompt_tokens": 64, "cached": 48}]))
    ctrl = run_gate.split_control(tmp_path)
    assert ctrl["aligned"] == [False, True]
    assert run_gate.split_differ(ctrl, ["x", "b"]) == (1, 0)


def test_split_control_prewarm_must_reach_last_full_block(tmp_path):
    """S6: the pre-warm counts only if ``split`` cached the prompt up to its
    last full block (one block more than split_ref is not enough)."""
    import json

    d = tmp_path / run_gate.GATE_DIR / "plain" / "gsm8k"
    d.mkdir(parents=True)
    outs = ["o0", "o1", "o2"]
    for name in ("cold", "split_ref", "split"):
        (d / f"{name}-outputs.json").write_text(json.dumps(outs))
    ref = [{"logprobs": [0.0], "cached": 32, "prompt_tokens": n} for n in (100, 64, 100)]
    got = [{"logprobs": [0.0], "cached": c, "prompt_tokens": n} for c, n in ((96, 100), (48, 64), (48, 100))]  # fmt: skip
    (d / "split_ref-tokens.json").write_text(json.dumps(ref))
    (d / "split-tokens.json").write_text(json.dumps(got))
    ctrl = run_gate.split_control(tmp_path)
    assert ctrl["prewarmed"] == 2 and ctrl["aligned"] == [False, True, False]
    assert ctrl["ref_equals_cold"]


def _cov(by_pass: dict) -> dict:
    v = {"pass": True}
    run_gate._coverage_verdict(v, {"split_invariant": False, "computed_by_pass": by_pass}, ("cold", "warm"))  # fmt: skip
    return v


@pytest.mark.parametrize(
    "warm,ok",
    [
        ({"compared": 95, "unmatched_split": 5, "unjudged_nonaligned": 0}, True),
        (
            {"compared": 0, "unmatched_split": 2136, "unjudged_nonaligned": 0},
            False,
        ),  # the reviewer's vacuous honest arm
        (
            {"compared": 80, "unmatched_split": 10, "unreferenced": 10, "unjudged_nonaligned": 0},
            False,
        ),
        ({}, False),  # no warm computed block judged at all
        # S7: 95% coverage, but one unjudged block on a non-aligned prompt.
        ({"compared": 95, "unmatched_split": 5, "unjudged_nonaligned": 1}, False),
        ({"compared": 95, "unmatched_split": 5}, False),  # count not reported
    ],
    ids=["covered", "vacuous", "low", "missing", "nonaligned-unjudged", "nonaligned-unknown"],
)
def test_split_variant_needs_computed_coverage_per_pass(warm, ok):
    v = _cov({"cold": {"compared": 10}, "warm": warm})
    assert v["pass"] is ok, v
    assert _cov({"cold": {"compared": 10}, "warm": warm} | {"cold": {}})["pass"] is False


def test_split_variant_still_fails_wrong_loaded_bytes(tmp_path):
    """S2: a loaded (external) block with wrong bytes fails on a
    split-variant model too; only computed blocks are judged per split."""
    from rsi.gates import digest_compare

    reqs, digests = [], []
    for i, (name, start, byte) in enumerate([("cold", 0, "A"), ("warm", 2 * BT, "flip")]):
        r = _req("p", 2, 10 * i + 1, name) | {"req_id": f"r{i}", "digest": "full"}
        r["episodes"] = [[10 * i + 1, 0, start]]
        r["progress"] = [[10 * i + 1, 2 * BT + 1]]
        reqs.append(r)
        blk = {"key": "p:1", "covered": 2 * BT, "exact": [byte], "l1": [1.0]}
        digests.append({"req_id": f"r{i}", "rank": "pp0-tp0", "group": 0, "layers": ["l0"], "blocks": [blk]})  # fmt: skip
    probe = _probe(tmp_path, reqs, digests)
    d = digest_compare.compare(probe, probe, invariant=True, split_invariant=False)
    assert d["external_mismatched"] == 1
    assert not run_gate._digest_verdict(d, 0.0)["pass"]


def test_workload_sessions_never_reuse_connections():
    """vLLM's uvicorn closes idle keep-alive connections after 5 s; a pooled
    client reuses a closing socket and the pass dies with "Server
    disconnected" (the gate then stops vLLM: looks like an external SIGTERM)."""
    import asyncio

    from rsi.gates import workloads

    async def force_close() -> bool:
        async with workloads.session() as s:
            return s.connector.force_close

    assert asyncio.run(force_close())
    # Every workload HTTP client goes through session(): no other
    # ClientSession / TCPConnector anywhere in the gate code.
    import re
    from pathlib import Path

    src = Path(workloads.__file__).parent
    for f in sorted(src.rglob("*.py")):
        if f.name.startswith("test_"):
            continue
        text = f.read_text()
        if f.name == "workloads.py":
            text = text.replace(text[text.index("def session(") : text.index("# ---", text.index("def session("))], "")  # fmt: skip
        assert not re.search(r"ClientSession\(|TCPConnector\(", text), f
