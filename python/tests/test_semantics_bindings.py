"""Replay the Rust semantics vectors through the Python bindings.

Guards against the bindings drifting from `pegaflow-semantics`. Skips when
the native extension is not built (source-only default gate).
"""

import json
from pathlib import Path

import pytest

# Skip only when the native extension is not built; a built extension that
# lacks `semantics` must fail loudly. Other test modules may install a
# file-less stub for `pegaflow.pegaflow` (tests/unit_stubs.py), so a module
# without `__file__` counts as not built.
_ext = pytest.importorskip("pegaflow.pegaflow", exc_type=ImportError)
if getattr(_ext, "__file__", None) is None:
    pytest.skip("native extension not built (unit stub installed)", allow_module_level=True)
import pegaflow.semantics as sem  # noqa: E402

VECTORS = Path(__file__).resolve().parents[2] / "pegaflow-semantics" / "vectors"


def _group(v: dict):
    kind = v["kind"]
    if kind == "window":
        return sem.Group.window(v["block_tokens"], v["tokens"])
    return getattr(sem.Group, kind)(v["block_tokens"])


def _cases(name: str) -> list[dict]:
    return json.loads((VECTORS / name).read_text())


@pytest.mark.parametrize("case", _cases("resolve.json"), ids=lambda c: c["id"])
def test_resolve_vectors(case: dict) -> None:
    groups = [_group(g) for g in case["groups"]]
    assert sem.resolve(groups, case["present"], case["prompt_len"]) == case["resolve"]
    for s in case.get("saveable", []):
        for g, (lo, hi) in zip(groups, s["ranges"], strict=True):
            want = (lo, hi) if lo < hi else (0, 0)
            assert sem.saveable(g, s["before"], s["after"]) == want, (g, s)


@pytest.mark.parametrize("case", _cases("key.json"), ids=lambda c: c["id"])
def test_key_vectors(case: dict) -> None:
    assert sem.key(bytes(case["hash"]), case["group"]) == bytes(case["key"])


def _shard(v: dict):
    if v["type"] == "replicated":
        return sem.Shard.replicated()
    if v["type"] == "heads":
        return sem.Shard.heads(v["index"], v["count"])
    return sem.Shard.tokens(v["index"], v["count"], v["interleave"])


@pytest.mark.parametrize("case", _cases("seal_plan.json"), ids=lambda c: c["id"])
def test_seal_plan_vectors(case: dict) -> None:
    ranks = [
        (
            r["rank"],
            [
                sem.Region(g["layer"], g["group"], g["segments"], _shard(g["shard"]))
                for g in r["regions"]
            ],
        )
        for r in case["ranks"]
    ]
    if "error" in case:
        with pytest.raises(ValueError):
            sem.seal_plan(ranks)
        return
    plan = sem.seal_plan(ranks)
    got = [
        {
            "group": g,
            "fragments": [
                {"layer": f.layer, "part": f.part, "holders": f.holders} for f in plan.fragments(g)
            ],
        }
        for g in plan.groups
    ]
    assert got == case["groups"]


def test_seal_plan_dedupes_heads_and_rejects_gaps() -> None:
    def regions(index: int, count: int) -> list:
        return [sem.Region("l0", 0, [32, 32], sem.Shard.heads(index, count))]

    plan = sem.seal_plan([(r, regions(r // 2, 2)) for r in range(4)])
    frags = plan.fragments(0)
    assert [f.holders for f in frags] == [[0, 1], [2, 3]]
    assert plan.slot_of(3, "l0") == (0, 1)
    assert plan.is_writer(2, 0, 1)
    with pytest.raises(ValueError, match="part 1"):
        sem.seal_plan([(0, regions(0, 2))])


def test_invalid_groups_raise_value_error() -> None:
    with pytest.raises(ValueError):
        sem.Group.full(0)
    with pytest.raises(ValueError):
        sem.Group.window(4, 0)
    assert sem.resolve([sem.Group.full(4)], [{0, 1, 2}], 13) == 12
