"""E2E correctness test: verify PegaFlow produces identical outputs to baseline vLLM.

The single most important test for PegaFlow. Verifies the core contract:

    Given the same prompt + model + greedy sampling,
    output must be identical with or without PegaFlow.

Covers all critical KV cache paths in one deterministic test:
- Cold save + warm load (basic round-trip)
- Multi-block prompts (block boundary alignment)
- Prefix extension (cached "A B C" -> request "A B C D E")
- Prefix rollback (cached "A B C D" -> request "A B")
- Multi-round decode (growing context across rounds)
- Cache metrics (directional assertions)

Usage:
    # Requires pegaflow-server running with metrics enabled:
    cargo run -r --bin pegaflow-server -- \
        --addr 0.0.0.0:50055 --device 0 --pool-size 30gb --http-addr 0.0.0.0:9091

    cd python && pytest tests/test_vllm_e2e_correctness.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .vllm_helpers import (
    VLLMServer,
    call_openai_api,
    check_pegaflow_server,
    fetch_pegaflow_metrics,
)

# ---------------------------------------------------------------------------
# Prompt design
#
# Each prompt group exercises a different cache path. Prompts are long enough
# to span multiple blocks (block_size=16 tokens typically) so that partial
# hits are meaningful, not degenerate single-block cases.
# ---------------------------------------------------------------------------

# Short prompt (likely < 1 full block) — edge case for incomplete blocks
SHORT_PROMPT = "2 + 2 ="

# Long prompt (multi-block) — tests block boundary alignment during save/load
LONG_PROMPT = (
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that can learn from and make decisions based on data. "
    "Deep learning, a further subset of machine learning, uses artificial "
    "neural networks with many layers to model complex patterns in large "
    "amounts of data. The key advantage of deep learning over traditional "
    "machine learning methods is its ability to automatically discover "
    "representations needed for feature detection or classification from "
    "raw data. This eliminates the need for manual feature engineering, "
    "which has traditionally been one of the most time-consuming aspects "
    "of applying machine learning to real-world problems. Convolutional "
    "neural networks have proven particularly effective for image recognition "
    "tasks, while recurrent neural networks and transformers have shown "
    "remarkable results in natural language processing. The transformer "
    "architecture, introduced in the paper Attention Is All You Need, has "
    "become the foundation for modern large language models. The main idea is"
)

# Prefix family: extension
# prefix_base is a strict prefix of prefix_extend.
# After caching prefix_base, requesting prefix_extend should partially hit.
PREFIX_BASE = (
    "The history of computing begins with mechanical calculators in the 17th "
    "century. Blaise Pascal built the Pascaline in 1642, which could perform "
    "addition and subtraction. Gottfried Wilhelm Leibniz later improved upon "
    "this design with the Stepped Reckoner, which could also multiply and "
    "divide. Charles Babbage conceived the Difference Engine in the 1820s and "
    "later designed the more ambitious Analytical Engine, which contained many "
    "features of modern computers including an arithmetic logic unit, control "
    "flow through conditional branching, and memory. Ada Lovelace wrote what "
    "is considered the first computer program for the Analytical Engine. The "
    "next major breakthrough came with"
)

PREFIX_EXTEND = (
    PREFIX_BASE + " the development of electronic computers in the mid-20th century. "
    "Alan Turing formalized the concept of computation with his theoretical "
    "Turing machine in 1936. During World War II, several electronic computing "
    "devices were built including Colossus and ENIAC. The invention of the "
    "transistor at Bell Labs in 1947 revolutionized electronics and led to "
    "increasingly powerful and compact computers. The integrated circuit, "
    "developed independently by Jack Kilby and Robert Noyce, further accelerated "
    "this trend. The impact of these innovations on modern society is"
)

# Prefix family: rollback
# rollback_long is cached first, then rollback_short (a strict prefix of
# rollback_long) requests a subset — the reverse of prefix extension.
ROLLBACK_LONG = (
    "In computer science, a hash table is a data structure that implements an "
    "associative array, also called a dictionary or map. A hash table uses a "
    "hash function to compute an index, also called a hash code, into an array "
    "of buckets or slots, from which the desired value can be found. During "
    "lookup, the key is hashed and the resulting hash indicates where the "
    "corresponding value is stored. Ideally, the hash function will assign each "
    "key to a unique bucket, but most hash table designs employ an imperfect "
    "hash function, which might cause hash collisions where the hash function "
    "generates the same index for more than one key. Such collisions are "
    "typically accommodated by chaining or open addressing. The main advantage is"
)

ROLLBACK_SHORT = (
    "In computer science, a hash table is a data structure that implements an "
    "associative array, also called a dictionary or map. A hash table uses a "
    "hash function to compute an index, also called a hash code, into an array "
    "of buckets or slots, from which the desired value can be found. During "
    "lookup, the key is hashed and the resulting hash indicates where the "
    "corresponding value is stored. Ideally, the hash function will assign each "
    "key to a unique bucket, but most hash table designs employ an imperfect "
    "hash function, which might cause hash collisions where the hash function "
    "generates the same index for more than one key. Such collisions are"
)

# Multi-round decode: three prompts where each is a prefix of the next.
# Simulates growing chat context across decode rounds.
_MULTI_ROUND_STEM = (
    "Quantum computing leverages quantum mechanical phenomena such as "
    "superposition and entanglement to perform computation. Unlike classical "
    "bits that exist in a state of either 0 or 1, quantum bits or qubits can "
    "exist in a superposition of both states simultaneously. This property "
    "allows quantum computers to explore many possible solutions at once. "
    "Quantum entanglement enables qubits that are entangled to be correlated "
    "with each other in ways that have no classical equivalent. When combined, "
    "these properties give quantum computers the potential to solve certain "
    "problems exponentially faster than classical computers. "
)

MULTI_ROUND = [
    _MULTI_ROUND_STEM + "The current state of quantum hardware is",
    (
        _MULTI_ROUND_STEM + "Key algorithms include Shor's algorithm for factoring large numbers "
        "and Grover's algorithm for searching unsorted databases. Current "
        "quantum computers face significant challenges including decoherence, "
        "error rates, and the need for extreme cooling. The path forward involves"
    ),
    (
        _MULTI_ROUND_STEM + "Key algorithms include Shor's algorithm for factoring large numbers "
        "and Grover's algorithm for searching unsorted databases. Current "
        "quantum computers face significant challenges including decoherence, "
        "error rates, and the need for extreme cooling. Major tech companies "
        "including Google, IBM, and Microsoft are investing heavily in quantum "
        "computing research. Google claimed quantum supremacy in 2019 with its "
        "Sycamore processor. IBM has been steadily increasing qubit counts with "
        "its Eagle, Osprey, and Condor processors. The most promising applications are"
    ),
]


# Ordered execution plan for PegaFlow phase.
# Each entry: (label, prompt, cache_expectation)
# cache_expectation: "cold" = first time, "warm" = exact repeat, "partial" = prefix hit
EXECUTION_PLAN: list[tuple[str, str, str]] = [
    # Round 1: cold saves — fill the cache
    ("short_cold", SHORT_PROMPT, "cold"),
    ("long_cold", LONG_PROMPT, "cold"),
    ("prefix_base", PREFIX_BASE, "cold"),
    ("rollback_long", ROLLBACK_LONG, "cold"),
    ("multi_r1", MULTI_ROUND[0], "cold"),
    # Round 2: warm hits — exact same prompts
    ("short_warm", SHORT_PROMPT, "warm"),
    ("long_warm", LONG_PROMPT, "warm"),
    # Round 3: prefix operations
    ("prefix_extend", PREFIX_EXTEND, "partial"),
    ("rollback_short", ROLLBACK_SHORT, "partial"),
    # Round 4: multi-round growth
    ("multi_r2", MULTI_ROUND[1], "partial"),
    ("multi_r3", MULTI_ROUND[2], "partial"),
]

# All unique prompts for baseline collection
ALL_PROMPTS: dict[str, str] = {
    "short": SHORT_PROMPT,
    "long": LONG_PROMPT,
    "prefix_base": PREFIX_BASE,
    "prefix_extend": PREFIX_EXTEND,
    "rollback_long": ROLLBACK_LONG,
    "rollback_short": ROLLBACK_SHORT,
    "multi_r1": MULTI_ROUND[0],
    "multi_r2": MULTI_ROUND[1],
    "multi_r3": MULTI_ROUND[2],
}

# Map execution plan labels to their baseline prompt key
_LABEL_TO_BASELINE: dict[str, str] = {
    "short_cold": "short",
    "long_cold": "long",
    "prefix_base": "prefix_base",
    "rollback_long": "rollback_long",
    "multi_r1": "multi_r1",
    "short_warm": "short",
    "long_warm": "long",
    "prefix_extend": "prefix_extend",
    "rollback_short": "rollback_short",
    "multi_r2": "multi_r2",
    "multi_r3": "multi_r3",
}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestE2ECorrectness:
    """E2E correctness: baseline vLLM vs PegaFlow-enabled vLLM.

    Two-phase structure:
      Phase 1 — run all unique prompts through baseline vLLM, collect golden outputs.
      Phase 2 — run execution plan through PegaFlow vLLM, collect outputs + metrics.
    Each test method asserts one cache path independently.
    """

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("e2e_logs")

    @pytest.fixture(scope="class")
    def baseline_outputs(
        self,
        model: str,
        base_port: int,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        max_model_len: int | None,
    ) -> dict[str, str]:
        """Phase 1: collect golden outputs from baseline vLLM (no PegaFlow)."""
        print("\n[Phase 1] Baseline vLLM — collecting golden outputs")
        outputs: dict[str, str] = {}

        with VLLMServer(
            model,
            base_port,
            use_pegaflow=False,
            log_file=log_dir / "baseline.log",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
        ):
            for key, prompt in ALL_PROMPTS.items():
                result = call_openai_api(base_port, model, prompt)
                outputs[key] = result["text"]
                print(f"  [{key}] {len(result['text'])} chars")

        print(f"[Phase 1] Done — {len(outputs)} golden outputs collected\n")
        return outputs

    @pytest.fixture(scope="class")
    def pegaflow_results(
        self,
        model: str,
        base_port: int,
        pega_metrics_port: int,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        max_model_len: int | None,
    ) -> dict:
        """Phase 2: run execution plan through PegaFlow vLLM."""
        if not check_pegaflow_server(pega_metrics_port):
            pytest.skip(
                f"PegaFlow server not reachable (metrics port {pega_metrics_port}). "
                "Start with --http-addr flag."
            )

        print("[Phase 2] PegaFlow vLLM — executing cache plan")
        pega_port = base_port + 1
        outputs: dict[str, str] = {}

        with VLLMServer(
            model,
            pega_port,
            use_pegaflow=True,
            log_file=log_dir / "pegaflow.log",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
        ):
            metrics_start = fetch_pegaflow_metrics(pega_metrics_port)

            for label, prompt, expectation in EXECUTION_PLAN:
                result = call_openai_api(pega_port, model, prompt)
                outputs[label] = result["text"]
                print(f"  [{label}] ({expectation}) {len(result['text'])} chars")

            metrics_end = fetch_pegaflow_metrics(pega_metrics_port)

        print("[Phase 2] Done\n")
        return {
            "outputs": outputs,
            "metrics_start": metrics_start,
            "metrics_end": metrics_end,
        }

    # -- Individual test methods: each asserts one cache path --

    def _assert_match(
        self,
        baseline_outputs: dict[str, str],
        pegaflow_results: dict,
        label: str,
    ):
        baseline_key = _LABEL_TO_BASELINE[label]
        baseline = baseline_outputs[baseline_key]
        pegaflow = pegaflow_results["outputs"][label]
        assert baseline == pegaflow, (
            f"[{label}] Output mismatch:\n"
            f"  baseline ({len(baseline)} chars): {baseline[:120]}...\n"
            f"  pegaflow ({len(pegaflow)} chars): {pegaflow[:120]}..."
        )

    def test_cold_short(self, baseline_outputs, pegaflow_results):
        """Short prompt cold save — incomplete block edge case."""
        self._assert_match(baseline_outputs, pegaflow_results, "short_cold")

    def test_cold_long(self, baseline_outputs, pegaflow_results):
        """Long prompt cold save — multi-block boundary alignment."""
        self._assert_match(baseline_outputs, pegaflow_results, "long_cold")

    def test_warm_short(self, baseline_outputs, pegaflow_results):
        """Short prompt warm hit — same prompt, full cache hit."""
        self._assert_match(baseline_outputs, pegaflow_results, "short_warm")

    def test_warm_long(self, baseline_outputs, pegaflow_results):
        """Long prompt warm hit — same prompt, full cache hit."""
        self._assert_match(baseline_outputs, pegaflow_results, "long_warm")

    def test_prefix_base(self, baseline_outputs, pegaflow_results):
        """Prefix base — cold save of prefix stem."""
        self._assert_match(baseline_outputs, pegaflow_results, "prefix_base")

    def test_prefix_extend(self, baseline_outputs, pegaflow_results):
        """Prefix extension — cached 'A B C', now requesting 'A B C D E'.

        This is the exact path changed by PR #204: external hit prefix
        + new block computation with correct global index alignment.
        """
        self._assert_match(baseline_outputs, pegaflow_results, "prefix_extend")

    def test_prefix_rollback(self, baseline_outputs, pegaflow_results):
        """Prefix rollback — cached 'A B C D', now requesting shorter 'A B'.

        Tests that a shorter request correctly reuses a subset of
        a longer cached prefix without index misalignment.
        """
        self._assert_match(baseline_outputs, pegaflow_results, "rollback_short")

    def test_rollback_long_cold(self, baseline_outputs, pegaflow_results):
        """Rollback long prompt — cold save baseline."""
        self._assert_match(baseline_outputs, pegaflow_results, "rollback_long")

    def test_multi_round_r1(self, baseline_outputs, pegaflow_results):
        """Multi-round decode R1 — cold save of initial context."""
        self._assert_match(baseline_outputs, pegaflow_results, "multi_r1")

    def test_multi_round_r2(self, baseline_outputs, pegaflow_results):
        """Multi-round decode R2 — extends R1 context, partial cache hit."""
        self._assert_match(baseline_outputs, pegaflow_results, "multi_r2")

    def test_multi_round_r3(self, baseline_outputs, pegaflow_results):
        """Multi-round decode R3 — extends R2 context, deeper partial hit."""
        self._assert_match(baseline_outputs, pegaflow_results, "multi_r3")

    def test_cache_metrics(self, pegaflow_results, pega_metrics_port):
        """Directional cache metrics — saves happened on cold, hits on warm/partial."""
        if not check_pegaflow_server(pega_metrics_port):
            pytest.skip("PegaFlow metrics not available")

        m_start = pegaflow_results["metrics_start"]
        m_end = pegaflow_results["metrics_end"]

        def delta(key: str) -> float:
            return m_end.get(key, 0) - m_start.get(key, 0)

        save_bytes = delta("pegaflow_save_bytes_total")
        insertions = delta("pegaflow_cache_block_insertions_total")
        load_bytes = delta("pegaflow_load_bytes_total")
        hits = delta("pegaflow_cache_block_hits_total")

        assert save_bytes > 0 or insertions > 0, (
            f"No SAVE activity: save_bytes={save_bytes}, insertions={insertions}"
        )
        assert load_bytes > 0 or hits > 0, f"No LOAD activity: load_bytes={load_bytes}, hits={hits}"

        print(
            f"\n[Metrics] saves={insertions:.0f} blocks ({save_bytes / 1e6:.1f}MB), "
            f"hits={hits:.0f} blocks ({load_bytes / 1e6:.1f}MB)"
        )
