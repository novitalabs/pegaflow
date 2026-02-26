"""E2E correctness test: verify PegaFlow produces identical outputs to baseline vLLM.

This is the "canary" test - the single most important test for PegaFlow.
It verifies the core contract:

    Given the same prompt + model + sampling params,
    output must be identical with or without PegaFlow.

Usage:
    # Requires pegaflow-server running with metrics enabled:
    cargo run -r --bin pegaflow-server -- \
        --addr 0.0.0.0:50055 --device 0 --pool-size 30gb --http-addr 0.0.0.0:9091

    cd python && pytest tests/test_vllm_e2e_correctness.py -v -s

    # Or with a specific model:
    pytest tests/test_vllm_e2e_correctness.py -v -s --model /path/to/model
"""

import time
from pathlib import Path

import pytest

from .vllm_helpers import (
    VLLMServer,
    call_openai_api,
    check_pegaflow_server,
    fetch_pegaflow_metrics,
)

# Test prompts - varied lengths and domains to test different code paths
TEST_PROMPTS = [
    # Short prompts
    ("short_fact", "The capital of France is"),
    ("short_math", "2 + 2 ="),
    # Medium prompts
    (
        "medium_cs",
        "In computer science, a hash table is a data structure that implements "
        "an associative array. The main advantage of hash tables is",
    ),
    (
        "medium_code",
        "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n    if n <= 1:\n        return n\n    return",
    ),
    # Longer prompts to test block boundaries
    (
        "long_ml",
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that can learn from data. Deep learning, a subset of machine "
        "learning, uses neural networks with many layers. The key benefit of deep learning is",
    ),
    (
        "long_history",
        "The Industrial Revolution began in Britain in the late 18th century "
        "and transformed manufacturing processes. Key inventions included the steam engine, "
        "spinning jenny, and power loom. The social impact of industrialization was",
    ),
]


@pytest.fixture(scope="module")
def model(request) -> str:
    """Get model from command line or use default."""
    return request.config.getoption("--model")


@pytest.fixture(scope="module")
def base_port(request) -> int:
    """Get base port from command line."""
    return request.config.getoption("--e2e-port")


@pytest.fixture(scope="module")
def pega_metrics_port(request) -> int:
    """Get PegaFlow metrics port from command line."""
    return request.config.getoption("--pega-metrics-port")


@pytest.fixture(scope="module")
def tensor_parallel_size(request) -> int:
    """Get tensor parallel size from command line."""
    return request.config.getoption("--tensor-parallel-size")


@pytest.fixture(scope="module")
def pipeline_parallel_size(request) -> int:
    """Get pipeline parallel size from command line."""
    return request.config.getoption("--pipeline-parallel-size")


class TestE2ECorrectness:
    """E2E correctness tests for PegaFlow.

    These tests verify that PegaFlow produces identical outputs to baseline vLLM.
    """

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        """Create a temporary directory for server logs."""
        return tmp_path_factory.mktemp("vllm_logs")

    def test_cache_roundtrip_correctness(
        self,
        model: str,
        base_port: int,
        pega_metrics_port: int,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Core contract: baseline, cold cache, and warm cache must produce identical outputs.

        This is THE canary test. If this fails, something is fundamentally broken.
        """
        # Pre-check: Ensure PegaFlow server is running with metrics enabled
        if not check_pegaflow_server(pega_metrics_port):
            pytest.skip(
                f"PegaFlow server not reachable (metrics port {pega_metrics_port}). "
                "Start with --http-addr flag."
            )

        results = {"baseline": [], "pegaflow_cold": [], "pegaflow_warm": []}

        # Phase 1: Baseline vLLM (no PegaFlow)
        print("\n[Phase 1] Baseline vLLM")
        baseline_log = log_dir / "baseline.log"
        with VLLMServer(
            model,
            base_port,
            use_pegaflow=False,
            log_file=baseline_log,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ):
            for _, prompt in TEST_PROMPTS:
                result = call_openai_api(base_port, model, prompt)
                results["baseline"].append(result)

        # Phase 2: PegaFlow vLLM (cold + warm cache)
        print("[Phase 2] PegaFlow vLLM (cold + warm)")
        metrics_before_cold = fetch_pegaflow_metrics(pega_metrics_port)

        pegaflow_log = log_dir / "pegaflow.log"
        with VLLMServer(
            model,
            base_port + 1,
            use_pegaflow=True,
            log_file=pegaflow_log,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ):
            # Cold cache run
            cold_total_time = 0.0
            for _, prompt in TEST_PROMPTS:
                start = time.time()
                result = call_openai_api(base_port + 1, model, prompt)
                cold_total_time += time.time() - start
                results["pegaflow_cold"].append(result)

            # Verify SAVE operations
            metrics_after_cold = fetch_pegaflow_metrics(pega_metrics_port)
            save_bytes = metrics_after_cold.get(
                "pegaflow_save_bytes_total", 0
            ) - metrics_before_cold.get("pegaflow_save_bytes_total", 0)
            insertions = metrics_after_cold.get(
                "pegaflow_cache_block_insertions_total", 0
            ) - metrics_before_cold.get("pegaflow_cache_block_insertions_total", 0)
            assert save_bytes > 0 or insertions > 0, (
                f"Cold cache did not SAVE: save_bytes={save_bytes}, insertions={insertions}"
            )

            # Warm cache run
            metrics_before_warm = fetch_pegaflow_metrics(pega_metrics_port)
            warm_total_time = 0.0
            for _, prompt in TEST_PROMPTS:
                start = time.time()
                result = call_openai_api(base_port + 1, model, prompt)
                warm_total_time += time.time() - start
                results["pegaflow_warm"].append(result)

            # Verify LOAD operations
            metrics_after_warm = fetch_pegaflow_metrics(pega_metrics_port)
            load_bytes = metrics_after_warm.get(
                "pegaflow_load_bytes_total", 0
            ) - metrics_before_warm.get("pegaflow_load_bytes_total", 0)
            hits = metrics_after_warm.get(
                "pegaflow_cache_block_hits_total", 0
            ) - metrics_before_warm.get("pegaflow_cache_block_hits_total", 0)
            assert load_bytes > 0 or hits > 0, (
                f"Warm cache did not LOAD: load_bytes={load_bytes}, hits={hits}"
            )

            print(
                f"[Metrics] SAVE: {save_bytes / 1e6:.1f}MB, LOAD: {load_bytes / 1e6:.1f}MB, "
                f"Speedup: {cold_total_time / warm_total_time:.2f}x"
            )

        # Phase 3: Verify correctness
        print("[Phase 3] Verifying outputs")
        for i, (prompt_id, _) in enumerate(TEST_PROMPTS):
            baseline = results["baseline"][i]["text"]
            cold = results["pegaflow_cold"][i]["text"]
            warm = results["pegaflow_warm"][i]["text"]

            assert cold == warm, (
                f"[{prompt_id}] Cold vs Warm mismatch:\n  Cold: {cold[:100]}\n  Warm: {warm[:100]}"
            )
            assert baseline == cold, (
                f"[{prompt_id}] Baseline vs PegaFlow mismatch:\n"
                f"  Baseline: {baseline[:100]}\n"
                f"  PegaFlow: {cold[:100]}"
            )

        print("[PASS] All outputs match!")
