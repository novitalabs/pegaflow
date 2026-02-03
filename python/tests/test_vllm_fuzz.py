"""Fuzz test: verify PegaFlow correctness under Zipf-distributed access patterns.

Uses ShareGPT dataset to simulate realistic LLM serving workloads with:
- Zipf distribution: few hot prompts, many cold prompts
- Partial hit simulation: prefix-sharing variants

Usage:
    # Requires pegaflow-server running with metrics enabled:
    cargo run -r --bin pegaflow-server -- \
        --addr 0.0.0.0:50055 --device 0 --pool-size 30gb --http-addr 0.0.0.0:9091

    # Run fuzz tests (skipped by default)
    cd python && pytest tests/test_vllm_fuzz.py -v -s -m fuzz

    # With custom parameters
    pytest tests/test_vllm_fuzz.py -v -s -m fuzz \
        --fuzz-corpus=500 --fuzz-requests=1000 --fuzz-seed=42
"""

import difflib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from .vllm_helpers import (
    VLLMServer,
    call_openai_api,
    check_pegaflow_server,
    fetch_pegaflow_metrics,
)


@dataclass
class AccessItem:
    """Single access in the request sequence."""

    prompt_id: str
    prompt: str
    access_type: str  # "original" | "prefix_50pct" | "extended"


def build_corpus(seed: int = 42, n: int = 500) -> list[dict]:
    """Load ShareGPT dataset and sample n unique prompts.

    Args:
        seed: Random seed for reproducibility.
        n: Number of prompts to sample.

    Returns:
        List of conversation dicts from ShareGPT.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    return [dataset[i] for i in indices]


def generate_access_sequence(
    corpus: list[dict],
    num_requests: int = 1000,
    zipf_alpha: float = 1.2,
    partial_ratio: float = 0.3,
    seed: int = 42,
) -> list[AccessItem]:
    """Generate deterministic Zipf-distributed access sequence.

    Args:
        corpus: List of ShareGPT conversation dicts.
        num_requests: Total number of requests to generate.
        zipf_alpha: Zipf distribution parameter (1.2 = moderate skew).
        partial_ratio: Fraction of repeat accesses that become partial-hit variants.
        seed: Random seed for reproducibility.

    Returns:
        List of AccessItem representing the request sequence.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Generate Zipf-distributed indices
    ranks = np_rng.zipf(zipf_alpha, num_requests)
    indices = np.clip(ranks - 1, 0, len(corpus) - 1)

    seen: set[str] = set()
    sequence: list[AccessItem] = []

    for i, idx in enumerate(indices):
        conversations = corpus[idx].get("conversations", [])
        if not conversations:
            continue

        # Extract first human message as prompt
        prompt = conversations[0].get("value", "")
        if not prompt:
            continue

        pid = f"r{i:04d}_c{idx:03d}"

        if prompt in seen and rng.random() < partial_ratio:
            # Generate partial-hit variant for repeated prompt
            variant_type = rng.choice(["prefix_50pct", "extended"])
            if variant_type == "prefix_50pct":
                cut = len(prompt) // 2
                variant = prompt[:cut] if cut > 5 else prompt
            else:
                variant = prompt + " Explain further."
            sequence.append(AccessItem(f"{pid}_{variant_type}", variant, variant_type))
        else:
            sequence.append(AccessItem(pid, prompt, "original"))
            seen.add(prompt)

    return sequence


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


@pytest.mark.fuzz
class TestFuzzCorrectness:
    """Fuzz tests for PegaFlow using Zipf-distributed ShareGPT prompts."""

    @pytest.fixture(scope="class")
    def access_sequence(self, request) -> list[AccessItem]:
        """Build corpus and generate access sequence from CLI options."""
        seed = request.config.getoption("--fuzz-seed")
        n_corpus = request.config.getoption("--fuzz-corpus")
        n_requests = request.config.getoption("--fuzz-requests")

        print(f"\n[Fuzz] Building corpus: {n_corpus} prompts, seed={seed}")
        corpus = build_corpus(seed=seed, n=n_corpus)

        print(f"[Fuzz] Generating {n_requests} requests with Zipf distribution")
        return generate_access_sequence(corpus, num_requests=n_requests, seed=seed)

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        """Create a temporary directory for server logs."""
        return tmp_path_factory.mktemp("fuzz_logs")

    def test_zipf_correctness(
        self,
        access_sequence: list[AccessItem],
        model: str,
        base_port: int,
        pega_metrics_port: int,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Verify PegaFlow produces identical outputs under Zipf workload.

        This test:
        1. Runs baseline vLLM and saves all outputs
        2. Runs PegaFlow-enabled vLLM and compares outputs
        3. Reports cache statistics
        """
        if not check_pegaflow_server(pega_metrics_port):
            pytest.skip(
                f"PegaFlow server not reachable (metrics port {pega_metrics_port}). "
                "Start with --http-addr flag."
            )

        baseline_results: dict[str, dict] = {}
        failures: list[dict] = []
        stats = {"original": 0, "prefix_50pct": 0, "extended": 0}

        baseline_log = log_dir / "baseline_fuzz.log"
        pegaflow_log = log_dir / "pegaflow_fuzz.log"

        # Phase 1: Run baseline and save results
        print("\n[Fuzz] Phase 1: Running baseline vLLM")
        with VLLMServer(
            model,
            base_port,
            use_pegaflow=False,
            log_file=baseline_log,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ):
            for i, item in enumerate(access_sequence):
                if (i + 1) % 100 == 0:
                    print(f"[Fuzz] Baseline progress: {i + 1}/{len(access_sequence)}")
                try:
                    result = call_openai_api(base_port, model, item.prompt)
                    baseline_results[item.prompt_id] = {
                        "text": result["text"],
                        "prompt": item.prompt,
                        "access_type": item.access_type,
                    }
                except Exception as e:
                    failures.append(
                        {
                            "prompt_id": item.prompt_id,
                            "access_type": item.access_type,
                            "prompt": item.prompt,
                            "error": f"Baseline error: {e}",
                        }
                    )
                    break

        if failures:
            raise AssertionError(self._format_failure(failures[0], log_dir, model, base_port))

        # Phase 2: Run PegaFlow and compare
        print(
            f"\n[Fuzz] Phase 2: Running PegaFlow vLLM (comparing {len(baseline_results)} results)"
        )
        processed = 0
        with VLLMServer(
            model,
            base_port,
            use_pegaflow=True,
            log_file=pegaflow_log,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ):
            metrics_start = fetch_pegaflow_metrics(pega_metrics_port)

            for i, item in enumerate(access_sequence):
                if item.prompt_id not in baseline_results:
                    continue

                processed = i + 1
                stats[item.access_type] += 1

                if (i + 1) % 100 == 0:
                    print(f"[Fuzz] PegaFlow progress: {i + 1}/{len(access_sequence)}")

                try:
                    pegaflow = call_openai_api(base_port, model, item.prompt)
                    baseline_text = baseline_results[item.prompt_id]["text"]

                    if baseline_text != pegaflow["text"]:
                        failures.append(
                            {
                                "prompt_id": item.prompt_id,
                                "access_type": item.access_type,
                                "prompt": item.prompt,
                                "baseline": baseline_text,
                                "pegaflow": pegaflow["text"],
                            }
                        )
                        break
                except Exception as e:
                    failures.append(
                        {
                            "prompt_id": item.prompt_id,
                            "access_type": item.access_type,
                            "prompt": item.prompt,
                            "error": f"PegaFlow error: {e}",
                        }
                    )
                    break

            metrics_end = fetch_pegaflow_metrics(pega_metrics_port)

        # Print report
        self._print_report(access_sequence, processed, stats, metrics_start, metrics_end, failures)

        if failures:
            raise AssertionError(self._format_failure(failures[0], log_dir, model, base_port))

    def _print_report(
        self,
        sequence: list[AccessItem],
        processed: int,
        stats: dict[str, int],
        m_start: dict[str, float],
        m_end: dict[str, float],
        failures: list[dict],
    ):
        """Print cache statistics report."""
        hits = m_end.get("pegaflow_cache_block_hits_total", 0) - m_start.get(
            "pegaflow_cache_block_hits_total", 0
        )
        misses = m_end.get("pegaflow_cache_block_misses_total", 0) - m_start.get(
            "pegaflow_cache_block_misses_total", 0
        )
        total = hits + misses if hits + misses > 0 else 1

        save_bytes = m_end.get("pegaflow_save_bytes_total", 0) - m_start.get(
            "pegaflow_save_bytes_total", 0
        )
        load_bytes = m_end.get("pegaflow_load_bytes_total", 0) - m_start.get(
            "pegaflow_load_bytes_total", 0
        )

        print(f"\n{'=' * 60}")
        print("[Fuzz Report]")
        print(f"  Processed: {processed}/{len(sequence)}")
        print(
            f"  Access types: original={stats['original']}, "
            f"prefix_50pct={stats['prefix_50pct']}, extended={stats['extended']}"
        )
        print(f"  Cache blocks: hits={hits:.0f}, misses={misses:.0f}, hit_rate={hits / total:.1%}")
        print(f"  Data transfer: save={save_bytes / 1e6:.1f}MB, load={load_bytes / 1e6:.1f}MB")
        print(f"  Failures: {len(failures)}")
        print(f"{'=' * 60}\n")

    def _format_failure(
        self, failure: dict, log_dir: Path | None = None, model: str = "", port: int = 8100
    ) -> str:
        """Format a readable failure message for quick diffing."""
        prompt = failure.get("prompt", "")
        prompt_id = failure.get("prompt_id", "unknown")

        # Dump the request to a JSON file for manual replay
        if log_dir is not None:
            self._dump_replay_request(failure, log_dir, model, port)

        if "error" in failure:
            return (
                "[Fuzz Error]\n"
                f"  prompt_id: {prompt_id}\n"
                f"  access_type: {failure.get('access_type')}\n"
                f"  prompt: {self._snippet(prompt)}\n"
                f"  error: {failure.get('error')}"
            )

        baseline = failure.get("baseline", "")
        pegaflow = failure.get("pegaflow", "")
        diff_lines = list(
            difflib.unified_diff(
                baseline.splitlines(),
                pegaflow.splitlines(),
                fromfile="baseline",
                tofile="pegaflow",
                lineterm="",
            )
        )
        max_lines = 40
        if len(diff_lines) > max_lines:
            diff_lines = diff_lines[:max_lines] + ["... (diff truncated)"]
        diff = "\n".join(diff_lines) if diff_lines else "(no diff output)"

        return (
            "[Fuzz Mismatch]\n"
            f"  prompt_id: {prompt_id}\n"
            f"  access_type: {failure.get('access_type')}\n"
            f"  prompt: {self._snippet(prompt)}\n"
            f"  baseline: {self._snippet(baseline)}\n"
            f"  pegaflow: {self._snippet(pegaflow)}\n"
            "  diff:\n"
            f"{diff}"
        )

    def _dump_replay_request(self, failure: dict, log_dir: Path, model: str, port: int) -> None:
        """Dump the failing request to JSON for manual replay with curl."""
        prompt = failure.get("prompt", "")
        prompt_id = failure.get("prompt_id", "unknown")

        # Save full failure info
        failure_file = log_dir / f"failure_{prompt_id}.json"
        with open(failure_file, "w") as f:
            json.dump(failure, f, indent=2, ensure_ascii=False)

        # Save curl-ready request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.0,
            "seed": 42,
            "logprobs": 5,
        }
        request_file = log_dir / f"request_{prompt_id}.json"
        with open(request_file, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"\n[Debug] Failure dumped to: {log_dir}")
        print(f"  - Full failure: {failure_file}")
        print(f"  - Request payload: {request_file}")
        print("\n[Debug] Replay with curl:")
        print(f"  curl -X POST http://localhost:{port}/v1/completions \\")
        print("    -H 'Content-Type: application/json' \\")
        print(f"    -d @{request_file}")

    @staticmethod
    def _snippet(text: str, max_len: int = 200) -> str:
        """Compact snippet to keep logs readable."""
        cleaned = text.replace("\n", "\\n")
        if len(cleaned) > max_len:
            return f"{cleaned[:max_len]}..."
        return cleaned
