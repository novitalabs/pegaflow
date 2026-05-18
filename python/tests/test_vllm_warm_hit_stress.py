"""Stress E2E for PegaFlow warm-hit pressure.

This test intentionally creates warm-hit traffic under constrained KV capacity.
It is not a replacement for the correctness E2E; it is an observability-heavy
proof that concurrent repeated prompts keep using the cache path and do not
leave obvious pending-probe release failures in the logs.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests

from .vllm_helpers import (
    PegaFlowServer,
    VLLMServer,
    call_openai_api,
    fetch_pegaflow_metrics,
)

pytestmark = [pytest.mark.e2e, pytest.mark.stress, pytest.mark.gpu]

STRESS_GPU_MEMORY_UTILIZATION = 0.82
STRESS_MAX_MODEL_LEN = 2048
STRESS_MAX_NUM_SEQS = 16
STRESS_CONCURRENCY = 12
STRESS_LONG_OUTPUT_WORKERS = 8
STRESS_LONG_MAX_TOKENS = 96
STRESS_SHORT_MAX_TOKENS = 32
MIN_WARM_HIT_LOOKUPS = STRESS_CONCURRENCY // 2
WARM_HIT_LOOKUP_RE = re.compile(r"cache_lookup: hit_blocks=(\d+)")

STRESS_PROMPT = (
    "Distributed systems often rely on caching to reduce latency and improve "
    "throughput, but cache correctness depends on carefully managed ownership. "
    "A request may observe that a prefix is available, wait while other work is "
    "scheduled, and later attempt to use that same prefix after allocation. "
    "During this gap, the cache layer must avoid double-counting reservations, "
    "must release abandoned references, and must preserve the exact hash order "
    "used for the eventual load. " * 12
)


def _post_completion(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": stream,
    }
    url = f"http://localhost:{port}/v1/completions"
    if not stream:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                return "stream-started"
    return "stream-closed"


def _grep_log(log_text: str, patterns: tuple[str, ...]) -> str:
    lines = [line for line in log_text.splitlines() if any(pattern in line for pattern in patterns)]
    return "\n".join(lines[-80:])


def _metric_delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    return after.get(key, 0) - before.get(key, 0)


def test_warm_hit_pressure_on_single_gpu_profile(
    model: str,
    base_port: int,
    tmp_path: Path,
    max_model_len: int | None,
):
    """Concurrent repeated prompts should produce cache hits and clean releases."""

    log_dir = tmp_path / "warm_hit_stress_logs"
    log_dir.mkdir()
    pega_log = log_dir / "pegaflow-vllm.log"
    server_log = log_dir / "pegaflow-server.log"

    with PegaFlowServer(log_file=server_log) as pega_server:
        with VLLMServer(
            model,
            base_port,
            use_pegaflow=True,
            pegaflow_port=pega_server.grpc_port,
            log_file=pega_log,
            max_model_len=max_model_len or STRESS_MAX_MODEL_LEN,
            gpu_memory_utilization=STRESS_GPU_MEMORY_UTILIZATION,
            extra_args=["--max-num-seqs", str(STRESS_MAX_NUM_SEQS)],
            startup_timeout=240,
        ):
            print(f"[Stress] vLLM log: {pega_log}")
            print(f"[Stress] PegaFlow server log: {server_log}")

            metrics_start = fetch_pegaflow_metrics(pega_server.metrics_port)

            cold = call_openai_api(base_port, model, STRESS_PROMPT, max_tokens=32)["text"]
            warm = call_openai_api(base_port, model, STRESS_PROMPT, max_tokens=32)["text"]
            assert warm == cold

            futures = []
            with ThreadPoolExecutor(max_workers=STRESS_CONCURRENCY) as pool:
                for idx in range(STRESS_CONCURRENCY):
                    futures.append(
                        pool.submit(
                            _post_completion,
                            base_port,
                            model,
                            STRESS_PROMPT,
                            (
                                STRESS_LONG_MAX_TOKENS
                                if idx < STRESS_LONG_OUTPUT_WORKERS
                                else STRESS_SHORT_MAX_TOKENS
                            ),
                            stream=False,
                            timeout=120,
                        )
                    )

                completed = 0
                for future in as_completed(futures, timeout=240):
                    future.result()
                    completed += 1

            assert completed == len(futures)

            final = call_openai_api(base_port, model, STRESS_PROMPT, max_tokens=32)["text"]
            assert final == cold

            metrics_end = fetch_pegaflow_metrics(pega_server.metrics_port)

        pega_text = pega_log.read_text(errors="replace")
        server_text = server_log.read_text(errors="replace")
        interesting = _grep_log(
            pega_text,
            (
                "cache_lookup",
                "pending query lease",
                "Preempt",
                "preempt",
                "abort",
            ),
        )
        print("[Stress] Interesting vLLM log lines:\n" + interesting)

        warm_hit_lookups = [
            int(match.group(1))
            for match in WARM_HIT_LOOKUP_RE.finditer(pega_text)
            if int(match.group(1)) > 0
        ]
        assert len(warm_hit_lookups) >= MIN_WARM_HIT_LOOKUPS, (
            "Expected repeated warm cache lookups under concurrent pressure.\n"
            f"warm_hit_lookups={warm_hit_lookups}\n"
            f"vLLM log: {pega_log}\n"
            f"PegaFlow log: {server_log}\n"
            f"Interesting lines:\n{interesting}"
        )
        assert "cache_lookup: hit_blocks=0" in pega_text
        assert "pending query lease release failed" not in pega_text
        assert "pending query lease release exception" not in pega_text

        hits = _metric_delta(metrics_start, metrics_end, "pegaflow_cache_block_hits_total")
        loads = _metric_delta(metrics_start, metrics_end, "pegaflow_load_bytes_total")
        assert hits > 0 or loads > 0, (
            f"Expected warm-hit/load activity, got hits={hits}, loads={loads}.\n"
            f"Server log tail:\n{_grep_log(server_text, ('hit=', 'load', 'release'))}"
        )
