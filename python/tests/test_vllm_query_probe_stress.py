"""Stress E2E for scheduler query-probe idempotency.

This test intentionally creates warm-hit traffic under constrained KV capacity.
It is not a replacement for the correctness E2E; it is an observability-heavy
proof that repeated scheduler probes reuse the memoized query result and that
abandoned probes do not leave obvious release failures in the logs.
"""

from __future__ import annotations

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

STRESS_PROMPT = (
    "Distributed systems often rely on caching to reduce latency and improve "
    "throughput, but cache correctness depends on carefully managed ownership. "
    "A request may observe that a prefix is available, wait while other work is "
    "scheduled, and later attempt to use that same prefix after allocation. "
    "During this gap, the cache layer must avoid double-counting reservations, "
    "must release abandoned references, and must preserve the exact hash order "
    "used for the eventual load. " * 24
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


def test_query_probe_reuse_under_pressure(
    model: str,
    base_port: int,
    tmp_path: Path,
    max_model_len: int | None,
):
    """Warm-hit concurrent requests should show query-probe reuse in vLLM logs."""

    log_dir = tmp_path / "query_probe_stress_logs"
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
            max_model_len=max_model_len or 4096,
            gpu_memory_utilization=0.3,
            extra_args=["--max-num-seqs", "96"],
            startup_timeout=240,
        ):
            print(f"[Stress] vLLM log: {pega_log}")
            print(f"[Stress] PegaFlow server log: {server_log}")

            metrics_start = fetch_pegaflow_metrics(pega_server.metrics_port)

            cold = call_openai_api(base_port, model, STRESS_PROMPT, max_tokens=32)["text"]
            warm = call_openai_api(base_port, model, STRESS_PROMPT, max_tokens=32)["text"]
            assert warm == cold

            futures = []
            with ThreadPoolExecutor(max_workers=40) as pool:
                for idx in range(40):
                    futures.append(
                        pool.submit(
                            _post_completion,
                            base_port,
                            model,
                            STRESS_PROMPT,
                            192 if idx < 32 else 64,
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
                "pending query unpin",
                "Preempt",
                "preempt",
                "abort",
            ),
        )
        print("[Stress] Interesting vLLM log lines:\n" + interesting)

        assert "cache_lookup_reuse" in pega_text, (
            "Expected repeated scheduler probe reuse in vLLM log.\n"
            f"vLLM log: {pega_log}\n"
            f"PegaFlow log: {server_log}\n"
            f"Interesting lines:\n{interesting}"
        )
        assert "cache_lookup: hit_blocks=0" in pega_text
        assert "pending query unpin failed" not in pega_text
        assert "pending query unpin exception" not in pega_text

        hits = _metric_delta(metrics_start, metrics_end, "pegaflow_cache_block_hits_total")
        loads = _metric_delta(metrics_start, metrics_end, "pegaflow_load_bytes_total")
        assert hits > 0 or loads > 0, (
            f"Expected warm-hit/load activity, got hits={hits}, loads={loads}.\n"
            f"Server log tail:\n{_grep_log(server_text, ('hit=', 'load', 'unpin'))}"
        )
