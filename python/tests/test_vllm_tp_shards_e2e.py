"""E2E coverage for one TP replica split across two PegaFlow servers."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest

from .vllm_helpers import (
    PegaFlowServer,
    VLLMServer,
    call_openai_api,
    fetch_pegaflow_metrics,
    fetch_pegaflow_rpc_failures,
)

PROMPT = (
    "Distributed inference splits one tensor-parallel model replica across GPUs. "
    "Each rank owns a different slice of the attention state, while the scheduler "
    "must treat a cached prefix as reusable only when every rank can load it. "
) * 8
REQUEST_COUNT = 96
CONCURRENCY = 8
POOL_SIZE = "2gb"
PROMPTS = tuple(
    PROMPT
    + (
        f" Workload variant {index} verifies an independent suffix while preserving "
        "the shared distributed-inference prefix."
    )
    * 4
    for index in range(REQUEST_COUNT)
)


def _test_devices(tensor_parallel_size: int) -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
        if not all(slot.isdigit() for slot in slots):
            pytest.skip("TP-shard E2E requires numeric CUDA_VISIBLE_DEVICES entries")
        devices = [int(slot) for slot in slots]
    else:
        torch = pytest.importorskip("torch")
        devices = list(range(torch.cuda.device_count()))

    if len(devices) < tensor_parallel_size:
        pytest.skip(f"TP{tensor_parallel_size} requires {tensor_parallel_size} visible GPUs")
    return devices[:tensor_parallel_size]


def _activity_delta(before: dict[str, float], after: dict[str, float], *names: str) -> float:
    return sum(after.get(name, 0) - before.get(name, 0) for name in names)


def _run_traffic(port: int, model: str) -> tuple[list[str], float, float, float]:
    def send(prompt: str) -> tuple[str, float]:
        started = time.perf_counter()
        result = call_openai_api(port, model, prompt, max_tokens=8)
        return result["text"], time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        samples = list(executor.map(send, PROMPTS))
    elapsed = time.perf_counter() - started
    latencies = sorted(latency for _, latency in samples)
    return (
        [output for output, _ in samples],
        elapsed,
        latencies[len(latencies) // 2],
        latencies[int(len(latencies) * 0.95) - 1],
    )


@pytest.mark.e2e
@pytest.mark.gpu
def test_vllm_tp_replica_uses_every_pegaflow_shard(
    model: str,
    base_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    max_model_len: int | None,
    pegaflow_transfer_backend: str,
    tmp_path: Path,
):
    if tensor_parallel_size < 2 or tensor_parallel_size % 2:
        pytest.skip("TP-shard E2E requires an even tensor parallel size of at least two")
    if pipeline_parallel_size != 1:
        pytest.skip("TP-shard E2E covers TP-only parallelism")

    server_binary = Path(__file__).parents[2] / "target/release/pegaflow-server"
    if not server_binary.is_file():
        pytest.skip("TP-shard E2E requires a release pegaflow-server build")

    devices = _test_devices(tensor_parallel_size)
    middle = tensor_parallel_size // 2
    device_shards = (devices[:middle], devices[middle:])

    with ExitStack() as stack:
        servers = tuple(
            stack.enter_context(
                PegaFlowServer(
                    log_file=tmp_path / f"pegaflow-shard-{index}.log",
                    pool_size=POOL_SIZE,
                    devices=",".join(map(str, shard)),
                    server_binary=str(server_binary),
                )
            )
            for index, shard in enumerate(device_shards)
        )
        endpoints = [f"http://127.0.0.1:{server.grpc_port}" for server in servers]
        kv_config = {
            "kv_connector": "PegaKVConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "pegaflow.connector",
            "kv_connector_extra_config": {
                "pegaflow.tp_shard_endpoints": endpoints,
                "pegaflow.transfer_backend": pegaflow_transfer_backend,
            },
        }
        metrics_before = [fetch_pegaflow_metrics(server.metrics_port) for server in servers]

        with VLLMServer(
            model,
            base_port,
            kv_transfer_config=kv_config,
            log_file=tmp_path / "vllm-cold.log",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            server_label="PegaFlow TP-shard cold",
        ):
            cold_outputs, cold_seconds, cold_p50, cold_p95 = _run_traffic(base_port, model)
        metrics_after_save = [fetch_pegaflow_metrics(server.metrics_port) for server in servers]

        with VLLMServer(
            model,
            base_port,
            kv_transfer_config=kv_config,
            log_file=tmp_path / "vllm-warm.log",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            server_label="PegaFlow TP-shard warm",
        ):
            warm_outputs, warm_seconds, warm_p50, warm_p95 = _run_traffic(base_port, model)
        metrics_after_load = [fetch_pegaflow_metrics(server.metrics_port) for server in servers]

        assert warm_outputs == cold_outputs
        for index, server in enumerate(servers):
            save_activity = _activity_delta(
                metrics_before[index],
                metrics_after_save[index],
                "pegaflow_save_bytes_total",
                "pegaflow_cache_block_insertions_total",
            )
            load_activity = _activity_delta(
                metrics_after_save[index],
                metrics_after_load[index],
                "pegaflow_load_bytes_total",
                "pegaflow_cache_block_hits_total",
            )
            assert save_activity > 0, f"PegaFlow shard {index} saved no KV blocks"
            assert load_activity > 0, f"PegaFlow shard {index} loaded no KV blocks"
            failures = fetch_pegaflow_rpc_failures(server.metrics_port)
            assert not failures, f"PegaFlow shard {index} RPC failures: {failures}"

        print(
            f"TP-shard E2E traffic ({REQUEST_COUNT} requests, concurrency={CONCURRENCY}): "
            f"cold={REQUEST_COUNT / cold_seconds:.2f} req/s "
            f"p50={cold_p50:.3f}s p95={cold_p95:.3f}s; "
            f"warm={REQUEST_COUNT / warm_seconds:.2f} req/s "
            f"p50={warm_p50:.3f}s p95={warm_p95:.3f}s"
        )
