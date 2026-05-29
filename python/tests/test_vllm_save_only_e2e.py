"""E2E coverage for PegaFlow save_only behind an external full-hit connector.

This test uses vLLM's DecodeBenchConnector as a deterministic stand-in for the
decode-side connector that claims a full prompt hit. The expected ownership is:

    MultiConnector child 0: DecodeBenchConnector -> load owner
    MultiConnector child 1: PegaKVConnector(save_only) -> no read, save only

Then a second vLLM instance with normal PegaKVConnector should load the blocks
saved by the save_only instance.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from .vllm_helpers import (
    PegaFlowServer,
    VLLMServer,
    call_openai_api,
    fetch_pegaflow_metrics,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

SAVE_ONLY_BLOCK_SIZE = 16
SAVE_ONLY_FULL_BLOCKS = 4
SAVE_ONLY_PROMPT_TOKENS = SAVE_ONLY_BLOCK_SIZE * SAVE_ONLY_FULL_BLOCKS + 1
SAVE_ONLY_MAX_TOKENS = 24
SAVE_ONLY_MAX_MODEL_LEN = 160
SAVE_ONLY_STARTUP_TIMEOUT = 600

SAVE_ONLY_PROMPT = [100 + (idx % 200) for idx in range(SAVE_ONLY_PROMPT_TOKENS)]


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    key: str,
) -> float:
    return after.get(key, 0) - before.get(key, 0)


def _wait_for_any_metric_delta(
    metrics_port: int,
    before: dict[str, float],
    keys: tuple[str, ...],
    *,
    timeout_s: float = 30,
) -> dict[str, float]:
    deadline = time.time() + timeout_s
    latest = before
    while time.time() < deadline:
        latest = fetch_pegaflow_metrics(metrics_port)
        if any(_metric_delta(before, latest, key) > 0 for key in keys):
            return latest
        time.sleep(1)
    return latest


def _save_only_multi_config(pegaflow_port: int) -> dict[str, Any]:
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "DecodeBenchConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "fill_mean": 0.015,
                        "fill_std": 0.0,
                    },
                },
                {
                    "kv_connector": "PegaKVConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "pegaflow.connector",
                    "kv_connector_extra_config": {
                        "pegaflow.port": pegaflow_port,
                        "pegaflow.mode": "save_only",
                    },
                },
            ],
        },
    }


def _pegaflow_read_write_config(pegaflow_port: int) -> dict[str, Any]:
    return {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
        "kv_connector_extra_config": {
            "pegaflow.port": pegaflow_port,
        },
    }


def test_save_only_external_full_hit_is_saved_for_later_pegaflow_read(
    model: str,
    base_port: int,
    tmp_path: Path,
    max_model_len: int | None,
):
    """save_only must not query PegaFlow, but must save external-hit blocks."""

    required_model_len = SAVE_ONLY_PROMPT_TOKENS + SAVE_ONLY_MAX_TOKENS
    if max_model_len is not None and max_model_len < required_model_len:
        pytest.skip(f"--max-model-len={max_model_len} is smaller than {required_model_len}")

    prompt = SAVE_ONLY_PROMPT
    assert len(prompt) == SAVE_ONLY_PROMPT_TOKENS
    assert (len(prompt) - 1) % SAVE_ONLY_BLOCK_SIZE == 0

    log_dir = tmp_path / "save_only_e2e_logs"
    log_dir.mkdir()
    server_log = log_dir / "pegaflow-server.log"
    save_only_log = log_dir / "save-only-multi.log"
    read_write_log = log_dir / "pegaflow-read-write.log"

    vllm_extra_args = [
        "--block-size",
        str(SAVE_ONLY_BLOCK_SIZE),
        "--max-num-seqs",
        "2",
        "--enforce-eager",
    ]
    vllm_env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    server_max_model_len = max_model_len or SAVE_ONLY_MAX_MODEL_LEN

    with PegaFlowServer(
        log_file=server_log,
        pool_size="1gb",
        log_level="debug",
    ) as pega_server:
        save_port = base_port
        read_port = base_port + 1

        with VLLMServer(
            model,
            save_port,
            pegaflow_port=pega_server.grpc_port,
            log_file=save_only_log,
            max_model_len=server_max_model_len,
            extra_args=vllm_extra_args,
            startup_timeout=SAVE_ONLY_STARTUP_TIMEOUT,
            kv_transfer_config=_save_only_multi_config(pega_server.grpc_port),
            server_label="SaveOnlyMulti",
            env_overrides=vllm_env,
        ):
            save_metrics_before = fetch_pegaflow_metrics(pega_server.metrics_port)
            save_only_result = call_openai_api(
                save_port,
                model,
                prompt,
                max_tokens=SAVE_ONLY_MAX_TOKENS,
            )
            save_metrics_after = _wait_for_any_metric_delta(
                pega_server.metrics_port,
                save_metrics_before,
                (
                    "pegaflow_save_bytes_total",
                    "pegaflow_cache_block_insertions_total",
                ),
            )

        with VLLMServer(
            model,
            read_port,
            pegaflow_port=pega_server.grpc_port,
            log_file=read_write_log,
            max_model_len=server_max_model_len,
            extra_args=vllm_extra_args,
            startup_timeout=SAVE_ONLY_STARTUP_TIMEOUT,
            kv_transfer_config=_pegaflow_read_write_config(pega_server.grpc_port),
            server_label="PegaFlowReadWrite",
            env_overrides=vllm_env,
        ):
            read_metrics_before = fetch_pegaflow_metrics(pega_server.metrics_port)
            read_write_result = call_openai_api(
                read_port,
                model,
                prompt,
                max_tokens=SAVE_ONLY_MAX_TOKENS,
            )
            read_metrics_after = _wait_for_any_metric_delta(
                pega_server.metrics_port,
                read_metrics_before,
                (
                    "pegaflow_load_bytes_total",
                    "pegaflow_cache_block_hits_total",
                ),
            )

    assert save_only_result["text"] == read_write_result["text"]

    save_bytes = _metric_delta(save_metrics_before, save_metrics_after, "pegaflow_save_bytes_total")
    insertions = _metric_delta(
        save_metrics_before,
        save_metrics_after,
        "pegaflow_cache_block_insertions_total",
    )
    assert save_bytes > 0 or insertions > 0, (
        f"save_only phase produced no PegaFlow save activity: "
        f"save_bytes={save_bytes}, insertions={insertions}; logs={log_dir}"
    )

    save_hits = _metric_delta(
        save_metrics_before,
        save_metrics_after,
        "pegaflow_cache_block_hits_total",
    )
    save_load_bytes = _metric_delta(
        save_metrics_before,
        save_metrics_after,
        "pegaflow_load_bytes_total",
    )
    assert save_hits == 0 and save_load_bytes == 0, (
        f"save_only phase unexpectedly read from PegaFlow: "
        f"hits={save_hits}, load_bytes={save_load_bytes}; logs={log_dir}"
    )

    read_hits = _metric_delta(
        read_metrics_before,
        read_metrics_after,
        "pegaflow_cache_block_hits_total",
    )
    read_load_bytes = _metric_delta(
        read_metrics_before,
        read_metrics_after,
        "pegaflow_load_bytes_total",
    )
    print(
        "[save_only metrics] "
        f"save_bytes={save_bytes:.0f} insertions={insertions:.0f} "
        f"hits={save_hits:.0f} load_bytes={save_load_bytes:.0f}"
    )
    print(f"[read_write metrics] hits={read_hits:.0f} load_bytes={read_load_bytes:.0f}")
    assert read_hits > 0 or read_load_bytes > 0, (
        f"read_write phase did not load saved blocks: "
        f"hits={read_hits}, load_bytes={read_load_bytes}; logs={log_dir}"
    )

    save_log_text = save_only_log.read_text(errors="replace")
    read_log_text = read_write_log.read_text(errors="replace")
    assert "cache_lookup:" not in save_log_text
    assert "cache_lookup: hit_blocks=" in read_log_text
