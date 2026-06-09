"""Shared helpers for P/D-specific vLLM connector facades."""

from __future__ import annotations

from typing import Any

from pegaflow.pd_connector.metrics import PdKVConnectorStats, PdMetricsTracker, PdPromMetrics
from pegaflow.pd_connector.worker import model_uses_mla


def assert_supported_config(vllm_config: Any) -> None:
    if not model_uses_mla(vllm_config):
        return
    parallel_config = getattr(vllm_config, "parallel_config", None)
    dcp_world_size = int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1)
    pcp_world_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
    assert dcp_world_size == 1, (
        "PdConnector MLA first version requires decode_context_parallel_size == 1"
    )
    assert pcp_world_size == 1, (
        "PdConnector MLA first version requires prefill_context_parallel_size == 1"
    )


class PdConnectorClassMixin:
    _metrics: PdMetricsTracker

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        if model_uses_mla(vllm_config):
            return None
        return "HND"

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return not bool(extra_config.get("pegaflow.pd.allow_full_decode_cudagraph", False))

    def get_kv_connector_stats(self) -> PdKVConnectorStats | None:
        return self._metrics.get_stats()

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> PdKVConnectorStats | None:
        if data is None:
            return None
        return PdKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ) -> PdPromMetrics:
        return PdPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)
