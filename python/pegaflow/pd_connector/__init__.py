"""Experimental P/D RDMA-push vLLM connector."""

from __future__ import annotations

from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from pegaflow.pd_connector.metadata import PdConnectorMetadata
from pegaflow.pd_connector.metrics import PdKVConnectorStats, PdMetricsTracker, PdPromMetrics
from pegaflow.pd_connector.scheduler import PdSchedulerConnector
from pegaflow.pd_connector.worker import PdWorkerConnector, model_uses_mla


class PdConnector(KVConnectorBase_V1, SupportsHMA):
    """Thin vLLM facade for the experimental P/D push connector."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        _assert_supported_config(vllm_config)
        self._scheduler: PdSchedulerConnector | None = None
        self._worker: PdWorkerConnector | None = None
        self._metrics = PdMetricsTracker()
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = PdSchedulerConnector(vllm_config, metrics=self._metrics)
        elif role == KVConnectorRole.WORKER:
            self._worker = PdWorkerConnector(
                vllm_config,
                kv_cache_config=kv_cache_config,
                metrics=self._metrics,
            )
        else:
            raise ValueError(f"unsupported KV connector role: {role}")

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        if model_uses_mla(vllm_config):
            return None
        return "HND"

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return not bool(extra_config.get("pegaflow.pd.allow_full_decode_cudagraph", False))

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self._worker is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PdConnectorMetadata)
        self._worker.start_load_kv(metadata, forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker is not None:
            self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if self._worker is not None:
            self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self._worker is None:
            return None, None
        return self._worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self._worker is None:
            return set()
        return self._worker.get_block_ids_with_load_errors()

    def build_connector_worker_meta(self) -> Any | None:
        if self._worker is None:
            return None
        return self._worker.build_connector_worker_meta()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
        if self._scheduler is not None:
            self._scheduler.shutdown()

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        if self._scheduler is not None:
            self._scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: Any,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)

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


__all__ = ["PdConnector"]


def _assert_supported_config(vllm_config: Any) -> None:
    _assert_mtp_not_supported(vllm_config)
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


def _assert_mtp_not_supported(vllm_config: Any) -> None:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_text_config", None)
    candidates = (model_config, hf_config)
    numeric_fields = (
        "num_nextn_predict_layers",
        "num_mtp_layers",
        "mtp_num_layers",
    )
    for config in candidates:
        if config is None:
            continue
        if bool(getattr(config, "use_mtp", False)):
            raise AssertionError("PdConnector does not support MTP layout")
        for field in numeric_fields:
            if int(getattr(config, field, 0) or 0) > 0:
                raise AssertionError("PdConnector does not support MTP layout")
