"""Experimental P/D RDMA-push vLLM connectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from pegaflow.pd_connector.base_connector import (
    PdConnectorClassMixin,
    assert_supported_config,
)
from pegaflow.pd_connector.metadata import PdConnectorMetadata
from pegaflow.pd_connector.metrics import PdMetricsTracker
from pegaflow.pd_connector.scheduler import (
    PdDecodeSchedulerConnector,
    PdPrefillSchedulerConnector,
)
from pegaflow.pd_connector.worker import (
    PdDecodeWorkerConnector,
    PdPrefillWorkerConnector,
)

if TYPE_CHECKING:
    import torch


class PdDecodeConnector(PdConnectorClassMixin, KVConnectorBase_V1, SupportsHMA):
    """Decode-side vLLM connector for P/D RDMA push."""

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return False

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert_supported_config(vllm_config)
        self._scheduler: PdDecodeSchedulerConnector | None = None
        self._worker: PdDecodeWorkerConnector | None = None
        self._metrics = PdMetricsTracker()
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = PdDecodeSchedulerConnector(vllm_config, metrics=self._metrics)
        elif role == KVConnectorRole.WORKER:
            self._worker = PdDecodeWorkerConnector(
                vllm_config,
                kv_cache_config=kv_cache_config,
                metrics=self._metrics,
            )
        else:
            raise ValueError(f"unsupported KV connector role: {role}")

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
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
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

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


class PdPrefillConnector(PdConnectorClassMixin, KVConnectorBase_V1, SupportsHMA):
    """Prefill-side vLLM connector for P/D RDMA push."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert_supported_config(vllm_config)
        self._scheduler: PdPrefillSchedulerConnector | None = None
        self._worker: PdPrefillWorkerConnector | None = None
        self._metrics = PdMetricsTracker()
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = PdPrefillSchedulerConnector(vllm_config, metrics=self._metrics)
        elif role == KVConnectorRole.WORKER:
            self._worker = PdPrefillWorkerConnector(
                vllm_config,
                kv_cache_config=kv_cache_config,
                metrics=self._metrics,
            )
        else:
            raise ValueError(f"unsupported KV connector role: {role}")

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self._worker is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PdConnectorMetadata)
        self._worker.start_load_kv(metadata, forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
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
        return set()

    def build_connector_worker_meta(self) -> Any | None:
        return None

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


class PdConnector(PdConnectorClassMixin, KVConnectorBase_V1, SupportsHMA):
    """Legacy facade that selects the split P/D connector from engine_id."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        engine_id = str(getattr(vllm_config.kv_transfer_config, "engine_id", "") or "")
        if engine_id.startswith("d"):
            self._delegate = PdDecodeConnector(vllm_config, role, kv_cache_config)
        elif engine_id.startswith("p"):
            self._delegate = PdPrefillConnector(vllm_config, role, kv_cache_config)
        else:
            raise ValueError(
                "PdConnector requires engine_id to start with 'd' or 'p'; "
                f"engine_id={engine_id!r}"
            )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return PdConnectorClassMixin.requires_piecewise_for_cudagraph(extra_config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        self._connector_metadata = connector_metadata
        bind = getattr(self._delegate, "bind_connector_metadata", None)
        if bind is not None:
            bind(connector_metadata)
        else:
            self._delegate._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None
        clear = getattr(self._delegate, "clear_connector_metadata", None)
        if clear is not None:
            clear()
        else:
            self._delegate._connector_metadata = None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        return self._delegate.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return self._delegate.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return self._delegate.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return self._delegate.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        return self._delegate.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        return self._delegate.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._delegate.get_block_ids_with_load_errors()

    def build_connector_worker_meta(self) -> Any | None:
        return self._delegate.build_connector_worker_meta()

    def shutdown(self) -> None:
        return self._delegate.shutdown()

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._delegate.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        return self._delegate.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        return self._delegate.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any) -> None:
        return self._delegate.update_connector_output(connector_output)

    def request_finished(
        self,
        request: Any,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._delegate.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._delegate.request_finished_all_groups(request, block_ids)


__all__ = ["PdConnector", "PdDecodeConnector", "PdPrefillConnector"]
