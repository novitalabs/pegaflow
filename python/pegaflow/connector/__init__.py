"""
Facade for the PegaFlow vLLM connector, split into scheduler/worker implementations.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from pegaflow.connector.common import (
    ConnectorContext,
    PegaConnectorMetadata,
    RequestPhase,
    RequestTracker,
    derive_namespace,
    logger,
    resolve_instance_id,
)
from pegaflow.connector.scheduler import SchedulerConnector
from pegaflow.connector.state_manager import ServiceStateManager
from pegaflow.connector.worker import WorkerConnector
from pegaflow.logging_utils import timing_wrapper
from pegaflow.pegaflow import EngineRpcClient


class PegaKVConnector(KVConnectorBase_V1):
    """v1 KV connector for PegaFlow with separated scheduler/worker logic."""

    def __init__(self, vllm_config, role: KVConnectorRole):
        super().__init__(vllm_config, role)

        instance_id = resolve_instance_id(vllm_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        namespace = derive_namespace(vllm_config, tp_size)
        num_layers = getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 0)
        block_size = vllm_config.cache_config.block_size

        tp_rank: int | None = None
        device_id: int | None = None
        if role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            if torch.cuda.is_available():
                device_id = _resolve_device_id()

        assert vllm_config.kv_transfer_config is not None
        server_host = os.environ.get(
            "PEGAFLOW_HOST"
        ) or vllm_config.kv_transfer_config.get_from_extra_config(
            "pegaflow.host", "http://127.0.0.1"
        )
        server_port = os.environ.get(
            "PEGAFLOW_PORT"
        ) or vllm_config.kv_transfer_config.get_from_extra_config("pegaflow.port", 50055)
        self._engine_endpoint = f"{server_host}:{server_port}"
        engine_client = EngineRpcClient(self._engine_endpoint)
        logger.info("[PegaKVConnector] Connected to engine server at %s", self._engine_endpoint)

        self._state_manager = ServiceStateManager(engine_client)

        self._ctx = ConnectorContext(
            instance_id=instance_id,
            namespace=namespace,
            block_size=block_size,
            num_layers=num_layers,
            tp_size=tp_size,
            tp_rank=tp_rank,
            device_id=device_id,
            engine_client=engine_client,
            state_manager=self._state_manager,
        )

        self._scheduler: SchedulerConnector | None = None
        self._worker: WorkerConnector | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = SchedulerConnector(self._ctx)
        else:
            self._worker = WorkerConnector(self._ctx)

        logger.info(
            "[PegaKVConnector] Initialized role=%s instance_id=%s device=%s tp_rank=%s tp_size=%d layers=%d namespace=%s",
            role.name,
            instance_id,
            device_id if device_id is not None else "cpu",
            tp_rank if tp_rank is not None else "N/A",
            tp_size,
            num_layers,
            namespace,
        )

    # ==============================
    # Worker-side methods
    # ==============================
    @timing_wrapper
    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        if not self._worker:
            return
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        self._worker.start_load_kv(metadata, forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._worker:
            return
        self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        if not self._worker:
            return
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        self._worker.save_kv_layer(metadata, layer_name, kv_layer, attn_metadata, **kwargs)

    @timing_wrapper
    def wait_for_save(self) -> None:
        if not self._worker:
            return
        self._worker.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if not self._worker:
            return (None, None)
        return self._worker.get_finished(finished_req_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if not self._worker:
            return
        self._worker.register_kv_caches(kv_caches)

    def unregister_context(self) -> None:
        if self._worker:
            self._worker.unregister_context()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def update_connector_output(self, connector_output) -> None:
        if self._scheduler:
            self._scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler:
            return self._scheduler.request_finished(request, block_ids)
        return (False, None)

    def take_events(self) -> Iterable:
        return ()

    @timing_wrapper
    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._scheduler:
            return (0, False)
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    @timing_wrapper
    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        if self._scheduler:
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    @timing_wrapper
    def build_connector_meta(self, scheduler_output) -> PegaConnectorMetadata:
        if not self._scheduler:
            return PegaConnectorMetadata()
        return self._scheduler.build_connector_meta(scheduler_output)

    # ==============================
    # Defaults and shutdown
    # ==============================
    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_connector_stats(self):
        return None

    def get_handshake_metadata(self):
        return None

    def set_host_xfer_buffer_ops(self, copy_operation):
        return

    def get_finished_count(self) -> int | None:
        return None

    def shutdown(self):
        if self._worker:
            self._worker.shutdown()
        if self._state_manager:
            self._state_manager.shutdown()


def _resolve_device_id() -> int:
    """
    Return the global CUDA device id even when CUDA_VISIBLE_DEVICES masks GPUs.

    torch.cuda.current_device() returns the local index within the visible set,
    but we need the actual global device ID for operations like CUDA IPC.
    This function maps the local index back to the global device ID.
    """
    local_id = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return local_id

    slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
    try:
        mapped = slots[local_id]
    except IndexError:
        return local_id

    try:
        return int(mapped)
    except ValueError:
        return local_id


__all__ = ["PegaKVConnector", "KVConnectorRole", "RequestPhase", "RequestTracker"]
