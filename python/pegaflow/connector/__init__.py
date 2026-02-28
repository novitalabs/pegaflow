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
    PegaKVConnectorStats,
    PegaPromMetrics,
    derive_namespace,
    detect_mla,
    logger,
    resolve_instance_id,
)
from pegaflow.connector.scheduler import SchedulerConnector
from pegaflow.connector.state_manager import ServiceStateManager
from pegaflow.connector.worker import WorkerConnector
from pegaflow.pegaflow import EngineRpcClient


class PegaKVConnector(KVConnectorBase_V1):
    """v1 KV connector for PegaFlow with separated scheduler/worker logic."""

    def __init__(self, vllm_config, role: KVConnectorRole):
        super().__init__(vllm_config, role)

        instance_id = resolve_instance_id(vllm_config)
        global_tp_size = vllm_config.parallel_config.tensor_parallel_size
        global_world_size = vllm_config.parallel_config.world_size
        is_mla = detect_mla(vllm_config)
        # Namespace uses global tp_size for consistent hashing across all ranks
        effective_tp_size = 1 if is_mla else global_tp_size
        namespace = derive_namespace(vllm_config, effective_tp_size)
        num_layers = getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 0)
        block_size = vllm_config.cache_config.block_size

        # Resolve server endpoints (supports multi-server for cross-node TP)
        endpoints = _resolve_endpoints(vllm_config)
        num_servers = len(endpoints)

        if num_servers > 1:
            assert global_tp_size % num_servers == 0, (
                f"tensor_parallel_size ({global_tp_size}) must be divisible by "
                f"number of servers ({num_servers})"
            )
            assert global_world_size % num_servers == 0, (
                f"world_size ({global_world_size}) must be divisible by "
                f"number of servers ({num_servers})"
            )

        # Each server manages a local subset of ranks
        local_tp_size = global_tp_size // num_servers
        local_world_size = global_world_size // num_servers

        tp_rank: int | None = None
        device_id: int | None = None
        engine_clients: tuple[EngineRpcClient, ...] = ()

        if role == KVConnectorRole.WORKER:
            global_tp_rank = get_tensor_model_parallel_rank()
            server_index = global_tp_rank // local_tp_size
            tp_rank = global_tp_rank % local_tp_size
            if torch.cuda.is_available():
                device_id = _resolve_device_id()
            engine_client = EngineRpcClient(endpoints[server_index])
            logger.info(
                "[PegaKVConnector] Worker connected to server %d/%d at %s "
                "(global_rank=%d, local_rank=%d)",
                server_index,
                num_servers,
                endpoints[server_index],
                global_tp_rank,
                tp_rank,
            )
        else:
            # Scheduler connects to all servers for coordinated queries
            engine_clients = tuple(EngineRpcClient(ep) for ep in endpoints)
            engine_client = engine_clients[0]
            logger.info(
                "[PegaKVConnector] Scheduler connected to %d server(s): %s",
                num_servers,
                endpoints,
            )

        self._state_manager = ServiceStateManager(engine_client)

        self._ctx = ConnectorContext(
            instance_id=instance_id,
            namespace=namespace,
            block_size=block_size,
            num_layers=num_layers,
            tp_size=local_tp_size,
            world_size=local_world_size,
            tp_rank=tp_rank,
            device_id=device_id,
            engine_client=engine_client,
            state_manager=self._state_manager,
            is_mla=is_mla,
            engine_clients=engine_clients,
        )

        self._scheduler: SchedulerConnector | None = None
        self._worker: WorkerConnector | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = SchedulerConnector(self._ctx)
        else:
            self._worker = WorkerConnector(self._ctx)

        logger.info(
            "[PegaKVConnector] Initialized role=%s instance_id=%s device=%s "
            "tp_rank=%s tp_size=%d world_size=%d layers=%d namespace=%s "
            "is_mla=%s num_servers=%d",
            role.name,
            instance_id,
            device_id if device_id is not None else "cpu",
            tp_rank if tp_rank is not None else "N/A",
            local_tp_size,
            local_world_size,
            num_layers,
            namespace,
            is_mla,
            num_servers,
        )

    # ==============================
    # Worker-side methods
    # ==============================
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

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self._worker:
            self._worker.handle_preemptions(preempted_req_ids)

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

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._scheduler:
            return (0, False)
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        if self._scheduler:
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output) -> PegaConnectorMetadata:
        if not self._scheduler:
            return PegaConnectorMetadata()
        return self._scheduler.build_connector_meta(scheduler_output)

    # ==============================
    # Defaults and shutdown
    # ==============================
    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_connector_stats(self) -> PegaKVConnectorStats | None:
        stats: PegaKVConnectorStats | None = None

        # Collect scheduler-side stats
        if self._scheduler:
            stats = self._scheduler.get_stats()

        # Collect worker-side stats
        if self._worker:
            worker_stats = self._worker.get_stats()
            if worker_stats is not None:
                stats = worker_stats if stats is None else stats.aggregate(worker_stats)

        return stats

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> PegaKVConnectorStats | None:
        if data is None:
            return None
        return PegaKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ) -> PegaPromMetrics:
        return PegaPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

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


def _resolve_endpoints(vllm_config) -> list[str]:
    """Resolve PegaFlow server endpoints.

    Priority:
        1. PEGAFLOW_ENDPOINTS env var (comma-separated)
        2. pegaflow.endpoints in kv_transfer_config.extra_config
        3. PEGAFLOW_HOST + PEGAFLOW_PORT (single server, backward compatible)
    """
    assert vllm_config.kv_transfer_config is not None

    # 1. PEGAFLOW_ENDPOINTS env var
    endpoints_env = os.environ.get("PEGAFLOW_ENDPOINTS")
    if endpoints_env:
        endpoints = [ep.strip() for ep in endpoints_env.split(",") if ep.strip()]
        if endpoints:
            return endpoints
        logger.warning(
            "[PegaKVConnector] PEGAFLOW_ENDPOINTS is set but contains no valid "
            "endpoints: %r, falling back to host+port",
            endpoints_env,
        )

    # 2. pegaflow.endpoints extra_config
    endpoints_config = vllm_config.kv_transfer_config.get_from_extra_config(
        "pegaflow.endpoints", None
    )
    if endpoints_config and isinstance(endpoints_config, list):
        endpoints = [str(ep).strip() for ep in endpoints_config if str(ep).strip()]
        if endpoints:
            return endpoints

    # 3. Fallback: single endpoint from host + port
    server_host = os.environ.get(
        "PEGAFLOW_HOST"
    ) or vllm_config.kv_transfer_config.get_from_extra_config(
        "pegaflow.host", "http://127.0.0.1"
    )
    server_port = os.environ.get(
        "PEGAFLOW_PORT"
    ) or vllm_config.kv_transfer_config.get_from_extra_config("pegaflow.port", 50055)
    return [f"{server_host}:{server_port}"]


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


__all__ = ["PegaKVConnector", "KVConnectorRole"]
