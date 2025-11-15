from __future__ import annotations

"""Baseline vLLM v1 KV connector for local development.

This module defines :class:`PegaKVConnector`, a thin subclass of
``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``.

At the moment it only mirrors the abstract API and raises
``NotImplementedError`` in all required methods, so that we have a
self-contained place inside this repo to start iterating on our own
PegaFlow-backed connector implementation.

Usage example (scheduler/worker side)::

    from pegaflow import PegaKVConnector, KVConnectorRole

    connector = PegaKVConnector(vllm_config, KVConnectorRole.WORKER)

Later we can register this class as a dynamic connector in vLLM by
referencing it via its full import path.
"""

from typing import Any, Optional, Tuple

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorMetadata,
)


class PegaKVConnector(KVConnectorBase_V1):
    """Skeleton v1 KV connector for PegaFlow.

    This class intentionally keeps the same method signatures as
    :class:`KVConnectorBase_V1` so that it can be used as a drop-in
    implementation once we fill in the logic. All abstract methods
    currently raise :class:`NotImplementedError`.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        """Create a new PegaKVConnector.

        Args:
            vllm_config: vLLM configuration object.
            role: Whether this connector instance runs in the scheduler
                process or the worker process.
        """
        super().__init__(vllm_config, role)
        # TODO: wire in PegaFlow client / configuration here.

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start asynchronously loading KV from external storage.

        This should load KV into vLLM's paged KV buffer based on the
        connector metadata bound via ``bind_connector_metadata``.
        """
        raise NotImplementedError("start_load_kv is not implemented yet in PegaKVConnector")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until KV for a specific layer has finished loading."""
        raise NotImplementedError("wait_for_layer_load is not implemented yet in PegaKVConnector")

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",  # type: ignore[name-defined]
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Start asynchronously saving KV for a single layer.

        ``kv_layer`` is the paged KV buffer for the current layer in vLLM.
        """
        raise NotImplementedError("save_kv_layer is not implemented yet in PegaKVConnector")

    def wait_for_save(self) -> None:
        """Block until all outstanding save operations have completed."""
        raise NotImplementedError("wait_for_save is not implemented yet in PegaKVConnector")

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Tuple[Optional[int], bool]:
        """Return how many external tokens can be loaded for this request.

        Returns a tuple ``(num_external_tokens, will_load_async)``. For more
        details see the vLLM ``KVConnectorBase_V1`` documentation.
        """
        raise NotImplementedError("get_num_new_matched_tokens is not implemented yet in PegaKVConnector")

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Update connector internal state after KV block allocation."""
        raise NotImplementedError("update_state_after_alloc is not implemented yet in PegaKVConnector")

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        """Build connector metadata for the current scheduler step."""
        raise NotImplementedError("build_connector_meta is not implemented yet in PegaKVConnector")


__all__ = ["PegaKVConnector", "KVConnectorRole"]

