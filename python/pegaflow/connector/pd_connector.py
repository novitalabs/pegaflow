"""
PegaPdConnector: MultiConnector combining PegaKVConnector + NixlConnector
for Prefill-Decode disaggregated serving with PegaFlow KV cache management.

Architecture:
    _connectors[0] = PegaKVConnector  (L2 cache: CPU/SSD via PegaFlow server)
    _connectors[1] = NixlConnector    (P->D KV transfer via RDMA)

Usage in --kv-transfer-config:
    {
        "kv_connector": "PegaPdConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector.pd_connector",
        "kv_connector_extra_config": {
            "connectors": [
                {"kv_connector": "PegaKVConnector",
                 "kv_connector_module_path": "pegaflow.connector",
                 "kv_role": "kv_both"},
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
            ]
        }
    }

Routing: Use with --no-router-kv-events on the Dynamo frontend for
approximate KV-aware routing. The router predicts cache state from its
own routing decisions, so no KV event publishing is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlHandshakePayload,
)
from vllm.v1.core.sched.output import SchedulerOutput

from pegaflow.connector import PegaKVConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class PegaPdConnectorMetadata(MultiKVConnectorMetadata):
    pass


class PegaPdConnector(MultiConnector):
    """
    A MultiConnector wrapper for PD disaggregated serving with PegaFlow.

    Combines PegaKVConnector (L2 cache via PegaFlow server sidecar) with
    NixlConnector (RDMA KV transfer between prefill and decode workers).

    Delegation logic:
    - Prefix matching: PegaKVConnector only (knows the L2 cache state)
    - Block allocation: PegaKVConnector gets real blocks, NixlConnector empty
    - Handshake: NixlConnector only (PegaFlow has no P/D handshake)
    - Save/load: Both connectors participate
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        if len(self._connectors) != 2:
            raise ValueError(
                "PegaPdConnector requires exactly two connectors "
                f"(got {len(self._connectors)})"
            )

        if not isinstance(self._connectors[0], PegaKVConnector):
            raise TypeError(
                "Expected first connector to be PegaKVConnector, "
                f"got {type(self._connectors[0]).__name__}"
            )
        if not isinstance(self._connectors[1], NixlConnector):
            raise TypeError(
                "Expected second connector to be NixlConnector, "
                f"got {type(self._connectors[1]).__name__}"
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Propagate handshake metadata to child connectors.

        PegaKVConnector ignores this (no-op), NixlConnector needs it
        to start its handshake listener for P/D coordination.
        """
        for c in self._connectors:
            c.set_xfer_handshake_metadata(metadata)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Get handshake metadata from NixlConnector only.

        PegaKVConnector returns None (no P/D handshake). Only NIXL provides
        handshake metadata so decode workers can connect for KV transfer.
        """
        nixl_connector = self._connectors[1]
        metadata = nixl_connector.get_handshake_metadata()
        if metadata is not None and not isinstance(metadata, NixlHandshakePayload):
            raise TypeError(
                "Expected NixlHandshakePayload from NIXL connector, "
                f"got {type(metadata).__name__}"
            )
        return metadata

    def bind_connector_metadata(
        self, connector_metadata: PegaPdConnectorMetadata
    ) -> None:
        """Bind per-step metadata to both child connectors.

        Skips MultiConnector.bind_connector_metadata() to avoid double-binding
        children. Calls KVConnectorBase_V1 directly for has_connector_metadata().
        """
        assert isinstance(connector_metadata, PegaPdConnectorMetadata)
        KVConnectorBase_V1.bind_connector_metadata(self, connector_metadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata, strict=True):
            c.bind_connector_metadata(cm)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get prefix match from PegaKVConnector only.

        Only PegaFlow knows the L2 cache state for prefix matching.
        """
        return self._connectors[0].get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        """PegaKVConnector gets real blocks, NixlConnector gets empty blocks."""
        empty_blocks = blocks.new_empty()
        self._connectors[0].update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        self._connectors[1].update_state_after_alloc(request, empty_blocks, 0)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PegaPdConnectorMetadata:
        """Build combined metadata from both connectors."""
        metadata = PegaPdConnectorMetadata(
            metadata=tuple(
                c.build_connector_meta(scheduler_output) for c in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata
