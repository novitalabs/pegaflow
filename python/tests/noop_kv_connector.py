"""No-op vLLM KV connector used as the E2E correctness baseline."""

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)


class NoOpKVConnector(KVConnectorBase_V1):
    """Enable vLLM's KV-transfer path without external KV save/load."""

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        return

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        return KVConnectorMetadata()
