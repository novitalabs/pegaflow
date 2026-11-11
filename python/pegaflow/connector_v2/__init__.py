"""vLLM KV connector on RPC V2 objects.

Use ``kv_connector="PegaKVConnectorV2"`` with
``kv_connector_module_path="pegaflow.connector_v2"``. The connector class is
imported lazily so ``semantics`` / ``layout`` / ``scheduler`` / ``worker``
import without vLLM or torch.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("PegaKVConnectorV2", "PegaConnectorMetadata", "PegaWorkerMetadata"):
        from . import connector

        return getattr(connector, name)
    raise AttributeError(name)
