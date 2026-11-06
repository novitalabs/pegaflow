"""PegaFlow vLLM KV connector, v2 (semantics-driven).

Load with ``kv_connector_module_path="pegaflow.connector_v2"`` and
``kv_connector="PegaKVConnectorV2"``. See README.md for the layering.

The class is exported lazily so ``core`` / ``policy`` / ``adapters`` import
without vLLM or torch (the default unit gate relies on this).
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "PegaKVConnectorV2":
        from .connector import PegaKVConnectorV2

        return PegaKVConnectorV2
    raise AttributeError(name)


__all__ = ["PegaKVConnectorV2"]
