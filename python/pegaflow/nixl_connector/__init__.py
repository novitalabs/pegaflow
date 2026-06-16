# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL KV-cache transfer connector (disaggregated prefill / decode)."""

from typing import Any

__all__ = [
    "NixlAgentMetadata",
    "NixlBaseConnector",
    "NixlBaseConnectorScheduler",
    "NixlBaseConnectorWorker",
    "NixlConnector",
    "NixlConnectorMetadata",
    "NixlConnectorScheduler",
    "NixlConnectorWorker",
    "NixlHandshakePayload",
    "NixlKVConnectorStats",
    "PegaNixlConnector",
    "NixlPullConnector",
    "NixlPullConnectorScheduler",
    "NixlPullConnectorWorker",
    "NixlPushConnector",
    "NixlPushConnectorScheduler",
    "NixlPushConnectorWorker",
]

_EXPORT_MODULES = {
    "NixlAgentMetadata": "pegaflow.nixl_connector.metadata",
    "NixlBaseConnector": "pegaflow.nixl_connector.connector",
    "NixlBaseConnectorScheduler": "pegaflow.nixl_connector.base_scheduler",
    "NixlBaseConnectorWorker": "pegaflow.nixl_connector.base_worker",
    "NixlConnector": "pegaflow.nixl_connector.connector",
    "NixlConnectorMetadata": "pegaflow.nixl_connector.metadata",
    "NixlConnectorScheduler": "pegaflow.nixl_connector.scheduler",
    "NixlConnectorWorker": "pegaflow.nixl_connector.worker",
    "NixlHandshakePayload": "pegaflow.nixl_connector.metadata",
    "NixlKVConnectorStats": "pegaflow.nixl_connector.stats",
    "PegaNixlConnector": "pegaflow.nixl_connector.connector",
    "NixlPullConnector": "pegaflow.nixl_connector.connector",
    "NixlPullConnectorScheduler": "pegaflow.nixl_connector.pull_scheduler",
    "NixlPullConnectorWorker": "pegaflow.nixl_connector.pull_worker",
    "NixlPushConnector": "pegaflow.nixl_connector.connector",
    "NixlPushConnectorScheduler": "pegaflow.nixl_connector.push_scheduler",
    "NixlPushConnectorWorker": "pegaflow.nixl_connector.push_worker",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
