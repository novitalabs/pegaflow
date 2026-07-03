"""
vLLM general plugin for PegaFlow.

Registered via the ``vllm.general_plugins`` entry-point so that
``load_general_plugins()`` executes it in **every** vLLM process
(API-server, engine-core, workers).

The sole purpose is to register PegaFlow KV connectors in the
``KVConnectorFactory`` registry.  The registration is lazy —
connector modules are only imported when the class is actually
looked up — so this module adds negligible overhead.
"""

import contextlib


def register() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PegaKVConnector",
            "pegaflow.connector",
            "PegaKVConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NoopKVConnector",
            "pegaflow.connector",
            "NoopKVConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdConnector",
            "pegaflow.pd_connector",
            "PdConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdDecodeConnector",
            "pegaflow.pd_connector",
            "PdDecodeConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PdPrefillConnector",
            "pegaflow.pd_connector",
            "PdPrefillConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlConnector",
            "pegaflow.nixl_connector",
            "NixlConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlPullConnector",
            "pegaflow.nixl_connector",
            "NixlPullConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "NixlPushConnector",
            "pegaflow.nixl_connector",
            "NixlPushConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PegaNixlConnector",
            "pegaflow.nixl_connector",
            "PegaNixlConnector",
        )
    with contextlib.suppress(ValueError):
        KVConnectorFactory.register_connector(
            "PegaNixlPullConnector",
            "pegaflow.nixl_connector",
            "PegaNixlPullConnector",
        )
