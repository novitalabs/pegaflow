"""
vLLM general plugin for PegaFlow.

Registered via the ``vllm.general_plugins`` entry-point so that
``load_general_plugins()`` executes it in **every** vLLM process
(API-server, engine-core, workers).

Registers ``PegaKVConnector`` and ``PegaPdConnector`` in the
``KVConnectorFactory`` registry.  The registration is lazy —
modules are only imported when the class is actually looked up —
so this plugin adds negligible overhead.
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
            "PegaPdConnector",
            "pegaflow.connector.pd_connector",
            "PegaPdConnector",
        )
