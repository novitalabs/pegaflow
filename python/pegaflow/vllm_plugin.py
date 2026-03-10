"""
vLLM general plugin for PegaFlow.

Registered via the ``vllm.general_plugins`` entry-point so that
``load_general_plugins()`` executes it in **every** vLLM process
(API-server, engine-core, workers).

The sole purpose is to register ``PegaKVConnector`` in the
``KVConnectorFactory`` registry.  The registration is lazy —
``pegaflow.connector`` is only imported when the class is actually
looked up — so this module adds negligible overhead.
"""


def register() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    try:
        KVConnectorFactory.register_connector(
            "PegaKVConnector",
            "pegaflow.connector",
            "PegaKVConnector",
        )
    except ValueError:
        pass  # Already registered (e.g. module was imported before plugin ran)
