"""Compatibility helpers for mixed-version PegaFlow deployments."""

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger

from pegaflow import __version__ as PEGAFLOW_VERSION
from pegaflow.pegaflow import PegaFlowBusinessError

_WARNED_QUERY_PREFETCH_ENDPOINTS: set[str] = set()


def _is_query_prefetch_unimplemented(error: PegaFlowBusinessError) -> bool:
    message = str(error).lower()
    return (
        "unimplemented" in message
        or "not implemented" in message
        or "not supported" in message
    )


def query_prefetch_with_fallback(
    engine_client,
    instance_id: str,
    block_hashes: Sequence[bytes],
    logger: Logger,
):
    """Call ``QueryPrefetch`` and fall back to legacy ``Query`` when unavailable."""
    try:
        return engine_client.query_prefetch(instance_id, list(block_hashes))
    except PegaFlowBusinessError as error:
        if not _is_query_prefetch_unimplemented(error):
            raise

        endpoint = getattr(engine_client, "endpoint", "unknown")
        if endpoint not in _WARNED_QUERY_PREFETCH_ENDPOINTS:
            logger.warning(
                "[PegaFlowCompat] remote pegaflow-server at %s does not support "
                "QueryPrefetch; falling back to legacy Query without SSD prefetch "
                "(connector_version=%s). Upgrade the server to restore "
                "QueryPrefetch/SSD-prefetch support. Original error: %s",
                endpoint,
                PEGAFLOW_VERSION,
                error,
            )
            _WARNED_QUERY_PREFETCH_ENDPOINTS.add(endpoint)

        return engine_client.query(instance_id, list(block_hashes))


__all__ = ["query_prefetch_with_fallback"]
