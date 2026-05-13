"""Integration tests for EngineRpcClient against PegaServer.

These tests verify the gRPC client can correctly communicate with
a running PegaServer instance. The server is automatically started
by the `pega_server` fixture.

Requirements:
- Rust extension built: maturin develop --release
- GPU available for PegaServer

Run with:
    cd python && pytest -m integration tests/test_engine_client.py -v
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


class TestEngineClientQuery:
    """Test query operations with various inputs."""

    def test_query_empty_hashes(self, engine_client, registered_instance: str):
        """Query with empty hashes should succeed."""
        result = engine_client.query_prefetch(registered_instance, [], req_id="test")

        assert isinstance(result, dict)
        assert "hit_blocks" in result or "ok" in result

    def test_query_unknown_hashes(
        self, engine_client, registered_instance: str, block_hashes: list[bytes]
    ):
        """Query for unknown hashes should return zero hits (miss)."""
        result = engine_client.query_prefetch(registered_instance, block_hashes[:5], req_id="test")

        assert isinstance(result, dict)
        hit_blocks = result.get("hit_blocks", 0)
        assert hit_blocks == 0, "Unknown hashes should have zero hits"
