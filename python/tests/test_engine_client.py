"""Integration tests for EngineRpcClient against PegaServer.

These tests verify the gRPC client can correctly communicate with
a running PegaServer instance. The server is automatically started
by the `pega_server` fixture.

Requirements:
- Rust extension built: maturin develop --release
- GPU available for PegaServer

Run with:
    cd python && pytest tests/ -v
"""


class TestEngineClientFixtures:
    """Test basic fixtures and server connectivity."""

    def test_client_connects(self, engine_client):
        """Verify client can connect to server."""
        assert engine_client is not None

    def test_server_is_running(self, pega_server):
        """Verify the test server is running."""
        assert pega_server.is_running()
        assert pega_server.endpoint.startswith("http://")


class TestEngineClientQuery:
    """Test query operations with various inputs."""

    def test_query_empty_hashes(self, engine_client, registered_instance: str):
        """Query with empty hashes should succeed."""
        result = engine_client.query(registered_instance, [], "")

        if isinstance(result, dict):
            assert "hit_blocks" in result or "ok" in result
        else:
            assert len(result) == 3

    def test_query_unknown_hashes(
        self, engine_client, registered_instance: str, block_hashes: list[bytes]
    ):
        """Query for unknown hashes should return zero hits (miss)."""
        result = engine_client.query(registered_instance, block_hashes[:5], "")

        if isinstance(result, dict):
            hit_blocks = result.get("hit_blocks", 0)
        else:
            _, _, hit_blocks = result

        assert hit_blocks == 0, "Unknown hashes should have zero hits"

    def test_query_single_hash(
        self, engine_client, registered_instance: str, block_hashes: list[bytes]
    ):
        """Query with a single hash."""
        result = engine_client.query(registered_instance, block_hashes[:1], "")
        assert result is not None
