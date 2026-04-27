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

import importlib
import time


def _prepare_snapshot(engine_client, instance_id: str, block_hashes: list[bytes]) -> dict:
    pegaflow_module = importlib.import_module("pegaflow.pegaflow")
    state = pegaflow_module._NativePrepareLoadState()
    ok, message = engine_client.prepare_load(
        instance_id,
        "test",
        block_hashes,
        len(block_hashes) * 16,
        0,
        16,
        state.shm_name(),
        None,
        0,
    )
    assert ok, message
    for _ in range(100):
        snapshot = state.snapshot()
        if not snapshot.get("preparing", False):
            return snapshot
        time.sleep(0.01)
    return state.snapshot()


class TestEngineClientFixtures:
    """Test basic fixtures and server connectivity."""

    def test_client_connects(self, engine_client):
        """Verify client can connect to server."""
        assert engine_client is not None

    def test_server_is_running(self, pega_server):
        """Verify the test server is running."""
        assert pega_server.is_running()
        assert pega_server.endpoint.startswith("http://")


class TestEngineClientPrepareLoad:
    """Test prepare-load operations with various inputs."""

    def test_prepare_empty_hashes(self, engine_client, registered_instance: str):
        """Prepare with empty hashes should return no load plan."""
        result = _prepare_snapshot(engine_client, registered_instance, [])

        assert isinstance(result, dict)
        assert result.get("ready_no_plan") is True

    def test_prepare_unknown_hashes(
        self, engine_client, registered_instance: str, block_hashes: list[bytes]
    ):
        """Prepare for unknown hashes should return no load plan."""
        result = _prepare_snapshot(engine_client, registered_instance, block_hashes[:5])

        assert isinstance(result, dict)
        assert result.get("ready_no_plan") is True

    def test_prepare_single_hash(
        self, engine_client, registered_instance: str, block_hashes: list[bytes]
    ):
        """Prepare with a single hash."""
        result = _prepare_snapshot(engine_client, registered_instance, block_hashes[:1])
        assert result is not None
