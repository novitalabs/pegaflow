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


@pytest.mark.parametrize(
    ("case", "hash_count", "expected_missing"),
    [
        pytest.param("empty_query", 0, 0, id="empty_query"),
        pytest.param("unknown_hashes", 5, 5, id="unknown_hashes"),
    ],
)
def test_query_prefetch_miss_contract(
    case: str,
    hash_count: int,
    expected_missing: int,
    engine_client,
    registered_instance: str,
    block_hashes: list[bytes],
):
    """A fresh server query returns a concrete miss contract, not only connectivity."""
    requested_hashes = block_hashes[:hash_count]

    result = engine_client.query_prefetch(
        registered_instance,
        requested_hashes,
        req_id=f"query-contract-{case}",
    )

    assert result == {
        "ok": True,
        "message": "",
        "hit_blocks": 0,
        "prefetch_state": "done",
        "loading_blocks": 0,
        "missing_blocks": expected_missing,
    }
