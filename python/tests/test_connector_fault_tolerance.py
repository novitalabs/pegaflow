"""Unit tests for load-path fault tolerance in the vLLM KV connector.

Mirrors NIXL's approach (vllm/tests/v1/kv_connector/unit/test_nixl_connector.py):
mock the transport, drive the connector's public API directly, assert that
failed blocks / reqs flow through `get_block_ids_with_load_errors` and
`get_finished` so vLLM can re-compute without dirty data or permanent leaks.

Covers:
- B.1 Load RPC returns ok=False → failure reported, no raise, no PyLoadState
  registered.
- B.1 Load RPC raises → same path.
- B.2 Load RPC ok=True but PyLoadState never ready → wall-clock timeout kicks
  in during get_finished, blocks/req reported as failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import (  # noqa: E402
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
)
from pegaflow.connector.worker import WorkerConnector  # noqa: E402


class FakeEngineClient:
    """Minimal stand-in for EngineRpcClient covering the load surface.

    Only implements what WorkerConnector touches in the load path. Save path is
    not exercised here since these tests are focused on load fault tolerance.
    """

    def __init__(self) -> None:
        self.fail_load_with_ok_false = False
        self.fail_load_with_exception: Exception | None = None
        self.load_calls: list[tuple] = []
        self.release_calls: list[bytes] = []

    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names,
        loads,
    ) -> tuple[bool, str]:
        block_ids = [block_id for _, ids in loads for block_id in ids]
        self.load_calls.append(
            (
                instance_id,
                tp_rank,
                device_id,
                load_state_shm,
                list(layer_names),
                list(block_ids),
            )
        )
        if self.fail_load_with_exception is not None:
            raise self.fail_load_with_exception
        if self.fail_load_with_ok_false:
            return (False, "simulated load failure")
        return (True, "ok")

    def health(self) -> tuple[bool, str]:
        return (True, "ok")

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        return (True, "ok")

    def release(self, lease: bytes) -> None:
        self.release_calls.append(lease)


def _make_worker() -> tuple[WorkerConnector, FakeEngineClient, MagicMock]:
    client = FakeEngineClient()
    state_manager = MagicMock()
    state_manager.is_available.return_value = True
    ctx = ConnectorContext(
        instance_id="test_instance",
        namespace="ns",
        block_size=16,
        num_layers=1,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=client,
        state_manager=state_manager,
    )
    worker = WorkerConnector(ctx)
    # cross-layer mode skips forward_context layer enumeration so we can drive
    # start_load_kv with a stub forward_context.
    worker._cross_layer_mode = True
    worker._cross_layer_key = "ALL_LAYERS"
    return worker, client, state_manager


def _stub_forward_context() -> MagicMock:
    ctx = MagicMock()
    ctx.no_compile_layers = {}
    return ctx


def _load_metadata(req_id: str, block_ids: tuple[int, ...]) -> PegaConnectorMetadata:
    return PegaConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids=block_ids,
                lease=f"lease-{req_id}".encode(),
                num_tokens=len(block_ids) * 16,
            )
        }
    )


@pytest.mark.parametrize(
    ("failure_mode", "req_id", "block_ids"),
    [
        ("ok_false", "req_fail_ok", (1, 2, 3)),
        ("exception", "req_fail_exc", (10, 20)),
    ],
)
def test_load_rpc_failure_reports_failures_without_raise(
    failure_mode: str,
    req_id: str,
    block_ids: tuple[int, ...],
):
    """B.1: failed load RPCs surface through vLLM recovery APIs instead of raising."""
    worker, client, state_mgr = _make_worker()
    if failure_mode == "ok_false":
        client.fail_load_with_ok_false = True
    elif failure_mode == "exception":
        client.fail_load_with_exception = ConnectionError("server gone")

    metadata = _load_metadata(req_id, block_ids)

    # Must not raise; used to crash the worker step instead of letting vLLM recompute.
    worker.start_load_kv(metadata, _stub_forward_context())

    assert len(client.load_calls) == 1
    assert worker.get_block_ids_with_load_errors() == set(block_ids)
    assert worker.get_block_ids_with_load_errors() == set()

    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {req_id}

    assert state_mgr.mark_unavailable.called
    assert client.release_calls == [f"lease-{req_id}".encode()]

    assert worker._pending_loads == {}
    assert worker._pending_load_reqs == {}
    assert worker._pending_load_meta == {}

    worker.shutdown()


def test_in_flight_load_timeout_respects_configured_boundary(monkeypatch):
    """B.2 boundary: elapsed < LOAD_TIMEOUT_SECONDS stays pending, > trips timeout.

    Mocks time.perf_counter so we can drive the wall-clock deterministically
    and verify the actual arithmetic — operand order and strict-greater-than
    behavior. Using LOAD_TIMEOUT_SECONDS=0 would exercise the same code path
    but would pass under a `>=` or swapped-operand regression.
    """
    worker, _client, state_mgr = _make_worker()
    timeout = worker.LOAD_TIMEOUT_SECONDS

    t0 = 10_000.0
    clock = {"now": t0}

    def fake_clock() -> float:
        return clock["now"]

    monkeypatch.setattr("pegaflow.connector.worker.time.perf_counter", fake_clock)

    metadata = _load_metadata("req_boundary", (5, 6, 7, 8))
    worker.start_load_kv(metadata, _stub_forward_context())
    assert "req_boundary" in worker._pending_loads

    # Just before the deadline: must NOT time out.
    clock["now"] = t0 + (timeout - 1)
    _, finished_recving = worker.get_finished(set())
    assert finished_recving is None, "load flagged as timed out before the deadline"
    assert "req_boundary" in worker._pending_loads
    assert worker.get_block_ids_with_load_errors() == set()
    assert not state_mgr.mark_unavailable.called

    # Just after the deadline: must time out.
    clock["now"] = t0 + (timeout + 1)
    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {"req_boundary"}
    assert worker.get_block_ids_with_load_errors() == {5, 6, 7, 8}
    assert state_mgr.mark_unavailable.called

    # In-flight state cleaned up — no permanent leak.
    assert worker._pending_loads == {}
    assert worker._pending_load_reqs == {}
    assert worker._pending_load_meta == {}

    worker.shutdown()


def test_get_block_ids_with_load_errors_drains_between_calls():
    """Repeated failures accumulate, but each call drains the set."""
    worker, client, _ = _make_worker()
    client.fail_load_with_ok_false = True

    worker.start_load_kv(_load_metadata("r1", (1,)), _stub_forward_context())
    worker.start_load_kv(_load_metadata("r2", (2, 3)), _stub_forward_context())

    assert worker.get_block_ids_with_load_errors() == {1, 2, 3}
    assert worker.get_block_ids_with_load_errors() == set()

    worker.shutdown()


def test_load_uses_registered_layer_names_before_forward_context_names():
    """Load must use the same layer names registered with the server."""
    worker, client, _ = _make_worker()
    worker._cross_layer_mode = False
    worker._registered_layers = ["registered.layer.0", "registered.layer.1"]

    forward_context = MagicMock()
    forward_layer = MagicMock()
    forward_layer.kv_cache = object()
    forward_context.no_compile_layers = {"model.layers.0.attn": forward_layer}

    worker.start_load_kv(_load_metadata("req_registered_layers", (1, 2)), forward_context)

    assert len(client.load_calls) == 1
    assert client.load_calls[0][4] == ["registered.layer.0", "registered.layer.1"]

    worker.shutdown()
