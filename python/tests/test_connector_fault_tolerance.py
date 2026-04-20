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

from pegaflow.connector.common import (
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
)
from pegaflow.connector.worker import WorkerConnector


class FakeEngineClient:
    """Minimal stand-in for EngineRpcClient covering the load surface.

    Only implements what WorkerConnector touches in the load path. Save path is
    not exercised here since these tests are focused on load fault tolerance.
    """

    def __init__(self) -> None:
        self.fail_load_with_ok_false = False
        self.fail_load_with_exception: Exception | None = None
        self.load_calls: list[tuple] = []

    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names,
        block_ids,
        block_hashes,
    ) -> tuple[bool, str]:
        self.load_calls.append((instance_id, tp_rank, device_id, load_state_shm, list(block_ids)))
        if self.fail_load_with_exception is not None:
            raise self.fail_load_with_exception
        if self.fail_load_with_ok_false:
            return (False, "simulated load failure")
        return (True, "ok")

    def health(self) -> tuple[bool, str]:
        return (True, "ok")


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
                block_hashes=tuple(f"h{b}".encode() for b in block_ids),
                num_tokens=len(block_ids) * 16,
            )
        }
    )


def test_load_rpc_ok_false_reports_failures_without_raise():
    """B.1: engine_client.load returns ok=False -> no raise, failure surface populated."""
    worker, client, state_mgr = _make_worker()
    client.fail_load_with_ok_false = True

    metadata = _load_metadata("req_fail_ok", (1, 2, 3))

    # Must not raise; used to raise RuntimeError and crash the worker step.
    worker.start_load_kv(metadata, _stub_forward_context())

    # RPC was actually attempted.
    assert len(client.load_calls) == 1

    # get_block_ids_with_load_errors returns the exact failed block ids, then drains.
    assert worker.get_block_ids_with_load_errors() == {1, 2, 3}
    assert worker.get_block_ids_with_load_errors() == set()

    # get_finished reports the failed request in finished_recving so vLLM unblocks.
    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {"req_fail_ok"}

    # State manager was asked to mark unavailable so subsequent queries short-circuit.
    assert state_mgr.mark_unavailable.called

    # No PyLoadState registered: no permanent leak.
    assert worker._pending_loads == {}
    assert worker._pending_load_reqs == {}
    assert worker._pending_load_meta == {}

    worker.shutdown()


def test_load_rpc_exception_reports_failures_without_raise():
    """B.1: engine_client.load raises -> same failure-reporting path as ok=False."""
    worker, client, state_mgr = _make_worker()
    client.fail_load_with_exception = ConnectionError("server gone")

    metadata = _load_metadata("req_fail_exc", (10, 20))

    worker.start_load_kv(metadata, _stub_forward_context())

    assert worker.get_block_ids_with_load_errors() == {10, 20}

    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {"req_fail_exc"}

    assert state_mgr.mark_unavailable.called

    assert worker._pending_loads == {}
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
