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

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: E402

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
        self.register_response: tuple[bool, str] = (True, "ok")
        self.register_exception: Exception | None = None
        self.register_calls: list[tuple] = []
        self.register_kwargs: list[dict] = []
        self.unregister_calls: list[str] = []
        self.release_calls: list[bytes] = []

    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_groups,
        loads,
    ) -> tuple[bool, str]:
        block_ids = [block_id for _, groups in loads for ids in groups for block_id in ids]
        self.load_calls.append(
            (
                instance_id,
                tp_rank,
                device_id,
                load_state_shm,
                [list(group) for group in layer_groups],
                list(block_ids),
            )
        )
        if self.fail_load_with_exception is not None:
            raise self.fail_load_with_exception
        if self.fail_load_with_ok_false:
            return (False, "simulated load failure")
        return (True, "ok")

    def register_context_batch(self, *args, **kwargs) -> tuple[bool, str]:
        self.register_calls.append(args)
        self.register_kwargs.append(kwargs)
        if self.register_exception is not None:
            raise self.register_exception
        return self.register_response

    def health(self) -> tuple[bool, str]:
        return (True, "ok")

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        self.unregister_calls.append(instance_id)
        return (True, "ok")

    def release(self, lease: bytes) -> None:
        self.release_calls.append(lease)


def _make_worker(
    pp_rank: int = 0,
    pp_size: int = 1,
    kv_cache_config=None,
    vllm_config=None,
    **ctx_kwargs,
) -> tuple[WorkerConnector, FakeEngineClient, MagicMock]:
    client = FakeEngineClient()
    state_manager = MagicMock()
    state_manager.is_available.return_value = True
    defaults = {
        "instance_id": "test_instance",
        "namespace": "ns",
        "block_size": 16,
        "tp_size": 1,
        "world_size": 1,
        "tp_rank": 0,
        "device_id": 0,
        "engine_client": client,
        "state_manager": state_manager,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
    }
    defaults.update(ctx_kwargs)
    ctx = ConnectorContext(**defaults)
    worker = WorkerConnector(
        ctx,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )
    # cross-layer mode skips forward_context layer enumeration so we can drive
    # start_load_kv with a stub forward_context.
    worker._cross_layer_mode = True
    worker._cross_layer_key = "ALL_LAYERS"
    return worker, client, state_manager


def _stub_forward_context() -> MagicMock:
    ctx = MagicMock()
    ctx.no_compile_layers = {}
    return ctx


def _single_attention_cache_group(*layer_names: str) -> MagicMock:
    spec = FullAttentionSpec()
    spec.block_size = 16
    return MagicMock(layer_names=layer_names, kv_cache_spec=spec)


def _load_metadata(req_id: str, block_ids: tuple[int, ...]) -> PegaConnectorMetadata:
    return PegaConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=(block_ids,),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=len(block_ids) * 16,
            )
        }
    )


def _configure_hma_worker(worker: WorkerConnector) -> None:
    worker._cross_layer_mode = False
    worker._cache_groups = MagicMock(group_count=2, has_recurrent_state=True)
    worker._registered_layers = ["attention", "recurrent"]
    worker._layer_to_group = {"attention": 0, "recurrent": 1}


def _hma_load_metadata(req_id: str) -> PegaConnectorMetadata:
    return PegaConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=((11,), (21,)),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=16,
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


@pytest.mark.parametrize("failure_mode", ["ok_false", "exception"])
def test_hma_load_rpc_failure_crashes_before_vllm_partial_recovery(failure_mode: str):
    worker, client, state_mgr = _make_worker()
    _configure_hma_worker(worker)
    if failure_mode == "ok_false":
        client.fail_load_with_ok_false = True
    else:
        client.fail_load_with_exception = ConnectionError("server gone")

    with pytest.raises(RuntimeError, match="cannot recover failed loads"):
        worker.start_load_kv(_hma_load_metadata("hma-failure"), _stub_forward_context())

    assert state_mgr.mark_unavailable.called
    assert client.release_calls == [b"lease-hma-failure"]
    assert worker._pending_loads == {}
    worker.shutdown()


def test_hma_load_distinguishes_block_zero_from_absent_recurrent_target():
    worker, client, _state_mgr = _make_worker()
    _configure_hma_worker(worker)
    metadata = PegaConnectorMetadata(
        load_intents={
            "hma-sparse": LoadIntent(
                block_ids_by_group=((0, 12), (None, 21)),
                leases=(b"lease-hma-sparse",),
                num_tokens=32,
            )
        }
    )

    worker.start_load_kv(metadata, _stub_forward_context())

    assert client.load_calls[0][5] == [0, 12, None, 21]
    worker.shutdown()


def test_hma_load_timeout_crashes_before_vllm_partial_recovery(monkeypatch):
    worker, _client, state_mgr = _make_worker()
    _configure_hma_worker(worker)
    clock = {"now": 10_000.0}
    monkeypatch.setattr("pegaflow.connector.worker.time.perf_counter", lambda: clock["now"])
    worker.start_load_kv(_hma_load_metadata("hma-timeout"), _stub_forward_context())
    clock["now"] += worker.LOAD_TIMEOUT_SECONDS + 1

    with pytest.raises(RuntimeError, match="cannot recover failed loads"):
        worker.get_finished(set())

    assert state_mgr.mark_unavailable.called
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
    assert client.load_calls[0][4] == [["registered.layer.0", "registered.layer.1"]]

    worker.shutdown()


class FakeTensor:
    shape = (1, 16)

    def storage_offset(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cuda:0"

    def stride(self) -> tuple[int, int]:
        return (16, 1)

    def element_size(self) -> int:
        return 2


class FakeCudaIPCWrapper:
    def __init__(self, _tensor) -> None:
        pass


def test_register_version_mismatch_raises_startup_error(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_response = (
        False,
        "PegaFlow version mismatch: client=0.22.4 server=0.22.5",
    )

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="PegaFlow version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert "for layer.0" not in str(exc_info.value)
    assert len(client.register_calls) == 1
    assert client.unregister_calls == []

    worker.shutdown()
    assert client.unregister_calls == []


def test_register_non_version_failure_reports_batch_layers(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_response = (False, "invalid tensor metadata")

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="invalid tensor metadata") as exc_info:
        worker.register_kv_caches(
            {
                "layer.0": FakeTensor(),
                "layer.1": FakeTensor(),
            }
        )

    message = str(exc_info.value)
    assert "Register context batch failed for layers ['layer.0', 'layer.1']" in message
    assert "for layer.1" not in message
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1"]

    worker.shutdown()


def test_register_kv_caches_ignores_shared_by_without_layer_split_opt_in(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
    ]
    worker, client, _ = _make_worker(
        kv_cache_config=kv_cache_config,
    )

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": FakeTensor(),
            "layer.1": FakeTensor(),
            "layer.2": FakeTensor(),
        }
    )

    assert worker._registered_layers == ["layer.0", "layer.1", "layer.2"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1", "layer.2"]

    worker.shutdown()


def test_register_kv_caches_uses_layer_split_shared_by_plan(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
        MagicMock(shared_by=()),
        MagicMock(shared_by=("layer.0",)),
    ]
    worker, client, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": FakeTensor(),
            "layer.1": FakeTensor(),
            "layer.2": FakeTensor(),
        }
    )

    assert worker._registered_layers == ["layer.1", "layer.0"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.1", "layer.0"]

    worker.shutdown()


def test_register_kv_caches_requires_shared_by_layers(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [_single_attention_cache_group("layer.0", "layer.1")]
    kv_cache_config.kv_cache_tensors = [MagicMock(shared_by=("layer.1",))]
    worker, _, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="missing layers"):
        worker.register_kv_caches({"layer.0": FakeTensor()})

    worker.shutdown()


def test_cross_layer_registration_uses_pp_suffixed_name(monkeypatch):
    worker, client, _ = _make_worker(pp_rank=1, pp_size=4)

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_cross_layers_kv_cache(FakeTensor(), attn_backend=object())

    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["ALL_LAYERS_pp1"]

    worker.shutdown()


def test_register_version_mismatch_rpc_error_stops_startup(monkeypatch):
    worker, client, _ = _make_worker()
    client.register_exception = RuntimeError(
        "register_context_batch RPC failed: status: FailedPrecondition, "
        'message: "PegaFlow version mismatch: client=0.22.4 server=0.22.5"'
    )

    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="PegaFlow version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "FailedPrecondition" in str(exc_info.value)
    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert len(client.register_calls) == 1

    worker.shutdown()


def test_local_recurrent_pool_oom_is_explicit_and_cleans_partial_state(monkeypatch):
    recurrent = MambaSpec()
    recurrent.block_size = 16
    recurrent.mamba_cache_mode = "align"
    recurrent.page_size_bytes = 32
    recurrent_2 = MambaSpec()
    recurrent_2.block_size = 16
    recurrent_2.mamba_cache_mode = "align"
    recurrent_2.page_size_bytes = 32
    attention = FullAttentionSpec()
    attention.block_size = 16
    kv_cache_config = MagicMock(
        kv_cache_groups=(
            MagicMock(layer_names=("recurrent",), kv_cache_spec=recurrent),
            MagicMock(layer_names=("recurrent_2",), kv_cache_spec=recurrent_2),
            MagicMock(layer_names=("attention",), kv_cache_spec=attention),
        )
    )
    worker, client, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        linear_state_cache_size_bytes=64,
    )
    monkeypatch.setattr(
        "pegaflow.connector.worker._raw_page_view",
        lambda _tensor, _page_bytes: object(),
    )
    monkeypatch.setattr(
        "pegaflow.connector.worker.torch.empty",
        MagicMock(side_effect=[object(), __import__("torch").OutOfMemoryError("oom")]),
        raising=False,
    )
    empty_cache = MagicMock()
    monkeypatch.setattr(
        "pegaflow.connector.worker.torch.cuda.empty_cache",
        empty_cache,
    )

    with pytest.raises(RuntimeError, match="outside vLLM KV-cache planning"):
        worker.register_kv_caches(
            {
                "recurrent": FakeTensor(),
                "recurrent_2": FakeTensor(),
                "attention": FakeTensor(),
            }
        )

    assert worker._local_recurrent_layers == {}
    assert client.register_calls == []
    empty_cache.assert_called_once()
    worker.shutdown()
