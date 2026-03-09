from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_test_stubs() -> None:
    repo_python = Path(__file__).resolve().parents[1]
    package_dir = repo_python / "pegaflow"

    pegaflow_pkg = types.ModuleType("pegaflow")
    pegaflow_pkg.__path__ = [str(package_dir)]
    pegaflow_pkg.__version__ = "0.0.16-test"
    sys.modules["pegaflow"] = pegaflow_pkg

    connector_pkg = types.ModuleType("pegaflow.connector")
    connector_pkg.__path__ = [str(package_dir / "connector")]
    sys.modules["pegaflow.connector"] = connector_pkg

    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    sys.modules["vllm"] = vllm_pkg

    distributed_pkg = types.ModuleType("vllm.distributed")
    distributed_pkg.__path__ = []
    sys.modules["vllm.distributed"] = distributed_pkg

    kv_transfer_pkg = types.ModuleType("vllm.distributed.kv_transfer")
    kv_transfer_pkg.__path__ = []
    sys.modules["vllm.distributed.kv_transfer"] = kv_transfer_pkg

    kv_connector_pkg = types.ModuleType("vllm.distributed.kv_transfer.kv_connector")
    kv_connector_pkg.__path__ = []
    sys.modules["vllm.distributed.kv_transfer.kv_connector"] = kv_connector_pkg

    v1_pkg = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    v1_pkg.__path__ = []
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1"] = v1_pkg

    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorMetadata:
        pass

    base_mod.KVConnectorMetadata = KVConnectorMetadata
    sys.modules[base_mod.__name__] = base_mod

    metrics_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

    class KVConnectorPromMetrics:
        pass

    class KVConnectorStats:
        def __init__(self, data=None):
            self.data = data or {}

    class PromMetric:
        pass

    metrics_mod.KVConnectorPromMetrics = KVConnectorPromMetrics
    metrics_mod.KVConnectorStats = KVConnectorStats
    metrics_mod.PromMetric = PromMetric
    metrics_mod.PromMetricT = object
    sys.modules[metrics_mod.__name__] = metrics_mod

    pegaflow_ext = types.ModuleType("pegaflow.pegaflow")

    class PegaFlowError(Exception):
        pass

    class PegaFlowBusinessError(PegaFlowError):
        pass

    class PegaFlowServiceError(PegaFlowError):
        pass

    pegaflow_ext.PegaFlowError = PegaFlowError
    pegaflow_ext.PegaFlowBusinessError = PegaFlowBusinessError
    pegaflow_ext.PegaFlowServiceError = PegaFlowServiceError
    pegaflow_ext.EngineRpcClient = object
    sys.modules[pegaflow_ext.__name__] = pegaflow_ext


@pytest.fixture()
def scheduler_module():
    _install_test_stubs()
    repo_python = Path(__file__).resolve().parents[1]

    for module_name in [
        "pegaflow.compat",
        "pegaflow.connector.common",
        "pegaflow.connector.connector_metrics",
        "pegaflow.connector.scheduler",
    ]:
        sys.modules.pop(module_name, None)

    compat_spec = importlib.util.spec_from_file_location(
        "pegaflow.compat",
        repo_python / "pegaflow" / "compat.py",
    )
    compat_module = importlib.util.module_from_spec(compat_spec)
    sys.modules["pegaflow.compat"] = compat_module
    assert compat_spec and compat_spec.loader
    compat_spec.loader.exec_module(compat_module)

    scheduler_spec = importlib.util.spec_from_file_location(
        "pegaflow.connector.scheduler",
        repo_python / "pegaflow" / "connector" / "scheduler.py",
    )
    module = importlib.util.module_from_spec(scheduler_spec)
    sys.modules["pegaflow.connector.scheduler"] = module
    assert scheduler_spec and scheduler_spec.loader
    scheduler_spec.loader.exec_module(module)
    yield module

    for module_name in [
        "pegaflow.compat",
        "pegaflow.connector",
        "pegaflow.connector.common",
        "pegaflow.connector.connector_metrics",
        "pegaflow.connector.scheduler",
        "pegaflow.pegaflow",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer",
        "vllm.distributed",
        "vllm",
        "pegaflow",
    ]:
        sys.modules.pop(module_name, None)


def test_scheduler_falls_back_to_legacy_query(scheduler_module):
    class FakeStateManager:
        def is_available(self) -> bool:
            return True

        def mark_unavailable(self, reason: str) -> None:
            raise AssertionError(f"unexpected mark_unavailable: {reason}")

    class FakeEngineClient:
        endpoint = "http://legacy-server:50055"

        def __init__(self):
            self.query_prefetch_calls = 0
            self.query_calls = 0

        def query_prefetch(self, instance_id: str, block_hashes: list[bytes]):
            self.query_prefetch_calls += 1
            raise scheduler_module.PegaFlowBusinessError(
                "rpc RPC failed: code: 'Operation is not implemented or not supported'"
            )

        def query(self, instance_id: str, block_hashes: list[bytes]):
            self.query_calls += 1
            return {
                "ok": True,
                "message": "legacy query fallback",
                "hit_blocks": 2,
                "prefetch_state": "done",
                "loading_blocks": 0,
                "missing_blocks": 1,
            }

    context = SimpleNamespace(
        engine_client=FakeEngineClient(),
        state_manager=FakeStateManager(),
        instance_id="instance-1",
    )
    connector = scheduler_module.SchedulerConnector(context)

    hit_blocks = connector._count_available_block_prefix([b"a", b"b", b"c"], "req-1")

    assert hit_blocks == 2
    assert context.engine_client.query_prefetch_calls == 1
    assert context.engine_client.query_calls == 1


def test_scheduler_preserves_non_compat_business_errors(scheduler_module):
    class FakeStateManager:
        def is_available(self) -> bool:
            return True

        def mark_unavailable(self, reason: str) -> None:
            raise AssertionError(f"unexpected mark_unavailable: {reason}")

    class FakeEngineClient:
        endpoint = "http://new-server:50055"

        def query_prefetch(self, instance_id: str, block_hashes: list[bytes]):
            raise scheduler_module.PegaFlowBusinessError("invalid block hash payload")

        def query(self, instance_id: str, block_hashes: list[bytes]):
            raise AssertionError("legacy query should not be used")

    context = SimpleNamespace(
        engine_client=FakeEngineClient(),
        state_manager=FakeStateManager(),
        instance_id="instance-1",
    )
    connector = scheduler_module.SchedulerConnector(context)

    with pytest.raises(scheduler_module.PegaFlowBusinessError):
        connector._count_available_block_prefix([b"a"], "req-2")
