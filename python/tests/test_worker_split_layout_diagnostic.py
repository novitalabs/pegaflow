from unittest.mock import MagicMock

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.common import ConnectorContext  # noqa: E402
from pegaflow.connector.worker import (  # noqa: E402
    WorkerConnector,
    _detect_split_block_layout,
    _infer_physical_block_size_tokens,
)


def _ctx(*, block_size: int = 128, is_mla: bool = True) -> ConnectorContext:
    return ConnectorContext(
        instance_id="test-instance",
        namespace="test-namespace",
        block_size=block_size,
        num_layers=1,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        engine_client=MagicMock(),
        state_manager=MagicMock(),
        is_mla=is_mla,
    )


def test_blocks_first_mla_split_layout_is_detected():
    assert _detect_split_block_layout(
        _ctx(block_size=128, is_mla=True),
        (16, 64, 656),
        "blocks-first",
    ) == (64, 2)


def test_kv_first_standard_split_layout_is_detected():
    assert _detect_split_block_layout(
        _ctx(block_size=128, is_mla=False),
        (2, 16, 64, 8, 128),
        "KV-first",
    ) == (64, 2)


def test_blocks_first_flashinfer_split_layout_is_detected():
    assert _detect_split_block_layout(
        _ctx(block_size=128, is_mla=False),
        (16, 2, 64, 8, 128),
        "blocks-first",
    ) == (64, 2)


def test_matching_physical_block_layout_is_supported():
    assert (
        _detect_split_block_layout(
            _ctx(block_size=128, is_mla=True),
            (16, 128, 656),
            "blocks-first",
        )
        is None
    )


def test_non_divisible_layout_is_not_reported_as_split():
    assert (
        _detect_split_block_layout(
            _ctx(block_size=128, is_mla=True),
            (16, 96, 656),
            "blocks-first",
        )
        is None
    )


def test_compressed_mla_storage_shape_is_not_reported_as_split():
    assert (
        _detect_split_block_layout(
            _ctx(block_size=256, is_mla=True),
            (16, 2, 584),
            "blocks-first",
        )
        is None
    )


def test_physical_block_size_inference_covers_common_vllm_layouts():
    assert _infer_physical_block_size_tokens((2, 16, 64, 8, 128), "KV-first") == 64
    assert _infer_physical_block_size_tokens((16, 2, 64, 8, 128), "blocks-first") == 64
    assert _infer_physical_block_size_tokens((16, 64, 656), "blocks-first") == 64
    assert _infer_physical_block_size_tokens((16, 2, 584), "blocks-first") is None


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.device = "cuda:0"

    def storage_offset(self) -> int:
        return 0

    def stride(self) -> tuple[int, ...]:
        if len(self.shape) == 3:
            return (self.shape[1] * self.shape[2], self.shape[2], 1)
        if len(self.shape) == 5 and self.shape[0] == 2:
            return (
                self.shape[1] * self.shape[2] * self.shape[3] * self.shape[4],
                self.shape[2] * self.shape[3] * self.shape[4],
                self.shape[3] * self.shape[4],
                self.shape[4],
                1,
            )
        if len(self.shape) == 5:
            return (
                self.shape[1] * self.shape[2] * self.shape[3] * self.shape[4],
                self.shape[2] * self.shape[3] * self.shape[4],
                self.shape[3] * self.shape[4],
                self.shape[4],
                1,
            )
        return tuple(reversed(range(1, len(self.shape) + 1)))

    def element_size(self) -> int:
        return 1


def _connector_for_registration(ctx: ConnectorContext) -> WorkerConnector:
    connector = WorkerConnector.__new__(WorkerConnector)
    connector._ctx = ctx
    connector._registered_layers = []
    connector._torch_device = None
    connector._layer_name_to_id = {}
    connector._cross_layer_mode = False
    connector._reported_split_block_layout = False
    return connector


def test_split_layout_registration_logs_once_and_continues(monkeypatch):
    ctx = _ctx(block_size=128, is_mla=False)
    connector = _connector_for_registration(ctx)
    ctx.engine_client.register_context_batch.return_value = (True, "")

    errors: list[str] = []
    monkeypatch.setattr(
        "pegaflow.connector.worker.logger.error", lambda msg, *args: errors.append(msg % args)
    )
    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", lambda _tensor: object())
    monkeypatch.setattr("pegaflow.connector.worker.pickle.dumps", lambda _wrapper: b"wrapper")

    connector.register_kv_caches(
        {
            "layer.0": FakeTensor((16, 2, 64, 8, 128)),
            "layer.1": FakeTensor((16, 2, 64, 8, 128)),
        }
    )

    assert len(errors) == 1
    assert "logical_block_size=128" in errors[0]
    assert "physical_page_size=64" in errors[0]
    assert "physical_pages_per_logical=2" in errors[0]
    assert "incorrect" in errors[0]
    ctx.engine_client.register_context_batch.assert_called_once()


def test_matching_layout_registration_does_not_log(monkeypatch):
    ctx = _ctx(block_size=128, is_mla=True)
    connector = _connector_for_registration(ctx)
    ctx.engine_client.register_context_batch.return_value = (True, "")

    error = MagicMock()
    monkeypatch.setattr("pegaflow.connector.worker.logger.error", error)
    monkeypatch.setattr("pegaflow.connector.worker.CudaIPCWrapper", lambda _tensor: object())
    monkeypatch.setattr("pegaflow.connector.worker.pickle.dumps", lambda _wrapper: b"wrapper")

    connector.register_kv_caches({"layer.0": FakeTensor((16, 128, 656))})

    error.assert_not_called()
    ctx.engine_client.register_context_batch.assert_called_once()
