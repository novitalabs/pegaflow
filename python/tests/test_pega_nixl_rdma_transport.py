from __future__ import annotations

from types import SimpleNamespace

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.nixl_connector.rdma_transport import PegaNixlRdmaTransport
from pegaflow.pd_connector.metadata import LayerRemoteLayout
from pegaflow.pd_connector.rdma import MockRdmaPort


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2),
        kv_transfer_config=SimpleNamespace(engine_id="d0"),
    )


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, element_size: int = 2) -> None:
        self.shape = shape
        self._element_size = element_size
        self._strides = self._contiguous_strides(shape)
        self._base_addr = 0x100000

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def element_size(self) -> int:
        return self._element_size

    def data_ptr(self) -> int:
        return self._base_addr

    def get_device(self) -> int:
        return 0

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = 1
        result = []
        for dim in reversed(shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))


class RegisteringRdmaPort(MockRdmaPort):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        registered = tuple(
            LayerRemoteLayout(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                block_ids=layer.block_ids,
                regions=layer.regions,
                mr_desc={"test_mr": layer.layer_idx},
            )
            for layer in layers
        )
        self.local_layers = registered
        return registered


def test_transport_registers_layers_and_builds_handshake() -> None:
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="d0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    kv_cache = FakeTensor((2, 4, 2, 1, 8))

    transport.register_kv_caches({"layer.0": kv_cache})
    handshake = transport.build_local_handshake("decode-req", [1, 3])

    assert handshake.request_id == "decode-req"
    assert handshake.engine_id == "d0"
    assert handshake.block_size == 2
    assert [layer.layer_name for layer in handshake.layers] == ["layer.0"]
    assert handshake.layers[0].block_ids == (1, 3)
    assert handshake.layers[0].mr_desc is not None


def test_transport_pushes_matching_local_blocks_to_remote_blocks() -> None:
    rdma = RegisteringRdmaPort()
    transport = PegaNixlRdmaTransport(
        vllm_config=_vllm_config(),
        engine_id="p0",
        tp_rank=0,
        tp_size=1,
        rdma=rdma,
    )
    kv_cache = FakeTensor((2, 8, 2, 1, 8))
    transport.register_kv_caches({"layer.0": kv_cache})
    handshake = transport.build_local_handshake("decode-req", [5, 7])

    transport.push_blocks(
        request_id="prefill-req",
        remote_handshake=handshake,
        local_block_ids=([1, 3],),
        remote_block_ids=([5, 7],),
    )

    pushed = rdma.pushed_layers["prefill-req"]
    assert len(pushed) == 1
    layer_idx, blocks = pushed[0]
    assert layer_idx == 0
    assert [block.regions[0].block_id for block in blocks] == [5, 7]
    assert [block.regions[0].src_offset_bytes for block in blocks] == [
        kv_cache.stride()[1] * kv_cache.element_size(),
        3 * kv_cache.stride()[1] * kv_cache.element_size(),
    ]
    assert "prefill-req" in rdma.pop_finished_sending()
