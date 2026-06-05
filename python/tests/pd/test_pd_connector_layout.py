from __future__ import annotations

# ruff: noqa: F403,F405,I001
from .pd_connector_test_utils import *


def test_flash_attn_hnd_layout_offsets() -> None:
    # Logical shape [2, num_blocks, block_size, num_kv_heads, head_size].
    # HND physical order [2, num_blocks, num_kv_heads, block_size, head_size].
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )

    layout = FlashAttnHndLayout.from_tensor("layer.0", tensor)

    assert layout.block_bytes == 16 * 4 * 32 * 2
    assert layout.block_offset_bytes(0, 3) == 3 * 4 * 16 * 32 * 2
    assert layout.block_offset_bytes(1, 3) == (8 * 4 * 16 * 32 + 3 * 4 * 16 * 32) * 2


def test_pd_worker_registers_mla_and_indexer_layouts_from_layer_specs() -> None:
    main_tensor = FakeTensor(
        shape=(8, 64, 656),
        stride=(64 * 656, 656, 1),
        ptr=0x1000,
        element_size=1,
    )
    indexer_tensor = FakeTensor(
        shape=(8, 64, 128),
        stride=(64 * 128, 128, 1),
        ptr=0x200000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": SimpleNamespace(block_size=64, page_size_bytes=64 * 656),
            "indexer.0": SimpleNamespace(block_size=64, page_size_bytes=64 * 128),
        },
    )
    worker = PdWorkerConnector(
        fake_mla_config(),
        kv_cache_config=kv_cache_config,
        rdma=MockRdmaPort(),
    )

    worker.register_kv_caches({"layer.0": main_tensor, "indexer.0": indexer_tensor})

    assert worker.layouts["layer.0"].block_size == 64
    assert worker.layouts["layer.0"].remote_layout(0).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x1000, block_len=64 * 656),
    )
    assert worker.layouts["indexer.0"].remote_layout(1).regions == (
        TransferRegionLayout(region_idx=0, base_addr=0x200000, block_len=64 * 128),
    )
    assert (
        worker.rdma.local_layers[0].regions[0].block_len
        != worker.rdma.local_layers[1].regions[0].block_len
    )


def test_pd_connector_mla_returns_default_layout_and_allows_128_block_size() -> None:
    assert PdConnector.get_required_kvcache_layout(fake_mla_config(block_size=64)) is None
    PdConnector(fake_mla_config(block_size=128), KVConnectorRole.WORKER)


def test_pd_worker_rejects_mla_physical_logical_block_split() -> None:
    tensor = FakeTensor(
        shape=(8, 32, 128),
        stride=(32 * 128, 128, 1),
        ptr=0x1000,
        element_size=1,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": SimpleNamespace(block_size=64, page_size_bytes=32 * 128),
        },
    )
    worker = PdWorkerConnector(
        fake_mla_config(block_size=64),
        kv_cache_config=kv_cache_config,
        rdma=MockRdmaPort(),
    )

    with pytest.raises(AssertionError, match="physical/logical block split"):
        worker.register_kv_caches({"layer.0": tensor})


def test_p_worker_maps_mla_prefill_tp_greater_than_decode_tp() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PdWorkerConnector(
        fake_mla_config(tp_rank=2, tp_size=8),
        rdma=MockRdmaPort(),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r2": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert worker.rdma.remote_handshakes["prefill-r2"] is handshakes[1]


def test_p_worker_skips_non_representative_mla_prefill_rank() -> None:
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=64,
            layers=(),
        )
        for rank in range(4)
    )
    worker = PdWorkerConnector(
        fake_mla_config(tp_rank=3, tp_size=8),
        rdma=MockRdmaPort(),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r3": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert "prefill-r3" not in worker.rdma.remote_handshakes
    assert worker.get_finished({"prefill-r3"}) == ({"prefill-r3"}, None)


def test_real_rdma_port_preserves_native_contract_for_pd_push() -> None:
    native_engine = FakeNativeRdmaEngine()
    rdma = RealRdmaPort(native_engine)
    layer = hnd_remote_layer(
        block_ids=(0, 1),
        k_base=0x1000,
        v_base=0x9000,
        block_len=4096,
    )

    registered = rdma.register_local_layers((layer,))

    assert registered[0].mr_desc == {
        "ptr": 0x1000,
        "addr_rkey_list": [["10.0.0.1:1", 17]],
    }
    assert registered[0].block_ids == (0, 1)
    assert registered[0].regions == layer.regions

    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=registered,
    )
    rdma.open_request("req-1", handshake)

    assert native_engine.remote_regs[0][0] == "req-1"
    assert native_engine.remote_regs[0][1]["layers"][0]["mr_desc"]["ptr"] == 0x1000
    assert native_engine.remote_regs[0][1]["layers"][0]["block_ids"] == [0, 1]
    assert native_engine.remote_regs[0][1]["layers"][0]["regions"] == [
        {"region_idx": 0, "base_addr": 0x1000, "block_len": 4096},
        {"region_idx": 1, "base_addr": 0x9000, "block_len": 4096},
    ]

    rdma.push_layer(
        "req-1",
        0,
        [
            FlashAttnHndLayout(
                "layer.0", (2, 8, 16, 4, 32), (16384, 2048, 32, 512, 1), 2, 0x1000
            ).block_slices(1)
        ],
    )
    rdma.push_done("req-1")
    rdma.wait_done("req-1")
    rdma.mark_done("req-1")

    _, _, pushed_blocks = native_engine.pushed_layers[0]
    assert pushed_blocks == [
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 1,
                    "src_offset_bytes": 2048 * 2,
                    "bytes": 4096,
                },
                {
                    "region_idx": 1,
                    "block_id": 1,
                    "src_offset_bytes": (16384 + 2048) * 2,
                    "bytes": 4096,
                },
            ],
        }
    ]
    assert native_engine.done_reqs == ["req-1"]
    assert native_engine.waited_reqs == ["req-1"]
    assert native_engine.marked_reqs == ["req-1"]
    assert rdma.pop_finished_sending() == {"sent-1"}
    assert rdma.pop_finished_sending() == set()
    assert rdma.pop_finished_recving() == {"recv-1"}
    assert rdma.pop_finished_recving() == set()


def test_real_rdma_port_rejects_native_layout_without_block_mapping() -> None:
    native_engine = FakeNativeRdmaEngine()

    def broken_register_local_layers(layers):
        registered = [{**layers[0], "mr_desc": {"ptr": 0x1000, "addr_rkey_list": []}}]
        registered[0].pop("block_ids")
        return registered

    native_engine.register_local_layers = broken_register_local_layers
    rdma = RealRdmaPort(native_engine)
    layer = hnd_remote_layer(block_ids=(0,), block_len=1024)

    with pytest.raises(KeyError, match="block_ids"):
        rdma.register_local_layers((layer,))


def test_pd_handshake_serializes_regions_layout() -> None:
    layer = hnd_remote_layer(
        block_ids=(8, 9, 10),
        k_base=0x10_000,
        v_base=0x20_000,
        block_len=0x400,
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(layer,),
    )

    data = handshake_to_dict(handshake)
    assert data is not None
    layer_data = data["layers"][0]
    assert layer_data["block_ids"] == [8, 9, 10]
    assert layer_data["regions"] == [
        {"region_idx": 0, "base_addr": 0x10_000, "block_len": 0x400},
        {"region_idx": 1, "base_addr": 0x20_000, "block_len": 0x400},
    ]
    assert "k_block_addrs" not in layer_data
    assert "v_block_addrs" not in layer_data
    assert "linear" not in layer_data

    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].block_ids == layer.block_ids
    assert restored.layers[0].regions == layer.regions


def test_pd_handshake_compact_serializes_shared_block_ids_once() -> None:
    layers = (
        hnd_remote_layer(layer_name="layer.0", layer_idx=0, block_ids=(8, 9, 10)),
        hnd_remote_layer(layer_name="layer.1", layer_idx=1, block_ids=(8, 9, 10)),
    )
    handshake = PdHandshake(
        request_id="req-1",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=layers,
    )

    data = handshake_to_compact_dict(handshake)

    assert data["block_ids"] == [8, 9, 10]
    assert "block_ids" not in data["layers"][0]
    assert "block_ids" not in data["layers"][1]
    restored = handshake_from_dict(data)
    assert restored is not None
    assert restored.layers[0].block_ids == (8, 9, 10)
    assert restored.layers[1].block_ids == (8, 9, 10)


def test_pd_worker_builds_native_rdma_by_default_when_extension_exists(monkeypatch) -> None:
    monkeypatch.setattr(native, "PdRdmaEngine", FakeNativeRdmaEngineCtor, raising=False)
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default: {
                "pegaflow.pd.rdma.rank_map": {"0": {"nic": "mlx5_2", "worker_cpu": 64}},
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = PdWorkerConnector(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert isinstance(worker.rdma, RealRdmaPort)
    assert FakeNativeRdmaEngineCtor.last_kwargs == {
        "cuda_device": 2,
        "numa_node": None,
        "domains": ["mlx5_2"],
        "device": "cuda",
        "pin_worker_cpu": 64,
    }


def test_pd_worker_uses_runtime_tp_rank_for_rdma_rank_map(monkeypatch) -> None:
    monkeypatch.setattr(native, "PdRdmaEngine", FakeNativeRdmaEngineCtor, raising=False)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_world_size", lambda: 8)
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default: {
                "pegaflow.pd.rdma.rank_map": {
                    "0": {"nic": "mlx5_1", "worker_cpu": 16},
                    "2": {"nic": "mlx5_2", "worker_cpu": 60},
                },
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=8),
    )

    worker = PdWorkerConnector(config)
    worker.register_kv_caches({"layer.0": tensor})

    assert FakeNativeRdmaEngineCtor.last_kwargs["domains"] == ["mlx5_2"]
    assert FakeNativeRdmaEngineCtor.last_kwargs["pin_worker_cpu"] == 60


def test_pd_worker_rejects_legacy_global_rdma_config(monkeypatch) -> None:
    monkeypatch.setattr(native, "PdRdmaEngine", FakeNativeRdmaEngineCtor, raising=False)
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        device_index=2,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default: {
                "pegaflow.pd.rdma.domains": ["mlx5_2"],
                "pegaflow.pd.rdma.rank_map": {"0": {"nic": "mlx5_2", "worker_cpu": 64}},
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0),
    )

    worker = PdWorkerConnector(config)
    with pytest.raises(RuntimeError, match="legacy keys"):
        worker.register_kv_caches({"layer.0": tensor})


def test_rdma_native_blocks_coalesce_contiguous_ranges() -> None:
    blocks = [
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=10, src_offset_bytes=0x1000, bytes=0x400),
                BlockRegionSlice(block_id=10, src_offset_bytes=0x9000, bytes=0x400),
            ),
        ),
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=11, src_offset_bytes=0x1400, bytes=0x400),
                BlockRegionSlice(block_id=11, src_offset_bytes=0x9400, bytes=0x400),
            ),
        ),
        LayerBlockSlices(
            regions=(
                BlockRegionSlice(block_id=13, src_offset_bytes=0x2000, bytes=0x400),
                BlockRegionSlice(block_id=13, src_offset_bytes=0xA000, bytes=0x400),
            ),
        ),
    ]

    native_blocks = _layer_blocks_to_native(blocks)

    assert native_blocks == [
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 10,
                    "src_offset_bytes": 0x1000,
                    "bytes": 0x800,
                },
                {
                    "region_idx": 1,
                    "block_id": 10,
                    "src_offset_bytes": 0x9000,
                    "bytes": 0x800,
                },
            ],
        },
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 13,
                    "src_offset_bytes": 0x2000,
                    "bytes": 0x400,
                },
                {
                    "region_idx": 1,
                    "block_id": 13,
                    "src_offset_bytes": 0xA000,
                    "bytes": 0x400,
                },
            ],
        },
    ]
