from __future__ import annotations

# ruff: noqa: F403,F405,I001
from .pd_connector_test_utils import *

from pegaflow.pd_connector.layout_mapping import (
    HeadSlice,
    build_push_layout_plan,
    decode_rank_source_counts,
)


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


def test_flash_attn_hnd_head_slice_layouts_use_full_block_stride() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        ptr=0x1000,
    )
    layout = FlashAttnHndLayout.from_tensor("layer.0", tensor)

    slices = layout.block_head_slices(block_id=3, start_head=1, end_head=3)

    assert slices.regions == (
        BlockRegionSlice(
            block_id=3,
            src_offset_bytes=(3 * 4 * 16 * 32 + 1 * 16 * 32) * 2,
            bytes=2 * 16 * 32 * 2,
        ),
        BlockRegionSlice(
            block_id=3,
            src_offset_bytes=(8 * 4 * 16 * 32 + 3 * 4 * 16 * 32 + 1 * 16 * 32) * 2,
            bytes=2 * 16 * 32 * 2,
        ),
    )

    remote = layout.remote_head_layout(
        layer_idx=0,
        block_ids=(3, 4),
        start_head=1,
        end_head=3,
    )

    assert remote.regions == (
        TransferRegionLayout(
            region_idx=0,
            base_addr=0x1000 + 1 * 16 * 32 * 2,
            block_len=2 * 16 * 32 * 2,
            block_stride=4 * 16 * 32 * 2,
        ),
        TransferRegionLayout(
            region_idx=1,
            base_addr=0x1000 + 8 * 4 * 16 * 32 * 2 + 1 * 16 * 32 * 2,
            block_len=2 * 16 * 32 * 2,
            block_stride=4 * 16 * 32 * 2,
        ),
    )


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
    worker = PdDecodeWorkerConnector(
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
    assert PdDecodeConnector.get_required_kvcache_layout(fake_mla_config(block_size=64)) is None
    PdDecodeConnector(fake_mla_config(block_size=128), KVConnectorRole.WORKER)


def test_pd_connectors_allow_mtp_layout() -> None:
    PdDecodeConnector(fake_mtp_config(), KVConnectorRole.WORKER)
    PdPrefillConnector(fake_mtp_config(), KVConnectorRole.WORKER)


def test_legacy_pd_connector_selects_decode_by_engine_id() -> None:
    config = fake_mtp_config()
    config.kv_transfer_config.engine_id = "d0"

    connector = PdConnector(config, KVConnectorRole.WORKER)

    assert isinstance(connector._delegate, PdDecodeConnector)


def test_legacy_pd_connector_selects_prefill_by_engine_id() -> None:
    config = fake_mtp_config()
    config.kv_transfer_config.engine_id = "p0"

    connector = PdConnector(config, KVConnectorRole.WORKER)

    assert isinstance(connector._delegate, PdPrefillConnector)


def test_legacy_pd_connector_forwards_bound_metadata() -> None:
    config = fake_mtp_config()
    config.kv_transfer_config.engine_id = "d0"
    connector = PdConnector(config, KVConnectorRole.WORKER)
    metadata = PdConnectorMetadata()

    connector.bind_connector_metadata(metadata)

    assert connector._connector_metadata is metadata
    assert connector._delegate._connector_metadata is metadata

    connector.clear_connector_metadata()

    assert connector._connector_metadata is None
    assert connector._delegate._connector_metadata is None


def test_legacy_pd_connector_preserves_piecewise_default_for_prefill() -> None:
    assert PdConnector.requires_piecewise_for_cudagraph({}) is True
    assert (
        PdConnector.requires_piecewise_for_cudagraph(
            {"pegaflow.pd.allow_full_decode_cudagraph": True}
        )
        is False
    )


def test_vllm_plugin_registers_split_pd_connectors(monkeypatch) -> None:
    import sys

    import pegaflow.vllm_plugin as vllm_plugin

    registered = []

    class FakeFactory:
        @staticmethod
        def register_connector(name: str, module: str, class_name: str) -> None:
            registered.append((name, module, class_name))

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.factory",
        SimpleNamespace(KVConnectorFactory=FakeFactory),
    )

    vllm_plugin.register()

    assert (
        "PdConnector",
        "pegaflow.pd_connector",
        "PdConnector",
    ) in registered
    assert (
        "PdDecodeConnector",
        "pegaflow.pd_connector",
        "PdDecodeConnector",
    ) in registered
    assert (
        "PdPrefillConnector",
        "pegaflow.pd_connector",
        "PdPrefillConnector",
    ) in registered


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
    worker = PdDecodeWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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


def test_layout_mapping_homogeneous_tp_reads_same_remote_rank() -> None:
    handshakes = decode_handshakes(tp_size=4)

    plan = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=2,
        remote_num_kv_heads=2,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [(2, ())]


def test_layout_mapping_decode_tp_greater_than_prefill_tp_splits_local_heads() -> None:
    handshakes = decode_handshakes(tp_size=4)

    plan = build_push_layout_plan(
        prefill_tp_rank=1,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=4,
        remote_num_kv_heads=2,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [
        (
            2,
            (
                HeadSlice(
                    local_start=0,
                    local_end=2,
                    remote_start=0,
                    remote_end=2,
                    global_heads=(4, 5),
                ),
            ),
        ),
        (
            3,
            (
                HeadSlice(
                    local_start=2,
                    local_end=4,
                    remote_start=0,
                    remote_end=2,
                    global_heads=(6, 7),
                ),
            ),
        ),
    ]


def test_layout_mapping_prefill_tp_greater_than_decode_tp_offsets_remote_heads() -> None:
    handshakes = decode_handshakes(tp_size=2)

    plan = build_push_layout_plan(
        prefill_tp_rank=3,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=2,
        remote_num_kv_heads=4,
        total_num_kv_heads=8,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in plan.targets] == [
        (
            1,
            (
                HeadSlice(
                    local_start=0,
                    local_end=2,
                    remote_start=2,
                    remote_end=4,
                    global_heads=(6, 7),
                ),
            ),
        )
    ]


def test_p_worker_prefill_tp_greater_than_decode_tp_registers_remote_head_slices() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
        ptr=0x1000,
    )
    decode_tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
        ptr=0x2000,
    )
    kv_cache_config = fake_kv_cache_config(
        num_blocks=8,
        specs={
            "layer.0": SimpleNamespace(block_size=16, page_size_bytes=2 * 16 * 4 * 32 * 2),
        },
    )
    decode_layer = FlashAttnHndLayout.from_tensor("layer.0", decode_tensor).remote_layout(
        0,
        (1, 2),
    )
    decode_handshake = PdHandshake(
        request_id="decode",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(decode_layer,),
    )

    def build_worker(rank: int) -> PdPrefillWorkerConnector:
        worker = PdPrefillWorkerConnector(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(engine_id="prefill"),
                model_config=SimpleNamespace(
                    use_mla=False,
                    get_total_num_kv_heads=lambda: 8,
                ),
                cache_config=SimpleNamespace(block_size=16),
                parallel_config=SimpleNamespace(
                    tensor_parallel_rank=rank,
                    tensor_parallel_size=2,
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                ),
            ),
            kv_cache_config=kv_cache_config,
            rdma=MockRdmaPort(),
        )
        worker.register_kv_caches({"layer.0": tensor})
        return worker

    rank0 = build_worker(0)
    rank1 = build_worker(1)

    for worker in (rank0, rank1):
        worker.start_load_kv(
            PdConnectorMetadata(
                reqs_to_push={
                    f"prefill-r{worker.tp_rank}": PushReqMeta(
                        local_block_ids=([1, 2],),
                        target_request_id="decode",
                        handshakes=(decode_handshake,),
                    )
                }
            ),
            None,
        )

    rank0_remote = rank0.rdma.remote_handshakes["prefill-r0"].layers[0]
    rank1_remote = rank1.rdma.remote_handshakes["prefill-r1"].layers[0]

    assert rank0_remote.regions == (
        TransferRegionLayout(
            region_idx=0,
            base_addr=0x2000,
            block_len=4 * 16 * 32 * 2,
            block_stride=8 * 16 * 32 * 2,
        ),
        TransferRegionLayout(
            region_idx=1,
            base_addr=0x2000 + 8 * 8 * 16 * 32 * 2,
            block_len=4 * 16 * 32 * 2,
            block_stride=8 * 16 * 32 * 2,
        ),
    )
    assert rank1_remote.regions == (
        TransferRegionLayout(
            region_idx=0,
            base_addr=0x2000 + 4 * 16 * 32 * 2,
            block_len=4 * 16 * 32 * 2,
            block_stride=8 * 16 * 32 * 2,
        ),
        TransferRegionLayout(
            region_idx=1,
            base_addr=0x2000 + 8 * 8 * 16 * 32 * 2 + 4 * 16 * 32 * 2,
            block_len=4 * 16 * 32 * 2,
            block_stride=8 * 16 * 32 * 2,
        ),
    )


def test_layout_mapping_counts_prefill_sources_per_decode_rank() -> None:
    assert decode_rank_source_counts(
        prefill_tp_size=2,
        decode_tp_size=1,
        local_num_kv_heads=2,
        remote_num_kv_heads=4,
        total_num_kv_heads=4,
        use_mla=False,
    ) == {0: 2}

    assert decode_rank_source_counts(
        prefill_tp_size=1,
        decode_tp_size=2,
        local_num_kv_heads=4,
        remote_num_kv_heads=2,
        total_num_kv_heads=4,
        use_mla=False,
    ) == {0: 1, 1: 1}


def test_layout_mapping_mla_uses_one_remote_rank_and_skips_duplicates() -> None:
    handshakes = decode_handshakes(tp_size=4)

    selected = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=8,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )
    skipped = build_push_layout_plan(
        prefill_tp_rank=3,
        prefill_tp_size=8,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in selected.targets] == [
        (1, ())
    ]
    assert skipped.targets == ()


def test_layout_mapping_mla_prefill_tp_less_than_decode_tp_fans_out() -> None:
    handshakes = decode_handshakes(tp_size=4)

    rank0 = build_push_layout_plan(
        prefill_tp_rank=0,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )
    rank1 = build_push_layout_plan(
        prefill_tp_rank=1,
        prefill_tp_size=2,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in rank0.targets] == [
        (0, ()),
        (1, ()),
    ]
    assert [(target.handshake.tp_rank, target.head_slices) for target in rank1.targets] == [
        (2, ()),
        (3, ()),
    ]
    assert decode_rank_source_counts(
        prefill_tp_size=2,
        decode_tp_size=4,
        local_num_kv_heads=1,
        remote_num_kv_heads=1,
        total_num_kv_heads=1,
        use_mla=True,
    ) == {0: 1, 1: 1, 2: 1, 3: 1}


def test_layout_mapping_gqa_dedup_skips_redundant_prefill_rank() -> None:
    handshakes = decode_handshakes(tp_size=1)

    owner = build_push_layout_plan(
        prefill_tp_rank=0,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=2,
        total_num_kv_heads=2,
        use_mla=False,
    )
    duplicate = build_push_layout_plan(
        prefill_tp_rank=2,
        prefill_tp_size=4,
        decode_handshakes=handshakes,
        local_num_kv_heads=1,
        remote_num_kv_heads=2,
        total_num_kv_heads=2,
        use_mla=False,
    )

    assert [(target.handshake.tp_rank, target.head_slices) for target in owner.targets] == [
        (
            0,
            (
                HeadSlice(
                    local_start=0,
                    local_end=1,
                    remote_start=0,
                    remote_end=1,
                    global_heads=(0,),
                ),
            ),
        )
    ]
    assert duplicate.targets == ()


def test_layout_mapping_rejects_non_divisible_tp_ratios() -> None:
    with pytest.raises(AssertionError, match="requires divisible TP sizes"):
        build_push_layout_plan(
            prefill_tp_rank=0,
            prefill_tp_size=2,
            decode_handshakes=decode_handshakes(tp_size=3),
            local_num_kv_heads=3,
            remote_num_kv_heads=2,
            total_num_kv_heads=6,
            use_mla=False,
        )


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


def test_pd_handshake_serializes_strided_regions_layout() -> None:
    layer = LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        block_ids=(8, 9, 10),
        regions=(
            TransferRegionLayout(
                region_idx=0,
                base_addr=0x10_400,
                block_len=0x400,
                block_stride=0x1000,
            ),
            TransferRegionLayout(
                region_idx=1,
                base_addr=0x20_400,
                block_len=0x400,
                block_stride=0x1000,
            ),
        ),
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

    assert data["layers"][0]["regions"] == [
        {
            "region_idx": 0,
            "base_addr": 0x10_400,
            "block_len": 0x400,
            "block_stride": 0x1000,
        },
        {
            "region_idx": 1,
            "base_addr": 0x20_400,
            "block_len": 0x400,
            "block_stride": 0x1000,
        },
    ]
    restored = handshake_from_dict(data)
    assert restored is not None
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

    worker = PdDecodeWorkerConnector(config)
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

    worker = PdDecodeWorkerConnector(config)
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

    worker = PdDecodeWorkerConnector(config)
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
