from __future__ import annotations

import asyncio

# ruff: noqa: F403,F405,I001
from .pd_connector_test_utils import *


def test_pd_worker_wait_handshake_uses_registered_native_mr_desc() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=RealRdmaPort(native_engine),
    )
    worker.register_kv_caches({"layer.0": tensor})
    meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1],),
                remote_request_id="req-1-p",
                done_request_id="req-1-d",
                prompt_token_ids=(11, 12, 13),
                prefill_url="http://p:8001",
            )
        }
    )

    worker.start_load_kv(meta, None)

    _, handshake = native_engine.remote_regs[-1]
    layer = handshake["layers"][0]
    assert layer["mr_desc"] == {
        "ptr": tensor.data_ptr(),
        "addr_rkey_list": [("10.0.0.1:1", 17)],
    }
    assert layer["block_ids"] == [1]
    assert handshake["expected_imm_count"] == 1


def test_d_worker_waits_for_all_prefill_ranks_when_prefill_tp_is_larger() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="decode",
                extra_config={"pegaflow.pd.prefill_tp_size": 2},
            ),
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 8),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=RealRdmaPort(native_engine),
    )
    worker.register_kv_caches({"layer.0": tensor})

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "req-1": WaitReqMeta(
                    local_block_ids=([1],),
                    remote_request_id="req-1-p",
                    done_request_id="req-1-d",
                    prompt_token_ids=(11, 12, 13),
                    prefill_url="http://p:8001",
                )
            }
        ),
        None,
    )

    _, handshake = native_engine.remote_regs[-1]
    assert handshake["expected_imm_count"] == 2


def test_d_worker_caches_expected_imm_counts(monkeypatch) -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="decode",
                extra_config={"pegaflow.pd.prefill_tp_size": 2},
            ),
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 8),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=RealRdmaPort(native_engine),
    )
    calls = 0
    original = decode_worker_mod.decode_rank_source_counts

    def tracking_decode_rank_source_counts(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        decode_worker_mod,
        "decode_rank_source_counts",
        tracking_decode_rank_source_counts,
    )

    worker.register_kv_caches({"layer.0": tensor})
    assert calls == 1
    for req_id in ("req-1", "req-2"):
        worker.start_load_kv(
            PdConnectorMetadata(
                reqs_to_wait={
                    req_id: WaitReqMeta(
                        local_block_ids=([1],),
                        remote_request_id=f"{req_id}-p",
                        done_request_id=f"{req_id}-d",
                        prompt_token_ids=(11, 12, 13),
                        prefill_url="http://p:8001",
                    )
                }
            ),
            None,
        )

    assert calls == 1
    assert [handshake["expected_imm_count"] for _, handshake in native_engine.remote_regs] == [
        2,
        2,
    ]


def test_pd_worker_pushes_flash_attn_hnd_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target_request_id="req-1",
                handshakes=(
                    PdHandshake(
                        request_id="req-1",
                        engine_id="decode",
                        tp_rank=0,
                        tp_size=1,
                        block_size=16,
                        layers=(),
                    ),
                ),
            )
        }
    )

    worker.start_load_kv(meta, None)
    attn_metadata = SimpleNamespace(slot_mapping=FakeSlotMapping([16, 17, 32, -1]))
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert unique_blocks_from_slot_mapping(attn_metadata.slot_mapping, 16) == {1, 2}
    worker.wait_for_save()
    drain_pd_pushes(worker)
    pushed_by_layer = pushed_layers_by_idx(worker.rdma, "req-1")
    assert set(pushed_by_layer) == {0, 1}
    first_layer_blocks = pushed_by_layer[0]
    assert len(first_layer_blocks) == 1
    assert first_layer_blocks[0].regions[0].block_id == 1
    assert first_layer_blocks[0].regions[0].bytes == tensor.element_size() * 16 * 4 * 32 * 2
    finished_sending, finished_recving = worker.get_finished({"req-1"})
    assert finished_sending == {"req-1"}
    assert finished_recving is None
    assert "req-1" not in worker.rdma.registered


def test_pd_worker_get_finished_does_not_poll_wait_reqs() -> None:
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=rdma
    )
    worker._wait_reqs["req-1"] = WaitReqMeta(
        local_block_ids=([1],),
        remote_request_id="req-1-p",
        done_request_id="req-1-d",
        prompt_token_ids=(1,),
        prefill_url="http://p:8001",
    )

    def fail_poll_done(req_id: str) -> bool:
        raise AssertionError(f"poll_done should not run for {req_id}")

    rdma.poll_done = fail_poll_done  # type: ignore[method-assign]
    rdma.mark_done("req-1")

    assert worker.get_finished(set()) == (None, {"req-1"})


def test_p_worker_save_kv_layer_noops_without_push_reqs() -> None:
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )

    worker.save_kv_layer("unknown-layer", object(), None)


def test_d_worker_idle_decode_step_skips_layer_hooks() -> None:
    class LayoutsThatShouldNotBeRead(dict):
        def __contains__(self, key: object) -> bool:
            raise AssertionError("idle decode step should not inspect layouts")

    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")), rdma=MockRdmaPort()
    )
    worker.layouts = LayoutsThatShouldNotBeRead()
    worker.start_load_kv(PdConnectorMetadata(), None)

    worker._prefill.save_kv_layer = MagicMock(
        side_effect=AssertionError("idle decode step should not delegate save")
    )
    worker._prefill.wait_for_save = MagicMock(
        side_effect=AssertionError("idle decode step should not delegate wait_for_save")
    )

    worker.wait_for_layer_load("layer.0")
    worker.save_kv_layer("layer.0", object(), None)
    worker.wait_for_save()


def test_d_worker_release_cancels_inflight_rdma_wait() -> None:
    class BlockingWaitRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.wait_started = threading.Event()
            self.wait_can_return = threading.Event()
            self.closed_reqs: list[str] = []

        def wait_done(self, req_id: str) -> None:
            self.wait_started.set()
            assert self.wait_can_return.wait(timeout=5), "wait_done was not released"

        def close_request(self, req_id: str) -> None:
            self.closed_reqs.append(req_id)
            super().close_request(req_id)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = BlockingWaitRdma()
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")), rdma=rdma
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "req-1": WaitReqMeta(
                    local_block_ids=([1],),
                    remote_request_id="req-1-p",
                    done_request_id="req-1-d",
                    prompt_token_ids=(1,),
                    prefill_url="",
                )
            }
        ),
        None,
    )
    assert rdma.wait_started.wait(timeout=5), "waiter did not start"

    worker.start_load_kv(PdConnectorMetadata(reqs_to_release={"req-1"}), None)

    waiter = worker._decode._rdma_waiter
    assert waiter is not None
    with waiter._lock:
        assert "req-1" not in waiter._submitted
    assert "req-1" not in worker._decode.wait_reqs
    assert "req-1" not in rdma.registered
    assert rdma.closed_reqs == ["req-1"]

    rdma.wait_can_return.set()


def test_d_worker_reregister_keeps_new_rdma_wait_after_old_wait_exits() -> None:
    class SequencedWaitRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.wait_calls = 0
            self.first_started = threading.Event()
            self.first_can_return = threading.Event()
            self.second_started = threading.Event()
            self.second_can_return = threading.Event()

        def wait_done(self, req_id: str) -> None:
            with self._lock:
                self.wait_calls += 1
                wait_call = self.wait_calls
            if wait_call == 1:
                self.first_started.set()
                assert self.first_can_return.wait(timeout=5), "first wait was not released"
                return
            if wait_call == 2:
                self.second_started.set()
                assert self.second_can_return.wait(timeout=5), "second wait was not released"
                return
            raise AssertionError(f"unexpected wait call {wait_call} for {req_id}")

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = SequencedWaitRdma()
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")), rdma=rdma
    )
    worker.register_kv_caches({"layer.0": tensor})
    wait_meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1],),
                remote_request_id="req-1-p",
                done_request_id="req-1-d",
                prompt_token_ids=(1,),
                prefill_url="",
            )
        }
    )
    worker.start_load_kv(wait_meta, None)
    assert rdma.first_started.wait(timeout=5), "first wait did not start"

    worker.start_load_kv(PdConnectorMetadata(reqs_to_release={"req-1"}), None)
    worker.start_load_kv(wait_meta, None)
    rdma.first_can_return.set()
    assert rdma.second_started.wait(timeout=5), "second wait did not start"

    waiter = worker._decode._rdma_waiter
    assert waiter is not None
    with waiter._lock:
        assert "req-1" in waiter._submitted

    worker.start_load_kv(PdConnectorMetadata(reqs_to_release={"req-1"}), None)
    rdma.second_can_return.set()


def test_p_worker_release_closes_all_physical_decode_targets() -> None:
    class TrackingRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.closed_reqs: list[str] = []

        def close_request(self, req_id: str) -> None:
            self.closed_reqs.append(req_id)
            super().close_request(req_id)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    handshakes = tuple(
        PdHandshake(
            request_id=f"decode-r{rank}",
            engine_id="decode",
            tp_rank=rank,
            tp_size=4,
            block_size=16,
            layers=(
                hnd_remote_layer(
                    layer_name="layer.0",
                    layer_idx=0,
                    block_ids=(1,),
                    block_len=2 * 16 * 32 * 2,
                ),
            ),
        )
        for rank in range(4)
    )
    rdma = TrackingRdma()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=1, tensor_parallel_size=2),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )
    assert sorted(rdma.registered) == ["prefill-r1#d2", "prefill-r1#d3"]

    worker.start_load_kv(PdConnectorMetadata(reqs_to_release={"prefill-r1"}), None)

    assert sorted(rdma.closed_reqs) == ["prefill-r1#d2", "prefill-r1#d3"]
    assert rdma.registered == set()


def test_p_worker_uses_scheduler_blocks_without_slot_mapping_cpu_sync() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )

    slot_mapping = FakeSlotMapping([999])
    attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    assert slot_mapping.cpu_calls == 0
    worker.wait_for_save()
    drain_pd_pushes(worker)
    pushed_by_layer = pushed_layers_by_idx(worker.rdma, "req-1")
    assert set(pushed_by_layer) == {0, 1}


def test_p_worker_save_does_not_require_slot_mapping() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )

    attn_metadata = SimpleNamespace()
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.wait_for_save()
    worker.save_kv_layer("layer.1", tensor, attn_metadata)

    drain_pd_pushes(worker)
    pushed_by_layer = pushed_layers_by_idx(worker.rdma, "req-1")
    assert set(pushed_by_layer) == {0, 1}


def test_pd_worker_publishes_wait_handshake_and_delays_done_until_all_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    d_rdma = MockRdmaPort()
    prefill_sender = FakePrefillSender()
    d_worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=d_rdma,
        prefill_sender=prefill_sender,
    )
    d_worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})

    wait_meta = PdConnectorMetadata(
        reqs_to_wait={
            "req-1": WaitReqMeta(
                local_block_ids=([1, 2],),
                remote_request_id="req-1",
                done_request_id="req-1",
                prompt_token_ids=(101, 102, 103),
                prefill_url="http://p:8001",
            )
        }
    )
    d_worker.start_load_kv(wait_meta, None)

    wait_handshake = d_rdma.remote_handshakes.get("req-1")
    assert wait_handshake is not None
    assert wait_handshake.engine_id == "decode"
    assert wait_handshake.block_size == 16
    assert isinstance(wait_handshake.imm_id, int)
    assert len(wait_handshake.layers) == 1
    assert wait_handshake.layers[0].block_ids == (1,)

    assert len(prefill_sender.tasks) == 1
    task = prefill_sender.tasks[0]
    handshake = handshake_from_dict(task.kv_transfer_params["pd_handshakes"][0])
    assert handshake is not None
    assert handshake.engine_id == "decode"
    assert handshake.block_size == 16
    assert handshake.imm_id == wait_handshake.imm_id
    assert handshake.layers[0].block_ids == (1, 2)
    assert handshake.layers[0].regions[0] == TransferRegionLayout(
        region_idx=0,
        base_addr=tensor.data_ptr(),
        block_len=4 * 16 * 32 * 2,
    )

    push_worker = PdWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )
    push_worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    push_meta = PdConnectorMetadata(
        reqs_to_push={
            "req-1": PushReqMeta(
                local_block_ids=([1, 2],),
                target_request_id="req-1",
                handshakes=(handshake,),
            )
        }
    )
    push_worker.start_load_kv(push_meta, None)
    assert push_worker.rdma.remote_handshakes["req-1"] is handshake

    push_worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="req-1",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )
    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    )
    push_worker.wait_for_save()
    drain_pd_pushes(push_worker)
    assert push_worker.get_finished(set()) == (None, None)

    push_worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "req-1": PushReqMeta(
                    local_block_ids=([2],),
                    target_request_id="req-1",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )
    push_worker.save_kv_layer(
        "layer.0", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    push_worker.save_kv_layer(
        "layer.1", tensor, SimpleNamespace(slot_mapping=FakeSlotMapping([32]))
    )
    push_worker.wait_for_save()
    drain_pd_pushes(push_worker)
    assert push_worker.get_finished({"req-1"}) == ({"req-1"}, None)


def test_scheduler_delays_producer_block_free_until_send_finishes() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "req-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())
    assert set(meta.reqs_to_push) == {"req-1"}

    delay_free, params = scheduler.request_finished(request, ([1, 2],))
    assert delay_free is True
    assert params is None

    early_release_meta = scheduler.build_connector_meta(SimpleNamespace())
    assert early_release_meta.reqs_to_release == set()

    scheduler.update_connector_output(
        SimpleNamespace(finished_sending={"req-1"}, finished_recving=None)
    )
    release_meta = scheduler.build_connector_meta(SimpleNamespace())
    assert release_meta.reqs_to_release == {"req-1"}


def test_scheduler_emits_cached_producer_chunks() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "decode-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    first = scheduler.build_connector_meta(
        SimpleNamespace(scheduled_cached_reqs=None, total_num_scheduled_tokens=8192)
    )
    assert first.reqs_to_push["req-1"].local_block_ids == ([1, 2],)

    cached = SimpleNamespace(req_ids=["req-1"], new_block_ids=[([3],)])
    second = scheduler.build_connector_meta(
        SimpleNamespace(scheduled_cached_reqs=cached, total_num_scheduled_tokens=1808)
    )

    assert second.reqs_to_push["req-1"].local_block_ids == ([3],)
    assert second.reqs_to_push["req-1"].target_request_id == "decode-1"


def test_scheduler_carries_prompt_tokens_for_d_to_p_oob() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={"do_remote_prefill": True, "prefill_url": "http://p:8001"},
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].prompt_token_ids == (11, 12, 13)


def test_scheduler_carries_prefill_max_tokens_for_d_to_p_oob() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://p:8001",
            "prefill_max_tokens": 3,
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].prefill_max_tokens == 3


def test_consumer_params_reject_invalid_prefill_max_tokens() -> None:
    with pytest.raises(ValueError, match="prefill_max_tokens must be positive"):
        parse_consumer(
            {
                "do_remote_prefill": True,
                "prefill_url": "http://p:8001",
                "prefill_max_tokens": 0,
            }
        )


def test_scheduler_carries_cross_process_rdma_handshake() -> None:
    handshake = {
        "request_id": "decode-1",
        "engine_id": "decode",
        "tp_rank": 0,
        "tp_size": 1,
        "block_size": 16,
        "layers": [
            {
                "layer_name": "layer.0",
                "layer_idx": 0,
                "block_ids": [1],
                "regions": [
                    {"region_idx": 0, "base_addr": 0x1000, "block_len": 1024},
                    {"region_idx": 1, "base_addr": 0x1400, "block_len": 1024},
                ],
                "mr_desc": {"addr_rkey_list": [["10.0.0.1:1", 17]]},
            }
        ],
    }
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="prefill-1",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "decode-1",
            "pd_handshakes": [handshake],
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    push_req = meta.reqs_to_push["prefill-1"]
    assert len(push_req.handshakes) == 1
    parsed = push_req.handshakes[0]
    assert parsed.request_id == "decode-1"
    assert parsed.layers[0].mr_desc == {"addr_rkey_list": [["10.0.0.1:1", 17]]}


def test_scheduler_ignores_legacy_fake_rdma_done_endpoint() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://p:8001",
            "done_endpoint": "tcp://127.0.0.1:7200",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_wait["req-1"].done_request_id == "req-1"


def test_scheduler_registers_remote_wait_once_until_done() -> None:
    scheduler = PdSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://127.0.0.1:8001",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    first = scheduler.build_connector_meta(SimpleNamespace())
    assert set(first.reqs_to_wait) == {"req-1"}

    # After finished_recving, _completed_waits prevents re-registration (preemption safety)
    scheduler.update_connector_output(
        SimpleNamespace(finished_sending=None, finished_recving={"req-1"})
    )
    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    third = scheduler.build_connector_meta(SimpleNamespace())
    assert third.reqs_to_wait == {}

    scheduler.request_finished(request, ([1],))
    scheduler.build_connector_meta(SimpleNamespace())
    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    after_release = scheduler.build_connector_meta(SimpleNamespace())
    assert set(after_release.reqs_to_wait) == {"req-1"}


def test_d_worker_rank0_dispatches_prefill_on_wait() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=MockRdmaPort(),
        prefill_sender=prefill_sender,
    )
    worker.register_kv_caches({"layer.0": tensor})
    meta = PdConnectorMetadata(
        reqs_to_wait={
            "internal-d": WaitReqMeta(
                local_block_ids=([1],),
                remote_request_id="external-p",
                done_request_id="external-d",
                prompt_token_ids=(11, 12, 13),
                model="/data/Qwen3-4B",
                prefill_url="http://127.0.0.1:8001",
                prefill_max_tokens=1,
            )
        }
    )

    worker.start_load_kv(meta, None)

    assert len(prefill_sender.tasks) == 1
    task = prefill_sender.tasks[0]
    assert task.request_id == "external-p"
    assert task.kv_transfer_params["target_request_id"] == "external-d"
    handshakes = task.kv_transfer_params["pd_handshakes"]
    assert len(handshakes) == 1
    assert handshakes[0]["request_id"] == "external-d"
    assert handshakes[0]["block_ids"] == [1]
    assert "block_ids" not in handshakes[0]["layers"][0]
    parsed = handshake_from_dict(handshakes[0])
    assert parsed is not None
    assert parsed.layers[0].block_ids == (1,)


def test_layer_push_sender_keeps_fifo_order() -> None:
    class Event:
        def __init__(self) -> None:
            self.ready = threading.Event()

        def synchronize(self) -> None:
            assert self.ready.wait(timeout=2)

    class RecordingRdma:
        def __init__(self) -> None:
            self.pushed: queue.Queue[int] = queue.Queue()

        def push_layer(self, req_id, layer_idx, blocks) -> None:
            self.pushed.put(layer_idx)

    first_event = Event()
    ready_event = Event()
    ready_event.ready.set()
    rdma = RecordingRdma()
    sender = prefill_worker_mod._AsyncLayerPushSender()
    try:
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="req",
                layer_idx=0,
                block_slices=[],
                event=first_event,
            )
        )
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="req",
                layer_idx=1,
                block_slices=[],
                event=ready_event,
            )
        )

        with pytest.raises(queue.Empty):
            rdma.pushed.get(timeout=0.1)
        first_event.ready.set()
        sender.wait_req("req")

        assert rdma.pushed.get(timeout=2) == 0
        assert rdma.pushed.get(timeout=2) == 1
    finally:
        first_event.ready.set()
        sender.close()


def _prefill_http_task(request_id: str) -> PrefillHttpTask:
    return PrefillHttpTask(
        request_id=request_id,
        prefill_url="http://p:8001",
        model="model",
        prompt_token_ids=(1,),
        max_tokens=1,
        kv_transfer_params={"target_request_id": request_id},
    )


def test_async_prefill_sender_runs_requests_concurrently(monkeypatch) -> None:
    entered: queue.Queue[str] = queue.Queue()
    release = threading.Event()

    async def fake_post_prefill_request(task: PrefillHttpTask, _client=None) -> None:
        entered.put(task.request_id)
        assert await asyncio.to_thread(release.wait, 2)

    monkeypatch.setattr(
        prefill_mod,
        "post_prefill_request_async",
        fake_post_prefill_request,
    )
    sender = AsyncPrefillSender(worker_count=2)
    try:
        sender.submit(_prefill_http_task("req-1"))
        sender.submit(_prefill_http_task("req-2"))

        assert {entered.get(timeout=2), entered.get(timeout=2)} == {"req-1", "req-2"}
    finally:
        release.set()
        sender.close()


def test_p_worker_selects_matching_tp_rank_handshake() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    handshakes = (
        PdHandshake(
            request_id="decode-r0",
            engine_id="decode",
            tp_rank=0,
            tp_size=2,
            block_size=16,
            layers=(),
        ),
        PdHandshake(
            request_id="decode-r1",
            engine_id="decode",
            tp_rank=1,
            tp_size=2,
            block_size=16,
            layers=(),
        ),
    )
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=1, tensor_parallel_size=2),
        ),
        rdma=MockRdmaPort(),
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=handshakes,
                )
            }
        ),
        None,
    )

    assert worker.rdma.remote_handshakes["prefill-r1"] is handshakes[1]


def test_p_worker_pushes_registered_blocks_from_save_kv_layer() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode",
                    handshakes=(
                        PdHandshake(
                            request_id="decode",
                            engine_id="decode",
                            tp_rank=0,
                            tp_size=1,
                            block_size=16,
                            layers=(),
                        ),
                    ),
                )
            }
        ),
        None,
    )

    attn_metadata = SimpleNamespace(slot_mapping=FakeSlotMapping([16]))
    worker.save_kv_layer("layer.0", tensor, attn_metadata)
    worker.save_kv_layer("layer.1", tensor, attn_metadata)
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert {layer_idx for layer_idx, _ in rdma.pushed_layers["prefill-r0"]} == {0, 1}
    assert worker.get_finished({"prefill-r0"})[0] == {"prefill-r0"}


def test_p_worker_pushes_to_multiple_decode_ranks_when_decode_tp_is_larger() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=1, tensor_parallel_size=2),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r1": PushReqMeta(
                    local_block_ids=([3],),
                    target_request_id="decode",
                    handshakes=tuple(
                        PdHandshake(
                            request_id=f"decode-r{rank}",
                            engine_id="decode",
                            tp_rank=rank,
                            tp_size=4,
                            block_size=16,
                            layers=(hnd_remote_layer(block_ids=(68,), block_len=2048),),
                        )
                        for rank in range(4)
                    ),
                )
            }
        ),
        None,
    )

    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16])),
    )
    drain_pd_pushes(worker)

    assert sorted(rdma.remote_handshakes) == ["prefill-r1#d2", "prefill-r1#d3"]
    assert rdma.remote_handshakes["prefill-r1#d2"].request_id == "decode-r2"
    assert rdma.remote_handshakes["prefill-r1#d3"].request_id == "decode-r3"
    d2_push = rdma.pushed_layers["prefill-r1#d2"][0][1]
    d3_push = rdma.pushed_layers["prefill-r1#d3"][0][1]
    assert d2_push[0].regions[0] == BlockRegionSlice(
        block_id=68,
        src_offset_bytes=(3 * 4 * 16 * 32) * 2,
        bytes=2 * 16 * 32 * 2,
    )
    assert d3_push[0].regions[0] == BlockRegionSlice(
        block_id=68,
        src_offset_bytes=(3 * 4 * 16 * 32 + 2 * 16 * 32) * 2,
        bytes=2 * 16 * 32 * 2,
    )
    assert worker.get_finished({"prefill-r1"})[0] == {"prefill-r1"}
    assert "prefill-r1#d2" not in rdma.registered
    assert "prefill-r1#d3" not in rdma.registered


def test_p_worker_offsets_remote_heads_when_prefill_tp_is_larger() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 2, 32),
        stride=(8 * 2 * 16 * 32, 2 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=3, tensor_parallel_size=4),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r3": PushReqMeta(
                    local_block_ids=([3],),
                    target_request_id="decode",
                    handshakes=tuple(
                        PdHandshake(
                            request_id=f"decode-r{rank}",
                            engine_id="decode",
                            tp_rank=rank,
                            tp_size=2,
                            block_size=16,
                            layers=(hnd_remote_layer(block_ids=(68,), block_len=4096),),
                        )
                        for rank in range(2)
                    ),
                )
            }
        ),
        None,
    )

    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16])),
    )
    drain_pd_pushes(worker)

    assert sorted(rdma.remote_handshakes) == ["prefill-r3"]
    assert rdma.remote_handshakes["prefill-r3"].request_id == "decode-r1"
    remote_layer = rdma.remote_handshakes["prefill-r3"].layers[0]
    assert remote_layer.regions[0].base_addr == 0x1000 + 2 * 16 * 32 * 2
    assert remote_layer.regions[0].block_len == 2 * 16 * 32 * 2
    assert remote_layer.regions[0].block_stride == 4 * 16 * 32 * 2
    pushed = rdma.pushed_layers["prefill-r3"][0][1]
    assert pushed[0].regions[0] == BlockRegionSlice(
        block_id=68,
        src_offset_bytes=(3 * 2 * 16 * 32) * 2,
        bytes=2 * 16 * 32 * 2,
    )


def test_p_worker_maps_local_blocks_to_remote_blocks_by_position() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([3, 4],),
                    target_request_id="decode",
                    handshakes=(
                        PdHandshake(
                            request_id="decode",
                            engine_id="decode",
                            tp_rank=0,
                            tp_size=1,
                            block_size=16,
                            layers=(
                                hnd_remote_layer(
                                    block_ids=(68, 69),
                                    block_len=4096,
                                ),
                            ),
                        ),
                    ),
                )
            }
        ),
        None,
    )

    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16, 4 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    _, pushed = rdma.pushed_layers["prefill-r0"][0]
    assert [block.regions[0].block_id for block in pushed] == [68]
    assert len(rdma.pushed_layers["prefill-r0"]) == 1
    assert [block.regions[0].src_offset_bytes for block in pushed] == [
        tensor.stride()[1] * 3 * tensor.element_size()
    ]
    assert pushed[0].regions[0].bytes == tensor.stride()[1] * tensor.element_size() * 2


def test_p_worker_advances_remote_blocks_across_chunk_prefill() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="prefill"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    handshake = PdHandshake(
        request_id="decode",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(
            hnd_remote_layer(
                block_ids=(68, 69, 70, 71),
                block_len=4096,
            ),
        ),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([3, 4],),
                    target_request_id="decode",
                    handshakes=(handshake,),
                )
            }
        ),
        None,
    )
    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([3 * 16, 4 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert worker.get_finished(set())[0] is None
    assert [block.regions[0].block_id for block in rdma.pushed_layers["prefill-r0"][0][1]] == [68]
    assert len(rdma.pushed_layers["prefill-r0"]) == 1

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([5, 6],),
                    target_request_id="decode",
                    handshakes=(handshake,),
                )
            }
        ),
        None,
    )
    worker.save_kv_layer(
        "layer.0",
        tensor,
        SimpleNamespace(slot_mapping=FakeSlotMapping([5 * 16, 6 * 16])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert [block.regions[0].block_id for block in rdma.pushed_layers["prefill-r0"][1][1]] == [70]
    assert worker.get_finished({"prefill-r0"})[0] == {"prefill-r0"}


def test_pd_proxy_only_sends_decode_request_with_prefill_hint() -> None:
    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
        timeout_s=30,
        prefill_max_tokens=1,
    )

    req = build_pd_proxy_request(
        {
            "model": "/data/Qwen3-4B",
            "prompt": "hello",
            "max_tokens": 4,
        },
        config,
        request_id="pd-test",
    )

    assert req.decode_body["request_id"] == "pd-test-d"
    assert req.decode_body["max_tokens"] == 4
    assert "stream" not in req.decode_body
    assert req.decode_body["kv_transfer_params"] == {
        "do_remote_prefill": True,
        "prefill_url": "http://127.0.0.1:8001",
        "remote_request_id": "pd-test-p",
        "done_request_id": "pd-test-d",
        "prefill_max_tokens": 1,
    }


def test_pd_proxy_preserves_streaming_decode_request() -> None:
    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
        timeout_s=30,
        prefill_max_tokens=1,
    )

    req = build_pd_proxy_request(
        {
            "model": "/data/Qwen3-4B",
            "prompt": "hello",
            "max_tokens": 4,
            "stream": True,
        },
        config,
        request_id="pd-test",
    )

    assert req.decode_body["stream"] is True


def test_pd_proxy_streams_sse_lines_without_large_read_buffering() -> None:
    class SlowSseBody:
        def __init__(self) -> None:
            self.lines = iter(
                [
                    b"data: {\"choices\":[{\"text\":\"a\"}]}\n",
                    b"\n",
                    b"data: {\"choices\":[{\"text\":\"b\"}]}\n",
                    b"\n",
                    b"data: [DONE]\n",
                    b"\n",
                ]
            )

        def readline(self):
            return next(self.lines, b"")

        def read(self, _size=-1):
            raise AssertionError("stream forwarding must not use large buffered reads")

    assert list(iter_http_stream_chunks(SlowSseBody())) == [
        b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n",
        b"data: {\"choices\":[{\"text\":\"b\"}]}\n\n",
        b"data: [DONE]\n\n",
    ]


def test_pd_proxy_fastapi_route_injects_request(monkeypatch) -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    responses = pytest.importorskip("fastapi.responses")

    from pegaflow.pd_connector import proxy as proxy_mod

    config = ProxyConfig(
        prefill_url="http://127.0.0.1:8001",
        decode_url="http://127.0.0.1:8002",
        timeout_s=30,
        prefill_max_tokens=1,
    )
    observed: dict[str, Any] = {}

    async def handle_openai_request(self, path: str, body: dict[str, Any]):
        observed["path"] = path
        observed["body"] = body
        return responses.Response(
            content=b"{\"ok\":true}",
            media_type="application/json",
        )

    monkeypatch.setattr(
        proxy_mod.PdProxy,
        "handle_openai_request",
        handle_openai_request,
    )

    app = proxy_mod.create_app(config)
    with testclient.TestClient(app) as client:
        response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}
    assert observed == {
        "path": "/v1/completions",
        "body": {"prompt": "hello"},
    }


def test_pd_connector_requires_piecewise_cudagraph_by_default() -> None:
    assert PdConnector.requires_piecewise_for_cudagraph({}) is True


def test_pd_connector_can_allow_full_decode_cudagraph() -> None:
    assert (
        PdConnector.requires_piecewise_for_cudagraph(
            {"pegaflow.pd.allow_full_decode_cudagraph": True}
        )
        is False
    )
