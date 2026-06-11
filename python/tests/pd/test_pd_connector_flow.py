from __future__ import annotations

import asyncio
import time
from http import HTTPStatus

# ruff: noqa: F403,F405,I001
from .pd_connector_test_utils import *


def test_pd_worker_wait_handshake_uses_registered_native_mr_desc() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdDecodeWorkerConnector(
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
        "addr_rkey_list": [["10.0.0.1:1", 17]],
    }
    assert layer["block_ids"] == [1]
    assert handshake["expected_imm_count"] == 1
    assert handshake["fail_imm_id"] == (handshake["imm_id"] ^ 0x8000_0000)
    assert handshake["abort_imm_id"] == (handshake["imm_id"] ^ 0x4000_0000)


def test_pd_connector_exposes_empty_stats_for_vllm_metrics() -> None:
    connector = PdDecodeConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")),
        KVConnectorRole.SCHEDULER,
    )

    stats = connector.get_kv_connector_stats()

    assert stats is not None
    assert stats.data["pd_decode_active_waits"] == 0
    assert stats.data["pd_prefill_active_pushes"] == 0
    assert stats.is_empty()


def test_pd_prom_metrics_observes_connector_stats(monkeypatch) -> None:
    class FakeGauge:
        def __init__(self, **_kwargs) -> None:
            self.value = None

        def set(self, value) -> None:
            self.value = value

    class FakeHistogram:
        def __init__(self, **_kwargs) -> None:
            self.values = []

        def observe(self, value) -> None:
            self.values.append(value)

    class FakeCounter:
        def __init__(self, **_kwargs) -> None:
            self.value = 0

        def inc(self, value=1) -> None:
            self.value += value

    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
    )

    import pegaflow.pd_connector.metrics as pd_metrics_mod

    monkeypatch.setattr(
        pd_metrics_mod,
        "_bind_metric_per_engine",
        lambda _prom_metrics, metric: {0: metric},
    )
    monkeypatch.setattr(
        KVConnectorPromMetrics,
        "__init__",
        lambda self, _vllm_config, _metric_types, _labelnames, per_engine_labelvalues: setattr(
            self,
            "per_engine_labelvalues",
            per_engine_labelvalues,
        ),
    )

    monkeypatch.setattr(pd_metrics_mod.PdPromMetrics, "_gauge_cls", FakeGauge, raising=False)
    monkeypatch.setattr(
        pd_metrics_mod.PdPromMetrics,
        "_histogram_cls",
        FakeHistogram,
        raising=False,
    )
    monkeypatch.setattr(pd_metrics_mod.PdPromMetrics, "_counter_cls", FakeCounter, raising=False)

    prom = PdDecodeConnector.build_prom_metrics(
        SimpleNamespace(kv_transfer_config=SimpleNamespace()),
        {PromMetric: PromMetric},
        [],
        {0: []},
    )
    stats = PdDecodeConnector.build_kv_connector_stats(
        {
            "pd_decode_active_waits": 2,
            "pd_prefill_active_pushes": 1,
            "pd_prefill_inflight_push_tasks": 3,
            "pd_prefill_inflight_finalize_tasks": 4,
            "pd_decode_wait_duration": [0.1],
            "pd_decode_rdma_wait_duration": [0.2],
            "pd_decode_prefill_http_submit_duration": [0.003],
            "pd_load_blocks": [8],
            "pd_prefill_push_duration": [0.4],
            "pd_prefill_first_save_to_done_duration": [0.5],
            "pd_prefill_wait_for_pushes_duration": [0.6],
            "pd_prefill_push_blocks": [8],
            "pd_prefill_push_bytes": [1024],
            "pd_prefill_push_gbps": [2.5],
            "pd_load_success_count": 1,
            "pd_load_failure_count": 2,
            "pd_prefill_push_success_count": 3,
            "pd_prefill_push_failure_count": 4,
            "pd_decode_abort_count": 5,
            "pd_prefill_release_count": 6,
            "pd_prefill_skipped_push_count": 7,
        }
    )

    prom.observe(stats.data, engine_idx=0)

    assert prom.gauge_decode_active_waits[0].value == 2
    assert prom.hist_decode_wait_duration[0].values == [0.1]
    assert prom.hist_prefill_push_gbps[0].values == [2.5]
    assert prom.counter_load_success[0].value == 1
    assert prom.counter_prefill_skipped_push[0].value == 7


def test_pd_proxy_metrics_render_low_cardinality_route_stats() -> None:
    router = RoundRobinPairRouter(
        prefill_endpoints=(
            PdEndpoint("http://p0:8000", instance_id="p0"),
            PdEndpoint("http://p1:8000", instance_id="p1"),
        ),
        decode_endpoints=(PdEndpoint("http://d0:8000", instance_id="d0"),),
    )
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
        router=router,
    )
    build_pd_proxy_request({"request_id": "secret-request"}, config)

    rendered = render_proxy_metrics(config).decode()

    assert "pega_pd_proxy_route_total" in rendered
    assert 'prefill_instance="p0"' in rendered
    assert 'decode_instance="d0"' in rendered
    assert "secret-request" not in rendered


def test_pd_proxy_counts_non_stream_decode_http_errors(monkeypatch) -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    def fake_post_json(*_args, **_kwargs):
        return HTTPStatus.BAD_GATEWAY, b'{"error":"decode failed"}', "application/json"

    monkeypatch.setattr(proxy_mod, "_post_json", fake_post_json)
    monkeypatch.setattr(proxy_mod.PdProxy, "_get_client", lambda self: object())
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    status, _payload, _content_type = proxy.handle_openai_request(
        "/v1/completions",
        {"request_id": "req-1", "model": "model", "prompt": "hello"},
    )

    metrics = config.metrics.snapshot()
    assert status == HTTPStatus.BAD_GATEWAY
    assert metrics["request_count"] == 1
    assert metrics["error_count"] == 1


def test_pd_proxy_reuses_non_stream_http_client(monkeypatch) -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    class FakeResponse:
        status_code = 200
        content = b'{"text":"ok"}'
        headers = {"Content-Type": "application/json"}

    class FakeClient:
        created = 0
        closed = 0

        def __init__(self, **_kwargs) -> None:
            type(self).created += 1

        def post(self, *_args, **_kwargs):
            return FakeResponse()

        def close(self) -> None:
            type(self).closed += 1

    monkeypatch.setattr(
        proxy_mod,
        "_httpx",
        lambda: SimpleNamespace(
            Client=FakeClient,
            Limits=lambda **kwargs: kwargs,
            RequestError=Exception,
        ),
    )
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    first = proxy.handle_openai_request(
        "/v1/completions",
        {"request_id": "req-1", "model": "model", "prompt": "hello"},
    )
    second = proxy.handle_openai_request(
        "/v1/completions",
        {"request_id": "req-2", "model": "model", "prompt": "world"},
    )

    assert first[0] == HTTPStatus.OK
    assert second[0] == HTTPStatus.OK
    assert FakeClient.created == 1
    assert FakeClient.closed == 0

    proxy.close()

    assert FakeClient.closed == 1


def test_pd_proxy_warms_decode_connections_with_non_stream_client(monkeypatch) -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    class FakeResponse:
        status_code = 200
        content = b"ok"
        headers = {"Content-Type": "text/plain"}

    class FakeClient:
        created = 0
        gets: list[str] = []
        posts: list[str] = []

        def __init__(self, **_kwargs) -> None:
            type(self).created += 1

        def get(self, url, **_kwargs):
            type(self).gets.append(url)
            return FakeResponse()

        def post(self, url, **_kwargs):
            type(self).posts.append(url)
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        proxy_mod,
        "_httpx",
        lambda: SimpleNamespace(
            Client=FakeClient,
            Limits=lambda **kwargs: kwargs,
            RequestError=Exception,
        ),
    )
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000/",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    proxy.warmup_decode_connections(connection_count=3)
    status, _payload, _content_type = proxy.handle_openai_request(
        "/v1/completions",
        {"request_id": "req-1", "model": "model", "prompt": "hello"},
    )

    assert status == HTTPStatus.OK
    assert FakeClient.created == 2
    assert FakeClient.gets == ["http://d0:8000/health"] * 6
    assert FakeClient.posts == ["http://d0:8000/v1/completions"]


def test_pd_proxy_warms_decode_connections_with_stream_client(monkeypatch) -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    class FakeResponse:
        status_code = 200
        content = b"ok"
        headers = {"Content-Type": "text/plain"}

    class FakeClient:
        created = 0
        gets: list[str] = []

        def __init__(self, **_kwargs) -> None:
            type(self).created += 1

        def get(self, url, **_kwargs):
            type(self).gets.append(url)
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        proxy_mod,
        "_httpx",
        lambda: SimpleNamespace(
            Client=FakeClient,
            Limits=lambda **kwargs: kwargs,
            RequestError=Exception,
        ),
    )
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    proxy.warmup_decode_connections(connection_count=2)

    assert FakeClient.created == 2
    assert FakeClient.gets == ["http://d0:8000/health"] * 4


def test_pd_proxy_unsupported_stream_does_not_record_decode_duration() -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    status, _content_type, _response, _request_id, start_ts_ns, payload = proxy.open_openai_stream(
        "/unsupported", {"stream": True}
    )
    if payload is not None and start_ts_ns > 0:
        config.metrics.finish_request((time.time_ns() - start_ts_ns) / 1_000_000_000)

    metrics = config.metrics.snapshot()
    assert status == HTTPStatus.NOT_FOUND
    assert metrics["request_count"] == 0
    assert metrics["decode_request_durations_s"] == []


def test_pd_proxy_reuses_stream_http_client(monkeypatch) -> None:
    from pegaflow.pd_connector import proxy as proxy_mod

    class FakeHeaders(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class FakeResponse:
        status_code = 200
        headers = FakeHeaders({"Content-Type": "text/event-stream"})

        def iter_bytes(self):
            yield b"data: {}\n\n"

    class FakeStream:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeClient:
        created = 0
        closed = 0

        def __init__(self, **_kwargs) -> None:
            type(self).created += 1

        def stream(self, *_args, **_kwargs):
            return FakeStream()

        def close(self) -> None:
            type(self).closed += 1

    monkeypatch.setattr(
        proxy_mod,
        "_httpx",
        lambda: SimpleNamespace(
            Client=FakeClient,
            Limits=lambda **kwargs: kwargs,
            RequestError=Exception,
        ),
    )
    config = ProxyConfig(
        prefill_url="http://p0:8000",
        decode_url="http://d0:8000",
        timeout_s=1.0,
        prefill_max_tokens=1,
    )
    proxy = proxy_mod.PdProxy(config)

    first = proxy.open_openai_stream(
        "/v1/completions",
        {"stream": True, "request_id": "req-1", "model": "model", "prompt": "a"},
    )
    second = proxy.open_openai_stream(
        "/v1/completions",
        {"stream": True, "request_id": "req-2", "model": "model", "prompt": "b"},
    )

    assert first[2] is not None
    assert second[2] is not None
    first[2].__exit__(None, None, None)
    second[2].__exit__(None, None, None)
    assert FakeClient.created == 1
    assert FakeClient.closed == 0

    proxy.close()

    assert FakeClient.closed == 1


def test_d_consumer_release_does_not_increment_prefill_release_metric() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "decode-1": WaitReqMeta(
                    local_block_ids=([1],),
                    remote_request_id="prefill-1",
                    done_request_id="decode-1",
                    prompt_token_ids=(1,),
                    prefill_url="http://p:8001",
                )
            }
        ),
        None,
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_release={"decode-1"},
            release_reasons={"decode-1": RELEASE_CONSUMER_ABORT},
        ),
        None,
    )

    stats = worker.get_stats()
    assert stats.data["pd_decode_abort_count"] == 1
    assert stats.data["pd_prefill_release_count"] == 0
    worker.shutdown()


def test_pd_worker_stats_record_decode_wait_completion() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "req-1": WaitReqMeta(
                    local_block_ids=([1, 2],),
                    remote_request_id="req-1-p",
                    done_request_id="req-1-d",
                    prompt_token_ids=(11, 12, 13),
                    prefill_url="http://p:8001",
                    scheduler_wait_ts_ns=time.time_ns(),
                )
            }
        ),
        None,
    )
    rdma._finished_recving.add("req-1")
    worker.get_finished(set())

    stats = worker.get_stats()

    assert stats is not None
    assert stats.data["pd_decode_active_waits"] == 0
    assert stats.data["pd_load_success_count"] == 1
    assert stats.data["pd_load_failure_count"] == 0
    assert stats.data["pd_load_blocks"] == [2]
    assert len(stats.data["pd_decode_wait_duration"]) == 1


def test_d_worker_waits_for_all_prefill_ranks_when_prefill_tp_is_larger() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 8, 32),
        stride=(8 * 8 * 16 * 32, 8 * 16 * 32, 32, 16 * 32, 1),
    )
    native_engine = FakeNativeRdmaEngine()
    worker = PdDecodeWorkerConnector(
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
    worker = PdDecodeWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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


def test_p_worker_closes_single_target_push_once_when_finished() -> None:
    class TrackingCloseRdma(MockRdmaPort):
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
    rdma = TrackingCloseRdma()
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="prefill")),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
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

    worker.save_kv_layer("layer.0", tensor, SimpleNamespace())
    drain_pd_pushes(worker)
    assert worker.get_finished({"req-1"}) == ({"req-1"}, None)
    assert rdma.closed_reqs == ["req-1"]


def test_pd_worker_get_finished_does_not_poll_wait_reqs() -> None:
    rdma = MockRdmaPort()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=rdma
    )
    worker._wait_reqs["req-1"] = WaitReqMeta(
        local_block_ids=([1],),
        remote_request_id="req-1-p",
        done_request_id="req-1-d",
        prompt_token_ids=(1,),
        prefill_url="http://p:8001",
    )

    rdma._finished_recving.add("req-1")

    assert worker.get_finished(set()) == (None, {"req-1"})


def test_p_worker_save_kv_layer_noops_without_push_reqs() -> None:
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="")), rdma=MockRdmaPort()
    )

    worker.save_kv_layer("unknown-layer", object(), None)


def test_p_worker_save_kv_layer_uses_registered_layout_fast_path() -> None:
    class RuntimeTensorThatShouldNotBeInspected:
        pass

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="prefill")),
        rdma=MockRdmaPort(),
    )
    worker.register_kv_caches({"layer.0": tensor})
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

    worker.save_kv_layer("layer.0", RuntimeTensorThatShouldNotBeInspected(), SimpleNamespace())
    drain_pd_pushes(worker)

    assert {layer_idx for layer_idx, _ in worker.rdma.pushed_layers["req-1"]} == {0}


def test_p_worker_runtime_layout_validation_can_be_enabled() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    changed_tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 64, 16 * 32, 1),
    )
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="prefill",
                extra_config={"pegaflow.pd.validate_runtime_layout": True},
            )
        ),
        rdma=MockRdmaPort(),
    )
    worker.register_kv_caches({"layer.0": tensor})
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

    with pytest.raises(AssertionError, match="KV strides changed"):
        worker.save_kv_layer("layer.0", changed_tensor, SimpleNamespace())


def test_d_worker_idle_decode_step_skips_layer_hooks() -> None:
    class LayoutsThatShouldNotBeRead(dict):
        def __contains__(self, key: object) -> bool:
            raise AssertionError("idle decode step should not inspect layouts")

    worker = PdDecodeWorkerConnector(
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


def test_d_worker_release_waits_for_abort_ack_before_finishing() -> None:
    class BlockingWaitRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.wait_started = threading.Event()
            self.wait_can_return = threading.Event()
            self.closed_reqs: list[str] = []

        def wait_done(self, req_id: str) -> None:
            self.wait_started.set()
            assert self.wait_can_return.wait(timeout=5), "wait_done was not released"
            self._finished_recving.add(req_id)

        def close_request(self, req_id: str) -> None:
            self.closed_reqs.append(req_id)
            super().close_request(req_id)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = BlockingWaitRdma()
    worker = PdDecodeWorkerConnector(
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

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_release={"req-1"},
            release_reasons={"req-1": RELEASE_CONSUMER_ABORT},
        ),
        None,
    )

    waiter = worker._decode._rdma_waiter
    assert waiter is not None
    with waiter._lock:
        assert "req-1" in waiter._submitted
    assert "req-1" in worker._decode.wait_reqs
    assert "req-1" in rdma.registered
    assert rdma.closed_reqs == []
    assert worker.get_finished(set()) == (None, None)

    rdma.wait_can_return.set()
    deadline = time.time() + 2
    finished = None
    while time.time() < deadline:
        _, finished = worker.get_finished(set())
        if finished:
            break
        time.sleep(0.01)

    assert finished == {"req-1"}
    assert worker.get_block_ids_with_load_errors() == set()
    assert "req-1" not in worker._decode.wait_reqs
    assert "req-1" not in rdma.registered
    assert rdma.closed_reqs == ["req-1"]


def test_d_worker_release_ack_does_not_record_successful_load() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdDecodeWorkerConnector(
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

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_release={"req-1"},
            release_reasons={"req-1": RELEASE_CONSUMER_ABORT},
        ),
        None,
    )
    rdma._finished_recving.add("req-1")
    worker.get_finished(set())
    stats = worker.get_stats()

    assert stats.data["pd_decode_abort_count"] == 1
    assert stats.data["pd_load_success_count"] == 0
    assert stats.data["pd_load_failure_count"] == 0
    assert stats.data["pd_load_blocks"] == []
    assert stats.data["pd_decode_wait_duration"] == []
    assert stats.data["pd_decode_active_waits"] == 0


def test_d_worker_release_cancels_remote_prefill_request() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=MockRdmaPort(),
        prefill_sender=prefill_sender,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "decode-1": WaitReqMeta(
                    local_block_ids=([1],),
                    remote_request_id="prefill-1",
                    done_request_id="decode-1",
                    prompt_token_ids=(1,),
                    prefill_url="http://p:8001",
                )
            }
        ),
        None,
    )
    assert [task.request_id for task in prefill_sender.tasks] == ["prefill-1"]

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_release={"decode-1"},
            release_reasons={"decode-1": RELEASE_CONSUMER_ABORT},
        ),
        None,
    )

    assert prefill_sender.cancelled == ["prefill-1"]


def test_decode_worker_prefill_sender_worker_count_comes_from_extra_config(monkeypatch) -> None:
    created_worker_counts: list[int] = []

    class FakeAsyncPrefillSender:
        def __init__(self, *, worker_count=16, failure_callback=None) -> None:
            created_worker_counts.append(worker_count)
            self.failure_callback = failure_callback

        def close(self) -> None:
            return None

    monkeypatch.setattr(decode_worker_mod, "AsyncPrefillSender", FakeAsyncPrefillSender)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            get_from_extra_config=lambda key, default=None: {
                "pegaflow.pd.prefill_sender_worker_count": 4,
            }.get(key, default),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
    )

    worker = PdDecodeWorkerConnector(vllm_config, rdma=MockRdmaPort())

    assert created_worker_counts == [4]
    worker.shutdown()


def test_decode_worker_prefill_sender_worker_count_defaults_to_sixteen(monkeypatch) -> None:
    created_worker_counts: list[int] = []

    class FakeAsyncPrefillSender:
        def __init__(self, *, worker_count=16, failure_callback=None) -> None:
            created_worker_counts.append(worker_count)
            self.failure_callback = failure_callback

        def close(self) -> None:
            return None

    monkeypatch.setattr(decode_worker_mod, "AsyncPrefillSender", FakeAsyncPrefillSender)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="decode"),
        parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
    )

    worker = PdDecodeWorkerConnector(vllm_config, rdma=MockRdmaPort())

    assert created_worker_counts == [16]
    worker.shutdown()


def test_prefill_worker_push_worker_counts_default_to_sixteen(monkeypatch) -> None:
    created_push_workers: list[int] = []
    created_finalizer_workers: list[int] = []

    class FakePushSender:
        def __init__(self, *, metrics=None, max_workers=16) -> None:
            created_push_workers.append(max_workers)

        def close(self) -> None:
            return None

        def is_idle(self) -> bool:
            return True

    class FakePushFinalizer:
        def __init__(self, push_sender, *, metrics=None, max_workers=16) -> None:
            created_finalizer_workers.append(max_workers)

        def close(self) -> None:
            return None

        def is_idle(self) -> bool:
            return True

    monkeypatch.setattr(prefill_worker_mod, "_AsyncLayerPushSender", FakePushSender)
    monkeypatch.setattr(prefill_worker_mod, "_AsyncPushFinalizer", FakePushFinalizer)
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="prefill")),
        rdma=MockRdmaPort(),
    )

    assert created_push_workers == [16]
    assert created_finalizer_workers == [16]
    worker.shutdown()


def test_prefill_worker_push_worker_counts_come_from_extra_config(monkeypatch) -> None:
    created_push_workers: list[int] = []
    created_finalizer_workers: list[int] = []

    class FakePushSender:
        def __init__(self, *, metrics=None, max_workers=16) -> None:
            created_push_workers.append(max_workers)

        def close(self) -> None:
            return None

        def is_idle(self) -> bool:
            return True

    class FakePushFinalizer:
        def __init__(self, push_sender, *, metrics=None, max_workers=16) -> None:
            created_finalizer_workers.append(max_workers)

        def close(self) -> None:
            return None

        def is_idle(self) -> bool:
            return True

    monkeypatch.setattr(prefill_worker_mod, "_AsyncLayerPushSender", FakePushSender)
    monkeypatch.setattr(prefill_worker_mod, "_AsyncPushFinalizer", FakePushFinalizer)
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                engine_id="prefill",
                get_from_extra_config=lambda key, default=None: {
                    "pegaflow.pd.push_worker_count": 7,
                    "pegaflow.pd.push_finalizer_worker_count": 9,
                }.get(key, default),
            )
        ),
        rdma=MockRdmaPort(),
    )

    assert created_push_workers == [7]
    assert created_finalizer_workers == [9]
    worker.shutdown()


def test_d_worker_prefill_failure_reports_load_error(monkeypatch) -> None:
    class BlockingWaitRdma(MockRdmaPort):
        def wait_done(self, req_id: str) -> None:
            time.sleep(10)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )

    async def fail_prefill_request(task: PrefillHttpTask, _client=None) -> None:
        raise prefill_mod.PrefillRequestError(task.request_id, "p aborted")

    monkeypatch.setattr(
        prefill_mod,
        "post_prefill_request_async",
        fail_prefill_request,
    )
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=BlockingWaitRdma(),
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "decode-1": WaitReqMeta(
                    local_block_ids=([3, 4],),
                    remote_request_id="prefill-1",
                    done_request_id="decode-1",
                    prompt_token_ids=(1,),
                    prefill_url="http://p:8001",
                )
            }
        ),
        None,
    )

    deadline = time.time() + 2
    finished_recving = None
    while time.time() < deadline:
        _, finished_recving = worker.get_finished(set())
        if finished_recving:
            break
        time.sleep(0.01)

    assert finished_recving == {"decode-1"}
    assert worker.get_block_ids_with_load_errors() == {3, 4}
    assert worker.get_block_ids_with_load_errors() == set()
    worker.shutdown()


def test_d_worker_rdma_wait_failure_reports_load_error() -> None:
    class FailingWaitRdma(MockRdmaPort):
        def wait_done(self, req_id: str) -> None:
            raise RuntimeError("rdma timeout")

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")),
        rdma=FailingWaitRdma(),
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "decode-1": WaitReqMeta(
                    local_block_ids=([5],),
                    remote_request_id="prefill-1",
                    done_request_id="decode-1",
                    prompt_token_ids=(1,),
                    prefill_url="",
                )
            }
        ),
        None,
    )

    deadline = time.time() + 2
    finished_recving = None
    while time.time() < deadline:
        _, finished_recving = worker.get_finished(set())
        if finished_recving:
            break
        time.sleep(0.01)

    assert finished_recving == {"decode-1"}
    assert worker.get_block_ids_with_load_errors() == {5}
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.failed_recving == {"decode-1"}
    worker.shutdown()


def test_d_worker_reports_background_rdma_wait_completion_without_native_poll() -> None:
    class CallbackOnlyRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.can_return = threading.Event()

        def wait_done(self, req_id: str) -> None:
            self.started.set()
            assert self.can_return.wait(timeout=5), f"wait for {req_id} was not released"

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = CallbackOnlyRdma()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "decode-1": WaitReqMeta(
                    local_block_ids=([5],),
                    remote_request_id="prefill-1",
                    done_request_id="decode-1",
                    prompt_token_ids=(1,),
                    prefill_url="",
                )
            }
        ),
        None,
    )
    assert rdma.started.wait(timeout=5), "RDMA waiter did not start"
    rdma.can_return.set()

    deadline = time.time() + 2
    finished_recving = None
    while time.time() < deadline:
        _, finished_recving = worker.get_finished(set())
        if finished_recving:
            break
        time.sleep(0.01)

    assert finished_recving == {"decode-1"}
    assert rdma.pop_finished_recving() == set()
    worker.shutdown()


def test_d_worker_finished_rdma_wait_prevents_idle_fast_path() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")),
        rdma=MockRdmaPort(),
    )
    worker.register_kv_caches({"layer.0": tensor})
    worker._decode._finished_rdma_waits.add("decode-1")

    _, finished_recving = worker.get_finished(set())

    assert finished_recving == {"decode-1"}
    worker.shutdown()


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
                self._finished_recving.add(req_id)
                return
            if wait_call == 2:
                self.second_started.set()
                assert self.second_can_return.wait(timeout=5), "second wait was not released"
                self._finished_recving.add(req_id)
                return
            raise AssertionError(f"unexpected wait call {wait_call} for {req_id}")

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = SequencedWaitRdma()
    worker = PdDecodeWorkerConnector(
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
    assert not rdma.second_started.wait(timeout=0.05)
    rdma.first_can_return.set()

    deadline = time.time() + 2
    finished = None
    while time.time() < deadline:
        _, finished = worker.get_finished(set())
        if finished:
            break
        time.sleep(0.01)
    assert finished == {"req-1"}

    worker.start_load_kv(wait_meta, None)
    assert rdma.second_started.wait(timeout=5), "second wait did not start"

    waiter = worker._decode._rdma_waiter
    assert waiter is not None
    with waiter._lock:
        assert "req-1" in waiter._submitted

    rdma.second_can_return.set()


def test_d_worker_starts_multiple_rdma_waits_concurrently() -> None:
    class BlockingWaitRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.started = {f"req-{idx}": threading.Event() for idx in (1, 2)}
            self.can_return = threading.Event()

        def wait_done(self, req_id: str) -> None:
            self.started[req_id].set()
            assert self.can_return.wait(timeout=5), f"wait for {req_id} was not released"
            self._finished_recving.add(req_id)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = BlockingWaitRdma()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="decode")), rdma=rdma
    )
    worker.register_kv_caches({"layer.0": tensor})

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                f"req-{idx}": WaitReqMeta(
                    local_block_ids=([idx],),
                    remote_request_id=f"req-{idx}-p",
                    done_request_id=f"req-{idx}-d",
                    prompt_token_ids=(idx,),
                    prefill_url="",
                )
                for idx in (1, 2)
            }
        ),
        None,
    )

    assert rdma.started["req-1"].wait(timeout=5)
    assert rdma.started["req-2"].wait(timeout=0.2)
    rdma.can_return.set()


def test_p_worker_release_closes_all_physical_decode_targets() -> None:
    class TrackingRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.closed_reqs: list[str] = []
            self.failed_reqs: list[str] = []
            self.aborted_reqs: list[str] = []
            self.drained_reqs: list[str] = []

        def fail_request(self, req_id: str) -> None:
            self.failed_reqs.append(req_id)

        def abort_request(self, req_id: str) -> None:
            self.aborted_reqs.append(req_id)

        def wait_for_pushes(self, req_id: str) -> None:
            self.drained_reqs.append(req_id)

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
    worker = PdPrefillWorkerConnector(
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

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_release={"prefill-r1"},
            release_reasons={"prefill-r1": RELEASE_CONSUMER_ABORT},
        ),
        None,
    )

    assert rdma.failed_reqs == []
    assert sorted(rdma.drained_reqs) == ["prefill-r1#d2", "prefill-r1#d3"]
    assert sorted(rdma.aborted_reqs) == ["prefill-r1#d2", "prefill-r1#d3"]
    assert sorted(rdma.closed_reqs) == ["prefill-r1#d2", "prefill-r1#d3"]
    assert rdma.registered == set()


def test_p_worker_completion_clears_physical_remote_block_offsets() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdPrefillWorkerConnector(
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
    worker._prefill._remote_block_offsets.update(
        {
            "prefill-r1#d2#l0": 1,
            "prefill-r1#d3#l0": 1,
            "prefill-r10#d2#l0": 1,
            "other#d0#l0": 1,
        }
    )

    worker._prefill._completed_pushes.add("prefill-r1")
    assert worker.get_finished({"prefill-r1"})[0] == {"prefill-r1"}

    assert worker._prefill._remote_block_offsets == {
        "prefill-r10#d2#l0": 1,
        "other#d0#l0": 1,
    }


def test_p_worker_preemption_cancels_push_without_waiting_for_done() -> None:
    class TrackingRdma(MockRdmaPort):
        def __init__(self) -> None:
            super().__init__()
            self.closed_reqs: list[str] = []
            self.failed_reqs: list[str] = []
            self.drained_reqs: list[str] = []

        def fail_request(self, req_id: str) -> None:
            self.failed_reqs.append(req_id)

        def wait_for_pushes(self, req_id: str) -> None:
            self.drained_reqs.append(req_id)

        def close_request(self, req_id: str) -> None:
            self.closed_reqs.append(req_id)
            super().close_request(req_id)

    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = TrackingRdma()
    worker = PdPrefillWorkerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="prefill")),
        rdma=rdma,
    )
    worker.register_kv_caches({"layer.0": tensor, "layer.1": tensor})
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-1": PushReqMeta(
                    local_block_ids=([1],),
                    target_request_id="decode-1",
                    handshakes=(DUMMY_HANDSHAKE,),
                )
            }
        ),
        None,
    )
    assert rdma.registered == {"prefill-1"}

    worker.start_load_kv(PdConnectorMetadata(preempted_req_ids={"prefill-1"}), None)

    assert rdma.failed_reqs == ["prefill-1"]
    assert rdma.drained_reqs == ["prefill-1"]
    assert rdma.closed_reqs == ["prefill-1"]
    assert rdma.registered == set()
    assert worker.get_finished({"prefill-1"}) == (None, None)


def test_p_worker_uses_scheduler_blocks_without_slot_mapping_cpu_sync() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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
    d_worker = PdDecodeWorkerConnector(
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
    assert len(wait_handshake.layers) == 2
    assert wait_handshake.layers[0].block_ids == (1,)
    assert wait_handshake.layers[1].block_ids == (1,)

    assert len(prefill_sender.tasks) == 1
    task = prefill_sender.tasks[0]
    assert task.kv_transfer_params["pd_consumer_abort_returns_ack"] is True
    handshake = handshakes_from_dicts(task.kv_transfer_params["pd_handshakes"])[0]
    assert handshake.engine_id == "decode"
    assert handshake.block_size == 16
    assert handshake.imm_id == wait_handshake.imm_id
    assert handshake.layers[0].block_ids == (1, 2)
    assert handshake.layers[1].block_ids == (1, 2)
    assert handshake.layers[0].regions[0] == TransferRegionLayout(
        region_idx=0,
        base_addr=tensor.data_ptr(),
        block_len=4 * 16 * 32 * 2,
    )

    push_worker = PdPrefillWorkerConnector(
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


def test_d_worker_wait_handshake_uses_layer_kv_cache_group_blocks_for_mtp() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdDecodeWorkerConnector(
        fake_mtp_config(),
        kv_cache_config=fake_mtp_kv_cache_config(),
        rdma=MockRdmaPort(),
        prefill_sender=prefill_sender,
    )
    base_layer = "model.layers.0.self_attn"
    mtp_layer = "model.layers.27.self_attn"
    worker.register_kv_caches({base_layer: tensor, mtp_layer: tensor})

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_wait={
                "req-1": WaitReqMeta(
                    local_block_ids=([1, 2], [7, 8]),
                    remote_request_id="req-1",
                    done_request_id="req-1",
                    prompt_token_ids=(101, 102, 103),
                    prefill_url="http://p:8001",
                )
            }
        ),
        None,
    )

    wait_handshake = worker.rdma.remote_handshakes["req-1"]
    assert wait_handshake.layers[0].layer_name == base_layer
    assert wait_handshake.layers[0].block_ids == (1,)
    assert wait_handshake.layers[1].layer_name == mtp_layer
    assert wait_handshake.layers[1].block_ids == (7,)

    task_handshake = handshakes_from_dicts(
        prefill_sender.tasks[0].kv_transfer_params["pd_handshakes"]
    )[0]
    assert [(layer.layer_name, layer.block_ids) for layer in task_handshake.layers] == [
        (base_layer, (1, 2)),
        (mtp_layer, (7, 8)),
    ]


def test_p_worker_pushes_mtp_layers_from_matching_kv_cache_group_blocks() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    base_layer = "model.layers.0.self_attn"
    mtp_layer = "model.layers.27.self_attn"
    rdma = MockRdmaPort()
    worker = PdPrefillWorkerConnector(
        fake_mtp_config(),
        kv_cache_config=fake_mtp_kv_cache_config(num_blocks=16),
        rdma=rdma,
    )
    worker.register_kv_caches({base_layer: tensor, mtp_layer: tensor})
    handshake = PdHandshake(
        request_id="decode",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        layers=(
            hnd_remote_layer(
                layer_name=base_layer,
                layer_idx=0,
                block_ids=(101, 102),
                block_len=4096,
            ),
            hnd_remote_layer(
                layer_name=mtp_layer,
                layer_idx=1,
                block_ids=(207, 208),
                block_len=4096,
            ),
        ),
    )
    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=([1, 2], [7, 8]),
                    target_request_id="decode",
                    handshakes=(handshake,),
                )
            }
        ),
        None,
    )

    worker.save_kv_layer(base_layer, tensor, SimpleNamespace())
    worker.save_kv_layer(mtp_layer, tensor, SimpleNamespace())
    worker.wait_for_save()
    drain_pd_pushes(worker)

    pushed_by_layer = pushed_layers_by_idx(rdma, "prefill-r0")
    assert [block.regions[0].block_id for block in pushed_by_layer[0]] == [101]
    assert [block.regions[0].src_offset_bytes for block in pushed_by_layer[0]] == [
        tensor.stride()[1] * 1 * tensor.element_size()
    ]
    assert [block.regions[0].block_id for block in pushed_by_layer[1]] == [207]
    assert [block.regions[0].src_offset_bytes for block in pushed_by_layer[1]] == [
        tensor.stride()[1] * 7 * tensor.element_size()
    ]


def test_scheduler_delays_producer_block_free_until_send_finishes() -> None:
    scheduler = PdPrefillSchedulerConnector(
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
    assert release_meta.release_reasons == {"req-1": RELEASE_PRODUCER_FINISHED}


def test_scheduler_does_not_delay_aborted_producer_block_free() -> None:
    scheduler = PdPrefillSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        status="FINISHED_ABORTED",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "req-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    scheduler.build_connector_meta(SimpleNamespace())

    delay_free, params = scheduler.request_finished(request, ([1, 2],))
    release_meta = scheduler.build_connector_meta(SimpleNamespace())

    assert delay_free is False
    assert params is None
    assert release_meta.reqs_to_release == {"req-1"}
    assert release_meta.release_reasons == {"req-1": RELEASE_PRODUCER_ABORT}


def test_scheduler_marks_remote_prefill_abort_as_consumer_abort_ack() -> None:
    scheduler = PdPrefillSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="p"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        status="FINISHED_ABORTED",
        kv_transfer_params={
            "do_remote_prefill_sender": True,
            "target_engine_id": "decode",
            "target_request_id": "req-1",
            "pd_consumer_abort_returns_ack": True,
        },
    )

    scheduler.update_state_after_alloc(request, ([1, 2],), num_external_tokens=0)
    scheduler.build_connector_meta(SimpleNamespace())

    delay_free, params = scheduler.request_finished(request, ([1, 2],))
    release_meta = scheduler.build_connector_meta(SimpleNamespace())

    assert delay_free is False
    assert params is None
    assert release_meta.reqs_to_release == {"req-1"}
    assert release_meta.release_reasons == {"req-1": RELEASE_CONSUMER_ABORT}


def test_scheduler_marks_preempted_producer_for_worker_release() -> None:
    scheduler = PdPrefillSchedulerConnector(
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
    meta = scheduler.build_connector_meta(SimpleNamespace(preempted_req_ids={"req-1"}))

    assert meta.preempted_req_ids == {"req-1"}
    assert meta.release_reasons == {"req-1": RELEASE_PRODUCER_PREEMPTED}
    assert "req-1" not in meta.reqs_to_release


def test_scheduler_marks_consumer_abort_release_reason() -> None:
    scheduler = PdDecodeSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="req-1",
        status="FINISHED_ABORTED",
        kv_transfer_params={"do_remote_prefill": True, "prefill_url": "http://p:8001"},
    )

    scheduler.request_finished(request, ([1],))
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.reqs_to_release == {"req-1"}
    assert meta.release_reasons == {"req-1": RELEASE_CONSUMER_ABORT}


def test_scheduler_emits_cached_producer_chunks() -> None:
    scheduler = PdPrefillSchedulerConnector(
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
    scheduler = PdDecodeSchedulerConnector(
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
    scheduler = PdDecodeSchedulerConnector(
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
    scheduler = PdPrefillSchedulerConnector(
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


def test_scheduler_carries_cross_process_rdma_handshake_list() -> None:
    handshake = {
        "request_id": "decode-1",
        "engine_id": "decode",
        "tp_rank": 0,
        "tp_size": 1,
        "block_size": 16,
        "block_ids": [1],
        "layers": [
            {
                "layer_name": "layer.0",
                "layer_idx": 0,
                "regions": [
                    {"region_idx": 0, "base_addr": 0x1000, "block_len": 1024},
                    {"region_idx": 1, "base_addr": 0x1400, "block_len": 1024},
                ],
                "mr_desc": {"addr_rkey_list": [["10.0.0.1:1", 17]]},
            }
        ],
    }
    scheduler = PdPrefillSchedulerConnector(
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
    assert parsed.layers[0].block_ids == (1,)
    assert parsed.layers[0].mr_desc == {"addr_rkey_list": [["10.0.0.1:1", 17]]}


def test_scheduler_ignores_legacy_fake_rdma_done_endpoint() -> None:
    scheduler = PdDecodeSchedulerConnector(
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
    scheduler = PdDecodeSchedulerConnector(
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


def test_scheduler_failed_recv_allows_remote_wait_retry() -> None:
    scheduler = PdDecodeSchedulerConnector(
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

    scheduler.update_connector_output(
        SimpleNamespace(
            finished_sending=None,
            finished_recving={"req-1"},
            kv_connector_worker_meta=PdWorkerMetadata(failed_recving={"req-1"}),
        )
    )
    scheduler.update_state_after_alloc(request, ([2],), num_external_tokens=3)
    retry = scheduler.build_connector_meta(SimpleNamespace())

    assert set(retry.reqs_to_wait) == {"req-1"}
    assert retry.reqs_to_wait["req-1"].local_block_ids == ([2],)


def test_d_failed_load_retry_dispatches_prefill_again() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdDecodeWorkerConnector(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="decode"),
            parallel_config=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=1),
        ),
        rdma=MockRdmaPort(),
        prefill_sender=prefill_sender,
    )
    worker.register_kv_caches({"layer.0": tensor})
    scheduler = PdDecodeSchedulerConnector(
        SimpleNamespace(kv_transfer_config=SimpleNamespace(engine_id="d"))
    )
    request = SimpleNamespace(
        request_id="decode-1",
        prompt_token_ids=[11, 12, 13],
        kv_transfer_params={
            "do_remote_prefill": True,
            "prefill_url": "http://p:8001",
            "remote_request_id": "prefill-1",
            "done_request_id": "decode-1",
        },
    )

    scheduler.update_state_after_alloc(request, ([1],), num_external_tokens=3)
    worker.start_load_kv(scheduler.build_connector_meta(SimpleNamespace()), None)
    assert [task.request_id for task in prefill_sender.tasks] == ["prefill-1"]

    worker._decode._mark_wait_failed("decode-1", RuntimeError("p preempted"))
    _, finished_recving = worker.get_finished(set())
    assert finished_recving == {"decode-1"}
    failed_blocks = worker.get_block_ids_with_load_errors()
    assert failed_blocks == {1}
    scheduler.update_connector_output(
        SimpleNamespace(
            finished_sending=None,
            finished_recving=finished_recving,
            invalid_block_ids=failed_blocks,
            kv_connector_worker_meta=worker.build_connector_worker_meta(),
        )
    )

    scheduler.update_state_after_alloc(request, ([2],), num_external_tokens=3)
    worker.start_load_kv(scheduler.build_connector_meta(SimpleNamespace()), None)

    assert [task.request_id for task in prefill_sender.tasks] == ["prefill-1", "prefill-1"]
    worker.shutdown()


def test_d_worker_rank0_dispatches_prefill_on_wait() -> None:
    tensor = FakeTensor(
        shape=(2, 8, 16, 4, 32),
        stride=(8 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    prefill_sender = FakePrefillSender()
    worker = PdDecodeWorkerConnector(
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
    handshakes = handshakes_from_dicts(task.kv_transfer_params["pd_handshakes"])
    assert len(handshakes) == 1
    assert handshakes[0].request_id == "external-d"
    assert handshakes[0].layers[0].block_ids == (1,)


def test_layer_push_sender_runs_requests_concurrently() -> None:
    class Event:
        def __init__(self) -> None:
            self.ready = threading.Event()

        def synchronize(self) -> None:
            assert self.ready.wait(timeout=2)

    class BlockingRdma:
        def __init__(self) -> None:
            self.entered: queue.Queue[str] = queue.Queue()
            self.release = threading.Event()

        def push_layer(self, req_id, layer_idx, blocks) -> None:
            self.entered.put(req_id)
            assert self.release.wait(timeout=2)

    ready_event = Event()
    ready_event.ready.set()
    rdma = BlockingRdma()
    sender = prefill_worker_mod._AsyncLayerPushSender()
    try:
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="req-1",
                layer_idx=0,
                block_slices=[],
                event=ready_event,
            )
        )
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="req-2",
                layer_idx=1,
                block_slices=[],
                event=ready_event,
            )
        )

        entered = {rdma.entered.get(timeout=2), rdma.entered.get(timeout=2)}
        assert entered == {"req-1", "req-2"}

        rdma.release.set()
        sender.wait_all()
    finally:
        rdma.release.set()
        sender.close()


def test_layer_push_sender_cancel_skips_queued_req() -> None:
    class Event:
        def __init__(self) -> None:
            self.ready = threading.Event()

        def synchronize(self) -> None:
            assert self.ready.wait(timeout=2)

    class RecordingRdma:
        def __init__(self) -> None:
            self.pushed: queue.Queue[str] = queue.Queue()

        def push_layer(self, req_id, layer_idx, blocks) -> None:
            self.pushed.put(req_id)

    hold_event = Event()
    ready_event = Event()
    ready_event.ready.set()
    rdma = RecordingRdma()
    sender = prefill_worker_mod._AsyncLayerPushSender(max_workers=1)
    try:
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="hold",
                layer_idx=0,
                block_slices=[],
                event=hold_event,
            )
        )
        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="cancelled",
                layer_idx=0,
                block_slices=[],
                event=ready_event,
            )
        )

        sender.cancel("cancelled")
        hold_event.ready.set()
        sender.wait_all()

        assert rdma.pushed.get(timeout=2) == "hold"
        with pytest.raises(queue.Empty):
            rdma.pushed.get(timeout=0.1)

        sender.submit(
            prefill_worker_mod._LayerPushTask(
                rdma=rdma,
                req_id="cancelled",
                layer_idx=1,
                block_slices=[],
                event=ready_event,
            )
        )
        sender.wait_req("cancelled")
        assert rdma.pushed.get(timeout=2) == "cancelled"
    finally:
        hold_event.ready.set()
        sender.close()


def test_push_finalizer_runs_requests_concurrently() -> None:
    class Sender:
        def wait_req(self, req_id: str) -> None:
            return None

    class BlockingRdma:
        def __init__(self) -> None:
            self.entered: queue.Queue[str] = queue.Queue()
            self.release = threading.Event()
            self.done: list[str] = []

        def wait_for_pushes(self, req_id: str) -> None:
            self.entered.put(req_id)
            assert self.release.wait(timeout=2)

        def push_done(self, req_id: str) -> None:
            self.done.append(req_id)

        def aggregated_link_speed(self) -> int:
            return 400_000_000_000

    rdma = BlockingRdma()
    finalizer = prefill_worker_mod._AsyncPushFinalizer(Sender())
    try:
        for req_id in ("req-1", "req-2"):
            finalizer.submit(
                prefill_worker_mod._PushFinalizeTask(
                    rdma=rdma,
                    req_ids=(req_id,),
                    target_request_id=req_id,
                    num_blocks=1,
                    chunk_count=1,
                    first_save_ts_ns=time.time_ns(),
                    finalize_queued_ts_ns=time.time_ns(),
                    schedule_queued_ts_ns=time.time_ns(),
                    rdma_bytes=1,
                )
            )

        entered = {rdma.entered.get(timeout=2), rdma.entered.get(timeout=2)}
        assert entered == {"req-1", "req-2"}

        rdma.release.set()
        finalizer.wait_all()
        assert sorted(rdma.done) == ["req-1", "req-2"]
    finally:
        rdma.release.set()
        finalizer.close()


def test_push_finalizer_records_schedule_to_done_duration() -> None:
    class Sender:
        def wait_req(self, req_id: str) -> None:
            return None

    class RecordingRdma:
        def wait_for_pushes(self, req_id: str) -> None:
            return None

        def push_done(self, req_id: str) -> None:
            return None

        def aggregated_link_speed(self) -> int:
            return 400_000_000_000

    from pegaflow.pd_connector.metrics import PdMetricsTracker

    metrics = PdMetricsTracker()
    now_ns = time.time_ns()
    finalizer = prefill_worker_mod._AsyncPushFinalizer(Sender(), metrics=metrics)
    try:
        finalizer.submit(
            prefill_worker_mod._PushFinalizeTask(
                rdma=RecordingRdma(),
                req_ids=("req-1",),
                target_request_id="req-1",
                num_blocks=1,
                chunk_count=1,
                first_save_ts_ns=now_ns - 800_000_000,
                finalize_queued_ts_ns=now_ns - 100_000_000,
                schedule_queued_ts_ns=now_ns - 1_000_000_000,
                rdma_bytes=1,
            )
        )
        finalizer.wait_all()
    finally:
        finalizer.close()

    stats = metrics.get_stats()
    assert stats.data["pd_prefill_push_duration"][0] >= 0.8
    assert stats.data["pd_prefill_push_gbps"][0] > 0


def _prefill_http_task(request_id: str) -> PrefillHttpTask:
    return PrefillHttpTask(
        request_id=request_id,
        prefill_url="http://p:8001",
        model="model",
        prompt_token_ids=(1,),
        max_tokens=1,
        kv_transfer_params={"target_request_id": request_id},
    )


def _fake_httpx_module(created_clients: list[object] | None = None):
    class Timeout:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class Limits:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class AsyncClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            if created_clients is not None:
                created_clients.append(self)

        async def aclose(self) -> None:
            self.closed = True

    return SimpleNamespace(Timeout=Timeout, Limits=Limits, AsyncClient=AsyncClient)


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
    monkeypatch.setattr(prefill_mod, "_httpx", lambda: _fake_httpx_module())
    sender = AsyncPrefillSender(worker_count=2)
    try:
        sender.submit(_prefill_http_task("req-1"))
        sender.submit(_prefill_http_task("req-2"))

        assert {entered.get(timeout=2), entered.get(timeout=2)} == {"req-1", "req-2"}
    finally:
        release.set()
        sender.close()


def test_async_prefill_sender_reuses_lazy_http_client(monkeypatch) -> None:
    entered: queue.Queue[str] = queue.Queue()
    clients: queue.Queue[object] = queue.Queue()
    created_clients: list[object] = []
    fake_httpx = _fake_httpx_module(created_clients)

    async def fake_post_prefill_request(task: PrefillHttpTask, client=None) -> None:
        entered.put(task.request_id)
        clients.put(client)

    monkeypatch.setattr(prefill_mod, "_httpx", lambda: fake_httpx)
    monkeypatch.setattr(
        prefill_mod,
        "post_prefill_request_async",
        fake_post_prefill_request,
    )
    sender = AsyncPrefillSender(worker_count=1)
    try:
        sender.submit(_prefill_http_task("req-1"))
        sender.submit(_prefill_http_task("req-2"))

        assert [entered.get(timeout=2), entered.get(timeout=2)] == ["req-1", "req-2"]
        first_client = clients.get(timeout=2)
        second_client = clients.get(timeout=2)
        assert first_client is not None
        assert first_client is second_client
        assert created_clients == [first_client]
    finally:
        sender.close()


def test_async_prefill_sender_cancel_cancels_running_request(monkeypatch) -> None:
    entered: queue.Queue[str] = queue.Queue()
    cancelled: queue.Queue[str] = queue.Queue()

    async def fake_post_prefill_request(task: PrefillHttpTask, _client=None) -> None:
        entered.put(task.request_id)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.put(task.request_id)
            raise

    monkeypatch.setattr(
        prefill_mod,
        "post_prefill_request_async",
        fake_post_prefill_request,
    )
    monkeypatch.setattr(prefill_mod, "_httpx", lambda: _fake_httpx_module())
    sender = AsyncPrefillSender(worker_count=1)
    try:
        sender.submit(_prefill_http_task("req-1"))
        assert entered.get(timeout=2) == "req-1"

        sender.cancel("req-1")

        assert cancelled.get(timeout=2) == "req-1"
    finally:
        sender.close()


def test_async_prefill_sender_reports_startup_failure(monkeypatch) -> None:
    def fail_new_event_loop():
        raise RuntimeError("loop init failed")

    monkeypatch.setattr(prefill_mod.asyncio, "new_event_loop", fail_new_event_loop)

    with pytest.raises(RuntimeError, match="failed to start"):
        AsyncPrefillSender(startup_timeout_s=0.1)


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
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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
    worker = PdPrefillWorkerConnector(
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


def test_p_worker_precomputes_layer_push_plan_before_save() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdPrefillWorkerConnector(
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

    prepared = worker._prefill._push_layer_plans["prefill-r0"][0]
    assert prepared.req_blocks == frozenset({3, 4})
    assert prepared.pushed_req_blocks == frozenset({3, 4})
    assert prepared.rdma_bytes == tensor.stride()[1] * tensor.element_size() * 2 * 2
    assert prepared.all_chunks_seen is True
    assert len(prepared.target_pushes) == 1
    assert len(prepared.target_pushes[0].block_slices) == 1
    assert prepared.target_pushes[0].block_slices[0].regions[0].block_id == 68

    worker.save_kv_layer("layer.0", object(), SimpleNamespace())
    drain_pd_pushes(worker)

    _, pushed = rdma.pushed_layers["prefill-r0"][0]
    assert [block.regions[0].block_id for block in pushed] == [68]


def test_p_worker_advances_remote_blocks_across_chunk_prefill() -> None:
    tensor = FakeTensor(
        shape=(2, 16, 16, 4, 32),
        stride=(16 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdPrefillWorkerConnector(
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


def test_p_worker_trims_extra_prefill_blocks_beyond_decode_handshake() -> None:
    tensor = FakeTensor(
        shape=(2, 32, 16, 4, 32),
        stride=(32 * 4 * 16 * 32, 4 * 16 * 32, 32, 16 * 32, 1),
    )
    rdma = MockRdmaPort()
    worker = PdPrefillWorkerConnector(
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
                block_ids=tuple(range(68, 96)),
                block_len=4096,
            ),
        ),
    )

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=(list(range(3, 11)),),
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
        SimpleNamespace(slot_mapping=FakeSlotMapping([block * 16 for block in range(3, 11)])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    assert worker.get_finished(set())[0] is None

    worker.start_load_kv(
        PdConnectorMetadata(
            reqs_to_push={
                "prefill-r0": PushReqMeta(
                    local_block_ids=(list(range(11, 32)),),
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
        SimpleNamespace(slot_mapping=FakeSlotMapping([block * 16 for block in range(11, 32)])),
    )
    worker.wait_for_save()
    drain_pd_pushes(worker)

    second_push = rdma.pushed_layers["prefill-r0"][1][1]
    assert [block.regions[0].block_id for block in second_push] == [76]
    assert second_push[0].regions[0].bytes == tensor.stride()[1] * tensor.element_size() * 20
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


def test_pd_proxy_round_robins_prefill_decode_pairs() -> None:
    router = RoundRobinPairRouter(
        prefill_endpoints=(
            PdEndpoint(url="http://127.0.0.1:8101", instance_id="p0"),
            PdEndpoint(url="http://127.0.0.1:8102", instance_id="p1"),
        ),
        decode_endpoints=(
            PdEndpoint(url="http://127.0.0.1:8201", instance_id="d0"),
            PdEndpoint(url="http://127.0.0.1:8202", instance_id="d1"),
        ),
    )
    config = ProxyConfig(
        prefill_url="http://unused-prefill",
        decode_url="http://unused-decode",
        timeout_s=30,
        prefill_max_tokens=1,
        router=router,
    )

    first = build_pd_proxy_request({"model": "m", "prompt": "a"}, config, request_id="r0")
    second = build_pd_proxy_request({"model": "m", "prompt": "b"}, config, request_id="r1")
    third = build_pd_proxy_request({"model": "m", "prompt": "c"}, config, request_id="r2")

    assert first.decode_url == "http://127.0.0.1:8201"
    assert first.decode_body["kv_transfer_params"]["prefill_url"] == "http://127.0.0.1:8101"
    assert second.decode_url == "http://127.0.0.1:8202"
    assert second.decode_body["kv_transfer_params"]["prefill_url"] == "http://127.0.0.1:8102"
    assert third.decode_url == "http://127.0.0.1:8201"
    assert third.decode_body["kv_transfer_params"]["prefill_url"] == "http://127.0.0.1:8101"


def test_pd_proxy_round_robin_pairs_all_prefill_decode_combinations() -> None:
    router = RoundRobinPairRouter(
        prefill_endpoints=(
            PdEndpoint(url="http://p0", instance_id="p0"),
            PdEndpoint(url="http://p1", instance_id="p1"),
        ),
        decode_endpoints=(
            PdEndpoint(url="http://d0", instance_id="d0"),
            PdEndpoint(url="http://d1", instance_id="d1"),
            PdEndpoint(url="http://d2", instance_id="d2"),
        ),
    )
    selected = [router.select().as_tuple() for _ in range(7)]

    assert selected == [
        ("http://p0", "http://d0"),
        ("http://p1", "http://d1"),
        ("http://p0", "http://d2"),
        ("http://p1", "http://d0"),
        ("http://p0", "http://d1"),
        ("http://p1", "http://d2"),
        ("http://p0", "http://d0"),
    ]


def test_pd_proxy_build_router_preserves_policy_abstraction() -> None:
    router = build_router(
        prefill_urls=("http://p0/", "http://p1/"),
        decode_urls=("http://d0/",),
        routing_policy="round_robin",
    )

    first = router.select()
    second = router.select()

    assert first.prefill.url == "http://p0"
    assert first.decode.url == "http://d0"
    assert second.prefill.url == "http://p1"
    assert second.decode.url == "http://d0"


def test_pd_proxy_build_request_fallback_normalizes_endpoint_urls() -> None:
    config = ProxyConfig(
        prefill_url="http://p0/",
        decode_url="http://d0/",
        timeout_s=30,
        prefill_max_tokens=1,
        router=None,
    )

    req = build_pd_proxy_request({"model": "m", "prompt": "a"}, config, request_id="r0")

    assert req.decode_url == "http://d0"
    assert req.decode_body["kv_transfer_params"]["prefill_url"] == "http://p0"


def test_pd_proxy_streams_decode_bytes_without_sse_event_aggregation() -> None:
    class SplitSseBody:
        def __init__(self) -> None:
            self.chunks = iter(
                [
                    b'data: {"choices"',
                    b':[{"text":"a"}]}\n\n',
                    b"data: [DONE]\n",
                    b"\n",
                ]
            )

        def read1(self, _size=-1):
            return next(self.chunks, b"")

        def read(self, _size=-1):
            raise AssertionError("stream forwarding must not use large buffered reads")

    assert list(iter_http_stream_bytes(SplitSseBody())) == [
        b'data: {"choices"',
        b':[{"text":"a"}]}\n\n',
        b"data: [DONE]\n",
        b"\n",
    ]


def test_pd_proxy_prefers_httpx_stream_chunks_over_fixed_size_reads() -> None:
    class HttpxLikeBody:
        def iter_bytes(self):
            yield b'data: {"text":"'
            yield b"\xe6\xb5\x8b"
            yield b'"}\n\n'

        def read1(self, _size=-1):
            raise AssertionError("httpx-style stream should not be re-chunked with read1")

        def read(self, _size=-1):
            raise AssertionError("httpx-style stream should not be re-chunked with read")

    assert list(iter_http_stream_bytes(HttpxLikeBody())) == [
        b'data: {"text":"',
        b"\xe6\xb5\x8b",
        b'"}\n\n',
    ]


def test_pd_decode_connector_allows_full_cudagraph_by_default() -> None:
    assert PdDecodeConnector.requires_piecewise_for_cudagraph({}) is False


def test_pd_prefill_connector_requires_piecewise_cudagraph_for_layerwise_push() -> None:
    assert PdPrefillConnector.requires_piecewise_for_cudagraph({}) is True


def test_pd_connector_can_allow_full_decode_cudagraph() -> None:
    assert (
        PdDecodeConnector.requires_piecewise_for_cudagraph(
            {"pegaflow.pd.allow_full_decode_cudagraph": True}
        )
        is False
    )
