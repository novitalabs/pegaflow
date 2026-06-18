"""Integration repro: MLA replica KV registration across multiple devices.

GLM-5.1-FP8 (``glm_moe_dsa``: MLA + DSA sparse indexer + 1 MTP layer) under
TP > 1. MLA KV data is identical on every TP rank, so the connector registers
every worker with ``effective_tp_rank=0`` / ``effective_tp_size=1`` while each
worker keeps its own CUDA device. The engine must accept those replica
registrations and keep save (submitted by rank 0 only) and load (issued per
device) working.

Observed failure (GLM-5.1-FP8 TP8, vLLM + PegaFlow, 2026-06-09):

    save RPC failed: code: 'Client specified an invalid argument', message:
    "invalid argument: slot 80 registered twice:
     device=3 pp_rank=0 tp_rank=0 layer=model.layers.40.self_attn.attn;
     device=6 pp_rank=0 tp_rank=0 layer=model.layers.40.self_attn.attn"

The engine seals the instance topology (one slot per ``layer_id * tp_size +
tp_rank``) once every worker has registered; MLA replicas mean several devices
own the same slot, which the seal must accept within one pipeline stage.

This test scales the production topology to 2 devices: both register the full
GLM-5.1 layer set (158 caches = (78 hidden + 1 MTP) layers x 2 caches, derived
from the real config.json in tests/data) with the same effective tp_rank. Save
from device 0 must succeed and the saved blocks must be loadable into device 1
with identical bytes.

Requires >= 2 CUDA devices (run on a multi-GPU box, e.g. jiuzhang 39).
"""

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pegaflow.connector.common import ConnectorContext, detect_mla  # noqa: E402
from pegaflow.connector.state_manager import ServiceStateManager  # noqa: E402
from pegaflow.connector.worker import WorkerConnector  # noqa: E402
from pegaflow.pegaflow import EngineRpcClient, PyLoadState, QueryReady  # noqa: E402

from .conftest import PegaServerProcess, find_available_port  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

GLM51_CONFIG_PATH = Path(__file__).parent / "data" / "glm51_fp8_config.json"

NUM_BLOCKS = 4
BLOCK_SIZE = 16
SAVED_BLOCK_IDS = [0, 1]
LOAD_DST_BLOCK_IDS = [2, 3]
WAIT_TIMEOUT_SECONDS = 30.0


def _glm51_topology() -> tuple[dict, list[str]]:
    """Derive the GLM-5.1 KV layer set from the real config.json."""
    hf_config = json.loads(GLM51_CONFIG_PATH.read_text())

    # GLM-5.1 must be detected as MLA: that is what flips every TP rank to
    # effective_tp_rank=0 in production.
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**hf_config))
    )
    assert detect_mla(vllm_config), "GLM-5.1 config must be detected as MLA"

    # vLLM registers two caches per transformer layer for DSA models (main MLA
    # attention + sparse indexer) and places the MTP layer at index
    # num_hidden_layers (model.layers.78 for GLM-5.1).
    num_model_layers = hf_config["num_hidden_layers"] + hf_config["num_nextn_predict_layers"]
    layer_names: list[str] = []
    for i in range(num_model_layers):
        layer_names.append(f"model.layers.{i}.self_attn.attn")
        layer_names.append(f"model.layers.{i}.self_attn.indexer.k_cache")

    return hf_config, layer_names


class ReplicaWorker:
    """One simulated TP worker holding GLM-5.1-shaped KV caches on its own GPU.

    Registers through the production ``WorkerConnector.register_kv_caches``
    path so the RPC carries exactly what vLLM workers send: per-cache CUDA IPC
    handles, the layer names, and the MLA-collapsed
    effective_tp_rank/effective_tp_size.
    """

    def __init__(
        self,
        engine_client,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        world_size: int,
        hf_config: dict,
        layer_names: list[str],
        fill: str,
    ):
        device = torch.device(f"cuda:{tp_rank}")
        mla_head_dim = hf_config["kv_lora_rank"] + hf_config["qk_rope_head_dim"]
        index_head_dim = hf_config["index_head_dim"]

        torch.manual_seed(42 + tp_rank)
        self.kv_caches: dict[str, torch.Tensor] = {}
        for name in layer_names:
            head_dim = index_head_dim if ".indexer." in name else mla_head_dim
            shape = (NUM_BLOCKS, BLOCK_SIZE, head_dim)
            if fill == "random":
                cache = torch.rand(shape, dtype=torch.bfloat16, device=device)
            else:
                cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            self.kv_caches[name] = cache

        self.ctx = ConnectorContext(
            instance_id=instance_id,
            namespace=namespace,
            block_size=BLOCK_SIZE,
            tp_size=world_size,
            world_size=world_size,
            tp_rank=tp_rank,
            device_id=tp_rank,
            engine_client=engine_client,
            state_manager=ServiceStateManager(engine_client),
            is_mla=True,
        )
        # Pin the production contract this repro depends on: MLA collapses the
        # engine-visible TP topology to a single rank.
        assert self.ctx.effective_tp_rank == 0
        assert self.ctx.effective_tp_size == 1
        self.connector = WorkerConnector(self.ctx)

    def register(self) -> None:
        self.connector.register_kv_caches(self.kv_caches)

    def close(self) -> None:
        self.connector.shutdown()
        self.kv_caches.clear()


@pytest.fixture(scope="module")
def dual_device_server():
    """A pegaflow-server managing two CUDA devices (production manages all 8)."""
    server = PegaServerProcess(port=find_available_port(), devices="0,1")
    if not server.start():
        pytest.skip("PegaServer binary not found or failed to start")
    yield server
    server.stop()


def _wait_for_ready_lease(engine_client, instance_id: str, block_hashes: list[bytes]) -> bytes:
    """Poll until every saved block is queryable, returning its load lease."""
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    last = None
    while time.time() < deadline:
        result = engine_client.query_prefetch(instance_id, block_hashes, req_id="mla-replica-load")
        last = result
        if isinstance(result, QueryReady):
            if result.num_hit_blocks == len(block_hashes):
                return result.lease
            if result.lease:
                engine_client.release(result.lease)
        time.sleep(0.2)
    raise AssertionError(f"saved blocks never became queryable, last query result: {last!r}")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 CUDA devices")
def test_mla_replica_devices_save_and_load(dual_device_server):
    hf_config, layer_names = _glm51_topology()

    engine_client = EngineRpcClient(dual_device_server.endpoint)
    instance_id = f"glm51-mla-{uuid.uuid4().hex[:8]}"
    workers: list[ReplicaWorker] = []
    try:
        # Worker 0 holds the data to save; worker 1 starts zeroed and is the
        # load target. Both register the full 158-cache layer set with
        # effective_tp_rank=0, exactly like GLM-5.1 TP8 ranks do. The engine
        # seals the topology when the second (world_size-th) worker registers.
        for tp_rank, fill in ((0, "random"), (1, "zeros")):
            worker = ReplicaWorker(
                engine_client,
                instance_id,
                namespace="glm51-mla-replica-test",
                tp_rank=tp_rank,
                world_size=2,
                hf_config=hf_config,
                layer_names=layer_names,
                fill=fill,
            )
            workers.append(worker)
            worker.register()

        # Drive the engine directly: rank-0 saves every layer (a valid server
        # path — replica slots accept any device as owner, see #336). Connector-
        # level per-rank layer sharding is covered in test_combine_hashes.py.
        block_hashes = [uuid.uuid4().bytes * 2 for _ in SAVED_BLOCK_IDS]
        saves = [(name, SAVED_BLOCK_IDS, block_hashes) for name in layer_names]
        ok, message = engine_client.save(
            instance_id,
            workers[0].ctx.effective_tp_rank,
            workers[0].ctx.pp_rank,
            workers[0].ctx.device_id,
            saves,
        )
        assert ok, f"save from the MLA rank-0 replica must succeed, got: {message}"

        lease = _wait_for_ready_lease(engine_client, instance_id, block_hashes)

        # Every replica device must be able to load the blocks back.
        load_state = PyLoadState()
        ok, message = engine_client.load(
            instance_id,
            workers[1].ctx.effective_tp_rank,
            workers[1].ctx.device_id,
            load_state.shm_name(),
            layer_names,
            [(lease, LOAD_DST_BLOCK_IDS)],
        )
        assert ok, f"load into the second MLA replica device must succeed, got: {message}"

        deadline = time.time() + WAIT_TIMEOUT_SECONDS
        while not load_state.is_ready() and time.time() < deadline:
            time.sleep(0.05)
        assert load_state.is_ready(), "load did not complete in time"
        assert load_state.get_state() == 1, f"load failed with state {load_state.get_state()}"

        src_blocks = slice(SAVED_BLOCK_IDS[0], SAVED_BLOCK_IDS[-1] + 1)
        dst_blocks = slice(LOAD_DST_BLOCK_IDS[0], LOAD_DST_BLOCK_IDS[-1] + 1)
        for name in layer_names:
            src = workers[0].kv_caches[name][src_blocks].cpu()
            dst = workers[1].kv_caches[name][dst_blocks].cpu()
            assert torch.equal(dst, src), f"loaded KV bytes differ from saved KV for {name}"
    finally:
        for worker in workers:
            worker.close()
        torch.cuda.empty_cache()
