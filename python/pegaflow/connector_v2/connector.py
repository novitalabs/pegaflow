"""vLLM entry point: ``PegaKVConnectorV2`` over RPC V2 objects.

Thin translation between vLLM's connector API and ``scheduler`` / ``worker``.
Supported: Full, MLA, sliding-window and Mamba (``align``) cache groups, TP
only. Failed loads follow vLLM's ``kv_load_failure_policy``. Stats and
Prometheus metrics are v1's.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from pegaflow.connector.connector_metrics import PegaKVConnectorStats, PegaPromMetrics

from .layout import Parallel, Unsupported, build_layout
from .policy import Policy
from .scheduler import Scheduler, StepPlan
from .worker import Region, Worker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


def _route_package_logs() -> None:
    """vLLM configures only the ``vllm`` logger: give this package a handler."""
    pkg = logging.getLogger(__package__)
    if pkg.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s"))
    pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False


_route_package_logs()


@dataclass
class PegaConnectorMetadata(KVConnectorMetadata):
    plan: StepPlan = field(default_factory=StepPlan)


@dataclass
class PegaWorkerMetadata(KVConnectorWorkerMetadata):
    """The latest plan step every worker finished saving."""

    saved_through: int

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, PegaWorkerMetadata)
        return PegaWorkerMetadata(min(self.saved_through, other.saved_through))


class PegaKVConnectorV2(KVConnectorBase_V1, SupportsHMA):
    _cache_hit_source = CacheHitSource.HOST

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        from pegaflow import EngineRpcClient

        pc = vllm_config.parallel_config
        for name in ("pipeline_parallel_size", "decode_context_parallel_size"):
            if getattr(pc, name, 1) > 1:
                raise Unsupported(f"connector v2 supports TP only, got {name}={getattr(pc, name)}")
        tp_rank = 0
        if role == KVConnectorRole.WORKER:
            from vllm.distributed.parallel_state import get_tp_group

            tp_rank = get_tp_group().rank_in_group
        mc, cc = vllm_config.model_config, vllm_config.cache_config
        identity = {
            "model": mc.model,
            "revision": getattr(mc, "revision", None),
            "dtype": str(mc.dtype),
            "kv_cache_dtype": str(cc.cache_dtype),
            "quantization": getattr(mc, "quantization", None),
            "hash_algo": getattr(cc, "prefix_caching_hash_algo", None),
        }
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        par = Parallel(pc.tensor_parallel_size)
        self._layout = build_layout(kv_cache_config, par, identity)
        is_mla = getattr(mc.hf_text_config, "kv_lora_rank", None) is not None  # v1 detect_mla
        self._policy = Policy.of(
            self._layout.groups, is_mla, extra.get("pegaflow.transfer_backend")
        )
        if self._policy.copy_per_socket:
            par = Parallel(pc.tensor_parallel_size, _tp_sockets(pc.tensor_parallel_size))
            self._layout = build_layout(kv_cache_config, par, identity)
        host = os.environ.get("PEGAFLOW_HOST") or extra.get("pegaflow.host", "http://127.0.0.1")
        port = os.environ.get("PEGAFLOW_PORT") or extra.get("pegaflow.port", 50055)
        self._rpc = EngineRpcClient(f"{host}:{port}")
        self._instance_id = _instance_id(vllm_config)
        logger.info(
            "[PegaKVConnectorV2] role=%s instance=%s endpoint=%s policy=%s groups=%s",
            role.name,
            self._instance_id,
            self._rpc.endpoint(),
            self._policy,
            [(ns.gid, ns.group, ns.slices, ns.savers) for ns in self._layout.namespaces],
        )

        self._scheduler: Scheduler | None = None
        self._worker: Worker | None = None
        self._pins_bound = False
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = Scheduler(self._layout, self._rpc, self._instance_id, self._policy)
            # The server logs namespaces by their first 8 bytes in hex.
            namespaces = ",".join(ns.namespace[:8].hex() for ns in self._layout.namespaces)
            self._rpc.start_session_watcher(self._instance_id, namespaces)
        else:
            from pegaflow import PyLoadState

            self._worker = Worker(
                self._layout,
                self._rpc,
                self._instance_id,
                rank=tp_rank,
                world_size=pc.tensor_parallel_size,
                device_id=_server_device_id(),
                new_load_state=PyLoadState,
                mark_compute=_mark_compute,
                transfer_backend=self._policy.transfer_backend,
            )

    @property
    def requires_kv_delivery(self) -> bool:
        return False  # a dropped save is a future miss

    # ---------------------------------------------------------- scheduler side
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        assert self._scheduler is not None
        return self._scheduler.get_loaded_group_ids(request.request_id)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(
            request, blocks.get_block_ids(), num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self._scheduler is not None
        manager = getattr(self, "_kv_cache_manager", None)
        if manager is not None and not self._pins_bound:
            self._pins_bound = True
            self._scheduler.bind_pins(*_block_pins(manager.block_pool))
        n_sched = scheduler_output.num_scheduled_tokens
        scheduled = [
            (r.req_id, r.num_computed_tokens, n_sched.get(r.req_id, 0))
            for r in scheduler_output.scheduled_new_reqs
        ]
        cached = scheduler_output.scheduled_cached_reqs
        scheduled.extend(
            (req_id, computed, n_sched.get(req_id, 0))
            for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        )
        state = scheduler_output.kv_connector_block_state
        self._scheduler.plan_saves(
            scheduled,
            state.get_block_ids if state is not None else lambda _: None,
            getattr(state, "boundary_state_offloads", None),
        )
        return PegaConnectorMetadata(self._scheduler.take_plan())

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self._scheduler is not None
        if connector_output.failed_recving:
            self._scheduler.on_loads_failed(connector_output.failed_recving)
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, PegaWorkerMetadata):
            self._scheduler.on_saved_through(meta.saved_through)

    def has_pending_push_work(self) -> bool:
        # Keep stepping until every save is reported done: pins clear then.
        return self._scheduler is not None and self._scheduler.has_pending_saves()

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        # Never held: blocks a pending save reads stay pinned instead.
        self._scheduler.request_finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    # ------------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        strides = {
            name: t.block_stride
            for t in self._kv_cache_config.kv_cache_tensors
            for name in t.layers
        }
        self._worker.register(kv_caches, strides, _share)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self._worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, PegaConnectorMetadata)
        self._worker.start(meta.plan)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # loads are async, between steps

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        return  # saves are issued once per step in wait_for_save

    def wait_for_save(self) -> None:
        assert self._worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, PegaConnectorMetadata)
        self._worker.save(meta.plan)

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        assert self._worker is not None
        finished, failed = self._worker.poll()
        return KVConnectorTransferResults(
            finished_sending=set(), finished_recving=finished, failed_recving=failed
        )

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self._worker is not None
        # Every step from every worker: aggregation takes the min.
        return PegaWorkerMetadata(self._worker.saved_through())

    # ------------------------------------------------------------------ stats
    def get_kv_connector_stats(self) -> PegaKVConnectorStats | None:
        side = self._scheduler or self._worker
        data = side.take_stats() if side is not None else {}
        return PegaKVConnectorStats(data=data) if data else None

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> PegaKVConnectorStats | None:
        return None if data is None else PegaKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ) -> PegaPromMetrics:
        return PegaPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown()
        if self._worker is not None:
            self._worker.shutdown()


def _instance_id(vllm_config: VllmConfig) -> str:
    instance_id = (
        vllm_config.kv_transfer_config.engine_id
        or os.environ.get("PEGAFLOW_INSTANCE_ID")
        or vllm_config.instance_id
        or uuid.uuid4().hex
    )
    pc = vllm_config.parallel_config
    if pc.data_parallel_size > 1 and pc.data_parallel_rank_local is not None:
        instance_id = f"{instance_id}_dp{pc.data_parallel_rank_local}"
    return instance_id


def _block_pins(pool: Any) -> tuple[Any, Any]:
    """Pin GPU blocks by id: one reference each, so a block vLLM frees
    meanwhile stays out of the free queue until the unpin."""

    def pin(block_ids: list[int]) -> Any:
        blocks = [pool.blocks[b] for b in block_ids]
        pool.touch(blocks)
        return blocks

    return pin, pool.free_blocks


def _share(tensor: torch.Tensor) -> Region:
    """The tensor's CUDA allocation as an RPC V2 region."""
    storage = tensor.untyped_storage()
    _, handle, storage_size, storage_offset, *_ = storage._share_cuda_()
    # torch prefixes the 64-byte cudaIpcMemHandle with a 2-byte type tag.
    if len(handle) != 66:
        raise Unsupported(f"unexpected CUDA IPC handle of {len(handle)} bytes")
    offset = storage_offset + tensor.data_ptr() - storage.data_ptr()
    return Region(bytes(handle[2:]), storage_offset + storage_size, offset)


def _server_device_id() -> int:
    """CUDA ordinal of this worker's GPU as the server sees it."""
    local = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return local
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if local >= len(entries) or not entries[local].isdigit():
        raise RuntimeError(f"cannot map CUDA device {local} through {visible!r}")
    return int(entries[local])


def _tp_sockets(tp_size: int) -> tuple[int, ...]:
    """CPU socket (``physical_package_id``) next to each TP rank's GPU, from
    NVML and sysfs without touching CUDA, so the scheduler and every worker
    compute the same list. Unknown: all 0, one socket."""
    from vllm.platforms import current_platform
    from vllm.third_party import pynvml

    try:
        pynvml.nvmlInit()
        try:
            out = []
            for r in range(tp_size):
                gpu = current_platform.device_id_to_physical_device_id(r)
                bus = pynvml.nvmlDeviceGetPciInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu)).busId
                out.append(_pci_socket(bus.decode() if isinstance(bus, bytes) else bus))
            return tuple(out)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        logger.warning("[PegaKVConnectorV2] GPU sockets unknown: one socket", exc_info=True)
        return (0,) * tp_size


def _pci_socket(bus_id: str) -> int:
    """Socket of a PCI device (NVML ``00000000:61:00.0``) via its NUMA node."""
    dev = bus_id.lower()[-12:]  # sysfs names it 0000:61:00.0
    with open(f"/sys/bus/pci/devices/{dev}/numa_node") as f:
        node = int(f.read())
    if node < 0:
        return 0
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        cpu = f.read().strip().split(",")[0].split("-")[0]
    with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as f:
        return int(f.read())


def _mark_compute() -> torch.cuda.Event:
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return event
