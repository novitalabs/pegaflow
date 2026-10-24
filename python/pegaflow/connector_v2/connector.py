"""vLLM entry point: ``PegaKVConnectorV2``.

Thin translation between vLLM's connector API and ``core``: every decision
(what to query, lease, load, save) lives in ``core`` + ``policy``; per-spec
knowledge lives in ``adapters``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput

from .adapters import Parallel
from .core import (
    Scheduler,
    StepPlan,
    Worker,
    build_layout,
    endpoint_of,
    layer_specs,
    may_zero_or_copy_blocks,
)
from .policy import load_policy

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _route_package_logs() -> None:
    """Emit ``pegaflow.connector_v2.*`` INFO lines (e.g. worker ``save stats``).

    vLLM 0.32 configures only the ``vllm`` logger (propagate=False). Loggers
    outside it have no handler, so Python's last-resort handler drops
    everything below WARNING.
    """
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

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        mc = vllm_config.model_config
        pc = vllm_config.parallel_config
        par = Parallel(
            tp_size=pc.tensor_parallel_size,
            pp_size=pc.pipeline_parallel_size,
            dcp_size=pc.decode_context_parallel_size,
            dcp_interleave=pc.cp_kv_cache_interleave_size,
            world_size=pc.world_size,
            kv_tp_replicas=getattr(kv_cache_config, "kv_tp_replicas", 1) or 1,
        )
        if role == KVConnectorRole.WORKER:
            par = _with_worker_ranks(par)
        cc = vllm_config.cache_config
        identity = {
            "model": mc.model,
            "revision": getattr(mc, "revision", None),
            "quantization": getattr(mc, "quantization", None),
            "dtype": str(mc.dtype),
            "kv_cache_dtype": str(cc.cache_dtype),
            "hash_algo": getattr(cc, "prefix_caching_hash_algo", None),
        }
        # Raises adapters.Unsupported on both roles for unknown spec classes.
        self._layout = build_layout(kv_cache_config, par, identity)
        self._instance_id = (
            os.environ.get("PEGAFLOW_INSTANCE_ID") or self._kv_transfer_config.engine_id
        )
        self._rpc = EngineRpcClient(endpoint_of(extra))
        logger.info(
            "[PegaKVConnectorV2] role=%s instance=%s namespace=%s endpoint=%s groups=%s from=%s",
            role.name,
            self._instance_id,
            self._layout.namespace,
            self._rpc.endpoint(),
            [(g.gid, repr(g.sem)) for g in self._layout.groups],
            # Which checkout vLLM imported (several trees share one venv).
            os.path.dirname(__file__),
        )

        self._scheduler: Scheduler | None = None
        self._worker: Worker | None = None
        if role == KVConnectorRole.SCHEDULER:
            if getattr(self._kv_transfer_config, "kv_load_failure_policy", None) == "fail":
                # vLLM's scheduler reads the TOP-LEVEL kv_transfer_config. Under
                # MultiConnector this child config always shows the default
                # "fail" (vLLM 0.32 builds it from the child dict), so this
                # cannot tell whether recompute is in effect.
                logger.info(
                    "[PegaKVConnectorV2] own kv_load_failure_policy=fail; the effective "
                    "policy is the top-level kv_transfer_config's. Under fail, a failed "
                    "load (e.g. an expired lease) errors the request: use recompute"
                )
            profile = extra.get("model_profile") or getattr(mc, "architecture", None)
            # vLLM maps a failed synchronous load (invalid_block_ids) back to
            # requests only with one KV cache group, and zeroes new blocks on
            # the compute stream, unordered against the server's copy.
            sync_loads = len(kv_cache_config.kv_cache_groups) == 1 and not getattr(
                kv_cache_config, "needs_kv_cache_zeroing", True
            )
            self._scheduler = Scheduler(
                self._layout,
                load_policy(profile),
                self._rpc,
                self._instance_id,
                sync_loads=sync_loads,
                # Steps vLLM schedules before it processes a step's output
                # (fallback: the largest value vLLM 0.32 uses, pp + 1).
                confirm_steps=getattr(
                    vllm_config, "max_concurrent_batches", pc.pipeline_parallel_size + 1
                ),
            )
            try:
                self._rpc.start_session_watcher(
                    self._instance_id, self._layout.namespace, par.tp_size, par.world_size
                )
            except Exception:
                logger.warning("[PegaKVConnectorV2] session watcher failed", exc_info=True)
        else:
            from pegaflow import PyLoadState

            sync_saves = may_zero_or_copy_blocks(kv_cache_config)
            logger.info("[PegaKVConnectorV2] saves=%s", "sync" if sync_saves else "async")
            self._worker = Worker(
                self._layout,
                self._rpc,
                self._instance_id,
                par,
                _server_device_id(),
                _ipc_bytes,
                PyLoadState,
                _mark_compute,
                sync_saves=sync_saves,
            )

    @property
    def requires_kv_delivery(self) -> bool:
        return False  # a dropped save is a future miss

    # ---------------------------------------------------------- scheduler side
    def on_new_request(self, request: Request) -> None:
        assert self._scheduler is not None
        self._scheduler.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        assert self._scheduler is not None
        return self._scheduler.loaded_group_ids(request.request_id)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(
            request, blocks.get_block_ids(), num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self._scheduler is not None
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
        block_state = scheduler_output.kv_connector_block_state
        self._scheduler.plan_saves(
            scheduled,
            block_state.get_block_ids if block_state is not None else lambda _: None,
            scheduler_output.preempted_req_ids or (),
        )
        return PegaConnectorMetadata(self._scheduler.take_plan())

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        # A failed sync load: KV computed on it may be reused, stop saving.
        if self._scheduler is not None and connector_output.invalid_block_ids:
            self._scheduler.on_load_errors(connector_output.invalid_block_ids)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        # True while a save of the request's blocks may be pending: vLLM keeps
        # the blocks until the workers report the request in finished_sending.
        return self._scheduler.request_finished(request.request_id), None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    # ------------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        specs: dict[str, Any] = {}
        for g in self._layout.groups:
            kv_group = self._kv_cache_config.kv_cache_groups[g.gid]
            specs.update(layer_specs(kv_group.kv_cache_spec, list(kv_group.layer_names)))
        self._worker.register(kv_caches, specs)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # Blocks vLLM freed (preemption) may be reused by this step while a
        # save still reads them. Runs before any KV write of the step; on the
        # V2 model runner after block zeroing and copy-on-write, which is why
        # layouts with those save synchronously (core.may_zero_or_copy_blocks).
        assert self._worker is not None
        assert isinstance(kv_connector_metadata, PegaConnectorMetadata)
        if kv_connector_metadata.plan.flush:
            self._worker.flush()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self._worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, PegaConnectorMetadata)
        self._worker.start(meta.plan)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # sync loads complete in start_load_kv, never layer-pipelined

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
        sending, finished, failed = self._worker.poll()
        return KVConnectorTransferResults(
            finished_sending=sending, finished_recving=finished, failed_recving=failed
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self._worker is not None
        return self._worker.take_invalid()

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown()
        if self._worker is not None:
            self._worker.shutdown()


def _with_worker_ranks(par: Parallel) -> Parallel:
    from vllm.distributed.parallel_state import (
        get_dcp_world_size_and_rank,
        get_pp_group,
        get_tp_group,
    )

    _, dcp_rank = get_dcp_world_size_and_rank()
    return dataclasses.replace(
        par,
        tp_rank=get_tp_group().rank_in_group,
        pp_rank=get_pp_group().rank_in_group,
        dcp_rank=dcp_rank,
    )


def _server_device_id() -> int:
    """CUDA ordinal of this worker's GPU as the server sees it.

    The server process is not restricted by this worker's CUDA_VISIBLE_DEVICES,
    so map the local ordinal back through it. Only plain indices can be mapped;
    UUID/MIG entries would silently pick the wrong GPU, so they are rejected.
    """
    local = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return local
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if local >= len(entries) or not entries[local].isdigit():
        raise RuntimeError(
            f"cannot map local CUDA device {local} through CUDA_VISIBLE_DEVICES="
            f"{visible!r} to the server's ordinal; use plain GPU indices"
        )
    return int(entries[local])


def _mark_compute() -> torch.cuda.Event:
    """Event after the work enqueued so far on the compute stream.

    The saver thread waits on it (``Event.synchronize`` releases the GIL), so
    the step thread never blocks on the GPU for a save.
    """
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return event


def _ipc_bytes(tensor: torch.Tensor) -> bytes:
    from pegaflow.ipc_wrapper import CudaIPCWrapper

    return pickle.dumps(CudaIPCWrapper(tensor))
