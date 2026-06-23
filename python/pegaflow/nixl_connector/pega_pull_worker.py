# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PegaFlow RDMA-backed pull worker for the Pega NIXL connector."""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    EngineTransferInfo,
    TransferTopology,
)

from pegaflow.nixl_connector.metadata import (
    GET_META_MSG,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
    ReqMeta,
    compute_nixl_compatibility_hash,
)
from pegaflow.nixl_connector.pull_worker import NixlPullConnectorWorker
from pegaflow.nixl_connector.rdma_transport import (
    PegaNixlRdmaTransport,
    handshake_from_wire,
    handshake_to_wire,
)
from pegaflow.nixl_connector.tp_mapping import ReadSpec, compute_tp_mapping
from pegaflow.nixl_connector.utils import zmq_ctx
from pegaflow.pd_connector.metadata import BlockIds, PdHandshake
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class PegaNixlPullConnectorWorker(NixlPullConnectorWorker):
    """NIXL pull worker with PegaFlow RDMA READ as the data plane."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self._use_pega_rdma_transport = True
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.pega_rdma = PegaNixlRdmaTransport(
            vllm_config=self.vllm_config,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
        )
        self._remote_rdma_handshakes: dict[EngineId, dict[int, PdHandshake]] = {}
        self._pending_rdma_recvs: dict[ReqId, int] = {}
        self._completed_rdma_recvs: queue.Queue[ReqId] = queue.Queue()
        self._rdma_pull_executor = ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="pega-nixl-rdma-pull",
        )

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )
        self.pega_rdma.register_kv_caches(
            kv_caches,
            layer_specs=self._layer_specs,
            expected_num_blocks=self._logical_num_blocks,
        )
        self.device_kv_caches = kv_caches
        self.device_id = _infer_cuda_device(kv_caches)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.num_regions = len(kv_caches) * (1 if self.use_mla else 2)
        self.num_descs = self.num_regions * self.num_blocks
        local_handshake = self.pega_rdma.build_local_handshake(
            request_id=f"{self.engine_id}:{self.tp_rank}",
            block_ids=tuple(range(self._logical_num_blocks)),
            imm_id=0,
        )
        self.xfer_handshake_metadata = encode_pega_rdma_handshake_payload(
            self.compat_hash,
            local_handshake,
        )

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        remote_rank_to_agent_name: dict[int, str] = {}
        path = make_zmq_path("tcp", host, port)
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_rank in p_remote_ranks:
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.send(msgspec.msgpack.encode((GET_META_MSG, remote_rank)))
                handshake_payload = msgspec.msgpack.decode(
                    sock.recv(),
                    type=NixlHandshakePayload,
                )
                assert self.compat_hash is not None
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        "Pega NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}."
                    )
                handshake = decode_pega_rdma_handshake_payload(handshake_payload)
                if handshake.engine_id != expected_engine_id:
                    raise RuntimeError(
                        "Remote Pega RDMA engine ID mismatch. "
                        f"Expected {expected_engine_id}, received {handshake.engine_id}."
                    )
                self._add_remote_rdma_handshake(
                    expected_engine_id,
                    remote_rank,
                    remote_tp_size,
                    handshake,
                )
                remote_rank_to_agent_name[remote_rank] = (
                    f"pega-rdma:{expected_engine_id}:{remote_rank}"
                )
        return remote_rank_to_agent_name

    def _add_remote_rdma_handshake(
        self,
        engine_id: EngineId,
        remote_rank: int,
        remote_tp_size: int,
        handshake: PdHandshake,
    ) -> None:
        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = _handshake_num_blocks(handshake)
        self._remote_rdma_handshakes.setdefault(engine_id, {})[remote_rank] = handshake
        assert self.transfer_topo is not None
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=handshake.block_size,
            remote_block_len=handshake.layers[0].regions[0].block_len,
            remote_physical_blocks_per_logical=1,
        )
        self.transfer_topo.register_remote_engine(engine_id, transfer_info)
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )
        self._engine_last_active[engine_id] = time.perf_counter()

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]

        if len(local_block_ids) == 0:
            self._completed_rdma_recvs.put(request_id)
            return

        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids,
            remote_block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        if not any(local_block_ids):
            self._completed_rdma_recvs.put(request_id)
            return

        try:
            remote_handshake = self._remote_rdma_handshakes[dst_engine_id][remote_rank]
            self.pega_rdma.start_pull_blocks(
                request_id=request_id,
                remote_handshake=remote_handshake,
                local_block_ids=_as_block_ids(local_block_ids),
                remote_block_ids=_as_block_ids(remote_block_ids),
            )
        except Exception as e:
            self._log_failure(
                failure_type="pega_rdma_pull_submit_failed",
                req_id=request_id,
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, None)
            self._pending_rdma_recvs.pop(request_id, None)
            return

        future = self._rdma_pull_executor.submit(
            self.pega_rdma.wait_for_pull_blocks,
            request_id=request_id,
        )

        def _done_callback(_future, req_id: str = request_id) -> None:
            try:
                _future.result()
            except Exception as e:
                self._log_failure(
                    failure_type="pega_rdma_pull_failed",
                    req_id=req_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                )
                self._handle_failed_transfer(req_id, None)
                return
            self._completed_rdma_recvs.put(req_id)

        future.add_done_callback(_done_callback)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]
        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        self._pending_rdma_recvs[req_id] = len(read_specs)
        for spec in read_specs:
            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=0,
                remote_xfer_side_handle=0,
            )

    def _get_new_notifs(self) -> set[str]:
        return set()

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        return None

    def get_finished(self) -> tuple[set[str], set[str]]:
        while not self._completed_rdma_recvs.empty():
            try:
                req_id = self._completed_rdma_recvs.get_nowait()
            except queue.Empty:
                break
            remaining = self._pending_rdma_recvs.get(req_id, 1) - 1
            if remaining > 0:
                self._pending_rdma_recvs[req_id] = remaining
                continue
            self._pending_rdma_recvs.pop(req_id, None)
            self._recving_transfers.setdefault(req_id, [])
        return super().get_finished()

    def shutdown(self):
        if not hasattr(self, "_handshake_initiation_executor"):
            return
        self._rdma_pull_executor.shutdown(wait=False, cancel_futures=True)
        self._handshake_initiation_executor.shutdown(wait=False)
        for req_id in list(self._recving_metadata):
            self.pega_rdma.close_request(req_id)
        self._recving_metadata.clear()
        self._recving_transfers.clear()
        self._pending_rdma_recvs.clear()


def encode_pega_rdma_handshake_payload(
    compatibility_hash: str,
    handshake: PdHandshake,
) -> NixlHandshakePayload:
    return NixlHandshakePayload(
        compatibility_hash=compatibility_hash,
        agent_metadata_bytes=msgspec.msgpack.encode(handshake_to_wire(handshake)),
    )


def decode_pega_rdma_handshake_payload(payload: NixlHandshakePayload) -> PdHandshake:
    return handshake_from_wire(msgspec.msgpack.decode(payload.agent_metadata_bytes))


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int:
    for tensor in kv_caches.values():
        get_device = getattr(tensor, "get_device", None)
        if get_device is not None:
            return max(int(get_device()), 0)
    return 0


def _handshake_num_blocks(handshake: PdHandshake) -> int:
    return max(block_id for layer in handshake.layers for block_id in layer.block_ids) + 1


def _as_block_ids(block_ids: Any) -> BlockIds:
    return tuple(list(group) for group in block_ids)
