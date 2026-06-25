# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""PegaFlow RDMA v1 data-plane worker for NIXL pull mode."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import init_logger

from pegaflow.nixl_connector.metadata import TransferHandle
from pegaflow.nixl_connector.pega_rdma_v1 import (
    PegaRdmaV1Config,
    PegaRdmaV1Read,
    build_memory_regions,
    build_read_descs,
    create_rdma_engine,
    decode_handshake_notif,
    encode_handshake_request,
    encode_handshake_response,
    make_peer_key,
    reverse_peer_key,
)
from pegaflow.nixl_connector.pull_worker import NixlPullConnectorWorker
from pegaflow.nixl_connector.tp_mapping import ReadSpec

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class PegaNixlPullConnectorWorker(NixlPullConnectorWorker):
    """NIXL pull worker with PegaFlow RDMA v1 for KV READ payloads.

    NIXL remains the connector/control plane: scheduler metadata,
    TP mapping, heartbeats, failure handling, and completion notifications
    are inherited from ``NixlPullConnectorWorker``.  Only the data READ
    submission/polling path is replaced with PegaFlow RDMA v1.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._pega_rdma_config = PegaRdmaV1Config.from_extra_config(extra_config)
        self._pega_rdma = create_rdma_engine(self._pega_rdma_config)
        self._pega_rdma_reads: dict[TransferHandle, PegaRdmaV1Read] = {}
        self._pega_local_handshake: dict[str, bytes] = {}
        self._pega_pending_requests = defaultdict[str, list[tuple[str, str]]](list)
        self._pega_pending_since: dict[str, float] = {}
        self._pega_response_agents: dict[str, str] = {}
        self._pega_expected_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_submitted_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_completed_peers: dict[str, set[str]] = defaultdict(set)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._register_pega_rdma_memory()

    def _register_pega_rdma_memory(self) -> None:
        if self.use_host_buffer:
            logger.warning(
                "PegaNixlPullConnector is registering host transfer buffers for RDMA v1"
            )
        regions = build_memory_regions(
            [(addr, size, device_id) for addr, size, device_id, _ in self.local_registered_blocks_data]
        )
        self._pega_rdma.register_memory(regions)
        logger.info(
            "Pega RDMA v1 registered %d local regions on nics=%s qps_per_peer=%d",
            len(regions),
            ",".join(self._pega_rdma_config.nics),
            self._pega_rdma_config.qps_per_peer,
        )

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
        block_size_ratio = self.transfer_topo.block_size_ratio(remote_info.remote_block_size)
        if block_size_ratio > 1:
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                local_block_ids_mapped = local_block_ids_mapped[: len(remote_block_ids0)]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]

        notif_id = f"{remote_request_id}:{self.world_size}".encode()
        if len(local_block_ids) == 0:
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        assert len(remote_block_ids) == len(local_block_ids) == len(
            self.kv_cache_config.kv_cache_groups
        )
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids,
            remote_block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )

        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )
        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        handle = None
        try:
            peer_key = make_peer_key(self.engine_id, self.tp_rank, dst_engine_id, remote_rank)
            self._pega_expected_peers[request_id].add(peer_key)
            if not self._ensure_pega_rdma_connection(
                peer_key,
                dst_engine_id,
                remote_rank,
                request_id,
            ):
                return
            if peer_key in self._pega_submitted_peers[request_id]:
                return

            local_blocks_data = self._local_blocks_data_for_handle(
                local_xfer_side_handle,
                block_size=remote_info.remote_block_size,
            )
            remote_blocks_data = self.dst_blocks_data[dst_engine_id][remote_rank]
            descs = build_read_descs(
                local_blocks_data,
                remote_blocks_data,
                local_block_descs_ids,
                remote_block_descs_ids,
            )
            handle = self._pega_rdma.read_async(peer_key, descs)
            self._pega_rdma_reads[handle] = PegaRdmaV1Read(
                remote_addr=peer_key,
                handle=handle,
                notif_agent=self._remote_agents[dst_engine_id][remote_rank],
                notif_msg=notif_id,
                submitted_at=time.perf_counter(),
                bytes_transferred=sum(desc["len"] for desc in descs),
                num_descriptors=len(descs),
            )
            self._recving_transfers[request_id].append(handle)
            self._pega_submitted_peers[request_id].add(peer_key)
            logger.debug(
                "Pega RDMA v1 READ submitted req=%s remote=%s rank=%s descs=%d",
                request_id,
                dst_engine_id,
                remote_rank,
                len(descs),
            )
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Pega RDMA v1 READ setup failed; marking blocks invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _local_blocks_data_for_handle(
        self,
        local_xfer_side_handle: int,
        *,
        block_size: int,
    ) -> list[tuple[int, int, int]]:
        if local_xfer_side_handle == self.src_xfer_handles_by_block_size[block_size]:
            return self.src_blocks_data_by_block_size[block_size]
        for tp_ratio, handles in self.src_xfer_handles_by_tp_ratio.items():
            for idx, handle in enumerate(handles):
                if handle == local_xfer_side_handle:
                    return self.src_blocks_data_by_tp_ratio[tp_ratio][idx]
        raise RuntimeError(f"unknown local xfer side handle {local_xfer_side_handle}")

    def _ensure_pega_rdma_connection(
        self,
        peer_key: str,
        dst_engine_id: str,
        remote_rank: int,
        request_id: str,
    ) -> bool:
        status = self._pega_rdma.get_or_prepare(peer_key)
        if status.status == "existing":
            return True
        if status.status == "connecting":
            self._add_pending_request(peer_key, request_id, dst_engine_id)
            self._pega_pending_since.setdefault(peer_key, time.perf_counter())
            return False
        if status.metadata is None:
            raise RuntimeError(f"RDMA v1 get_or_prepare({peer_key}) returned no metadata")

        local_metadata = bytes(status.metadata)
        self._pega_local_handshake[peer_key] = local_metadata
        agent_name = self._remote_agents[dst_engine_id][remote_rank]
        self.nixl_wrapper.send_notif(
            agent_name,
            notif_msg=encode_handshake_request(
                peer_key,
                local_metadata,
                self.nixl_wrapper.get_agent_metadata(),
            ),
        )
        self._add_pending_request(peer_key, request_id, dst_engine_id)
        self._pega_pending_since.setdefault(peer_key, time.perf_counter())
        return False

    def _add_pending_request(self, peer_key: str, request_id: str, dst_engine_id: str) -> None:
        pending = self._pega_pending_requests[peer_key]
        entry = (request_id, dst_engine_id)
        if entry not in pending:
            pending.append(entry)

    def _get_new_notifs(self) -> set[str]:
        assert self.transfer_topo is not None
        self._expire_pega_rdma_handshakes()
        notified_req_ids: set[str] = set()
        for _source_agent, notifs in self.nixl_wrapper.get_new_notifs().items():
            for notif in notifs:
                handshake = decode_handshake_notif(notif)
                if handshake is not None:
                    self._handle_pega_rdma_handshake_notif(handshake)
                    continue

                msg = notif.decode("utf-8")
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)
                if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                    logger.error(
                        "Potentially invalid KV blocks for unrecognized request %s "
                        "were retrieved by a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
                consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1

                self.consumer_notification_counts_by_req[req_id] += 1
                if self.consumer_notification_counts_by_req[req_id] == consumers_per_producer:
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _handle_pega_rdma_handshake_notif(
        self,
        payload: dict[str, object],
    ) -> None:
        kind = payload.get("kind")
        peer_key = payload.get("peer_key")
        metadata = payload.get("metadata")
        response_agent_metadata = payload.get("response_agent_metadata")
        if not isinstance(kind, str) or not isinstance(peer_key, str) or not isinstance(
            metadata, bytes
        ):
            raise ValueError("invalid Pega RDMA v1 handshake notification")

        if kind == "request":
            if not isinstance(response_agent_metadata, bytes):
                raise ValueError(
                    "Pega RDMA v1 handshake request missing response agent metadata"
                )
            local_peer_key = reverse_peer_key(peer_key)
            local = self._prepare_or_get_local_metadata(local_peer_key)
            self._pega_rdma.complete_handshake(local_peer_key, local, metadata)
            agent_name = self._get_pega_response_agent(peer_key, response_agent_metadata)
            self.nixl_wrapper.send_notif(
                agent_name,
                notif_msg=encode_handshake_response(peer_key, local),
            )
            logger.debug("Pega RDMA v1 handshake request accepted peer=%s", local_peer_key)
            return

        if kind == "response":
            local = self._pega_local_handshake.pop(peer_key, None)
            self._pega_pending_since.pop(peer_key, None)
            if local is None:
                local = self._prepare_or_get_local_metadata(peer_key)
            self._pega_rdma.complete_handshake(peer_key, local, metadata)
            pending = self._pega_pending_requests.pop(peer_key, [])
            for req_id, engine_id in pending:
                meta = self._recving_metadata.get(req_id)
                if meta is not None:
                    self._ready_requests.put((req_id, meta))
                else:
                    logger.debug(
                        "Skipping stale request %s after RDMA handshake with %s",
                        req_id,
                        engine_id,
                    )
            logger.debug("Pega RDMA v1 handshake response completed peer=%s", peer_key)
            return

        raise ValueError(f"unknown Pega RDMA v1 handshake kind {kind!r}")

    def _get_pega_response_agent(self, peer_key: str, agent_metadata: bytes) -> str:
        agent_name = self._pega_response_agents.get(peer_key)
        if agent_name is not None:
            return agent_name
        agent_name = self.nixl_wrapper.add_remote_agent(agent_metadata)
        self._pega_response_agents[peer_key] = agent_name
        return agent_name

    def _expire_pega_rdma_handshakes(self) -> None:
        now = time.perf_counter()
        timeout_s = self._pega_rdma_config.handshake_timeout_s
        expired_peers = [
            peer_key
            for peer_key, started_at in self._pega_pending_since.items()
            if now - started_at >= timeout_s
        ]
        for peer_key in expired_peers:
            self._pega_pending_since.pop(peer_key, None)
            local = self._pega_local_handshake.pop(peer_key, None)
            if local is not None:
                try:
                    self._pega_rdma.abort_handshake(peer_key, local)
                except Exception:
                    logger.debug(
                        "Failed to abort expired Pega RDMA v1 handshake peer=%s",
                        peer_key,
                        exc_info=True,
                    )

            pending = self._pega_pending_requests.pop(peer_key, [])
            for req_id, engine_id in pending:
                if req_id not in self._recving_metadata:
                    continue
                self._log_failure(
                    failure_type="pega_rdma_handshake_timeout",
                    req_id=req_id,
                    msg=(
                        "Pega RDMA v1 handshake timed out before READ setup; "
                        "marking blocks invalid"
                    ),
                    remote_engine_id=engine_id,
                    peer_key=peer_key,
                    timeout_s=timeout_s,
                )
                self._handle_failed_transfer(req_id, None)

    def _prepare_or_get_local_metadata(self, peer_key: str) -> bytes:
        local = self._pega_local_handshake.get(peer_key)
        if local is not None:
            return local
        status = self._pega_rdma.get_or_prepare(peer_key)
        if status.status == "existing":
            metadata = self._pega_rdma.local_meta_for(peer_key)
            if metadata is None:
                raise RuntimeError(f"RDMA v1 existing peer {peer_key} has no local metadata")
            local = bytes(metadata)
        elif status.status == "prepared":
            if status.metadata is None:
                raise RuntimeError(f"RDMA v1 prepared peer {peer_key} has no metadata")
            local = bytes(status.metadata)
        else:
            raise RuntimeError(f"RDMA v1 peer {peer_key} is already connecting")
        self._pega_local_handshake[peer_key] = local
        return local

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                read = self._pega_rdma_reads.get(handle)
                if read is None:
                    continue
                try:
                    state = self._pega_rdma.check_read(handle)
                    if state == "done":
                        self._pega_rdma_reads.pop(handle, None)
                        self._pega_completed_peers[req_id].add(read.remote_addr)
                        self.xfer_stats.record_transfer_values(
                            transfer_duration=time.perf_counter() - read.submitted_at,
                            post_duration=0.0,
                            bytes_transferred=read.bytes_transferred,
                            num_descriptors=read.num_descriptors,
                        )
                        self.nixl_wrapper.send_notif(read.notif_agent, notif_msg=read.notif_msg)
                    elif state == "pending":
                        in_progress.append(handle)
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            req_id=req_id,
                            msg=f"unexpected Pega RDMA v1 state {state}",
                        )
                        self._handle_failed_transfer(req_id, handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        req_id=req_id,
                        msg="Pega RDMA v1 READ failed",
                        error=e,
                    )
                    self._handle_failed_transfer(req_id, handle)

            expected = self._pega_expected_peers.get(req_id, set())
            completed = self._pega_completed_peers.get(req_id, set())
            if not in_progress and expected and completed.issuperset(expected):
                done_req_ids.add(req_id)
                del transfers[req_id]
                self._pega_expected_peers.pop(req_id, None)
                self._pega_submitted_peers.pop(req_id, None)
                self._pega_completed_peers.pop(req_id, None)
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    def _handle_failed_transfer(self, req_id: str, handle: int | None):
        if handle is not None:
            read = self._pega_rdma_reads.pop(handle, None)
            if read is not None:
                self._pega_rdma.release_read(handle)
                self._pega_rdma.invalidate_connection(read.remote_addr)
                handle = None
        self._pega_expected_peers.pop(req_id, None)
        self._pega_submitted_peers.pop(req_id, None)
        self._pega_completed_peers.pop(req_id, None)
        super()._handle_failed_transfer(req_id, handle)

    def shutdown(self):
        for read in list(self._pega_rdma_reads.values()):
            self._pega_rdma.release_read(read.handle)
        self._pega_rdma_reads.clear()
        self._pega_pending_since.clear()
        self._pega_pending_requests.clear()
        self._recving_transfers.clear()
        for agent_name in self._pega_response_agents.values():
            try:
                self.nixl_wrapper.remove_remote_agent(agent_name)
            except Exception:
                logger.debug("Failed to remove Pega RDMA v1 response agent", exc_info=True)
        self._pega_response_agents.clear()
        try:
            self._pega_rdma.unregister_memory(
                [addr for addr, _size, _device_id, _tag in self.local_registered_blocks_data]
            )
        except Exception:
            logger.debug("Failed to unregister Pega RDMA v1 memory", exc_info=True)
        super().shutdown()
