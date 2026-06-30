# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""PegaFlow RDMA v1 data-plane worker for NIXL pull mode."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import zmq
from vllm import envs
from vllm.logger import init_logger

from pegaflow.nixl_connector.metadata import TransferHandle
from pegaflow.nixl_connector.pega_rdma_v1 import (
    PEGA_RDMA_V1_ACCEPT_ACK,
    PEGA_RDMA_V1_ACCEPT_REGISTER,
    PEGA_RDMA_V1_ACCEPT_REQUEST,
    PegaRdmaV1Config,
    PegaRdmaV1Perf,
    PegaRdmaV1Read,
    build_handshake_request_extension,
    build_memory_regions,
    create_rdma_engine,
    make_accept_broker_endpoint,
    make_peer_key,
    parse_handshake_response_extension,
    worker_identity,
)
from pegaflow.nixl_connector.pull_worker import NixlPullConnectorWorker
from pegaflow.nixl_connector.tp_mapping import ReadSpec
from pegaflow.nixl_connector.utils import zmq_ctx

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
        self._pega_rdma_perf = PegaRdmaV1Perf(
            enabled=self._pega_rdma_config.perf_enabled,
            log_every=self._pega_rdma_config.perf_log_every,
            logger_name=__name__,
        )
        if self._pega_rdma_config.perf_enabled:
            logger.warning(
                "Pega RDMA v1 perf probes enabled; log_every=%d",
                self._pega_rdma_config.perf_log_every,
            )
        self._pega_rdma_reads: dict[TransferHandle, PegaRdmaV1Read] = {}
        self._pega_block_tables: dict[object, int] = {}
        self._pega_expected_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_submitted_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_completed_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_accept_stop = None
        self._pega_accept_thread = None
        side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        self._pega_accept_broker_endpoint = make_accept_broker_endpoint(
            self.engine_id,
            side_channel_port,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register NIXL KV caches, then register the same buffers for RDMA v1."""
        super().register_kv_caches(kv_caches)
        self._register_pega_rdma_memory()
        self._start_pega_rdma_accept_service()

    def _register_pega_rdma_memory(self) -> None:
        """Register locally exported KV blocks with the Pega RDMA v1 engine."""
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

    def _start_pega_rdma_accept_service(self) -> None:
        """Connect to the scheduler-owned broker for worker-owned RDMA accepts."""
        if self._pega_accept_thread is not None:
            return
        import threading

        stop_event = threading.Event()
        ready_event = threading.Event()
        thread = threading.Thread(
            target=self._pega_rdma_accept_loop,
            args=(self._pega_accept_broker_endpoint, ready_event, stop_event),
            daemon=True,
            name=f"pega-rdma-v1-accept-{self.engine_id}-{self.tp_rank}",
        )
        thread.start()
        ready_event.wait(timeout=1.0)
        self._pega_accept_stop = stop_event
        self._pega_accept_thread = thread

    def _pega_rdma_accept_loop(self, endpoint, ready_event, stop_event) -> None:
        """Serve scheduler broker requests that need worker-owned RDMA accept."""
        identity = worker_identity(self.engine_id, self.tp_rank)
        with zmq_ctx(zmq.DEALER, endpoint, identity=identity) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.setsockopt(zmq.SNDTIMEO, int(self._pega_rdma_config.handshake_timeout_s * 1000))
            ready_event.set()
            register_msg = msgspec.msgpack.encode(
                {
                    "kind": PEGA_RDMA_V1_ACCEPT_REGISTER,
                    "tp_rank": self.tp_rank,
                }
            )
            while not stop_event.is_set():
                try:
                    sock.send(register_msg)
                    response = msgspec.msgpack.decode(sock.recv())
                    if (
                        isinstance(response, dict)
                        and response.get("ok")
                        and response.get("kind") == PEGA_RDMA_V1_ACCEPT_ACK
                    ):
                        break
                except zmq.Again:
                    pass
                time.sleep(0.05)
            while not stop_event.is_set():
                try:
                    msg = sock.recv()
                except zmq.Again:
                    continue
                try:
                    request = msgspec.msgpack.decode(msg)
                    if not isinstance(request, dict):
                        raise ValueError("accept request must be a dict")
                    if request.get("kind") != PEGA_RDMA_V1_ACCEPT_REQUEST:
                        raise ValueError(f"unexpected accept request kind {request.get('kind')!r}")
                    peer_key = request.get("peer_key")
                    metadata = request.get("metadata")
                    request_id = request.get("request_id")
                    if not isinstance(peer_key, str) or not isinstance(metadata, bytes):
                        raise ValueError("accept request missing peer_key/metadata")
                    with self._pega_rdma_perf.measure("handle_hs_request"):
                        response_metadata = self._pega_rdma.accept_handshake(peer_key, metadata)
                    response = {
                        "ok": True,
                        "request_id": request_id,
                        "metadata": bytes(response_metadata),
                    }
                except Exception as exc:
                    logger.debug("Pega RDMA v1 local accept failed", exc_info=True)
                    response = {
                        "ok": False,
                        "request_id": request.get("request_id") if isinstance(request, dict) else None,
                        "error": str(exc),
                    }
                sock.send(msgspec.msgpack.encode(response))

    def _build_handshake_request_extensions(
        self,
        remote_engine_id: str,
        remote_rank: int,
    ) -> dict[str, object] | None:
        """Attach D-side RDMA metadata to the outgoing NIXL handshake."""
        peer_key = make_peer_key(self.engine_id, self.tp_rank, remote_engine_id, remote_rank)
        status = self._pega_rdma.prepare_handshake(peer_key)
        if status.status == "existing":
            self._pega_rdma_perf.record_ns("rdma_conn_existing", 0)
            return None
        if status.status == "connecting":
            self._pega_rdma_perf.record_ns("rdma_conn_connecting", 0)
            raise RuntimeError(f"RDMA v1 handshake already in progress for {peer_key}")
        if status.metadata is None:
            raise RuntimeError(f"RDMA v1 prepare_handshake({peer_key}) returned no metadata")
        self._pega_rdma_perf.record_ns("rdma_conn_prepared", 0)
        return build_handshake_request_extension(peer_key, bytes(status.metadata))

    def _handle_handshake_response_extensions(
        self,
        remote_engine_id: str,
        remote_rank: int,
        response_extensions: dict[str, object] | None,
        request_extensions: dict[str, object] | None,
    ) -> None:
        """Finish RDMA setup from metadata folded into the NIXL response."""
        response_metadata = parse_handshake_response_extension(response_extensions)
        if response_metadata is None:
            if request_extensions:
                raise RuntimeError(
                    "Pega RDMA v1 handshake response did not include RDMA metadata"
                )
            return
        peer_key = make_peer_key(self.engine_id, self.tp_rank, remote_engine_id, remote_rank)
        with self._pega_rdma_perf.measure("handle_hs_response"):
            self._pega_rdma.finish_handshake(peer_key, response_metadata)

    def _handle_handshake_request_failure(
        self,
        remote_engine_id: str,
        remote_rank: int,
        request_extensions: dict[str, object] | None,
    ) -> None:
        """Abort locally prepared RDMA metadata when NIXL handshake fails."""
        if not request_extensions:
            return
        peer_key = make_peer_key(self.engine_id, self.tp_rank, remote_engine_id, remote_rank)
        self._pega_rdma.abort_handshake(peer_key)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """Submit a pull-mode KV READ through Pega RDMA v1.

        The inherited NIXL worker computes all request/block/TP mapping state.
        This override only changes how the final READ payload is transferred.
        """
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
        # Keep NIXL's mapping contract intact: the connector computes logical
        # request blocks, prefix-cache trims, TP ratios, and physical KV block
        # descriptors exactly as the vendored pull worker does.  Pega RDMA only
        # consumes the resulting descriptor indices to submit the READ payload.
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
            self._require_pega_rdma_connection(peer_key)
            if peer_key in self._pega_submitted_peers[request_id]:
                return

            # Fast path: register stable local/remote block tables once, then
            # pass compact descriptor indices into Rust for every READ.  This
            # avoids rebuilding Python dict descriptors in the hot path while
            # preserving NIXL's block ordering and TP-rank decisions above.
            local_table = self._local_blocks_table_for_handle(
                local_xfer_side_handle,
                block_size=remote_info.remote_block_size,
            )
            remote_table = self._remote_blocks_table(dst_engine_id, remote_rank)
            with self._pega_rdma_perf.measure("read_async_submit"):
                handle, bytes_transferred, num_descriptors = self._pega_rdma.read_async_indices(
                    peer_key,
                    local_table,
                    remote_table,
                    local_block_descs_ids.tolist(),
                    remote_block_descs_ids.tolist(),
                )
            self._pega_rdma_reads[handle] = PegaRdmaV1Read(
                remote_addr=peer_key,
                handle=handle,
                notif_agent=self._remote_agents[dst_engine_id][remote_rank],
                notif_msg=notif_id,
                submitted_at=time.perf_counter(),
                bytes_transferred=bytes_transferred,
                num_descriptors=num_descriptors,
            )
            self._recving_transfers[request_id].append(handle)
            self._pega_submitted_peers[request_id].add(peer_key)
            logger.debug(
                "Pega RDMA v1 READ submitted req=%s remote=%s rank=%s descs=%d",
                request_id,
                dst_engine_id,
                remote_rank,
                num_descriptors,
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
        """Resolve the local NIXL transfer handle to registered block data."""
        if local_xfer_side_handle == self.src_xfer_handles_by_block_size[block_size]:
            return self.src_blocks_data_by_block_size[block_size]
        for tp_ratio, handles in self.src_xfer_handles_by_tp_ratio.items():
            for idx, handle in enumerate(handles):
                if handle == local_xfer_side_handle:
                    return self.src_blocks_data_by_tp_ratio[tp_ratio][idx]
        raise RuntimeError(f"unknown local xfer side handle {local_xfer_side_handle}")

    def _local_blocks_table_for_handle(
        self,
        local_xfer_side_handle: int,
        *,
        block_size: int,
    ) -> int:
        """Return a cached native block-table handle for local KV blocks."""
        key = ("local", local_xfer_side_handle)
        table = self._pega_block_tables.get(key)
        if table is not None:
            return table
        blocks = self._local_blocks_data_for_handle(local_xfer_side_handle, block_size=block_size)
        table = self._pega_rdma.register_blocks_table(blocks)
        self._pega_block_tables[key] = table
        return table

    def _remote_blocks_table(self, dst_engine_id: str, remote_rank: int) -> int:
        """Return a cached native block-table handle for remote KV blocks."""
        key = ("remote", dst_engine_id, remote_rank)
        table = self._pega_block_tables.get(key)
        if table is not None:
            return table
        blocks = self.dst_blocks_data[dst_engine_id][remote_rank]
        table = self._pega_rdma.register_blocks_table(blocks)
        self._pega_block_tables[key] = table
        return table

    def _require_pega_rdma_connection(self, peer_key: str) -> None:
        """Require the RDMA connection created by the initial NIXL handshake."""
        status = self._pega_rdma.prepare_handshake(peer_key)
        if status.status == "existing":
            self._pega_rdma_perf.record_ns("rdma_conn_existing", 0)
            return
        if status.status == "connecting":
            self._pega_rdma_perf.record_ns("rdma_conn_connecting", 0)
            raise RuntimeError(f"RDMA v1 handshake already in progress for {peer_key}")
        self._pega_rdma_perf.record_ns("rdma_conn_missing", 0)
        self._pega_rdma.abort_handshake(peer_key)
        raise RuntimeError(
            f"Pega RDMA v1 connection for {peer_key} is missing after NIXL handshake"
        )

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """Poll Pega RDMA READs and emit normal NIXL completion notifications."""
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                read = self._pega_rdma_reads.get(handle)
                if read is None:
                    continue
                try:
                    with self._pega_rdma_perf.measure("check_read_poll"):
                        state = self._pega_rdma.check_read(handle)
                    if state == "done":
                        with self._pega_rdma_perf.measure("read_complete_to_notif"):
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
            # A request may require READs from multiple remote TP ranks.  The
            # request is visible to the inherited scheduler only after every
            # expected Pega RDMA peer has completed and sent its normal NIXL
            # completion notification.
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
        """Release Pega RDMA state, then delegate request failure to NIXL."""
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
        """Release Pega RDMA resources before the inherited NIXL shutdown."""
        self._pega_rdma_perf.log_final_summary()
        if self._pega_accept_stop is not None:
            self._pega_accept_stop.set()
        if self._pega_accept_thread is not None:
            self._pega_accept_thread.join(timeout=2.0)
        self._pega_accept_stop = None
        self._pega_accept_thread = None
        for read in list(self._pega_rdma_reads.values()):
            self._pega_rdma.release_read(read.handle)
        self._pega_rdma_reads.clear()
        self._recving_transfers.clear()
        for table in self._pega_block_tables.values():
            try:
                self._pega_rdma.drop_blocks_table(table)
            except Exception:
                logger.debug("Failed to drop Pega RDMA v1 block table", exc_info=True)
        self._pega_block_tables.clear()
        try:
            self._pega_rdma.unregister_memory(
                [addr for addr, _size, _device_id, _tag in self.local_registered_blocks_data]
            )
        except Exception:
            logger.debug("Failed to unregister Pega RDMA v1 memory", exc_info=True)
        super().shutdown()
