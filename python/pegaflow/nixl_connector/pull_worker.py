# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import zmq
from vllm.logger import init_logger

from pegaflow.nixl_connector.base_worker import (
    NixlBaseConnectorWorker,
)
from pegaflow.nixl_connector.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
    TransferHandle,
)
from pegaflow.nixl_connector.pega_rdma_v1 import (
    PEGA_RDMA_V1_ACCEPT_ACK,
    PEGA_RDMA_V1_ACCEPT_REGISTER,
    PEGA_RDMA_V1_ACCEPT_REQUEST,
    PEGA_RDMA_V1_ACCEPT_RESPONSE,
    PegaRdmaV1Config,
    PegaRdmaV1Perf,
    PegaRdmaV1Read,
    build_handshake_request_extension,
    build_memory_regions,
    create_rdma_engine,
    make_accept_broker_endpoint_from_config,
    make_peer_key,
    parse_handshake_response_extension,
    worker_identity,
)
from pegaflow.nixl_connector.tp_mapping import (
    ReadSpec,
)
from pegaflow.nixl_connector.utils import zmq_ctx

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class _PegaLocalBlockTables:
    """Track NIXL handle objects without exporting Python object ids to Rust."""

    def __init__(self) -> None:
        self._items: list[tuple[object, int]] = []

    def ensure_unregistered(self, handle: object) -> None:
        for existing, existing_table_id in self._items:
            if existing is handle:
                raise RuntimeError(
                    "Pega RDMA v1 local NIXL handle registered twice "
                    f"(table_id={existing_table_id})"
                )

    def register(self, handle: object, table_id: int) -> None:
        self.ensure_unregistered(handle)
        self._items.append((handle, table_id))

    def get(self, handle: object) -> int:
        for existing, table_id in self._items:
            if existing is handle:
                return table_id
        raise KeyError("unknown Pega RDMA v1 local NIXL handle")

    def table_ids(self) -> list[int]:
        return [table_id for _handle, table_id in self._items]

    def clear(self) -> None:
        self._items.clear()


@dataclass(frozen=True)
class PegaReadPlan:
    request_id: str
    dst_engine_id: str
    remote_rank: int
    peer_key: str
    notif_msg: bytes
    local_desc_ids: list[int]
    remote_desc_ids: list[int]
    has_payload: bool


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(meta.local_block_ids)
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
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
                    list(local_block_ids[g]) if rank in plan.source_ranks_per_group[g] else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g]) if rank in plan.source_ranks_per_group[g] else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        # D may have to perform multiple reads from different remote ranks.
        # MLA opt: when P TP > D TP, only a single read is executed for
        # the first remote rank (cache is duplicated)..
        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[remote_block_size]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

        if self.use_mla and tp_ratio < 0 and read_specs:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != read_specs[0].remote_rank:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: object,
        remote_xfer_side_handle: object,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(remote_info.remote_block_size)
        if block_size_ratio > 1:
            # TODO (NickLucche) assume HMA is off. Change to handle multiple KV groups.
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expanded decode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids_mapped = local_block_ids_mapped[: len(remote_block_ids0)]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids, remote_block_ids, remote_physical_per_logical
        )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
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

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)
                if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if self.consumer_notification_counts_by_req[req_id] == consumers_per_producer:
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids


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
        self._pega_rdma_config = PegaRdmaV1Config.from_extra_config(
            extra_config,
            tp_rank=self.tp_rank,
        )
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
        self._pega_local_memory_regions: list[tuple[int, int, int, str]] = []
        self._pega_rdma_reads: dict[TransferHandle, PegaRdmaV1Read] = {}
        self._pega_expected_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_submitted_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_completed_peers: dict[str, set[str]] = defaultdict(set)
        self._pega_local_block_tables = _PegaLocalBlockTables()
        self._pega_accept_stop = None
        self._pega_accept_thread = None
        self._pega_accept_broker_endpoint = make_accept_broker_endpoint_from_config(
            self.engine_id,
            vllm_config,
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
            [(addr, size, device_id) for addr, size, device_id, _ in self._pega_local_memory_regions]
        )
        self._pega_rdma.register_memory(regions)
        logger.info(
            "Pega RDMA v1 registered %d local regions on nics=%s qps_per_peer=%d",
            len(regions),
            ",".join(self._pega_rdma_config.nics),
            self._pega_rdma_config.qps_per_peer,
        )

    def _on_local_memory_registered(
        self,
        regions: list[tuple[int, int, int, str]],
    ) -> None:
        self._pega_local_memory_regions = regions

    def _on_local_blocks_registered(
        self,
        nixl_handle: object,
        blocks_data: list[tuple[int, int, int]],
    ) -> None:
        self._pega_local_block_tables.ensure_unregistered(nixl_handle)
        table_id = self._pega_rdma.register_local_blocks(blocks_data)
        self._pega_local_block_tables.register(nixl_handle, table_id)

    def _on_remote_blocks_registered(
        self,
        engine_id: str,
        tp_rank: int,
        blocks_data: list[tuple[int, int, int]],
    ) -> None:
        self._pega_rdma.register_remote_blocks(engine_id, tp_rank, blocks_data)

    def _cleanup_remote_engine(self, engine_id: str, *, log_eviction: bool = True) -> None:
        """Remove inherited NIXL state and matching Pega RDMA remote block tables."""
        try:
            self._pega_rdma.unregister_remote_blocks(engine_id)
        except Exception:
            logger.debug(
                "Failed to unregister Pega RDMA v1 remote blocks for engine %s",
                engine_id,
                exc_info=True,
            )
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)

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
                request: dict[str, object] | None = None
                try:
                    decoded = msgspec.msgpack.decode(msg)
                    if not isinstance(decoded, dict):
                        raise ValueError("accept request must be a dict")
                    request = decoded
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
                        "kind": PEGA_RDMA_V1_ACCEPT_RESPONSE,
                        "ok": True,
                        "request_id": request_id,
                        "metadata": bytes(response_metadata),
                    }
                except Exception as exc:
                    logger.debug("Pega RDMA v1 local accept failed", exc_info=True)
                    response = {
                        "kind": PEGA_RDMA_V1_ACCEPT_RESPONSE,
                        "ok": False,
                        "request_id": request.get("request_id") if request is not None else None,
                        "error": str(exc),
                    }
                sock.send(msgspec.msgpack.encode(response))

    def _build_handshake_request_extensions(
        self,
        remote_engine_id: str,
        remote_rank: int,
    ) -> dict[str, object]:
        """Attach D-side RDMA metadata to the outgoing NIXL handshake."""
        peer_key = make_peer_key(self.engine_id, self.tp_rank, remote_engine_id, remote_rank)
        status = self._pega_rdma.prepare_handshake(peer_key)
        if status.status == "existing":
            self._pega_rdma_perf.record_ns("rdma_conn_existing", 0)
            return {}
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
        request_extensions: dict[str, object],
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
        request_extensions: dict[str, object],
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
        local_xfer_side_handle: object,
        remote_xfer_side_handle: object,
    ):
        """Submit a pull-mode KV READ through Pega RDMA v1.

        The inherited NIXL worker computes all request/block/TP mapping state.
        This override only changes how the final READ payload is transferred.
        """
        plan = self._build_pega_read_plan(
            read_spec=read_spec,
            dst_engine_id=dst_engine_id,
            request_id=request_id,
            remote_request_id=remote_request_id,
        )
        if not plan.has_payload:
            self._send_empty_pega_read_notification(plan)
            return

        self._submit_pega_read(plan, local_xfer_side_handle)

    def _build_pega_read_plan(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
    ) -> PegaReadPlan:
        """Build NIXL-compatible descriptor ids for a Pega RDMA READ."""
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids
        peer_key = make_peer_key(self.engine_id, self.tp_rank, dst_engine_id, remote_rank)
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

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

        if len(local_block_ids) == 0:
            return PegaReadPlan(
                request_id=request_id,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
                peer_key=peer_key,
                notif_msg=notif_id,
                local_desc_ids=[],
                remote_desc_ids=[],
                has_payload=False,
            )

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

        return PegaReadPlan(
            request_id=request_id,
            dst_engine_id=dst_engine_id,
            remote_rank=remote_rank,
            peer_key=peer_key,
            notif_msg=notif_id,
            local_desc_ids=local_block_descs_ids.tolist(),
            remote_desc_ids=remote_block_descs_ids.tolist(),
            has_payload=True,
        )

    def _send_empty_pega_read_notification(self, plan: PegaReadPlan) -> None:
        """Notify the P worker that a full prefix-cache hit needs no READ."""
        agent_name = self._remote_agents[plan.dst_engine_id][plan.remote_rank]
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=plan.notif_msg)
        except Exception as e:
            self._log_failure(
                failure_type="notification_failed",
                msg="P worker blocks will be freed after timeout.",
                req_id=plan.request_id,
                error=e,
                dst_engine_id=plan.dst_engine_id,
                remote_rank=plan.remote_rank,
                remote_agent_name=agent_name,
            )
            self.xfer_stats.record_failed_notification()

    def _submit_pega_read(
        self,
        plan: PegaReadPlan,
        local_xfer_side_handle: object,
    ) -> None:
        """Submit a Pega RDMA READ for a prepared descriptor plan."""
        handle = None
        try:
            self._pega_expected_peers[plan.request_id].add(plan.peer_key)
            self._require_pega_rdma_connection(plan.peer_key)
            if plan.peer_key in self._pega_submitted_peers[plan.request_id]:
                return

            with self._pega_rdma_perf.measure("read_async_submit"):
                local_table_id = self._pega_local_block_tables.get(local_xfer_side_handle)
                handle, bytes_transferred, num_descriptors = self._pega_rdma.read_async_indices(
                    plan.peer_key,
                    local_table_id,
                    plan.dst_engine_id,
                    plan.remote_rank,
                    plan.local_desc_ids,
                    plan.remote_desc_ids,
                )
            self._pega_rdma_reads[handle] = PegaRdmaV1Read(
                remote_addr=plan.peer_key,
                handle=handle,
                notif_agent=self._remote_agents[plan.dst_engine_id][plan.remote_rank],
                notif_msg=plan.notif_msg,
                submitted_at=time.perf_counter(),
                bytes_transferred=bytes_transferred,
                num_descriptors=num_descriptors,
            )
            self._recving_transfers[plan.request_id].append(handle)
            self._pega_submitted_peers[plan.request_id].add(plan.peer_key)
            logger.debug(
                "Pega RDMA v1 READ submitted req=%s remote=%s rank=%s descs=%d",
                plan.request_id,
                plan.dst_engine_id,
                plan.remote_rank,
                num_descriptors,
            )
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=plan.request_id,
                msg="Pega RDMA v1 READ setup failed; marking blocks invalid",
                error=e,
                dst_engine_id=plan.dst_engine_id,
                remote_rank=plan.remote_rank,
            )
            self._handle_failed_transfer(plan.request_id, handle)

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
            failed = False
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
                        failed = True
                        break
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        req_id=req_id,
                        msg="Pega RDMA v1 READ failed",
                        error=e,
                    )
                    self._handle_failed_transfer(req_id, handle)
                    failed = True
                    break

            if failed:
                continue

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
        handles_to_release = set(self._recving_transfers.pop(req_id, []))
        if handle is not None:
            handles_to_release.add(handle)
        for pending_handle in handles_to_release:
            read = self._pega_rdma_reads.pop(pending_handle, None)
            if read is not None:
                self._pega_rdma.release_read(pending_handle)
                self._pega_rdma.invalidate_connection(read.remote_addr)
        self._pega_expected_peers.pop(req_id, None)
        self._pega_submitted_peers.pop(req_id, None)
        self._pega_completed_peers.pop(req_id, None)
        super()._handle_failed_transfer(req_id, None)

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
        for table_id in self._pega_local_block_tables.table_ids():
            try:
                self._pega_rdma.unregister_local_blocks(table_id)
            except Exception:
                logger.debug(
                    "Failed to unregister Pega RDMA v1 local blocks for table %s",
                    table_id,
                    exc_info=True,
                )
        self._pega_local_block_tables.clear()
        try:
            self._pega_rdma.unregister_memory(
                [addr for addr, _size, _device_id, _tag in self._pega_local_memory_regions]
            )
        except Exception:
            logger.debug("Failed to unregister Pega RDMA v1 memory", exc_info=True)
        super().shutdown()
