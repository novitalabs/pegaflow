# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PegaFlow RDMA-backed push worker for the Pega NIXL connector."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology

from pegaflow.nixl_connector.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlHandshakePayload,
    RemoteMeta,
    ReqId,
    ReqMeta,
    compute_nixl_compatibility_hash,
)
from pegaflow.nixl_connector.push_worker import NixlPushConnectorWorker
from pegaflow.nixl_connector.rdma_transport import (
    PegaNixlRdmaTransport,
    handshake_from_wire,
    handshake_to_wire,
)
from pegaflow.nixl_connector.tp_mapping import compute_tp_mapping
from pegaflow.nixl_connector.utils import get_base_request_id, zmq_ctx
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


class PegaNixlPushConnectorWorker(NixlPushConnectorWorker):
    """NIXL push worker with PegaFlow RDMA as the data plane."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._use_pega_rdma_transport = True
        super().__init__(*args, **kwargs)
        self.pega_rdma = PegaNixlRdmaTransport(
            vllm_config=self.vllm_config,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
        )
        self._next_imm_id = 1
        self._finished_rdma_waits: set[ReqId] = set()
        self._rdma_wait_lock = threading.Lock()
        self._rdma_wait_executor = ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="pega-nixl-rdma-done-waiter",
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
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=b"pega-rdma",
        )
        self._ensure_push_writer_started()

    def _ensure_push_writer_started(self) -> None:
        if self._push_writer_thread is not None:
            return
        self._push_writer_thread = threading.Thread(
            target=self._push_writer_loop,
            daemon=True,
            name="pega-nixl-push-writer",
        )
        self._push_writer_thread.start()
        logger.info("pega-nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def shutdown(self):
        self._push_writer_stop.set()
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        self._rdma_wait_executor.shutdown(wait=False, cancel_futures=True)
        self._handshake_initiation_executor.shutdown(wait=False, cancel_futures=True)

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        reg_data = dict(reg_data)
        imm_id = self._alloc_imm_id()
        local_block_ids = self._as_grouped_block_ids(reg_data["local_block_ids"])
        handshake = self.pega_rdma.build_local_handshake(
            req_id,
            local_block_ids,
            imm_id=imm_id,
            fail_imm_id=_fail_imm_id(imm_id),
            abort_imm_id=_abort_imm_id(imm_id),
        )
        self.pega_rdma.open_request(req_id, handshake)
        self._submit_rdma_wait(req_id)
        reg_data["pega_rdma_handshake"] = handshake_to_wire(
            handshake
        )
        self._do_send_reg_notif(req_id, reg_data)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        path = make_zmq_path(
            "tcp",
            reg_data["remote_host"],
            int(reg_data["remote_port"]),
        )
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        try:
            with zmq_ctx(zmq.REQ, path) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.setsockopt(zmq.SNDTIMEO, 5000)
                sock.send(msgspec.msgpack.encode((PUSH_REG_NOTIF_PREFIX, notif_msg)))
                ack = sock.recv()
                if ack != b"OK":
                    raise RuntimeError(f"Pega PUSH_REG side-channel ack={ack!r}")
        except Exception as e:
            self._log_failure(
                failure_type="pega_push_reg_side_channel_failed",
                req_id=req_id,
                error=e,
                remote_engine_id=reg_data.get("remote_engine_id"),
                remote_host=reg_data.get("remote_host"),
                remote_port=reg_data.get("remote_port"),
            )
            self._handle_failed_transfer(req_id, None)
            return
        logger.debug(
            "Sent Pega NIXL RDMA PUSH_REG for %s to %s (%dB)",
            req_id,
            path,
            len(notif_msg),
        )

    def _ensure_d_handshake(
        self,
        decode_engine_id: str,
        decode_host: str,
        decode_port: int,
        decode_tp_size: int,
        request_id: str,
    ) -> bool:
        return True

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None
        reg_data = getattr(meta, "registration_data", None)
        if not isinstance(reg_data, dict):
            raise RuntimeError(f"missing Pega RDMA registration data for {req_id}")
        handshake = handshake_from_wire(reg_data["pega_rdma_handshake"])
        if not self._should_push_to_decode_rank(reg_data):
            logger.debug(
                "Skipping Pega NIXL RDMA push req=%s from rank=%d to decode rank=%d",
                req_id,
                self.tp_rank,
                handshake.tp_rank,
            )
            return
        self.pega_rdma.push_blocks(
            request_id=req_id,
            remote_handshake=handshake,
            local_block_ids=meta.local_physical_block_ids,
            remote_block_ids=self._logical_to_kernel_block_ids(
                self._as_grouped_block_ids(reg_data["local_block_ids"])
            ),
        )

    def _get_new_notifs(self) -> set[str]:
        return set()

    def get_finished(self) -> tuple[set[str], set[str]]:
        self._push_writer_wake.set()
        done_sending = set[ReqId]()
        done_recving = self.pega_rdma.pop_finished_recving()
        with self._rdma_wait_lock:
            done_recving.update(self._finished_rdma_waits)
            self._finished_rdma_waits.clear()
        while not self._failed_recv_reqs.empty():
            try:
                done_recving.add(self._failed_recv_reqs.get_nowait())
            except Exception:
                break
        done_sending.update(self.pega_rdma.pop_finished_sending())
        for req_id in list(done_recving):
            self._recving_metadata.pop(req_id, None)
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            if now < expires:
                break
            self._reqs_to_process.discard(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)
        for req_id in done_sending | done_recving:
            self.pega_rdma.close_request(req_id)
        return done_sending, done_recving

    @staticmethod
    def _attach_registration(meta: ReqMeta, registration_data: dict[str, Any]) -> ReqMeta:
        setattr(meta, "registration_data", registration_data)
        return meta

    def start_load_kv(self, metadata):
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            self._recving_metadata[req_id] = meta

        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        for reg_key, reg_data in metadata.push_incoming_registrations.items():
            if not self._should_push_to_decode_rank(reg_data):
                continue
            decode_req_id = reg_data["request_id"]
            self._pending_d_registrations[reg_key] = reg_data
            match = self._pop_matching_finished_blocks(decode_req_id)
            if match is not None:
                self._pending_d_registrations.pop(reg_key, None)
                fin_id, blocks = match
                self._do_start_push_kv(fin_id, blocks, reg_data)
        if metadata.push_incoming_registrations:
            self._push_writer_wake.set()

        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

    def _push_writer_loop(self) -> None:
        sleep_s = 0.001
        while not self._push_writer_stop.is_set():
            try:
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
            except Exception:
                logger.exception("pega-nixl-push-writer error; continuing")

            if self._push_finished_blocks:
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: Any,
        registration_data: dict[str, Any],
    ) -> None:
        if "pega_rdma_handshake" not in registration_data:
            logger.error("Pega NIXL registration missing RDMA handshake for %s", request_id)
            self._handle_failed_transfer(request_id, None)
            return

        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        logical_local = self._as_grouped_block_ids(local_block_ids)
        physical_local = self._logical_to_kernel_block_ids(logical_local)
        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=self._as_grouped_block_ids(remote_block_ids),
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )
        self._attach_registration(push_meta, registration_data)
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        base_id = get_base_request_id(request_id)
        for reg_key, reg_data in list(self._pending_d_registrations.items()):
            decode_req_id = reg_data.get("request_id")
            if (
                decode_req_id == request_id
                or get_base_request_id(decode_req_id) == base_id
            ):
                return self._pending_d_registrations.pop(reg_key)
        return super()._pop_matching_registration(request_id)

    def _should_push_to_decode_rank(self, registration_data: dict[str, Any]) -> bool:
        handshake = handshake_from_wire(registration_data["pega_rdma_handshake"])
        decode_tp_size = int(registration_data.get("decode_tp_size", handshake.tp_size))
        if self.transfer_topo is None:
            return handshake.tp_rank == self.tp_rank
        plan = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=decode_tp_size,
            group_spec_types=self._group_spec_types,
        )
        return handshake.tp_rank in plan.all_source_ranks

    def _submit_rdma_wait(self, req_id: str) -> None:
        self._rdma_wait_executor.submit(self._wait_done, req_id)

    def _wait_done(self, req_id: str) -> None:
        try:
            self.pega_rdma.wait_done(req_id)
            with self._rdma_wait_lock:
                self._finished_rdma_waits.add(req_id)
        except Exception as e:
            self._log_failure(
                failure_type="pega_rdma_wait_done_failed",
                req_id=req_id,
                error=e,
            )
            self._handle_failed_transfer(req_id, None)

    def _alloc_imm_id(self) -> int:
        imm_id = self._next_imm_id
        self._next_imm_id += 1
        if self._next_imm_id > 0x3FFF_FFFF:
            self._next_imm_id = 1
        return imm_id


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int:
    for tensor in kv_caches.values():
        get_device = getattr(tensor, "get_device", None)
        if get_device is None:
            continue
        try:
            return max(int(get_device()), 0)
        except RuntimeError:
            continue
    return 0


def _fail_imm_id(imm_id: int) -> int:
    return imm_id ^ 0x8000_0000


def _abort_imm_id(imm_id: int) -> int:
    return imm_id ^ 0x4000_0000
