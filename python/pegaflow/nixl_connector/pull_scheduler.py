# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Pull-specific scheduler-side logic for the NIXL connector."""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.logger import init_logger

from pegaflow.nixl_connector.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from pegaflow.nixl_connector.metadata import NixlHandshakePayload
from pegaflow.nixl_connector.pull_worker import (
    PEGA_RDMA_V1_ACCEPT_ACK,
    PEGA_RDMA_V1_ACCEPT_ENDPOINT,
    PEGA_RDMA_V1_ACCEPT_REGISTER,
    PEGA_RDMA_V1_ACCEPT_REQUEST,
    PEGA_RDMA_V1_ACCEPT_RESPONSE,
    PEGA_RDMA_V1_EXTENSION,
    PegaRdmaV1BrokerState,
    PegaRdmaV1Config,
    accept_handshake_via_zmq,
    make_accept_broker_endpoint_from_config,
    reverse_peer_key,
)
from pegaflow.nixl_connector.utils import zmq_ctx

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._mamba_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # The tokens will be loaded if not already present
            # in the prefill node local cache
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            if count > 0:
                # Check kv_recompute_threshold: skip pull if
                # remote tokens are below the threshold.
                if self.kv_recompute_threshold > 0 and count < self.kv_recompute_threshold:
                    logger.debug(
                        "Skipping remote pull for %s: %d remote tokens < threshold %d",
                        request.request_id,
                        count,
                        self.kv_recompute_threshold,
                    )
                    return 0, False
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the local node. We need to call
                    # send_notif in _read_blocks to free the memory on the remote node.

                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_sw_clipped_blocks(unhashed_local_block_ids)

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                # For blocks pinned on D, use a simpler timeout for now instead of a
                # lease mechanism as turn2 request is client-driven.
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            self._reqs_need_send[request.request_id] = time.perf_counter() + request_kv_blocks_ttl
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            block_ids = self.get_sw_clipped_blocks(block_ids)

            remote_num_tokens = request.num_computed_tokens

        return delay_free_blocks, {
            "do_remote_prefill": is_p_node,
            "do_remote_decode": is_d_node,
            "remote_block_ids": block_ids,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "tp_size": self.vllm_config.parallel_config.tensor_parallel_size,
            "remote_num_tokens": remote_num_tokens,
        }


class PegaNixlPullConnectorScheduler(NixlPullConnectorScheduler):
    """Fold Pega RDMA v1 accept metadata into NIXL's GET_META response."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._pega_rdma_config = PegaRdmaV1Config.from_extra_config(extra_config)
        self._pega_accept_broker_endpoint = make_accept_broker_endpoint_from_config(
            self.engine_id,
            self.vllm_config,
        )
        self._pega_accept_broker_stop = threading.Event()
        self._pega_accept_broker_thread: threading.Thread | None = None

    def set_xfer_handshake_metadata(self, metadata):
        """Advertise one scheduler-owned broker endpoint to all remote workers."""
        self._start_pega_rdma_accept_broker()
        for payload in metadata.values():
            if not isinstance(payload, NixlHandshakePayload):
                continue
            extensions = dict(payload.extensions or {})
            extensions[PEGA_RDMA_V1_ACCEPT_ENDPOINT] = self._pega_accept_broker_endpoint
            payload.extensions = extensions
        super().set_xfer_handshake_metadata(metadata)

    def _start_pega_rdma_accept_broker(self) -> None:
        """Start the scheduler-owned ROUTER that workers register with."""
        if self._pega_accept_broker_thread is not None:
            return
        if self._pega_accept_broker_endpoint.startswith("ipc://"):
            with suppress(FileNotFoundError):
                os.unlink(self._pega_accept_broker_endpoint[len("ipc://") :])
        ready_event = threading.Event()
        thread = threading.Thread(
            target=self._pega_rdma_accept_broker_loop,
            args=(
                self._pega_accept_broker_endpoint,
                ready_event,
                self._pega_accept_broker_stop,
            ),
            daemon=True,
            name=f"pega-rdma-v1-accept-broker-{self.engine_id}",
        )
        thread.start()
        if not ready_event.wait(timeout=self._pega_rdma_config.handshake_timeout_s):
            self._pega_accept_broker_stop.set()
            raise RuntimeError(
                f"Pega RDMA v1 accept broker failed to bind " f"{self._pega_accept_broker_endpoint}"
            )
        self._pega_accept_broker_thread = thread

    def _pega_rdma_accept_broker_loop(
        self,
        endpoint: str,
        ready_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        """Route scheduler accept requests to the worker owning target TP rank."""
        state = PegaRdmaV1BrokerState(self._pega_rdma_config.handshake_timeout_s)
        with zmq_ctx(zmq.ROUTER, endpoint) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 100)
            ready_event.set()
            while not stop_event.is_set():
                now = time.monotonic()
                for client_reply_prefix, request_id in state.pop_timed_out(now):
                    response = {
                        "kind": PEGA_RDMA_V1_ACCEPT_RESPONSE,
                        "request_id": request_id,
                        "ok": False,
                        "error": "Pega RDMA v1 worker accept timed out",
                    }
                    sock.send_multipart((*client_reply_prefix, msgspec.msgpack.encode(response)))
                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    continue
                try:
                    identity, reply_prefix, msg = self._decode_broker_frames(frames)
                except ValueError:
                    logger.debug("Pega RDMA v1 broker got malformed frames: %s", frames)
                    continue
                try:
                    payload = msgspec.msgpack.decode(msg)
                    if not isinstance(payload, dict):
                        raise ValueError("broker payload must be a dict")
                    kind = payload.get("kind")
                    if kind == PEGA_RDMA_V1_ACCEPT_REGISTER:
                        tp_rank = payload.get("tp_rank")
                        if not isinstance(tp_rank, int):
                            raise ValueError("worker register missing tp_rank")
                        queued_payloads = state.register_worker(tp_rank, identity)
                        sock.send_multipart(
                            (
                                identity,
                                msgspec.msgpack.encode(
                                    {"ok": True, "kind": PEGA_RDMA_V1_ACCEPT_ACK}
                                ),
                            )
                        )
                        for payload_to_send in queued_payloads:
                            sock.send_multipart((identity, msgspec.msgpack.encode(payload_to_send)))
                    elif kind == PEGA_RDMA_V1_ACCEPT_RESPONSE:
                        request_id = payload.get("request_id")
                        if not isinstance(request_id, int):
                            raise ValueError("worker response missing request_id")
                        client_reply_prefix = state.complete_request(request_id)
                        if client_reply_prefix is None:
                            logger.warning(
                                "Pega RDMA v1 broker got response for unknown request %s",
                                request_id,
                            )
                            continue
                        sock.send_multipart((*client_reply_prefix, msgspec.msgpack.encode(payload)))
                    elif kind == PEGA_RDMA_V1_ACCEPT_REQUEST:
                        target_tp_rank = payload.get("target_tp_rank")
                        if not isinstance(target_tp_rank, int):
                            raise ValueError("accept request missing target_tp_rank")
                        worker, payload_to_send = state.add_request(
                            reply_prefix=reply_prefix,
                            target_tp_rank=target_tp_rank,
                            payload=payload,
                            now=time.monotonic(),
                        )
                        if worker is not None:
                            sock.send_multipart((worker, msgspec.msgpack.encode(payload_to_send)))
                    else:
                        raise ValueError(f"unexpected broker payload kind {kind!r}")
                except Exception as exc:
                    logger.debug("Pega RDMA v1 broker request failed", exc_info=True)
                    response = {"ok": False, "error": str(exc)}
                    sock.send_multipart((*reply_prefix, msgspec.msgpack.encode(response)))

    @staticmethod
    def _decode_broker_frames(frames: list[bytes]) -> tuple[bytes, tuple[bytes, ...], bytes]:
        """Decode ROUTER frames from either a REQ client or DEALER worker.

        ``accept_handshake_via_zmq`` uses REQ, so ROUTER receives
        ``identity, empty, payload`` and must include the empty delimiter when
        replying.  Worker threads use DEALER identities and exchange a single
        payload frame after the identity.  Keeping those shapes separate avoids
        leaking an empty frame into worker ``recv()``.
        """
        if len(frames) == 2:
            identity, msg = frames
            return identity, (identity,), msg
        if len(frames) == 3 and frames[1] == b"":
            identity, _empty, msg = frames
            return identity, (identity, b""), msg
        raise ValueError(f"unexpected ROUTER frame shape with {len(frames)} frames")

    def _handle_handshake_extensions(
        self,
        target_tp_rank: int,
        payload: NixlHandshakePayload,
        request_extensions: dict[str, Any],
    ) -> NixlHandshakePayload:
        """Accept D-side RDMA metadata before replying to NIXL GET_META."""
        rdma_request = request_extensions.get(PEGA_RDMA_V1_EXTENSION)
        if rdma_request is None:
            return payload
        if not isinstance(rdma_request, dict):
            raise ValueError("Pega RDMA v1 request extension must be a dict")

        peer_key = rdma_request.get("peer_key")
        metadata = rdma_request.get("metadata")
        if not isinstance(peer_key, str) or not isinstance(metadata, bytes):
            raise ValueError("Pega RDMA v1 request extension missing peer_key/metadata")

        endpoint = (payload.extensions or {}).get(PEGA_RDMA_V1_ACCEPT_ENDPOINT)
        if not isinstance(endpoint, str):
            raise RuntimeError(
                "Pega RDMA v1 accept endpoint missing from target rank handshake payload"
            )

        timeout_ms = int(self._pega_rdma_config.handshake_timeout_s * 1000)
        response_extensions = dict(payload.extensions or {})
        try:
            response_metadata = accept_handshake_via_zmq(
                endpoint,
                target_tp_rank,
                reverse_peer_key(peer_key),
                metadata,
                timeout_ms,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "metadata": response_metadata,
            }
        except Exception as exc:
            logger.warning(
                "Failed to accept Pega RDMA v1 handshake target_rank=%s peer=%s: %s",
                target_tp_rank,
                peer_key,
                exc,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "error": str(exc),
            }
        logger.debug(
            "Accepted Pega RDMA v1 handshake target_rank=%s peer=%s",
            target_tp_rank,
            peer_key,
        )
        return NixlHandshakePayload(
            compatibility_hash=payload.compatibility_hash,
            agent_metadata_bytes=payload.agent_metadata_bytes,
            extensions=response_extensions,
        )

    def shutdown(self):
        """Stop the Pega RDMA broker before the base scheduler listener exits."""
        self._pega_accept_broker_stop.set()
        if self._pega_accept_broker_thread is not None:
            self._pega_accept_broker_thread.join(timeout=2.0)
            self._pega_accept_broker_thread = None
        if self._pega_accept_broker_endpoint.startswith("ipc://"):
            with suppress(FileNotFoundError):
                os.unlink(self._pega_accept_broker_endpoint[len("ipc://") :])
        super().shutdown()
