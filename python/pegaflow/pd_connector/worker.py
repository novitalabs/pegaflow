"""Worker-side logic for the experimental P/D connector.

PdWorkerConnector is the public facade. It composes:
- DecodeHandler  (D side — receives KV via RDMA)
- PrefillHandler (P side — pushes KV via RDMA)
"""

from __future__ import annotations

from typing import Any

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.decode_worker import DecodeHandler
from pegaflow.pd_connector.layout import FlashAttnHndLayout
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdWorkerMetadata,
)
from pegaflow.pd_connector.prefill_worker import PrefillHandler
from pegaflow.pd_connector.rdma import RdmaPort, build_rdma_port

logger = get_connector_logger()


class PdWorkerConnector:
    def __init__(
        self,
        vllm_config: Any,
        rdma: RdmaPort | None = None,
        prefill_sender: Any | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rdma = rdma
        self._rdma_is_injected = rdma is not None
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""
        self.tp_rank, self.tp_size = _tensor_parallel_identity(vllm_config)
        logger.info(
            "[PdConnector] worker initialized engine=%s tp_rank=%d tp_size=%d",
            self.engine_id,
            self.tp_rank,
            self.tp_size,
        )
        self.layouts: dict[str, FlashAttnHndLayout] = {}
        self.layer_names: list[str] = []
        self._registered_layers: dict[str, LayerRemoteLayout] = {}
        self._forward_step_id = 0

        self._decode = DecodeHandler(self, prefill_sender=prefill_sender)
        self._prefill = PrefillHandler(self)

    # ------------------------------------------------------------------
    # Backward-compatible attribute access for tests / internal callers
    # ------------------------------------------------------------------

    @property
    def _wait_reqs(self) -> dict:
        return self._decode._wait_reqs

    @_wait_reqs.setter
    def _wait_reqs(self, value: dict) -> None:
        self._decode._wait_reqs = value

    @property
    def _push_reqs(self) -> dict:
        return self._prefill._push_reqs

    @property
    def _push_sender(self):
        return self._prefill._push_sender

    @property
    def _push_finalizer(self):
        return self._prefill._push_finalizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.layouts = {
            layer_name: FlashAttnHndLayout.from_tensor(layer_name, tensor)
            for layer_name, tensor in kv_caches.items()
        }
        self.layer_names = list(kv_caches.keys())
        if not self._rdma_is_injected:
            self.rdma = build_rdma_port(
                self.vllm_config,
                _infer_cuda_device(kv_caches),
                tp_rank=self.tp_rank,
            )
            self._decode.init_rdma_waiter()
        assert self.rdma is not None
        registered_layers = self.rdma.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        self._registered_layers = {layer.layer_name: layer for layer in registered_layers}
        self._decode.gather_peer_info()
        logger.info(
            "[PdConnector] registered %d FlashAttention HND KV cache layers, gathered %d peer ranks",
            len(self.layouts),
            len(self._decode._peer_layouts),
        )

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        self._forward_step_id += 1
        logger.debug(
            "[PdConnector] worker start_load_kv metadata=%s wait_reqs=%s push_reqs=%s release=%s known_wait=%s known_push=%s",
            metadata,
            sorted(metadata.reqs_to_wait),
            sorted(metadata.reqs_to_push),
            sorted(metadata.reqs_to_release),
            sorted(self._decode.wait_reqs),
            sorted(self._prefill.push_reqs),
        )
        assert self.rdma is not None, "PdConnector RDMA port is not initialized"

        self._decode.process_wait_reqs(metadata.reqs_to_wait)
        self._prefill.process_push_reqs(metadata.reqs_to_push)

        for req_id in metadata.reqs_to_release:
            logger.debug("[PdConnector] worker release req=%s", req_id)
            self._decode.release(req_id)
            self._prefill.release(req_id)
            self.rdma.close_request(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert layer_name in self.layouts, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        self._prefill.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self._prefill.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        logger.debug(
            "[PdConnector] worker get_finished enter finished_req_ids=%s wait_reqs=%s push_reqs=%s",
            sorted(finished_req_ids),
            sorted(self._decode.wait_reqs),
            sorted(self._prefill.push_reqs),
        )

        releasable_sending = self._prefill.get_finished_sending(finished_req_ids)
        finished_recving = self.rdma.pop_finished_recving()
        self._decode.finish_recving(finished_recving)

        logger.debug(
            "[PdConnector] worker get_finished exit sending=%s recving=%s remaining_wait=%s remaining_push=%s",
            sorted(releasable_sending),
            sorted(finished_recving),
            sorted(self._decode.wait_reqs),
            sorted(self._prefill.push_reqs),
        )
        return releasable_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def build_connector_worker_meta(self) -> PdWorkerMetadata | None:
        return None

    def shutdown(self) -> None:
        self._decode.shutdown()
        self._prefill.shutdown()

    def _layer_idx(self, layer_name: str) -> int:
        try:
            return self.layer_names.index(layer_name)
        except ValueError as exc:
            raise AssertionError(f"unknown layer {layer_name}") from exc


# ---------------------------------------------------------------------------
# Module-level helpers (kept here so monkeypatching worker_mod.X still works)
# ---------------------------------------------------------------------------


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int | None:
    for tensor in kv_caches.values():
        device = getattr(tensor, "device", None)
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
    return None


def _tensor_parallel_identity(vllm_config: Any) -> tuple[int, int]:
    try:
        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        return (
            int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0),
            int(getattr(parallel_config, "tensor_parallel_size", 1) or 1),
        )
