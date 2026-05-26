"""Typed definitions for the two kv_transfer_params formats in P/D push.

Router → D (consumer):  ConsumerKvParams  — "do a remote prefill for this request"
D → P (producer):       ProducerKvParams  — "run prefill and push KV back via RDMA"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegaflow.pd_connector.metadata import (
    PdHandshake,
    handshake_from_dict,
    handshake_to_dict,
    handshakes_from_dicts,
)


@dataclass(frozen=True)
class ConsumerKvParams:
    """Router → D: tells Decode to wait for a remote prefill."""

    prefill_url: str
    remote_request_id: str = ""
    done_request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"do_remote_prefill": True, "prefill_url": self.prefill_url}
        if self.remote_request_id:
            d["remote_request_id"] = self.remote_request_id
        if self.done_request_id:
            d["done_request_id"] = self.done_request_id
        return d


@dataclass(frozen=True)
class ProducerKvParams:
    """D → P: tells Prefill to run prefill and RDMA-push KV to D."""

    target_engine_id: str
    target_request_id: str
    handshakes: tuple[PdHandshake, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "do_remote_prefill_sender": True,
            "target_engine_id": self.target_engine_id,
            "target_request_id": self.target_request_id,
        }
        if self.handshakes:
            result["pd_handshakes"] = [
                handshake_to_dict(handshake) for handshake in self.handshakes
            ]
        return result


def parse_consumer(params: dict[str, Any]) -> ConsumerKvParams | None:
    if not params.get("do_remote_prefill"):
        return None
    prefill_url = params.get("prefill_url")
    if not prefill_url:
        return None
    return ConsumerKvParams(
        prefill_url=str(prefill_url),
        remote_request_id=str(params.get("remote_request_id") or ""),
        done_request_id=str(params.get("done_request_id") or ""),
    )


def parse_producer(params: dict[str, Any]) -> ProducerKvParams | None:
    if not params.get("do_remote_prefill_sender"):
        return None
    handshakes = handshakes_from_dicts(params.get("pd_handshakes"))
    if not handshakes:
        single = handshake_from_dict(params.get("pd_handshake"))
        handshakes = (single,) if single is not None else ()
    return ProducerKvParams(
        target_engine_id=str(params.get("target_engine_id") or ""),
        target_request_id=str(params.get("target_request_id") or ""),
        handshakes=handshakes,
    )


def is_consumer(params: dict[str, Any]) -> bool:
    return bool(params.get("do_remote_prefill"))


def is_producer(params: dict[str, Any]) -> bool:
    return bool(params.get("do_remote_prefill_sender"))
