"""Thin RDMA port abstraction used by the P/D connector skeleton."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from pegaflow.pd_connector.layout import LayerBlockSlices
from pegaflow.pd_connector.metadata import LayerRemoteLayout, PdHandshake


class RdmaPort(Protocol):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def push_done(self, req_id: str) -> None: ...

    def wait_done(self, req_id: str) -> None: ...

    def mark_done(self, req_id: str) -> None: ...

    def pop_finished_sending(self) -> set[str]: ...

    def pop_finished_recving(self) -> set[str]: ...


class NoopRdmaPort:
    """A non-blocking RDMA stub that records calls and completes immediately."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.remote_handshakes: dict[str, PdHandshake | None] = {}
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = defaultdict(list)
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None:
        self.registered.add(req_id)
        self.remote_handshakes[req_id] = handshake

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pushed_layers[req_id].append((layer_idx, blocks))

    def push_done(self, req_id: str) -> None:
        self._finished_sending.add(req_id)

    def wait_done(self, req_id: str) -> None:
        return None

    def mark_done(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def pop_finished_sending(self) -> set[str]:
        finished = self._finished_sending
        self._finished_sending = set()
        return finished

    def pop_finished_recving(self) -> set[str]:
        finished = self._finished_recving
        self._finished_recving = set()
        return finished


class MockRdmaPort(NoopRdmaPort):
    """Alias for now; later tests can add stricter copy semantics here."""
