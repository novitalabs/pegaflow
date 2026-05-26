"""Small request progress tracker for chunked prefill push."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestChunks:
    pushed_pairs: set[tuple[int, int]] = field(default_factory=set)
    done: bool = False


class ChunkTracker:
    def __init__(self) -> None:
        self._requests: dict[str, RequestChunks] = {}

    def add_request(self, req_id: str) -> None:
        self._requests.setdefault(req_id, RequestChunks())

    def mark_blocks_pushed(self, req_id: str, layer_idx: int, block_ids: set[int]) -> None:
        self.add_request(req_id)
        self._requests[req_id].pushed_pairs.update((layer_idx, block_id) for block_id in block_ids)

    def has_pushed_all_blocks(
        self,
        req_id: str,
        block_ids: set[int],
        *,
        num_layers: int,
    ) -> bool:
        self.add_request(req_id)
        expected = {
            (layer_idx, block_id) for layer_idx in range(num_layers) for block_id in block_ids
        }
        return expected.issubset(self._requests[req_id].pushed_pairs)

    def is_done(self, req_id: str) -> bool:
        self.add_request(req_id)
        return self._requests[req_id].done

    def mark_done(self, req_id: str) -> None:
        self.add_request(req_id)
        self._requests[req_id].done = True

    def remove(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
