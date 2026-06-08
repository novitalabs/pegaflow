"""Tensor-parallel KV-head mapping for the experimental P/D connector."""

from __future__ import annotations

from dataclasses import dataclass

from pegaflow.pd_connector.metadata import PdHandshake


@dataclass(frozen=True)
class HeadSlice:
    local_start: int
    local_end: int
    remote_start: int
    remote_end: int
    global_heads: tuple[int, ...]

    def __post_init__(self) -> None:
        assert 0 <= self.local_start < self.local_end
        assert 0 <= self.remote_start < self.remote_end
        assert self.local_end - self.local_start == self.remote_end - self.remote_start
        assert len(self.global_heads) == self.local_end - self.local_start


@dataclass(frozen=True)
class PushTargetPlan:
    handshake: PdHandshake
    head_slices: tuple[HeadSlice, ...]


@dataclass(frozen=True)
class PushLayoutPlan:
    targets: tuple[PushTargetPlan, ...]

    @property
    def should_skip(self) -> bool:
        return not self.targets


def build_push_layout_plan(
    *,
    prefill_tp_rank: int,
    prefill_tp_size: int,
    decode_handshakes: tuple[PdHandshake, ...],
    local_num_kv_heads: int,
    remote_num_kv_heads: int,
    total_num_kv_heads: int,
    use_mla: bool,
) -> PushLayoutPlan:
    assert decode_handshakes, "PdConnector push request has no handshakes"
    assert 0 <= prefill_tp_rank < prefill_tp_size
    assert local_num_kv_heads > 0
    assert remote_num_kv_heads > 0
    assert total_num_kv_heads > 0
    _assert_handshake_tp_consistency(decode_handshakes)
    handshakes_by_rank = {handshake.tp_rank: handshake for handshake in decode_handshakes}
    decode_tp_size = decode_handshakes[0].tp_size
    assert len(handshakes_by_rank) == decode_tp_size, (
        "PdConnector requires one decode handshake per TP rank; "
        f"decode_tp={decode_tp_size} available={sorted(handshakes_by_rank)}"
    )

    if use_mla:
        return _build_mla_plan(
            prefill_tp_rank=prefill_tp_rank,
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            handshakes_by_rank=handshakes_by_rank,
        )

    assert (
        prefill_tp_size == decode_tp_size
        or prefill_tp_size % decode_tp_size == 0
        or decode_tp_size % prefill_tp_size == 0
    ), (
        "PdConnector heterogeneous layout mapping requires divisible TP sizes; "
        f"prefill_tp={prefill_tp_size} decode_tp={decode_tp_size}"
    )
    assert prefill_tp_size * local_num_kv_heads >= total_num_kv_heads, (
        "PdConnector prefill KV-head coverage is smaller than total KV heads; "
        f"prefill_tp={prefill_tp_size} local_heads={local_num_kv_heads} "
        f"total_heads={total_num_kv_heads}"
    )
    assert decode_tp_size * remote_num_kv_heads >= total_num_kv_heads, (
        "PdConnector decode KV-head coverage is smaller than total KV heads; "
        f"decode_tp={decode_tp_size} remote_heads={remote_num_kv_heads} "
        f"total_heads={total_num_kv_heads}"
    )

    target_slices: dict[int, list[HeadSlice]] = {}
    local_heads = _rank_global_heads(
        rank=prefill_tp_rank,
        local_heads=local_num_kv_heads,
        total_heads=total_num_kv_heads,
    )
    for local_idx, global_head in enumerate(local_heads):
        if (
            _owner_rank_for_global_head(
                global_head,
                rank_count=prefill_tp_size,
                local_heads=local_num_kv_heads,
                total_heads=total_num_kv_heads,
            )
            != prefill_tp_rank
        ):
            continue
        decode_rank = _owner_rank_for_global_head(
            global_head,
            rank_count=decode_tp_size,
            local_heads=remote_num_kv_heads,
            total_heads=total_num_kv_heads,
        )
        remote_heads = _rank_global_heads(
            rank=decode_rank,
            local_heads=remote_num_kv_heads,
            total_heads=total_num_kv_heads,
        )
        remote_idx = remote_heads.index(global_head)
        target_slices.setdefault(decode_rank, []).append(
            HeadSlice(
                local_start=local_idx,
                local_end=local_idx + 1,
                remote_start=remote_idx,
                remote_end=remote_idx + 1,
                global_heads=(global_head,),
            )
        )

    targets = tuple(
        PushTargetPlan(
            handshake=handshakes_by_rank[rank],
            head_slices=_drop_full_rank_slice(
                _coalesce_head_slices(slices),
                local_num_kv_heads=local_num_kv_heads,
                remote_num_kv_heads=remote_num_kv_heads,
            ),
        )
        for rank, slices in sorted(target_slices.items())
    )
    return PushLayoutPlan(targets=targets)


def decode_rank_source_counts(
    *,
    prefill_tp_size: int,
    decode_tp_size: int,
    local_num_kv_heads: int,
    remote_num_kv_heads: int,
    total_num_kv_heads: int,
    use_mla: bool,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for prefill_tp_rank in range(prefill_tp_size):
        placeholder_handshakes = tuple(
            PdHandshake(
                request_id=f"rank-{rank}",
                engine_id="",
                tp_rank=rank,
                tp_size=decode_tp_size,
                block_size=1,
                layers=(),
            )
            for rank in range(decode_tp_size)
        )
        plan = build_push_layout_plan(
            prefill_tp_rank=prefill_tp_rank,
            prefill_tp_size=prefill_tp_size,
            decode_handshakes=placeholder_handshakes,
            local_num_kv_heads=local_num_kv_heads,
            remote_num_kv_heads=remote_num_kv_heads,
            total_num_kv_heads=total_num_kv_heads,
            use_mla=use_mla,
        )
        for target in plan.targets:
            counts[target.handshake.tp_rank] = counts.get(target.handshake.tp_rank, 0) + 1
    return counts


def _build_mla_plan(
    *,
    prefill_tp_rank: int,
    prefill_tp_size: int,
    decode_tp_size: int,
    handshakes_by_rank: dict[int, PdHandshake],
) -> PushLayoutPlan:
    if prefill_tp_size >= decode_tp_size:
        assert prefill_tp_size % decode_tp_size == 0, (
            "PdConnector MLA heterogeneous TP requires divisible TP sizes; "
            f"prefill_tp={prefill_tp_size} decode_tp={decode_tp_size}"
        )
        ratio = prefill_tp_size // decode_tp_size
        if prefill_tp_rank % ratio != 0:
            return PushLayoutPlan(targets=())
        decode_rank = prefill_tp_rank // ratio
    else:
        assert decode_tp_size % prefill_tp_size == 0, (
            "PdConnector MLA heterogeneous TP requires divisible TP sizes; "
            f"prefill_tp={prefill_tp_size} decode_tp={decode_tp_size}"
        )
        ratio = decode_tp_size // prefill_tp_size
        decode_ranks = tuple(range(prefill_tp_rank * ratio, (prefill_tp_rank + 1) * ratio))
        return PushLayoutPlan(
            targets=tuple(
                PushTargetPlan(
                    handshake=handshakes_by_rank[decode_rank],
                    head_slices=(),
                )
                for decode_rank in decode_ranks
            )
        )
    return PushLayoutPlan(
        targets=(
            PushTargetPlan(
                handshake=handshakes_by_rank[decode_rank],
                head_slices=(),
            ),
        )
    )


def _rank_global_heads(
    *,
    rank: int,
    local_heads: int,
    total_heads: int,
) -> tuple[int, ...]:
    start = rank * local_heads
    return tuple((start + offset) % total_heads for offset in range(local_heads))


def _owner_rank_for_global_head(
    global_head: int,
    *,
    rank_count: int,
    local_heads: int,
    total_heads: int,
) -> int:
    for rank in range(rank_count):
        if global_head in _rank_global_heads(
            rank=rank,
            local_heads=local_heads,
            total_heads=total_heads,
        ):
            return rank
    raise AssertionError(
        f"PdConnector could not find owner for global KV head {global_head}; "
        f"rank_count={rank_count} local_heads={local_heads} total_heads={total_heads}"
    )


def _coalesce_head_slices(slices: list[HeadSlice]) -> tuple[HeadSlice, ...]:
    if not slices:
        return ()
    ordered = sorted(slices, key=lambda item: (item.local_start, item.remote_start))
    coalesced: list[HeadSlice] = [ordered[0]]
    for item in ordered[1:]:
        prev = coalesced[-1]
        if (
            prev.local_end == item.local_start
            and prev.remote_end == item.remote_start
            and prev.global_heads[-1] + 1 == item.global_heads[0]
        ):
            coalesced[-1] = HeadSlice(
                local_start=prev.local_start,
                local_end=item.local_end,
                remote_start=prev.remote_start,
                remote_end=item.remote_end,
                global_heads=prev.global_heads + item.global_heads,
            )
            continue
        coalesced.append(item)
    return tuple(coalesced)


def _drop_full_rank_slice(
    slices: tuple[HeadSlice, ...],
    *,
    local_num_kv_heads: int,
    remote_num_kv_heads: int,
) -> tuple[HeadSlice, ...]:
    if len(slices) == 1:
        only = slices[0]
        if (
            only.local_start == 0
            and only.local_end == local_num_kv_heads
            and only.remote_start == 0
            and only.remote_end == remote_num_kv_heads
        ):
            return ()
    return slices


def _assert_handshake_tp_consistency(handshakes: tuple[PdHandshake, ...]) -> None:
    tp_size = handshakes[0].tp_size
    assert all(handshake.tp_size == tp_size for handshake in handshakes), (
        "PdConnector handshakes disagree on decode TP size: "
        f"{[handshake.tp_size for handshake in handshakes]}"
    )
