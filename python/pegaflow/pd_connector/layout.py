"""KV-cache layout helpers for the experimental P/D connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegaflow.pd_connector.metadata import LayerRemoteLayout, LinearBlockAddrLayout


@dataclass(frozen=True)
class BlockSlice:
    block_id: int
    src_offset_bytes: int
    bytes: int


@dataclass(frozen=True)
class LayerBlockSlices:
    k: BlockSlice
    v: BlockSlice


@dataclass(frozen=True)
class FlashAttnHndLayout:
    """FlashAttention HND layout.

    Logical tensor shape:
        [2, num_blocks, block_size, num_kv_heads, head_size]

    HND physical order:
        [2, num_blocks, num_kv_heads, block_size, head_size]
    """

    layer_name: str
    shape: tuple[int, int, int, int, int]
    strides: tuple[int, int, int, int, int]
    element_size: int
    base_addr: int

    @classmethod
    def from_tensor(cls, layer_name: str, tensor: Any) -> FlashAttnHndLayout:
        shape = tuple(int(dim) for dim in tensor.shape)
        assert len(shape) == 5, (
            f"PdConnector only supports FlashAttention 5D KV cache; "
            f"layer={layer_name} shape={shape}"
        )
        assert shape[0] == 2, (
            f"PdConnector only supports FlashAttention KV-first layout "
            f"[2, num_blocks, block_size, num_kv_heads, head_size]; "
            f"layer={layer_name} shape={shape}"
        )

        strides = tuple(int(stride) for stride in tensor.stride())
        _, _, block_size, num_kv_heads, head_size = shape
        expected_token_stride = head_size
        expected_head_stride = block_size * head_size
        if block_size > 1:
            assert strides[2] == expected_token_stride, (
                f"PdConnector requires HND KV cache; layer={layer_name} "
                f"expected token stride {expected_token_stride}, got {strides[2]} "
                f"shape={shape} strides={strides}"
            )
        if num_kv_heads > 1:
            assert strides[3] == expected_head_stride, (
                f"PdConnector requires HND KV cache; layer={layer_name} "
                f"expected head stride {expected_head_stride}, got {strides[3]} "
                f"shape={shape} strides={strides}"
            )

        return cls(
            layer_name=layer_name,
            shape=shape,  # type: ignore[arg-type]
            strides=strides,  # type: ignore[arg-type]
            element_size=int(tensor.element_size()),
            base_addr=int(tensor.data_ptr()),
        )

    @property
    def num_blocks(self) -> int:
        return self.shape[1]

    @property
    def block_size(self) -> int:
        return self.shape[2]

    @property
    def num_kv_heads(self) -> int:
        return self.shape[3]

    @property
    def head_size(self) -> int:
        return self.shape[4]

    @property
    def block_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * self.element_size

    def block_offset_bytes(self, kv_idx: int, block_id: int) -> int:
        assert kv_idx in (0, 1), f"kv_idx must be 0 (K) or 1 (V), got {kv_idx}"
        assert 0 <= block_id < self.num_blocks, (
            f"block_id {block_id} out of range for layer={self.layer_name} "
            f"num_blocks={self.num_blocks}"
        )
        return (kv_idx * self.strides[0] + block_id * self.strides[1]) * self.element_size

    def block_slices(self, block_id: int) -> LayerBlockSlices:
        return LayerBlockSlices(
            k=BlockSlice(
                block_id=block_id,
                src_offset_bytes=self.block_offset_bytes(0, block_id),
                bytes=self.block_bytes,
            ),
            v=BlockSlice(
                block_id=block_id,
                src_offset_bytes=self.block_offset_bytes(1, block_id),
                bytes=self.block_bytes,
            ),
        )

    def remote_layout(self, layer_idx: int, block_ids: set[int] | None = None) -> LayerRemoteLayout:
        if block_ids is None:
            block_ids = set(range(self.num_blocks))
        ordered = tuple(sorted(block_ids))
        linear = self._linear_block_addr_layout(ordered)
        if linear is not None:
            return LayerRemoteLayout(
                layer_name=self.layer_name,
                layer_idx=layer_idx,
                base_addr=self.base_addr,
                block_bytes=self.block_bytes,
                block_ids=ordered,
                k_block_addrs=(),
                v_block_addrs=(),
                linear=linear,
            )
        return LayerRemoteLayout(
            layer_name=self.layer_name,
            layer_idx=layer_idx,
            base_addr=self.base_addr,
            block_bytes=self.block_bytes,
            block_ids=ordered,
            k_block_addrs=tuple(
                self.base_addr + self.block_offset_bytes(0, block_id) for block_id in ordered
            ),
            v_block_addrs=tuple(
                self.base_addr + self.block_offset_bytes(1, block_id) for block_id in ordered
            ),
            linear=None,
        )

    def _linear_block_addr_layout(
        self,
        ordered_block_ids: tuple[int, ...],
    ) -> LinearBlockAddrLayout | None:
        if not ordered_block_ids:
            return None
        block_id_stride = _constant_stride(ordered_block_ids)
        if block_id_stride is None:
            return None
        addr_stride = block_id_stride * self.block_bytes
        return LinearBlockAddrLayout(
            block_id_start=ordered_block_ids[0],
            block_id_stride=block_id_stride,
            num_blocks=len(ordered_block_ids),
            k_addr_start=self.base_addr + self.block_offset_bytes(0, ordered_block_ids[0]),
            v_addr_start=self.base_addr + self.block_offset_bytes(1, ordered_block_ids[0]),
            addr_stride=addr_stride,
        )


def block_slices_bytes(block_slices: list[LayerBlockSlices]) -> int:
    return sum(block.k.bytes + block.v.bytes for block in block_slices)


def block_ranges_for_remote_write(
    layout: FlashAttnHndLayout,
    local_block_ids: set[int],
    remote_block_ids: dict[int, int],
) -> list[LayerBlockSlices]:
    sorted_local = sorted(local_block_ids)
    if not sorted_local:
        return []
    ranges: list[LayerBlockSlices] = []
    start_local = prev_local = sorted_local[0]
    start_remote = prev_remote = remote_block_ids[start_local]
    count = 1
    for local_id in sorted_local[1:]:
        remote_id = remote_block_ids[local_id]
        if local_id == prev_local + 1 and remote_id == prev_remote + 1:
            prev_local, prev_remote, count = local_id, remote_id, count + 1
            continue
        ranges.append(_coalesced_block_slice(layout, start_local, start_remote, count))
        start_local = prev_local = local_id
        start_remote = prev_remote = remote_id
        count = 1
    ranges.append(_coalesced_block_slice(layout, start_local, start_remote, count))
    return ranges


def _coalesced_block_slice(
    layout: FlashAttnHndLayout,
    local_block_id: int,
    remote_block_id: int,
    count: int,
) -> LayerBlockSlices:
    local = layout.block_slices(local_block_id)
    bytes_total = layout.block_bytes * count
    return LayerBlockSlices(
        k=BlockSlice(
            block_id=remote_block_id, src_offset_bytes=local.k.src_offset_bytes, bytes=bytes_total
        ),
        v=BlockSlice(
            block_id=remote_block_id, src_offset_bytes=local.v.src_offset_bytes, bytes=bytes_total
        ),
    )


def unique_blocks_from_slot_mapping(slot_mapping: Any, block_size: int) -> set[int]:
    """Extract touched KV block ids from vLLM attention metadata slot_mapping."""
    slots = slot_mapping.detach().cpu().tolist()
    return {int(slot) // block_size for slot in slots if int(slot) >= 0}


def _constant_stride(values: tuple[int, ...]) -> int | None:
    if len(values) <= 1:
        return 1
    stride = values[1] - values[0]
    for prev, current in zip(values[:-1], values[1:], strict=True):
        if current - prev != stride:
            return None
    return stride
