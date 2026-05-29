"""KV-cache layout helpers for the experimental P/D connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pegaflow.pd_connector.metadata import LayerRemoteLayout, TransferRegionLayout

BlockIdSelection = set[int] | tuple[int, ...] | None


@dataclass(frozen=True)
class BlockRegionSlice:
    block_id: int
    src_offset_bytes: int
    bytes: int

    def __post_init__(self) -> None:
        assert self.block_id >= 0
        assert self.src_offset_bytes >= 0
        assert self.bytes > 0


@dataclass(frozen=True)
class LayerBlockSlices:
    regions: tuple[BlockRegionSlice, ...]

    def __post_init__(self) -> None:
        assert self.regions


class KvCacheLayout(Protocol):
    layer_name: str
    shape: tuple[int, ...]

    @property
    def num_blocks(self) -> int: ...

    @property
    def block_size(self) -> int: ...

    def block_slices(self, block_id: int) -> LayerBlockSlices: ...

    def remote_layout(
        self, layer_idx: int, block_ids: BlockIdSelection = None
    ) -> LayerRemoteLayout: ...


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
    def from_tensor(
        cls,
        layer_name: str,
        tensor: Any,
        *,
        layer_spec: Any | None = None,
        expected_num_blocks: int | None = None,
    ) -> FlashAttnHndLayout:
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
        if expected_num_blocks is not None:
            assert shape[1] == expected_num_blocks, (
                f"PdConnector requires all KV cache tensors to share num_blocks; "
                f"layer={layer_name} expected={expected_num_blocks} shape={shape}"
            )

        layout = cls(
            layer_name=layer_name,
            shape=shape,  # type: ignore[arg-type]
            strides=strides,  # type: ignore[arg-type]
            element_size=int(tensor.element_size()),
            base_addr=int(tensor.data_ptr()),
        )
        if layer_spec is not None:
            region_block_len = _region_block_len_from_spec(
                _unwrap_layer_spec(layer_spec, layer_name),
                region_count=2,
            )
            assert region_block_len == layout.block_bytes, (
                f"PdConnector HND layer block bytes must match KVCacheSpec page size; "
                f"layer={layer_name} tensor_block_bytes={layout.block_bytes} "
                f"spec_region_block_len={region_block_len}"
            )
        return layout

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
            regions=(
                BlockRegionSlice(
                    block_id=block_id,
                    src_offset_bytes=self.block_offset_bytes(0, block_id),
                    bytes=self.block_bytes,
                ),
                BlockRegionSlice(
                    block_id=block_id,
                    src_offset_bytes=self.block_offset_bytes(1, block_id),
                    bytes=self.block_bytes,
                ),
            ),
        )

    def remote_layout(
        self, layer_idx: int, block_ids: BlockIdSelection = None
    ) -> LayerRemoteLayout:
        ordered = _ordered_block_ids(block_ids, self.num_blocks)
        return LayerRemoteLayout(
            layer_name=self.layer_name,
            layer_idx=layer_idx,
            block_ids=ordered,
            regions=(
                TransferRegionLayout(
                    region_idx=0,
                    base_addr=self.base_addr + self.block_offset_bytes(0, 0),
                    block_len=self.block_bytes,
                ),
                TransferRegionLayout(
                    region_idx=1,
                    base_addr=self.base_addr + self.block_offset_bytes(1, 0),
                    block_len=self.block_bytes,
                ),
            ),
        )


@dataclass(frozen=True)
class MlaBlocksLayout:
    """FlashMLA blocks-first layout used by MLA and indexer cache tensors."""

    layer_name: str
    shape: tuple[int, int, int]
    strides: tuple[int, int, int]
    element_size: int
    base_addr: int
    logical_block_size: int
    block_bytes: int

    @classmethod
    def from_tensor(
        cls,
        layer_name: str,
        tensor: Any,
        *,
        layer_spec: Any,
        logical_block_size: int,
        expected_num_blocks: int | None = None,
    ) -> MlaBlocksLayout:
        assert logical_block_size > 0
        shape = tuple(int(dim) for dim in tensor.shape)
        assert len(shape) == 3, (
            f"PdConnector MLA cache must be 3D blocks-first; layer={layer_name} shape={shape}"
        )
        physical_block_size = shape[1]
        assert physical_block_size == logical_block_size, (
            "PdConnector MLA first version does not support physical/logical "
            f"block split; layer={layer_name} logical_block_size={logical_block_size} "
            f"physical_block_size={physical_block_size}"
        )
        if expected_num_blocks is not None:
            assert shape[0] == expected_num_blocks, (
                f"PdConnector requires all KV cache tensors to share num_blocks; "
                f"layer={layer_name} expected={expected_num_blocks} shape={shape}"
            )
        strides = tuple(int(stride) for stride in tensor.stride())
        if shape[2] > 1:
            assert strides[2] == 1, (
                f"PdConnector MLA cache requires contiguous head dimension; "
                f"layer={layer_name} shape={shape} strides={strides}"
            )
        if shape[1] > 1:
            assert strides[1] == shape[2], (
                f"PdConnector MLA cache requires blocks-first row stride; "
                f"layer={layer_name} expected={shape[2]} got={strides[1]} "
                f"shape={shape} strides={strides}"
            )
        layer_spec = _unwrap_layer_spec(layer_spec, layer_name)
        spec_block_size = int(getattr(layer_spec, "block_size", logical_block_size))
        assert spec_block_size == logical_block_size, (
            "PdConnector MLA first version does not support physical/logical "
            f"block split; layer={layer_name} spec_block_size={spec_block_size} "
            f"logical_block_size={logical_block_size}"
        )
        region_block_len = _region_block_len_from_spec(layer_spec, region_count=1)
        tensor_block_len = strides[0] * int(tensor.element_size())
        assert region_block_len == tensor_block_len, (
            f"PdConnector MLA layer block bytes must match KVCacheSpec page size; "
            f"layer={layer_name} tensor_block_bytes={tensor_block_len} "
            f"spec_region_block_len={region_block_len}"
        )
        return cls(
            layer_name=layer_name,
            shape=shape,  # type: ignore[arg-type]
            strides=strides,  # type: ignore[arg-type]
            element_size=int(tensor.element_size()),
            base_addr=int(tensor.data_ptr()),
            logical_block_size=logical_block_size,
            block_bytes=region_block_len,
        )

    @property
    def num_blocks(self) -> int:
        return self.shape[0]

    @property
    def block_size(self) -> int:
        return self.logical_block_size

    def block_offset_bytes(self, block_id: int) -> int:
        assert 0 <= block_id < self.num_blocks, (
            f"block_id {block_id} out of range for layer={self.layer_name} "
            f"num_blocks={self.num_blocks}"
        )
        return block_id * self.strides[0] * self.element_size

    def block_slices(self, block_id: int) -> LayerBlockSlices:
        return LayerBlockSlices(
            regions=(
                BlockRegionSlice(
                    block_id=block_id,
                    src_offset_bytes=self.block_offset_bytes(block_id),
                    bytes=self.block_bytes,
                ),
            ),
        )

    def remote_layout(
        self, layer_idx: int, block_ids: BlockIdSelection = None
    ) -> LayerRemoteLayout:
        return LayerRemoteLayout(
            layer_name=self.layer_name,
            layer_idx=layer_idx,
            block_ids=_ordered_block_ids(block_ids, self.num_blocks),
            regions=(
                TransferRegionLayout(
                    region_idx=0,
                    base_addr=self.base_addr + self.block_offset_bytes(0),
                    block_len=self.block_bytes,
                ),
            ),
        )


def layout_from_tensor(
    layer_name: str,
    cache_tensor: Any,
    *,
    layer_spec: Any | None,
    logical_block_size: int,
    expected_num_blocks: int | None = None,
) -> KvCacheLayout:
    shape = tuple(int(dim) for dim in cache_tensor.shape)
    if len(shape) == 3:
        assert layer_spec is not None, (
            f"PdConnector MLA/indexer layer requires KVCacheSpec; layer={layer_name}"
        )
        return MlaBlocksLayout.from_tensor(
            layer_name,
            cache_tensor,
            layer_spec=layer_spec,
            logical_block_size=logical_block_size,
            expected_num_blocks=expected_num_blocks,
        )
    if len(shape) == 5:
        return FlashAttnHndLayout.from_tensor(
            layer_name,
            cache_tensor,
            layer_spec=layer_spec,
            expected_num_blocks=expected_num_blocks,
        )
    raise AssertionError(
        f"PdConnector unsupported KV cache tensor rank; layer={layer_name} shape={shape}"
    )


def block_slices_bytes(block_slices: list[LayerBlockSlices]) -> int:
    return sum(region.bytes for block in block_slices for region in block.regions)


def _ordered_block_ids(block_ids: BlockIdSelection, num_blocks: int) -> tuple[int, ...]:
    if block_ids is None:
        return tuple(range(num_blocks))
    if isinstance(block_ids, tuple):
        return block_ids
    return tuple(sorted(block_ids))


def block_ranges_for_remote_write(
    layout: KvCacheLayout,
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
    layout: KvCacheLayout,
    local_block_id: int,
    remote_block_id: int,
    count: int,
) -> LayerBlockSlices:
    local = layout.block_slices(local_block_id)
    return LayerBlockSlices(
        regions=tuple(
            BlockRegionSlice(
                block_id=remote_block_id,
                src_offset_bytes=region.src_offset_bytes,
                bytes=region.bytes * count,
            )
            for region in local.regions
        ),
    )


def unique_blocks_from_slot_mapping(slot_mapping: Any, block_size: int) -> set[int]:
    """Extract touched KV block ids from vLLM attention metadata slot_mapping."""
    slots = slot_mapping.detach().cpu().tolist()
    return {int(slot) // block_size for slot in slots if int(slot) >= 0}


def _unwrap_layer_spec(layer_spec: Any, layer_name: str) -> Any:
    specs = getattr(layer_spec, "kv_cache_specs", None)
    if isinstance(specs, dict):
        assert layer_name in specs, (
            f"PdConnector layer {layer_name} missing from UniformTypeKVCacheSpecs"
        )
        return specs[layer_name]
    return layer_spec


def _region_block_len_from_spec(layer_spec: Any, *, region_count: int) -> int:
    assert region_count > 0
    page_size_bytes = int(layer_spec.page_size_bytes)
    # Matches NIXL: physical_page_size = page_size_bytes / physical_blocks_per_logical,
    # then split the page evenly across transfer regions. PD MLA first version
    # only supports physical_blocks_per_logical == 1.
    assert page_size_bytes % region_count == 0, (
        f"KVCacheSpec page_size_bytes={page_size_bytes} cannot split into "
        f"{region_count} transfer regions"
    )
    return page_size_bytes // region_count
