"""Scheduler-side index for connector-owned recurrent-state GPU slots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearStateSlot:
    """Stable reference to one generation of a compound recurrent-state slot."""

    slot: int
    generation: int
    block_hash: bytes


@dataclass(slots=True)
class _Entry:
    ref: LinearStateSlot
    pins: int = 0


class LinearStateCache:
    """Fixed-capacity LRU with explicit save reservation and load pinning.

    Reservations are intentionally invisible to lookup until ``commit``. This is
    the scheduler half of the worker-completion protocol: every TP worker reports
    an exact slot-generation result through connector worker metadata, and only a
    unanimous success lets the scheduler publish the hash as reusable.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"linear-state cache capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._entries: dict[bytes, _Entry] = {}
        self._slots: list[_Entry | None] = [None] * capacity
        self._generations = [0] * capacity
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._reservations: dict[tuple[int, int], LinearStateSlot] = {}
        self._pending_hashes: set[bytes] = set()

    def lookup(self, block_hash: bytes, *, pin: bool = False) -> LinearStateSlot | None:
        entry = self._entries.get(block_hash)
        if entry is None:
            return None
        self._touch_slot(entry.ref.slot)
        if pin:
            entry.pins += 1
        return entry.ref

    def unpin(self, ref: LinearStateSlot) -> None:
        entry = self._entry_for_ref(ref)
        if entry.pins <= 0:
            raise RuntimeError(f"linear-state slot {ref.slot}/{ref.generation} is not pinned")
        entry.pins -= 1

    def reserve(self, block_hash: bytes) -> LinearStateSlot | None:
        """Reserve an unpublished slot, or return ``None`` when saving is skipped."""
        existing = self._entries.get(block_hash)
        if existing is not None:
            self._touch_slot(existing.ref.slot)
            return None
        if block_hash in self._pending_hashes:
            return None

        slot = self._find_reservable_slot()
        if slot is None:
            return None

        victim = self._slots[slot]
        if victim is not None:
            del self._entries[victim.ref.block_hash]
            self._lru.pop(slot, None)
            self._slots[slot] = None

        self._generations[slot] += 1
        ref = LinearStateSlot(slot, self._generations[slot], block_hash)
        self._reservations[(slot, ref.generation)] = ref
        self._pending_hashes.add(block_hash)
        return ref

    def commit(self, ref: LinearStateSlot) -> None:
        reserved = self._pop_reservation(ref)
        entry = _Entry(reserved)
        self._entries[reserved.block_hash] = entry
        self._slots[reserved.slot] = entry
        self._touch_slot(reserved.slot)

    def cancel(self, ref: LinearStateSlot) -> None:
        self._pop_reservation(ref)

    def clear(self) -> None:
        self._entries.clear()
        self._slots = [None] * self.capacity
        self._lru.clear()
        self._reservations.clear()
        self._pending_hashes.clear()

    def _find_reservable_slot(self) -> int | None:
        reserved_slots = {slot for slot, _generation in self._reservations}
        for slot, entry in enumerate(self._slots):
            if entry is None and slot not in reserved_slots:
                return slot
        for slot in self._lru:
            entry = self._slots[slot]
            if entry is not None and entry.pins == 0 and slot not in reserved_slots:
                return slot
        return None

    def _entry_for_ref(self, ref: LinearStateSlot) -> _Entry:
        if ref.slot < 0 or ref.slot >= self.capacity:
            raise RuntimeError(f"linear-state slot {ref.slot} is out of range")
        entry = self._slots[ref.slot]
        if entry is None or entry.ref != ref:
            raise RuntimeError(f"stale linear-state slot reference {ref.slot}/{ref.generation}")
        return entry

    def _pop_reservation(self, ref: LinearStateSlot) -> LinearStateSlot:
        key = (ref.slot, ref.generation)
        reserved = self._reservations.pop(key, None)
        if reserved != ref:
            raise RuntimeError(f"stale linear-state reservation {ref.slot}/{ref.generation}")
        self._pending_hashes.remove(ref.block_hash)
        return reserved

    def _touch_slot(self, slot: int) -> None:
        self._lru.pop(slot, None)
        self._lru[slot] = None
