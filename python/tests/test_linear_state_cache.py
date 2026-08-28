from __future__ import annotations

import pytest

from .unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from pegaflow.connector.linear_state_cache import LinearStateCache  # noqa: E402


def test_reservations_are_invisible_until_committed():
    cache = LinearStateCache(1)
    ref = cache.reserve(b"a")

    assert ref is not None
    assert cache.lookup(b"a") is None

    cache.commit(ref)
    assert cache.lookup(b"a") == ref


def test_lru_evicts_oldest_unpinned_entry():
    cache = LinearStateCache(2)
    a = cache.reserve(b"a")
    b = cache.reserve(b"b")
    assert a is not None and b is not None
    cache.commit(a)
    cache.commit(b)
    cache.lookup(b"a")

    c = cache.reserve(b"c")
    assert c is not None
    cache.commit(c)

    assert cache.lookup(b"a") == a
    assert cache.lookup(b"b") is None
    assert cache.lookup(b"c") == c


def test_pinned_and_reserved_slots_are_not_reused():
    cache = LinearStateCache(2)
    a = cache.reserve(b"a")
    b = cache.reserve(b"b")
    assert a is not None and b is not None
    cache.commit(a)
    cache.commit(b)
    assert cache.lookup(b"a", pin=True) == a
    assert cache.lookup(b"b", pin=True) == b

    assert cache.reserve(b"c") is None
    cache.unpin(a)
    c = cache.reserve(b"c")
    assert c is not None
    assert cache.reserve(b"d") is None


def test_generation_rejects_stale_reference_after_reuse():
    cache = LinearStateCache(1)
    first = cache.reserve(b"first")
    assert first is not None
    cache.commit(first)
    second = cache.reserve(b"second")
    assert second is not None
    cache.commit(second)

    assert second.slot == first.slot
    assert second.generation == first.generation + 1
    with pytest.raises(RuntimeError, match="stale"):
        cache.unpin(first)


def test_duplicate_hash_touches_without_overwrite():
    cache = LinearStateCache(2)
    a = cache.reserve(b"a")
    b = cache.reserve(b"b")
    assert a is not None and b is not None
    cache.commit(a)
    cache.commit(b)

    assert cache.reserve(b"a") is None
    c = cache.reserve(b"c")
    assert c is not None
    cache.commit(c)
    assert cache.lookup(b"a") == a
    assert cache.lookup(b"b") is None


def test_cancel_and_clear_remove_unpublished_state():
    cache = LinearStateCache(1)
    ref = cache.reserve(b"a")
    assert ref is not None
    cache.cancel(ref)
    replacement = cache.reserve(b"b")
    assert replacement is not None
    cache.commit(replacement)

    cache.clear()
    assert cache.lookup(b"b") is None
    assert cache.reserve(b"c") is not None
