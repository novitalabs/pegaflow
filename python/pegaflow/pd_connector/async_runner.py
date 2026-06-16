"""Shared async task-execution primitives for the P/D connector.

These factor out the threading skeleton that the prefill push sender, push
finalizer, and decode RDMA-done waiter previously duplicated:

- ``AsyncTaskPool``: thin owner of a ``ThreadPoolExecutor`` with a shared lock.
  Used by callback-driven pools that track their own per-request generation
  state (the decode RDMA-done waiter).
- ``InflightTaskRunner``: adds an inflight counter, error propagation, and
  ``wait_all`` / ``is_idle`` semantics on top, for pools whose callers block on
  completion (the prefill push sender and finalizer).

The runner deliberately keeps only what all in-flight pools share. Per-request
counting, cancellation, and dedup live in the subclasses because their
semantics differ.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class AsyncTaskPool:
    """Owns a ``ThreadPoolExecutor`` guarded by a shared lock.

    Subclasses implement ``_execute(task)`` with the work body and call
    ``_spawn(task)`` to schedule it. The pool only manages executor lifecycle;
    all task bookkeeping is the subclass's responsibility.
    """

    def __init__(self, thread_name_prefix: str, max_workers: int = 16) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix=thread_name_prefix,
        )

    def _spawn(self, task: Any) -> None:
        self._executor.submit(self._execute, task)

    def _execute(self, task: Any) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class InflightTaskRunner(AsyncTaskPool, Generic[T]):
    """Task pool that tracks inflight count, errors, and supports ``wait_all``.

    Shared by the prefill push sender and finalizer. Uses a ``Condition`` so
    callers can block until all (or a subset of) submitted tasks drain. The
    first task exception is captured and re-raised to the next waiter, matching
    the original fail-fast behaviour.

    Subclasses implement ``_run(task)`` (the work body) and may override
    ``_on_submit_locked`` / ``_on_finish_locked`` to maintain per-request state.
    ``_set_inflight_metric_locked`` is called whenever the inflight count
    changes so subclasses can publish their own gauge.
    """

    def __init__(
        self,
        thread_name_prefix: str,
        metrics: Any | None = None,
        max_workers: int = 16,
    ) -> None:
        super().__init__(thread_name_prefix, max_workers=max_workers)
        self._condition = threading.Condition()
        self._inflight = 0
        self._error: BaseException | None = None
        self._closed = False
        self._metrics = metrics

    # -- submission ---------------------------------------------------------

    def _submit(self, task: T, closed_message: str) -> bool:
        """Register and schedule a task.

        Returns False if the subclass's ``_on_submit_locked`` rejected the task
        (e.g. dedup); raises if the pool is closed or holds a pending error.
        """
        with self._condition:
            if self._closed:
                raise RuntimeError(closed_message)
            if self._error is not None:
                raise self._error
            if not self._on_submit_locked(task):
                return False
            self._inflight += 1
            self._set_inflight_metric_locked()
        try:
            self._spawn(task)
        except BaseException:
            with self._condition:
                self._finish_locked(task)
            raise
        return True

    # -- waiting ------------------------------------------------------------

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            self._raise_pending_error_locked()

    def is_idle(self) -> bool:
        with self._condition:
            return self._inflight == 0 and self._error is None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        super().close()

    # -- execution ----------------------------------------------------------

    def _execute(self, task: T) -> None:
        try:
            self._run(task)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            self._on_error(task, exc)
        finally:
            with self._condition:
                self._finish_locked(task)

    def _finish_locked(self, task: T) -> None:
        self._inflight -= 1
        self._on_finish_locked(task)
        self._set_inflight_metric_locked()
        self._condition.notify_all()

    def _raise_pending_error_locked(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise error

    # -- hooks for subclasses ----------------------------------------------

    def _run(self, task: T) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _on_submit_locked(self, task: T) -> bool:
        """Update per-request state under the condition lock. Return False to
        skip scheduling (e.g. duplicate task)."""
        return True

    def _on_finish_locked(self, task: T) -> None:
        """Update per-request state when a task completes (lock held)."""

    def _on_error(self, task: T, exc: BaseException) -> None:
        """Side effects on task failure (e.g. metrics), outside the lock."""

    def _set_inflight_metric_locked(self) -> None:
        """Publish the inflight gauge. Default no-op."""


__all__ = ["AsyncTaskPool", "InflightTaskRunner"]
