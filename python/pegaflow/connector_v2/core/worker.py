"""Worker-side glue: register KV caches, run the step's loads and saves.

vLLM- and torch-free: tensors are opaque, and the connector injects ``wrap``
(tensor -> CUDA IPC bytes), ``new_load_state`` and ``mark_compute`` (returns a
handle whose ``synchronize()`` waits until the work enqueued so far on the
compute stream is done, e.g. a recorded CUDA event), so unit tests can drive it
with fakes.

Saves are asynchronous: ``save`` only marks the compute stream and queues the
job; a saver thread waits for the mark and then issues the save RPC (the server
copies GPU->CPU on its own stream). The source blocks must stay untouched until
the copy is done, so:

- finished requests that had saves are held by vLLM (``request_finished``
  returns True) until ``poll`` reports them in ``finished_sending``, which
  happens once every job submitted before vLLM told us they finished is done;
- a step that reuses a preempted request's blocks first calls ``flush``.

``flush`` runs in vLLM's ``handle_preemptions``, which vLLM's V1 model runner
calls before it zeroes or copies (copy-on-write) any block, but the V2 runner
calls after. Layouts where vLLM may zero or copy blocks (``sync_saves``) so
wait for every save inside ``wait_for_save`` instead, before the next step.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..adapters import Parallel, Unsupported, adapter_for
from .layout import Layout
from .scheduler import StepPlan

logger = logging.getLogger(__name__)


@dataclass
class _Batch:
    state: Any  # PyLoadState
    req_ids: list[str]


@dataclass
class _SaveJob:
    seq: int
    ready: Any  # mark_compute() handle
    saves: list[tuple[str, list[int], list[bytes]]]


@dataclass
class SaveStats:
    """Save cost since the last report (grep "save stats" in vllm.log)."""

    steps: int = 0  # wait_for_save calls
    save_steps: int = 0  # steps that queued a save job
    blocks: int = 0  # (layer, block) copies requested
    bytes: int = 0
    submit_s: float = 0.0  # step thread: building and queueing jobs (on the critical path)
    flushes: int = 0  # preemption flushes that had to wait
    flush_s: float = 0.0  # step thread: blocked in flush (on the critical path)
    wait_s: float = 0.0  # saver thread: waiting for the compute stream
    rpc_s: float = 0.0  # saver thread: save RPC (server copies GPU->CPU inside it)
    max_job_s: float = 0.0  # saver thread: wait + rpc of one job
    max_pending: int = 0  # jobs queued or running at submit time
    failed: int = 0  # jobs that did not store their blocks
    _since: float = field(default=0.0, repr=False)


class Worker:
    # wait_for_save calls between save-cost log lines.
    SAVE_LOG_EVERY = 500
    SHUTDOWN_FLUSH_S = 30.0
    # Waits that never give up (sync loads, flush) log this often.
    STALL_LOG_S = 10.0
    # Sync-load poll interval: doubles from the first value up to the cap.
    SYNC_POLL_S = (20e-6, 200e-6)

    def __init__(
        self,
        layout: Layout,
        rpc: Any,
        instance_id: str,
        par: Parallel,
        device_id: int,
        wrap: Callable[[Any], bytes],
        new_load_state: Callable[[], Any],
        mark_compute: Callable[[], Any],
        clock: Callable[[], float] = time.perf_counter,
        sync_saves: bool = False,
    ):
        """``sync_saves``: ``save`` returns only after the step's saves are done
        (vLLM may zero or copy blocks before ``handle_preemptions`` runs)."""
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._par = par
        self._device_id = device_id
        self._wrap = wrap
        self._new_load_state = new_load_state
        self._mark_compute = mark_compute
        self._clock = clock
        self._sync_saves = sync_saves
        self._layer_bytes: dict[str, int] = {}  # bytes one block of a layer stores
        self.save_stats = SaveStats(_since=clock())
        # Registered layer names per storage group (index into layout.groups).
        self._layers: list[list[str]] = [[] for _ in layout.groups]
        self._batches: list[_Batch] = []
        self._failed: set[str] = set()  # failed, not yet reported
        self._done: set[str] = set()  # finished without a load batch, not yet reported
        self._invalid: set[int] = set()  # GPU blocks of failed sync loads, not yet reported
        # Async saves. Jobs complete in submit order (one saver thread), so a
        # sequence number is enough to know what has landed.
        self._cv = threading.Condition()
        self._submitted = 0  # step thread only
        self._completed = 0  # saver thread, under _cv
        self._finishing: dict[str, int] = {}  # req -> last job that may read its blocks
        # The saver thread exited: it reads no block any more, so every hold
        # is released and later saves are dropped (future misses).
        self._saver_dead = False
        self._jobs: queue.SimpleQueue[_SaveJob | None] = queue.SimpleQueue()
        self._saver = threading.Thread(
            target=self._save_loop, name="pegaflow-v2-saver", daemon=True
        )
        self._saver.start()

    def register(self, kv_caches: dict[str, Any], specs: dict[str, Any]) -> None:
        """``specs``: layer name -> vLLM KVCacheSpec for every transferable layer."""
        names: list[str] = []
        wrappers: list[bytes] = []
        num_blocks: list[int] = []
        bytes_per_block: list[int] = []
        kv_strides: list[int] = []
        segments: list[int] = []
        group_ids: list[int] = []
        for sid, g in enumerate(self._layout.groups):
            for name in g.layers:
                if name not in kv_caches:
                    continue  # another PP rank owns this layer
                spec = specs[name]
                adapter = adapter_for(spec, g.gid)
                decl = adapter.layer(spec, name, self._par)
                view = adapter.view(spec, name, kv_caches[name])
                # The server copies bytes_per_block per segment: a declaration
                # that disagrees would store bytes other than the declared ones.
                if not decl.segments or any(s != view.bytes_per_block for s in decl.segments):
                    raise Unsupported(
                        f"layer {name}: declared segments {decl.segments} != server copy "
                        f"size bytes_per_block={view.bytes_per_block} per segment"
                    )
                if len(decl.segments) > 1 and view.kv_stride_bytes < view.bytes_per_block:
                    raise Unsupported(
                        f"layer {name}: kv_stride_bytes={view.kv_stride_bytes} < "
                        f"bytes_per_block={view.bytes_per_block}: segments overlap"
                    )
                names.append(name)
                wrappers.append(self._wrap(view.tensor))
                num_blocks.append(view.num_blocks)
                bytes_per_block.append(view.bytes_per_block)
                kv_strides.append(view.kv_stride_bytes)
                segments.append(len(decl.segments))
                group_ids.append(sid)
                self._layers[sid].append(name)
                self._layer_bytes[name] = view.bytes_per_block * len(decl.segments)
        if not names:
            raise Unsupported("no transferable KV cache layer on this worker")
        par = self._par
        ok, msg = self._rpc.register_context_batch(
            self._instance_id,
            self._layout.namespace,
            par.tp_rank,
            par.pp_rank,
            par.tp_size,
            par.world_size,
            self._device_id,
            names,
            wrappers,
            num_blocks,
            bytes_per_block,
            kv_strides,
            segments,
            self._layout.transfer_backend,
            self._layout.page_first,
            group_ids,
        )
        if not ok:
            raise RuntimeError(f"PegaFlow register_context_batch failed: {msg}")
        logger.info(
            "[PegaKVConnectorV2] registered %d layers instance=%s namespace=%s device=%d",
            len(names),
            self._instance_id,
            self._layout.namespace,
            self._device_id,
        )

    def start(self, plan: StepPlan) -> None:
        """Start this step's loads and track requests vLLM holds for our saves.

        Called before ``save`` of the same step, so every job that can read a
        finished request's blocks (it was scheduled at most in the previous
        step) is already submitted: its fence is ``_submitted``.
        """
        for req_id in plan.finished:
            self._finishing[req_id] = self._submitted
        self._start_loads(plan)

    def _start_loads(self, plan: StepPlan) -> None:
        """One ``load`` RPC per request.

        The server consumes leases one by one and fails the whole call on the
        first bad one (e.g. an expired lease), so batching requests would fail
        every request of the step for one bad lease.

        Sync loads are waited for before returning (vLLM calls this before
        the forward when the step has any); a failed one reports its blocks
        as invalid. Async loads report through ``poll``.
        """
        sync: list[tuple[Any, list[int]]] = []
        for lp in plan.loads:
            if lp.fail:
                if lp.sync:
                    self._invalid.update(lp.covered)
                else:
                    self._failed.add(lp.req_id)
            elif not lp.loads:
                if not lp.sync:
                    self._done.add(lp.req_id)  # nothing to copy, still owes a completion
            else:
                state = self._start_load(lp.req_id, lp.loads)
                if state is None:
                    if lp.sync:
                        self._invalid.update(lp.covered)
                    else:
                        self._failed.add(lp.req_id)
                elif lp.sync:
                    sync.append((state, lp.covered))
                else:
                    self._batches.append(_Batch(state, [lp.req_id]))
        if sync:
            self._wait_sync(sync)

    def _start_load(self, req_id: str, loads: list[tuple[bytes, list[list[int | None]]]]) -> Any:
        """Issue the load RPC; the load state, or None if the RPC failed."""
        try:
            state = self._new_load_state()
            ok, msg = self._rpc.load(
                self._instance_id,
                self._par.tp_rank,
                self._device_id,
                state.shm_name(),
                self._layers,
                loads,
            )
        except Exception as e:
            ok, msg = False, repr(e)
        if not ok:
            logger.warning("[PegaKVConnectorV2] load failed req=%s: %s", req_id, msg)
            return None
        return state

    def _wait_sync(self, loads: list[tuple[Any, list[int]]]) -> None:
        """Wait until every sync load reached a final state.

        Short by construction (policy ``sync_load_tokens``): ~1 ms per load.
        There is no deadline: until the server signals, its copy may still
        land in the blocks, so neither "failed" (vLLM would recompute into
        them) nor "done" is safe to report. Polls with a growing sleep.
        """
        t0 = self._clock()
        next_log = t0 + self.STALL_LOG_S
        for state, covered in loads:
            delay, cap = self.SYNC_POLL_S
            while not state.is_ready():
                time.sleep(delay)
                delay = min(delay * 2, cap)
                if self._clock() >= next_log:
                    next_log += self.STALL_LOG_S
                    logger.warning(
                        "[PegaKVConnectorV2] sync load still pending after %.0f s",
                        self._clock() - t0,
                    )
            if state.get_state() != 1:
                logger.warning(
                    "[PegaKVConnectorV2] sync load failed state=%s blocks=%d",
                    state.get_state(),
                    len(covered),
                )
                self._invalid.update(covered)

    def take_invalid(self) -> set[int]:
        """GPU blocks whose sync load failed since the last call."""
        out, self._invalid = self._invalid, set()
        return out

    # ----------------------------------------------------------------- saves
    def save(self, plan: StepPlan) -> None:
        """Queue this step's saves behind a mark on the compute stream.

        Never blocks, except with ``sync_saves`` (then it waits for them).
        """
        t0 = self._clock()
        saves: list[tuple[str, list[int], list[bytes]]] = []
        for gs in plan.saves:
            for name in self._layers[gs.sid]:
                saves.append((name, gs.block_ids, gs.block_hashes))
        if saves and self._saver_dead:
            with self._cv:
                self.save_stats.failed += 1
            saves = []
        if saves:
            # The mark covers this step's forward (and draft forward): the
            # saver only reads the blocks after the KV writes completed.
            ready = self._mark_compute()
            self._submitted += 1
            self._jobs.put(_SaveJob(self._submitted, ready, saves))
        t1 = self._clock()
        with self._cv:
            stats = self.save_stats
            stats.steps += 1
            stats.submit_s += t1 - t0
            if saves:
                stats.save_steps += 1
                stats.blocks += sum(len(ids) for _, ids, _ in saves)
                stats.bytes += sum(len(ids) * self._layer_bytes[name] for name, ids, _ in saves)
                stats.max_pending = max(stats.max_pending, self._submitted - self._completed)
            report = stats.steps >= self.SAVE_LOG_EVERY
        if saves and self._sync_saves:
            self.flush()
        if report:
            self.report_save_stats()

    def _settled(self, target: int) -> bool:
        return self._saver_dead or self._completed >= target

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every submitted save job is done; False on timeout.

        Without a timeout it waits for good: a save RPC has no deadline (the
        client offers none), and a timed-out one may still be copying on the
        server, so its blocks cannot be handed back early. It logs while stuck.
        """
        target = self._submitted
        t0 = self._clock()
        with self._cv:
            if self._settled(target):
                return True
            if timeout is not None:
                done = self._cv.wait_for(lambda: self._settled(target), timeout)
            else:
                while not (
                    done := self._cv.wait_for(lambda: self._settled(target), self.STALL_LOG_S)
                ):
                    logger.error(
                        "[PegaKVConnectorV2] save flush waiting %.0f s (job %d of %d done)",
                        self._clock() - t0,
                        self._completed,
                        target,
                    )
            self.save_stats.flushes += 1
            self.save_stats.flush_s += self._clock() - t0
        return done

    def _save_loop(self) -> None:
        try:
            while (job := self._jobs.get()) is not None:
                self._run_job(job)
        finally:
            with self._cv:
                self._saver_dead = True
                self._cv.notify_all()

    def _run_job(self, job: _SaveJob) -> None:
        t0 = self._clock()
        t1 = t2 = t0
        ok = False
        try:
            try:
                job.ready.synchronize()
            except Exception as e:  # the forward failed: its KV is not valid
                ok, msg = False, f"compute wait failed: {e!r}"
            else:
                t1 = self._clock()
                try:
                    ok, msg = self._rpc.save(
                        self._instance_id,
                        self._par.tp_rank,
                        self._par.pp_rank,
                        self._device_id,
                        job.saves,
                    )
                except Exception as e:
                    ok, msg = False, repr(e)
            t2 = self._clock()
            if not ok:
                # Best effort: a dropped save is a future miss, not a wrong hit.
                logger.warning("[PegaKVConnectorV2] save failed: %s", msg)
        finally:
            with self._cv:
                stats = self.save_stats
                stats.wait_s += t1 - t0
                stats.rpc_s += t2 - t1
                stats.max_job_s = max(stats.max_job_s, t2 - t0)
                stats.failed += not ok
                # Always advance, failed or not: held blocks must be released.
                self._completed = job.seq
                self._cv.notify_all()

    def report_save_stats(self) -> None:
        """Log and reset the save-cost window (grep "save stats" in vllm.log)."""
        now = self._clock()
        with self._cv:
            s, self.save_stats = self.save_stats, SaveStats(_since=now)
        if not s.steps:
            return
        logger.info(
            "[PegaKVConnectorV2] save stats tp=%d window_s=%.1f steps=%d save_steps=%d "
            "blocks=%d bytes=%d submit_ms=%.1f flushes=%d flush_ms=%.1f wait_ms=%.1f "
            "rpc_ms=%.1f max_job_ms=%.2f max_pending=%d failed=%d",
            self._par.tp_rank,
            now - s._since,
            s.steps,
            s.save_steps,
            s.blocks,
            s.bytes,
            s.submit_s * 1e3,
            s.flushes,
            s.flush_s * 1e3,
            s.wait_s * 1e3,
            s.rpc_s * 1e3,
            s.max_job_s * 1e3,
            s.max_pending,
            s.failed,
        )

    def shutdown(self) -> None:
        if not self.flush(self.SHUTDOWN_FLUSH_S):
            logger.warning("[PegaKVConnectorV2] saves still pending at shutdown")
        self._jobs.put(None)
        self._saver.join(timeout=1.0)
        self.report_save_stats()

    # ------------------------------------------------------------------ poll
    def poll(self) -> tuple[set[str], set[str], set[str]]:
        """Returns (finished_sending, finished_recving, failed_recving).

        Failed receives are in both receive sets. A finished request is
        reported as sent once every job that could read its blocks is done.
        """
        sending: set[str] = set()
        if self._finishing:
            with self._cv:
                completed = self._submitted if self._saver_dead else self._completed
            sending = {r for r, fence in self._finishing.items() if fence <= completed}
            for r in sending:
                del self._finishing[r]
        finished, self._done = self._done, set()
        pending: list[_Batch] = []
        for b in self._batches:
            if not b.state.is_ready():
                pending.append(b)
                continue
            finished.update(b.req_ids)
            state = b.state.get_state()
            if state != 1:
                logger.warning("[PegaKVConnectorV2] load state=%s reqs=%s", state, b.req_ids)
                self._failed.update(b.req_ids)
        self._batches = pending
        failed, self._failed = self._failed, set()
        return sending, finished | failed, failed
