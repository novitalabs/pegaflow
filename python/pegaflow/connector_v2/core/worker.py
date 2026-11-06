"""Worker-side glue: register KV caches, run the step's loads and saves.

vLLM- and torch-free: tensors are opaque, and the connector injects ``wrap``
(tensor -> CUDA IPC bytes), ``new_load_state`` and ``mark_compute`` (returns a
handle whose ``synchronize()`` waits until the work enqueued so far on the
compute stream is done, e.g. a recorded CUDA event), so unit tests can drive it
with fakes.

Saves are asynchronous: ``save`` only marks the compute stream and queues the
job; a saver thread waits for the mark and then issues the save RPC (the server
copies GPU->CPU on its own stream and replies once the copy is done). The
source blocks must stay untouched until the copy is done, so before a step
writes any KV:

- a step that writes a block vLLM freed while a save may still read it
  (``StepPlan.fence_step``) first calls ``wait_saved``: every save of plans up
  to that step is done;
- a step that reuses a preempted request's blocks first calls ``flush``.

Both run in vLLM's ``handle_preemptions``, which vLLM's V1 model runner
calls before it zeroes or copies (copy-on-write) any block, but the V2 runner
calls after. Layouts where vLLM may zero or copy blocks, or free a checkpoint
state block right after offering it, use pinned saves instead: the scheduler
keeps every block it plans to save allocated in vLLM's block pool until all
workers report the step's saves done (``saved_through``), so no such block is
reallocated meanwhile.
"""

from __future__ import annotations

import collections
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


def _save_settled(e: Exception) -> bool:
    """True if a raised save RPC proves that no copy of the job runs.

    Only an InvalidArgument status does: the binding raises ``ValueError``
    for it, tonic's client never makes it, and the save handler returns it
    before it copies or after its copy synchronized. Every other error may
    leave a copy reading the job's blocks:

    - Internal: the handler also returns it when a GPU copy failed
      part-way with earlier copies still queued, and the client itself
      reports decode and HTTP/2 errors as Internal;
    - no reply (Unavailable, Cancelled, DeadlineExceeded, Unknown): the
      handler may still run;
    - FailedPrecondition (unknown instance) raises the same exception
      class as those, so it fails closed too.
    """
    return isinstance(e, ValueError)


@dataclass
class _Batch:
    state: Any  # PyLoadState
    req_ids: list[str]
    polls: int = 0  # poll() calls that saw it, the landing one included
    issued_at: float = 0.0  # worker clock when its step's start_load_kv began


@dataclass
class _SaveJob:
    seq: int
    ready: Any  # mark_compute() handle
    saves: list[tuple[str, list[int], list[bytes]]]


def _wire_bytes(saves: list[tuple[str, list[int], list[bytes]]]) -> int:
    """Upper bound on the protobuf size of these SaveLayer entries.

    Per entry: name plus framing; per block: a uint32 id (<= 5 B varint) and a
    length-prefixed hash (+ 2 B tag/length, +1 B packed-field slack).
    """
    return sum(
        len(name) + 16 + sum(len(h) for h in hashes) + 8 * len(ids) for name, ids, hashes in saves
    )


@dataclass
class SaveStats:
    """Save cost since the last report (grep "save stats" in vllm.log), and
    the sync-load wait, the worker's other cost on the step's critical path."""

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
    raised: int = 0  # save RPCs that raised with a copy possibly running (each quiesces)
    rpcs: int = 0  # save RPCs sent (one per released batch not dropped by a quiesce)
    pressure_rpcs: int = 0  # batches released early on KV pressure (StepPlan.send_saves)
    kv_usage_unknown_steps: int = 0  # steps without vLLM's KV usage (batching off)
    fences: int = 0  # steps that waited for saves before writing a freed block
    fence_s: float = 0.0  # step thread: blocked on those waits (on the critical path)
    max_fence_s: float = 0.0
    sync_loads: int = 0  # sync loads whose RPC was accepted (failed RPCs are not waited for)
    sync_first_steps: int = 0  # steps where a sync load was moved ahead of an async one
    sync_wait_s: float = 0.0  # step thread: waiting for their copies (on the critical path)
    max_sync_wait_s: float = 0.0  # longest wait of one step
    load_steps: int = 0  # steps with any load plan
    load_issue_s: float = 0.0  # step thread: all of _start_loads (RPCs + sync wait)
    max_load_issue_s: float = 0.0
    async_done: int = 0  # async load RPCs seen ready (landed or failed) by poll()
    async_first_poll: int = 0  # of those, ready at the first poll (the issuing step's)
    async_polls: int = 0  # their polls, the landing one included
    max_async_polls: int = 0
    async_ready_s: float = 0.0  # their step's start_load_kv -> the poll that saw them ready
    max_async_ready_s: float = 0.0
    _since: float = field(default=0.0, repr=False)


class Worker:
    # wait_for_save calls between save-cost log lines.
    SAVE_LOG_EVERY = 500
    SHUTDOWN_FLUSH_S = 30.0
    # Waits that never give up (sync loads, flush) log this often.
    STALL_LOG_S = 10.0
    # Sync-load poll interval: doubles from the first value up to the cap.
    SYNC_POLL_S = (20e-6, 200e-6)
    # Staleness cap for batched saves, not a tuned policy: held saves are sent
    # once the oldest has waited this long, so other requests see a step's
    # blocks in the store within about this bound when steps are slow. It is
    # checked once per step, so a hold can exceed it by about one step.
    SAVE_HOLD_MAX_S = 1.0
    # Wire-size budget of one merged save RPC: a quarter of the 64 MiB gRPC
    # message limit (python/src/lib.rs). A step whose job would push the held
    # batch past it sends the batch first, so a merged RPC is at most this big
    # (or one step's size, the unbatched behaviour, if a single step is bigger).
    SAVE_RPC_MAX_BYTES = 16 << 20
    # Quiesce window after a save RPC raised without proving its copy over
    # (``_save_settled``): no save RPC is sent and the job's fences and pins
    # hold for this long, then the job counts as done and saving resumes.
    # A heuristic, not a proof: the server reports no bound on a copy it
    # may still run. Gate logs (277 v2 runs, 3853 ``save stats`` windows)
    # show a whole job (compute wait + RPC) at p99 0.63 s and at most 2.98 s
    # (one 557 MB RPC); 10 s is over 3x that. It is also the longest a fence
    # or pin on the raised job can hold vLLM's step. Each consecutive raise
    # doubles the window up to SAVE_QUIESCE_MAX_S, so a cause that persists
    # (a pinned pool that stays exhausted) stalls fences less often; a save
    # RPC that stores its blocks resets it.
    SAVE_QUIESCE_S = 10.0
    SAVE_QUIESCE_MAX_S = 60.0

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
        batch_steps: int = 1,
        pinned_saves: bool = False,
    ):
        """``batch_steps`` (policy ``save_batch_steps``): a step's save job is
        held and merged with later steps' jobs into one save RPC until the
        oldest held job has seen ``batch_steps`` steps or waited
        ``SAVE_HOLD_MAX_S``, whichever comes first (<= 1 sends every step).
        ``flush`` and ``StepPlan.send_saves`` send held jobs at once.

        ``pinned_saves``: the scheduler pins the blocks of each plan's saves
        until ``saved_through`` reports that plan's step (``Scheduler``
        ``pin_saves``), so saves need no batching.
        """
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._par = par
        # Server-side rank: TP replicas share one (layout.StorageRanks).
        storage = layout.storage
        self._storage_rank = storage.rank[par.tp_rank]
        self._writer = storage.writer[par.tp_rank]
        self._device_id = device_id
        self._wrap = wrap
        self._new_load_state = new_load_state
        self._mark_compute = mark_compute
        self._clock = clock
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
        # (plan step, _submitted after that step's save) for steps whose jobs
        # are not known done, in step order; ``_done_seq`` is the newest
        # sequence number dropped from the front (done). A fence on a step
        # maps to that step's entry (``_fence_of``).
        self._step_seqs: collections.deque[tuple[int, int]] = collections.deque()
        self._done_seq = 0
        # Save batching (step thread only). Held jobs already own their
        # sequence numbers, so fences and flush cover them.
        self._batch_steps = batch_steps
        self._held: list[_SaveJob] = []
        self._held_steps = 0  # steps seen by the oldest held job
        self._held_since = 0.0
        self._held_bytes = 0  # estimated wire size of the held jobs' save entries
        # (job seq, plan step) of submitted jobs not known done, and the last
        # plan step seen by ``save`` (``saved_through``).
        self._pinned = pinned_saves
        self._job_steps: collections.deque[tuple[int, int]] = collections.deque()
        self._last_step = -1
        # The saver thread exited: it reads no block any more, so every hold
        # is released and later saves are dropped (future misses).
        self._saver_dead = False
        # A save RPC raised and its window (SAVE_QUIESCE_S) runs: new saves
        # are dropped (future misses). Set and cleared by the saver thread.
        self._quiescing = False
        self._quiesce_s = self.SAVE_QUIESCE_S  # next window (saver thread only)
        self._stop = threading.Event()  # shutdown: ends a quiesce window early
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
            self._storage_rank,
            par.pp_rank,
            self._layout.storage.tp_size,
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
            "[PegaKVConnectorV2] registered %d layers instance=%s namespace=%s device=%d "
            "storage_rank=%d writer=%s",
            len(names),
            self._instance_id,
            self._layout.namespace,
            self._device_id,
            self._storage_rank,
            self._writer,
        )

    def start(self, plan: StepPlan) -> None:
        """Start this step's loads."""
        self._start_loads(plan)

    def _fence_of(self, step: int) -> int:
        """The sequence number of the last job submitted by the end of plan
        ``step``'s ``save``: once it is done, so is every job of that step and
        of the steps before it (one saver thread, jobs complete in order)."""
        if step > self._last_step:
            return self._submitted  # not seen yet (cannot happen in vLLM): wait for all
        fence = self._done_seq
        for s, seq in self._step_seqs:
            if s > step:
                break
            fence = seq
        return fence

    def _forget_done_steps(self, completed: int) -> None:
        steps = self._step_seqs
        while steps and steps[0][1] <= completed:
            self._done_seq = steps.popleft()[1]

    def _start_loads(self, plan: StepPlan) -> None:
        """One ``load`` RPC per request, the step's sync loads first.

        The server consumes leases one by one and fails the whole call on the
        first bad one (e.g. an expired lease), so batching requests would fail
        every request of the step for one bad lease.

        Sync loads are waited for before returning (vLLM calls this before
        the forward when the step has any); a failed one reports its blocks
        as invalid. Async loads report through ``poll``. The server copies a
        device's loads one at a time in arrival order, so a sync load sent
        after the step's async loads would stall the step for their copies
        too.
        """
        if not plan.loads:
            return
        t0 = self._clock()
        sync: list[tuple[Any, list[int]]] = []
        ordered = sorted(plan.loads, key=lambda lp: not lp.sync)
        reordered = ordered != plan.loads
        for lp in ordered:
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
                    self._batches.append(_Batch(state, [lp.req_id], issued_at=t0))
        if sync:
            self._wait_sync(sync)
        issued = self._clock() - t0
        with self._cv:
            stats = self.save_stats
            stats.sync_first_steps += reordered
            stats.load_steps += 1
            stats.load_issue_s += issued
            stats.max_load_issue_s = max(stats.max_load_issue_s, issued)

    def _start_load(self, req_id: str, loads: list[tuple[bytes, list[list[int | None]]]]) -> Any:
        """Issue the load RPC; the load state, or None if the RPC failed."""
        try:
            state = self._new_load_state()
            ok, msg = self._rpc.load(
                self._instance_id,
                self._storage_rank,
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
        waited = self._clock() - t0
        with self._cv:
            stats = self.save_stats
            stats.sync_loads += len(loads)
            stats.sync_wait_s += waited
            stats.max_sync_wait_s = max(stats.max_sync_wait_s, waited)

    def take_invalid(self) -> set[int]:
        """GPU blocks whose sync load failed since the last call."""
        out, self._invalid = self._invalid, set()
        return out

    # ----------------------------------------------------------------- saves
    def save(self, plan: StepPlan) -> None:
        """Queue this step's saves behind a mark on the compute stream.

        Never blocks. With batching, the job is held and sent merged with later steps' jobs
        (``_release_due``). vLLM calls this every engine step, also steps
        without a forward, so held jobs keep aging while requests wait on
        ``finished_sending``.
        """
        t0 = self._clock()
        saves: list[tuple[str, list[int], list[bytes]]] = []
        # A replica that is not its part's writer stores nothing: the writer's
        # copy seals the block and every holder loads from it.
        for gs in plan.saves if self._writer else ():
            for name in self._layers[gs.sid]:
                saves.append((name, gs.block_ids, gs.block_hashes))
        if saves and (self._saver_dead or self._quiescing):
            with self._cv:
                self.save_stats.failed += 1
            saves = []
        if saves:
            # The mark covers this step's forward (and draft forward): the
            # saver only reads the blocks after the KV writes completed.
            ready = self._mark_compute()
            self._submitted += 1
            self._job_steps.append((self._submitted, plan.step))
            size = _wire_bytes(saves)
            if self._held and self._held_bytes + size > self.SAVE_RPC_MAX_BYTES:
                # One merged RPC must stay well under the gRPC message limit.
                self._release_held()
            if not self._held:
                self._held_steps, self._held_since = 0, t0
            self._held.append(_SaveJob(self._submitted, ready, saves))
            self._held_bytes += size
        if self._held:
            self._held_steps += 1
            if self._release_due(t0):
                self._release_held()
            elif plan.send_saves:
                # KV pressure: freed blocks are reused soon, and a fence on a
                # held job would wait for the whole batch.
                with self._cv:
                    self.save_stats.pressure_rpcs += 1
                self._release_held()
        if plan.step > self._last_step:
            self._last_step = plan.step
            self._step_seqs.append((plan.step, self._submitted))
        t1 = self._clock()
        with self._cv:
            stats = self.save_stats
            stats.steps += 1
            stats.submit_s += t1 - t0
            stats.kv_usage_unknown_steps += plan.kv_usage_unknown
            if saves:
                stats.save_steps += 1
                stats.blocks += sum(len(ids) for _, ids, _ in saves)
                stats.bytes += sum(len(ids) * self._layer_bytes[name] for name, ids, _ in saves)
                stats.max_pending = max(stats.max_pending, self._submitted - self._completed)
            report = stats.steps >= self.SAVE_LOG_EVERY
        if report:
            self.report_save_stats()

    @property
    def pinned_saves(self) -> bool:
        return self._pinned

    def saved_through(self) -> int:
        """The latest plan step whose saves, and every earlier one's, are done
        (failed or not: a failed save is a future miss, its blocks may go)."""
        with self._cv:
            completed = self._submitted if self._saver_dead else self._completed
        steps = self._job_steps
        while steps and steps[0][0] <= completed:
            steps.popleft()
        return steps[0][1] - 1 if steps else self._last_step

    def _release_due(self, now: float) -> bool:
        if self._batch_steps <= 1:
            return True
        return (
            self._held_steps >= self._batch_steps or now - self._held_since >= self.SAVE_HOLD_MAX_S
        )

    def _release_held(self) -> None:
        """Send the held jobs as one: the last job's sequence number and mark.

        The compute stream is in order, so the last mark covers every earlier
        step's forward, and completing the last sequence number completes them
        all. Each step keeps its own layer entries (a layer may appear once per
        held step), so the server sees the same entries as unbatched saves.
        The server pins one region per entry, freed only once all of its
        blocks are evicted: folding steps into one entry per layer makes
        regions span more requests and strands more evicted bytes (measured
        in the full-only batching A/B, rsi-notes).
        """
        if not self._held:
            return
        jobs, self._held = self._held, []
        self._held_bytes = 0
        job = jobs[-1]
        if len(jobs) > 1:
            job = _SaveJob(job.seq, job.ready, [s for j in jobs for s in j.saves])
        self._jobs.put(job)

    def _settled(self, target: int) -> bool:
        return self._saver_dead or self._completed >= target

    def _log_stall(self, what: str, t0: float, target: int) -> None:
        """A stuck fence or flush (under ``_cv``). Waiting out a quiesce window
        is expected (bounded), not a stall."""
        log = logger.warning if self._quiescing else logger.error
        log(
            "[PegaKVConnectorV2] save %s waiting %.0f s (job %d of %d done%s)",
            what,
            self._clock() - t0,
            self._completed,
            target,
            ", save quiesce window" if self._quiescing else "",
        )

    def wait_saved(self, step: int) -> None:
        """Block until the saves of plans up to ``step`` are done (the GPU->CPU
        copies completed, or the save failed): this step writes blocks they
        read. Held (batched) jobs among them are sent first. Waits for good,
        like ``flush``."""
        target = self._fence_of(step)
        if self._held and self._held[0].seq <= target:
            self._release_held()
        t0 = self._clock()
        with self._cv:
            if self._settled(target):
                return
            while not self._cv.wait_for(lambda: self._settled(target), self.STALL_LOG_S):
                self._log_stall("fence", t0, target)
            waited = self._clock() - t0
            stats = self.save_stats
            stats.fences += 1
            stats.fence_s += waited
            stats.max_fence_s = max(stats.max_fence_s, waited)

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every submitted save job is done; False on timeout.

        Without a timeout it waits for good: a save RPC has no deadline (the
        client offers none), and a timed-out one may still be copying on the
        server, so its blocks cannot be handed back early. It logs while stuck.
        Held (batched) jobs are sent first.
        """
        self._release_held()
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
                    self._log_stall("flush", t0, target)
            self.save_stats.flushes += 1
            self.save_stats.flush_s += self._clock() - t0
        return done

    def _save_loop(self) -> None:
        try:
            while (job := self._jobs.get()) is not None:
                if not self._run_job(job) and not self._quiesce(job):
                    break
        finally:
            with self._cv:
                self._saver_dead = True
                self._cv.notify_all()

    def _quiesce(self, raised: _SaveJob) -> bool:
        """After ``raised``'s RPC raised: send nothing for the window
        (``SAVE_QUIESCE_S``, doubled per consecutive raise) while its fences
        and pins hold (its copy may still run), then count it done and
        resume. Jobs queued behind it, held (batched) jobs released during
        the window included, are completed as failed without an RPC (no copy
        of theirs started): under the pressure that made the RPC raise they
        would likely raise too, each holding fences for another window.
        Shutdown ends the window early. False if shutdown came during it."""
        window = self._quiesce_s
        self._quiesce_s = min(2 * window, self.SAVE_QUIESCE_MAX_S)
        self._quiesce_wait(window)
        last, dropped, stop = raised.seq, 0, False
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is None:
                stop = True
                break
            last, dropped = job.seq, dropped + 1
        with self._cv:
            self._completed = last
            self.save_stats.failed += dropped
            self._quiescing = False
            self._cv.notify_all()
        logger.warning(
            "[PegaKVConnectorV2] saves resume after %.0f s quiesce (%d queued jobs dropped)",
            window,
            dropped,
        )
        return not stop

    def _quiesce_wait(self, seconds: float) -> None:  # tests replace it
        self._stop.wait(seconds)

    def _run_job(self, job: _SaveJob) -> bool:
        """Run one save job; False if its copy may still run.

        A save RPC that raises (``_save_settled`` false) is not a finished
        save: the server may still copy the job's blocks. The job is not
        counted as completed, so its fences and pins (and every later job's)
        hold, and ``_quiescing`` drops new saves; ``_quiesce`` then waits
        out the window and resumes. A saver thread that dies mid-job stops
        saving for good instead (``_saver_dead``): it reads no block any
        more, so every fence and hold is released and later saves are
        dropped. Only the raised job's copy may race a reuse, and only if
        it outlasts the window; closing that needs a server that reports
        such a copy only once it stopped (a server limitation).
        A failed reply, a rejected request (``_save_settled``), or a failed
        compute wait (no RPC sent) completes the job as a failed save: a
        future miss.
        """
        t0 = self._clock()
        t1 = t2 = t0
        # over: no copy of this job can still be running
        ok = over = raised = sent = False
        try:
            try:
                job.ready.synchronize()
            except Exception as e:  # the forward failed: its KV is not valid
                ok, over, msg = False, True, f"compute wait failed: {e!r}"
            else:
                t1 = self._clock()
                sent = True
                try:
                    ok, msg = self._rpc.save(
                        self._instance_id,
                        self._storage_rank,
                        self._par.pp_rank,
                        self._device_id,
                        job.saves,
                    )
                    over = True
                except Exception as e:
                    ok, over, msg = False, _save_settled(e), repr(e)
                    raised = not over
            t2 = self._clock()
            if not over:
                logger.error(
                    "[PegaKVConnectorV2] save RPC raised, saves pause %.0f s (fail closed): %s",
                    self._quiesce_s,
                    msg,
                )
            elif not ok:
                # Best effort: a dropped save is a future miss, not a wrong hit.
                logger.warning("[PegaKVConnectorV2] save failed: %s", msg)
        finally:
            with self._cv:
                stats = self.save_stats
                stats.wait_s += t1 - t0
                stats.rpc_s += t2 - t1
                stats.max_job_s = max(stats.max_job_s, t2 - t0)
                stats.failed += not ok
                stats.rpcs += sent
                if ok:
                    self._quiesce_s = self.SAVE_QUIESCE_S  # stored: backoff resets
                if over:
                    # Failed or not, the copy is over: held blocks are released.
                    self._completed = job.seq
                elif raised:  # the thread lives on and quiesces
                    stats.raised += 1
                    self._quiescing = True
                else:  # the thread dies mid-job
                    self._saver_dead = True
                self._cv.notify_all()
        return over

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
            "rpc_ms=%.1f max_job_ms=%.2f max_pending=%d failed=%d raised=%d rpcs=%d "
            "pressure_rpcs=%d fences=%d fence_ms=%.1f max_fence_ms=%.2f kv_usage_unknown_steps=%d "
            "sync_loads=%d sync_first_steps=%d sync_wait_ms=%.1f max_sync_wait_ms=%.2f "
            "load_steps=%d load_issue_ms=%.1f max_load_issue_ms=%.2f "
            "async_done=%d async_first_poll=%d async_polls=%d max_async_polls=%d "
            "async_ready_ms=%.1f max_async_ready_ms=%.2f",
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
            s.raised,
            s.rpcs,
            s.pressure_rpcs,
            s.fences,
            s.fence_s * 1e3,
            s.max_fence_s * 1e3,
            s.kv_usage_unknown_steps,
            s.sync_loads,
            s.sync_first_steps,
            s.sync_wait_s * 1e3,
            s.max_sync_wait_s * 1e3,
            s.load_steps,
            s.load_issue_s * 1e3,
            s.max_load_issue_s * 1e3,
            s.async_done,
            s.async_first_poll,
            s.async_polls,
            s.max_async_polls,
            s.async_ready_s * 1e3,
            s.max_async_ready_s * 1e3,
        )

    def shutdown(self) -> None:
        self._stop.set()  # a quiesce window ends now: its queued jobs are dropped
        if not self.flush(self.SHUTDOWN_FLUSH_S):
            logger.warning("[PegaKVConnectorV2] saves still pending at shutdown")
        self._jobs.put(None)
        self._saver.join(timeout=1.0)
        self.report_save_stats()

    # ------------------------------------------------------------------ poll
    def poll(self) -> tuple[set[str], set[str], set[str]]:
        """Returns (finished_sending, finished_recving, failed_recving).

        Failed receives are in both receive sets. Nothing is ever sending:
        finished requests are not held (fences cover their blocks).
        """
        sending: set[str] = set()
        with self._cv:
            completed = self._submitted if self._saver_dead else self._completed
        self._forget_done_steps(completed)
        finished, self._done = self._done, set()
        pending: list[_Batch] = []
        landed: list[int] = []
        ready_s: list[float] = []
        now = self._clock() if self._batches else 0.0
        for b in self._batches:
            b.polls += 1
            if not b.state.is_ready():
                pending.append(b)
                continue
            landed.append(b.polls)
            ready_s.append(now - b.issued_at)
            finished.update(b.req_ids)
            state = b.state.get_state()
            if state != 1:
                logger.warning("[PegaKVConnectorV2] load state=%s reqs=%s", state, b.req_ids)
                self._failed.update(b.req_ids)
        self._batches = pending
        if landed:
            with self._cv:
                stats = self.save_stats
                stats.async_done += len(landed)
                stats.async_first_poll += landed.count(1)
                stats.async_polls += sum(landed)
                stats.max_async_polls = max(stats.max_async_polls, *landed)
                stats.async_ready_s += sum(ready_s)
                stats.max_async_ready_s = max(stats.max_async_ready_s, *ready_s)
        failed, self._failed = self._failed, set()
        return sending, finished | failed, failed
