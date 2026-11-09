"""Worker side: register this rank's KV cache, run each step's loads and saves.

torch-free: the connector injects ``share`` (tensor -> region facts),
``new_load_state`` and ``mark_compute`` (a handle whose ``synchronize()``
waits for the work enqueued so far on the compute stream), so unit tests can
drive it with fakes.

Spans: vLLM 0.32 puts the whole KV cache of a rank in one allocation, layer
``l`` of block ``b`` at ``offset_l + b * block_stride_l``. One object (one
block of one namespace) is one ``page_bytes`` span per layer.

Savers: per slice, one rank on each CPU socket saves (``NsSpec.savers``), so
a load reads a copy on its own socket.

Loads: one ``load_objects`` per step for every request it loads, as v1. Every
failure is reported as a failed receive; vLLM applies its
``kv_load_failure_policy`` (recompute, or fail the request).

Saves are asynchronous: ``save`` marks the compute stream and queues the job;
a saver thread waits for the marks, then sends every queued job in one
``save_objects`` (the server replies once its copy is done). The scheduler
keeps the blocks pinned until every worker reports the step through
``saved_through``.

Stats: each load and save RPC is recorded under v1's
``PegaKVConnectorStats`` keys (``take_stats``).
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .layout import Layout, Unsupported
from .scheduler import StepPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """Where a tensor sits in its CUDA allocation (from ``share``)."""

    handle: bytes  # 64-byte cudaIpcMemHandle of the allocation
    size: int  # bytes from the allocation base to the end of the storage
    offset: int  # tensor data_ptr - allocation base


@dataclass
class _NsGeometry:
    region: int
    offsets: np.ndarray  # per layer, bytes into the region
    strides: np.ndarray  # per layer, bytes between blocks
    page: int


@dataclass
class _SaveJob:
    seq: int
    ready: Any
    writes: list[tuple[tuple[bytes, bytes], int, bytes]]


class Worker:
    # After a save RPC raised without proving its copy over, wait this long
    # (pins held) before counting the job done.
    SAVE_QUIESCE_S = 10.0
    SHUTDOWN_FLUSH_S = 30.0

    def __init__(
        self,
        layout: Layout,
        rpc: Any,
        instance_id: str,
        rank: int,
        world_size: int,
        device_id: int,
        new_load_state: Callable[[], Any],
        mark_compute: Callable[[], Any],
        transfer_backend: str = "kernel",
    ):
        self._layout = layout
        self._rpc = rpc
        self._instance_id = instance_id
        self._rank = rank
        self._world_size = world_size
        self._device_id = device_id
        self._new_load_state = new_load_state
        self._mark_compute = mark_compute
        self._transfer_backend = transfer_backend
        self._geo: list[_NsGeometry] = []
        self._saver_of: list[bool] = []
        # One load state per step: (state, req ids, start, blocks).
        self._loads: list[tuple[Any, list[str], float, int]] = []
        self._failed: set[str] = set()
        # Async saves complete in submit order (one saver thread).
        self._cv = threading.Condition()
        self._submitted = 0
        self._completed = 0
        self._job_steps: collections.deque[tuple[int, int]] = collections.deque()
        self._last_step = -1
        self._stats: dict[str, Any] = {}  # guarded by _cv (the saver thread records)
        self._stop = threading.Event()
        self._jobs: queue.SimpleQueue[_SaveJob | None] = queue.SimpleQueue()
        self._saver = threading.Thread(
            target=self._save_loop, name="pegaflow-v2-saver", daemon=True
        )
        self._saver.start()

    # -------------------------------------------------------------- register
    def register(
        self,
        kv_caches: dict[str, Any],
        block_strides: dict[str, int],
        share: Callable[[Any], Region],
        sockets: list[int],
    ) -> None:
        """``block_strides``: layer -> bytes between its blocks (vLLM
        ``KVCacheTensor.block_stride``). ``sockets``: every rank's CPU socket."""
        handles: dict[bytes, int] = {}
        regions: list[tuple[bytes, int, str]] = []
        metas = []
        for ns in self._layout.namespaces:
            region = None
            offsets, strides = [], []
            for name in ns.layers:
                if name not in kv_caches:
                    raise Unsupported(f"layer {name} has no KV cache on this rank")
                r = share(kv_caches[name])
                idx = handles.get(r.handle)
                if idx is None:
                    idx = handles[r.handle] = len(regions)
                    regions.append((r.handle, r.size, f"kv{idx}"))
                elif r.size > regions[idx][1]:
                    regions[idx] = (r.handle, r.size, regions[idx][2])
                if region is None:
                    region = idx
                elif region != idx:
                    raise Unsupported(f"namespace of group {ns.gid} spans several allocations")
                stride = block_strides[name]
                if stride < ns.page_bytes:
                    raise Unsupported(f"layer {name}: block stride {stride} < page {ns.page_bytes}")
                offsets.append(r.offset)
                strides.append(stride)
            assert region is not None
            self._geo.append(
                _NsGeometry(
                    region,
                    np.array(offsets, dtype=np.uint64),
                    np.array(strides, dtype=np.uint64),
                    ns.page_bytes,
                )
            )
            savers = ns.savers(sockets)
            self._saver_of.append(self._rank in savers)
            metas.append(ns.meta(self._world_size, self._rank, self._device_id, savers))
        self._rpc.register_context_v2(self._instance_id, metas, regions, self._transfer_backend)
        logger.info(
            "[PegaKVConnectorV2] registered instance=%s rank=%d device=%d regions=%d "
            "sockets=%s savers=%s",
            self._instance_id,
            self._rank,
            self._device_id,
            len(regions),
            sockets,
            [m["saver_rank"] for m in metas],
        )

    def _spans(self, i: int, block_ids: list[int]) -> tuple[int, list[bytes]]:
        """Region index and one packed span list per block of namespace ``i``."""
        g = self._geo[i]
        ids = np.array(block_ids, dtype=np.uint64)
        out = np.empty((len(block_ids), len(g.offsets), 2), dtype="<u8")
        out[:, :, 0] = g.offsets[None, :] + ids[:, None] * g.strides[None, :]
        out[:, :, 1] = g.page
        return g.region, [row.tobytes() for row in out]

    def _objects(self, items: list[tuple[int, bytes, int]]) -> list[tuple[Any, int, bytes]]:
        by_ns: dict[int, list[tuple[bytes, int]]] = {}
        for i, prefix, block_id in items:
            by_ns.setdefault(i, []).append((prefix, block_id))
        out = []
        for i, blocks in by_ns.items():
            ns = self._layout.namespaces[i].namespace
            region, spans = self._spans(i, [b for _, b in blocks])
            out.extend(
                ((ns, prefix), region, s) for (prefix, _), s in zip(blocks, spans, strict=True)
            )
        return out

    # ----------------------------------------------------------------- loads
    def start(self, plan: StepPlan) -> None:
        """One ``load_objects`` for all of this step's loads, as v1 does: a
        failure fails every request of the step."""
        req_ids: list[str] = []
        blocks: list[tuple[int, bytes, int]] = []
        leases: list[bytes] = []
        for lp in plan.loads:
            if lp.fail:
                self._failed.add(lp.req_id)
                continue
            req_ids.append(lp.req_id)
            blocks.extend(lp.blocks)
            leases.extend(lp.leases)
        if not req_ids:
            return
        start = time.perf_counter()
        try:
            state = self._new_load_state()
            self._rpc.load_objects(
                self._instance_id,
                self._rank,
                self._device_id,
                state.shm_name(),
                self._objects(blocks),
                leases,
            )
        except Exception as e:
            logger.warning("[PegaKVConnectorV2] load failed reqs=%s: %r", req_ids, e)
            self._failed.update(req_ids)
            self._record("load", time.perf_counter() - start, len(blocks), False)
            return
        self._loads.append((state, req_ids, start, len(blocks)))

    def poll(self) -> tuple[set[str], set[str]]:
        """(finished_recving, failed_recving); failed ones are in both."""
        finished: set[str] = set()
        pending = []
        for load in self._loads:
            state, req_ids, start, n_blocks = load
            if not state.is_ready():
                pending.append(load)
                continue
            finished.update(req_ids)
            ok = state.get_state() == 1
            if not ok:
                logger.warning(
                    "[PegaKVConnectorV2] load state=%s reqs=%s", state.get_state(), req_ids
                )
                self._failed.update(req_ids)
            self._record("load", time.perf_counter() - start, n_blocks, ok)
        self._loads = pending
        failed, self._failed = self._failed, set()
        return finished | failed, failed

    # ----------------------------------------------------------------- saves
    def save(self, plan: StepPlan) -> None:
        """Queue this step's saves behind a mark on the compute stream."""
        items = [s for s in plan.saves if self._saver_of[s[0]]]
        if items:
            self._submitted += 1
            self._job_steps.append((self._submitted, plan.step))
            self._jobs.put(_SaveJob(self._submitted, self._mark_compute(), self._objects(items)))
        self._last_step = max(self._last_step, plan.step)

    def saved_through(self) -> int:
        """The latest plan step whose saves, and all earlier ones, are done."""
        with self._cv:
            completed = self._completed
        steps = self._job_steps
        while steps and steps[0][0] <= completed:
            steps.popleft()
        return steps[0][1] - 1 if steps else self._last_step

    def _save_loop(self) -> None:
        stop = False
        while not stop and (job := self._jobs.get()) is not None:
            # As v1: every job queued meanwhile goes out in the same RPC.
            batch = [job]
            while True:
                try:
                    nxt = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
            writes, seen = [], set()
            for j in batch:
                for w in j.writes:
                    if w[0] not in seen:  # one write per object
                        seen.add(w[0])
                        writes.append(w)
            start, ok = time.perf_counter(), False
            try:
                for j in batch:
                    j.ready.synchronize()
                self._rpc.save_objects(self._instance_id, self._rank, writes)
                ok = True
            except ValueError as e:
                # Rejected by the server before or after its copy: a future miss.
                logger.warning("[PegaKVConnectorV2] save rejected: %s", e)
            except BaseException as e:
                # The server may still copy: keep the pins for a while.
                logger.error(
                    "[PegaKVConnectorV2] save RPC failed, holding its blocks %.0f s: %r",
                    self.SAVE_QUIESCE_S,
                    e,
                )
                self._stop.wait(self.SAVE_QUIESCE_S)
            self._record("save", time.perf_counter() - start, len(writes), ok)
            with self._cv:
                self._completed = batch[-1].seq
                self._cv.notify_all()

    # ----------------------------------------------------------------- stats
    def _record(self, op: str, seconds: float, blocks: int, ok: bool) -> None:
        with self._cv:
            st = self._stats
            st.setdefault(f"{op}_duration", []).append(seconds)
            st.setdefault(f"{op}_blocks", []).append(blocks)
            key = f"{op}_{'success' if ok else 'failure'}_count"
            st[key] = st.get(key, 0) + 1

    def take_stats(self) -> dict[str, Any]:
        """Stats since the last call, as ``PegaKVConnectorStats`` data."""
        with self._cv:
            stats, self._stats = self._stats, {}
            if self._submitted > self._completed:
                stats["pending_save_requests"] = self._submitted - self._completed
        return stats

    def shutdown(self) -> None:
        target = self._submitted
        self._stop.set()
        with self._cv:
            done = self._cv.wait_for(lambda: self._completed >= target, self.SHUTDOWN_FLUSH_S)
        if not done:
            logger.warning("[PegaKVConnectorV2] saves still pending at shutdown")
        self._jobs.put(None)
        self._saver.join(timeout=1.0)
