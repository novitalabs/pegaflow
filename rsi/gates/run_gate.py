"""RSI gate orchestrator (human-owned, see docs/rsi/GATES.md).

Runs arms on one model and writes a run record:

    RUN_DIR/
      summary.json            numbers, verdicts, ``official`` flag
      <arm>/<workload>/       what the stage's processes write: probe dir,
                              vllm.log, pegaflow.log (untrusted, M9)
      .gate/<arm>/<workload>/ what the gate writes and judges from: pass
                              outputs / tokens, result.json, stage-error.json

Arms:
  plain                    vLLM without any KV connector (gsm8k only; control)
  baseline                 vLLM + RsiProbe only (no PegaFlow)
  old                      reference connector (pegaflow.connector:PegaKVConnector), black box
  native                   reference: vLLM's OffloadingConnector (CPU offload), record only
  lmcache                  reference: LMCacheConnectorV1, local CPU backend only, record only
  selftest:<mode>          gate self-test connector (rsi.gates.selftest), gsm8k only
  <module>:<Class>         connector under test

Every arm/workload pair gets fresh servers. Passes: ``cold``, then before each
warm pass ``/reset_prefix_cache`` (local only) and ``rsi_poison`` (every GPU KV
byte overwritten), so warm reuse can only come from the connector. Replay runs
``--replay-repeats`` warm passes; the first carries digest tags (G2/G3), the
others are timed for G4. The ``cap`` workload (G5) is its own, larger
replay (``--cap-convs`` x ``--cap-turns``): the baseline arm runs it first
(``cap_cold``, ``cap_warm``) to fix ``U`` and the reference local hits, then
every store arm runs ``cap_cold``, ``cap_warm``, ``cap_warm2..N`` per
fraction ``f`` on a fresh store of ``C = f x U`` bytes; warm pass ``j``
starts the conversations rotated by ``j/4`` of their count.

Per-model gate settings (attention backend, batch invariance, extra vLLM args)
come from ``docs/rsi/models/<model>/gate.json``; the committed spec dump is
``docs/rsi/models/<model>/spec.json``.

A run is ``official`` only with the official profile (constants below, all
required arms and workloads). Every required verdict that cannot be evaluated
is a failure. Exit 0 only for an official run where every verdict passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import sysconfig
import time
from contextlib import ExitStack, contextmanager, suppress
from datetime import date
from pathlib import Path
from typing import Any

import requests

from rsi.gates import digest_compare, oracle, provenance, workloads

REPO = Path(__file__).resolve().parents[2]
OLD = "pegaflow.connector:PegaKVConnector"
SELFTEST = "rsi.gates.selftest.connector:SelftestConnector"
NEGATIVE_SELFTESTS = ("liar", "bitflip", "offbyone")

# Official profile: thresholds and workload sizes (GATES.md is the source).
OFFICIAL = {
    "gsm8k_limit": 0,  # full test split
    "shots": 8,
    "replay_convs": 32,
    "replay_skip": 0,  # > 0: a held-out slice of the trace, never official
    "replay_turns": 12,
    "replay_out_tokens": 64,
    "replay_repeats": 3,
    "replay_digest_every": 4,
    "cap_convs": "auto",  # >= CAP_MIN_CONVS, grown until the capacity floor holds
    "cap_turns": 12,
    "cap_fractions": "0.25,0.5,0.75",
    "concurrency": 32,
    "max_model_len": 8192,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 256,
}
ACC_TOL = 0.01
# Hit-rate levels are reported, never gated (user, 2026-10-24: "命中率那块，不要
# 设置硬性门槛，只要agent自己判断命中率足够就行了，不要差太多"). These are the
# former bars, kept as reference points next to the numbers (``reported``);
# the onboarding agent judges the hit rate in ITERATIONS.md (GATES.md).
MIN_EFFICIENCY = 0.95  # G3 E_ext reference
ORACLE_FLOOR = {"gsm8k": 0.5, "replay": 0.5}  # warm oracle hit rate below it: E says little
MIN_JUDGED_GSM8K = 0.9  # every gsm8k request is digested: nearly all loads are judged
# G4: passes with nothing to reuse from PegaFlow (replay cold: vLLM's own
# prefix cache serves the earlier turns; gsm8k: the shared few-shot prefix) can
# only tie baseline, so they bound the connector's overhead; warm replay must win.
MAX_OVERHEAD = 0.02
# G5: store capacity as a fraction of the cap trace's unique KV bytes; only
# CAPACITY_FRACTION is gated, the others are a reported curve. The bar is the
# clairvoyant oracle at that capacity.
CAPACITY_FRACTION = 0.5
MIN_EFFICIENCY_C = 0.9  # G5 E_C reference (reported, not gated)
# Workload validity, not a bar on the arm: a trace on which a store frozen
# after its first fill reaches this share of ``H*_C`` cannot separate a policy
# from that bar, so the run is not official (``g5_problems``).
FREEZE_SEPARATION = 0.9
MIN_CAPACITY_GPU_KV = 2.0  # official: gated C >= 2 x vLLM GPU KV bytes
CAP_MIN_CONVS = 64
CAP_FLOOR_MARGIN = 1.1  # the auto size aims 10% above the floor (estimate vs measured U)
# Cap digests every request: a loaded block may stay unjudged only when the
# key's undigested computations are preempted episodes, at most this share.
MAX_PREEMPTED_UNCERTAIN = 0.01
WORKLOADS = ("gsm8k", "replay", "cap")
# Server teardown: SIGTERM grace for the whole process group, then SIGKILL and
# a bound for the kernel to tear down (unpinning a 160 GB pool takes > 30 s).
TERM_GRACE = 120
KILL_GRACE = 300
SELFTEST_LIMIT = 128  # gsm8k questions per self-test arm
# Split control (gsm8k, plain and baseline arms): ``split_ref`` repeats the
# cold pass with logprobs, ``split`` pre-warms vLLM's prefix cache with each
# whole prompt so the request starts at its last full prompt block (where a
# connector's warm hit ends). The plain arm's pair establishes gate.json
# ``split_invariant``; on split-variant models an arm's warm completions must
# equal the plain ``split`` ones and the baseline's ``split`` digests are the
# same-split references for warm computed blocks.
SPLIT_PASSES = ("split_ref", "split")
MIN_PREWARMED = 0.9  # share of split requests whose hit reaches the last full block
MIN_COMPUTED_COVERAGE = 0.9  # split-variant: computed blocks judged, per pass


def free_port() -> int:
    """An OS-assigned ephemeral port (tests only: a stage port must be a
    ``PortLease``; the ephemeral range is shared with every client socket)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Stage ports come from a host-reserved range (net.ipv4.ip_local_reserved_ports
# on the 4090 host: 20000-29999), which the kernel never hands out as an
# ephemeral port, so no outgoing connection can take one. Concurrent gate runs
# (several agents share the host) coordinate through per-port lock files: a
# port is leased while a live process holds ``flock`` on its file.
PORT_RANGE = (20000, 29999)
PORT_LOCK_DIR = Path.home() / "kexi" / "rsi-ports"


class PortRace(RuntimeError):
    """A stage process failed to bind a port (EADDRINUSE): another process
    took it. Fail-closed: the stage fails with this reason, never retried."""


# vLLM's own TCP ports (``get_open_port``: a TCP rendezvous / DP master,
# fault-tolerance sentinel) scan upward from ``VLLM_PORT`` when it is set,
# so a contiguous leased block pinned there keeps them in the reserved range
# and out of other gate runs. On CUDA, TP>1 rendezvous through a file store
# (no TCP master port) and the engine core's ZMQ sockets are ``ipc://``
# paths, so the block is normally unused.
VLLM_PORT_BLOCK = 8


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


# Every lease of this gate process (``record``): summary.json ``port_leases``.
_LEASES: list[PortLease] = []


class PortLease:
    """``n`` ports of ``rng`` plus, if ``block``, ``block`` contiguous ones
    (``.block``), held until ``release`` (or the gate's exit: the kernel
    drops the flocks); each was free at lease time. ``log``: the stage
    process the ports are for (recorded)."""

    def __init__(
        self,
        n: int,
        lock_dir: Path | None = None,
        rng: tuple[int, int] = PORT_RANGE,
        block: int = 0,
        log: Path | None = None,
    ):
        import random

        self.lock_dir = lock_dir or PORT_LOCK_DIR
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.ports: list[int] = []
        self.block: list[int] = []
        self._fds: list[int] = []
        self.record: dict[str, Any] = {
            "log": str(log) if log else None,
            "pid": os.getpid(),
            "leased_at": time.time(),
            "released_at": None,
        }
        try:
            if block:
                starts = list(range(rng[0], rng[1] - block + 2, block))
                random.shuffle(starts)
                for b in starts:
                    got = [self._take(q) for q in range(b, b + block)]
                    if all(got):
                        self.block = list(range(b, b + block))
                        break
                    for fd in got:
                        if fd:
                            self._fds.remove(fd)
                            os.close(fd)
                if not self.block:
                    raise RuntimeError(f"no {block} contiguous free ports in {rng} (lock dir {self.lock_dir})")  # fmt: skip
            order = [q for q in range(rng[0], rng[1] + 1) if q not in self.block]
            random.shuffle(order)
            for q in order:
                if len(self.ports) == n:
                    break
                if self._take(q):
                    self.ports.append(q)
            if len(self.ports) < n:
                raise RuntimeError(f"no {n} free ports in {rng} (lock dir {self.lock_dir})")
        except BaseException:
            self.release()
            raise
        self.record |= {
            "ports": self.ports,
            "block": self.block,
            "files": [str(self.lock_dir / str(q)) for q in self.ports + self.block],
        }
        _LEASES.append(self)

    def _take(self, port: int) -> int | None:
        import fcntl

        fd = os.open(self.lock_dir / str(port), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        if not port_free(port):
            os.close(fd)  # in use by a process outside the lock protocol
            return None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fds.append(fd)
        return fd

    def release(self) -> None:
        for fd in self._fds:
            with suppress(OSError):
                os.close(fd)
        if self._fds or self.record.get("released_at") is None:
            self.record["released_at"] = time.time()
        self._fds = []


def release_leases(since: int = 0) -> list[dict[str, Any]]:
    """Release every lease from ``_LEASES[since:]`` (idempotent): a stage's
    leases once it is over, whatever way it ended (a lease whose stage never
    launched included); their records."""
    for lease in _LEASES[since:]:
        lease.release()
    return [lease.record for lease in _LEASES[since:]]


_BIND_FAIL = re.compile(r"address already in use|EADDRINUSE|errno 98\b|os error 98\b", re.I)
_BIND_PORT = (
    re.compile(r"\(\s*['\"][^'\"]*['\"]\s*,\s*(\d{1,5})\s*\)"),  # ('0.0.0.0', 8000)
    re.compile(r"\bport[\s:=]+(\d{1,5})\b", re.I),  # torch TCPStore: port: 29500
    re.compile(
        r"(?:tcp://|\b)(?:\d{1,3}(?:\.\d{1,3}){3}|localhost|\*|\[[0-9a-f:]*\]):(\d{1,5})\b", re.I
    ),  # fmt: skip
)


def port_race(log: Path, leases: list[PortLease], proc: subprocess.Popen) -> PortRace | None:
    """A stage process failed before it was healthy: if its log shows a bind
    failure (any bind: the gate's leased ports or a port vLLM / the server
    picked itself), a ``PortRace`` naming the port(s) parsed from the failing
    lines plus any leased port another process now holds (or "unknown"),
    whether the gate leased them, and the lease files."""
    text = log.read_text(errors="replace")[-50000:] if log.exists() else ""
    lines = text.splitlines()
    hit = [i for i, line in enumerate(lines) if _BIND_FAIL.search(line)]
    if not hit:
        return None
    ports: list[int] = []
    for i in hit:
        for line in lines[max(0, i - 2) : i + 1]:  # the address is often a line above
            for rx in _BIND_PORT:
                ports += [int(m) for m in rx.findall(line) if 0 < int(m) < 65536]
    leased = {q for lease in leases for q in lease.ports + lease.block}
    files = [f for lease in leases for f in lease.record.get("files", [])]
    # A bare bind error often names no port. A leased port still bound once
    # the stage process is dead is held by another process: the raced one.
    held = sorted(q for q in leased if not port_free(q)) if proc.poll() is not None else []
    ports = sorted(set(ports) | set(held))
    where = (
        ", ".join(f"{q} ({'leased' if q in leased else 'not leased by the gate'})" for q in ports)
        if ports
        else "unknown (not in the log)"
    )
    return PortRace(
        f"port race: {log.name} (pid {proc.pid}, exit {proc.poll()}) failed to bind: "
        f"EADDRINUSE on port {where}; leased ports held by another process: {held}; "
        f"gate leases {sorted(leased)} (files {files}); "
        f"failing line: {lines[hit[0]].strip()[:300]!r}; not retried"
    )


def wait_http(url: str, proc: subprocess.Popen, timeout: float, log: Path) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            tail = log.read_text()[-3000:] if log.exists() else ""
            raise RuntimeError(f"process exited {proc.returncode} before {url} was ready\n{tail}")
        with suppress(requests.RequestException):
            if requests.get(url, timeout=2).status_code == 200:
                return
        time.sleep(2)
    raise TimeoutError(f"{url} not ready after {timeout}s")


@contextmanager
def launched(
    cmd: list[str],
    env: dict[str, str],
    log: Path,
    health: str,
    timeout: float,
    lease: PortLease | None = None,
):
    """``lease``: the stage's ports, released once its group is gone.

    N5: a stop signal from just before ``Popen`` until the group is in
    ``_LIVE`` is deferred (``_STOP["teardown"]``), then honored inside the
    ``try``. One landing after that but before ``stop`` owns the group (a
    few bytecodes before ``try:`` or into ``finally:``) cannot be deferred
    by a counter; the group is still in ``_LIVE``, so ``stop_live`` (at
    exit) stops it."""
    try:
        with log.open("w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            # S11: an isolated arm's helper waits after its unshare until the
            # gate holds the new mount namespace (``_record_mnt_ns``).
            gate_r = gate_w = None
            if str(ISOLATE) in cmd:
                gate_r, gate_w = os.pipe()
                cut = cmd.index("--")
                cmd = [*cmd[:cut], "--wait-fd", str(gate_r), *cmd[cut:]]
            _STOP["teardown"] += 1
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=REPO,
                    start_new_session=True,
                    pass_fds=() if gate_r is None else (gate_r,),
                )
                _LIVE[proc.pid] = (proc, cmd[0])
                if gate_w is not None:
                    _MNT_NS[proc.pid] = None  # stop() also empties its mount ns
            except BaseException:
                if gate_w is not None:
                    os.close(gate_w)
                raise
            finally:
                _STOP["teardown"] -= 1
                if gate_r is not None:
                    os.close(gate_r)
            try:
                if gate_w is not None:
                    try:
                        _record_mnt_ns(proc.pid, NS_RECORD_TIMEOUT)
                    finally:
                        os.close(gate_w)  # EOF: the helper goes on
                check_stop()
                try:
                    wait_http(health, proc, timeout, log)
                except Exception as e:  # died or hung: a bind failure is a PortRace
                    race = port_race(log, [lease] if lease else [], proc)
                    if race is not None:
                        raise race from e
                    raise
                yield proc
            finally:
                stop(proc, cmd[0], fh)
    finally:
        if lease is not None:
            lease.release()


_STOP: dict[str, Any] = {"signal": None, "teardown": 0}
# Every launched process group not yet stopped (pid -> (proc, name)).
_LIVE: dict[int, tuple[subprocess.Popen, str]] = {}
# S11: isolated leader pid -> an open fd of the private mount namespace
# isolate.py unshared (None until read). A root arm can setsid() out of its
# process group, but not out of this namespace (setns / unshare need
# CAP_SYS_ADMIN, which isolate.py drops), so every process in it is the
# arm's. The fd keeps the namespace (and so its inode number) from being
# freed and reused until ``stop`` has emptied it.
_MNT_NS: dict[int, int | None] = {}
NS_RECORD_TIMEOUT = 30.0


def _mnt_ns(pid: int | str) -> tuple[int, int] | None:
    """The mount namespace of a live process (None: gone, or a zombie)."""
    try:
        st = os.stat(f"/proc/{pid}/ns/mnt")
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _record_mnt_ns(pid: int, timeout: float) -> None:
    """Hold the leader's mount namespace once isolate.py has unshared it
    (it differs from the gate's). The helper blocks on ``--wait-fd`` right
    after the unshare, so this runs before any of the arm's code can. Left
    None if the leader exits first or is still in the gate's namespace
    after ``timeout``; ``stop`` tries once more."""
    own = _mnt_ns("self")
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(f"/proc/{pid}/ns/mnt", os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            return  # gone
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) != own:
            _MNT_NS[pid] = fd
            return
        os.close(fd)
        if time.monotonic() - t0 >= timeout:
            return
        time.sleep(0.01)


def _ns_signal(fd: int, sig: int) -> list[int]:
    """Send ``sig`` to every live process in the mount namespace held by
    ``fd``; return their pids. Each is signaled through a pidfd opened
    before its namespace is checked, so a pid reused in between is never
    signaled."""
    st = os.fstat(fd)
    ns = (st.st_dev, st.st_ino)
    hit = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            pidfd = os.pidfd_open(int(p))
        except OSError:
            continue
        try:
            if _mnt_ns(p) == ns:
                with suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, sig)
                hit.append(int(p))
        finally:
            os.close(pidfd)
    return hit


def check_stop() -> None:
    """Raise SystemExit if a stop signal was received (``install_stop_handlers``)."""
    if _STOP["signal"] is not None:
        raise SystemExit(128 + _STOP["signal"])


class TeardownError(RuntimeError):
    """A stage's process group survived SIGTERM + SIGKILL: the GPU / host
    memory it holds would skew every later stage, so the run aborts (never
    recorded as a reference arm's error)."""


def stop(proc: subprocess.Popen, name: str, fh) -> None:
    """Stop ``proc``'s whole process group and return only once every member
    is gone, so the next stage cannot start while this one still holds its
    port, GPU memory or pinned host memory. Children (vLLM's EngineCore)
    outlive the API server while they free GPU / CPU-offload memory, and
    pegaflow-server can take minutes after SIGKILL to unpin a large pool, so
    the wait is on the group, with a generous bound; the exit time is logged.

    S11: for an isolated arm the same signals and wait also cover every
    process in its mount namespace (``_MNT_NS``), so one that setsid()'d out
    of the group is stopped too."""
    t0 = time.monotonic()
    _STOP["teardown"] += 1  # a stop signal now waits for this group
    try:
        if proc.pid in _MNT_NS and _MNT_NS[proc.pid] is None:
            _record_mnt_ns(proc.pid, 0)
        ns = _MNT_NS.get(proc.pid)
        _signal_all(proc, ns, signal.SIGTERM)
        how = "SIGTERM"
        if not _all_gone(proc, ns, TERM_GRACE):
            _signal_all(proc, ns, signal.SIGKILL)
            how = "SIGKILL"
            if not _all_gone(proc, ns, KILL_GRACE):
                left = f", mount ns members {_ns_signal(ns, 0)}" if ns is not None else ""
                raise TeardownError(
                    f"process group {proc.pid} ({name}) still alive"
                    f" {time.monotonic() - t0:.0f}s after SIGTERM + SIGKILL{left}"
                )
    finally:
        _LIVE.pop(proc.pid, None)  # gone, or given up on (TeardownError)
        fd = _MNT_NS.pop(proc.pid, None)
        if fd is not None:
            os.close(fd)
        _STOP["teardown"] -= 1
    msg = f"[gate] {name} group {proc.pid} gone after {how} in {time.monotonic() - t0:.1f}s"
    with suppress(ValueError, OSError, AttributeError):
        fh.write(f"\n{msg}\n")
        fh.flush()
    print(msg, flush=True)
    check_stop()


def stop_live() -> None:
    """At exit: stop every group a stop signal kept ``launched`` from
    stopping (N5); a no-op on every ordinary path."""
    for proc, name in list(_LIVE.values()):
        print(f"[gate] {name} group {proc.pid} left by a signal window: stopping it", flush=True)
        with suppress(SystemExit, TeardownError):
            stop(proc, name, None)


def _signal_all(proc: subprocess.Popen, ns: int | None, sig: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)
    if ns is not None:
        _ns_signal(ns, sig)


def _all_gone(proc: subprocess.Popen, ns: int | None, timeout: float) -> bool:
    """Reap the leader (an unreaped zombie still counts as a group member)
    and wait until no process of its group, nor of its mount namespace
    ``ns``, is left. Signal 0 probes only: it delivers nothing."""
    t0 = time.monotonic()
    while True:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            if ns is None or not _ns_signal(ns, 0):
                return True
        if time.monotonic() - t0 >= timeout:
            return False
        time.sleep(1)


def pycache_env(env: dict[str, str], log: Path) -> None:
    """Every stage process caches bytecode in its stage's own cache dir
    (``stage_pycache``), never in a source tree: an in-tree ``__pycache__``
    (planted, or from another interpreter) is never read."""
    env["PYTHONPYCACHEPREFIX"] = str(stage_pycache(log.parent))


PYCACHE_DIR = ".pycache"
# M10: one bytecode cache per stage, made fresh before the stage and outside
# every arm dir (<run>/.pycache-stages/<label>/<wl>): a stage cannot plant a
# ``.pyc`` that a later stage runs.
STAGE_PYCACHE_DIR = ".pycache-stages"


def stage_pycache(stage: Path) -> Path:
    """``<run>/<label>/<wl>`` -> ``<run>/.pycache-stages/<label>/<wl>``."""
    return stage.parents[1] / STAGE_PYCACHE_DIR / stage.parent.name / stage.name


# M9: everything the gate judges from that the gate itself writes (pass
# outputs and tokens, result.json, stage-error.json) lives here, never in a
# dir an arm gets: <run>/.gate/<label>/<workload>. The runs root is under the
# arm's tmpfs and only its current stage dir is bound back.
GATE_DIR = ".gate"


def gate_dir(stage: Path) -> Path:
    """``<run>/<label>/<wl>`` -> ``<run>/.gate/<label>/<wl>``."""
    return stage.parents[1] / GATE_DIR / stage.parent.name / stage.name


def untrusted_problems(stage: Path) -> list[str]:
    """What an arm left in its stage dir (bound rw) that the gate must not
    follow: a symlink or non-regular file (``probe/digest-*.jsonl`` ->
    the baseline's would make the gate judge the baseline's records)."""
    out = []
    for dirpath, dirs, names in os.walk(stage, followlinks=False):
        for n in [*dirs, *names]:
            q = Path(dirpath) / n
            if q.is_symlink():
                out.append(f"symlink {q.relative_to(stage)}")
            elif n in names and not q.is_file():
                out.append(f"non-regular file {q.relative_to(stage)}")
    return out


ISOLATE = Path(__file__).resolve().parent / "isolate.py"
# What hiding does not stop (GATES.md "Boundary"): it is deterrence.
ISOLATION_RESIDUAL = (
    "deterrence, not a boundary. Mechanism: private mount namespace (runs root "
    "under tmpfs) and a bounding set without CAP_SYS_ADMIN, CAP_SYS_PTRACE, "
    "CAP_SYS_MODULE, CAP_SYS_RAWIO, CAP_SYS_CHROOT, CAP_SYS_BOOT, CAP_MKNOD, CAP_BPF, "
    "CAP_CHECKPOINT_RESTORE; opening /proc/<pid>/ns/* or /proc/<pid>/root of a "
    "full-cap process then fails the kernel's ptrace access check (EACCES), so "
    "setns / nsenter into the host mount namespace fail (tested), and the cover "
    "cannot be unmounted. Not covered: the arm stays uid 0 with CAP_DAC_OVERRIDE "
    "in the host pid / IPC / network namespaces, so it can write any host file "
    "outside the runs root, including code that later stages or other root "
    "processes run with full capabilities (gate tree, venv, system files). "
    "Controls: the gate tree, the venv pegaflow tree, the snapshot and the "
    "pegaflow-server binary are re-fingerprinted before every stage and after "
    "the last one (a change makes the run non-official, naming the path); each "
    "stage gets a fresh bytecode cache no later stage reads; the static scan and "
    "the review. Not covered: other venv / system files, and a change made and "
    "undone within one stage."
)


def isolation_available() -> bool:
    return sys.platform.startswith("linux") and os.geteuid() == 0


def isolation(args, stage: Path) -> dict[str, Any]:
    """M7(a), M9, M10: what the arm under test sees of the runs root: only
    the current stage's dir and that stage's bytecode cache (rw) and its
    snapshot (ro). Every other arm's outputs, digests and requests (the
    baseline's included), its own earlier stages and their caches, and the
    gate dir (``GATE_DIR``) are under a tmpfs in its private mount namespace
    (``isolate.py``)."""
    stage = stage.resolve()
    return {
        "helper": str(ISOLATE),
        "hide": str(Path(args.runs_root).resolve()),
        "rw": [str(stage), str(stage_pycache(stage))],
        "ro": [str(Path(args.under_test_overlay).resolve())] if args.under_test_path else [],
    }


# Stage dir -> the isolate.py prefix its processes ran with (summary.json).
_ISOLATED: dict[str, list[str]] = {}


def isolated(args, arm: str, log: Path, cmd: list[str]) -> list[str]:
    """``cmd`` for a stage process: the arm under test's go through
    ``isolate.py`` (exec in place: same pid and process group, so ``stop``
    is unchanged; pid / IPC / network namespaces stay the host's, so CUDA
    IPC between its vLLM and pegaflow-server and the loopback ports the gate
    reads work as before)."""
    if arm_kind(arm) != "under_test" or not isolation_available():
        return cmd
    iso = isolation(args, log.parent)
    for d in iso["rw"]:
        Path(d).mkdir(parents=True, exist_ok=True)  # a bind needs its target
    pre = isolation_prefix(iso)
    _ISOLATED[str(log.parent.resolve())] = pre
    return [*pre, *cmd]


def isolation_prefix(iso: dict[str, Any]) -> list[str]:
    pre = [sys.executable, "-I", "-S", "-B", iso["helper"], "--hide", iso["hide"]]
    pre += [x for d in iso["rw"] for x in ("--rw", d)]
    pre += [x for d in iso["ro"] for x in ("--ro", d)]
    return [*pre, "--"]


def pegaflow_server(args, log: Path, pool_size: str, arm: str = "old"):
    lease = PortLease(2, log=log)
    grpc, http = lease.ports
    env = os.environ.copy()
    pycache_env(env, log)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYO3_PYTHON"] = sys.executable
    env["PYTHONHOME"] = sys.base_prefix
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (env.get("LD_LIBRARY_PATH"), libdir) if x)
    env["PYTHONPATH"] = f"{REPO / 'python'}:{sysconfig.get_path('purelib')}"
    cmd = [
        args.pegaflow_server,
        "--addr",
        f"127.0.0.1:{grpc}",
        "--http-addr",
        f"127.0.0.1:{http}",
        "--pool-size",
        pool_size,
        "--devices",
        args.gpu,
        "--enable-prometheus",
        # One pool for the whole size: with per-NUMA pools a single GPU only
        # sees pool_size / numa_nodes, and the oracle assumes no eviction.
        "--disable-numa-affinity",
    ]
    cmd = isolated(args, arm, log, cmd)
    return grpc, http, launched(cmd, env, log, f"http://127.0.0.1:{http}/health", 300, lease)


_REFERENCE = ("plain", "baseline", "old", "native", "lmcache")
# Reference arms: numbers recorded, never required, a crash is recorded.
_RECORD_ONLY = ("old", "native", "lmcache")
# Reference stores without a PegaFlow server (a CPU cache inside vLLM).
_OFFLOAD = ("native", "lmcache")
LMCACHE_OVERLAY = Path.home() / "kexi" / "overlays" / "lmcache"


def lmcache_env(cpu_bytes: int) -> dict[str, str]:
    """LMCache config (env, read by ``LMCacheEngineConfig.from_env``): local
    CPU backend only, sized to exactly ``cpu_bytes``. ``max_local_cpu_size``
    is GiB and LMCache allocates ``int(GiB * 1024**3)`` bytes; ``bytes / 2**30``
    is exact in a float and ``repr`` round-trips it. Chunk size stays at
    LMCache's default (256 tokens); telemetry is off. Disk and remote stay at
    their defaults (None); the caller drops any inherited ``LMCACHE_*``."""
    return {
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": repr(cpu_bytes / (1 << 30)),
        "LMCACHE_MAX_LOCAL_DISK_SIZE": "0",
        "LMCACHE_TRACK_USAGE": "false",
        "DO_NOT_TRACK": "1",
    }


def lmcache_config(log: Path, cpu_bytes: int) -> dict[str, Any]:
    """What the lmcache arm ran with: the env we set and LMCache's own
    ``LMCache Configuration:`` log line (chunk size, backends, CPU GiB)."""
    line = next((x for x in log.read_text().splitlines() if "LMCache Configuration:" in x), None)
    return {
        "cpu_bytes": cpu_bytes,
        "env": lmcache_env(cpu_bytes),
        "logged": line.split("LMCache Configuration:", 1)[1].strip()[:4000] if line else None,
    }


def arm_kind(arm: str) -> str:
    if arm in _REFERENCE:
        return arm
    return "selftest" if arm.startswith("selftest:") else "under_test"


# The venv's pegaflow ``vllm.general_plugins`` entry point (python/pyproject.toml).
PEGAFLOW_PLUGIN = "pegaflow"


def vllm_plugins(arm: str) -> str:
    """``VLLM_PLUGINS`` of an arm's vLLM processes: "" (no plugin) except for
    the ``old`` reference arm, which gets the pegaflow plugin alone.

    vLLM 0.32's API server rebuilds each MultiConnector child's stats by
    class name through the connector registry
    (``MultiConnector.build_kv_connector_stats`` ->
    ``KVConnectorFactory.get_connector_class_by_name``). ``PegaKVConnector``
    reports stats and only the pegaflow plugin registers its name, so
    without it every request of the old arm is an HTTP 500. The other arms'
    children report no stats (probe, selftest, the arm under test today) or
    are vLLM builtins registered under their class name (native, lmcache).

    An arm under test whose connector starts reporting stats hits the same
    500 (loud, never silent). The fix is in the connector, not here: its own
    module registers its class under its class name at import
    (``KVConnectorFactory.register_connector(cls_name, __name__, cls_name)``;
    the module is imported in the API server by module path). Never enable
    the plugin for the arm under test: it would run the venv's pegaflow code
    in the arm's processes."""
    return PEGAFLOW_PLUGIN if arm_kind(arm) == "old" else ""


def arm_label(arm: str) -> str:
    if arm.startswith("selftest:"):
        return "selftest-" + arm.split(":", 1)[1]
    return arm if arm in _REFERENCE else arm.split(":")[-1]


def parse_size(s: str) -> int:
    """Bytes of a ``--pool-size`` string (same units as pegaflow-server)."""
    s = s.strip().lower()
    for unit, mult in (("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * mult)
    return int(float(s.removesuffix("b")))


def kv_transfer_config(
    arm: str, probe_dir: Path, digest: str, cpu_bytes: int = 0
) -> dict[str, Any] | None:
    kind = arm_kind(arm)
    if kind == "plain":
        return None
    probe = {
        "kv_connector": "RsiProbe",
        "kv_connector_module_path": "rsi.gates.rsi_probe",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"probe_dir": str(probe_dir), "digest": digest},
    }
    if kind == "baseline":
        return probe
    if kind == "native":
        # vLLM's own CPU offload, as ``--kv-offloading-size`` would set it up.
        under_test = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
        }
    elif kind == "lmcache":
        # LMCache's vLLM connector; sized by env (``lmcache_env``).
        under_test = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    else:
        if kind == "selftest":
            module, cls = SELFTEST.split(":")
            extra = {"mode": arm.split(":", 1)[1]}
        else:
            module, cls = (OLD if kind == "old" else arm).split(":")
            extra = {}
        under_test = {
            "kv_connector": cls,
            "kv_connector_module_path": module,
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
        # G0.provenance: the probe records where this module was imported from.
        probe["kv_connector_extra_config"]["connector_modules"] = [module]
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        # A failed load must be recomputed, never served from garbage.
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {"connectors": [under_test, probe]},
    }


def vllm_server(
    args,
    gate: dict[str, Any],
    arm: str,
    probe_dir: Path,
    invariant: bool,
    pf_port,
    log,
    cpu_bytes: int = 0,
):
    lease = PortLease(1, block=VLLM_PORT_BLOCK, log=log)
    (port,) = lease.ports
    env = os.environ.copy()
    pycache_env(env, log)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["VLLM_PORT"] = str(lease.block[0])  # vLLM's own ports: from the leased block
    env["VLLM_SERVER_DEV_MODE"] = "1"  # /reset_prefix_cache, /collective_rpc
    # No ``vllm.general_plugins`` (M7d): the pegaflow plugin would run a
    # tree's code in every vLLM process, the API server included. Every arm
    # names its connector by module path; the only other plugins in the venv
    # are vLLM's LoRA resolvers (unused). The ``old`` reference arm is the
    # exception: it needs the plugin's connector names (``vllm_plugins``).
    env["VLLM_PLUGINS"] = vllm_plugins(arm)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = ":".join(x for x in (str(REPO), env.get("PYTHONPATH")) if x)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    backend: list[str] = []
    if gate.get("attention_backend"):
        backend = ["--attention-backend", gate["attention_backend"]]
    if invariant:
        # Batch invariance makes computed KV independent of batch composition,
        # so loaded blocks can be compared byte-for-byte with a baseline run.
        env["VLLM_BATCH_INVARIANT"] = "1"
        if gate.get("invariant_attention_backend"):
            backend = ["--attention-backend", gate["invariant_attention_backend"]]
    else:
        env.pop("VLLM_BATCH_INVARIANT", None)
    if pf_port is not None:
        env["PEGAFLOW_PORT"] = str(pf_port)  # contract: the connector under test reads it
    for k in [k for k in env if k.startswith("LMCACHE_")]:
        del env[k]
    if arm_kind(arm) == "under_test" and args.under_test_path:
        # The connector under test (and its native extension) from another
        # tree, through a verified snapshot holding only its ``pegaflow``
        # (``provenance.stage``): the tree cannot shadow the probe, vLLM or
        # sitecustomize. Reference arms keep the venv's pegaflow.
        env["PYTHONPATH"] = f"{args.under_test_overlay}:{env['PYTHONPATH']}"
    if arm_kind(arm) == "lmcache":
        # LMCache lives in an overlay (its pins would downgrade the shared
        # venv); only this arm's vLLM sees it.
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{args.lmcache_overlay}"
        env |= lmcache_env(cpu_bytes)
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(port),
        "--served-model-name",
        "m",
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(args.gpu_mem),
        "--generation-config",
        "vllm",
        "--seed",
        "42",
        "--worker-extension-cls",
        "rsi.gates.rsi_probe.RsiWorkerExt",
        *backend,
        *gate.get("vllm_args", []),
        *args.vllm_arg,
    ]
    if arm_kind(arm) == "plain":
        # Usage reports cached prompt tokens: shows the split pre-warm hit.
        cmd.append("--enable-prompt-tokens-details")
    cfg = kv_transfer_config(arm, probe_dir, "tagged", cpu_bytes)
    if cfg is not None:
        cmd += ["--kv-transfer-config", json.dumps(cfg)]
    if args.num_gpu_blocks:
        cmd += ["--num-gpu-blocks-override", str(args.num_gpu_blocks)]
    cmd = isolated(args, arm, log, cmd)
    url = f"http://127.0.0.1:{port}"
    return url, launched(cmd, env, log, f"{url}/health", 900, lease)


def reset_prefix_cache(url: str) -> None:
    for _ in range(30):
        r = requests.post(f"{url}/reset_prefix_cache", timeout=30)
        if r.ok and r.json().get("success", True) is not False:
            return
        time.sleep(1)
    raise RuntimeError(f"reset_prefix_cache kept failing: {r.status_code} {r.text}")


def reset_and_poison(url: str) -> dict[str, Any]:
    """Drop the local prefix cache, then overwrite all GPU KV bytes."""
    reset_prefix_cache(url)
    r = requests.post(f"{url}/collective_rpc", json={"method": "rsi_poison"}, timeout=120)
    r.raise_for_status()
    results = r.json()["results"]
    if not results or not all(isinstance(x, dict) and x.get("bytes") for x in results):
        raise RuntimeError(f"rsi_poison did not cover the KV cache: {results}")
    return {"poisoned": results}


def cap_names(repeats: int) -> list[str]:
    """G5 pass labels: the replay passes, prefixed ``cap_``."""
    return [f"cap_{n}" for n in ["cold", "warm"] + [f"warm{k}" for k in range(2, repeats + 1)]]


def cap_order(name: str, n: int) -> list[int]:
    """Conversation start order of a cap pass. Warm pass ``j`` (``cap_warm`` =
    1) rotates by ``j * n // 4``, so a store frozen after the cold pass does
    not see the same arrival order again and recency differs between passes."""
    stage = name.removeprefix("cap_")
    j = 0 if stage == "cold" else 1 if stage == "warm" else int(stage.removeprefix("warm"))
    k = (j * n // 4) % n if n else 0
    return list(range(k, n)) + list(range(k))


def cap_label(f: float) -> str:
    """Workload dir / result key of capacity fraction ``f``."""
    return "cap" if f == CAPACITY_FRACTION else f"cap{round(f * 100):03d}"


def parse_fractions(s: str) -> list[float]:
    out = sorted({float(x) for x in s.split(",") if x.strip()})
    if CAPACITY_FRACTION not in out:
        raise ValueError(f"--cap-fractions must include the gated {CAPACITY_FRACTION}")
    if any(not 0 < f <= 1 for f in out):
        raise ValueError(f"--cap-fractions must be in (0, 1]: {out}")
    return out


def gpu_kv_bytes(spec: dict[str, Any]) -> int:
    """vLLM's GPU KV bytes in ``U``'s units: the allocated KV tensors of one
    rank (``KVCacheTensor.size``; groups sharing a tensor count once, host
    tensors not at all) x distinct copies over the TP ranks."""
    kc = spec["kv_cache_config"]
    per_rank = sum(t["size"] for t in kc["tensors"] if not t.get("host_resident"))
    return per_rank * oracle.tp_copies(spec)


def _pos(r: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    if r.get("conv") is None or r.get("turn") is None:
        return None
    return (r.get("pass"), r["conv"], r["turn"])


def reference_locals(reqs: list[dict[str, Any]]) -> dict[str, int]:
    """Baseline local hit per workload position ``pass/conv/turn``."""
    return {"/".join(map(str, p)): r["local_hit"] or 0 for r in reqs if (p := _pos(r))}


def match_locals(reqs: list[dict[str, Any]], ref: dict[str, int] | None) -> list[int | None]:
    """Reference local hit per request: the baseline's at the same
    conversation / turn; ``cap_cold`` -> baseline ``cap_cold``, every warm pass
    -> baseline ``cap_warm`` (local hits after reset + poison). None if absent."""
    out: list[int | None] = []
    for r in reqs:
        p = _pos(r)
        if ref is None or p is None:
            out.append(None)
            continue
        label = "cap_cold" if p[0] == "cap_cold" else "cap_warm"
        out.append(ref.get(f"{label}/{p[1]}/{p[2]}"))
    return out


def run_arm_workload(
    args,
    gate,
    arm: str,
    workload: str,
    out: Path,
    capacity: int = 0,
    ref_local: dict[str, int] | None = None,
) -> dict[str, Any]:
    """``capacity`` (``cap*`` workloads): store bytes, for PegaFlow's pool or
    the native arm's CPU cache. Other workloads use ``--pool-size``.
    ``ref_local``: baseline cap local hits (``reference_locals``)."""
    out.mkdir(parents=True, exist_ok=True)
    probe = out / "probe"
    if probe.exists():
        shutil.rmtree(probe)
    gdir = gate_dir(out)
    for d in (gdir, stage_pycache(out)):  # fresh per stage (M9, M10)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    kind = arm_kind(arm)
    gsm = workload == "gsm8k"
    cap = workload.startswith("cap")
    if cap and capacity <= 0 and kind != "baseline":
        raise ValueError("cap workload needs a capacity")
    store_bytes = capacity if cap else parse_size(args.pool_size)
    invariant = gsm and bool(gate.get("batch_invariant"))
    split_inv = bool(gate.get("split_invariant", True))
    result: dict[str, Any] = {
        "arm": arm,
        "workload": workload,
        "invariant": invariant,
        "passes": {},
    }
    if cap:
        result["capacity_bytes"] = capacity
    with ExitStack() as stack:
        pf_port = pf_http = None
        if kind in ("old", "under_test"):
            pf_port, pf_http, cm = pegaflow_server(
                args, out / "pegaflow.log", str(capacity) if cap else args.pool_size, arm
            )
            stack.enter_context(cm)
        url, cm = vllm_server(
            args,
            gate,
            arm,
            probe,
            invariant,
            pf_port,
            out / "vllm.log",
            store_bytes if kind in _OFFLOAD else 0,
        )
        stack.enter_context(cm)
        if kind == "lmcache":
            result["lmcache"] = lmcache_config(out / "vllm.log", store_bytes)
        if cap and pf_http:
            # What the server actually allocated: must not exceed C.
            m = asyncio.run(workloads.scrape(f"http://127.0.0.1:{pf_http}", ("pegaflow",)))
            pool = [n for k, n in m.items() if k.startswith("pegaflow_pool_capacity_bytes")]
            result["pool_capacity_bytes"] = int(sum(pool)) if pool else None

        if gsm:
            limit = SELFTEST_LIMIT if kind == "selftest" else args.gsm8k_limit
            prompts = workloads.gsm8k_prompts(Path(args.gsm8k_dir), args.shots, limit)
            names = ["cold", "warm"]
            if kind == "plain":
                names += list(SPLIT_PASSES)  # the split control
            elif kind == "baseline" and not split_inv:
                names.append("split")  # same-split references for warm computed KV
        else:
            convs = workloads.build_turns(
                Path(args.replay_file),
                args.model,
                args.cap_plan["cap_convs"] if cap else args.replay_convs,
                args.max_model_len - args.replay_out_tokens - 8,
                args.cap_turns if cap else args.replay_turns,
                skip=args.replay_skip,
            )
            names = ["cold", "warm"] + [f"warm{k}" for k in range(2, args.replay_repeats + 1)]
            if cap:
                # Baseline: U (cold) and reference local hits (cold, warm).
                names = cap_names(args.replay_repeats)[: 2 if kind == "baseline" else None]
        if not cap:
            # Untimed first request (first-use costs), then drop it from
            # vLLM's prefix cache; cap timing is not gated and its blocks
            # would take capped store space.
            result["passes"][workloads.WARMUP_PASS] = asyncio.run(workloads.run_warmup(url, "m"))
            reset_prefix_cache(url)
        for name in names:
            check_stop()
            stage = name.removeprefix("cap_")
            if stage != "cold":
                time.sleep(args.settle)  # let async saves land before the warm pass
                result["passes"].setdefault("_poison", {})[name] = reset_and_poison(url)
            before = asyncio.run(workloads.scrape(url))
            pf_metrics = f"http://127.0.0.1:{pf_http}" if pf_http else None
            pf_before = (
                asyncio.run(workloads.scrape(pf_metrics, ("pegaflow",))) if pf_metrics else {}
            )
            if gsm:
                # Digest every request: under batch invariance every copy is a
                # reference, so nothing a connector loads goes unjudged.
                xargs = {"rsi_pass": name, "rsi_digest": "full"}
                res = asyncio.run(
                    workloads.run_gsm8k(
                        url,
                        "m",
                        prompts,
                        args.concurrency,
                        xargs=xargs,
                        logprobs=kind == "plain" and name in SPLIT_PASSES,
                        prewarm=name == "split",
                    )
                )
                (gdir / f"{name}-outputs.json").write_text(json.dumps(res.pop("outputs")))
                if "tokens" in res:
                    (gdir / f"{name}-tokens.json").write_text(json.dumps(res.pop("tokens")))
            else:
                # Replay: digest a conversation sample in the untimed passes;
                # all turns of a sampled conversation, so its references are
                # digested too. Cap: every request of every pass, so every
                # scored load is judged and the reference is exhaustive (no
                # undigested computation of any key; cap timing is not gated).
                timed = stage not in ("cold", "warm") and not cap
                every = 1 if cap else args.replay_digest_every
                xs = [
                    {"rsi_pass": name} | ({} if timed or c % every else {"rsi_digest": "full"})
                    for c in range(len(convs))
                ]
                res = asyncio.run(
                    workloads.run_replay(
                        url,
                        "m",
                        convs,
                        args.concurrency,
                        args.replay_out_tokens,
                        xargs=xs,
                        order=cap_order(name, len(convs)) if cap else None,
                    )
                )
            time.sleep(2)  # late finish records and metrics land in this pass
            res["vllm_metrics"] = workloads.diff(asyncio.run(workloads.scrape(url)), before)
            if pf_metrics:
                pf_after = asyncio.run(workloads.scrape(pf_metrics, ("pegaflow",)))
                res["pegaflow_metrics"] = workloads.diff(pf_after, pf_before)
                if cap:
                    # Absolute occupancy at pass end vs C: shows fragmentation.
                    res["pegaflow_end"] = {
                        k.split("{")[0]: v
                        for k, v in pf_after.items()
                        if k.startswith(
                            ("pegaflow_cache_resident_bytes{", "pegaflow_pool_used_bytes")
                        )
                    }
            check_stop()  # a pass cut short by a stop signal is not recorded
            result["passes"][name] = res
            print(f"[{arm}/{workload}/{name}] {summary_line(res)}", flush=True)

    if kind == "plain":
        (gdir / "result.json").write_text(json.dumps(result, indent=1))
        return result
    bad = untrusted_problems(out)
    if bad:
        raise RuntimeError(f"{arm}/{workload}: stage dir holds what the gate must not read: {bad[:10]}")  # fmt: skip
    spec = json.loads((probe / "spec.json").read_text())
    reqs = oracle.load_requests(probe / "requests.jsonl")
    bad = oracle.warmup_problems(reqs, workloads.WARMUP_PASS)
    if bad:
        raise RuntimeError(f"{arm}/{workload}: warmup not isolated: {bad}")
    result["hit"] = oracle.summarize(spec, reqs, {n: {n} for n in names})
    if cap and kind == "baseline":
        result["unique_bytes"] = oracle.unique_bytes(spec, reqs, {"cap_cold"})
        result["gpu_kv_bytes"] = gpu_kv_bytes(spec)
        result["ref_local"] = reference_locals(reqs)
    elif cap:
        timed = {n for n in names if n.removeprefix("cap_") not in ("cold", "warm")}
        result["capacity"] = oracle.summarize_capacity(
            spec,
            reqs,
            capacity,
            {"warm_timed": timed} | {n: {n} for n in names},
            match_locals(reqs, ref_local),
        )
        result["external_load_bytes"] = {
            n: oracle.external_load_bytes(spec, reqs, {n}) for n in names
        }
    errs = probe / "load_errors.jsonl"
    result["load_errors"] = (
        [json.loads(x) for x in errs.read_text().splitlines()] if errs.exists() else []
    )
    result["digest_self"] = digest_compare.compare(probe, probe, invariant, split_inv)
    (gdir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def summary_line(res: dict[str, Any]) -> str:
    keys = ("accuracy", "ttft_p50", "ttft_p99", "req_per_s", "seconds")
    return " ".join(f"{k}={res[k]:.4g}" for k in keys if k in res)


# ------------------------------------------------------------------- G0

# Depend on GPU memory / CLI, not on the model layout.
# ``modules`` (host paths of the instrument and runtime) is judged by
# G0.provenance, not compared across arms or with the committed dump.
_VOLATILE = {"num_blocks", "size", "layer_stride", "model", "max_model_len", "probe", "modules"}


def _strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in _VOLATILE}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _worker_layout(probe: Path) -> dict[str, Any]:
    out = {}
    for f in sorted(probe.glob("spec-*.json")):
        d = json.loads(f.read_text())
        out[d["rank"]] = {
            name: [
                {"shape": t["shape"][1:], "stride": t["stride"], "dtype": t["dtype"]}
                for t in layer["tensors"]
            ]
            for name, layer in d["layers"].items()
        }
    return out


def spec_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += spec_diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in spec_diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def g0(run_dir: Path, label: str, committed: Path) -> dict[str, Any]:
    """Arm layout == baseline layout (gsm8k), baseline layout == committed dump."""
    base = run_dir / "baseline" / "gsm8k" / "probe"
    mine = run_dir / label / "gsm8k" / "probe"
    if not (base / "spec.json").exists() or not (mine / "spec.json").exists():
        return {"pass": False, "reason": "spec dump missing"}
    diffs = spec_diff(
        _strip(json.loads((base / "spec.json").read_text())),
        _strip(json.loads((mine / "spec.json").read_text())),
    ) + spec_diff(_worker_layout(base), _worker_layout(mine), ".worker")
    if committed.exists():
        diffs += [
            "committed" + d
            for d in spec_diff(
                _strip(json.loads(committed.read_text())),
                _strip(json.loads((base / "spec.json").read_text())),
            )
        ]
    else:
        diffs.append(f"committed spec dump {committed} missing")
    return {"pass": not diffs, "diffs": diffs[:20], "num_diffs": len(diffs)}


def expected_pegaflow(args, kind: str) -> Path | None:
    """The ``pegaflow`` dir an arm's processes must import."""
    if kind == "under_test" and getattr(args, "under_test_path", ""):
        return (Path(args.under_test_overlay) / "pegaflow").resolve()
    return provenance.gate_pegaflow_dir()


def expected_connector(arm: str) -> str | None:
    """The connector module an arm's probe dumps must record (``None``: a
    vLLM built-in, no module path)."""
    kind = arm_kind(arm)
    if kind in ("selftest", "under_test", "old"):
        return {"selftest": SELFTEST, "old": OLD}.get(kind, arm).split(":")[0]
    return None


def connector_root(arm: str, pegaflow_dir: Path | None) -> Path | None:
    """Where an arm's connector module must resolve, by arm kind (M8), never
    by module name: the declared self-test arms -> the gate tree; every other
    arm (the arm under test whatever its module, ``old``) -> its expected
    ``pegaflow`` dir (the snapshot for the arm under test)."""
    return REPO.resolve() if arm_kind(arm) == "selftest" else pegaflow_dir


def g0_provenance(
    run_dir: Path,
    label: str,
    pegaflow_dir: Path | None,
    connector: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Every spec dump of the arm (scheduler and workers, every workload)
    recorded its module files; they resolve under the gate tree, the shared
    venv and the arm's expected ``pegaflow``. ``connector``: the arm's
    connector module, which must be recorded and resolve under ``root``
    (``connector_root``)."""
    dumps = sorted((run_dir / label).glob("*/probe/spec*.json"))
    if not dumps:
        return {"pass": False, "reason": "no spec dump"}
    problems, pegaflow_files = [], set()
    for f in dumps:
        files = json.loads(f.read_text()).get("modules")
        where = str(f.relative_to(run_dir))
        if not files:
            problems.append(f"{where}: module files not recorded")
            continue
        problems += provenance.module_problems(files, REPO, pegaflow_dir, where)
        pegaflow_files |= {files[k] for k in ("pegaflow", "pegaflow.pegaflow") if files.get(k)}
        if connector is not None:
            f = files.get(f"connector:{connector}")
            if not f:
                problems.append(f"{where}: connector module {connector} not recorded")
            elif root is None or not provenance.under(f, root):
                problems.append(f"{where}: connector {connector} = {f} not under {root}")
            else:
                pegaflow_files.add(f)
    return {
        "pass": not problems,
        "problems": problems[:20],
        "dumps": len(dumps),
        "pegaflow_dir": str(pegaflow_dir),
        "pegaflow_files": sorted(pegaflow_files),
    }


# ------------------------------------------------------------- verdicts


def _outputs(run_dir: Path, label: str, name: str) -> list[str] | None:
    f = run_dir / GATE_DIR / label / "gsm8k" / f"{name}-outputs.json"
    return json.loads(f.read_text()) if f.exists() else None


def _differ(ref: list[str] | None, got: list[str] | None) -> int | None:
    if ref is None or got is None or len(got) > len(ref):
        return None
    return sum(a != b for a, b in zip(ref[: len(got)], got, strict=True))


def _tokens(run_dir: Path, label: str, name: str) -> list[dict[str, Any] | None] | None:
    f = run_dir / GATE_DIR / label / "gsm8k" / f"{name}-tokens.json"
    return json.loads(f.read_text()) if f.exists() else None


def _hit_alignment(run_dir: Path) -> int:
    """vLLM's prefix-cache hit granularity (baseline spec dump; 16 if absent)."""
    f = run_dir / "baseline" / "gsm8k" / "probe" / "spec.json"
    if f.exists():
        kc = json.loads(f.read_text()).get("kv_cache_config", {})
        for k in ("cache_hit_alignment_tokens", "hash_block_size"):
            if kc.get(k):
                return int(kc[k])
    return 16


def split_control(run_dir: Path) -> dict[str, Any] | None:
    """The plain arm's split control: ``split`` (each prompt pre-warmed into
    vLLM's prefix cache, so the request starts at its last full prompt block)
    vs ``split_ref`` (same prompts, hits end at the shared few-shot prefix).
    Per request: completion differs, sampled-token logprobs differ, and the
    pre-warm took (``split`` cached the whole prompt up to its last full
    block: ``cached >= (prompt_tokens - 1) // align * align``)."""
    ref, got = _outputs(run_dir, "plain", "split_ref"), _outputs(run_dir, "plain", "split")
    rt, gt = _tokens(run_dir, "plain", "split_ref"), _tokens(run_dir, "plain", "split")
    if not ref or not got or rt is None or gt is None or len(ref) != len(got):
        return None
    differ = [a != b for a, b in zip(ref, got, strict=True)]
    lp = [
        a is None or b is None or a.get("logprobs") != b.get("logprobs")
        for a, b in zip(rt, gt, strict=True)
    ]
    al = _hit_alignment(run_dir)
    warmed = [
        b is not None
        and b.get("prompt_tokens") is not None
        and (b.get("cached") or 0) >= (b["prompt_tokens"] - 1) // al * al
        for b in gt
    ]
    # prompt_len % align == 0: vLLM's own hit stops one block short of the
    # prompt (it must compute the last token), a connector's need not, so the
    # split differs from a connector's warm request on exactly these prompts.
    # A missing token record is not aligned: it is never exempted (S7).
    aligned = [b is not None and bool(b.get("prompt_tokens")) and b["prompt_tokens"] % al == 0 for b in gt]  # fmt: skip
    cold = _outputs(run_dir, "plain", "cold")
    return {
        "n": len(ref),
        "differ": differ,
        "logprob_differ": lp,
        "prewarmed": sum(warmed),
        "ref_equals_cold": cold is not None and cold == ref,
        "split_outputs": got,
        "aligned": aligned,
    }


def _split_control_verdict(ctrl: dict[str, Any] | None, claim: bool) -> dict[str, Any]:
    """gate.json's ``split_invariant`` must be what the control measures:
    true needs identical completions and logprobs at both splits, false needs
    some divergence. The pre-warm must have taken for nearly every request,
    else the control compared nothing."""
    if ctrl is None:
        return {"pass": False, "reason": "not evaluated (plain arm split passes missing)"}
    n = ctrl["n"]
    differ, lp = sum(ctrl["differ"]), sum(ctrl["logprob_differ"])
    measured = differ == 0 and lp == 0
    ok = measured == claim and ctrl["prewarmed"] >= MIN_PREWARMED * n
    # split_ref is the cold pass repeated: if it differs from plain cold the
    # control's basis is not the baseline the arms are compared to.
    ok = ok and ctrl["ref_equals_cold"]
    out = {
        "pass": ok,
        "split_invariant": claim,
        "measured_invariant": measured,
        "n": n,
        "differ": differ,
        "rate": differ / n if n else None,
        "logprob_differ": lp,
        "prewarmed": ctrl["prewarmed"],
        "ref_equals_cold": ctrl["ref_equals_cold"],
    }
    if measured != claim:
        out["reason"] = (
            f"gate.json split_invariant={claim} but the control measured "
            f"{'no' if measured else 'a'} divergence"
        )
    elif not ctrl["ref_equals_cold"]:
        out["reason"] = "split_ref differs from plain cold: the control has no stable basis"
    elif not ok:
        out["reason"] = "pre-warm did not take: the control is vacuous"
    return out


def split_differ(
    ctrl: dict[str, Any] | None, got: list[str] | None
) -> tuple[int | None, int | None]:
    """Warm completions vs the plain ``split`` ones of the same prompts:
    (differing among non-aligned prompts, differing among aligned prompts)."""
    if ctrl is None or got is None or len(got) > ctrl["n"]:
        return None, None
    pairs = zip(ctrl["split_outputs"], got, ctrl["aligned"], strict=False)
    diff = [(a != b, al) for a, b, al in pairs]
    return sum(d for d, al in diff if not al), sum(d for d, al in diff if al)


def _outputs_verdict(
    diffs: dict[str, int | None],
    invariant: bool,
    split_inv: bool,
    split_diffs: tuple[int | None, int | None] = (None, None),
) -> dict[str, Any]:
    """G2.gsm8k_outputs. Split-invariant: every cold and warm completion
    equals baseline cold. Split-variant: cold (same split as the baseline)
    must equal baseline cold; warm (hit ends at the last full prompt block)
    must equal the plain arm's ``split`` completions of the same prompts
    exactly, except on ``prompt_len % align == 0`` prompts, where the plain
    split stops one block earlier than a connector's hit (reported only); the
    warm differ count vs baseline cold is reported only."""
    out: dict[str, Any] = {"invariant": invariant, "split_invariant": split_inv}
    out |= {f"{p}_differ": d for p, d in diffs.items()}
    if not invariant:
        return {"pass": True} | out
    if split_inv:
        return {"pass": all(d == 0 for d in diffs.values())} | out
    out["warm_differ_vs_split"], out["warm_differ_vs_split_aligned"] = split_diffs
    if split_diffs[0] is None or diffs.get("cold") is None:
        return {"pass": False, "reason": "plain split outputs not comparable"} | out
    return {"pass": diffs["cold"] == 0 and split_diffs[0] == 0} | out


def _coverage_verdict(
    v: dict[str, Any],
    d: dict[str, Any] | None,
    passes: tuple[str, ...],
    exact: tuple[str, ...] = ("warm",),
) -> None:
    """Split-variant: in each of ``passes`` the candidate's computed blocks
    must be judged against same-split references: judged > 0 and at least
    ``MIN_COMPUTED_COVERAGE`` of them (unjudged = only other-split copies in
    REF, or REF never computed the key). In ``exact`` passes every unjudged
    block must sit in an aligned prompt (``prompt_len % align == 0``, where
    the plain split stops one block short of a connector's hit): zero
    unjudged computed blocks on non-aligned prompts (S7)."""
    if d is None or d.get("split_invariant") is not False:
        return
    cov, ok = {}, True
    by_pass = d.get("computed_by_pass", {})
    for p in passes:
        c = by_pass.get(p, {})
        judged = c.get("compared", 0)
        seen = judged + c.get("unmatched_split", 0) + c.get("unreferenced", 0)
        frac = judged / seen if seen else None
        nonaligned = c.get("unjudged_nonaligned")
        cov[p] = {"judged": judged, "seen": seen, "coverage": frac, "unjudged_nonaligned": nonaligned}  # fmt: skip
        ok = ok and judged > 0 and frac is not None and frac >= MIN_COMPUTED_COVERAGE
        if p in exact:
            ok = ok and nonaligned == 0
    v["computed_coverage_by_pass"] = cov
    if not ok:
        v["pass"] = False
        v.setdefault(
            "reason",
            f"computed coverage below {MIN_COMPUTED_COVERAGE}, nothing judged,"
            " or unjudged warm blocks on non-aligned prompts",
        )


def _digest_verdict(
    d: dict[str, Any] | None,
    min_judged: float | None,
    vacuous_ok: bool = False,
    exhaustive: bool = False,
) -> dict[str, Any]:
    """``min_judged``: fraction of loaded external blocks that must be judged
    (matched or mismatched, not uncertain); None = no external load required.
    ``vacuous_ok``: no external block at all is a (flagged) pass, not a fail
    (cap, reference arms only: a store at small C may load nothing; G5 scores
    that). ``exhaustive`` (cap: every request digested): every loaded block
    must be judged; the only allowance, flagged, is a block whose undigested
    computations are all preempted episodes (``*_uncertain_preempted``), at
    most ``MAX_PREEMPTED_UNCERTAIN`` of the loaded blocks."""
    if d is None:
        return {"pass": False, "reason": "not evaluated"}
    judged = d["external_matched"] + d["external_mismatched"]
    seen = d["external_compared"] + d["external_uncertain"] + d["external_unreferenced"]
    vacuous = vacuous_ok and seen == 0
    ok = (
        d["local_mismatched"] == 0
        and d["external_mismatched"] == 0
        and d["local_unreferenced"] == 0
        and d["external_unreferenced"] == 0
    )
    if min_judged is not None and not vacuous:
        ok = ok and judged > 0 and judged >= min_judged * seen
    preempted = 0
    if exhaustive:
        loaded = 0
        for t in ("local", "external"):
            unc, pre = d[f"{t}_uncertain"], d.get(f"{t}_uncertain_preempted", 0)
            ok = ok and unc == pre
            preempted += pre
            loaded += d[f"{t}_compared"] + unc + d[f"{t}_unreferenced"]
        ok = ok and preempted <= MAX_PREEMPTED_UNCERTAIN * loaded
    if d["invariant"]:
        # Split-variant: only same-split copies were judged (digest_compare).
        ok = ok and d["computed_exact"] == d["computed_compared"]
    keys = [k for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    cov = {}
    if d.get("split_invariant") is False:
        seen_c = d["computed_compared"] + d.get("computed_unmatched_split", 0)
        cov = {"computed_coverage": d["computed_compared"] / seen_c if seen_c else None}
    return {
        "pass": ok,
        **cov,
        **({"vacuous": "no external block in the digested sample"} if vacuous else {}),
        **({"preempted_uncertain": preempted} if preempted else {}),
        **{k: d[k] for k in keys},
        "first_bad": next((g["first_bad"] for g in d["groups"].values() if g["first_bad"]), None),
    }


def digest_groups(probe: Path) -> dict[str, Any]:
    """G2.digest_groups: every group whose blocks a connector transfers (not
    skipped by the oracle, not scratch) has digest records in the probe dir,
    and the probe did not skip it. Otherwise G2's digest checks judge only
    some of the loaded bytes (e.g. Mamba state) and a connector loading wrong
    bytes into the others passes them."""
    f = probe / "spec.json"
    if not f.exists():
        return {"pass": False, "reason": "spec dump missing"}
    spec = json.loads(f.read_text())
    skipped = spec.get("probe", {}).get("digest_skipped_groups", {})
    seen: set[int] = set()
    for d in probe.glob("digest-*.jsonl"):
        for line in d.read_text().splitlines():
            if line.strip():
                seen.add(int(json.loads(line)["group"]))
    need = [
        gid for gid, (label, g, _) in enumerate(oracle.group_kinds(spec))
        if g is not None and label != "scratch"
    ]  # fmt: skip
    missing = {str(gid): skipped.get(str(gid), "no digest record") for gid in need if gid not in seen or str(gid) in skipped}  # fmt: skip
    return {"pass": bool(need) and not missing, "groups": need, "undigested": missing}


def _reported(**levels: tuple[float | None, float]) -> dict[str, Any]:
    """Hit-rate levels (value, former bar): reported next to the verdict,
    never part of ``pass``. ``below``: the value is missing or under it."""
    return {
        k: {"value": v, "reference": ref, "below": v is None or v < ref}
        for k, (v, ref) in levels.items()
    }


def _hit_verdict(r: dict[str, Any] | None, workload: str) -> dict[str, Any]:
    """G3: ``pass`` = correctness only (excess within one block per request,
    no load errors); ``E_ext`` and the oracle hit rate are ``reported``."""
    if r is None:
        return {"pass": False, "reason": "not evaluated"}
    w = r["hit"]["warm"]
    e = w["efficiency"]
    ok = w["excess_tokens"] <= w["excess_limit"] and not r["load_errors"]
    return {
        "pass": ok,
        "reported": _reported(
            efficiency=(e, MIN_EFFICIENCY),
            oracle_hit_rate=(w["oracle_hit_rate"], ORACLE_FLOOR[workload]),
        ),
        "efficiency": e,
        "oracle_hit_rate": w["oracle_hit_rate"],
        "hit_rate": w["hit_rate"],
        "local_hit_rate": w["local_hit_tokens"] / w["prompt_tokens"] if w["prompt_tokens"] else 0.0,
        "external_hit_rate": w["external_hit_rate"],
        "excess_tokens": w["excess_tokens"],
        "excess_limit": w["excess_limit"],
        "load_errors": len(r["load_errors"]),
    }


def _metric(metrics: dict[str, float], name: str) -> float:
    return sum(n for k, n in metrics.items() if k.split("{")[0] == name)


def _evictions(metrics: dict[str, float]) -> float:
    return _metric(metrics, "pegaflow_cache_block_evictions_total")


_WARM_TIMED_FIELDS = (
    "efficiency_c", "efficiency_c_own_local", "excess_tokens", "excess_limit",
    "unmatched_requests", "hit_rate", "oracle_hit_rate", "oracle_c_hit_rate",
    "lru_hit_rate", "freeze_hit_rate", "lru_vs_oracle_c", "freeze_vs_oracle_c",
)  # fmt: skip
# Every field ``_capacity_verdict`` reads by index (dotted paths).
_CAPACITY_FIELDS = (
    "passes", "capacity_bytes", "load_errors", "capacity.warm_timed",
    *(f"capacity.warm_timed.{k}" for k in _WARM_TIMED_FIELDS),
    "capacity.diagnostics.oracle_c.groups", "capacity.diagnostics.oracle_c.passes",
    "capacity.diagnostics.lru.groups", "capacity.diagnostics.lru.passes",
)  # fmt: skip


def _missing_fields(r: dict[str, Any], paths: tuple[str, ...]) -> list[str]:
    out = []
    for path in paths:
        x: Any = r
        for k in path.split("."):
            if not isinstance(x, dict) or k not in x:
                out.append(path)
                break
            x = x[k]
    return out


def _capacity_verdict(r: dict[str, Any] | None, pegaflow: bool, gated: bool) -> dict[str, Any]:
    """G5 for one cap stage. ``pegaflow``: the arm stores in a PegaFlow server
    (pool size and load bytes must be proven). ``gated``: the gated fraction
    (else the numbers are a reported curve point, ``pass`` informational).
    ``pass`` is correctness and measurement validity only; ``E_C`` (and the
    LRU / freeze shares) are ``reported``."""
    if r is None or "capacity" not in r:
        reason = (r or {}).get("error", "not evaluated")
        return {"pass": False, "reason": reason}
    # S13: a record without a field this verdict needs (e.g. written before
    # the field existed) cannot be evaluated, which is a fail, not a crash
    # that loses every other verdict of the run.
    missing = _missing_fields(r, _CAPACITY_FIELDS)
    if missing:
        return {"pass": False, "reason": f"cannot evaluate: missing {', '.join(missing)}"}
    c = r["capacity"]
    w = c["warm_timed"]
    cold = c.get("cap_cold", {})
    e = w["efficiency_c"]
    passes = {n: p for n, p in r["passes"].items() if not n.startswith("_")}
    pm = {n: p.get("pegaflow_metrics", {}) for n, p in passes.items()}
    evicted = {n: _evictions(m) for n, m in pm.items()}
    resident_end = {n: p["pegaflow_end"] for n, p in passes.items() if "pegaflow_end" in p}
    io = {
        n: {
            "saved_bytes": _metric(m, "pegaflow_save_bytes_total"),
            "loaded_bytes": _metric(m, "pegaflow_load_bytes_total"),
            "external_hit_bytes": r.get("external_load_bytes", {}).get(n),
        }
        for n, m in pm.items()
    }
    pool = r.get("pool_capacity_bytes")
    # A server that grants itself more memory than C is not measured at C;
    # a PegaFlow arm that cannot show its pool size fails.
    pool_ok = pool <= r["capacity_bytes"] if pool is not None else not pegaflow
    # Every external hit must have been loaded from the C-byte server: load
    # bytes over the scored passes cover the external-hit blocks.
    scored = [n for n in passes if n.removeprefix("cap_") not in ("cold", "warm")]
    need = sum(io[n]["external_hit_bytes"] or 0 for n in scored)
    loaded = sum(io[n]["loaded_bytes"] for n in scored)
    loads_ok = not pegaflow or (
        all(io[n]["external_hit_bytes"] is not None for n in scored) and loaded >= need
    )
    ok = (
        w["excess_tokens"] <= w["excess_limit"]
        and not r["load_errors"]
        and pool_ok
        and loads_ok
        and w["unmatched_requests"] == 0
    )
    diag = c["diagnostics"]
    return {
        "pass": ok,
        "gated": gated,
        "reported": _reported(efficiency_c=(e, MIN_EFFICIENCY_C)),
        "efficiency_c": e,
        # Within-pass multi-turn reuse (turn k+1 after GPU eviction): reported.
        "efficiency_c_cold": cold.get("efficiency_c"),
        "cold_ext_oracle_c_share": cold.get("ext_oracle_c_share"),
        "freeze_vs_oracle_c_cold": cold.get("freeze_vs_oracle_c"),
        "efficiency_c_own_local": w["efficiency_c_own_local"],
        "unique_bytes": r.get("unique_bytes"),
        "capacity_bytes": r["capacity_bytes"],
        "capacity_fraction": r.get("capacity_fraction"),
        "pool_capacity_bytes": pool,
        "pool_ok": pool_ok,
        "loads_ok": loads_ok,
        "loaded_bytes": loaded,
        "external_hit_bytes": need,
        "unmatched_requests": w["unmatched_requests"],
        "hit_rate": w["hit_rate"],
        "oracle_hit_rate": w["oracle_hit_rate"],
        "oracle_c_hit_rate": w["oracle_c_hit_rate"],
        "lru_hit_rate": w["lru_hit_rate"],
        "freeze_hit_rate": w["freeze_hit_rate"],
        "lru_vs_oracle_c": w["lru_vs_oracle_c"],
        "freeze_vs_oracle_c": w["freeze_vs_oracle_c"],
        "excess_tokens": w["excess_tokens"],
        "excess_limit": w["excess_limit"],
        "load_errors": len(r["load_errors"]),
        "evicted": evicted,
        "server_end_bytes": resident_end,
        "server_io_bytes": io,
        "oracle_c_groups": diag["oracle_c"]["groups"],
        "lru_groups": diag["lru"]["groups"],
        "oracle_c_passes": diag["oracle_c"]["passes"],
        "lru_passes": diag["lru"]["passes"],
    }


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _cmp(mine: list[float], theirs: list[float], lower: bool, slack: float) -> dict[str, Any]:
    """Median of ``mine`` vs median of ``theirs``. ``slack`` 0: strictly better;
    otherwise at most ``slack`` (fraction) worse."""
    a, b = _median(mine), _median(theirs)
    if a is None or b is None:
        ok = False
    elif slack == 0:
        ok = a < b if lower else a > b
    else:
        ok = a <= b * (1 + slack) if lower else a >= b * (1 - slack)
    return {"pass": ok, "value": mine, "baseline": theirs, "slack": slack}


def _perf_verdicts(
    rep: dict[str, Any] | None,
    base_rep: dict[str, Any] | None,
    gsm: dict[str, Any] | None,
    base_gsm: dict[str, Any] | None,
    replay_ok: bool,
) -> dict[str, Any]:
    """G4 vs baseline vLLM. Replay numbers only count when the replay KV was
    correct (``replay_ok``)."""

    def passes(r: dict[str, Any] | None, names: tuple[str, ...], key: str) -> list[float]:
        ps = (r or {}).get("passes", {})
        return [ps[n][key] for n in names if n in ps and key in ps[n]]

    def timed(r: dict[str, Any] | None) -> tuple[str, ...]:
        ps = (r or {}).get("passes", {})
        return tuple(n for n in ps if n.startswith("warm") and n != "warm")

    g: dict[str, Any] = {}
    for key, lower in (("ttft_p50", True), ("req_per_s", False)):
        for name, names, slack in (
            ("replay_warm", timed(rep), 0.0),
            ("replay_cold", ("cold",), MAX_OVERHEAD),
        ):
            mine = passes(rep, names, key)
            theirs = passes(base_rep, timed(base_rep) if name == "replay_warm" else names, key)
            v = _cmp(mine, theirs, lower, slack)
            v["pass"] = v["pass"] and replay_ok
            g[f"G4.{name}.{key}"] = v
    for name in ("cold", "warm"):
        g[f"G4.gsm8k_{name}.seconds"] = _cmp(
            passes(gsm, (name,), "seconds"),
            passes(base_gsm, (name,), "seconds"),
            True,
            MAX_OVERHEAD,
        )
    # S9: the untimed warmup moves one-time costs (first save, gRPC setup)
    # out of the cold passes; reported per workload, never gated.
    g["G4.warmup_seconds"] = {
        wl: {
            "value": (passes(r, (workloads.WARMUP_PASS,), "seconds") or [None])[0],
            "baseline": (passes(b, (workloads.WARMUP_PASS,), "seconds") or [None])[0],
        }
        for wl, r, b in (("replay", rep, base_rep), ("gsm8k", gsm, base_gsm))
    }
    return g


def verdicts(
    results: dict[str, dict[str, Any]],
    run_dir: Path,
    committed: Path,
    gate: dict[str, Any] | None = None,
    pegaflow_dirs: dict[str, Path | None] | None = None,
    connectors: dict[str, tuple[str | None, Path | None]] | None = None,
) -> dict[str, Any]:
    """``connectors``: label -> (connector module, ``connector_root``)."""
    v: dict[str, Any] = {}
    connectors = connectors or {}
    split_inv = bool((gate or {}).get("split_invariant", True))
    ctrl = split_control(run_dir)
    base = results.get("baseline", {})
    ref_cold = _outputs(run_dir, "baseline", "cold")
    ref_acc = base.get("gsm8k", {}).get("passes", {}).get("cold", {}).get("accuracy")
    base_probe = run_dir / "baseline" / "gsm8k" / "probe"
    invariant = base.get("gsm8k", {}).get("invariant", False)

    # Controls: the probe changes nothing (plain == baseline), poisoning and
    # the local reset change nothing (baseline warm == cold), local hits are
    # byte-identical to their computed copies.
    v["baseline"] = {
        "G2.plain_equals_baseline": {
            "pass": _differ(ref_cold, _outputs(run_dir, "plain", "cold")) == 0,
            "differ": _differ(ref_cold, _outputs(run_dir, "plain", "cold")),
        },
        # Output equality needs batch invariance; gate.json declares when a
        # model cannot have it, and then only accuracy and digests count.
        "G2.warm_equals_cold": {
            "pass": not invariant or _differ(ref_cold, _outputs(run_dir, "baseline", "warm")) == 0,
            "differ": _differ(ref_cold, _outputs(run_dir, "baseline", "warm")),
            "invariant": invariant,
        },
        "G2.digest_self.gsm8k": _digest_verdict(base.get("gsm8k", {}).get("digest_self"), None),
        "G2.digest_groups": digest_groups(base_probe),
        **(
            {"G0.provenance": g0_provenance(run_dir, "baseline", pegaflow_dirs.get("baseline"))}
            if pegaflow_dirs is not None
            else {}
        ),
        # gate.json's split_invariant is established here, not declared.
        "G2.split_control": _split_control_verdict(ctrl, split_inv),
        "G2.digest_self.cap": _digest_verdict(
            base.get("cap", {}).get("digest_self"), None, exhaustive=True
        )
        if "cap" in base
        else {"pass": True, "na": "no cap workload"},
    }
    base_errors = {wl: r["error"] for wl, r in base.items() if "error" in r}
    if base_errors:
        v["baseline"]["errors"] = base_errors

    for label, per_all in results.items():
        kind = "selftest" if label.startswith("selftest-") else label
        if kind in ("plain", "baseline"):
            continue
        # A crashed stage is recorded under "errors" and evaluates as missing.
        per_wl = {wl: r for wl, r in per_all.items() if "error" not in r}
        g: dict[str, Any] = {}
        gsm = per_wl.get("gsm8k")
        g["G0.spec"] = g0(run_dir, label, committed)
        if pegaflow_dirs is not None:
            mod, root = connectors.get(label, (None, None))
            g["G0.provenance"] = g0_provenance(run_dir, label, pegaflow_dirs.get(label), mod, root)
        g["G2.digest_self.gsm8k"] = _digest_verdict(gsm and gsm["digest_self"], MIN_JUDGED_GSM8K)
        g["G2.digest_groups"] = digest_groups(run_dir / label / "gsm8k" / "probe")
        vs_base = (
            digest_compare.compare(
                base_probe,
                run_dir / label / "gsm8k" / "probe",
                invariant,
                split_inv,
                _hit_alignment(run_dir),
            )
            if gsm and base_probe.exists()
            else None
        )
        g["G2.digest_vs_baseline.gsm8k"] = _digest_verdict(vs_base, MIN_JUDGED_GSM8K)
        # Split-variant: cold computed KV is judged against baseline cold,
        # warm against the baseline's split pass (same split).
        _coverage_verdict(g["G2.digest_vs_baseline.gsm8k"], vs_base, ("cold", "warm"))
        diffs = {p: _differ(ref_cold, _outputs(run_dir, label, p)) for p in ("cold", "warm")}
        split_diffs = split_differ(ctrl, _outputs(run_dir, label, "warm"))
        g["G2.gsm8k_outputs"] = _outputs_verdict(diffs, invariant, split_inv, split_diffs)
        if kind == "selftest":
            # Negative arms must be caught by both digest checks (a single
            # flipped byte need not change a greedy output, so outputs are
            # only reported); the honest arm must pass every G2 check.
            digests = ("G2.digest_self.gsm8k", "G2.digest_vs_baseline.gsm8k")
            if label == "selftest-honest":
                g["pass"] = all(
                    g[k]["pass"] for k in (*digests, "G2.gsm8k_outputs", "G2.digest_groups")
                )
                g["expect"] = "pass every G2 check"
            else:
                g["pass"] = not any(g[k]["pass"] for k in digests)
                g["expect"] = "fail both G2 digest checks"
            v[label] = g
            continue
        accs = {p: gsm["passes"][p]["accuracy"] for p in ("cold", "warm")} if gsm else {}
        g["G2.gsm8k"] = {
            "pass": ref_acc is not None
            and len(accs) == 2
            and all(abs(a - ref_acc) <= ACC_TOL for a in accs.values()),
            "baseline": ref_acc,
            **{
                f"{p}_delta_pp": (a - ref_acc) * 100 for p, a in accs.items() if ref_acc is not None
            },
        }
        rep = per_wl.get("replay")
        # Replay shares prefixes with untagged conversations, so many loads stay
        # uncertain; only require that some were judged and none mismatched.
        g["G2.digest_self.replay"] = _digest_verdict(rep and rep["digest_self"], 0.0)
        g["G3.gsm8k_warm"] = _hit_verdict(gsm, "gsm8k")
        # G3/G4 replay numbers only count when the replay KV was correct.
        g["G3.replay_warm"] = _hit_verdict(rep, "replay")
        if not g["G2.digest_self.replay"]["pass"]:
            g["G3.replay_warm"]["pass"] = False
        evicted = {
            f"{wl}.{p}": _evictions(r["passes"][p].get("pegaflow_metrics", {}))
            for wl, r in per_wl.items()
            if not wl.startswith("cap") and "passes" in r
            for p in r["passes"]
            if not p.startswith("_")
        }
        # The oracle has unlimited capacity: an evicting pool measures its size.
        if kind in _OFFLOAD:
            # No PegaFlow server: the CPU cache exports no eviction counter.
            g["G3.no_eviction"] = {"pass": True, "na": f"{kind} arm has no PegaFlow server"}
        else:
            g["G3.no_eviction"] = {
                "pass": bool(evicted) and not any(evicted.values()),
                "evicted": evicted,
            }
        g |= _perf_verdicts(
            rep, base.get("replay"), gsm, base.get("gsm8k"), g["G2.digest_self.replay"]["pass"]
        )
        cap = per_wl.get("cap")
        pegaflow = kind not in _OFFLOAD
        # The arm under test must load in the cap passes (a passing G5 needs
        # E_C >= 0.9) and every load is judged; only reference arms may load
        # nothing.
        ref_arm = kind in _RECORD_ONLY
        g["G2.digest_self.cap"] = _digest_verdict(
            cap and cap.get("digest_self"), 0.0, vacuous_ok=ref_arm, exhaustive=True
        )
        g["G5.replay_warm"] = _capacity_verdict(cap, pegaflow, gated=True)
        if not g["G2.digest_self.cap"]["pass"]:
            g["G5.replay_warm"]["pass"] = False
        # Other fractions: a reported curve, never gated.
        curve = {}
        for wl in sorted(per_wl):
            if wl.startswith("cap") and wl != "cap":
                cv = _capacity_verdict(per_wl[wl], pegaflow, gated=False)
                dv = _digest_verdict(
                    per_wl[wl].get("digest_self"), 0.0, vacuous_ok=ref_arm, exhaustive=True
                )
                cv["digest_self"] = dv["pass"]
                curve[wl] = {
                    k: cv.get(k)
                    for k in (
                        "pass",
                        "reason",
                        "capacity_fraction",
                        "capacity_bytes",
                        "efficiency_c",
                        "oracle_c_hit_rate",
                        "lru_hit_rate",
                        "freeze_vs_oracle_c",
                        "lru_vs_oracle_c",
                        "hit_rate",
                        "evicted",
                        "loads_ok",
                        "digest_self",
                    )
                }
        g["G5.curve"] = curve
        errors = {wl: r["error"] for wl, r in per_all.items() if "error" in r}
        if errors:
            g["errors"] = errors
        g["pass"] = not errors and all(
            x["pass"] for x in g.values() if isinstance(x, dict) and "pass" in x
        )
        if kind in _RECORD_ONLY:
            # Reference arms: every number is recorded, nothing is gated.
            g["required"] = False
        v[label] = g
    v["baseline"]["pass"] = not base_errors and all(
        x["pass"] for x in v["baseline"].values() if isinstance(x, dict) and "pass" in x
    )
    return v


_HIT_KEYS = {
    "G3.gsm8k_warm": ("hit_rate", "local_hit_rate", "external_hit_rate", "oracle_hit_rate", "efficiency"),
    "G3.replay_warm": ("hit_rate", "local_hit_rate", "external_hit_rate", "oracle_hit_rate", "efficiency"),
    "G5.replay_warm": ("hit_rate", "oracle_hit_rate", "oracle_c_hit_rate", "lru_hit_rate",
                       "freeze_hit_rate", "efficiency_c", "lru_vs_oracle_c", "freeze_vs_oracle_c"),
}  # fmt: skip


def hit_report(v: dict[str, Any]) -> dict[str, Any]:
    """Every hit-rate number per arm (H, H*, E_ext, H*_C, H_lru, freeze, E_C),
    marked ``reported``: no hit-rate level is gated; the onboarding agent
    judges them in ITERATIONS.md and the review agent checks that judgment."""
    out: dict[str, Any] = {}
    for label, g in v.items():
        if label == "baseline" or not isinstance(g, dict):
            continue
        arm = {
            k: {f: g[k].get(f) for f in fields} | {"below_reference": [
                n for n, x in g[k].get("reported", {}).items() if x["below"]
            ]}
            for k, fields in _HIT_KEYS.items()
            if isinstance(g.get(k), dict) and "reason" not in g[k]
        }  # fmt: skip
        if arm:
            out[label] = {"status": "reported"} | arm
    return out


def git_rev(root: Path = REPO) -> str:
    with suppress(Exception):
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    return (root / ".rsi-rev").read_text().strip() if (root / ".rsi-rev").exists() else "unknown"


def capacity_floor(base_cap: dict[str, Any] | None) -> dict[str, Any]:
    """Gated ``C`` vs ``MIN_CAPACITY_GPU_KV`` x vLLM GPU KV bytes (baseline cap)."""
    if not base_cap or "unique_bytes" not in base_cap:
        return {"pass": False, "reason": "no baseline cap result"}
    u, kv = base_cap["unique_bytes"], base_cap["gpu_kv_bytes"]
    c = int(CAPACITY_FRACTION * u)
    return {
        "pass": c >= MIN_CAPACITY_GPU_KV * kv,
        "unique_bytes": u,
        "capacity_bytes": c,
        "gpu_kv_bytes": kv,
        "capacity_over_gpu_kv": c / kv if kv else None,
        "min_capacity_over_gpu_kv": MIN_CAPACITY_GPU_KV,
    }


class UniqueEstimate:
    """``U`` of a cap trace before running it, accumulated per conversation:
    distinct prompt-prefix blocks per group (the trace's shared history) plus
    each request's generated complete blocks, at ``oracle.block_bytes``. For
    sizing only; the floor is checked on the measured ``U``."""

    def __init__(self, spec: dict[str, Any], out_tokens: int):
        kinds = [(label, bt) for label, g, bt in oracle.group_kinds(spec) if g is not None]
        sizes = oracle.block_bytes(spec)
        self._groups = [
            (bt, sizes[gi], set()) for gi, (label, bt) in enumerate(kinds) if label != "scratch"
        ]
        self._out = out_tokens
        self._gen = 0

    def add(self, turns: list[list[int]]) -> None:
        for bt, size, seen in self._groups:
            for ids in turns:
                h = 0
                for b in range(len(ids) // bt):
                    h = hash((h, tuple(ids[b * bt : (b + 1) * bt])))
                    seen.add(h)
                self._gen += size * ((len(ids) + self._out - 1) // bt - len(ids) // bt)

    @property
    def bytes(self) -> int:
        return self._gen + sum(size * len(seen) for _, size, seen in self._groups)


def plan_cap_convs(args, spec: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ``--cap-convs auto``: the fewest conversations (from
    ``CAP_MIN_CONVS``, all turns up to ``--cap-turns``) whose estimated
    ``C(0.5)`` is ``>= CAP_FLOOR_MARGIN x MIN_CAPACITY_GPU_KV x`` GPU KV.
    Fails closed: if the whole file is not enough, all of it is used and the
    measured floor marks the run not official."""
    if args.cap_convs != "auto":
        return {"cap_convs": int(args.cap_convs), "mode": "fixed"}
    if spec is None:
        return {"cap_convs": CAP_MIN_CONVS, "mode": "auto", "reason": "no baseline spec"}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    budget = args.max_model_len - args.replay_out_tokens - 8
    lines = Path(args.replay_file).read_text().splitlines()[args.replay_skip :]
    need = CAP_FLOOR_MARGIN * MIN_CAPACITY_GPU_KV * gpu_kv_bytes(spec) / CAPACITY_FRACTION
    est = UniqueEstimate(spec, args.replay_out_tokens)
    used = 0
    for line in lines:
        used += 1
        est.add(workloads.conv_turns(tok, json.loads(line), budget, args.cap_turns))
        if used >= CAP_MIN_CONVS and est.bytes >= need:
            break
    return {
        "cap_convs": used,
        "mode": "auto",
        "estimated_unique_bytes": est.bytes,
        "needed_unique_bytes": int(need),
        "enough": est.bytes >= need,
        "available_convs": len(lines),
    }


class PoolTooSmall(RuntimeError):
    """An uncapped stage's ``U`` exceeds ``--pool-size``: the store would
    evict and G3 would measure the pool size, not the connector."""


_STORES = ("old", "native", "lmcache", "under_test")


def estimate_unique(args, spec: dict[str, Any], wl: str) -> dict[str, Any]:
    """``UniqueEstimate`` of an uncapped workload before any stage. Replay
    (``ignore_eos``): prompts + exactly ``--replay-out-tokens`` per request.
    gsm8k: prompts only, a lower bound (completions stop on data)."""
    est = UniqueEstimate(spec, args.replay_out_tokens if wl == "replay" else 1)
    if wl == "replay":
        for turns in workloads.build_turns(
            Path(args.replay_file),
            args.model,
            args.replay_convs,
            args.max_model_len - args.replay_out_tokens - 8,
            args.replay_turns,
            skip=args.replay_skip,
        ):
            est.add(turns)
        how = "estimate"
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for prompt, _ in workloads.gsm8k_prompts(
            Path(args.gsm8k_dir), args.shots, args.gsm8k_limit
        ):
            est.add([tok(prompt)["input_ids"]])
        how = "lower bound (prompts only)"
    return {"workload": wl, "unique_bytes": est.bytes, "how": how}


def check_pool(args, u: dict[str, Any]) -> dict[str, Any]:
    pool = parse_size(args.pool_size)
    rec = u | {"pool_bytes": pool, "fits": u["unique_bytes"] <= pool}
    print(f"[pool] {rec}", flush=True)
    if not rec["fits"]:
        raise PoolTooSmall(
            f"{u['workload']}: U {u['unique_bytes'] / 2**30:.1f} GiB ({u['how']}) exceeds "
            f"--pool-size {args.pool_size}"
        )
    return rec


def g5_problems(v: dict[str, Any], floor: dict[str, Any], plan: dict[str, Any] | None) -> list[str]:
    """Why a run's G5 cannot be official: the gated store is below the
    capacity floor (a smaller store measures vLLM's GPU cache), or the trace
    does not separate a store frozen after its first fill from ``H*_C``."""
    out = []
    if not floor["pass"]:
        why = " (whole replay file used)" if (plan or {}).get("enough") is False else ""
        out.append(f"capacity below {MIN_CAPACITY_GPU_KV:g}x GPU KV{why}")
    for label, g in v.items():
        if label in _REFERENCE or label.startswith("selftest-"):
            continue
        fz = g.get("G5.replay_warm", {}).get("freeze_vs_oracle_c")
        if fz is None or fz >= FREEZE_SEPARATION:
            out.append("G5 trace does not separate freeze from H*_C")
    return out


CONTAMINATION_P99 = 10.0  # x the baseline's p99 TTFT of the same pass


def contamination(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """G4-timed replay passes with a stall from outside the arm (another
    process's pinned-pool allocation, JIT, a shared GPU), both ways:
    - an arm's p99 TTFT above ``CONTAMINATION_P99`` x the in-run baseline's
      p99 for the same pass;
    - the baseline's p99 above ``CONTAMINATION_P99`` x the median of the
      arms' p99 for that pass (a baseline stall inflates every arm's G4
      margin, S8).
    Nothing is dropped: the run becomes non-official."""
    base = results.get("baseline", {}).get("replay", {}).get("passes", {})
    out = []
    arms: dict[str, list[float]] = {}
    for label, per in results.items():
        if label == "baseline":
            continue
        for name, p in per.get("replay", {}).get("passes", {}).items():
            mine = p.get("ttft_p99") if isinstance(p, dict) else None
            # G4-timed passes only: cold, warm2.. (``warm`` is the digested one).
            if name.startswith("_") or name == "warm" or mine is None:
                continue
            arms.setdefault(name, []).append(mine)
            b = base.get(name, {}).get("ttft_p99")
            if b and mine > CONTAMINATION_P99 * b:
                out.append({"arm": label, "pass": name, "ttft_p99": mine, "baseline_p99": b})
    for name, ps in sorted(arms.items()):
        b = base.get(name, {}).get("ttft_p99")
        med = statistics.median(ps)
        if b and med and b > CONTAMINATION_P99 * med:
            out.append({"arm": "baseline", "pass": name, "ttft_p99": b, "arms_median_p99": med})
    return out


def official_problems(
    args, arms: list[str], wls: list[str], gate_file: Path, committed: Path
) -> list[str]:
    def same(k: str, v: Any) -> bool:
        got = getattr(args, k)
        if k == "cap_fractions":
            try:
                return parse_fractions(got) == parse_fractions(v)
            except ValueError:
                return False
        return got == v

    out = [
        f"{k}={getattr(args, k)} (held-out slice; official {v})"
        if k == "replay_skip"
        else f"{k}={getattr(args, k)} (official {v})"
        for k, v in OFFICIAL.items()
        if not same(k, v)
    ]
    kinds = {arm_kind(a) for a in arms}
    for need in ("plain", "baseline", "under_test"):
        if need not in kinds:
            out.append(f"arm {need} missing")
    if set(wls) != set(WORKLOADS):
        out.append(f"workloads {wls} != {','.join(WORKLOADS)}")
    if args.num_gpu_blocks or args.vllm_arg:
        out.append("--num-gpu-blocks / --vllm-arg override")
    for a in arms:
        mod = a.split(":")[0]
        if arm_kind(a) == "under_test" and not (mod == "pegaflow" or mod.startswith("pegaflow.")):
            # M8: the snapshot holds only ``pegaflow``; a module elsewhere
            # (e.g. ``rsi.*`` in the gate tree) is never staged or scanned.
            out.append(f"arm under test {a}: module {mod} is outside the pegaflow package")
    if "under_test" in kinds and not args.under_test_path:
        # Without it the arm imports the venv's (unverified, unsnapshotted)
        # pegaflow tree: G0.provenance would accept any file there.
        out.append("--under-test-path missing: the connector under test has no verified snapshot")
    for f in (gate_file, committed):
        if not f.exists():
            out.append(f"{f} missing")
    return out


def _nvcc_version(nvcc: str | None) -> str | None:
    if not nvcc:
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else None


def cuda_env(env: dict[str, str] | None = None, torch_cuda: str | None = None) -> dict[str, Any]:
    """The CUDA toolchain every stage inherits (vLLM / FlashInfer JIT-compile
    kernels with the ``nvcc`` of ``CUDA_HOME`` or ``PATH``): paths, nvcc
    versions and the venv torch's CUDA version (from a subprocess, so the
    gate never imports torch)."""
    env = dict(os.environ) if env is None else env
    if torch_cuda is None:
        with suppress(OSError, subprocess.SubprocessError):
            torch_cuda = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.cuda)"],
                capture_output=True, text=True, timeout=300, env=env,
            ).stdout.strip() or None  # fmt: skip
    home = env.get("CUDA_HOME") or None
    home_nvcc = str(Path(home) / "bin" / "nvcc") if home else None
    path_nvcc = shutil.which("nvcc", path=env.get("PATH", ""))
    return {
        "torch_cuda": torch_cuda,
        "CUDA_HOME": home,
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"),
        "PATH": env.get("PATH"),
        "nvcc_path": path_nvcc,
        "nvcc_path_version": _nvcc_version(path_nvcc),
        "nvcc_cuda_home": home_nvcc,
        "nvcc_cuda_home_version": _nvcc_version(home_nvcc)
        if home_nvcc and Path(home_nvcc).is_file()
        else None,  # fmt: skip
    }


def cuda_problems(rec: dict[str, Any]) -> list[str]:
    """Official only with one CUDA major for torch and every nvcc a stage can
    pick (a mismatched nvcc fails a JIT build mid-run, or builds kernels for
    another toolkit)."""
    major = (rec.get("torch_cuda") or "").split(".")[0]
    if not major:
        return ["CUDA env: torch CUDA version unknown"]
    out = []
    if rec.get("CUDA_HOME") and rec.get("nvcc_cuda_home_version") is None:
        out.append(f"CUDA env: CUDA_HOME={rec['CUDA_HOME']} has no working bin/nvcc")
    found = [
        (k, rec[f"{k}_version"]) for k in ("nvcc_path", "nvcc_cuda_home") if rec.get(f"{k}_version")
    ]
    if not found:
        out.append("CUDA env: no nvcc on PATH or under CUDA_HOME (JIT kernels cannot build)")
    for k, ver in found:
        if ver.split(".")[0] != major:
            out.append(f"CUDA env: {rec[k]} is CUDA {ver}, torch is CUDA {rec['torch_cuda']}")
    return out


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def install_stop_handlers() -> None:
    """SIGINT / SIGTERM / SIGHUP abort the whole run: the handler raises
    SystemExit, so every ``launched`` context unwinds and stops its process
    group (vLLM / pegaflow-server run in their own sessions and never get
    the signal), and the gate exits 128 + signal. Installing it for SIGINT
    also keeps ``asyncio.run`` from installing its own SIGINT handler, which
    only cancels the running task: a pass that swallows or converts the
    CancelledError would let the run go on. The request is also recorded
    (``check_stop``), so no later pass or stage starts even if an exception
    is converted or caught on the way. A signal during ``stop`` is deferred
    until that group is gone; later signals are ignored."""

    def handler(signum: int, _frame: Any) -> None:
        for s in STOP_SIGNALS:
            signal.signal(s, signal.SIG_IGN)
        _STOP["signal"] = signum
        print(f"[gate] {signal.Signals(signum).name}: stopping the run", flush=True)
        if not _STOP["teardown"]:
            raise SystemExit(128 + signum)

    for s in STOP_SIGNALS:
        signal.signal(s, handler)
    import atexit

    atexit.register(stop_live)


def record_provenance(args, run_dir: Path) -> dict[str, Any]:
    """Revs (verified against ``provenance.REMOTE``) and hashes of what the
    run measured: the gate tree, the venv's ``pegaflow`` (reference arms, the
    oracle's ``pegaflow.semantics``), the ``--under-test-path`` tree and the
    pegaflow-server binary (sha256; its tree's ``.rsi-rev`` is a claim)."""
    trees = provenance.Trees(run_dir / ".provenance")
    # In-tree ``__pycache__`` of the gate / venv trees is never read when the
    # gate itself runs with PYTHONPYCACHEPREFIX (stage processes always do).
    pyc_ok = bool(sys.pycache_prefix)
    native = provenance.native_module()
    try:
        out: dict[str, Any] = {
            "gate": trees.tree(REPO, ("rsi", "python/pegaflow"), native, pyc_ok)
            | {"pycache_prefix": sys.pycache_prefix}
        }
        venv = provenance.gate_pegaflow_dir()
        out["venv_pegaflow"] = (
            trees.tree(provenance.tree_root(venv), ("python/pegaflow",), native, pyc_ok)
            | {"pegaflow_dir": str(venv)}
            if venv
            else {"verified": False, "reason": "pegaflow not importable"}
        )
        if args.under_test_path:
            # The arm imports this snapshot, not the tree (M5).
            out["under_test"] = provenance.stage(
                Path(args.under_test_path), Path(args.under_test_overlay), trees
            )
        server = Path(args.pegaflow_server).resolve()
        rev_file = server.parents[2] / ".rsi-rev"
        out["pegaflow_server"] = {
            "path": str(server),
            "sha256": provenance.sha256(server) if server.is_file() else None,
            # A binary cannot be tied to a rev: the tree's claim, unverified.
            "tree_rev_claimed": rev_file.read_text().strip() if rev_file.is_file() else None,
            "python_path_head": str(REPO / "python"),
        }
        return out
    finally:
        trees.cleanup()


class ProvenanceWatch:
    """M10: what ``record_provenance`` verified must stay as it was. The gate
    tree (``rsi/``, ``python/pegaflow``, the model's committed
    ``docs/rsi/models/<model>``), the venv ``pegaflow`` tree, the snapshot
    the arm under test imports and the pegaflow-server binary are
    fingerprinted (blob ids) right after verification and compared before
    every stage and after the last one. Any change makes the run
    non-official, naming the path; the run goes on."""

    def __init__(self, args, model_dir: Path):
        pyc_ok = bool(sys.pycache_prefix)
        gate_subs = ("rsi", "python/pegaflow")
        if model_dir.is_dir():
            gate_subs += (str(model_dir.resolve().relative_to(REPO.resolve())),)
        self.roots: dict[str, tuple[Path, tuple[str, ...], bool]] = {
            "gate": (REPO.resolve(), gate_subs, pyc_ok)
        }
        venv = provenance.gate_pegaflow_dir()
        if venv:
            self.roots["venv_pegaflow"] = (provenance.tree_root(venv), ("python/pegaflow",), pyc_ok)  # fmt: skip
        if getattr(args, "under_test_path", ""):
            self.roots["under_test"] = (Path(args.under_test_overlay).resolve(), ("pegaflow",), False)  # fmt: skip
        server = Path(args.pegaflow_server).resolve()
        self.roots["pegaflow_server"] = (server.parent, (server.name,), False)
        self.base = {r: provenance.fingerprint(*v) for r, v in self.roots.items()}
        self.checks: list[dict[str, Any]] = []

    def check(self, when: str) -> dict[str, list[str]]:
        changes = {}
        for role, v in self.roots.items():
            d = provenance.fingerprint_diff(self.base[role], provenance.fingerprint(*v))
            if d:
                changes[role] = d[:20]
        self.checks.append({"before": when, "at": time.time(), "changes": changes})
        if changes:
            print(f"[gate] provenance changed before {when}: {changes}", flush=True)
        return changes

    def record(self) -> dict[str, Any]:
        return {
            "roots": {
                r: {"root": str(v[0]), "subpaths": list(v[1]), "files": len(self.base[r])}
                for r, v in self.roots.items()
            },  # fmt: skip
            "checks": self.checks,
        }

    def problems(self) -> list[str]:
        return [
            f"provenance: {_PROVENANCE_ROLES.get(role, role)} changed before {c['before']}: "
            + ", ".join(d[:5])
            for c in self.checks
            for role, d in c["changes"].items()
        ]


_PROVENANCE_ROLES = {
    "gate": "gate tree (rsi/, the judge)",
    "venv_pegaflow": "venv pegaflow (oracle semantics, reference arms)",
    "under_test": "connector under test (snapshot)",
}


def scan_record(args, allow: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """M7(b): static scan of the snapshot the arm under test imports; hits
    on ``provenance.SCAN_ALLOWLIST`` are recorded with their reason."""
    if not getattr(args, "under_test_path", ""):
        return {"scanned": None, "allowlisted": [], "flagged": []}
    hits = provenance.scan(Path(args.under_test_overlay) / "pegaflow")
    allowed, flagged = [], []
    for h in hits:
        a = provenance.allowlisted(h, allow)
        if a is None:
            flagged.append(h)
        else:
            allowed.append(h | {"reason": a["reason"]})
    return {
        "scanned": str(Path(args.under_test_overlay) / "pegaflow"),
        "patterns": provenance.SCAN_PATTERNS,
        "allowlisted": allowed,
        "flagged": flagged,
    }


def provenance_problems(prov: dict[str, Any]) -> list[str]:
    """Official only if every tree the run imports is verified: the gate
    tree, the venv's ``pegaflow`` and, with ``--under-test-path``, the
    snapshot of the tree under test (M6)."""
    out = []
    for key, role in _PROVENANCE_ROLES.items():
        rec = prov.get(key)
        if rec is None:
            if key != "under_test":
                out.append(f"{role}: provenance not recorded")
            continue
        if not rec.get("verified"):
            root = rec.get("root") or rec.get("source", {}).get("root")
            out.append(f"{role} {root} at an unverified rev ({rec.get('reason')})")
    return out


def _teardown_failed(e: BaseException) -> bool:
    """``e`` or any exception it was raised from/during is a TeardownError
    (``launched``'s finally raises it while another error is in flight)."""
    seen: set[int] = set()
    x: BaseException | None = e
    while x is not None and id(x) not in seen:
        if isinstance(x, TeardownError):
            return True
        seen.add(id(x))
        x = x.__cause__ or x.__context__
    return False


def run_stage(args, gate, arm: str, wl: str, out: Path, **kw) -> dict[str, Any]:
    """Run one stage; a reference arm's crash is recorded, not fatal, unless
    its teardown failed (the next stage would share a live GPU) or a stop
    signal was received (whatever the exception became on the way)."""
    check_stop()
    since = len(_LEASES)
    try:
        return run_arm_workload(args, gate, arm, wl, out, **kw)
    except Exception as e:
        # Every failed stage leaves its cause in the run record (a port race
        # names the port and the lease files); a required arm's failure still
        # aborts the run.
        with suppress(OSError):
            gdir = gate_dir(out)
            gdir.mkdir(parents=True, exist_ok=True)
            err = {"arm": arm, "workload": wl, "type": type(e).__name__, "error": str(e)[:4000]}
            err["port_leases"] = release_leases(since)
            (gdir / "stage-error.json").write_text(json.dumps(err, indent=1))
        check_stop()
        if arm_kind(arm) not in _RECORD_ONLY or _teardown_failed(e):
            raise
        print(f"[{arm}/{wl}] reference arm failed: {e!r}", flush=True)
        return {"arm": arm, "workload": wl, "error": f"{type(e).__name__}: {e}"[:2000]}
    finally:
        release_leases(since)  # also on a stop signal, before the gate exits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="<model>-<iter>, used in the run dir name")
    ap.add_argument(
        "--arms",
        default="plain,baseline,old",
        help="comma list; selftest arms: selftest:honest,selftest:liar,...",
    )
    ap.add_argument(
        "--workloads",
        default=",".join(WORKLOADS),
        help="cap (G5) needs the baseline arm in the same run",
    )
    ap.add_argument("--runs-root", default=str(Path.home() / "kexi" / "rsi-runs"))
    ap.add_argument("--gpu", default="0", help="CUDA device ids (comma list, len = tp)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=OFFICIAL["max_model_len"])
    ap.add_argument(
        "--max-num-batched-tokens", type=int, default=OFFICIAL["max_num_batched_tokens"]
    )
    ap.add_argument("--max-num-seqs", type=int, default=OFFICIAL["max_num_seqs"])
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--vllm-arg", action="append", default=[], help="extra vllm serve arg")
    ap.add_argument(
        "--pool-size",
        default="64gb",
        help="store size of the uncapped stages (PegaFlow pool, native/lmcache CPU cache); "
        "the run fails before a store stage whose estimated or measured U exceeds it",
    )
    ap.add_argument(
        "--pegaflow-server", default=str(REPO / "target" / "release" / "pegaflow-server")
    )
    ap.add_argument(
        "--lmcache-overlay",
        default=str(LMCACHE_OVERLAY),
        help="dir with lmcache + its missing pure deps, on PYTHONPATH of the lmcache arm only",
    )
    ap.add_argument(
        "--under-test-path",
        default="",
        help="python dir put first on PYTHONPATH of the connector-under-test arm only "
        "(e.g. another tree's python/); its rev (.rsi-rev / git) is recorded",
    )
    ap.add_argument("--concurrency", type=int, default=OFFICIAL["concurrency"])
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--gsm8k-dir", default=str(Path.home() / "kexi" / "datasets" / "gsm8k-main"))
    ap.add_argument("--shots", type=int, default=OFFICIAL["shots"])
    ap.add_argument("--gsm8k-limit", type=int, default=OFFICIAL["gsm8k_limit"], help="0 = full")
    ap.add_argument(
        "--replay-file", default=str(Path.home() / "kexi" / "datasets" / "r2egym-256.jsonl")
    )
    ap.add_argument("--replay-convs", type=int, default=OFFICIAL["replay_convs"])
    ap.add_argument(
        "--replay-skip",
        type=int,
        default=OFFICIAL["replay_skip"],
        help="skip the first N conversations of --replay-file (replay and cap): "
        "a held-out slice, never official",
    )
    ap.add_argument("--replay-turns", type=int, default=OFFICIAL["replay_turns"])
    ap.add_argument("--replay-out-tokens", type=int, default=OFFICIAL["replay_out_tokens"])
    ap.add_argument("--replay-repeats", type=int, default=OFFICIAL["replay_repeats"])
    ap.add_argument("--replay-digest-every", type=int, default=OFFICIAL["replay_digest_every"])
    ap.add_argument(
        "--cap-convs",
        default=OFFICIAL["cap_convs"],
        help="int, or auto: the fewest conversations (>= 64) whose estimated "
        "C(0.5) clears the capacity floor (needs an earlier baseline stage)",
    )
    ap.add_argument("--cap-turns", type=int, default=OFFICIAL["cap_turns"])
    ap.add_argument(
        "--cap-fractions",
        default=OFFICIAL["cap_fractions"],
        help=f"C/U list; {CAPACITY_FRACTION} is gated, the rest are reported",
    )
    args = ap.parse_args(argv)
    install_stop_handlers()
    try:
        fractions = parse_fractions(args.cap_fractions)
    except ValueError as e:
        ap.error(str(e))
    args.model = str(Path(args.model).expanduser())
    if len(args.gpu.split(",")) != args.tp:
        ap.error(f"--gpu {args.gpu} does not have --tp {args.tp} devices")
    if args.replay_repeats < 2:
        ap.error("--replay-repeats must be >= 2 (the first warm pass is digested, not timed)")

    model_dir = REPO / "docs" / "rsi" / "models" / Path(args.model).name
    gate_file, committed = model_dir / "gate.json", model_dir / "spec.json"
    gate = json.loads(gate_file.read_text()) if gate_file.exists() else {}
    run_dir = Path(args.runs_root) / f"{date.today().isoformat()}-{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gate_pyc = (run_dir / PYCACHE_DIR).resolve()
    if argv is None and sys.pycache_prefix != str(gate_pyc):
        # Re-exec (same pid) with a fresh bytecode cache: the gate and the
        # oracle never read an in-tree ``__pycache__`` of the gate or venv
        # ``pegaflow`` trees (M5). Nothing ran yet but argument parsing.
        if gate_pyc.exists():
            shutil.rmtree(gate_pyc)
        gate_pyc.mkdir(parents=True)
        env = os.environ | {"PYTHONPYCACHEPREFIX": str(gate_pyc)}
        os.execve(sys.executable, [sys.executable, "-m", "rsi.gates.run_gate", *sys.argv[1:]], env)  # fmt: skip
    arms = [a for a in args.arms.split(",") if a]
    wls = [w for w in args.workloads.split(",") if w]
    if "baseline" in arms:  # every comparison references the baseline
        arms = ["baseline"] + [a for a in arms if a != "baseline"]
    if "cap" in wls:  # G5 sizes every store from the baseline cap trace
        wls = [w for w in wls if w != "cap"] + ["cap"]
    results: dict[str, dict[str, Any]] = {}
    if args.under_test_path:
        args.under_test_overlay = str((run_dir / "under-test-overlay").resolve())
    # Before any stage: what the run is about to measure.
    prov = record_provenance(args, run_dir)
    print(f"[gate] provenance {json.dumps(prov, default=str)}", flush=True)
    watch = ProvenanceWatch(args, model_dir)
    cuda = cuda_env()
    cuda_bad = cuda_problems(cuda)
    print(f"[gate] cuda env {json.dumps(cuda)} problems {cuda_bad}", flush=True)
    pool_checks: list[dict[str, Any]] = []
    uncapped = [w for w in wls if w != "cap"]
    if any(arm_kind(a) in _STORES for a in arms) and uncapped:
        if committed.exists():
            spec = json.loads(committed.read_text())
            for w in uncapped:
                pool_checks.append(check_pool(args, estimate_unique(args, spec, w)))
        else:
            print(f"[pool] no committed {committed}: estimate skipped", flush=True)

    def stage(arm: str, wl: str, out: Path, **kw) -> dict[str, Any]:
        watch.check(f"{arm_label(arm)}/{out.name}")  # M10: before every stage
        return run_stage(args, gate, arm, wl, out, **kw)

    measured_u: dict[str, dict[str, Any]] = {}

    for arm in arms:
        kind = arm_kind(arm)
        label = arm_label(arm)
        for wl in wls:
            if wl != "gsm8k" and kind in ("plain", "selftest"):
                continue
            if wl != "cap":
                if kind in _STORES and wl not in measured_u:
                    # Measured on the baseline's trace of the same workload.
                    base_res = results.get("baseline", {}).get(wl, {})
                    probe = run_dir / "baseline" / wl / "probe"
                    if "error" not in base_res and (probe / "spec.json").exists():
                        u = oracle.unique_bytes(
                            json.loads((probe / "spec.json").read_text()),
                            oracle.load_requests(probe / "requests.jsonl"),
                        )
                        measured_u[wl] = check_pool(
                            args, {"workload": wl, "unique_bytes": u, "how": "baseline trace"}
                        )
                        pool_checks.append(measured_u[wl])
                results.setdefault(label, {})[wl] = stage(arm, wl, run_dir / label / wl)
                continue
            if kind == "baseline":
                spec_file = next(
                    (
                        run_dir / label / w / "probe" / "spec.json"
                        for w in wls
                        if (run_dir / label / w / "probe" / "spec.json").exists()
                    ),
                    None,
                )
                args.cap_plan = plan_cap_convs(
                    args, json.loads(spec_file.read_text()) if spec_file else None
                )
                print(f"[cap] plan {args.cap_plan}", flush=True)
                results.setdefault(label, {})[wl] = stage(arm, wl, run_dir / label / wl)
                continue
            base_cap = results.get("baseline", {}).get("cap", {})
            u = base_cap.get("unique_bytes")
            if not u:
                results.setdefault(label, {})[wl] = {
                    "arm": arm,
                    "workload": wl,
                    "error": "no baseline cap trace in this run to size the store",
                }
                continue
            for f in fractions:
                name = cap_label(f)
                out = run_dir / label / name
                r = stage(arm, name, out, capacity=int(f * u), ref_local=base_cap.get("ref_local"))
                r["unique_bytes"] = u
                r["capacity_fraction"] = f
                if gate_dir(out).exists():
                    (gate_dir(out) / "result.json").write_text(json.dumps(r, indent=1))
                results.setdefault(label, {})[name] = r

    watch.check("verdicts")  # ... and after the last one (M10)

    def strip(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if k not in ("digest_self", "ref_local")}

    problems = official_problems(args, arms, wls, gate_file, committed)
    contaminated = contamination(results)
    problems += [
        f"contaminated: {c['arm']} replay/{c['pass']} p99 TTFT {c['ttft_p99']:.3g}s > "
        + (
            f"{CONTAMINATION_P99:g}x baseline {c['baseline_p99']:.3g}s"
            if "baseline_p99" in c
            else f"{CONTAMINATION_P99:g}x the arms' median {c['arms_median_p99']:.3g}s"
        )
        for c in contaminated
    ]
    problems += provenance_problems(prov)
    problems += watch.problems()
    problems += cuda_bad
    scan = scan_record(args)
    problems += [f"static scan hit not on the allowlist: {h['id']}" for h in scan["flagged"]]
    iso_rec = None
    if any(arm_kind(a) == "under_test" for a in arms):
        label = next(arm_label(a) for a in arms if arm_kind(a) == "under_test")
        iso_rec = isolation(args, run_dir / label / "<workload>")
        iso_rec |= {
            "available": isolation_available(),
            "command_prefix": isolation_prefix(iso_rec),
            # What each stage of the arm ran with (rw: that stage's dir only).
            "stages": dict(sorted(_ISOLATED.items())),
            "residual": ISOLATION_RESIDUAL,
        }
        if not iso_rec["available"]:
            problems.append("arm under test not isolated (isolate.py needs Linux and root)")
    dirs = {arm_label(a): expected_pegaflow(args, arm_kind(a)) for a in arms}
    connectors = {
        arm_label(a): (expected_connector(a), connector_root(a, dirs[arm_label(a)])) for a in arms
    }
    v = verdicts(results, run_dir, committed, gate, dirs, connectors)
    floor = None
    if "cap" in wls:
        # Not a verdict: a store smaller than this measures vLLM's GPU cache,
        # so the run cannot be official (G5 is then a flow + digest smoke).
        floor = capacity_floor(results.get("baseline", {}).get("cap"))
        problems += g5_problems(v, floor, getattr(args, "cap_plan", None))
    summary = {
        "tag": args.tag,
        "rev": git_rev(),
        "under_test_rev": prov.get("under_test", {}).get("rev"),
        "provenance": prov,
        "provenance_watch": watch.record(),
        "pool_checks": pool_checks,
        "contamination": contaminated,
        "isolation": iso_rec,
        "cuda_env": cuda,
        "scan": scan,
        "port_leases": release_leases(),
        "args": vars(args),
        "vllm_plugins": {arm_label(a): vllm_plugins(a) for a in arms},
        "gate": gate,
        "official": not problems,
        "held_out": args.replay_skip > 0,
        "replay_skip": args.replay_skip,
        "not_official_because": problems,
        "capacity_floor": floor,
        "cap_plan": getattr(args, "cap_plan", None),
        "results": {a: {w: strip(r) for w, r in per.items()} for a, per in results.items()},
        "verdicts": v,
        "hit_report": hit_report(v),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(
        json.dumps(
            {"official": not problems, "problems": problems, "verdicts": v}, indent=1, default=str
        )
    )
    print(
        f"[gate] hit rates (reported, not gated) {json.dumps(summary['hit_report'], default=str)}"
    )
    print(f"run record: {run_dir}")
    required = [x for x in v.values() if x.get("required", True)]
    return 0 if not problems and all(x["pass"] for x in required) else 1


if __name__ == "__main__":
    sys.exit(main())
