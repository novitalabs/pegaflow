"""Run a command with a directory tree hidden (M7a, GATES.md "Boundary").

    python -I -S -B isolate.py --hide ROOT [--rw DIR]... [--ro DIR]... [--wait-fd FD] -- CMD...

In a private mount namespace (``os.unshare(CLONE_NEWNS)``, propagation
private, so nothing is visible on the host), ``ROOT`` is covered by an empty
tmpfs and only the ``--rw`` / ``--ro`` dirs (each under ``ROOT``) are bound
back at their own paths, ``--ro`` read-only; block devices and
``/dev/{mem,kmem,port}`` are covered by ``/dev/null``; then ``CMD`` is
exec'd. Only the mount namespace is private. The
helper execs in place (same pid, process group and session), so the gate's
teardown of the group reaches every process, and the namespace ends with
its last process. Binds go through ``/proc/self/fd/N`` of dirs opened (in the
new namespace) before the tmpfs is mounted: the kernel follows the fd to the original dir, and no
path is canonicalized after ROOT is covered. The dir fds are not inherited
by ``CMD``. ``DROP_CAPS`` leave the bounding set before the exec.
``--wait-fd`` (S11): right after the unshare, block until the read end
``FD`` of a pipe reaches EOF, i.e. until the gate (holding the write end) has
opened the new namespace, so it is recorded before ``CMD`` can run.

Standalone: stdlib only, run with ``-I -S`` (no ``PYTHONPATH``, no site, so the
arm's snapshot dir on it cannot shadow a module of this helper). Needs
Linux and root (CAP_SYS_ADMIN).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
PR_CAPBSET_DROP = 24
# Dropped from the bounding set before exec, so CMD (uid 0) never holds
# them: without CAP_SYS_ADMIN it cannot unmount the tmpfs or mount anything;
# without CAP_SYS_PTRACE its permitted set is not a superset of a full-cap
# process's, so ``/proc/<pid>/{root,cwd,fd,mem,environ}`` of the gate (or any
# other process outside the arm) is EACCES: the host mount namespace stays
# out of reach.
DROP_CAPS = {
    "CAP_SYS_MODULE": 16,
    "CAP_SYS_RAWIO": 17,
    "CAP_SYS_CHROOT": 18,
    "CAP_SYS_PTRACE": 19,
    "CAP_SYS_ADMIN": 21,
    "CAP_SYS_BOOT": 22,
    "CAP_MKNOD": 27,
    "CAP_BPF": 39,
    "CAP_CHECKPOINT_RESTORE": 40,
}
# Raw disk / memory: read-masked with /dev/null (a hidden file's bytes are
# on the disk). GPU (/dev/nvidia*), /dev/shm and the host's pid / IPC /
# network namespaces stay shared: legacy CUDA IPC handles (vLLM workers ->
# pegaflow-server) and the loopback ports the gate reads need them.
_MASK_DIRS = ("/dev", "/dev/disk", "/dev/mapper", "/dev/block")
_MASK_CHAR = ("/dev/mem", "/dev/kmem", "/dev/port")

_libc = ctypes.CDLL(None, use_errno=True)
_libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]  # fmt: skip
_libc.mount.argtypes = [ctypes.c_char_p] * 3 + [ctypes.c_ulong, ctypes.c_char_p]


def _mount(source: str, target: str, fstype: str | None, flags: int, data: str | None = None):
    enc = [x.encode() if x is not None else None for x in (source, target, fstype, data)]
    if _libc.mount(enc[0], enc[1], enc[2], flags, enc[3]) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"mount({source}, {target}, {fstype}, {flags:#x}): {os.strerror(e)}")


def _mask_devices() -> list[str]:
    import stat

    masked = []
    seen = set()
    for d in _MASK_DIRS:
        if not os.path.isdir(d):
            continue
        for dirpath, _, names in os.walk(d):
            for n in names:
                f = os.path.join(dirpath, n)
                try:
                    st = os.stat(f)
                except OSError:
                    continue
                if (stat.S_ISBLK(st.st_mode) or f in _MASK_CHAR) and f not in seen:
                    seen.add(f)
                    # Bind over the path (a symlink resolves to its target).
                    _mount("/dev/null", f, None, MS_BIND)
                    masked.append(f)
    return masked


def _inside(p: str, root: str) -> bool:
    return p != root and p.startswith(root.rstrip("/") + "/")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: isolate.py --hide ROOT [--rw DIR]... [--ro DIR]... -- CMD...")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hide", required=True)
    ap.add_argument("--rw", action="append", default=[])
    ap.add_argument("--ro", action="append", default=[])
    ap.add_argument("--wait-fd", type=int)
    args = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1 :]
    if not cmd:
        raise SystemExit("isolate.py: no command")
    hide = os.path.realpath(args.hide)
    keep = [(os.path.realpath(d), False) for d in args.rw] + [
        (os.path.realpath(d), True) for d in args.ro
    ]
    for d, _ in keep:
        if not _inside(d, hide):
            raise SystemExit(f"isolate.py: {d} is not under {hide}")
    os.unshare(os.CLONE_NEWNS)
    if args.wait_fd is not None:
        os.read(args.wait_fd, 1)  # EOF once the gate holds this namespace
        os.close(args.wait_fd)
    _mount("none", "/", None, MS_REC | MS_PRIVATE)
    # Opened in the new namespace (a bind source must be one of its mounts)
    # before ROOT is covered; O_CLOEXEC (Python's default): CMD never gets an
    # fd into the hidden tree.
    fds = [(os.open(d, os.O_RDONLY | os.O_DIRECTORY), d, ro) for d, ro in keep]
    _mask_devices()
    _mount("rsi-hidden", hide, "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755,size=1m")
    for fd, d, ro in fds:
        os.makedirs(d, exist_ok=True)  # on the tmpfs
        _mount(f"/proc/self/fd/{fd}", d, None, MS_BIND)
        if ro:
            _mount("none", d, None, MS_REMOUNT | MS_BIND | MS_RDONLY)
        os.close(fd)
    for name, cap in DROP_CAPS.items():
        if _libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0:
            e = ctypes.get_errno()
            if e != 22:  # EINVAL: the kernel does not know this cap
                raise OSError(e, f"PR_CAPBSET_DROP {name}: {os.strerror(e)}")
    os.execvpe(cmd[0], cmd, os.environ)
    return 127  # not reached


if __name__ == "__main__":
    sys.exit(main())
