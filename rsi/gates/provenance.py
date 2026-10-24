"""Provenance of the code a gate run measures (G0.provenance, official rule).

- ``Trees.tree``: a tree's rev, *verified*. The rev is HEAD of a git toplevel
  or the ``.rsi-rev`` file's (that rev fetched blob-less from ``remote``).
  The rev's ``git ls-tree -r`` under each subpath is the whole allowed
  content: every file on disk there must be a tracked regular blob with the
  same blob id, or the one expected native module (``native_module``, its
  sha256 recorded: a ``.so`` cannot be tied to a rev). Anything else rejects
  the tree: a symlink (file or dir, tracked or not), an untracked or ignored
  file, any other ``.so`` / ``.pyd`` / ``.dylib``, a ``.pyc`` outside
  ``__pycache__`` (importable without its source) and, unless
  ``pycache_ok``, any ``.pyc`` at all. ``pycache_ok`` is only for trees
  every consumer imports with ``PYTHONPYCACHEPREFIX`` set, which makes
  Python ignore in-tree ``__pycache__``.
- ``stage``: the connector-under-test arm imports a *snapshot*: the rev's
  tracked blobs of ``python/pegaflow`` plus the expected native module,
  copied into ``<dest>/pegaflow`` and verified again after the copy (blob
  ids, sha256), so editing the source tree mid-run changes nothing the run
  imports. ``dest`` is the only dir put on that arm's ``PYTHONPATH``: the
  tree cannot shadow the probe, its worker extension, vLLM, torch or
  ``sitecustomize``.
- ``scan``: static scan of the snapshot for instrument / run-artifact
  access (M7b); a hit makes the run non-official until a reviewer clears it.
- ``module_problems``: the module files a probe recorded in its spec dumps
  (``rsi_probe.module_files``) must resolve under the gate tree (probe,
  worker extension), the shared venv (vLLM, torch, interpreter prefix) and
  the arm's expected ``pegaflow`` dir.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from contextlib import suppress
from pathlib import Path
from typing import Any

REMOTE = "https://github.com/gentlecold/pegaflow.git"
PKG = "python/pegaflow"
_REGULAR = ("100644", "100755")


def native_module(pkg: str = PKG) -> str:
    """The one native module a ``pegaflow`` tree may carry untracked."""
    return f"{pkg}/pegaflow{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    p = Path(path).resolve()
    return p == root or root in p.parents


def venv_purelib() -> Path:
    return Path(sysconfig.get_path("purelib")).resolve()


def gate_pegaflow_dir() -> Path | None:
    """Where ``pegaflow`` resolves for the gate's interpreter (venv .pth)."""
    import importlib.util

    spec = importlib.util.find_spec("pegaflow")
    return Path(spec.origin).resolve().parent if spec and spec.origin else None


def module_problems(
    files: dict[str, Any], repo: Path, pegaflow_dir: Path | None, where: str
) -> list[str]:
    out = []
    for k in ("rsi.gates.rsi_probe", "worker_extension"):
        if k in files and not under(files[k], repo.resolve()):
            out.append(f"{where}: {k} = {files[k]} not under the gate tree {repo}")
    if "rsi.gates.rsi_probe" not in files:
        out.append(f"{where}: probe module file not recorded")
    purelib = venv_purelib()
    for k in ("vllm", "torch"):
        if not under(files.get(k), purelib):
            out.append(f"{where}: {k} = {files.get(k)} not in the shared venv {purelib}")
    if Path(files.get("prefix") or "").resolve() != Path(sys.prefix).resolve():
        out.append(f"{where}: interpreter prefix {files.get('prefix')} != {sys.prefix}")
    for k in ("sitecustomize", "usercustomize"):
        f = files.get(k)
        if f and not (under(f, purelib) or under(f, Path(sys.base_prefix).resolve())):
            out.append(f"{where}: {k} = {f} outside the venv / base interpreter")
    for k in ("pegaflow", "pegaflow.pegaflow"):
        f = files.get(k)
        if f and (pegaflow_dir is None or not under(f, pegaflow_dir)):
            out.append(f"{where}: {k} = {f} not under the arm's pegaflow {pegaflow_dir}")
    return out


def _git(*a: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *a], cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=300
    )


def _blob_id(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _walk(root: Path, sub: str) -> tuple[list[str], list[str]]:
    """(regular files, symlinks) under ``root/sub``, relative to ``root``.
    Every directory is entered (``__pycache__`` included); a symlinked one
    is listed as a symlink, never followed."""
    base = root / sub
    if base.is_symlink():
        return [], [sub]
    if not base.is_dir():
        return ([sub] if base.is_file() else []), []
    files, links = [], []
    for dirpath, dirs, names in os.walk(base, followlinks=False):
        for n in [*dirs, *names]:
            p = Path(dirpath) / n
            rel = str(p.relative_to(root))
            if p.is_symlink():
                links.append(rel)
            elif n in names:
                files.append(rel)
    return sorted(files), sorted(links)


def _ls_tree(rev: str, cwd: Path) -> dict[str, tuple[str, str]]:
    """``path -> (mode, blob id)`` of every blob of ``rev`` (git dir/tree ``cwd``)."""
    out = {}
    for line in _git("ls-tree", "-r", rev, cwd=cwd).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, oid = meta.split()
        if kind == "blob":
            out[path] = (mode, oid)
    return out


def _inside(p: str, subpaths: tuple[str, ...]) -> bool:
    return any(p == s or p.startswith(s.rstrip("/") + "/") for s in subpaths)


def check(
    root: Path,
    subpaths: tuple[str, ...],
    listing: dict[str, tuple[str, str]],
    native: str | None,
    pycache_ok: bool = False,
) -> dict[str, Any]:
    """Files under ``root/subpaths`` vs ``listing`` (module doc)."""
    want = {p: m for p, m in listing.items() if _inside(p, subpaths)}
    files: list[str] = []
    links: list[str] = []
    for sub in subpaths:
        f, link = _walk(root, sub)
        files += f
        links += link
    rejected = [f"symlink {p}" for p in links]
    rejected += [f"tracked non-regular {p} (mode {m[0]})" for p, m in want.items() if m[0] not in _REGULAR]  # fmt: skip
    missing = sorted(set(want) - set(files))
    differ = sorted(f for f in set(files) & set(want) if _blob_id(root / f) != want[f][1])
    extra, cached = [], 0
    for f in sorted(set(files) - set(want)):
        p = Path(f)
        if f == native:
            continue
        if pycache_ok and p.suffix == ".pyc" and p.parent.name == "__pycache__":
            cached += 1  # never read: every consumer runs with PYTHONPYCACHEPREFIX
            continue
        extra.append(f)
    rec: dict[str, Any] = {
        "native_sha256": {native: sha256(root / native)} if native in files else {},
        "pycache_pyc_ignored": cached,
        "verified": False,
    }
    if not want:
        rec["reason"] = f"rev tracks nothing under {list(subpaths)}"
    elif rejected or missing or extra or differ:
        rec["reason"] = "tree differs from rev"
        found = {"rejected": rejected, "missing": missing, "extra": extra, "differ": differ}
        rec |= {k: v[:20] for k, v in found.items() if v}
    else:
        rec["verified"] = True
        rec["files"] = len(want)
    return rec


class Trees:
    """Verified revs, one blob-less fetch per rev (git dir under ``work``)."""

    def __init__(self, work: Path, remote: str = REMOTE):
        self.work, self.remote = work, remote
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _fetched(self, rev: str) -> dict[str, tuple[str, str]] | str:
        gd = self.work / "git"
        try:
            if not gd.exists():
                _git("init", "-q", "--bare", str(gd))
            with_tree = subprocess.run(
                ["git", "-C", str(gd), "cat-file", "-e", f"{rev}^{{tree}}"],
                capture_output=True,
                timeout=60,
            )
            if with_tree.returncode:
                _git("-C", str(gd), "fetch", "-q", "--depth=1", "--filter=blob:none", self.remote, rev)  # fmt: skip
            return _ls_tree(rev, gd)
        except (subprocess.SubprocessError, OSError) as e:
            return f"cannot fetch {rev} from {self.remote}: {type(e).__name__}"

    def listing(self, root: Path) -> tuple[dict[str, Any], dict[str, tuple[str, str]] | None]:
        """(``how`` / ``rev`` / ``reason``, the rev's blobs or None) of ``root``."""
        try:
            top = Path(_git("rev-parse", "--show-toplevel", cwd=root).strip()).resolve()
        except (subprocess.SubprocessError, OSError):
            top = None
        if top == root:
            rev = _git("rev-parse", "HEAD", cwd=root).strip()
            return {"how": "git", "rev": rev}, _ls_tree(rev, root)
        rev_file = root / ".rsi-rev"
        if rev_file.is_symlink() or not rev_file.is_file():
            return {"how": "unknown", "rev": "unknown", "reason": "not a git toplevel, no .rsi-rev"}, None  # fmt: skip
        rev = rev_file.read_text().strip()
        got = self._fetched(rev)
        if isinstance(got, str):
            return {"how": "rsi-rev", "rev": rev, "reason": got}, None
        return {"how": "rsi-rev", "rev": rev}, got

    def tree(
        self,
        root: Path,
        subpaths: tuple[str, ...],
        native: str | None = None,
        pycache_ok: bool = False,
    ) -> dict[str, Any]:
        root = root.resolve()
        key = (root, subpaths, native, pycache_ok)
        if key not in self._cache:
            head, listing = self.listing(root)
            rec: dict[str, Any] = {"root": str(root), "subpaths": list(subpaths)} | head
            if listing is None:
                rec["verified"] = False
            else:
                rec |= check(root, subpaths, listing, native, pycache_ok)
            self._cache[key] = rec
        return self._cache[key]

    def cleanup(self) -> None:
        if (self.work / "git").exists():
            shutil.rmtree(self.work / "git")
        if self.work.is_dir() and not any(self.work.iterdir()):
            self.work.rmdir()


def fingerprint(root: Path, subpaths: tuple[str, ...], pycache_ok: bool = False) -> dict[str, str]:
    """M10: ``path -> blob id`` of every file under ``root/subpaths`` (a
    symlink as ``symlink -> <target>``; a subpath may be one file). With
    ``pycache_ok``, in-tree ``__pycache__/*.pyc`` are skipped: no process
    reads them (every one runs with PYTHONPYCACHEPREFIX)."""
    out: dict[str, str] = {}
    for sub in subpaths:
        files, links = _walk(root, sub)
        for f in files:
            p = Path(f)
            if pycache_ok and p.suffix == ".pyc" and p.parent.name == "__pycache__":
                continue
            with suppress(OSError):
                out[f] = _blob_id(root / f)
        for link in links:
            with suppress(OSError):
                out[link] = f"symlink -> {os.readlink(root / link)}"
    return out


def fingerprint_diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """What changed between two ``fingerprint``s, by path."""
    return (
        [f"changed {p}" for p in sorted(set(before) & set(after)) if before[p] != after[p]]
        + [f"added {p}" for p in sorted(set(after) - set(before))]
        + [f"removed {p}" for p in sorted(set(before) - set(after))]
    )


def tree_root(pegaflow_dir: Path) -> Path:
    """``<tree>/python/pegaflow`` -> ``<tree>``."""
    return pegaflow_dir.resolve().parents[1]


def stage(under_test: Path, dest: Path, trees: Trees) -> dict[str, Any]:
    """Snapshot ``<under_test>/pegaflow`` (``under_test`` = a tree's
    ``python/``) into ``<dest>/pegaflow``. The source must verify with no
    ``.pyc`` at all; the rev's blobs and the native module are copied and
    the copy is verified again. ``verified`` is false if either check failed
    (the snapshot is still written: the run goes on, non-official)."""
    pkg = under_test / "pegaflow"
    if pkg.is_symlink() or not (pkg / "__init__.py").is_file():
        raise ValueError(f"{under_test}: no pegaflow package dir (or it is a symlink)")
    root = tree_root(pkg)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "pegaflow").mkdir(parents=True)
    src = trees.tree(root, (PKG,), native_module())
    rec: dict[str, Any] = {
        "source": src,
        "rev": src.get("rev"),
        "pegaflow_dir": str((dest / "pegaflow").resolve()),
        "verified": False,
    }
    head, listing = trees.listing(root)
    if listing is None:
        rec["reason"] = f"source: {src.get('reason')}"
        return rec
    # Paths relative to ``python/`` (the snapshot's root is ``dest``).
    blobs = {
        str(Path(p).relative_to("python")): m for p, m in listing.items() if _inside(p, (PKG,))
    }
    for p, (mode, _) in blobs.items():
        s = root / "python" / p
        if mode in _REGULAR and s.is_file() and not s.is_symlink():
            (dest / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, dest / p)
    native = native_module("pegaflow")
    if (under_test / native).is_file() and not (under_test / native).is_symlink():
        shutil.copyfile(under_test / native, dest / native)
    copy = check(dest, ("pegaflow",), blobs, native)
    rec |= {"copy": copy, "native": native, "native_sha256": copy["native_sha256"].get(native)}
    if not src["verified"]:
        rec["reason"] = f"source: {src.get('reason')}"
    elif not copy["verified"]:
        rec["reason"] = f"snapshot: {copy.get('reason')}"
    elif copy["native_sha256"].get(native) != src["native_sha256"].get(native_module()):
        rec["reason"] = "snapshot native module differs from the source's"
    else:
        rec["verified"] = True
    return rec


# M7(b): instrument / run-artifact access a connector has no reason for
# (GATES.md, "Boundary"). A hit is not a verdict: the run is non-official
# unless the hit is on ``SCAN_ALLOWLIST``. ``rsi`` matches the gate package
# as code (``import rsi``, ``from rsi``, ``"rsi.``, ``rsi.gates``,
# ``rsi/gates``), not a path that merely contains ``rsi`` (``docs/rsi/``,
# a ``*-rsi/`` build dir).
SCAN_PATTERNS = {
    "rsi": r"\b(import|from)\s+rsi\b|[\"']rsi[.\"']|\brsi[./]gates\b",
    "rsi_probe": r"rsi_probe",
    "RsiProbe": r"RsiProbe",
    "RsiWorkerExt": r"RsiWorkerExt",
    "rsi_poison": r"rsi_poison",
    "rsi-runs": r"rsi-runs",
    "probe_dir": r"probe_dir",
    "digest-": r"digest-",
    "-outputs.json": r"-outputs\.json",
    "-tokens.json": r"-tokens\.json",
    "requests.jsonl": r"requests\.jsonl",
    "sys.modules write": r"sys\.modules\s*\[[^\]]*\]\s*=|sys\.modules\.(setdefault|update|pop|popitem|clear|__setitem__)\b|\bdel\s+sys\.modules",
    "gc introspection": r"\bgc\.get_(objects|referrers|referents)\b",
    "trace / audit hook": r"\bsys\.(settrace|setprofile|addaudithook|_getframe)\b|\bthreading\.(settrace|setprofile)\b",
    "import hook": r"\bsys\.(meta_path|path_hooks)\b|\bimportlib\.reload\b|\bbuiltins\.__import__\b",
}  # fmt: skip
_SCAN = {k: re.compile(v) for k, v in SCAN_PATTERNS.items()}


def _strings(data: bytes, n: int = 4) -> list[str]:
    return [m.decode() for m in re.findall(rb"[\x20-\x7e]{%d,}" % n, data)]


def scan(pkg_dir: Path) -> list[dict[str, Any]]:
    """``SCAN_PATTERNS`` hits in a snapshot: every line of every text file,
    and the printable strings of binary files (the native module). Hit id:
    ``<file>:<line>:<pattern>``, ``<file>:strings:<pattern>`` for binaries."""
    hits = []
    for f in sorted(p for p in pkg_dir.rglob("*") if p.is_file()):
        rel = str(f.relative_to(pkg_dir.parent))
        data = f.read_bytes()
        binary = f.suffix in (".so", ".pyd", ".dylib") or b"\0" in data
        if not binary:
            try:
                lines = list(enumerate(data.decode().splitlines(), 1))
            except UnicodeDecodeError:
                binary = True
        if binary:
            lines = [(0, s) for s in _strings(data)]
        counts: dict[str, int] = {}
        for i, line in lines:
            for name, rx in _SCAN.items():
                if not rx.search(line):
                    continue
                if binary:
                    counts[name] = counts.get(name, 0) + 1
                    if counts[name] == 1:
                        hits.append({"id": f"{rel}:strings:{name}", "text": line[:200]})
                else:
                    hits.append({"id": f"{rel}:{i}:{name}", "text": line.strip()[:200]})
    return hits


# Reviewed scan hits (path glob relative to the snapshot root, pattern name,
# reason). Adding an entry is a gate change: review agent sign-off, as for
# any GATES.md change. Allowlisted hits are recorded in ``summary.scan``.
SCAN_ALLOWLIST: list[dict[str, str]] = []


def allowlisted(
    hit: dict[str, Any], allow: list[dict[str, str]] | None = None
) -> dict[str, str] | None:
    """The ``SCAN_ALLOWLIST`` entry (or ``allow``'s) covering ``hit``."""
    import fnmatch

    path, _, name = hit["id"].split(":", 2)  # <file>:<line|strings>:<pattern>
    for a in SCAN_ALLOWLIST if allow is None else allow:
        if fnmatch.fnmatchcase(path, a["path"]) and a["pattern"] == name:
            return a
    return None
