"""Utility to locate Rust binaries bundled with pegaflow."""

import os
import shutil
import site
import sysconfig
from pathlib import Path

# pegaflow/ directory (where this file lives)
_MODULE_DIR = Path(__file__).parent
# Repo root: pegaflow/python/pegaflow/../../ -> pegaflow/
_REPO_ROOT = _MODULE_DIR.parent.parent


def find_binary(name: str) -> str:
    """Locate a pegaflow binary by name.

    Search order:
    1. Cargo target/release/ (dev mode — always freshest build)
    2. Cargo target/debug/
    3. Installed package directory (pip install from wheel)
    4. PATH fallback
    """
    # 1. Dev mode: cargo target/release/
    path = _REPO_ROOT / "target" / "release" / name
    if path.is_file():
        return str(path)

    # 2. Dev mode: cargo target/debug/
    path = _REPO_ROOT / "target" / "debug" / name
    if path.is_file():
        return str(path)

    # 3. Wheel install: binary next to this module
    path = _MODULE_DIR / name
    if path.is_file():
        return str(path)

    # 4. Fallback: PATH
    found = shutil.which(name)
    if found:
        return found

    return name


def _prepend_env_paths(env: dict[str, str], key: str, paths: list[str]) -> None:
    paths = [path for path in paths if path]
    if not paths:
        return
    current = env.get(key)
    if current:
        paths.append(current)
    env[key] = os.pathsep.join(paths)


def binary_env() -> dict[str, str]:
    """Return an environment that can load binaries linked to this Python."""
    env = os.environ.copy()
    libdir = sysconfig.get_config_var("LIBDIR")
    _prepend_env_paths(env, "LD_LIBRARY_PATH", [libdir] if libdir else [])
    _prepend_env_paths(
        env,
        "PYTHONPATH",
        [str(_MODULE_DIR.parent), *site.getsitepackages()],
    )
    return env
