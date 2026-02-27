"""Utility to locate Rust binaries bundled with pegaflow."""

import shutil
from pathlib import Path

# pegaflow/ directory (where this file lives)
_MODULE_DIR = Path(__file__).parent
# Repo root: pegaflow/python/pegaflow/../../ -> pegaflow/
_REPO_ROOT = _MODULE_DIR.parent.parent


def find_binary(name: str) -> str:
    """Locate a pegaflow binary by name.

    Search order:
    1. Installed package directory (pip install from wheel)
    2. Cargo target/release/ (dev mode / editable install)
    3. Cargo target/debug/
    4. PATH fallback
    """
    # 1. Wheel install: binary next to this module
    path = _MODULE_DIR / name
    if path.is_file():
        return str(path)

    # 2. Dev mode: cargo target/release/
    path = _REPO_ROOT / "target" / "release" / name
    if path.is_file():
        return str(path)

    # 3. Dev mode: cargo target/debug/
    path = _REPO_ROOT / "target" / "debug" / name
    if path.is_file():
        return str(path)

    # 4. Fallback: PATH
    found = shutil.which(name)
    if found:
        return found

    return name
