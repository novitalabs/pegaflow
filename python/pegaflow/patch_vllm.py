"""Simple script to apply vLLM patch.

This module provides functionality to apply vLLM patch files.
It automatically finds the latest vllm-*.patch file and applies it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Get the package directory
PACKAGE_DIR = Path(__file__).parent
PATCHES_DIR = PACKAGE_DIR / "patches"


def get_vllm_path(vllm_path: str | None = None) -> Path:
    """Get the installation path of vLLM package.

    Args:
        vllm_path: Optional path to vLLM installation. If not provided,
                   auto-detects from installed vllm package.

    Returns:
        Path to vLLM package directory.

    Raises:
        SystemExit: If vLLM is not found.
    """
    if vllm_path:
        path = Path(vllm_path)
        if not path.exists():
            print(f"Error: vLLM path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        return path

    try:
        import vllm

        # Get the file path of vllm module
        vllm_file = Path(vllm.__file__)
        # Return the parent directory (package root)
        return vllm_file.parent
    except ImportError:
        print("Error: vLLM is not installed or cannot be found.", file=sys.stderr)
        print("  Please install vLLM first: pip install vllm", file=sys.stderr)
        print("  Or specify vLLM path: python -m pegaflow.patch_vllm <vllm_path>", file=sys.stderr)
        sys.exit(1)


def find_latest_patch() -> Path:
    """Find the latest vllm-*.patch file in patches directory.

    Returns:
        Path to the latest patch file.

    Raises:
        SystemExit: If no patch file is found.
    """
    patch_files = sorted(
        PATCHES_DIR.glob("vllm-*.patch"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not patch_files:
        print(f"Error: No vLLM patch file found in {PATCHES_DIR}", file=sys.stderr)
        sys.exit(1)

    return patch_files[0]


def apply_patch(vllm_path: Path, patch_file: Path) -> None:
    """Apply the patch to vLLM installation.

    Args:
        vllm_path: Path to vLLM package directory.
        patch_file: Path to the patch file.

    Raises:
        SystemExit: If patch application fails.
    """
    if not patch_file.exists():
        print(f"Error: Patch file not found: {patch_file}", file=sys.stderr)
        sys.exit(1)

    if not vllm_path.exists():
        print(f"Error: vLLM path does not exist: {vllm_path}", file=sys.stderr)
        sys.exit(1)

    # Try git apply first (preferred method)
    git_cmd = shutil.which("git")
    if git_cmd is not None:
        git_dir = vllm_path / ".git"
        parent_git_dir = vllm_path.parent / ".git"
        if git_dir.exists() or parent_git_dir.exists():
            try:
                cmd = [git_cmd, "apply", "--whitespace=fix"]
                with patch_file.open("rb") as f:
                    result = subprocess.run(
                        cmd,
                        stdin=f,
                        cwd=str(vllm_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                if result.returncode == 0:
                    print("Patch applied successfully using git apply!")
                    return
            except Exception:
                # git apply failed, fall through to patch command
                pass

    # Fall back to patch command
    patch_cmd = shutil.which("patch")
    if patch_cmd is None:
        print("Error: 'patch' command not found. Please install it first.", file=sys.stderr)
        print("  On Ubuntu/Debian: apt-get install patch", file=sys.stderr)
        sys.exit(1)

    # Apply patch using patch command
    # -p2: strip 'a/vllm/' or 'b/vllm/' from paths in patch file
    # -f: force mode, skip interactive prompts
    cmd = [patch_cmd, "-p2", "-f", "-d", str(vllm_path)]

    with patch_file.open("rb") as f:
        result = subprocess.run(
            cmd,
            stdin=f,
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        print(f"Error applying patch:\n{result.stderr}", file=sys.stderr)
        if result.stdout:
            print(f"Output:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)

    print("Patch applied successfully using patch command!")


def main() -> int:
    """CLI entry point for applying vLLM patch.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Get vLLM path from argument or auto-detect
    vllm_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    vllm_path = get_vllm_path(vllm_path_arg)

    # Find latest patch file
    patch_file = find_latest_patch()

    print(f"Applying patch: {patch_file}")
    print(f"To vLLM path: {vllm_path}")

    # Apply patch
    apply_patch(vllm_path, patch_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
