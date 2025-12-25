#!/usr/bin/env python3
"""
Wrapper script to launch pegaflow-server binary from the installed package.
"""
import os
import subprocess
import sys
from pathlib import Path


def get_server_binary():
    """Locate the pegaflow-server-py binary in the installed package."""
    # The binary is in the same directory as this Python module
    module_dir = Path(__file__).parent
    binary_path = module_dir / "pegaflow-server-py"

    if binary_path.exists() and binary_path.is_file():
        return str(binary_path)

    # Fallback: try to find in PATH
    return "pegaflow-server-py"


def main():
    """Launch pegaflow-server with command-line arguments."""
    server_binary = get_server_binary()

    try:
        # Pass through all command-line arguments
        result = subprocess.run(
            [server_binary] + sys.argv[1:],
            check=False,
        )
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: pegaflow-server-py binary not found at {server_binary}", file=sys.stderr)
        print("Please ensure pegaflow is properly installed.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
