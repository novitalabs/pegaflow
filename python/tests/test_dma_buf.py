"""Real FD handoff across independent processes; no CUDA/native dependency."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pegaflow.dma_buf import DmaBufExports, receive_dma_buf

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux DMA-BUF FD transport")


def test_fd_handoff_and_registration_cleanup(tmp_path):
    payload = tmp_path / "allocation"
    payload.write_bytes(b"owned allocation")
    with DmaBufExports() as exports:
        fd = os.open(payload, os.O_RDONLY)
        address, token = exports.add(fd)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; from pegaflow.dma_buf import receive_dma_buf; "
                "fd=receive_dma_buf(sys.argv[1], bytes.fromhex(sys.argv[2])); "
                "assert not os.get_inheritable(fd); "
                "print(os.read(fd, 64).decode()); os.close(fd)",
                address,
                token.hex(),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        assert result.stdout.strip() == "owned allocation"
        with pytest.raises(RuntimeError, match="exactly one descriptor"):
            receive_dma_buf(address, token)
        unclaimed = os.open(payload, os.O_RDONLY)
        exports.add(unclaimed)
    assert not Path(address).exists()
    with pytest.raises(OSError):
        os.fstat(unclaimed)
