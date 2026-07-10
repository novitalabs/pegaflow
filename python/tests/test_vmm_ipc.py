"""Unit tests for the VMM-IPC layer (sleep-mode KV sharing).

CPU-only: the FD handoff runs over a real abstract unix socket with plain
file descriptors, and the CUDA driver is replaced by a call-recording fake.
No torch, vLLM, GPU, or server required (default gate compatible).
"""

import array
import ctypes
import gc
import os
import socket
import tempfile

import pytest

from pegaflow import vmm_ipc
from pegaflow.vmm_ipc import (
    _FdServer,
    _import_mapping,
    _Mapping,
    _VmmMappedView,
    is_cumem_tensor,
)


@pytest.fixture()
def fd_server():
    server = _FdServer()
    yield server
    server.close()  # ends the accept loop; per-test instances must not leak threads


def _receive_fd(uds_path: str, token: str) -> tuple[bytes, int | None]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(uds_path)
        s.sendall(token.encode().ljust(32))
        msg, ancillary, _, _ = s.recvmsg(1, socket.CMSG_SPACE(4))
        if not ancillary:
            return msg, None
        fds = array.array("i")
        fds.frombytes(ancillary[0][2][:4])
        return msg, fds[0]
    finally:
        s.close()


class TestFdServer:
    def test_handoff_delivers_the_same_file(self, fd_server):
        with tempfile.NamedTemporaryFile() as f:
            fd = os.dup(f.fileno())
            token = fd_server.publish_allocation(0x1000, 4096, lambda: fd)

            msg, received = _receive_fd(fd_server.path, token)

            assert msg == b"F"
            assert received is not None
            # SCM_RIGHTS dup'd the descriptor: same underlying file.
            assert os.fstat(received).st_ino == os.fstat(fd).st_ino
            os.close(received)

    def test_unknown_token_is_refused(self, fd_server):
        msg, received = _receive_fd(fd_server.path, "nope")
        assert msg == b"E"
        assert received is None

    def test_close_all_invalidates_exports(self, fd_server):
        r, w = os.pipe()
        os.close(w)
        token = fd_server.publish_allocation(0x2000, 1, lambda: r)

        fd_server.close_all()

        with pytest.raises(OSError):
            os.fstat(r)
        msg, received = _receive_fd(fd_server.path, token)
        assert msg == b"E"
        assert received is None

    def test_one_export_per_allocation_is_reused(self, fd_server):
        r, w = os.pipe()
        os.close(w)
        calls = []

        def mint():
            calls.append(1)
            return r

        t1 = fd_server.publish_allocation(0x3000, 64, mint)
        t2 = fd_server.publish_allocation(0x3000, 64, mint)
        assert t1 == t2
        assert len(calls) == 1  # layers in one allocation share the export

    def test_garbage_request_does_not_kill_the_handoff_thread(self, fd_server):
        # Abstract sockets are connectable by any local process: junk bytes
        # or a stalled client must never wedge or kill the serving thread.
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(fd_server.path)
        s.sendall(b"\xff\xfe\xfd")
        s.close()

        with tempfile.NamedTemporaryFile() as f:
            fd = os.dup(f.fileno())
            token = fd_server.publish_allocation(0x4000, 128, lambda: fd)
            msg, received = _receive_fd(fd_server.path, token)
            assert msg == b"F"
            assert received is not None
            os.close(received)


class _FakeCuda:
    """Records driver calls; returns success and hands out fake handles."""

    def __init__(self):
        self.calls = []

    def _record(self, name):
        def call(*args):
            self.calls.append(name)
            if name == "cuMemImportFromShareableHandle":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_uint64))[0] = 0xBEEF
                return 0
            if name == "cuMemAddressReserve":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_uint64))[0] = 0xA000
                return 0
            return 0

        return call

    def __getattr__(self, name):
        return self._record(name)


@pytest.fixture()
def fake_cuda(monkeypatch):
    fake = _FakeCuda()
    monkeypatch.setattr(vmm_ipc, "_libcuda", fake)
    return fake


class TestImportMapping:
    def _publish(self, fd_server) -> str:
        r, w = os.pipe()
        os.close(w)
        return fd_server.publish_allocation(0x5000, 8192, lambda: r)

    def test_full_import_flow_and_cache(self, fd_server, fake_cuda):
        token = self._publish(fd_server)

        mapping = _import_mapping(fd_server.path, token, 8192, device_index=0)

        assert mapping.va == 0xA000
        for expected in (
            "cuMemImportFromShareableHandle",
            "cuMemAddressReserve",
            "cuMemMap",
            "cuMemSetAccess",
        ):
            assert expected in fake_cuda.calls
        # Second wrapper inside the same allocation reuses the mapping
        # (no second handoff/import).
        n_calls = len(fake_cuda.calls)
        again = _import_mapping(fd_server.path, token, 8192, device_index=0)
        assert again is mapping
        assert len(fake_cuda.calls) == n_calls

    def test_view_gc_releases_mapping_and_fd(self, fd_server, fake_cuda):
        token = self._publish(fd_server)
        mapping = _import_mapping(fd_server.path, token, 8192, device_index=0)
        received_fd = mapping.fd
        view = _VmmMappedView(mapping, offset=256, nbytes=1024, device_index=0)

        assert view.data_ptr() == 0xA000 + 256
        assert view.untyped_storage().nbytes() == 1024
        assert view.device.index == 0

        del view, mapping
        gc.collect()

        assert "cuMemUnmap" in fake_cuda.calls
        assert "cuMemRelease" in fake_cuda.calls
        with pytest.raises(OSError):
            os.fstat(received_fd)

    def test_release_is_idempotent(self, fake_cuda):
        r, w = os.pipe()
        os.close(w)
        m = _Mapping(va=1, size=2, handle=3, fd=r)
        m.release()
        n = fake_cuda.calls.count("cuMemRelease")
        m.release()
        assert fake_cuda.calls.count("cuMemRelease") == n


class TestImportFailureCleanup:
    def test_map_failure_releases_handle_and_fd(self, fd_server, monkeypatch):
        class FailingMapCuda(_FakeCuda):
            def _record(self, name):
                base = super()._record(name)

                def call(*args):
                    rc = base(*args)
                    return 999 if name == "cuMemMap" else rc

                return call

        fake = FailingMapCuda()
        monkeypatch.setattr(vmm_ipc, "_libcuda", fake)

        r, w = os.pipe()
        os.close(w)
        token = fd_server.publish_allocation(0x6000, 4096, lambda: r)

        with pytest.raises(RuntimeError, match="cuMemMap"):
            _import_mapping(fd_server.path, token, 4096, device_index=0)

        # A failed import must not pin the exporter's physical memory or
        # leak the received descriptor.
        assert "cuMemRelease" in fake.calls
        assert "cuMemAddressFree" in fake.calls


class TestCumemDetection:
    def test_fake_and_plain_objects_take_the_legacy_path(self):
        class FakeTensor:
            pass

        assert is_cumem_tensor(FakeTensor()) is False
        assert is_cumem_tensor(object()) is False
