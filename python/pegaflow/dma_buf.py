"""Registration-scoped, same-host file descriptor handoff for CUDA DMA-BUF."""

import array
import os
import secrets
import socket
import tempfile
import threading
from contextlib import AbstractContextManager


class DmaBufExports(AbstractContextManager):
    """Keep owner exports alive until the registration RPC has consumed them."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="pega-dmabuf-")
        self.address = os.path.join(self._directory.name, "fd.sock")
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._socket.bind(self.address)
        self._socket.listen()
        self._socket.settimeout(0.2)
        self._fds: dict[bytes, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="pega-dmabuf", daemon=True)
        self._thread.start()

    def add(self, fd: int) -> tuple[str, bytes]:
        """Take ownership of fd and return a serializable one-use capability."""
        token = secrets.token_bytes(32)
        with self._lock:
            self._fds[token] = fd
        return self.address, token

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(5)
                try:
                    token = connection.recv(33)
                    with self._lock:
                        fd = self._fds.pop(token, None)
                    if fd is None:
                        connection.send(b"E")
                        continue
                    try:
                        connection.sendmsg(
                            [b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                        )
                    finally:
                        os.close(fd)
                except OSError:
                    # The RPC may have been cancelled while this peer connected.
                    continue

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._socket.close()
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
        self._directory.cleanup()


def receive_dma_buf(address: str, token: bytes) -> int:
    """Receive a new process-local descriptor; caller must close it."""
    fds = array.array("i")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(10)
            connection.connect(address)
            connection.sendall(token)
            message, ancillary, flags, _ = connection.recvmsg(
                1, socket.CMSG_SPACE(fds.itemsize), socket.MSG_CMSG_CLOEXEC
            )
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(data[: len(data) - len(data) % fds.itemsize])
        if message != b"F" or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(fds) != 1:
            raise RuntimeError("CUDA DMA-BUF handoff did not return exactly one descriptor")
        return fds.pop()
    finally:
        for fd in fds:
            os.close(fd)
