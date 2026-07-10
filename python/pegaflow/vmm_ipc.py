"""CUDA VMM (cuMemCreate) cross-process sharing for sleep-mode KV caches.

vLLM's --enable-sleep-mode allocates KV under its cumem allocator; legacy
CUDA IPC (storage._share_cuda_()) cannot share those allocations, so
PegaFlow registration used to die with "invalid argument". This module
shares them the VMM way instead:

  worker:  cuMemExportToShareableHandle(POSIX_FD)  [stock vLLM allocations
           are exportable on any GPU whose driver reports fabric/posix
           handle support — the C extension requests the type at create]
  handoff: unix socket + SCM_RIGHTS (pidfd_getfd is blocked between
           sibling processes under yama ptrace_scope=1)
  server:  cuMemImportFromShareableHandle + cuMemAddressReserve + cuMemMap
           inside pegaflow-server's embedded python; the returned object
           duck-types the registry contract (data_ptr / device.index /
           untyped_storage().nbytes).

Sleep/wake lifecycle (driven by the orchestrator via /collective_rpc):
an imported handle REFCOUNTS the physical memory, so the server must drop
its mappings and the worker must close its exported FDs BEFORE vLLM
sleeps, or sleep frees nothing. Wake maps fresh physical chunks into the
SAME virtual addresses (tensor objects stay valid), so re-registration
just re-exports and re-registers the same tensors.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import os
import socket
import threading
import uuid
import weakref
from dataclasses import dataclass

from pegaflow.logging_utils import get_connector_logger

logger = get_connector_logger()

_CU_MEM_HANDLE_TYPE_POSIX_FD = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_READWRITE = 3


class _AccessDesc(ctypes.Structure):
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


_libcuda = None


def _cuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: CUresult={rc}")


# --------------------------------------------------------------------------
# Worker side: export + FD handoff server
# --------------------------------------------------------------------------


@dataclass
class _Export:
    fd: int
    d_mem: int
    size: int


class _FdServer:
    """One per worker process: serves exported allocation FDs over an
    abstract unix socket. The server connects during registration, sends a
    token, and receives the FD via SCM_RIGHTS."""

    _instance: _FdServer | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.path = f"\0pegaflow-vmm-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._exports: dict[str, _Export] = {}
        # One export per cumem allocation, shared by every KV tensor inside
        # it: d_mem -> token of the existing export.
        self._token_by_dmem: dict[int, str] = {}
        self._exports_lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(16)
        t = threading.Thread(target=self._serve, name="pegaflow-vmm-fds", daemon=True)
        t.start()

    @classmethod
    def has_instance(cls) -> bool:
        with cls._lock:
            return cls._instance is not None

    @classmethod
    def instance(cls) -> _FdServer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = _FdServer()
            return cls._instance

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                # Abstract sockets have no filesystem permissions: any local
                # process can connect. The 128-bit token is the secret, and
                # the connection must never be able to wedge (timeout) or
                # kill (broad except) this thread — a dead handoff thread
                # hangs every later registration.
                conn.settimeout(5.0)
                token = conn.recv(64).decode(errors="replace")
                with self._exports_lock:
                    exp = self._exports.get(token)
                    # Send under the lock with a dup: close_all may close
                    # the original fd concurrently, and the fd NUMBER can
                    # be reused by a new export — sending a dup taken
                    # while the export is provably alive cannot hand out
                    # the wrong descriptor.
                    dup_fd = os.dup(exp.fd) if exp is not None else -1
                if dup_fd < 0:
                    conn.sendall(b"E")
                else:
                    try:
                        conn.sendmsg(
                            [b"F"],
                            [
                                (
                                    socket.SOL_SOCKET,
                                    socket.SCM_RIGHTS,
                                    array.array("i", [dup_fd]).tobytes(),
                                )
                            ],
                        )
                    finally:
                        os.close(dup_fd)
            except Exception:
                logger.exception("[pegaflow.vmm] fd handoff request failed")
            finally:
                conn.close()

    def publish_allocation(self, d_mem: int, size: int, export_fd) -> str:
        """Publish (or reuse) the export for the allocation at d_mem.
        export_fd is a callable minting the FD only when a new export is
        actually needed."""
        with self._exports_lock:
            token = self._token_by_dmem.get(d_mem)
            if token is not None and token in self._exports:
                return token
        fd = export_fd()
        with self._exports_lock:
            token = uuid.uuid4().hex
            self._exports[token] = _Export(fd=fd, d_mem=d_mem, size=size)
            self._token_by_dmem[d_mem] = token
            return token

    def close_all(self) -> None:
        """Drop every exported FD. MUST run before vLLM sleeps: an exported
        FD holds a reference to the physical memory."""
        with self._exports_lock:
            for exp in self._exports.values():
                with contextlib.suppress(OSError):
                    os.close(exp.fd)
            self._exports.clear()
            self._token_by_dmem.clear()


def _find_cumem_allocation(storage_ptr: int):
    """(d_mem, size, p_memHandle) of the vLLM cumem allocation covering
    storage_ptr, or None if the pointer is not cumem-managed."""
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except ImportError:
        return None
    try:
        alloc = CuMemAllocator.get_instance()
    except Exception:
        return None
    for _, data in alloc.pointer_to_data.items():
        device, size, d_mem, p_mem_handle = data.handle
        if d_mem <= storage_ptr < d_mem + size:
            return d_mem, size, p_mem_handle
    return None


def is_cumem_tensor(tensor) -> bool:
    """True only for tensors provably inside a vLLM cumem allocation;
    anything unreadable (test fakes, CPU tensors) takes the legacy path."""
    try:
        ptr = tensor.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return False
    return _find_cumem_allocation(ptr) is not None


class VmmCudaIPCWrapper:
    """Drop-in sibling of CudaIPCWrapper for cumem (sleep-mode) tensors.

    Pickled to the server; to_tensor() runs THERE and returns a duck-typed
    mapped view (registry only needs data_ptr / device.index /
    untyped_storage().nbytes)."""

    def __init__(self, tensor) -> None:
        from pegaflow.ipc_wrapper import CudaIPCWrapper

        storage = tensor.untyped_storage()
        assert tensor.storage_offset() == 0, "Tensor must have zero storage offset"
        found = _find_cumem_allocation(storage.data_ptr())
        if found is None:
            raise RuntimeError("tensor is not managed by the vLLM cumem allocator")
        d_mem, size, p_mem_handle = found

        def _export_fd() -> int:
            # The generic allocation handle lives at the C-side pointer
            # vLLM hands back to python; re-read AT EXPORT TIME — wake_up
            # creates a fresh physical handle at the same address after
            # every sleep.
            generic = ctypes.c_uint64.from_address(p_mem_handle).value
            fd = ctypes.c_int(-1)
            rc = _cuda().cuMemExportToShareableHandle(
                ctypes.byref(fd), ctypes.c_uint64(generic), _CU_MEM_HANDLE_TYPE_POSIX_FD, 0
            )
            if rc != 0:
                raise RuntimeError(
                    f"cuMemExportToShareableHandle failed (CUresult={rc}): the "
                    "allocation was not created with an exportable handle type. "
                    "vLLM's cumem allocator requests one only when the driver "
                    "reports CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED "
                    "(observed true on recent drivers even for consumer GPUs); "
                    "on this platform sleep-mode KV cannot be shared with "
                    "PegaFlow — disable sleep mode or the connector."
                )
            return fd.value

        server = _FdServer.instance()
        self.token = server.publish_allocation(d_mem, size, _export_fd)

        self.uds_path = server.path
        self.alloc_size = size
        self.offset = storage.data_ptr() - d_mem
        self.nbytes = storage.nbytes()
        self.device_uuid = CudaIPCWrapper._get_device_uuid(tensor.device.index)

    # ---- server side ----
    def to_tensor(self):
        device_index = _resolve_device_index(self.device_uuid)
        mapping = _import_mapping(self.uds_path, self.token, self.alloc_size, device_index)
        return _VmmMappedView(mapping, self.offset, self.nbytes, device_index)


def _resolve_device_index(device_uuid: str) -> int:
    from pegaflow.ipc_wrapper import CudaIPCWrapper

    return CudaIPCWrapper._get_device_index_from_uuid(device_uuid)


# --------------------------------------------------------------------------
# Server side: import + mapping cache (one mapping per allocation, shared
# by every layer wrapper that lives inside it)
# --------------------------------------------------------------------------


class _Mapping:
    """Released when the LAST view holding it is garbage-collected — the
    registry's drop_instance decrefs its held views and runs gc.collect(),
    which is exactly the moment the server must stop pinning the worker's
    physical memory (so sleep can actually free VRAM)."""

    def __init__(self, va: int, size: int, handle: int, fd: int) -> None:
        self.va, self.size, self.handle, self.fd = va, size, handle, fd
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        cuda = _cuda()
        try:
            cuda.cuMemUnmap(ctypes.c_uint64(self.va), ctypes.c_size_t(self.size))
            cuda.cuMemAddressFree(ctypes.c_uint64(self.va), ctypes.c_size_t(self.size))
            cuda.cuMemRelease(ctypes.c_uint64(self.handle))
        finally:
            with contextlib.suppress(OSError):
                os.close(self.fd)

    def __del__(self) -> None:
        # Interpreter-shutdown GC must never raise through __del__.
        with contextlib.suppress(Exception):
            self.release()


_mappings: dict[str, weakref.ref[_Mapping]] = {}
_mappings_lock = threading.Lock()


def _import_mapping(uds_path: str, token: str, size: int, device_index: int) -> _Mapping:
    with _mappings_lock:
        ref = _mappings.get(token)
        cached = ref() if ref is not None else None
        if cached is not None:
            return cached

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(uds_path)
        s.sendall(token.encode())
        msg, ancillary, _, _ = s.recvmsg(1, socket.CMSG_SPACE(4))
        if msg != b"F" or not ancillary:
            raise RuntimeError(f"FD handoff failed for token {token!r}")
        fds = array.array("i")
        fds.frombytes(ancillary[0][2][:4])
        fd = fds[0]
    finally:
        s.close()

    cuda = _cuda()
    handle = ctypes.c_uint64()
    va = ctypes.c_uint64()
    mapped = False
    try:
        _check(
            cuda.cuMemImportFromShareableHandle(
                ctypes.byref(handle), ctypes.c_int(fd), _CU_MEM_HANDLE_TYPE_POSIX_FD
            ),
            "cuMemImportFromShareableHandle",
        )
        _check(
            cuda.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(size), 0, 0, 0),
            "cuMemAddressReserve",
        )
        _check(cuda.cuMemMap(va, ctypes.c_size_t(size), 0, handle, 0), "cuMemMap")
        mapped = True
        desc = _AccessDesc(
            _CU_MEM_LOCATION_TYPE_DEVICE, device_index, _CU_MEM_ACCESS_FLAGS_READWRITE
        )
        _check(
            cuda.cuMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(desc), 1),
            "cuMemSetAccess",
        )
    except Exception:
        # A failed import must not pin the exporter's physical memory or
        # leak the received descriptor.
        if mapped:
            cuda.cuMemUnmap(va, ctypes.c_size_t(size))
        if va.value:
            cuda.cuMemAddressFree(va, ctypes.c_size_t(size))
        if handle.value:
            cuda.cuMemRelease(handle)
        with contextlib.suppress(OSError):
            os.close(fd)
        raise

    mapping = _Mapping(va.value, size, handle.value, fd)
    with _mappings_lock:
        _mappings[token] = weakref.ref(mapping)
    logger.info(
        "[pegaflow.vmm] imported %d bytes on device %d (token %s)", size, device_index, token[:8]
    )
    return mapping


class _Device:
    def __init__(self, index: int) -> None:
        self.index = index


class _VmmMappedView:
    """Registry-contract duck: data_ptr() / device.index /
    untyped_storage().nbytes(). Keeping the object alive keeps the CUDA
    mapping alive (mirrors how the registry holds IPC tensors); when the
    registry drops it, the shared _Mapping releases and the worker's
    physical memory is unpinned."""

    def __init__(self, mapping: _Mapping, offset: int, nbytes: int, device_index: int) -> None:
        self._mapping = mapping
        self._ptr = mapping.va + offset
        self._nbytes = nbytes
        self.device = _Device(device_index)

    def data_ptr(self) -> int:
        return self._ptr

    def untyped_storage(self):
        return self

    def nbytes(self) -> int:
        return self._nbytes
