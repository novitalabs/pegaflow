import importlib.util
from pathlib import Path


_IPC_WRAPPER_PATH = Path(__file__).resolve().parents[1] / "pegaflow" / "ipc_wrapper.py"
_IPC_WRAPPER_SPEC = importlib.util.spec_from_file_location("ipc_wrapper", _IPC_WRAPPER_PATH)
assert _IPC_WRAPPER_SPEC is not None
assert _IPC_WRAPPER_SPEC.loader is not None
ipc_wrapper = importlib.util.module_from_spec(_IPC_WRAPPER_SPEC)
_IPC_WRAPPER_SPEC.loader.exec_module(ipc_wrapper)
CudaIPCWrapper = ipc_wrapper.CudaIPCWrapper


class FakeTensor:
    def __init__(self):
        self.set_args = None
        self.view_shape = None

    def set_(self, *args):
        self.set_args = args
        return self

    def view(self, shape):
        self.view_shape = shape
        return self


class FakeUntypedStorage:
    @staticmethod
    def _new_shared_cuda(device, *handle_args):
        return ("storage", device, handle_args)


class FakeTorch:
    UntypedStorage = FakeUntypedStorage

    def __init__(self):
        self.created_tensor = FakeTensor()

    def tensor(self, *args, **kwargs):
        return self.created_tensor


def _wrapper_without_init(stride=None, storage_offset=0):
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = ("ignored_device", "ipc_handle", "size")
    wrapper.dtype = "fake-dtype"
    wrapper.shape = (2, 3)
    wrapper.device_uuid = "GPU-fake"
    if stride is not None:
        wrapper.stride = stride
        wrapper.storage_offset = storage_offset
    return wrapper


def test_to_tensor_preserves_strided_storage(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(ipc_wrapper, "torch", fake_torch)
    monkeypatch.setattr(CudaIPCWrapper, "_get_device_index_from_uuid", staticmethod(lambda _: 7))

    tensor = _wrapper_without_init(stride=(4, 1), storage_offset=5).to_tensor()

    assert tensor.set_args == (("storage", 7, ("ipc_handle", "size")), 5, (2, 3), (4, 1))
    assert tensor.view_shape is None


def test_to_tensor_supports_legacy_pickles_without_stride(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(ipc_wrapper, "torch", fake_torch)
    monkeypatch.setattr(CudaIPCWrapper, "_get_device_index_from_uuid", staticmethod(lambda _: 3))

    tensor = _wrapper_without_init().to_tensor()

    assert tensor.set_args == (("storage", 3, ("ipc_handle", "size")),)
    assert tensor.view_shape == (2, 3)
