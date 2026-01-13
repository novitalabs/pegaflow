"""CUDA IPC Wrapper for cross-process GPU memory sharing.

This module provides a wrapper class for PyTorch tensors that enables
cross-process GPU memory sharing via CUDA IPC handles. The wrapper can
be serialized (via pickle) and sent across process boundaries.

This implementation handles CUDA_VISIBLE_DEVICES correctly by using GPU UUIDs
instead of device indices for device identification.
"""

import threading

import torch


class CudaIPCWrapper:
    """Wrapper for CUDA IPC handle with tensor metadata.

    This class wraps a PyTorch CUDA tensor and extracts its IPC handle,
    allowing the tensor to be reconstructed in another process. It correctly
    handles CUDA_VISIBLE_DEVICES by using GPU UUIDs for device identification.

    Attributes:
        handle: CUDA IPC handle tuple (device, ipc_handle, size, offset, ...)
        dtype: PyTorch dtype of the tensor
        shape: Shape tuple of the tensor
        device_uuid: UUID string of the GPU device

    Example:
        # Process 1 (sender)
        tensor = torch.randn(10, device='cuda:0')
        wrapper = CudaIPCWrapper(tensor)
        serialized = pickle.dumps(wrapper)
        # ... send serialized bytes to another process ...

        # Process 2 (receiver)
        wrapper = pickle.loads(serialized)
        tensor = wrapper.to_tensor()  # Reconstruct tensor
        ptr = tensor.data_ptr()  # Get GPU pointer
    """

    # Class-level cache for device UUID to index mapping
    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """Get the UUID of a GPU device given its index.

        Args:
            device_index: CUDA device index (relative to CUDA_VISIBLE_DEVICES)

        Returns:
            UUID string of the GPU device
        """
        return str(torch.cuda.get_device_properties(device_index).uuid)

    @staticmethod
    def _discover_gpu_devices():
        """Discover all available GPU devices and map their UUIDs to
        the physical device ordinals (relative to CUDA_VISIBLE_DEVICES).
        """
        if not torch.cuda.is_available():
            return

        num_devices = torch.cuda.device_count()
        with CudaIPCWrapper._device_mapping_lock:
            if CudaIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = CudaIPCWrapper._get_device_uuid(i)
                CudaIPCWrapper._discovered_device_mapping[device_uuid] = i

    @staticmethod
    def _get_device_index_from_uuid(device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID.

        Args:
            device_uuid: UUID string of the GPU device

        Returns:
            Device index relative to CUDA_VISIBLE_DEVICES

        Raises:
            RuntimeError: If the device UUID is not found
        """
        CudaIPCWrapper._discover_gpu_devices()

        with CudaIPCWrapper._device_mapping_lock:
            device_index = CudaIPCWrapper._discovered_device_mapping.get(device_uuid, None)

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered devices. "
                "Please make sure the process can see all the GPU devices."
            )
        return device_index

    def __init__(self, tensor: torch.Tensor):
        """Create IPC wrapper from a CUDA tensor.

        Args:
            tensor: PyTorch CUDA tensor to wrap. Must be contiguous and
                   have zero storage offset.

        Raises:
            AssertionError: If tensor is not contiguous or has non-zero offset.
        """
        assert tensor.storage_offset() == 0, "Tensor must have zero storage offset"
        assert tensor.is_contiguous(), "Tensor must be contiguous"

        # Get the underlying storage and create IPC handle
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        # Store metadata needed to reconstruct the tensor
        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape

        # Store device UUID instead of device index to handle CUDA_VISIBLE_DEVICES
        device_index = tensor.device.index
        self.device_uuid = CudaIPCWrapper._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct tensor from IPC handle.

        This method creates a new tensor in the current process that shares
        the same GPU memory as the original tensor (via CUDA IPC).

        Returns:
            PyTorch tensor that shares GPU memory with the original tensor.

        Note:
            The reconstructed tensor shares memory with the original. Any
            modifications to one will be visible in the other.

            This function may break if torch.cuda is not initialized.
            Call torch.cuda.init() before using this function if needed.
        """
        # Get the correct device index in the current process based on UUID
        device = CudaIPCWrapper._get_device_index_from_uuid(self.device_uuid)

        # Reconstruct storage from IPC handle
        storage = torch.UntypedStorage._new_shared_cuda(device, *self.handle[1:])

        # Create empty tensor on the correct device
        t = torch.tensor([], device=device, dtype=self.dtype)

        # Set the tensor to use the shared storage
        t.set_(storage)

        # Reshape to original shape
        return t.view(self.shape)

    def __eq__(self, other) -> bool:
        """Check equality with another CudaIPCWrapper.

        Args:
            other: Object to compare with

        Returns:
            True if the wrappers refer to the same tensor, False otherwise
        """
        if not isinstance(other, CudaIPCWrapper):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.device_uuid == other.device_uuid
        )

    def __repr__(self) -> str:
        return (
            f"CudaIPCWrapper(shape={self.shape}, dtype={self.dtype}, "
            f"device_uuid={self.device_uuid})"
        )


__all__ = ["CudaIPCWrapper"]
