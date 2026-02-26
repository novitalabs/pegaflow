#!/usr/bin/env python3
"""
IPC Receiver: Receives IPC handle via ZMQ, reconstructs tensor, then uses Rust to access GPU memory.

Usage:
    Terminal 1: python examples/ipc_sender.py
    Terminal 2: python examples/ipc_receiver.py
"""

import pickle

import numpy as np
import torch
import zmq

from pegaflow import GpuMemory


class CudaIPCWrapper:
    """Wrapper for CUDA IPC handle with tensor metadata."""

    def __init__(self, tensor: torch.Tensor):
        assert tensor.storage_offset() == 0, "Tensor must have zero storage offset"
        assert tensor.is_contiguous(), "Tensor must be contiguous"

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device.index

    def to_tensor(self):
        """Reconstruct tensor from IPC handle."""
        storage = torch.UntypedStorage._new_shared_cuda(*self.handle)
        device = self.handle[0]
        t = torch.tensor([], device=device, dtype=self.dtype)
        t.set_(storage)
        return t.view(self.shape)


def main():
    print("=" * 60)
    print("IPC Receiver - Cross-Process GPU Memory Sharing")
    print("=" * 60)

    # Initialize CUDA
    torch.cuda.init()

    # Step 1: Connect to sender via ZMQ
    print("\n[Receiver] Step 1: Connecting to sender at tcp://localhost:5555...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    print("  ✓ Connected!")

    # Step 2: Request data
    print("\n[Receiver] Step 2: Requesting IPC handle...")
    socket.send(b"REQUEST_IPC_HANDLE")

    # Step 3: Receive IPC wrapper
    print("[Receiver] Waiting for data...")
    data = socket.recv()
    print(f"  Received {len(data)} bytes")

    ipc_wrapper = pickle.loads(data)
    print("  ✓ IPC wrapper deserialized!")

    print("\n[Receiver] Step 3: IPC wrapper info:")
    print(f"  Shape: {ipc_wrapper.shape}")
    print(f"  Dtype: {ipc_wrapper.dtype}")
    print(f"  Device: {ipc_wrapper.device}")
    print(f"  Handle type: {type(ipc_wrapper.handle)}")
    print(f"  Handle length: {len(ipc_wrapper.handle)}")

    # Step 4: Reconstruct tensor from IPC handle (PyTorch)
    print("\n[Receiver] Step 4: Reconstructing tensor from IPC handle (PyTorch)...")

    try:
        tensor = ipc_wrapper.to_tensor()

        print("  ✓ Tensor reconstructed!")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Tensor device: {tensor.device}")
        print(f"  Tensor dtype: {tensor.dtype}")
        print(f"  Tensor data_ptr: 0x{tensor.data_ptr():x}")

        # Step 5: Verify data via PyTorch
        print("\n[Receiver] Step 5: Verifying tensor data (PyTorch)...")
        print(f"  Received tensor: {tensor}")

        expected = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            device=f"cuda:{ipc_wrapper.device}",
        )

        if torch.allclose(tensor, expected):
            print("  ✓ SUCCESS: Data matches expected values!")
            print("  ✓ Cross-process GPU memory sharing works!")
        else:
            print("  ✗ FAILED: Data mismatch!")
            print(f"  Expected: {expected}")
            print(f"  Got: {tensor}")

        # Step 6: Now use Rust to access the same GPU memory
        print("\n[Receiver] Step 6: Accessing GPU memory via Rust...")
        data_ptr = tensor.data_ptr()
        size_bytes = tensor.numel() * tensor.element_size()

        gpu_mem = GpuMemory(data_ptr, size_bytes)
        print("  ✓ GpuMemory handle created!")
        print(f"  Data pointer: 0x{gpu_mem.data_ptr():x}")
        print(f"  Size: {gpu_mem.size_bytes()} bytes")

        # Read via Rust
        print("\n[Receiver] Step 7: Reading GPU memory via Rust CUDA API...")
        read_bytes = gpu_mem.read_to_host(size_bytes)
        read_data = np.frombuffer(read_bytes, dtype=np.float32)
        print(f"  ✓ Read {len(read_bytes)} bytes from GPU via Rust")
        print(f"  Data: {read_data}")

        # Step 8: Modify via Rust
        print("\n[Receiver] Step 8: Modifying GPU memory via Rust...")
        modified_data = (read_data * 2.0).astype(np.float32)
        modified_bytes = modified_data.tobytes()
        print(f"  Writing modified data: {modified_data}")
        gpu_mem.write_from_host(modified_bytes)
        print("  ✓ Write successful via Rust!")

        # Verify via PyTorch
        print("\n[Receiver] Step 9: Verifying modification via PyTorch...")
        print(f"  Tensor after Rust modification: {tensor}")
        expected_modified = expected * 2.0

        if torch.allclose(tensor, expected_modified):
            print("  ✓ SUCCESS: Rust modification visible in PyTorch tensor!")
            print("  ✓ Rust can read/write cross-process GPU memory!")
            print("  ✓ Sender's tensor should also show modified values!")
        else:
            print("  ✗ FAILED: Modification not visible!")
            print(f"  Expected: {expected_modified}")
            print(f"  Got: {tensor}")

    except Exception as e:
        print("  ✗ ERROR: Failed to reconstruct tensor!")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    print("\n[Receiver] Cleaning up...")
    socket.close()
    context.term()

    print("\n" + "=" * 60)
    print("Receiver completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
