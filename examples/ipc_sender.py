#!/usr/bin/env python3
"""
IPC Sender: Creates a GPU tensor, sends GPU pointer via ZMQ.

Usage:
    Terminal 1: python examples/ipc_sender.py
    Terminal 2: python examples/ipc_receiver.py
"""

import pickle
import time

import torch
import zmq


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
    print("IPC Sender - Cross-Process GPU Memory Sharing")
    print("=" * 60)

    # Initialize CUDA
    torch.cuda.init()
    device = torch.device("cuda:0")

    # Step 1: Create a GPU tensor with test data
    print("\n[Sender] Step 1: Creating GPU tensor with test data...")
    tensor = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=torch.float32,
        device=device,
    )
    print(f"  Tensor: {tensor}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Data pointer: 0x{tensor.data_ptr():x}")

    # Step 2: Wrap tensor in IPC wrapper
    print("\n[Sender] Step 2: Creating CUDA IPC wrapper...")
    ipc_wrapper = CudaIPCWrapper(tensor)
    print("  ✓ IPC wrapper created!")
    print(f"  Handle type: {type(ipc_wrapper.handle)}")
    print(f"  Handle length: {len(ipc_wrapper.handle)}")
    print(f"  Shape: {ipc_wrapper.shape}")
    print(f"  Dtype: {ipc_wrapper.dtype}")
    print(f"  Device: {ipc_wrapper.device}")

    # Step 3: Setup ZMQ and send
    print("\n[Sender] Step 3: Setting up ZMQ server on tcp://*:5555...")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("  ✓ ZMQ server ready!")
    print("\n[Sender] Waiting for receiver to connect...")
    print("  (Start receiver in another terminal: python examples/ipc_receiver.py)")

    # Wait for request from receiver
    message = socket.recv()
    print(f"\n[Sender] Received request: {message.decode()}")

    # Send IPC wrapper via pickle
    print("[Sender] Sending IPC wrapper via pickle...")
    serialized = pickle.dumps(ipc_wrapper)
    print(f"  Serialized size: {len(serialized)} bytes")
    socket.send(serialized)
    print("  ✓ Data sent!")

    # Keep tensor alive while receiver is using it
    print("\n[Sender] Keeping tensor alive and monitoring for changes...")
    print("  (Receiver should access and modify the data)")
    print(f"\n  Initial tensor value: {tensor}")

    modified = False
    for i in range(30):
        time.sleep(1)

        # Check tensor value every second
        current_value = tensor.clone()
        expected_original = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            dtype=torch.float32,
            device=device,
        )
        expected_modified = expected_original * 2.0

        # Check if tensor was modified by receiver
        if not modified and torch.allclose(current_value, expected_modified):
            print(f"\n  ✓✓✓ DETECTED MODIFICATION at {i + 1} seconds! ✓✓✓")
            print(f"  Modified tensor value: {tensor}")
            print("  ✓ Cross-process memory sharing CONFIRMED!")
            print("  ✓ Receiver's changes are visible in sender's process!")
            modified = True

        # Print status every 5 seconds
        if i % 5 == 0:
            print(
                f"  ... {30 - i} seconds remaining (tensor: {tensor[:3].tolist()}...)"
            )

    # Final check
    print(f"\n[Sender] Final tensor value: {tensor}")
    if modified:
        print("  ✓ SUCCESS: Tensor was modified by receiver!")
    else:
        print("  ⚠ WARNING: Tensor was NOT modified by receiver")
        print("  (This might indicate IPC sharing didn't work as expected)")

    print("\n[Sender] Done! Cleaning up...")
    socket.close()
    context.term()

    print("\n" + "=" * 60)
    print("Sender completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
