#!/usr/bin/env python3
"""PegaEngine Server - Independent process for KV cache management.

This server runs as a separate process from vLLM and handles GPU KV cache
operations via CUDA IPC. It receives tensor IPC handles from vLLM, reconstructs
the tensors, and uses the Rust PegaEngine backend for efficient storage/retrieval.

Architecture:
    vLLM Process                    Engine Server Process
    ┌──────────────┐               ┌──────────────────┐
    │ Connector    │  ─── ZMQ ──→  │ ZMQ Server       │
    │ (Client)     │  ← Request →  │ + PegaEngine     │
    └──────────────┘               └──────────────────┘
         ↓                                  ↓
    GPU Tensor ────── CUDA IPC ────→ Reconstructed Tensor
                    (shared memory)         ↓
                                      tensor.data_ptr()
                                            ↓
                                      Rust PegaEngine

Protocol:
    Request:  (command: str, payload: dict)
    Response: (status: str, result: Any)

Commands:
    - REGISTER: Register KV cache layer from IPC handle
    - SAVE: Save blocks to CPU storage
    - LOAD: Load blocks to GPU
    - QUERY: Query cache hit count
    - SHUTDOWN: Clean shutdown
"""

import argparse
import logging
import os
import pickle
import signal
import sys
from typing import Any, Dict, Optional

import torch
import zmq

from pegaflow.pegaflow import PegaEngine

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CudaIPCWrapper:
    """Wrapper for CUDA IPC handle with tensor metadata.

    This is a copy of the class from ipc_wrapper.py to avoid import issues.
    """

    def __init__(self, tensor: torch.Tensor):
        assert tensor.storage_offset() == 0, "Tensor must have zero storage offset"
        assert tensor.is_contiguous(), "Tensor must be contiguous"

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device.index

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct tensor from IPC handle."""
        storage = torch.UntypedStorage._new_shared_cuda(*self.handle)
        device = self.handle[0]
        t = torch.tensor([], device=device, dtype=self.dtype)
        t.set_(storage)
        return t.view(self.shape)


class PegaEngineServer:
    """ZMQ server wrapping Rust PegaEngine for cross-process KV cache access."""

    def __init__(self, socket_path: str, device: int = 0):
        """Initialize the engine server.

        Args:
            socket_path: ZMQ socket endpoint (e.g., "ipc:///tmp/pega_engine.sock")
            device: CUDA device index
        """
        self.socket_path = socket_path
        self.device = device

        # Initialize CUDA
        torch.cuda.init()
        torch.cuda.set_device(device)
        logger.info("Initialized CUDA device %d", device)

        # Initialize Rust PegaEngine
        self.engine = PegaEngine()
        logger.info("Initialized PegaEngine")

        # Store reconstructed tensors to keep them alive
        # Key: layer_name, Value: torch.Tensor
        self._tensors: Dict[str, torch.Tensor] = {}

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Clean up existing IPC socket if needed
        if socket_path.startswith("ipc://"):
            ipc_path = socket_path.replace("ipc://", "")
            try:
                os.unlink(ipc_path)
            except FileNotFoundError:
                pass

        self.socket.bind(socket_path)
        logger.info("ZMQ server bound to %s", socket_path)

        self.running = False

    def _handle_register(self, payload: dict) -> dict:
        """Handle REGISTER command - register KV cache from IPC handle.

        Args:
            payload: {
                'layer_name': str,
                'wrapper_bytes': bytes,  # pickled CudaIPCWrapper
                'num_blocks': int,
                'bytes_per_block': int,
                'kv_stride_bytes': int,
                'segments': int,
            }

        Returns:
            {'status': 'success'} or {'status': 'error', 'message': str}
        """
        try:
            layer_name = payload['layer_name']
            wrapper_bytes = payload['wrapper_bytes']
            num_blocks = payload['num_blocks']
            bytes_per_block = payload['bytes_per_block']
            kv_stride_bytes = payload['kv_stride_bytes']
            segments = payload['segments']

            # Reconstruct tensor from IPC handle
            wrapper = pickle.loads(wrapper_bytes)
            tensor = wrapper.to_tensor()

            # Store tensor reference to keep GPU memory alive
            self._tensors[layer_name] = tensor

            # Register with Rust PegaEngine using raw pointer
            data_ptr = tensor.data_ptr()
            size_bytes = tensor.untyped_storage().nbytes()

            self.engine.register_kv_cache(
                layer_name,
                data_ptr,
                size_bytes,
                num_blocks,
                bytes_per_block,
                kv_stride_bytes,
                segments,
            )

            logger.info(
                "Registered layer '%s': %d blocks, %d bytes/block, ptr=0x%x",
                layer_name, num_blocks, bytes_per_block, data_ptr
            )

            return {'status': 'success'}

        except Exception as e:
            logger.error("Failed to register KV cache: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _handle_save(self, payload: dict) -> dict:
        """Handle SAVE command - save blocks to CPU storage.

        Args:
            payload: {
                'layer_name': str,
                'block_ids': list[int],
                'block_hashes': list[bytes],
            }

        Returns:
            {'status': 'success'} or {'status': 'error', 'message': str}
        """
        try:
            layer_name = payload['layer_name']
            block_ids = payload['block_ids']
            block_hashes = payload['block_hashes']

            self.engine.save_kv_blocks_from_ipc(
                layer_name,
                block_ids,
                block_hashes
            )

            logger.debug(
                "Saved %d blocks for layer '%s'",
                len(block_ids), layer_name
            )

            return {'status': 'success'}

        except Exception as e:
            logger.error("Failed to save blocks: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _handle_load(self, payload: dict) -> dict:
        """Handle LOAD command - batch load blocks to GPU.

        Args:
            payload: {
                'layer_names': list[str],
                'block_ids': list[int],
                'block_hashes': list[bytes],
            }

        Returns:
            {
                'status': 'success',
                'num_layers_loaded': int,
                'total_bytes': int,
            }
            or {'status': 'error', 'message': str}
        """
        try:
            layer_names = payload['layer_names']
            block_ids = payload['block_ids']
            block_hashes = payload['block_hashes']

            num_layers_loaded, total_bytes = self.engine.batch_load_kv_blocks(
                layer_names,
                block_ids,
                block_hashes
            )

            logger.debug(
                "Loaded %d blocks across %d layers (%d bytes)",
                len(block_ids), num_layers_loaded, total_bytes
            )

            return {
                'status': 'success',
                'num_layers_loaded': num_layers_loaded,
                'total_bytes': total_bytes,
            }

        except Exception as e:
            logger.error("Failed to load blocks: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _handle_query(self, payload: dict) -> dict:
        """Handle QUERY command - count available blocks.

        Args:
            payload: {
                'block_hashes': list[bytes],
            }

        Returns:
            {'status': 'success', 'hit_blocks': int}
            or {'status': 'error', 'message': str}
        """
        try:
            block_hashes = payload['block_hashes']

            hit_blocks = self.engine.count_prefix_hit_blocks(block_hashes)

            return {
                'status': 'success',
                'hit_blocks': hit_blocks,
            }

        except Exception as e:
            logger.error("Failed to query blocks: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _handle_wait_layer(self, payload: dict) -> dict:
        """Handle WAIT_LAYER command - wait for layer transfer.

        Args:
            payload: {
                'layer_name': str,
            }

        Returns:
            {'status': 'success'} or {'status': 'error', 'message': str}
        """
        try:
            layer_name = payload['layer_name']
            self.engine.wait_for_layer_transfer(layer_name)
            return {'status': 'success'}
        except Exception as e:
            logger.error("Failed to wait for layer: %s", e, exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _handle_shutdown(self, payload: dict) -> dict:
        """Handle SHUTDOWN command - graceful shutdown."""
        logger.info("Received shutdown command")
        self.running = False
        return {'status': 'success'}

    def run(self):
        """Main server loop - receive and handle requests."""
        self.running = True
        logger.info("PegaEngine server running, waiting for requests...")

        while self.running:
            try:
                # Receive request
                message = self.socket.recv()

                # Deserialize request
                command, payload = pickle.loads(message)

                # Dispatch command
                if command == 'REGISTER':
                    response = self._handle_register(payload)
                elif command == 'SAVE':
                    response = self._handle_save(payload)
                elif command == 'LOAD':
                    response = self._handle_load(payload)
                elif command == 'QUERY':
                    response = self._handle_query(payload)
                elif command == 'WAIT_LAYER':
                    response = self._handle_wait_layer(payload)
                elif command == 'SHUTDOWN':
                    response = self._handle_shutdown(payload)
                else:
                    response = {
                        'status': 'error',
                        'message': f'Unknown command: {command}'
                    }

                # Send response
                self.socket.send(pickle.dumps(response))

            except zmq.ZMQError as e:
                if not self.running:
                    break
                logger.error("ZMQ error: %s", e)
            except Exception as e:
                logger.error("Error handling request: %s", e, exc_info=True)
                # Send error response
                try:
                    response = {'status': 'error', 'message': str(e)}
                    self.socket.send(pickle.dumps(response))
                except Exception:
                    pass

        logger.info("Server loop exited")

    def shutdown(self):
        """Clean shutdown - close sockets and cleanup resources."""
        logger.info("Shutting down server...")
        self.running = False

        # Unregister all KV caches
        try:
            self.engine.unregister_all_kv_caches()
        except Exception as e:
            logger.error("Error unregistering KV caches: %s", e)

        # Clear tensor references
        self._tensors.clear()

        # Close ZMQ socket
        try:
            self.socket.close()
        except Exception as e:
            logger.error("Error closing socket: %s", e)

        # Terminate ZMQ context
        try:
            self.context.term()
        except Exception as e:
            logger.error("Error terminating context: %s", e)

        logger.info("Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='PegaEngine Server')
    parser.add_argument(
        '--socket',
        type=str,
        default='ipc:///tmp/pega_engine.sock',
        help='ZMQ socket path (default: ipc:///tmp/pega_engine.sock)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='CUDA device index (default: 0)'
    )
    args = parser.parse_args()

    # Create server
    server = PegaEngineServer(socket_path=args.socket, device=args.device)

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received signal %d, shutting down...", sig)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run server
    try:
        server.run()
    except Exception as e:
        logger.error("Server error: %s", e, exc_info=True)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
