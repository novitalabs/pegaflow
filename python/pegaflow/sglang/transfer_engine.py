from __future__ import annotations

import base64
import json
import logging
import os

from sglang.srt.environ import envs
from sglang.srt.utils import get_free_port, is_valid_ipv6_address, maybe_wrap_ipv6_address

from pegaflow import TransferEngine

logger = logging.getLogger(__name__)


def _has_nonzero_port(hostname: str) -> bool:
    hostname = hostname.strip()
    if not hostname:
        return False

    # [ipv6]:port
    if hostname.startswith("[") and "]:" in hostname:
        _, port_str = hostname.rsplit("]:", 1)
        return port_str.isdigit() and int(port_str) > 0

    # Bare IPv6 should not be treated as host:port.
    if is_valid_ipv6_address(hostname):
        return False

    # ipv4:port or hostname:port
    if ":" in hostname:
        _, port_str = hostname.rsplit(":", 1)
        return port_str.isdigit() and int(port_str) > 0

    return False


def _ensure_hostname_with_port(hostname: str) -> str:
    if _has_nonzero_port(hostname):
        return hostname
    return f"{maybe_wrap_ipv6_address(hostname)}:{get_free_port()}"


def get_ib_devices_for_gpu(ib_device_str: str | None, gpu_id: int) -> str | None:
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()
    is_json_file = ib_device_str.endswith(".json")
    if is_json_file:
        try:
            if os.path.isfile(ib_device_str):
                with open(ib_device_str, "r") as f:
                    ib_device_str = f.read()
            else:
                raise RuntimeError(f"File {ib_device_str} does not exist.")
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e

    try:
        parsed_json = json.loads(ib_device_str)
        if isinstance(parsed_json, dict):
            gpu_mapping: dict[int, str] = {}
            for gpu_key, ib_devices in parsed_json.items():
                if (
                    isinstance(gpu_key, str)
                    and gpu_key.isdigit()
                    and isinstance(ib_devices, str)
                ):
                    gpu_mapping[int(gpu_key)] = ib_devices.strip()
                elif isinstance(gpu_key, int) and isinstance(ib_devices, str):
                    gpu_mapping[gpu_key] = ib_devices.strip()
                else:
                    raise ValueError(
                        "Invalid format: keys must be integers (or integer strings) and values must be strings"
                    )

            if not gpu_mapping:
                raise ValueError("No valid GPU mappings found in JSON")

            if gpu_id in gpu_mapping:
                return gpu_mapping[gpu_id]
            raise ValueError(
                f"No IB devices configured for GPU {gpu_id}. Available GPUs: {list(gpu_mapping.keys())}"
            )
    except json.JSONDecodeError:
        if is_json_file:
            raise RuntimeError(f"Failed to parse JSON content from file {ib_device_str}")
        return ib_device_str


def _encode_session_id(raw_session_id: bytes) -> str:
    return base64.urlsafe_b64encode(raw_session_id).decode("ascii")


def _decode_session_id(session_id: str) -> bytes:
    # Preferred format: "<base64_raw_session>:<rpc_port>"
    # Keep backward compatibility with old "<base64_raw_session>" format.
    encoded = session_id
    if ":" in session_id:
        encoded = session_id.rsplit(":", 1)[0]
    return base64.urlsafe_b64decode(encoded.encode("ascii"))


class MooncakeTransferEngine:
    def __init__(self, hostname: str, gpu_id: int, ib_device: str | None = None):
        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = get_ib_devices_for_gpu(ib_device, gpu_id)

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        raw_session_id = self.engine.get_session_id()
        raw_session_id = (
            raw_session_id
            if isinstance(raw_session_id, (bytes, bytearray))
            else bytes(raw_session_id)
        )
        if not raw_session_id:
            raise RuntimeError("PegaFlow Transfer Engine get_session_id failed.")
        rpc_port = self.engine.get_rpc_port()
        if rpc_port <= 0:
            raise RuntimeError(
                f"PegaFlow Transfer Engine get_rpc_port failed, got {rpc_port}."
            )
        self.session_id = f"{_encode_session_id(bytes(raw_session_id))}:{rpc_port}"

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            ret_value = -1

        if ret_value != 0:
            logger.debug("PegaFlow memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            ret_value = -1

        if ret_value != 0:
            logger.debug("PegaFlow memory deregistration %s failed.", ptr)

    def batch_register(self, ptrs: list[int], lengths: list[int]) -> int:
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            ret_value = -1

        if ret_value != 0:
            logger.debug("PegaFlow batch memory registration failed.")
        return ret_value

    def initialize(
        self,
        hostname: str,
        device_name: str | None,
    ) -> None:
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            if npu_phy_id == -1:
                hostname += f":{get_free_port()}:npu_{self.gpu_id}"
            else:
                hostname += f":{get_free_port()}:npu_{npu_phy_id}"
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name if device_name is not None else "",
            )
        else:
            rdma_hostname = _ensure_hostname_with_port(hostname)
            ret_value = self.engine.initialize(
                rdma_hostname,
                "P2PHANDSHAKE",
                "rdma",
                device_name if device_name is not None else "",
            )
        if ret_value != 0:
            logger.error("PegaFlow Transfer Engine initialization failed.")
            raise RuntimeError("PegaFlow Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        try:
            raw_session_id = _decode_session_id(session_id)
            ret = self.engine.transfer_sync_write(
                raw_session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            ret = -1

        if ret < 0:
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )
            return ret

        # Keep Mooncake ABI semantics for sglang: 0 means success, <0 means failure.
        # pegaflow-transfer returns bytes on success; map any non-negative value to 0.
        return 0

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> int:
        try:
            raw_session_id = _decode_session_id(session_id)
            ret = self.engine.batch_transfer_sync_write(
                raw_session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
            return ret

        # Keep Mooncake ABI semantics for sglang: 0 means success, <0 means failure.
        # pegaflow-transfer returns bytes on success; map any non-negative value to 0.
        return 0

    def get_session_id(self):
        return self.session_id
