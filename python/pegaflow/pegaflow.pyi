"""Private native PegaFlow extension bindings."""

from typing import Any

__version__: str

class PegaFlowError(Exception): ...
class PegaFlowServiceError(PegaFlowError): ...
class PegaFlowBusinessError(PegaFlowError): ...

class _NativeEngineClient:
    def __init__(self, endpoint: str | None = None) -> None: ...
    def endpoint(self) -> str: ...
    def health(self) -> tuple[bool, str]: ...
    def register_context_batch(
        self,
        instance_id: str,
        namespace: str,
        tp_rank: int,
        tp_size: int,
        world_size: int,
        device_id: int,
        num_layers: int,
        layer_names: list[str],
        wrapper_bytes_list: list[bytes],
        num_blocks_list: list[int],
        bytes_per_block_list: list[int],
        kv_stride_bytes_list: list[int],
        segments_list: list[int],
    ) -> tuple[bool, str]: ...
    def save(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        saves: list[tuple[str, list[int], list[bytes]]],
    ) -> tuple[bool, str]: ...
    def load(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        load_state_shm: str,
        layer_names: list[str],
        items: list[tuple[int, list[int]]],
    ) -> tuple[bool, str]: ...
    def prepare_load(
        self,
        instance_id: str,
        request_id: str,
        block_hashes: list[bytes],
        num_prompt_tokens: int,
        num_computed_tokens: int,
        virtual_block_size: int,
        prepare_state_shm: str,
        decode_request_id: str | None = None,
        decode_expected_writes: int = 0,
    ) -> tuple[bool, str]: ...
    def get_pd_receive_descriptor(
        self,
        dst_instance_id: str,
        request_id: str,
        receive_rank: int = -1,
        handle: str | None = None,
    ) -> dict[str, Any]: ...
    def unregister_context(self, instance_id: str) -> tuple[bool, str]: ...
    def shutdown(self) -> tuple[bool, str]: ...
    def start_session_watcher(
        self,
        instance_id: str,
        namespace: str,
        tp_size: int,
        world_size: int,
    ) -> None: ...

class _NativeLoadState:
    def __init__(self) -> None: ...
    def shm_name(self) -> str: ...
    def get_state(self) -> int: ...
    def is_ready(self) -> bool: ...

class _NativePrepareLoadState:
    def __init__(self) -> None: ...
    def shm_name(self) -> str: ...
    def snapshot(self) -> dict[str, Any]: ...

class KvEgressRuntime:
    def __init__(self, nic_names: list[str]) -> None: ...
    def _register_memory(self, ptr: int, len: int) -> None: ...
    def _unregister_memory(self, ptr: int) -> None: ...
    def _ensure_connected(
        self,
        remote_addr: str,
        requester_id: str,
        engine_client: _NativeEngineClient,
    ) -> None: ...
    def _write_registered(
        self,
        remote_addr: str,
        descs: list[tuple[int, int, int]],
    ) -> tuple[int, int]: ...
    def _preferred_nic_for_gpu(self, device_id: int) -> int | None: ...
    def _write_registered_on_nic(
        self,
        remote_addr: str,
        descs: list[tuple[int, int, int]],
        nic_idx: int,
    ) -> tuple[int, int]: ...
    def _write_imm(self, remote_addr: str, imm_data: int) -> tuple[int, int]: ...
