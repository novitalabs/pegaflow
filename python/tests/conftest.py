"""Pytest fixtures for PegaFlow connector integration tests.

Provides fixtures for automatically starting/stopping PegaServer
and test helpers for connector testing against a running server.
"""

import hashlib
import logging
import os
import pickle
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

# Import CudaIPCWrapper for IPC communication
try:
    from pegaflow.ipc_wrapper import CudaIPCWrapper
except ImportError:
    CudaIPCWrapper = None

logger = logging.getLogger(__name__)


# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_POOL_SIZE = "100mb"
SERVER_STARTUP_TIMEOUT = 60  # seconds (increased for slow GPU init)
SERVER_READY_CHECK_INTERVAL = 1.0  # seconds


# =============================================================================
# Utility Functions
# =============================================================================


def find_available_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_server_binary() -> str | None:
    """
    Locate the pegaflow-server binary.

    Search order:
    1. Installed pegaflow-server-py in package directory
    2. cargo target/release/pegaflow-server
    3. cargo target/debug/pegaflow-server
    """
    # 1. Check installed package binary
    try:
        from pegaflow._server import get_server_binary

        binary = get_server_binary()
        if Path(binary).exists():
            return binary
    except ImportError:
        pass

    # 2. Check cargo build outputs
    project_root = Path(__file__).parent.parent.parent  # python/tests -> pegaflow
    for build_type in ["release", "debug"]:
        cargo_binary = project_root / "target" / build_type / "pegaflow-server"
        if cargo_binary.exists():
            return str(cargo_binary)

    return None


def wait_for_server_ready(endpoint: str, timeout: float = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for server to become ready by attempting connections."""
    # Import directly from submodule to avoid triggering __init__.py imports (vllm dependency)
    import importlib

    pegaflow_module = importlib.import_module("pegaflow.pegaflow")
    EngineRpcClient = pegaflow_module.EngineRpcClient

    start_time = time.time()
    last_error = None
    while time.time() - start_time < timeout:
        try:
            client = EngineRpcClient(endpoint)
            ok, _ = client.health()
            if ok:
                return True
        except Exception as e:
            last_error = str(e)
        time.sleep(SERVER_READY_CHECK_INTERVAL)

    if last_error:
        print(f"Last health check error: {last_error}", file=sys.stderr)
    return False


# =============================================================================
# Test Helpers: ClientContext
# =============================================================================


def initialize_kv_cache(
    device: torch.device,
    num_blocks: int = 64,
    num_layers: int = 1,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """
    Initialize KV cache tensors on GPU for testing.

    Creates tensors in KV-first layout: (2, num_blocks, block_size, num_heads, head_size)
    where the first dimension is [K, V].
    """
    torch.random.manual_seed(42)

    gpu_tensors = [
        torch.rand(
            (2, num_blocks, block_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    return gpu_tensors


class ClientContext:
    """
    Client context that represents a vLLM instance.

    This class abstracts a vLLM instance by managing:
    - GPU KV cache tensors (like WorkerConnector)
    - Query operations (like SchedulerConnector)
    - Context registration/unregistration
    """

    def __init__(
        self,
        engine_client,
        instance_id: str,
        namespace: str,
        device_id: int = 0,
        num_blocks: int = 64,
        num_layers: int = 1,
        block_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"device_id {device_id} >= available GPUs ({torch.cuda.device_count()})"
            )

        self.engine_client = engine_client
        self.instance_id = instance_id
        self.namespace = namespace
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        # Initialize KV cache tensors
        self.gpu_kv_caches = initialize_kv_cache(
            self.device, num_blocks, num_layers, block_size, num_heads, head_size, dtype
        )

        # Map layer index to layer name (for compatibility with vLLM)
        self._layer_names = [f"layer_{i}" for i in range(num_layers)]
        self._layer_name_to_id = {name: i for i, name in enumerate(self._layer_names)}
        self._registered = False

    def register_kv_caches(self) -> None:
        """Register KV cache tensors with the engine server (like WorkerConnector.register_kv_caches)."""
        if CudaIPCWrapper is None:
            raise RuntimeError("CudaIPCWrapper not available")

        if self._registered:
            return

        kv_caches = {name: self.gpu_kv_caches[i] for i, name in enumerate(self._layer_names)}

        for layer_name, kv_cache in kv_caches.items():
            if not kv_cache.is_contiguous():
                kv_cache = kv_cache.contiguous()
            if kv_cache.storage_offset() != 0:
                raise ValueError(f"KV cache for {layer_name} must have zero storage offset")

            wrapper = CudaIPCWrapper(kv_cache)
            wrapper_bytes = pickle.dumps(wrapper)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            if len(shape) >= 2 and shape[0] == 2:
                num_blocks = shape[1]
                bytes_per_block = stride[1] * element_size
                kv_stride_bytes = stride[0] * element_size
                segments = 2
            else:
                num_blocks = shape[0]
                bytes_per_block = stride[0] * element_size
                kv_stride_bytes = 0
                segments = 1

            ok, message = self.engine_client.register_context(
                self.instance_id,
                self.namespace,
                0,  # tp_rank
                1,  # tp_size
                self.device_id,
                self.num_layers,
                layer_name,
                wrapper_bytes,
                num_blocks,
                bytes_per_block,
                kv_stride_bytes,
                segments,
            )

            if not ok:
                raise RuntimeError(f"Register context failed for {layer_name}: {message}")

        self._registered = True

    def unregister_context(self) -> None:
        """Unregister context from server (like WorkerConnector.unregister_context)."""
        if not self._registered:
            return

        try:
            ok, message = self.engine_client.unregister_context(self.instance_id)
            if not ok:
                logger.warning(f"Unregister context failed: {message}")
        except Exception as e:
            logger.warning(f"Unregister context exception: {e}")

        self._registered = False

    def query(self, block_hashes: list[bytes]) -> dict | tuple:
        """Query available blocks (like SchedulerConnector._count_available_block_prefix).

        Args:
            block_hashes: List of block hashes to query

        Returns:
            Query result (dict or tuple format)
        """
        return self.engine_client.query(self.instance_id, block_hashes)

    def get_kv_cache(self, layer: int = 0) -> torch.Tensor:
        """Get KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer]

    def get_tensor_slice(self, layer: int, start_block: int, num_blocks: int) -> torch.Tensor:
        """Get a slice of the KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer][:, start_block : start_block + num_blocks]


# =============================================================================
# Fixtures: Test Data
# =============================================================================


@pytest.fixture
def instance_id() -> str:
    """Generate a unique instance ID for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def namespace() -> str:
    """Generate a test namespace."""
    return "test_namespace"


@pytest.fixture
def block_hashes() -> list[bytes]:
    """Generate deterministic block hashes for testing."""
    hashes = []
    for i in range(20):
        content = f"test_block_{i}".encode()
        hash_bytes = hashlib.sha256(content).digest()
        hashes.append(hash_bytes)
    return hashes


# =============================================================================
# Fixtures: Server Management
# =============================================================================


class PegaServerProcess:
    """Manages a PegaServer subprocess for testing."""

    def __init__(self, port: int, pool_size: str = DEFAULT_POOL_SIZE):
        self.port = port
        self.pool_size = pool_size
        self.endpoint = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen | None = None
        self._binary_path = find_server_binary()

    def start(self) -> bool:
        """Start the server process. Returns True if successful."""
        if not self._binary_path:
            return False

        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable

        # Add libpython to LD_LIBRARY_PATH if available
        if libdir := sysconfig.get_config_var("LIBDIR"):
            env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"

        # Set PYTHONPATH to include python package and venv site-packages
        python_dir = Path(__file__).parent.parent
        site_packages = next((p for p in sys.path if "site-packages" in p), None)
        env["PYTHONPATH"] = f"{python_dir}" + (f":{site_packages}" if site_packages else "")

        cmd = [
            self._binary_path,
            "--addr",
            f"127.0.0.1:{self.port}",
            "--pool-size",
            self.pool_size,
            "--device",
            "0",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/tmp",
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
        except (FileNotFoundError, PermissionError):
            return False

        return wait_for_server_ready(self.endpoint)

    def stop(self) -> None:
        """Stop the server process."""
        if self.process is None:
            return
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            if self.process:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=2)
        finally:
            self.process = None

    def is_running(self) -> bool:
        """Check if server process is still running."""
        return self.process is not None and self.process.poll() is None


@pytest.fixture(scope="session")
def pega_server() -> Generator[PegaServerProcess, None, None]:
    """Session-scoped fixture that starts a PegaServer for integration tests."""
    port = find_available_port()
    server = PegaServerProcess(port=port)

    if not server.start() or not server._binary_path:
        pytest.skip("PegaServer binary not found or failed to start")

    yield server
    server.stop()


@pytest.fixture
def server_endpoint(pega_server: PegaServerProcess) -> str:
    """Return the endpoint URL of the running test server."""
    return pega_server.endpoint


@pytest.fixture
def engine_client(pega_server: PegaServerProcess):
    """Create an EngineRpcClient connected to the test server."""
    # Import directly from submodule to avoid triggering __init__.py imports (vllm dependency)
    import importlib

    pegaflow_module = importlib.import_module("pegaflow.pegaflow")
    EngineRpcClient = pegaflow_module.EngineRpcClient

    return EngineRpcClient(pega_server.endpoint)


@pytest.fixture
def client_context(
    engine_client, instance_id: str, namespace: str
) -> Generator[ClientContext, None, None]:
    """Fixture that provides a ClientContext representing a vLLM instance.

    Args:
        engine_client: EngineRpcClient connected to server
        instance_id: Unique instance identifier
        namespace: Namespace for the instance

    Returns:
        ClientContext instance (automatically registered)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    ctx = ClientContext(
        engine_client=engine_client,
        instance_id=instance_id,
        namespace=namespace,
        device_id=0,
    )

    # Auto-register on creation
    ctx.register_kv_caches()

    yield ctx

    # Cleanup
    ctx.unregister_context()
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture
def registered_instance(client_context: ClientContext) -> Generator[str, None, None]:
    """Fixture that returns the instance_id of a registered ClientContext.

    The client_context fixture already handles registration/unregistration.
    This fixture just provides the instance_id for convenience.
    """
    yield client_context.instance_id


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options for E2E tests."""
    parser.addoption(
        "--model",
        action="store",
        default="Qwen/Qwen3-0.6B",
        help="Model to use for E2E testing",
    )
    parser.addoption(
        "--e2e-port",
        action="store",
        default=8100,
        type=int,
        help="Base port for vLLM servers in E2E tests",
    )
    parser.addoption(
        "--pega-metrics-port",
        action="store",
        default=9091,
        type=int,
        help="PegaFlow server metrics port for E2E tests",
    )
    parser.addoption(
        "--tensor-parallel-size",
        action="store",
        default=1,
        type=int,
        help="Tensor parallel size for vLLM servers in E2E/Fuzz tests (tp * pp <= 4)",
    )
    parser.addoption(
        "--pipeline-parallel-size",
        action="store",
        default=1,
        type=int,
        help="Pipeline parallel size for vLLM servers in E2E/Fuzz tests (tp * pp <= 4)",
    )
    # Fuzz test options
    parser.addoption(
        "--fuzz-seed",
        action="store",
        default=42,
        type=int,
        help="Random seed for fuzz test reproducibility",
    )
    parser.addoption(
        "--fuzz-corpus",
        action="store",
        default=500,
        type=int,
        help="Number of unique prompts to sample from ShareGPT",
    )
    parser.addoption(
        "--fuzz-requests",
        action="store",
        default=1000,
        type=int,
        help="Number of requests to generate in fuzz test",
    )


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require PegaServer with GPU)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (require vLLM + PegaFlow)",
    )
    config.addinivalue_line(
        "markers",
        "fuzz: marks tests as fuzz tests (long-running, skipped by default)",
    )
