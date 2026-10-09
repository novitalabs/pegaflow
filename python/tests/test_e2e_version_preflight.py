"""Check E2E artifact versions before starting vLLM or PegaFlow."""

import os
import sys
import venv
from pathlib import Path

import pytest

from .vllm_helpers import preflight_pegaflow_versions


def _artifacts(
    tmp_path: Path, client_version: str, server_version: str, interpreter: str | None = None
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm = bin_dir / "vllm"
    vllm.write_text(f"#!{interpreter or sys.executable}\n", encoding="utf-8")
    vllm.chmod(0o755)

    package = tmp_path / "site" / "pegaflow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pegaflow.py").write_text(f"__version__ = {client_version!r}\n", encoding="utf-8")

    server = bin_dir / "pegaflow-server"
    server.write_text(f"#!/bin/sh\nprintf '%s\\n' 'pega-engine-server {server_version}'\n")
    server.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir), "PYTHONPATH": str(tmp_path / "site")}
    return server, env, package / "pegaflow.py"


def test_preflight_accepts_matching_versions_from_vllm_python(tmp_path: Path):
    server, env, extension = _artifacts(tmp_path, "0.24.6", "0.24.6")
    result = preflight_pegaflow_versions(server_binary=str(server), env=env)
    assert result.server_version == "0.24.6"
    assert result.client_version == "0.24.6"
    assert result.client_path == str(extension)


def test_preflight_rejects_mismatch_before_startup(tmp_path: Path):
    server, env, extension = _artifacts(tmp_path, "0.24.5", "0.24.6")
    with pytest.raises(RuntimeError, match="PegaFlow version mismatch before E2E startup") as error:
        preflight_pegaflow_versions(server_binary=str(server), env=env)
    message = str(error.value)
    assert "0.24.6" in message
    assert "0.24.5" in message
    assert str(server) in message
    assert str(extension) in message
    assert "maturin develop" in message


def test_preflight_uses_vllm_interpreter_instead_of_pytest_interpreter(tmp_path: Path):
    vllm_env = tmp_path / "vllm-env"
    venv.EnvBuilder(with_pip=False).create(vllm_env)
    server, env, extension = _artifacts(tmp_path, "0.24.5", "0.24.6", str(vllm_env / "bin/python"))
    extension.write_text(
        "import sys\n__version__ = '0.24.6' if sys.prefix.endswith('vllm-env') else '0.24.5'\n",
        encoding="utf-8",
    )
    result = preflight_pegaflow_versions(server_binary=str(server), env=env)
    assert result.client_version == "0.24.6"


def test_preflight_reports_missing_native_extension(tmp_path: Path):
    server, env, _ = _artifacts(tmp_path, "0.24.6", "0.24.6")
    (tmp_path / "site" / "pegaflow" / "pegaflow.py").unlink()
    with pytest.raises(RuntimeError, match="cannot import the pegaflow native extension"):
        preflight_pegaflow_versions(server_binary=str(server), env=env)
