# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Backward-compatible re-export of NixlPullConnectorWorker."""

from pegaflow.nixl_connector.pull_worker import (
    NixlPullConnectorWorker,
)

# Backward compatibility: NixlConnectorWorker is the pull-based worker.
NixlConnectorWorker = NixlPullConnectorWorker


__all__ = ["NixlConnectorWorker", "NixlPullConnectorWorker"]
