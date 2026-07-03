# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Backward-compatible re-export of NixlPullConnectorScheduler."""

from pegaflow.nixl_connector.pull_scheduler import (
    NixlPullConnectorScheduler,
)

# Backward compatibility: NixlConnectorScheduler is the pull-based scheduler.
NixlConnectorScheduler = NixlPullConnectorScheduler

__all__ = ["NixlConnectorScheduler", "NixlPullConnectorScheduler"]
