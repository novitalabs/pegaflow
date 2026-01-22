"""Logging utilities for PegaFlow connector.

This module provides logger configuration.
"""

import logging


def get_connector_logger() -> logging.Logger:
    """Get a logger for the connector module."""
    connector_logger = logging.getLogger("pegaflow.connector")
    connector_logger.setLevel(logging.INFO)
    if not connector_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter("%(message)s"))
        connector_logger.addHandler(handler)
        connector_logger.propagate = False
    return connector_logger


__all__ = ["get_connector_logger"]
