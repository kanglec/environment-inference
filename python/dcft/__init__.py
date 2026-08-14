"""Unified DCFT simulation package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dcft-simulation")
except PackageNotFoundError:  # Source-tree imports before installation.
    __version__ = "0.1.0"

SCHEMA_VERSION = "DCFT_PARQUET_V1"

__all__ = [
    "SCHEMA_VERSION",
    "__version__",
]
