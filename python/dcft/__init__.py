"""Unified DCFT simulation package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dcft-simulation")
except PackageNotFoundError:  # Source-tree imports before installation.
    __version__ = "0.1.0"

SCHEMA_VERSION = "DCFT_PARQUET_V1"
QUALIFICATION_STATUS = "cluster-qualified"
CLUSTER_QUALIFICATION_STATUS = "complete"

__all__ = [
    "CLUSTER_QUALIFICATION_STATUS",
    "QUALIFICATION_STATUS",
    "SCHEMA_VERSION",
    "__version__",
]
