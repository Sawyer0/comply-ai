"""Data structures shared by storage manager components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class StorageBackend(Enum):
    """Supported storage backends."""

    CLICKHOUSE = "clickhouse"
    POSTGRESQL = "postgresql"


@dataclass
class StorageRecord:
    """Record structure for storing mapping results."""

    # pylint: disable=too-many-instance-attributes

    id: str
    source_data: str
    mapped_data: str
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tenant_id: str = "unknown"
    s3_key: Optional[str] = None
    encrypted: bool = False


class StorageAccessError(PermissionError):
    """Raised when tenant isolation rules forbid an operation."""


__all__ = ["StorageBackend", "StorageRecord", "StorageAccessError"]
