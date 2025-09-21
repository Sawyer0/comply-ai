"""Storage manager package exposing high-level persistence APIs."""

from .manager import StorageManager
from .models import StorageBackend, StorageRecord

__all__ = ["StorageManager", "StorageBackend", "StorageRecord"]
