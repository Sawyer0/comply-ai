"""Lifecycle utilities for the storage manager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import StorageBackend


class StorageLifecycleMixin:
    """Adds cleanup and shutdown helpers."""

    # pylint: disable=too-few-public-methods

    backend: StorageBackend
    logger: Any
    _db_pool: Any

    async def cleanup_expired_records(self) -> int:
        """Remove records older than 90 days from hot storage."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)

            if self.backend == StorageBackend.POSTGRESQL:
                count = await self._cleanup_postgresql(cutoff_date)
            else:
                count = await self._cleanup_clickhouse(cutoff_date)

            self.logger.info("Cleanup completed", records_removed=count)
            return count

        except Exception as exc:  # pragma: no cover - defensive logging  # pylint: disable=broad-exception-caught
            self.logger.error("Cleanup failed", error=str(exc))
            raise

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            if self._db_pool:
                await self._db_pool.close()

            self.logger.info("Storage manager closed successfully")

        except Exception as exc:  # pragma: no cover - defensive logging  # pylint: disable=broad-exception-caught
            self.logger.error("Error closing storage manager", error=str(exc))


__all__ = ["StorageLifecycleMixin"]
