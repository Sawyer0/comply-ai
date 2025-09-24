"""Composite storage manager that coordinates persistence across backends."""

from __future__ import annotations

from typing import Any, Optional

import structlog

from ...config.settings import Settings as GlobalSettings
from ...config.settings import StorageConfig
from ..privacy_logger import PrivacyLogger
from ..tenant_isolation import TenantIsolationManager
from .database import StorageDatabaseMixin
from .initialization import StorageInitializationMixin
from .lifecycle import StorageLifecycleMixin
from .models import StorageBackend, StorageRecord
from .operations import StorageOperationsMixin
from .s3 import StorageS3Mixin

logger = structlog.get_logger(__name__)


class StorageManager(
    StorageInitializationMixin,
    StorageDatabaseMixin,
    StorageS3Mixin,
    StorageLifecycleMixin,
    StorageOperationsMixin,
):
    """Manages storage operations across S3 and database backends."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, settings: StorageConfig):
        self.settings: StorageConfig = settings
        self.logger = logger.bind(component="storage_manager")

        self._s3_client: Any = None
        self._kms_client: Any = None
        self._db_pool: Any = None
        self._clickhouse_client: Any = None
        self._fernet: Optional[Any] = None

        self.backend = StorageBackend(settings.storage_backend)

        global_settings = GlobalSettings()
        self.tenant_manager = TenantIsolationManager(global_settings)
        self.privacy_logger = PrivacyLogger(global_settings)

    async def initialize(self) -> None:
        """Initialize all storage connections and configurations."""
        try:
            await self._init_s3()
            await self._init_database()
            await self._init_encryption()
            await self._setup_worm_policy()
            self.logger.info("Storage manager initialized successfully")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to initialize storage manager", error=str(exc))
            raise


__all__ = ["StorageManager", "StorageRecord", "StorageBackend"]
