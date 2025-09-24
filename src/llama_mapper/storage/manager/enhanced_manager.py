"""Enhanced storage manager with Azure Database support and production features."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from ...config.settings import Settings as GlobalSettings
from ...config.settings import StorageConfig
from ..database.azure_config import AzureDatabaseConfig, AzureDatabaseConnectionManager, DatabaseErrorHandler
from ..database.enhanced_database import EnhancedStorageDatabaseMixin
from ..database.migrations import DatabaseMigrationManager, create_production_migrations
from ..manager.models import (
    StorageRecord, AuditRecord, TenantConfig, ModelMetric, DetectorExecution,
    StorageBackend
)
from ..monitoring.azure_monitor import AzureDatabaseMonitor, DatabasePerformanceAnalyzer
from ..privacy_logger import PrivacyLogger
from ..security.encryption import FieldEncryption, EnhancedRowLevelSecurity
from ..tenant_isolation import TenantIsolationManager
from .database import StorageDatabaseMixin
from .initialization import StorageInitializationMixin
from .lifecycle import StorageLifecycleMixin
from .operations import StorageOperationsMixin
from .s3 import StorageS3Mixin

logger = structlog.get_logger(__name__)


class EnhancedStorageManager(
    StorageInitializationMixin,
    EnhancedStorageDatabaseMixin,
    StorageS3Mixin,
    StorageLifecycleMixin,
    StorageOperationsMixin,
):
    """Enhanced storage manager with Azure Database support and production features."""

    def __init__(self, settings: StorageConfig):
        self.settings: StorageConfig = settings
        self.logger = logger.bind(component="enhanced_storage_manager")

        # Azure-specific components
        self.azure_config: Optional[AzureDatabaseConfig] = None
        self.connection_manager: Optional[AzureDatabaseConnectionManager] = None
        self.error_handler: Optional[DatabaseErrorHandler] = None
        self.migration_manager: Optional[DatabaseMigrationManager] = None
        self.azure_monitor: Optional[AzureDatabaseMonitor] = None
        self.performance_analyzer: Optional[DatabasePerformanceAnalyzer] = None
        
        # Security components
        self.field_encryption: Optional[FieldEncryption] = None
        self.rls_manager: Optional[EnhancedRowLevelSecurity] = None
        
        # Legacy components (for compatibility)
        self._s3_client: Any = None
        self._kms_client: Any = None
        self._db_pool: Any = None  # Legacy pool reference
        self._clickhouse_client: Any = None
        self._fernet: Optional[Any] = None

        self.backend = StorageBackend(settings.storage_backend)

        global_settings = GlobalSettings()
        self.tenant_manager = TenantIsolationManager(global_settings)
        self.privacy_logger = PrivacyLogger(global_settings)

    async def initialize(self) -> None:
        """Initialize enhanced storage manager with Azure Database support."""
        try:
            # Initialize Azure Database components
            await self._init_azure_database()
            
            # Initialize security components
            await self._init_security()
            
            # Initialize monitoring
            await self._init_monitoring()
            
            # Initialize legacy components for compatibility
            await self._init_legacy_components()
            
            # Apply database migrations
            await self._apply_database_migrations()
            
            # Setup enhanced RLS policies
            await self._setup_enhanced_security()
            
            self.logger.info("Enhanced storage manager initialized successfully")
        except Exception as exc:
            self.logger.error("Failed to initialize enhanced storage manager", error=str(exc))
            raise

    async def _init_azure_database(self) -> None:
        """Initialize Azure Database connection manager."""
        if self.backend == StorageBackend.POSTGRESQL and self.settings.azure:
            # Create Azure Database configuration
            self.azure_config = AzureDatabaseConfig(
                subscription_id=self.settings.azure.subscription_id or "",
                resource_group=self.settings.azure.resource_group or "",
                server_name=self.settings.azure.server_name or self.settings.db_server_name or "",
                azure_db_host=self.settings.azure.azure_db_host or self.settings.db_host,
                database_name=self.settings.db_name,
                username=self.settings.db_user or "",
                password=self.settings.db_password or "",
                ssl_mode=self.settings.azure.ssl_mode,
                connection_timeout=self.settings.azure.connection_timeout,
                command_timeout=self.settings.azure.command_timeout,
                min_pool_size=self.settings.azure.min_pool_size,
                max_pool_size=self.settings.azure.max_pool_size,
                read_replica_regions=self.settings.azure.read_replica_regions,
                enable_azure_monitor=self.settings.azure.enable_azure_monitor,
                log_analytics_workspace_id=self.settings.azure.log_analytics_workspace_id,
                backup_retention_days=self.settings.azure.backup_retention_days,
                geo_redundant_backup=self.settings.azure.geo_redundant_backup
            )
            
            # Create connection manager
            self.connection_manager = AzureDatabaseConnectionManager(self.azure_config)
            await self.connection_manager.initialize_pools()
            
            # Create error handler
            self.error_handler = DatabaseErrorHandler(self.connection_manager)
            
            # Create migration manager
            self.migration_manager = DatabaseMigrationManager(self.connection_manager)
            self.migration_manager.migrations = create_production_migrations()
            
            # Set legacy pool reference for compatibility
            self._db_pool = self.connection_manager.write_pool
            
            self.logger.info("Azure Database components initialized")

    async def _init_security(self) -> None:
        """Initialize security components."""
        if self.settings.field_encryption_enabled:
            # Initialize field encryption
            self.field_encryption = FieldEncryption(
                key_vault_url=self.settings.azure.key_vault_url if self.settings.azure else None
            )
            await self.field_encryption.initialize()
            
        if self.connection_manager:
            # Initialize enhanced RLS manager
            self.rls_manager = EnhancedRowLevelSecurity(self.connection_manager)
            
        self.logger.info("Security components initialized")

    async def _init_monitoring(self) -> None:
        """Initialize monitoring components."""
        if self.connection_manager and self.azure_config:
            # Initialize Azure Database monitor
            self.azure_monitor = AzureDatabaseMonitor(self.connection_manager, self.azure_config)
            
            # Initialize performance analyzer
            self.performance_analyzer = DatabasePerformanceAnalyzer(self.connection_manager)
            
            # Start monitoring (non-blocking)
            asyncio.create_task(self.azure_monitor.start_monitoring())
            
        self.logger.info("Monitoring components initialized")

    async def _init_legacy_components(self) -> None:
        """Initialize legacy components for compatibility."""
        # Initialize S3 and other legacy components
        await self._init_s3()
        await self._init_encryption()
        await self._setup_worm_policy()

    async def _apply_database_migrations(self) -> None:
        """Apply database migrations."""
        if self.migration_manager:
            await self.migration_manager.initialize_migration_tracking()
            results = await self.migration_manager.apply_all_migrations()
            
            self.logger.info(
                "Database migrations applied",
                applied=len(results['applied']),
                failed=len(results['failed']),
                skipped=len(results['skipped'])
            )

    async def _setup_enhanced_security(self) -> None:
        """Setup enhanced security policies."""
        if self.rls_manager:
            await self.rls_manager.create_enhanced_rls_policies()
            self.logger.info("Enhanced RLS policies created")

    # Enhanced storage operations
    async def store_record_enhanced(self, record: StorageRecord) -> None:
        """Store record using enhanced database operations."""
        if hasattr(self, '_store_in_database_enhanced'):
            await self._store_in_database_enhanced(record)
        else:
            # Fallback to legacy method
            await self._store_in_database(record)

    async def retrieve_record_enhanced(
        self, record_id: str, tenant_id: str
    ) -> Optional[StorageRecord]:
        """Retrieve record using enhanced database operations."""
        from ..tenant_isolation import TenantContext
        
        tenant_context = TenantContext(tenant_id=tenant_id)
        
        if hasattr(self, '_retrieve_from_database_enhanced'):
            return await self._retrieve_from_database_enhanced(record_id, tenant_context)
        else:
            # Fallback to legacy method
            return await self._retrieve_from_database(record_id, tenant_context)

    async def update_record(
        self, record_id: str, updates: Dict[str, Any], tenant_id: str
    ) -> bool:
        """Update storage record with audit trail."""
        from ..tenant_isolation import TenantContext
        
        tenant_context = TenantContext(tenant_id=tenant_id)
        
        if hasattr(self, '_update_storage_record'):
            return await self._update_storage_record(record_id, updates, tenant_context)
        else:
            self.logger.warning("Record update not supported in legacy mode")
            return False

    # Tenant configuration management
    async def create_tenant_config(self, config: TenantConfig) -> bool:
        """Create tenant configuration."""
        if hasattr(self, '_create_tenant_config'):
            return await self._create_tenant_config(config)
        else:
            self.logger.warning("Tenant config creation not supported in legacy mode")
            return False

    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        if hasattr(self, '_get_tenant_config'):
            return await self._get_tenant_config(tenant_id)
        else:
            self.logger.warning("Tenant config retrieval not supported in legacy mode")
            return None

    # Performance monitoring
    async def get_performance_metrics(
        self, tenant_id: str, time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get database performance metrics."""
        if hasattr(self, '_get_performance_metrics'):
            return await self._get_performance_metrics(tenant_id, time_range_hours)
        else:
            return {"error": "Performance metrics not supported in legacy mode"}

    async def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze database performance."""
        if self.performance_analyzer:
            query_analysis = await self.performance_analyzer.analyze_query_performance(hours)
            table_analysis = await self.performance_analyzer.analyze_table_growth()
            
            return {
                "query_performance": query_analysis,
                "table_growth": table_analysis,
                "analysis_timestamp": asyncio.get_event_loop().time()
            }
        else:
            return {"error": "Performance analyzer not available"}

    # Security operations
    async def encrypt_sensitive_data(
        self, data: Dict[str, Any], sensitive_fields: List[str]
    ) -> Dict[str, Any]:
        """Encrypt sensitive fields in data."""
        if self.field_encryption:
            return self.field_encryption.encrypt_dict(data, sensitive_fields)
        else:
            self.logger.warning("Field encryption not available")
            return data

    async def decrypt_sensitive_data(
        self, encrypted_data: Dict[str, Any], sensitive_fields: List[str]
    ) -> Dict[str, Any]:
        """Decrypt sensitive fields in data."""
        if self.field_encryption:
            return self.field_encryption.decrypt_dict(encrypted_data, sensitive_fields)
        else:
            self.logger.warning("Field encryption not available")
            return encrypted_data

    async def validate_tenant_isolation(self, tenant_id: str) -> Dict[str, Any]:
        """Validate tenant isolation is working correctly."""
        if self.rls_manager:
            return await self.rls_manager.validate_tenant_isolation(tenant_id)
        else:
            return {"error": "RLS manager not available"}

    # Health and monitoring
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {}
        
        try:
            # Database health
            if hasattr(self, '_get_database_health'):
                health_status['database'] = await self._get_database_health()
            elif self.connection_manager:
                health_status['database'] = await self.connection_manager.health_check()
            
            # Azure monitoring health
            if self.azure_monitor:
                health_status['monitoring'] = await self.azure_monitor.get_monitoring_summary()
            
            # Migration status
            if self.migration_manager:
                health_status['migrations'] = await self.migration_manager.get_migration_status()
            
            # Overall health calculation
            db_healthy = health_status.get('database', {}).get('overall_healthy', False)
            monitoring_healthy = health_status.get('monitoring', {}).get('monitoring_active', False)
            migrations_healthy = len(health_status.get('migrations', {}).get('failed_migrations', [])) == 0
            
            health_status['overall_healthy'] = db_healthy and migrations_healthy
            
        except Exception as e:
            self.logger.error("Failed to get health status", error=str(e))
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False
        
        return health_status

    async def cleanup_expired_data(self, tenant_id: str) -> Dict[str, Any]:
        """Clean up expired data for tenant."""
        results = {"tenant_id": tenant_id}
        
        try:
            # Get tenant config for retention policy
            tenant_config = await self.get_tenant_config(tenant_id)
            retention_days = tenant_config.storage_retention_days if tenant_config else self.settings.retention_days
            
            # Cleanup database records
            if hasattr(self, '_cleanup_expired_records'):
                deleted_count = await self._cleanup_expired_records(tenant_id, retention_days)
                results['deleted_records'] = deleted_count
            else:
                # Fallback to legacy cleanup
                from datetime import datetime, timedelta
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                deleted_count = await self._cleanup_postgresql(cutoff_date)
                results['deleted_records'] = deleted_count
            
            results['success'] = True
            
        except Exception as e:
            self.logger.error("Failed to cleanup expired data", tenant_id=tenant_id, error=str(e))
            results['error'] = str(e)
            results['success'] = False
        
        return results

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            # Stop monitoring
            if self.azure_monitor:
                await self.azure_monitor.stop_monitoring()
            
            # Close Azure Database connections
            if self.connection_manager:
                await self.connection_manager.close_pools()
            
            # Close legacy connections
            if hasattr(self, '_close_legacy_connections'):
                await self._close_legacy_connections()
            
            self.logger.info("Enhanced storage manager closed")
            
        except Exception as e:
            self.logger.error("Error closing enhanced storage manager", error=str(e))

    # Compatibility methods for legacy StorageDatabaseMixin
    async def _store_in_database(self, record: StorageRecord) -> None:
        """Legacy database storage method for compatibility."""
        if self.backend == StorageBackend.POSTGRESQL:
            await self._store_postgresql_legacy(record)
        else:
            await self._store_clickhouse(record)

    async def _store_postgresql_legacy(self, record: StorageRecord) -> None:
        """Legacy PostgreSQL storage method."""
        if self.connection_manager:
            # Use enhanced storage
            await self.store_record_enhanced(record)
        else:
            # Use original implementation
            from .database import StorageDatabaseMixin
            await StorageDatabaseMixin._store_postgresql(self, record)


__all__ = ["EnhancedStorageManager"]
