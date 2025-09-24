"""Enhanced database operations with Azure PostgreSQL support."""

from __future__ import annotations

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
import structlog

from ..manager.models import (
    StorageRecord, AuditRecord, TenantConfig, ModelMetric, DetectorExecution,
    DatabaseConnectionError, DatabaseOperationError
)
from ..tenant_isolation import TenantContext
from .azure_config import AzureDatabaseConnectionManager, DatabaseErrorHandler
from .migrations import DatabaseMigrationManager

logger = structlog.get_logger(__name__)


class EnhancedStorageDatabaseMixin:
    """Enhanced database operations with Azure PostgreSQL support and production features."""

    # Required attributes from the storage manager
    connection_manager: AzureDatabaseConnectionManager
    error_handler: DatabaseErrorHandler
    migration_manager: DatabaseMigrationManager
    tenant_manager: Any
    logger: Any
    settings: Any

    async def _store_in_database_enhanced(self, record: StorageRecord) -> None:
        """Store record with enhanced Azure PostgreSQL support."""
        
        async def operation():
            # Create source data hash for privacy compliance
            if record.source_data and not record.source_data_hash:
                record.source_data_hash = hashlib.sha256(record.source_data.encode()).hexdigest()
            
            # Enhanced insert SQL with all new fields
            insert_sql = """
                INSERT INTO storage_records (
                    id, source_data, source_data_hash, mapped_data, model_version, 
                    detector_type, confidence_score, timestamp, metadata, tenant_id, 
                    s3_key, encrypted, correlation_id, azure_region, backup_status,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """
            
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, record.tenant_id)
                
                await conn.execute(
                    insert_sql,
                    record.id,
                    record.source_data,
                    record.source_data_hash,
                    record.mapped_data,
                    record.model_version,
                    record.detector_type,
                    record.confidence_score,
                    record.timestamp,
                    json.dumps(record.metadata) if record.metadata else '{}',
                    record.tenant_id,
                    record.s3_key,
                    record.encrypted,
                    record.correlation_id,
                    record.azure_region,
                    record.backup_status,
                    record.created_at or datetime.utcnow(),
                    record.updated_at or datetime.utcnow()
                )
                
                # Create audit trail
                await self._create_audit_record(
                    conn, record.tenant_id, 'storage_records', record.id,
                    'INSERT', new_values=record.to_dict()
                )
        
        await self.error_handler.execute_with_retry(operation)

    async def _retrieve_from_database_enhanced(
        self, record_id: str, tenant_context: TenantContext
    ) -> Optional[StorageRecord]:
        """Retrieve record with enhanced field support."""
        
        async def operation():
            base_query = """
                SELECT id, source_data, source_data_hash, mapped_data, model_version, 
                       detector_type, confidence_score, timestamp, metadata, tenant_id, 
                       s3_key, encrypted, correlation_id, azure_region, backup_status,
                       created_at, updated_at
                FROM storage_records WHERE id = $1
            """

            filtered_query = self.tenant_manager.apply_tenant_filter(
                base_query, tenant_context
            )

            async with self.connection_manager.get_read_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, tenant_context.tenant_id)
                
                row = await conn.fetchrow(filtered_query, record_id)
                
                # Create audit trail for SELECT operations
                await self._create_audit_record(
                    conn, tenant_context.tenant_id, 'storage_records', record_id,
                    'SELECT'
                )
            
            if row:
                return StorageRecord(
                    id=row['id'],
                    source_data=row['source_data'],
                    source_data_hash=row['source_data_hash'],
                    mapped_data=row['mapped_data'],
                    model_version=row['model_version'],
                    detector_type=row['detector_type'],
                    confidence_score=row['confidence_score'],
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    tenant_id=row['tenant_id'],
                    s3_key=row['s3_key'],
                    encrypted=row['encrypted'],
                    correlation_id=row['correlation_id'],
                    azure_region=row['azure_region'],
                    backup_status=row['backup_status'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
        
        return await self.error_handler.execute_with_retry(operation)

    async def _update_storage_record(
        self, record_id: str, updates: Dict[str, Any], tenant_context: TenantContext
    ) -> bool:
        """Update storage record with audit trail."""
        
        async def operation():
            # Get current record for audit trail
            current_record = await self._retrieve_from_database_enhanced(record_id, tenant_context)
            if not current_record:
                return False
            
            # Build update query dynamically
            set_clauses = []
            values = []
            param_index = 1
            
            for field, value in updates.items():
                set_clauses.append(f"{field} = ${param_index}")
                values.append(value)
                param_index += 1
            
            # Always update updated_at
            set_clauses.append(f"updated_at = ${param_index}")
            values.append(datetime.utcnow())
            param_index += 1
            
            # Add WHERE clause parameters
            values.extend([record_id])
            
            update_sql = f"""
                UPDATE storage_records 
                SET {', '.join(set_clauses)}
                WHERE id = ${param_index}
            """
            
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, tenant_context.tenant_id)
                
                result = await conn.execute(update_sql, *values)
                
                if result == "UPDATE 1":
                    # Create audit trail
                    await self._create_audit_record(
                        conn, tenant_context.tenant_id, 'storage_records', record_id,
                        'UPDATE', old_values=current_record.to_dict(), new_values=updates
                    )
                    return True
                    
            return False
        
        return await self.error_handler.execute_with_retry(operation)

    async def _create_tenant_config(self, config: TenantConfig) -> bool:
        """Create tenant configuration."""
        
        async def operation():
            insert_sql = """
                INSERT INTO tenant_configs (
                    tenant_id, confidence_threshold, detector_whitelist, detector_blacklist,
                    storage_retention_days, encryption_enabled, audit_level,
                    custom_taxonomy_mappings, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            async with self.connection_manager.get_write_connection() as conn:
                await conn.execute(
                    insert_sql,
                    config.tenant_id,
                    config.confidence_threshold,
                    config.detector_whitelist,
                    config.detector_blacklist,
                    config.storage_retention_days,
                    config.encryption_enabled,
                    config.audit_level,
                    json.dumps(config.custom_taxonomy_mappings),
                    config.created_at,
                    config.updated_at
                )
                
                # Create audit trail
                await self._create_audit_record(
                    conn, config.tenant_id, 'tenant_configs', config.tenant_id,
                    'INSERT', new_values=config.to_dict()
                )
                
            return True
        
        return await self.error_handler.execute_with_retry(operation)

    async def _get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        
        async def operation():
            async with self.connection_manager.get_read_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM tenant_configs WHERE tenant_id = $1",
                    tenant_id
                )
                
            if row:
                return TenantConfig(
                    tenant_id=row['tenant_id'],
                    confidence_threshold=row['confidence_threshold'],
                    detector_whitelist=row['detector_whitelist'],
                    detector_blacklist=row['detector_blacklist'],
                    storage_retention_days=row['storage_retention_days'],
                    encryption_enabled=row['encryption_enabled'],
                    audit_level=row['audit_level'],
                    custom_taxonomy_mappings=json.loads(row['custom_taxonomy_mappings']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
        
        return await self.error_handler.execute_with_retry(operation)

    async def _store_model_metric(self, metric: ModelMetric) -> bool:
        """Store model performance metric."""
        
        async def operation():
            insert_sql = """
                INSERT INTO model_metrics (
                    id, model_version, tenant_id, metric_type, metric_value, timestamp, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, metric.tenant_id)
                
                await conn.execute(
                    insert_sql,
                    metric.id,
                    metric.model_version,
                    metric.tenant_id,
                    metric.metric_type,
                    metric.metric_value,
                    metric.timestamp,
                    json.dumps(metric.metadata)
                )
                
            return True
        
        return await self.error_handler.execute_with_retry(operation)

    async def _store_detector_execution(self, execution: DetectorExecution) -> bool:
        """Store detector execution log."""
        
        async def operation():
            insert_sql = """
                INSERT INTO detector_executions (
                    id, tenant_id, detector_type, execution_time_ms, confidence_score,
                    success, error_message, timestamp, correlation_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, execution.tenant_id)
                
                await conn.execute(
                    insert_sql,
                    execution.id,
                    execution.tenant_id,
                    execution.detector_type,
                    execution.execution_time_ms,
                    execution.confidence_score,
                    execution.success,
                    execution.error_message,
                    execution.timestamp,
                    execution.correlation_id
                )
                
            return True
        
        return await self.error_handler.execute_with_retry(operation)

    async def _create_audit_record(
        self,
        conn: asyncpg.Connection,
        tenant_id: str,
        table_name: str,
        record_id: str,
        operation: str,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Create audit trail record."""
        
        insert_sql = """
            INSERT INTO audit_trail (
                tenant_id, table_name, record_id, operation, user_id, timestamp,
                old_values, new_values, correlation_id, ip_address, user_agent
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        try:
            await conn.execute(
                insert_sql,
                tenant_id,
                table_name,
                record_id,
                operation,
                user_id,
                datetime.utcnow(),
                json.dumps(old_values) if old_values else None,
                json.dumps(new_values) if new_values else None,
                correlation_id,
                ip_address,
                user_agent
            )
        except Exception as e:
            # Don't fail the main operation if audit logging fails
            logger.warning("Failed to create audit record", error=str(e))

    async def _set_tenant_context(self, conn: asyncpg.Connection, tenant_id: str) -> None:
        """Set tenant context for RLS policies."""
        try:
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
        except Exception as e:
            logger.warning("Failed to set tenant context", tenant_id=tenant_id, error=str(e))

    async def _get_performance_metrics(
        self, tenant_id: str, time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get database performance metrics."""
        
        async def operation():
            async with self.connection_manager.get_read_connection() as conn:
                # Query performance metrics
                metrics = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_records,
                        AVG(confidence_score) as avg_confidence,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY confidence_score) as p95_confidence,
                        COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_count
                    FROM storage_records 
                    WHERE tenant_id = $1 
                    AND timestamp >= NOW() - INTERVAL '%s hours'
                """, tenant_id, time_range_hours)
                
                # Query detector performance
                detector_metrics = await conn.fetch("""
                    SELECT 
                        detector_type,
                        COUNT(*) as execution_count,
                        AVG(execution_time_ms) as avg_execution_time,
                        COUNT(CASE WHEN success THEN 1 END)::float / COUNT(*) as success_rate
                    FROM detector_executions 
                    WHERE tenant_id = $1 
                    AND timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY detector_type
                """, tenant_id, time_range_hours)
                
                return {
                    "overall_metrics": dict(metrics) if metrics else {},
                    "detector_metrics": [dict(row) for row in detector_metrics],
                    "time_range_hours": time_range_hours
                }
        
        return await self.error_handler.execute_with_retry(operation)

    async def _cleanup_expired_records(self, tenant_id: str, retention_days: int) -> int:
        """Clean up expired records based on tenant retention policy."""
        
        async def operation():
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - retention_days)
            
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context for RLS
                await self._set_tenant_context(conn, tenant_id)
                
                # Delete expired records
                result = await conn.execute(
                    "DELETE FROM storage_records WHERE tenant_id = $1 AND timestamp < $2",
                    tenant_id, cutoff_date
                )
                
                # Parse result to get count
                try:
                    deleted_count = int(result.split()[-1])
                except (ValueError, AttributeError):
                    deleted_count = 0
                
                if deleted_count > 0:
                    logger.info(
                        "Cleaned up expired records",
                        tenant_id=tenant_id,
                        deleted_count=deleted_count,
                        cutoff_date=cutoff_date.isoformat()
                    )
                
                return deleted_count
        
        return await self.error_handler.execute_with_retry(operation)

    async def _create_database_tables_enhanced(self) -> None:
        """Create all enhanced database tables using migrations."""
        try:
            # Initialize migration tracking
            await self.migration_manager.initialize_migration_tracking()
            
            # Apply all migrations
            results = await self.migration_manager.apply_all_migrations()
            
            logger.info(
                "Enhanced database tables created",
                applied=len(results['applied']),
                failed=len(results['failed']),
                skipped=len(results['skipped'])
            )
            
        except Exception as e:
            logger.error("Failed to create enhanced database tables", error=str(e))
            raise

    async def _get_database_health(self) -> Dict[str, Any]:
        """Get comprehensive database health status."""
        health_status = {}
        
        try:
            # Connection health
            connection_health = await self.connection_manager.health_check()
            health_status['connections'] = connection_health
            
            # Azure connection info
            azure_info = await self.connection_manager.get_azure_connection_info()
            health_status['azure_info'] = azure_info
            
            # Schema validation
            schema_status = await self.migration_manager.validate_schema_integrity()
            health_status['schema'] = schema_status
            
            # Migration status
            migration_status = await self.migration_manager.get_migration_status()
            health_status['migrations'] = migration_status
            
            # Overall health
            all_connections_healthy = all(connection_health.values())
            schema_valid = schema_status['schema_valid']
            no_failed_migrations = len(migration_status.get('failed_migrations', [])) == 0
            
            health_status['overall_healthy'] = all_connections_healthy and schema_valid and no_failed_migrations
            
        except Exception as e:
            logger.error("Failed to get database health", error=str(e))
            health_status['error'] = str(e)
            health_status['overall_healthy'] = False
        
        return health_status
