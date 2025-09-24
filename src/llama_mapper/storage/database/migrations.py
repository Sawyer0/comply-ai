"""Database migration system for Azure PostgreSQL with versioned schema management."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from ...manager.models import DatabaseMigrationError
from .azure_config import AzureDatabaseConnectionManager

logger = structlog.get_logger(__name__)


@dataclass
class Migration:
    """Database migration definition."""
    
    version: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        
        # Calculate checksum if not provided
        if not self.checksum:
            content = f"{self.version}{self.description}{self.up_sql}{self.down_sql}"
            self.checksum = hashlib.sha256(content.encode()).hexdigest()


class DatabaseMigrationManager:
    """Manages versioned database schema migrations."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.migration_table = "schema_migrations"
        self.migrations: List[Migration] = []
        
    async def initialize_migration_tracking(self):
        """Create migration tracking table."""
        async with self.connection_manager.get_write_connection() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.migration_table} (
                    version VARCHAR(50) PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    checksum VARCHAR(64) NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    success BOOLEAN NOT NULL DEFAULT TRUE,
                    error_message TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
                ON {self.migration_table}(applied_at);
            """)
            
        logger.info("Migration tracking table initialized")
    
    def register_migration(self, migration: Migration):
        """Register a migration for execution."""
        self.migrations.append(migration)
        logger.debug("Migration registered", version=migration.version)
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration with rollback support."""
        start_time = time.time()
        
        async with self.connection_manager.get_write_connection() as conn:
            async with conn.transaction():
                try:
                    # Validate migration hasn't been applied
                    existing = await conn.fetchrow(
                        f"SELECT version FROM {self.migration_table} WHERE version = $1",
                        migration.version
                    )
                    
                    if existing:
                        logger.info("Migration already applied", version=migration.version)
                        return True
                    
                    # Validate dependencies
                    if migration.dependencies:
                        for dep in migration.dependencies:
                            dep_exists = await conn.fetchrow(
                                f"SELECT version FROM {self.migration_table} WHERE version = $1 AND success = TRUE",
                                dep
                            )
                            if not dep_exists:
                                raise DatabaseMigrationError(
                                    f"Migration {migration.version} depends on {dep} which hasn't been applied"
                                )
                    
                    # Execute migration
                    logger.info("Applying migration", version=migration.version)
                    await conn.execute(migration.up_sql)
                    
                    # Record migration
                    execution_time = int((time.time() - start_time) * 1000)
                    await conn.execute(f"""
                        INSERT INTO {self.migration_table} 
                        (version, description, checksum, execution_time_ms, success)
                        VALUES ($1, $2, $3, $4, $5)
                    """, migration.version, migration.description, 
                        migration.checksum, execution_time, True)
                    
                    logger.info(
                        "Migration applied successfully", 
                        version=migration.version, 
                        execution_time_ms=execution_time
                    )
                    return True
                    
                except Exception as e:
                    execution_time = int((time.time() - start_time) * 1000)
                    
                    # Record failed migration
                    try:
                        await conn.execute(f"""
                            INSERT INTO {self.migration_table} 
                            (version, description, checksum, execution_time_ms, success, error_message)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, migration.version, migration.description, 
                            migration.checksum, execution_time, False, str(e))
                    except Exception:
                        pass  # Don't fail on recording error
                    
                    logger.error("Migration failed", version=migration.version, error=str(e))
                    raise DatabaseMigrationError(f"Migration {migration.version} failed: {e}") from e
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a migration."""
        start_time = time.time()
        
        async with self.connection_manager.get_write_connection() as conn:
            async with conn.transaction():
                try:
                    # Check if migration was applied
                    existing = await conn.fetchrow(
                        f"SELECT version FROM {self.migration_table} WHERE version = $1 AND success = TRUE",
                        migration.version
                    )
                    
                    if not existing:
                        logger.warning("Migration not found or not successful", version=migration.version)
                        return False
                    
                    # Execute rollback
                    logger.info("Rolling back migration", version=migration.version)
                    await conn.execute(migration.down_sql)
                    
                    # Remove migration record
                    await conn.execute(
                        f"DELETE FROM {self.migration_table} WHERE version = $1",
                        migration.version
                    )
                    
                    execution_time = int((time.time() - start_time) * 1000)
                    logger.info(
                        "Migration rolled back successfully", 
                        version=migration.version,
                        execution_time_ms=execution_time
                    )
                    return True
                    
                except Exception as e:
                    logger.error("Migration rollback failed", version=migration.version, error=str(e))
                    raise DatabaseMigrationError(f"Rollback {migration.version} failed: {e}") from e
    
    async def apply_all_migrations(self) -> Dict[str, Any]:
        """Apply all registered migrations in order."""
        results = {
            "total_migrations": len(self.migrations),
            "applied": [],
            "failed": [],
            "skipped": []
        }
        
        # Sort migrations by version
        sorted_migrations = sorted(self.migrations, key=lambda m: m.version)
        
        for migration in sorted_migrations:
            try:
                success = await self.apply_migration(migration)
                if success:
                    results["applied"].append(migration.version)
                else:
                    results["skipped"].append(migration.version)
            except DatabaseMigrationError:
                results["failed"].append(migration.version)
                # Stop on first failure for safety
                break
        
        logger.info(
            "Migration batch completed",
            applied=len(results["applied"]),
            failed=len(results["failed"]),
            skipped=len(results["skipped"])
        )
        
        return results
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get status of all migrations."""
        async with self.connection_manager.get_read_connection() as conn:
            # Get applied migrations
            applied_migrations = await conn.fetch(f"""
                SELECT version, description, applied_at, execution_time_ms, success, error_message
                FROM {self.migration_table}
                ORDER BY applied_at DESC
            """)
            
            # Get pending migrations
            applied_versions = {row['version'] for row in applied_migrations if row['success']}
            registered_versions = {m.version for m in self.migrations}
            pending_versions = registered_versions - applied_versions
            
            return {
                "applied_count": len(applied_versions),
                "pending_count": len(pending_versions),
                "total_registered": len(self.migrations),
                "applied_migrations": [dict(row) for row in applied_migrations],
                "pending_migrations": list(pending_versions)
            }
    
    async def validate_schema_integrity(self) -> Dict[str, Any]:
        """Validate schema integrity and consistency."""
        async with self.connection_manager.get_read_connection() as conn:
            # Check for missing tables
            required_tables = [
                'storage_records', 'audit_trail', 'tenant_configs', 
                'model_metrics', 'detector_executions', self.migration_table
            ]
            
            existing_tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            existing_table_names = {row['table_name'] for row in existing_tables}
            
            missing_tables = set(required_tables) - existing_table_names
            
            # Check for missing indexes
            required_indexes = [
                'idx_storage_records_tenant_timestamp',
                'idx_storage_records_model_version',
                'idx_audit_trail_tenant_timestamp',
                'idx_model_metrics_version_type'
            ]
            
            existing_indexes = await conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            existing_index_names = {row['indexname'] for row in existing_indexes}
            
            missing_indexes = set(required_indexes) - existing_index_names
            
            # Check RLS policies
            rls_tables = await conn.fetch("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' AND rowsecurity = true
            """)
            rls_enabled_tables = {row['tablename'] for row in rls_tables}
            
            return {
                "schema_valid": len(missing_tables) == 0 and len(missing_indexes) == 0,
                "missing_tables": list(missing_tables),
                "missing_indexes": list(missing_indexes),
                "total_tables": len(existing_table_names),
                "total_indexes": len(existing_index_names),
                "rls_enabled_tables": list(rls_enabled_tables)
            }


def create_production_migrations() -> List[Migration]:
    """Create all production-ready migrations."""
    migrations = []
    
    # Migration 001: Enhanced storage_records table
    migrations.append(Migration(
        version="001_enhance_storage_records",
        description="Add Azure-specific fields and enhanced constraints to storage_records",
        up_sql="""
            -- Add new columns to storage_records
            ALTER TABLE storage_records 
            ADD COLUMN IF NOT EXISTS source_data_hash VARCHAR(64),
            ADD COLUMN IF NOT EXISTS detector_type VARCHAR(50),
            ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(5,4),
            ADD COLUMN IF NOT EXISTS correlation_id UUID,
            ADD COLUMN IF NOT EXISTS azure_region VARCHAR(50) DEFAULT 'eastus',
            ADD COLUMN IF NOT EXISTS backup_status VARCHAR(20) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
            
            -- Add constraints
            ALTER TABLE storage_records 
            ADD CONSTRAINT IF NOT EXISTS valid_confidence 
            CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
            ADD CONSTRAINT IF NOT EXISTS valid_backup_status 
            CHECK (backup_status IN ('pending', 'completed', 'failed')),
            ADD CONSTRAINT IF NOT EXISTS valid_tenant_id 
            CHECK (length(tenant_id) > 0);
            
            -- Add indexes for new fields
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_detector_confidence 
            ON storage_records (detector_type, confidence_score DESC) 
            WHERE confidence_score IS NOT NULL;
            
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_correlation 
            ON storage_records (correlation_id) 
            WHERE correlation_id IS NOT NULL;
            
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_azure_region 
            ON storage_records (azure_region, timestamp DESC);
        """,
        down_sql="""
            -- Remove indexes
            DROP INDEX IF EXISTS idx_storage_records_detector_confidence;
            DROP INDEX IF EXISTS idx_storage_records_correlation;
            DROP INDEX IF EXISTS idx_storage_records_azure_region;
            
            -- Remove constraints
            ALTER TABLE storage_records 
            DROP CONSTRAINT IF EXISTS valid_confidence,
            DROP CONSTRAINT IF EXISTS valid_backup_status,
            DROP CONSTRAINT IF EXISTS valid_tenant_id;
            
            -- Remove columns
            ALTER TABLE storage_records 
            DROP COLUMN IF EXISTS source_data_hash,
            DROP COLUMN IF EXISTS detector_type,
            DROP COLUMN IF EXISTS confidence_score,
            DROP COLUMN IF EXISTS correlation_id,
            DROP COLUMN IF EXISTS azure_region,
            DROP COLUMN IF EXISTS backup_status,
            DROP COLUMN IF EXISTS updated_at;
        """,
        checksum=""
    ))
    
    # Migration 002: Audit trail table
    migrations.append(Migration(
        version="002_create_audit_trail",
        description="Create audit trail table for compliance tracking",
        up_sql="""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(100) NOT NULL,
                table_name VARCHAR(100) NOT NULL,
                record_id UUID NOT NULL,
                operation VARCHAR(20) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE', 'SELECT')),
                user_id VARCHAR(100),
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                old_values JSONB,
                new_values JSONB,
                correlation_id UUID,
                ip_address INET,
                user_agent TEXT,
                
                CONSTRAINT valid_operation CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE', 'SELECT')),
                CONSTRAINT valid_tenant_id CHECK (length(tenant_id) > 0)
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_audit_trail_tenant_timestamp 
            ON audit_trail (tenant_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_audit_trail_record_operation 
            ON audit_trail (record_id, operation, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_audit_trail_correlation 
            ON audit_trail (correlation_id) 
            WHERE correlation_id IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_audit_trail_table_operation 
            ON audit_trail (table_name, operation, timestamp DESC);
            
            -- Enable RLS
            ALTER TABLE audit_trail ENABLE ROW LEVEL SECURITY;
            
            -- Create RLS policies
            CREATE POLICY IF NOT EXISTS tenant_isolation_audit ON audit_trail
                FOR ALL TO application_role
                USING (tenant_id = current_setting('app.current_tenant_id', true));
        """,
        down_sql="""
            DROP TABLE IF EXISTS audit_trail CASCADE;
        """,
        dependencies=["001_enhance_storage_records"],
        checksum=""
    ))
    
    # Migration 003: Tenant configuration table
    migrations.append(Migration(
        version="003_create_tenant_configs",
        description="Create tenant configuration table",
        up_sql="""
            CREATE TABLE IF NOT EXISTS tenant_configs (
                tenant_id VARCHAR(100) PRIMARY KEY,
                confidence_threshold DECIMAL(5,4) DEFAULT 0.6,
                detector_whitelist TEXT[],
                detector_blacklist TEXT[],
                storage_retention_days INTEGER DEFAULT 90,
                encryption_enabled BOOLEAN DEFAULT TRUE,
                audit_level VARCHAR(20) DEFAULT 'standard' CHECK (audit_level IN ('minimal', 'standard', 'verbose')),
                custom_taxonomy_mappings JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                
                CONSTRAINT valid_confidence_threshold 
                CHECK (confidence_threshold >= 0 AND confidence_threshold <= 1),
                CONSTRAINT valid_retention_days 
                CHECK (storage_retention_days > 0),
                CONSTRAINT valid_tenant_id 
                CHECK (length(tenant_id) > 0),
                CONSTRAINT valid_taxonomy_mappings 
                CHECK (jsonb_typeof(custom_taxonomy_mappings) = 'object')
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_tenant_configs_retention 
            ON tenant_configs (storage_retention_days);
            
            CREATE INDEX IF NOT EXISTS idx_tenant_configs_encryption 
            ON tenant_configs (encryption_enabled);
            
            -- Enable RLS
            ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;
            
            -- Create RLS policies
            CREATE POLICY IF NOT EXISTS tenant_isolation_configs ON tenant_configs
                FOR ALL TO application_role
                USING (tenant_id = current_setting('app.current_tenant_id', true));
        """,
        down_sql="""
            DROP TABLE IF EXISTS tenant_configs CASCADE;
        """,
        checksum=""
    ))
    
    # Migration 004: Model metrics table
    migrations.append(Migration(
        version="004_create_model_metrics",
        description="Create model performance metrics table",
        up_sql="""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_version VARCHAR(100) NOT NULL,
                tenant_id VARCHAR(100) NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                metric_value DECIMAL(10,6) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}',
                
                CONSTRAINT valid_tenant_id CHECK (length(tenant_id) > 0),
                CONSTRAINT valid_model_version CHECK (length(model_version) > 0),
                CONSTRAINT valid_metric_type CHECK (length(metric_type) > 0),
                CONSTRAINT valid_metadata CHECK (jsonb_typeof(metadata) = 'object')
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_model_metrics_version_type 
            ON model_metrics (model_version, metric_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_model_metrics_tenant_timestamp 
            ON model_metrics (tenant_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_model_metrics_type_value 
            ON model_metrics (metric_type, metric_value DESC);
            
            -- Enable RLS
            ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
            
            -- Create RLS policies
            CREATE POLICY IF NOT EXISTS tenant_isolation_metrics ON model_metrics
                FOR ALL TO application_role
                USING (tenant_id = current_setting('app.current_tenant_id', true));
        """,
        down_sql="""
            DROP TABLE IF EXISTS model_metrics CASCADE;
        """,
        checksum=""
    ))
    
    # Migration 005: Detector execution table
    migrations.append(Migration(
        version="005_create_detector_executions",
        description="Create detector execution logging table",
        up_sql="""
            CREATE TABLE IF NOT EXISTS detector_executions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(100) NOT NULL,
                detector_type VARCHAR(50) NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                confidence_score DECIMAL(5,4),
                success BOOLEAN NOT NULL,
                error_message TEXT,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                correlation_id UUID,
                
                CONSTRAINT valid_tenant_id CHECK (length(tenant_id) > 0),
                CONSTRAINT valid_detector_type CHECK (length(detector_type) > 0),
                CONSTRAINT valid_execution_time CHECK (execution_time_ms >= 0),
                CONSTRAINT valid_confidence 
                CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1))
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_detector_executions_performance 
            ON detector_executions (detector_type, timestamp DESC, execution_time_ms);
            
            CREATE INDEX IF NOT EXISTS idx_detector_executions_tenant_timestamp 
            ON detector_executions (tenant_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_detector_executions_success 
            ON detector_executions (success, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_detector_executions_correlation 
            ON detector_executions (correlation_id) 
            WHERE correlation_id IS NOT NULL;
            
            -- Enable RLS
            ALTER TABLE detector_executions ENABLE ROW LEVEL SECURITY;
            
            -- Create RLS policies
            CREATE POLICY IF NOT EXISTS tenant_isolation_executions ON detector_executions
                FOR ALL TO application_role
                USING (tenant_id = current_setting('app.current_tenant_id', true));
        """,
        down_sql="""
            DROP TABLE IF EXISTS detector_executions CASCADE;
        """,
        checksum=""
    ))
    
    return migrations
