"""Database migration system for Detector Orchestration Service."""

import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations for the orchestration service."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)

    async def initialize_migration_table(self) -> None:
        """Create the migrations tracking table."""
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    applied_at TIMESTAMPTZ DEFAULT NOW(),
                    checksum VARCHAR(64) NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
                ON schema_migrations(version);
            """
            )
        finally:
            await conn.close()

    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        conn = await asyncpg.connect(self.database_url)
        try:
            rows = await conn.fetch(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            return [row["version"] for row in rows]
        finally:
            await conn.close()

    async def apply_migration(self, version: str, name: str, sql: str) -> None:
        """Apply a single migration."""
        import hashlib

        checksum = hashlib.sha256(sql.encode()).hexdigest()

        conn = await asyncpg.connect(self.database_url)
        try:
            async with conn.transaction():
                # Apply the migration
                await conn.execute(sql)

                # Record the migration
                await conn.execute(
                    """
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES ($1, $2, $3)
                """,
                    version,
                    name,
                    checksum,
                )

                logger.info(f"Applied migration {version}: {name}")
        finally:
            await conn.close()

    async def migrate_from_monolith(self) -> None:
        """Migrate data from the monolithic database structure."""
        await self.initialize_migration_table()

        # Check if we need to migrate from existing data
        applied = await self.get_applied_migrations()

        if "001_initial_schema" not in applied:
            # Read and apply the initial schema
            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    schema_sql = f.read()

                await self.apply_migration(
                    "001_initial_schema",
                    "Initial orchestration service schema",
                    schema_sql,
                )

        if "002_migrate_detector_data" not in applied:
            await self.apply_migration(
                "002_migrate_detector_data",
                "Migrate detector data from monolith",
                self._get_detector_migration_sql(),
            )

        if "003_migrate_orchestration_data" not in applied:
            await self.apply_migration(
                "003_migrate_orchestration_data",
                "Migrate orchestration data from monolith",
                self._get_orchestration_migration_sql(),
            )

    def _get_detector_migration_sql(self) -> str:
        """SQL to migrate detector-related data from monolith."""
        return """
        -- Migrate detector execution data from existing storage_records
        INSERT INTO detector_executions (
            id,
            request_id,
            detector_id,
            tenant_id,
            input_hash,
            execution_status,
            started_at,
            completed_at,
            execution_time_ms,
            confidence_score,
            result_data,
            correlation_id
        )
        SELECT 
            gen_random_uuid(),
            COALESCE(correlation_id, gen_random_uuid()),
            gen_random_uuid(), -- Will need to create detector records first
            tenant_id,
            COALESCE(source_data_hash, encode(sha256(source_data::bytea), 'hex')),
            CASE 
                WHEN confidence_score IS NOT NULL THEN 'completed'
                ELSE 'failed'
            END,
            created_at,
            updated_at,
            EXTRACT(EPOCH FROM (updated_at - created_at)) * 1000,
            confidence_score,
            jsonb_build_object(
                'mapped_data', mapped_data,
                'metadata', metadata
            ),
            correlation_id
        FROM storage_records 
        WHERE detector_type IS NOT NULL
        ON CONFLICT DO NOTHING;
        
        -- Create default detector entries for existing detector types
        INSERT INTO detectors (
            detector_type,
            detector_name,
            endpoint_url,
            version,
            tenant_id,
            status
        )
        SELECT DISTINCT
            detector_type,
            detector_type || '_default',
            'http://localhost:8080/' || detector_type,
            '1.0.0',
            tenant_id,
            'active'
        FROM storage_records 
        WHERE detector_type IS NOT NULL
        ON CONFLICT (detector_type, detector_name, tenant_id) DO NOTHING;
        """

    def _get_orchestration_migration_sql(self) -> str:
        """SQL to migrate orchestration-related data from monolith."""
        return """
        -- Create orchestration requests from grouped detector executions
        INSERT INTO orchestration_requests (
            id,
            tenant_id,
            input_hash,
            detector_types,
            status,
            started_at,
            completed_at,
            detectors_executed,
            detectors_successful,
            correlation_id
        )
        SELECT 
            correlation_id,
            tenant_id,
            source_data_hash,
            array_agg(DISTINCT detector_type),
            CASE 
                WHEN COUNT(*) = COUNT(CASE WHEN confidence_score IS NOT NULL THEN 1 END) THEN 'completed'
                ELSE 'partial'
            END,
            MIN(created_at),
            MAX(updated_at),
            COUNT(*),
            COUNT(CASE WHEN confidence_score IS NOT NULL THEN 1 END),
            correlation_id
        FROM storage_records 
        WHERE correlation_id IS NOT NULL 
        AND detector_type IS NOT NULL
        GROUP BY correlation_id, tenant_id, source_data_hash
        ON CONFLICT (id) DO NOTHING;
        
        -- Create rate limiting entries for existing tenants
        INSERT INTO rate_limits (
            tenant_id,
            endpoint,
            requests_count,
            window_start,
            window_duration_seconds,
            limit_per_window
        )
        SELECT DISTINCT
            tenant_id,
            '/api/v1/orchestrate',
            0,
            DATE_TRUNC('hour', NOW()),
            3600,
            1000
        FROM storage_records
        ON CONFLICT DO NOTHING;
        """

    async def rollback_migration(self, version: str) -> None:
        """Rollback a specific migration (if rollback SQL is available)."""
        # This would require storing rollback SQL in migration files
        # For now, we'll just remove the migration record
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute(
                "DELETE FROM schema_migrations WHERE version = $1", version
            )
            logger.info(f"Rolled back migration {version}")
        finally:
            await conn.close()

    async def validate_schema(self) -> Dict[str, Any]:
        """Validate the current database schema."""
        conn = await asyncpg.connect(self.database_url)
        try:
            # Check required tables exist
            tables = await conn.fetch(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """
            )

            table_names = {row["table_name"] for row in tables}

            required_tables = {
                "detectors",
                "detector_executions",
                "orchestration_requests",
                "policies",
                "policy_enforcements",
                "rate_limits",
                "service_registry",
                "async_jobs",
                "orchestration_audit",
            }

            missing_tables = required_tables - table_names

            # Check indexes
            indexes = await conn.fetch(
                """
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """
            )

            index_names = {row["indexname"] for row in indexes}

            return {
                "valid": len(missing_tables) == 0,
                "missing_tables": list(missing_tables),
                "existing_tables": list(table_names),
                "index_count": len(index_names),
                "rls_enabled": await self._check_rls_enabled(conn),
            }
        finally:
            await conn.close()

    async def _check_rls_enabled(self, conn: asyncpg.Connection) -> bool:
        """Check if Row Level Security is properly enabled."""
        result = await conn.fetch(
            """
            SELECT schemaname, tablename, rowsecurity 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('detectors', 'detector_executions', 'orchestration_requests')
        """
        )

        return all(row["rowsecurity"] for row in result)


async def main():
    """Run migrations for the orchestration service."""
    import os

    database_url = os.getenv(
        "ORCHESTRATION_DATABASE_URL",
        "postgresql://orchestration:password@localhost:5432/orchestration_db",
    )

    manager = MigrationManager(database_url)

    try:
        await manager.migrate_from_monolith()
        validation = await manager.validate_schema()

        if validation["valid"]:
            print("✅ Orchestration service database migration completed successfully")
            print(f"📊 Tables: {len(validation['existing_tables'])}")
            print(f"📊 Indexes: {validation['index_count']}")
            print(f"🔒 RLS Enabled: {validation['rls_enabled']}")
        else:
            print("❌ Migration validation failed")
            print(f"Missing tables: {validation['missing_tables']}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
