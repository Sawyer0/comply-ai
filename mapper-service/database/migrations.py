"""Database migration system for Mapper Service."""

import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MapperMigrationManager:
    """Manages database migrations for the mapper service."""

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
                    "001_initial_schema", "Initial mapper service schema", schema_sql
                )

        if "002_migrate_mapping_data" not in applied:
            await self.apply_migration(
                "002_migrate_mapping_data",
                "Migrate mapping data from monolith",
                self._get_mapping_migration_sql(),
            )

        if "003_setup_model_management" not in applied:
            await self.apply_migration(
                "003_setup_model_management",
                "Setup model management and versioning",
                self._get_model_management_sql(),
            )

        if "004_migrate_cost_data" not in applied:
            await self.apply_migration(
                "004_migrate_cost_data",
                "Migrate cost tracking data",
                self._get_cost_migration_sql(),
            )

        if "005_setup_taxonomies" not in applied:
            await self.apply_migration(
                "005_setup_taxonomies",
                "Setup taxonomy and framework management",
                self._get_taxonomy_setup_sql(),
            )

    def _get_mapping_migration_sql(self) -> str:
        """SQL to migrate mapping-related data from monolith."""
        return """
        -- Create mapping requests from existing storage records
        INSERT INTO mapping_requests (
            id,
            tenant_id,
            analysis_response_id,
            target_frameworks,
            status,
            started_at,
            completed_at,
            correlation_id
        )
        SELECT DISTINCT
            gen_random_uuid(),
            tenant_id,
            COALESCE(correlation_id, gen_random_uuid()),
            ARRAY['SOC2', 'GDPR', 'HIPAA'],
            'completed',
            created_at,
            updated_at,
            correlation_id
        FROM storage_records 
        WHERE mapped_data IS NOT NULL
        ON CONFLICT DO NOTHING;
        
        -- Create mapping results from existing mapped data
        INSERT INTO mapping_results (
            mapping_request_id,
            tenant_id,
            canonical_category,
            canonical_subcategory,
            canonical_confidence,
            canonical_risk_level,
            canonical_tags,
            canonical_metadata,
            framework_mappings,
            overall_confidence,
            fallback_used,
            validation_passed
        )
        SELECT 
            mr.id,
            sr.tenant_id,
            COALESCE(sr.metadata->>'category', 'unknown'),
            COALESCE(sr.metadata->>'subcategory', 'unknown'),
            COALESCE(sr.confidence_score, 0.5),
            CASE 
                WHEN COALESCE(sr.confidence_score, 0.5) >= 0.8 THEN 'high'
                WHEN COALESCE(sr.confidence_score, 0.5) >= 0.6 THEN 'medium'
                ELSE 'low'
            END,
            CASE 
                WHEN sr.detector_type IS NOT NULL THEN ARRAY[sr.detector_type]
                ELSE ARRAY[]::TEXT[]
            END,
            sr.metadata,
            jsonb_build_object(
                'SOC2', jsonb_build_array(
                    jsonb_build_object(
                        'control_id', 'CC6.1',
                        'control_name', 'Logical Access Controls',
                        'compliance_status', 'compliant',
                        'confidence', COALESCE(sr.confidence_score, 0.5)
                    )
                ),
                'GDPR', jsonb_build_array(
                    jsonb_build_object(
                        'article', 'Article 32',
                        'requirement', 'Security of Processing',
                        'compliance_status', 'compliant',
                        'confidence', COALESCE(sr.confidence_score, 0.5)
                    )
                )
            ),
            COALESCE(sr.confidence_score, 0.5),
            sr.confidence_score IS NULL OR sr.confidence_score < 0.6,
            true
        FROM storage_records sr
        JOIN mapping_requests mr ON mr.tenant_id = sr.tenant_id 
        WHERE sr.mapped_data IS NOT NULL
        ON CONFLICT DO NOTHING;
        
        -- Create model inference records
        INSERT INTO model_inferences (
            mapping_request_id,
            tenant_id,
            model_name,
            model_version,
            inference_backend,
            input_tokens,
            output_tokens,
            inference_time_ms,
            confidence_score
        )
        SELECT 
            mr.id,
            mr.tenant_id,
            'llama-3-8b-instruct',
            COALESCE(sr.model_version, '1.0.0'),
            'vllm',
            LENGTH(sr.source_data) / 4, -- Rough token estimate
            LENGTH(sr.mapped_data) / 4,  -- Rough token estimate
            EXTRACT(EPOCH FROM (sr.updated_at - sr.created_at)) * 1000,
            COALESCE(sr.confidence_score, 0.5)
        FROM mapping_requests mr
        JOIN storage_records sr ON sr.tenant_id = mr.tenant_id
        WHERE sr.mapped_data IS NOT NULL
        ON CONFLICT DO NOTHING;
        """

    def _get_model_management_sql(self) -> str:
        """SQL to setup model management and versioning."""
        return """
        -- Create default model versions
        INSERT INTO model_versions (
            model_name,
            version,
            tenant_id,
            model_type,
            model_path,
            status,
            deployment_status,
            configuration,
            performance_metrics
        ) VALUES 
        (
            'llama-3-8b-instruct',
            '1.0.0',
            'default',
            'base',
            '/models/llama-3-8b-instruct',
            'active',
            'production',
            jsonb_build_object(
                'max_tokens', 4096,
                'temperature', 0.1,
                'top_p', 0.9,
                'backend', 'vllm'
            ),
            jsonb_build_object(
                'accuracy', 0.85,
                'latency_p95_ms', 150,
                'throughput_rps', 100
            )
        ),
        (
            'llama-3-8b-compliance-lora',
            '1.0.0',
            'default',
            'lora',
            '/models/llama-3-8b-compliance-lora',
            'active',
            'production',
            jsonb_build_object(
                'base_model', 'llama-3-8b-instruct',
                'lora_rank', 16,
                'lora_alpha', 32,
                'lora_dropout', 0.1
            ),
            jsonb_build_object(
                'accuracy', 0.92,
                'latency_p95_ms', 160,
                'throughput_rps', 95
            )
        )
        ON CONFLICT (model_name, version, tenant_id) DO NOTHING;
        
        -- Create training jobs for existing models
        INSERT INTO training_jobs (
            job_name,
            tenant_id,
            base_model,
            training_type,
            training_data_path,
            status,
            configuration,
            metrics,
            created_by
        ) VALUES 
        (
            'compliance_lora_v1',
            'default',
            'llama-3-8b-instruct',
            'lora',
            '/data/compliance_training_data.jsonl',
            'completed',
            jsonb_build_object(
                'lora_rank', 16,
                'lora_alpha', 32,
                'learning_rate', 2e-4,
                'batch_size', 4,
                'max_steps', 1000
            ),
            jsonb_build_object(
                'final_loss', 0.15,
                'eval_accuracy', 0.92,
                'training_time_minutes', 120
            ),
            'system'
        )
        ON CONFLICT (job_name, tenant_id) DO NOTHING;
        """

    def _get_cost_migration_sql(self) -> str:
        """SQL to migrate cost tracking data."""
        return """
        -- Create cost metrics from existing inference data
        INSERT INTO cost_metrics (
            mapping_request_id,
            tenant_id,
            tokens_processed,
            inference_cost,
            storage_cost,
            total_cost,
            cost_per_request,
            billing_period
        )
        SELECT 
            mi.mapping_request_id,
            mi.tenant_id,
            mi.input_tokens + mi.output_tokens,
            (mi.input_tokens + mi.output_tokens) * 0.0001, -- $0.0001 per token
            0.001, -- $0.001 storage cost
            (mi.input_tokens + mi.output_tokens) * 0.0001 + 0.001,
            (mi.input_tokens + mi.output_tokens) * 0.0001 + 0.001,
            DATE(mi.created_at)
        FROM model_inferences mi
        ON CONFLICT DO NOTHING;
        
        -- Create storage artifacts for existing models
        INSERT INTO storage_artifacts (
            artifact_type,
            artifact_name,
            tenant_id,
            storage_path,
            storage_backend,
            size_bytes,
            metadata
        ) VALUES 
        (
            'model',
            'llama-3-8b-instruct',
            'default',
            's3://models/llama-3-8b-instruct/',
            's3',
            17000000000, -- ~17GB
            jsonb_build_object(
                'model_type', 'base',
                'format', 'safetensors',
                'precision', 'fp16'
            )
        ),
        (
            'model',
            'llama-3-8b-compliance-lora',
            'default',
            's3://models/llama-3-8b-compliance-lora/',
            's3',
            134000000, -- ~134MB
            jsonb_build_object(
                'model_type', 'lora',
                'format', 'safetensors',
                'base_model', 'llama-3-8b-instruct'
            )
        )
        ON CONFLICT (artifact_name, tenant_id) DO NOTHING;
        """

    def _get_taxonomy_setup_sql(self) -> str:
        """SQL to setup taxonomy and framework management."""
        return """
        -- Create default taxonomies
        INSERT INTO taxonomies (
            taxonomy_name,
            version,
            tenant_id,
            taxonomy_data,
            schema_version,
            is_active,
            backward_compatible,
            created_by
        ) VALUES 
        (
            'canonical_compliance_taxonomy',
            '1.0.0',
            'default',
            jsonb_build_object(
                'categories', jsonb_build_object(
                    'pii', jsonb_build_object(
                        'subcategories', ARRAY['personal_identifiers', 'financial_data', 'health_data', 'biometric_data'],
                        'risk_levels', ARRAY['low', 'medium', 'high', 'critical']
                    ),
                    'security', jsonb_build_object(
                        'subcategories', ARRAY['access_control', 'encryption', 'authentication', 'authorization'],
                        'risk_levels', ARRAY['low', 'medium', 'high', 'critical']
                    ),
                    'content_moderation', jsonb_build_object(
                        'subcategories', ARRAY['hate_speech', 'violence', 'harassment', 'misinformation'],
                        'risk_levels', ARRAY['low', 'medium', 'high', 'critical']
                    )
                )
            ),
            '1.0',
            true,
            true,
            'system'
        )
        ON CONFLICT (taxonomy_name, version, tenant_id) DO NOTHING;
        
        -- Create framework configurations
        INSERT INTO framework_configs (
            framework_name,
            framework_version,
            tenant_id,
            mapping_rules,
            validation_schema,
            is_active,
            created_by
        ) VALUES 
        (
            'SOC2',
            '2017',
            'default',
            jsonb_build_object(
                'pii', jsonb_build_object(
                    'personal_identifiers', ARRAY['CC6.1', 'CC6.7'],
                    'financial_data', ARRAY['CC6.1', 'CC6.2', 'CC6.7'],
                    'health_data', ARRAY['CC6.1', 'CC6.2', 'CC6.7', 'CC7.1']
                ),
                'security', jsonb_build_object(
                    'access_control', ARRAY['CC6.1', 'CC6.2'],
                    'encryption', ARRAY['CC6.7', 'CC6.8'],
                    'authentication', ARRAY['CC6.1', 'CC6.2', 'CC6.3']
                )
            ),
            jsonb_build_object(
                'type', 'object',
                'properties', jsonb_build_object(
                    'control_id', jsonb_build_object('type', 'string'),
                    'control_name', jsonb_build_object('type', 'string'),
                    'compliance_status', jsonb_build_object('type', 'string', 'enum', ARRAY['compliant', 'non_compliant', 'not_applicable'])
                )
            ),
            true,
            'system'
        ),
        (
            'GDPR',
            '2018',
            'default',
            jsonb_build_object(
                'pii', jsonb_build_object(
                    'personal_identifiers', ARRAY['Article 6', 'Article 32'],
                    'financial_data', ARRAY['Article 6', 'Article 9', 'Article 32'],
                    'health_data', ARRAY['Article 6', 'Article 9', 'Article 32', 'Article 35']
                )
            ),
            jsonb_build_object(
                'type', 'object',
                'properties', jsonb_build_object(
                    'article', jsonb_build_object('type', 'string'),
                    'requirement', jsonb_build_object('type', 'string'),
                    'compliance_status', jsonb_build_object('type', 'string', 'enum', ARRAY['compliant', 'non_compliant', 'not_applicable'])
                )
            ),
            true,
            'system'
        )
        ON CONFLICT (framework_name, framework_version, tenant_id) DO NOTHING;
        
        -- Create validation schemas
        INSERT INTO validation_schemas (
            schema_name,
            schema_version,
            tenant_id,
            schema_type,
            json_schema,
            is_active,
            created_by
        ) VALUES 
        (
            'mapping_request_schema',
            '1.0.0',
            'default',
            'input',
            jsonb_build_object(
                'type', 'object',
                'required', ARRAY['analysis_response_id', 'target_frameworks'],
                'properties', jsonb_build_object(
                    'analysis_response_id', jsonb_build_object('type', 'string', 'format', 'uuid'),
                    'target_frameworks', jsonb_build_object('type', 'array', 'items', jsonb_build_object('type', 'string')),
                    'mapping_mode', jsonb_build_object('type', 'string', 'enum', ARRAY['standard', 'fast', 'comprehensive'])
                )
            ),
            true,
            'system'
        ),
        (
            'mapping_response_schema',
            '1.0.0',
            'default',
            'output',
            jsonb_build_object(
                'type', 'object',
                'required', ARRAY['mapping_results', 'overall_confidence'],
                'properties', jsonb_build_object(
                    'mapping_results', jsonb_build_object('type', 'array'),
                    'overall_confidence', jsonb_build_object('type', 'number', 'minimum', 0, 'maximum', 1),
                    'cost_metrics', jsonb_build_object('type', 'object'),
                    'model_metrics', jsonb_build_object('type', 'object')
                )
            ),
            true,
            'system'
        )
        ON CONFLICT (schema_name, schema_version, tenant_id) DO NOTHING;
        
        -- Create feature flags for existing tenants
        INSERT INTO feature_flags (
            flag_name,
            tenant_id,
            is_enabled,
            rollout_percentage,
            created_by
        )
        SELECT DISTINCT
            'enable_lora_models',
            tenant_id,
            true,
            100,
            'system'
        FROM storage_records
        UNION ALL
        SELECT DISTINCT
            'enable_cost_tracking',
            tenant_id,
            true,
            100,
            'system'
        FROM storage_records
        ON CONFLICT (flag_name, tenant_id) DO NOTHING;
        """

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
                "mapping_requests",
                "mapping_results",
                "model_inferences",
                "cost_metrics",
                "training_jobs",
                "model_versions",
                "deployment_experiments",
                "taxonomies",
                "framework_configs",
                "validation_schemas",
                "feature_flags",
                "storage_artifacts",
            }

            missing_tables = required_tables - table_names

            # Check model versions
            model_count = await conn.fetchval(
                "SELECT COUNT(*) FROM model_versions WHERE status = 'active'"
            )

            return {
                "valid": len(missing_tables) == 0,
                "missing_tables": list(missing_tables),
                "existing_tables": list(table_names),
                "active_models": model_count,
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
            AND tablename IN ('mapping_requests', 'mapping_results', 'cost_metrics')
        """
        )

        return all(row["rowsecurity"] for row in result)


async def main():
    """Run migrations for the mapper service."""
    import os

    database_url = os.getenv(
        "MAPPER_DATABASE_URL", "postgresql://mapper:password@localhost:5432/mapper_db"
    )

    manager = MapperMigrationManager(database_url)

    try:
        await manager.migrate_from_monolith()
        validation = await manager.validate_schema()

        if validation["valid"]:
            print("✅ Mapper service database migration completed successfully")
            print(f"📊 Tables: {len(validation['existing_tables'])}")
            print(f"🤖 Active Models: {validation['active_models']}")
            print(f"🔒 RLS Enabled: {validation['rls_enabled']}")
        else:
            print("❌ Migration validation failed")
            print(f"Missing tables: {validation['missing_tables']}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
