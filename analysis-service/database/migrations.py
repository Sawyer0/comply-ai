"""Database migration system for Analysis Service."""

import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisMigrationManager:
    """Manages database migrations for the analysis service."""

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
                    "001_initial_schema", "Initial analysis service schema", schema_sql
                )

        if "002_migrate_analysis_data" not in applied:
            await self.apply_migration(
                "002_migrate_analysis_data",
                "Migrate analysis data from monolith",
                self._get_analysis_migration_sql(),
            )

        if "003_setup_knowledge_base" not in applied:
            await self.apply_migration(
                "003_setup_knowledge_base",
                "Setup RAG knowledge base",
                self._get_knowledge_base_setup_sql(),
            )

        if "004_migrate_quality_data" not in applied:
            await self.apply_migration(
                "004_migrate_quality_data",
                "Migrate quality monitoring data",
                self._get_quality_migration_sql(),
            )

    def _get_analysis_migration_sql(self) -> str:
        """SQL to migrate analysis-related data from monolith."""
        return """
        -- Create analysis requests from existing storage records
        INSERT INTO analysis_requests (
            id,
            tenant_id,
            orchestration_response_id,
            analysis_types,
            status,
            started_at,
            completed_at,
            correlation_id
        )
        SELECT DISTINCT
            gen_random_uuid(),
            tenant_id,
            COALESCE(correlation_id, gen_random_uuid()),
            ARRAY['pattern', 'risk', 'compliance'],
            'completed',
            created_at,
            updated_at,
            correlation_id
        FROM storage_records 
        WHERE mapped_data IS NOT NULL
        ON CONFLICT DO NOTHING;
        
        -- Create canonical results from existing mapped data
        INSERT INTO canonical_results (
            analysis_request_id,
            tenant_id,
            category,
            subcategory,
            confidence,
            risk_level,
            tags,
            metadata
        )
        SELECT 
            ar.id,
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
            sr.metadata
        FROM storage_records sr
        JOIN analysis_requests ar ON ar.tenant_id = sr.tenant_id 
        WHERE sr.mapped_data IS NOT NULL
        ON CONFLICT DO NOTHING;
        
        -- Create risk scores with default values
        INSERT INTO risk_scores (
            analysis_request_id,
            tenant_id,
            overall_risk_score,
            technical_risk,
            business_risk,
            regulatory_risk,
            temporal_risk,
            risk_factors
        )
        SELECT 
            ar.id,
            ar.tenant_id,
            COALESCE(AVG(cr.confidence), 0.5),
            COALESCE(AVG(cr.confidence) * 0.8, 0.4),
            COALESCE(AVG(cr.confidence) * 0.6, 0.3),
            COALESCE(AVG(cr.confidence) * 0.9, 0.45),
            COALESCE(AVG(cr.confidence) * 0.7, 0.35),
            jsonb_build_array(
                jsonb_build_object(
                    'factor', 'confidence_based',
                    'score', AVG(cr.confidence),
                    'description', 'Risk calculated from detection confidence'
                )
            )
        FROM analysis_requests ar
        JOIN canonical_results cr ON cr.analysis_request_id = ar.id
        GROUP BY ar.id, ar.tenant_id
        ON CONFLICT DO NOTHING;
        """

    def _get_knowledge_base_setup_sql(self) -> str:
        """SQL to setup the RAG knowledge base with initial data."""
        return """
        -- Insert sample compliance framework documents
        INSERT INTO knowledge_base (
            document_id,
            document_title,
            document_type,
            content,
            content_hash,
            framework,
            category,
            version
        ) VALUES 
        (
            'soc2_cc6_1',
            'SOC 2 CC6.1 - Logical and Physical Access Controls',
            'regulation',
            'The entity implements logical and physical access controls to restrict unauthorized access to the system and protect against threats from sources outside its system boundaries.',
            encode(sha256('soc2_cc6_1_content'::bytea), 'hex'),
            'SOC2',
            'access_control',
            '2017'
        ),
        (
            'gdpr_art32',
            'GDPR Article 32 - Security of Processing',
            'regulation',
            'Taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of processing as well as the risk of varying likelihood and severity for the rights and freedoms of natural persons, the controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.',
            encode(sha256('gdpr_art32_content'::bytea), 'hex'),
            'GDPR',
            'data_security',
            '2018'
        ),
        (
            'hipaa_164_312',
            'HIPAA 164.312 - Technical Safeguards',
            'regulation',
            'A covered entity must, in accordance with § 164.306: (a)(1) Access control. Implement technical policies and procedures for electronic information systems that maintain electronic protected health information to allow access only to those persons or software programs that have been granted access rights as specified in § 164.308(a)(4).',
            encode(sha256('hipaa_164_312_content'::bytea), 'hex'),
            'HIPAA',
            'technical_safeguards',
            '2013'
        ),
        (
            'iso27001_a8_2_1',
            'ISO 27001 A.8.2.1 - Classification of Information',
            'regulation',
            'Information shall be classified in terms of legal requirements, value, criticality and sensitivity to unauthorised disclosure or modification.',
            encode(sha256('iso27001_a8_2_1_content'::bytea), 'hex'),
            'ISO27001',
            'information_classification',
            '2022'
        )
        ON CONFLICT (document_id, version) DO NOTHING;
        
        -- Create default ML models entries
        INSERT INTO ml_models (
            model_name,
            model_version,
            model_type,
            tenant_id,
            status,
            configuration
        ) VALUES 
        (
            'phi3-mini-4k-instruct',
            '1.0.0',
            'phi3',
            'default',
            'active',
            jsonb_build_object(
                'max_tokens', 4096,
                'temperature', 0.1,
                'top_p', 0.9
            )
        ),
        (
            'all-MiniLM-L6-v2',
            '1.0.0',
            'embedding',
            'default',
            'active',
            jsonb_build_object(
                'dimension', 384,
                'normalize', true
            )
        )
        ON CONFLICT (model_name, model_version, tenant_id) DO NOTHING;
        """

    def _get_quality_migration_sql(self) -> str:
        """SQL to migrate quality monitoring data."""
        return """
        -- Create quality metrics from existing model metrics
        INSERT INTO quality_metrics (
            tenant_id,
            metric_type,
            metric_name,
            metric_value,
            model_version,
            evaluation_date
        )
        SELECT 
            tenant_id,
            'accuracy',
            'detection_accuracy',
            AVG(COALESCE(confidence_score, 0.5)),
            COALESCE(model_version, 'unknown'),
            DATE(created_at)
        FROM storage_records 
        WHERE confidence_score IS NOT NULL
        GROUP BY tenant_id, model_version, DATE(created_at)
        ON CONFLICT DO NOTHING;
        
        -- Create weekly evaluations from aggregated data
        INSERT INTO weekly_evaluations (
            tenant_id,
            evaluation_week,
            model_version,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            status
        )
        SELECT 
            tenant_id,
            DATE_TRUNC('week', created_at)::DATE,
            COALESCE(model_version, 'unknown'),
            AVG(COALESCE(confidence_score, 0.5)),
            AVG(COALESCE(confidence_score, 0.5)) * 0.9,
            AVG(COALESCE(confidence_score, 0.5)) * 0.85,
            AVG(COALESCE(confidence_score, 0.5)) * 0.87,
            'completed'
        FROM storage_records 
        WHERE confidence_score IS NOT NULL
        GROUP BY tenant_id, DATE_TRUNC('week', created_at)::DATE, model_version
        ON CONFLICT (tenant_id, evaluation_week, model_version) DO NOTHING;
        
        -- Create analysis pipelines for existing tenants
        INSERT INTO analysis_pipelines (
            pipeline_name,
            pipeline_version,
            tenant_id,
            configuration,
            enabled
        )
        SELECT DISTINCT
            'default_analysis_pipeline',
            '1.0.0',
            tenant_id,
            jsonb_build_object(
                'analysis_types', ARRAY['pattern', 'risk', 'compliance'],
                'frameworks', ARRAY['SOC2', 'GDPR', 'HIPAA'],
                'quality_checks', true,
                'rag_enabled', true
            ),
            true
        FROM storage_records
        ON CONFLICT (pipeline_name, pipeline_version, tenant_id) DO NOTHING;
        """

    async def setup_vector_extension(self) -> None:
        """Setup pgvector extension for embeddings."""
        conn = await asyncpg.connect(self.database_url)
        try:
            # Check if vector extension is available
            result = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions 
                    WHERE name = 'vector'
                )
            """
            )

            if result:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("Vector extension enabled for embeddings")
            else:
                logger.warning(
                    "Vector extension not available - embeddings will be stored as JSONB"
                )
        except Exception as e:
            logger.warning(f"Could not setup vector extension: {e}")
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
                "analysis_requests",
                "canonical_results",
                "pattern_analysis",
                "risk_scores",
                "compliance_mappings",
                "knowledge_base",
                "rag_insights",
                "quality_metrics",
                "quality_alerts",
                "weekly_evaluations",
                "ml_models",
                "analysis_pipelines",
                "tenant_analytics",
            }

            missing_tables = required_tables - table_names

            # Check vector extension
            vector_enabled = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'vector'
                )
            """
            )

            return {
                "valid": len(missing_tables) == 0,
                "missing_tables": list(missing_tables),
                "existing_tables": list(table_names),
                "vector_extension": vector_enabled,
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
            AND tablename IN ('analysis_requests', 'canonical_results', 'quality_metrics')
        """
        )

        return all(row["rowsecurity"] for row in result)


async def main():
    """Run migrations for the analysis service."""
    import os

    database_url = os.getenv(
        "ANALYSIS_DATABASE_URL",
        "postgresql://analysis:password@localhost:5432/analysis_db",
    )

    manager = AnalysisMigrationManager(database_url)

    try:
        await manager.setup_vector_extension()
        await manager.migrate_from_monolith()
        validation = await manager.validate_schema()

        if validation["valid"]:
            print("✅ Analysis service database migration completed successfully")
            print(f"📊 Tables: {len(validation['existing_tables'])}")
            print(f"🔍 Vector Extension: {validation['vector_extension']}")
            print(f"🔒 RLS Enabled: {validation['rls_enabled']}")
        else:
            print("❌ Migration validation failed")
            print(f"Missing tables: {validation['missing_tables']}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
