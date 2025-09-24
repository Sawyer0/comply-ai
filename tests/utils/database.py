"""
Test database management for multi-service testing.

This module provides utilities for managing test databases across:
- PostgreSQL (Core Mapper and Detector Orchestration)
- ClickHouse (Analysis Service)
- Redis (All services for caching)
"""

import asyncio
import asyncpg
import redis.asyncio as redis
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)


class TestDatabaseManager:
    """Manages test databases for all services."""
    
    def __init__(self, test_config):
        self.config = test_config
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.clickhouse_client = None  # Would use clickhouse-driver in real implementation
        
        # Database schemas for each service
        self.schemas = {
            "core_mapper": self._get_core_mapper_schema(),
            "detector_orchestration": self._get_detector_orchestration_schema(),
            "analysis_service": self._get_analysis_service_schema()
        }
    
    async def setup_databases(self) -> None:
        """Setup all test databases."""
        logger.info("Setting up test databases")
        
        # Setup PostgreSQL
        await self._setup_postgres()
        
        # Setup Redis
        await self._setup_redis()
        
        # Setup ClickHouse (if available)
        await self._setup_clickhouse()
        
        logger.info("Test databases setup complete")
    
    async def cleanup_databases(self) -> None:
        """Cleanup all test databases."""
        logger.info("Cleaning up test databases")
        
        # Cleanup PostgreSQL
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        # Cleanup Redis
        if self.redis_client:
            await self.redis_client.flushall()
            await self.redis_client.aclose()
        
        # Cleanup ClickHouse
        if self.clickhouse_client:
            # Would cleanup ClickHouse here
            pass
        
        logger.info("Test databases cleanup complete")
    
    async def get_postgres_pool(self) -> asyncpg.Pool:
        """Get PostgreSQL connection pool."""
        if not self.postgres_pool:
            await self._setup_postgres()
        return self.postgres_pool
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            await self._setup_redis()
        return self.redis_client
    
    async def get_clickhouse_client(self):
        """Get ClickHouse client."""
        if not self.clickhouse_client:
            await self._setup_clickhouse()
        return self.clickhouse_client
    
    async def clean_postgres_state(self) -> None:
        """Clean PostgreSQL state for test isolation."""
        if not self.postgres_pool:
            return
        
        async with self.postgres_pool.acquire() as conn:
            # Truncate all tables (in reverse dependency order)
            tables = [
                "compliance_mappings", "detector_outputs", "analysis_results",
                "tenant_configurations", "audit_trail", "model_metrics"
            ]
            
            for table in tables:
                try:
                    await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                except Exception as e:
                    logger.debug(f"Error truncating table", table=table, error=str(e))
            
            # Reset sequences
            sequences = [
                "compliance_mappings_id_seq", "detector_outputs_id_seq",
                "analysis_results_id_seq", "audit_trail_id_seq"
            ]
            
            for sequence in sequences:
                try:
                    await conn.execute(f"ALTER SEQUENCE {sequence} RESTART WITH 1")
                except Exception as e:
                    logger.debug(f"Error resetting sequence", sequence=sequence, error=str(e))
    
    async def clean_redis_state(self) -> None:
        """Clean Redis state for test isolation."""
        if self.redis_client:
            await self.redis_client.flushdb()
    
    async def clean_clickhouse_state(self) -> None:
        """Clean ClickHouse state for test isolation."""
        if self.clickhouse_client:
            # Would implement ClickHouse cleanup here
            pass
    
    async def create_test_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Create test tenant with isolated data."""
        if not self.postgres_pool:
            await self._setup_postgres()
        
        async with self.postgres_pool.acquire() as conn:
            # Insert tenant configuration
            tenant_config = {
                "tenant_id": tenant_id,
                "name": f"Test Tenant {tenant_id}",
                "compliance_frameworks": ["SOC2", "ISO27001"],
                "privacy_settings": {
                    "data_retention_days": 90,
                    "pii_redaction": True,
                    "audit_logging": True
                },
                "created_at": "NOW()",
                "updated_at": "NOW()"
            }
            
            try:
                await conn.execute("""
                    INSERT INTO tenant_configurations (
                        tenant_id, name, compliance_frameworks, privacy_settings,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        compliance_frameworks = EXCLUDED.compliance_frameworks,
                        privacy_settings = EXCLUDED.privacy_settings,
                        updated_at = EXCLUDED.updated_at
                """, 
                tenant_id,
                tenant_config["name"],
                tenant_config["compliance_frameworks"],
                tenant_config["privacy_settings"],
                tenant_config["created_at"],
                tenant_config["updated_at"]
                )
                
                logger.info(f"Created test tenant", tenant_id=tenant_id)
                return tenant_config
                
            except Exception as e:
                logger.error(f"Failed to create test tenant", tenant_id=tenant_id, error=str(e))
                raise
    
    async def insert_test_data(self, service: str, data: List[Dict[str, Any]]) -> None:
        """Insert test data for specific service."""
        if service == "core_mapper":
            await self._insert_mapper_test_data(data)
        elif service == "detector_orchestration":
            await self._insert_orchestration_test_data(data)
        elif service == "analysis_service":
            await self._insert_analysis_test_data(data)
        else:
            raise ValueError(f"Unknown service: {service}")
    
    async def get_test_data(self, service: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get test data for specific service."""
        if not self.postgres_pool:
            await self._setup_postgres()
        
        async with self.postgres_pool.acquire() as conn:
            if service == "core_mapper":
                query = "SELECT * FROM compliance_mappings"
                params = []
                
                if filters:
                    conditions = []
                    if "tenant_id" in filters:
                        conditions.append("tenant_id = $1")
                        params.append(filters["tenant_id"])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
            elif service == "detector_orchestration":
                query = "SELECT * FROM detector_outputs"
                params = []
                
                if filters:
                    conditions = []
                    if "detector_type" in filters:
                        conditions.append("detector_type = $1")
                        params.append(filters["detector_type"])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
            elif service == "analysis_service":
                query = "SELECT * FROM analysis_results"
                params = []
                
                if filters:
                    conditions = []
                    if "analysis_type" in filters:
                        conditions.append("analysis_type = $1")
                        params.append(filters["analysis_type"])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            
            else:
                raise ValueError(f"Unknown service: {service}")
    
    async def _setup_postgres(self) -> None:
        """Setup PostgreSQL test database."""
        try:
            # Create connection pool
            self.postgres_pool = await asyncpg.create_pool(
                host=getattr(self.config, 'postgres_host', 'localhost'),
                port=getattr(self.config, 'postgres_port', 5432),
                database=getattr(self.config, 'postgres_test_db', 'llama_mapper_test'),
                user=getattr(self.config, 'postgres_user', 'test_user'),
                password=getattr(self.config, 'postgres_password', 'test_password'),
                min_size=2,
                max_size=10
            )
            
            # Create database schema
            await self._create_postgres_schema()
            
            logger.info("PostgreSQL test database setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL", error=str(e))
            raise
    
    async def _setup_redis(self) -> None:
        """Setup Redis test client."""
        try:
            self.redis_client = redis.Redis(
                host=getattr(self.config, 'redis_host', 'localhost'),
                port=getattr(self.config, 'redis_port', 6379),
                db=getattr(self.config, 'redis_test_db', 1),
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Clear test database
            await self.redis_client.flushdb()
            
            logger.info("Redis test client setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup Redis", error=str(e))
            raise
    
    async def _setup_clickhouse(self) -> None:
        """Setup ClickHouse test client."""
        try:
            # In a real implementation, would setup ClickHouse client here
            # self.clickhouse_client = clickhouse_driver.Client(...)
            logger.info("ClickHouse test client setup complete (mock)")
            
        except Exception as e:
            logger.error(f"Failed to setup ClickHouse", error=str(e))
            # Don't raise for ClickHouse as it's optional in tests
    
    async def _create_postgres_schema(self) -> None:
        """Create PostgreSQL test schema."""
        async with self.postgres_pool.acquire() as conn:
            # Create tables for each service
            for service, schema_sql in self.schemas.items():
                try:
                    await conn.execute(schema_sql)
                    logger.debug(f"Created schema for service", service=service)
                except Exception as e:
                    logger.error(f"Failed to create schema", service=service, error=str(e))
                    raise
    
    async def _insert_mapper_test_data(self, data: List[Dict[str, Any]]) -> None:
        """Insert test data for Core Mapper service."""
        async with self.postgres_pool.acquire() as conn:
            for item in data:
                await conn.execute("""
                    INSERT INTO compliance_mappings (
                        tenant_id, canonical_result, framework_mappings,
                        confidence_score, processing_metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                item.get("tenant_id", "test_tenant"),
                item.get("canonical_result", {}),
                item.get("framework_mappings", []),
                item.get("confidence_score", 0.95),
                item.get("processing_metadata", {})
                )
    
    async def _insert_orchestration_test_data(self, data: List[Dict[str, Any]]) -> None:
        """Insert test data for Detector Orchestration service."""
        async with self.postgres_pool.acquire() as conn:
            for item in data:
                await conn.execute("""
                    INSERT INTO detector_outputs (
                        detector_type, findings, metadata, created_at
                    ) VALUES ($1, $2, $3, NOW())
                """,
                item.get("detector_type", "test_detector"),
                item.get("findings", []),
                item.get("metadata", {})
                )
    
    async def _insert_analysis_test_data(self, data: List[Dict[str, Any]]) -> None:
        """Insert test data for Analysis service."""
        async with self.postgres_pool.acquire() as conn:
            for item in data:
                await conn.execute("""
                    INSERT INTO analysis_results (
                        analysis_type, risk_assessment, remediation_actions,
                        compliance_evidence, processing_metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                item.get("analysis_type", "compliance_analysis"),
                item.get("risk_assessment", {}),
                item.get("remediation_actions", []),
                item.get("compliance_evidence", {}),
                item.get("processing_metadata", {})
                )
    
    def _get_core_mapper_schema(self) -> str:
        """Get Core Mapper database schema."""
        return """
        CREATE TABLE IF NOT EXISTS compliance_mappings (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            canonical_result JSONB NOT NULL,
            framework_mappings JSONB NOT NULL,
            confidence_score FLOAT NOT NULL,
            processing_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_compliance_mappings_tenant_id 
        ON compliance_mappings(tenant_id);
        
        CREATE INDEX IF NOT EXISTS idx_compliance_mappings_created_at 
        ON compliance_mappings(created_at);
        
        CREATE TABLE IF NOT EXISTS tenant_configurations (
            tenant_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            compliance_frameworks TEXT[] NOT NULL,
            privacy_settings JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
    
    def _get_detector_orchestration_schema(self) -> str:
        """Get Detector Orchestration database schema."""
        return """
        CREATE TABLE IF NOT EXISTS detector_outputs (
            id SERIAL PRIMARY KEY,
            detector_type VARCHAR(255) NOT NULL,
            findings JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_detector_outputs_type 
        ON detector_outputs(detector_type);
        
        CREATE INDEX IF NOT EXISTS idx_detector_outputs_created_at 
        ON detector_outputs(created_at);
        
        CREATE TABLE IF NOT EXISTS detector_registry (
            name VARCHAR(255) PRIMARY KEY,
            version VARCHAR(50) NOT NULL,
            endpoint VARCHAR(255) NOT NULL,
            capabilities TEXT[] NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'active',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
    
    def _get_analysis_service_schema(self) -> str:
        """Get Analysis Service database schema."""
        return """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            analysis_type VARCHAR(255) NOT NULL,
            risk_assessment JSONB NOT NULL,
            remediation_actions JSONB NOT NULL,
            compliance_evidence JSONB NOT NULL,
            processing_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_analysis_results_type 
        ON analysis_results(analysis_type);
        
        CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at 
        ON analysis_results(created_at);
        
        CREATE TABLE IF NOT EXISTS audit_trail (
            id SERIAL PRIMARY KEY,
            event_type VARCHAR(255) NOT NULL,
            event_data JSONB NOT NULL,
            tenant_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255),
            correlation_id VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_trail_tenant_id 
        ON audit_trail(tenant_id);
        
        CREATE INDEX IF NOT EXISTS idx_audit_trail_created_at 
        ON audit_trail(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_audit_trail_correlation_id 
        ON audit_trail(correlation_id);
        
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            metrics JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_model_metrics_name_version 
        ON model_metrics(model_name, model_version);
        """
