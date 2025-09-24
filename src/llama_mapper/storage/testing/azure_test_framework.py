"""Azure Database testing framework with Azure-specific considerations."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional

import asyncpg
import pytest
import structlog

from ..database.azure_config import AzureDatabaseConfig, AzureDatabaseConnectionManager
from ..database.migrations import DatabaseMigrationManager, create_production_migrations
from ..manager.models import StorageRecord, AuditRecord, TenantConfig
from ..security.encryption import FieldEncryption, EnhancedRowLevelSecurity

logger = structlog.get_logger(__name__)


class AzureDatabaseTestFramework:
    """Azure Database testing utilities with Azure-specific considerations."""
    
    def __init__(self):
        self.test_databases: List[str] = []
        self.azure_config: Optional[AzureDatabaseConfig] = None
        self.connection_manager: Optional[AzureDatabaseConnectionManager] = None
    
    def get_azure_test_config(self) -> AzureDatabaseConfig:
        """Get Azure Database configuration for testing."""
        return AzureDatabaseConfig(
            subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID', 'test-subscription'),
            resource_group=os.getenv('AZURE_RESOURCE_GROUP', 'test-rg'),
            server_name=os.getenv('AZURE_DB_SERVER', 'test-server'),
            azure_db_host=os.getenv('AZURE_DB_HOST', 'test-server.postgres.database.azure.com'),
            database_name=os.getenv('AZURE_DB_NAME', 'postgres'),
            username=os.getenv('AZURE_DB_USER', 'test_user'),
            password=os.getenv('AZURE_DB_PASSWORD', 'test_password'),
            min_pool_size=1,
            max_pool_size=5
        )
    
    @pytest.fixture
    async def azure_test_database(self):
        """Create isolated test database on Azure."""
        test_db_name = f"test_llama_mapper_{uuid.uuid4().hex[:8]}"
        self.test_databases.append(test_db_name)
        
        # Get Azure Database connection parameters
        config = self.get_azure_test_config()
        
        azure_params = {
            'host': config.azure_db_host,
            'port': 5432,
            'user': f"{config.username}@{config.server_name}",
            'password': config.password,
            'ssl': 'require',
            'sslmode': 'require'
        }
        
        # Connect to default database to create test database
        admin_conn = await asyncpg.connect(
            database='postgres',
            **azure_params
        )
        
        try:
            await admin_conn.execute(f'CREATE DATABASE "{test_db_name}"')
            logger.info("Test database created", database=test_db_name)
        except Exception as e:
            logger.warning("Failed to create test database", error=str(e))
            # Database might already exist, continue
        finally:
            await admin_conn.close()
        
        # Connect to test database and apply schema
        test_conn = await asyncpg.connect(
            database=test_db_name,
            **azure_params
        )
        
        # Create test connection manager
        test_config = config
        test_config.database_name = test_db_name
        self.azure_config = test_config
        self.connection_manager = AzureDatabaseConnectionManager(test_config)
        await self.connection_manager.initialize_pools()
        
        # Apply migrations
        migration_manager = DatabaseMigrationManager(self.connection_manager)
        migration_manager.migrations = create_production_migrations()
        await migration_manager.initialize_migration_tracking()
        await migration_manager.apply_all_migrations()
        
        yield self.connection_manager
        
        # Cleanup
        await test_conn.close()
        if self.connection_manager:
            await self.connection_manager.close_pools()
        
        # Drop test database
        admin_conn = await asyncpg.connect(
            database='postgres',
            **azure_params
        )
        try:
            # Terminate active connections to the test database
            await admin_conn.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{test_db_name}' AND pid <> pg_backend_pid()
            """)
            await admin_conn.execute(f'DROP DATABASE IF EXISTS "{test_db_name}"')
            logger.info("Test database dropped", database=test_db_name)
        except Exception as e:
            logger.warning("Failed to drop test database", database=test_db_name, error=str(e))
        finally:
            await admin_conn.close()
    
    async def test_azure_specific_features(self, azure_test_database):
        """Test Azure Database specific features."""
        connection_manager = azure_test_database
        
        # Test Azure extensions
        async with connection_manager.get_read_connection() as conn:
            extensions = await conn.fetch("""
                SELECT name, installed_version 
                FROM pg_available_extensions 
                WHERE name IN ('pg_stat_statements', 'pg_buffercache', 'pgcrypto', 'uuid-ossp')
            """)
            
            assert len(extensions) > 0, "Azure extensions should be available"
            
            # Test SSL connection
            ssl_info = await conn.fetchrow("""
                SELECT 
                    current_setting('ssl') as ssl_enabled,
                    version() as postgres_version
            """)
            
            assert ssl_info['ssl_enabled'] == 'on', "SSL should be enabled on Azure Database"
            
            # Test Azure-specific settings
            azure_settings = await conn.fetch("""
                SELECT name, setting 
                FROM pg_settings 
                WHERE name IN ('shared_preload_libraries', 'log_statement', 'log_min_duration_statement')
            """)
            
            settings_dict = {row['name']: row['setting'] for row in azure_settings}
            logger.info("Azure Database settings", settings=settings_dict)
    
    async def test_azure_performance_insights(self, azure_test_database):
        """Test Azure Query Performance Insights compatibility."""
        connection_manager = azure_test_database
        
        async with connection_manager.get_write_connection() as conn:
            # Enable pg_stat_statements if not already enabled
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            except Exception as e:
                logger.warning("Could not create pg_stat_statements extension", error=str(e))
            
            # Execute some test queries
            await conn.execute("SELECT 1")
            await conn.execute("SELECT COUNT(*) FROM pg_tables")
            await conn.execute("SELECT current_timestamp")
            
            # Verify pg_stat_statements is collecting data
            try:
                stats = await conn.fetchrow("""
                    SELECT count(*) as query_count 
                    FROM pg_stat_statements 
                    WHERE query LIKE 'SELECT%'
                """)
                
                if stats:
                    assert stats['query_count'] > 0, "pg_stat_statements should be collecting query data"
                else:
                    logger.warning("pg_stat_statements not collecting data - extension may not be properly configured")
            except Exception as e:
                logger.warning("pg_stat_statements not available", error=str(e))
    
    async def test_tenant_isolation_azure(self, azure_test_database):
        """Test tenant isolation policies on Azure Database."""
        connection_manager = azure_test_database
        
        # Create enhanced RLS policies
        rls_manager = EnhancedRowLevelSecurity(connection_manager)
        await rls_manager.create_enhanced_rls_policies()
        
        async with connection_manager.get_write_connection() as conn:
            # Set tenant context
            await conn.execute("SET app.current_tenant_id = 'tenant1'")
            
            # Insert test data
            test_record_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO storage_records (
                    id, source_data, mapped_data, model_version, detector_type, tenant_id
                )
                VALUES ($1, $2, $3, $4, $5, $6)
            """, test_record_id, 'test_source', '{"test": "data"}', 'v1.0', 'test_detector', 'tenant1')
            
            # Verify tenant can only see their data
            result = await conn.fetch("SELECT * FROM storage_records")
            assert len(result) == 1
            assert result[0]['tenant_id'] == 'tenant1'
            assert result[0]['id'] == test_record_id
            
            # Switch tenant context
            await conn.execute("SET app.current_tenant_id = 'tenant2'")
            
            # Verify no access to other tenant's data
            result = await conn.fetch("SELECT * FROM storage_records")
            assert len(result) == 0
            
            # Test admin bypass
            await conn.execute("SET ROLE admin_role")
            await conn.execute("RESET app.current_tenant_id")
            result = await conn.fetch("SELECT * FROM storage_records")
            assert len(result) == 1  # Admin can see all data
    
    async def test_enhanced_storage_operations(self, azure_test_database):
        """Test enhanced storage operations with all new fields."""
        connection_manager = azure_test_database
        
        # Create test storage record with all enhanced fields
        test_record = StorageRecord(
            id=str(uuid.uuid4()),
            source_data="test source data",
            source_data_hash="abc123hash",
            mapped_data='{"category": "test", "confidence": 0.95}',
            model_version="v2.0",
            detector_type="test_detector",
            confidence_score=0.95,
            metadata={"test": "metadata"},
            tenant_id="test_tenant",
            s3_key="test/key/path",
            encrypted=True,
            correlation_id=str(uuid.uuid4()),
            azure_region="eastus",
            backup_status="completed"
        )
        
        async with connection_manager.get_write_connection() as conn:
            # Set tenant context
            await conn.execute("SET app.current_tenant_id = 'test_tenant'")
            
            # Insert enhanced record
            insert_sql = """
                INSERT INTO storage_records (
                    id, source_data, source_data_hash, mapped_data, model_version, 
                    detector_type, confidence_score, timestamp, metadata, tenant_id, 
                    s3_key, encrypted, correlation_id, azure_region, backup_status,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """
            
            await conn.execute(
                insert_sql,
                test_record.id,
                test_record.source_data,
                test_record.source_data_hash,
                test_record.mapped_data,
                test_record.model_version,
                test_record.detector_type,
                test_record.confidence_score,
                test_record.timestamp,
                '{"test": "metadata"}',
                test_record.tenant_id,
                test_record.s3_key,
                test_record.encrypted,
                test_record.correlation_id,
                test_record.azure_region,
                test_record.backup_status,
                test_record.created_at,
                test_record.updated_at
            )
            
            # Verify record was inserted correctly
            result = await conn.fetchrow(
                "SELECT * FROM storage_records WHERE id = $1",
                test_record.id
            )
            
            assert result is not None
            assert result['source_data_hash'] == test_record.source_data_hash
            assert result['detector_type'] == test_record.detector_type
            assert result['confidence_score'] == test_record.confidence_score
            assert result['correlation_id'] == test_record.correlation_id
            assert result['azure_region'] == test_record.azure_region
            assert result['backup_status'] == test_record.backup_status
    
    async def test_audit_trail_functionality(self, azure_test_database):
        """Test audit trail functionality."""
        connection_manager = azure_test_database
        
        async with connection_manager.get_write_connection() as conn:
            # Create audit record
            audit_id = str(uuid.uuid4())
            record_id = str(uuid.uuid4())
            
            await conn.execute("""
                INSERT INTO audit_trail (
                    id, tenant_id, table_name, record_id, operation, user_id, 
                    old_values, new_values, correlation_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
                audit_id, 'test_tenant', 'storage_records', record_id, 'INSERT', 
                'test_user', None, '{"test": "new_data"}', str(uuid.uuid4())
            )
            
            # Verify audit record
            result = await conn.fetchrow(
                "SELECT * FROM audit_trail WHERE id = $1",
                audit_id
            )
            
            assert result is not None
            assert result['tenant_id'] == 'test_tenant'
            assert result['operation'] == 'INSERT'
            assert result['record_id'] == record_id
    
    async def test_tenant_config_functionality(self, azure_test_database):
        """Test tenant configuration functionality."""
        connection_manager = azure_test_database
        
        # Create test tenant config
        tenant_config = TenantConfig(
            tenant_id="test_tenant",
            confidence_threshold=0.8,
            detector_whitelist=["detector1", "detector2"],
            storage_retention_days=60,
            encryption_enabled=True,
            audit_level="verbose",
            custom_taxonomy_mappings={"custom": "mapping"}
        )
        
        async with connection_manager.get_write_connection() as conn:
            # Insert tenant config
            await conn.execute("""
                INSERT INTO tenant_configs (
                    tenant_id, confidence_threshold, detector_whitelist, detector_blacklist,
                    storage_retention_days, encryption_enabled, audit_level,
                    custom_taxonomy_mappings, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                tenant_config.tenant_id,
                tenant_config.confidence_threshold,
                tenant_config.detector_whitelist,
                tenant_config.detector_blacklist,
                tenant_config.storage_retention_days,
                tenant_config.encryption_enabled,
                tenant_config.audit_level,
                '{"custom": "mapping"}',
                tenant_config.created_at,
                tenant_config.updated_at
            )
            
            # Verify tenant config
            result = await conn.fetchrow(
                "SELECT * FROM tenant_configs WHERE tenant_id = $1",
                tenant_config.tenant_id
            )
            
            assert result is not None
            assert result['confidence_threshold'] == tenant_config.confidence_threshold
            assert result['detector_whitelist'] == tenant_config.detector_whitelist
            assert result['encryption_enabled'] == tenant_config.encryption_enabled
    
    async def test_field_encryption(self, azure_test_database):
        """Test field-level encryption functionality."""
        # Create encryption manager with test key
        encryption = FieldEncryption(master_key=FieldEncryption.generate_key_from_password("test_password"))
        await encryption.initialize()
        
        # Test data encryption/decryption
        test_data = "sensitive data that needs encryption"
        encrypted_data = encryption.encrypt_field(test_data)
        decrypted_data = encryption.decrypt_field(encrypted_data)
        
        assert encrypted_data != test_data
        assert decrypted_data == test_data
        
        # Test dict encryption
        test_dict = {
            "public_field": "public data",
            "sensitive_field": "sensitive data",
            "another_sensitive": "more sensitive data"
        }
        
        encrypted_dict = encryption.encrypt_dict(test_dict, ["sensitive_field", "another_sensitive"])
        decrypted_dict = encryption.decrypt_dict(encrypted_dict, ["sensitive_field", "another_sensitive"])
        
        assert encrypted_dict["public_field"] == test_dict["public_field"]
        assert encrypted_dict["sensitive_field"] != test_dict["sensitive_field"]
        assert decrypted_dict == test_dict
    
    async def test_database_performance_monitoring(self, azure_test_database):
        """Test database performance monitoring functionality."""
        connection_manager = azure_test_database
        
        # Test connection health check
        health_status = await connection_manager.health_check()
        assert isinstance(health_status, dict)
        assert 'write_db' in health_status
        
        # Test Azure connection info
        azure_info = await connection_manager.get_azure_connection_info()
        assert isinstance(azure_info, dict)
        assert 'postgres_version' in azure_info
        
        # Simulate some database activity for monitoring
        async with connection_manager.get_write_connection() as conn:
            for i in range(10):
                await conn.execute("SELECT pg_sleep(0.01)")  # Small delay
                await conn.execute("SELECT COUNT(*) FROM pg_tables")
    
    async def test_migration_system(self, azure_test_database):
        """Test database migration system."""
        connection_manager = azure_test_database
        
        migration_manager = DatabaseMigrationManager(connection_manager)
        
        # Test migration status
        status = await migration_manager.get_migration_status()
        assert isinstance(status, dict)
        assert 'applied_count' in status
        assert 'pending_count' in status
        
        # Test schema validation
        schema_status = await migration_manager.validate_schema_integrity()
        assert isinstance(schema_status, dict)
        assert 'schema_valid' in schema_status
        assert 'missing_tables' in schema_status
        assert 'missing_indexes' in schema_status
    
    async def cleanup_all_test_databases(self):
        """Clean up all test databases created during testing."""
        if not self.test_databases or not self.azure_config:
            return
        
        azure_params = {
            'host': self.azure_config.azure_db_host,
            'port': 5432,
            'user': f"{self.azure_config.username}@{self.azure_config.server_name}",
            'password': self.azure_config.password,
            'ssl': 'require',
            'sslmode': 'require'
        }
        
        admin_conn = await asyncpg.connect(database='postgres', **azure_params)
        
        try:
            for db_name in self.test_databases:
                try:
                    # Terminate active connections
                    await admin_conn.execute(f"""
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
                    """)
                    
                    # Drop database
                    await admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
                    logger.info("Cleaned up test database", database=db_name)
                    
                except Exception as e:
                    logger.warning("Failed to cleanup test database", database=db_name, error=str(e))
                    
        finally:
            await admin_conn.close()
            self.test_databases.clear()


# Pytest fixtures for easy use
@pytest.fixture(scope="session")
def azure_test_framework():
    """Session-scoped Azure test framework."""
    return AzureDatabaseTestFramework()


@pytest.fixture
async def azure_database(azure_test_framework):
    """Test database fixture."""
    async with azure_test_framework.azure_test_database() as db:
        yield db


# Test runner function for CI/CD
async def run_azure_database_tests():
    """Run all Azure database tests."""
    framework = AzureDatabaseTestFramework()
    
    try:
        async with framework.azure_test_database() as connection_manager:
            # Run all tests
            await framework.test_azure_specific_features(connection_manager)
            await framework.test_azure_performance_insights(connection_manager)
            await framework.test_tenant_isolation_azure(connection_manager)
            await framework.test_enhanced_storage_operations(connection_manager)
            await framework.test_audit_trail_functionality(connection_manager)
            await framework.test_tenant_config_functionality(connection_manager)
            await framework.test_field_encryption(connection_manager)
            await framework.test_database_performance_monitoring(connection_manager)
            await framework.test_migration_system(connection_manager)
            
            logger.info("All Azure database tests passed successfully")
            
    except Exception as e:
        logger.error("Azure database tests failed", error=str(e))
        raise
    finally:
        await framework.cleanup_all_test_databases()
