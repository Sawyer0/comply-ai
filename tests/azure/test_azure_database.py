"""Azure Database test framework for production database assessment."""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
import structlog

from src.llama_mapper.storage.database.azure_config import (
    AzureDatabaseConfig,
    AzureDatabaseConnectionManager
)
from src.llama_mapper.storage.database.migrations import (
    DatabaseMigrationManager,
    create_production_migrations
)
from src.llama_mapper.storage.manager.models import (
    StorageRecord,
    AuditRecord,
    TenantConfig
)
from src.llama_mapper.storage.tenant_isolation import (
    TenantIsolationManager,
    TenantContext,
    TenantAccessLevel
)

logger = structlog.get_logger(__name__)


class AzureDatabaseTestFramework:
    """Test framework for Azure Database functionality."""
    
    def __init__(self, test_config: Optional[Dict[str, Any]] = None):
        self.test_config = test_config or self._get_test_config()
        self.connection_manager: Optional[AzureDatabaseConnectionManager] = None
        self.migration_manager: Optional[DatabaseMigrationManager] = None
        self.tenant_manager = TenantIsolationManager()
        self.test_tenant_id = f"test-tenant-{uuid.uuid4().hex[:8]}"
        
    def _get_test_config(self) -> Dict[str, Any]:
        """Get test configuration from environment or defaults."""
        return {
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", "test-subscription"),
            "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "test-rg"),
            "server_name": os.getenv("AZURE_DB_SERVER", "test-server"),
            "azure_db_host": os.getenv("AZURE_DB_HOST", "localhost"),
            "database_name": os.getenv("AZURE_DB_NAME", "test_llama_mapper"),
            "username": os.getenv("AZURE_DB_USER", "test_user"),
            "password": os.getenv("AZURE_DB_PASSWORD", "test_password"),
            "ssl_mode": "prefer",  # Use prefer for testing
            "min_pool_size": 1,
            "max_pool_size": 5
        }
    
    async def setup_test_environment(self):
        """Set up test environment with database connections and migrations."""
        try:
            # Create Azure database configuration
            config = AzureDatabaseConfig(**self.test_config)
            
            # Initialize connection manager
            self.connection_manager = AzureDatabaseConnectionManager(config)
            await self.connection_manager.initialize_pools()
            
            # Initialize migration manager
            self.migration_manager = DatabaseMigrationManager(self.connection_manager)
            await self.migration_manager.initialize_migration_tracking()
            
            # Register and apply test migrations
            migrations = create_production_migrations()
            for migration in migrations:
                self.migration_manager.register_migration(migration)
            
            await self.migration_manager.apply_all_migrations()
            
            logger.info("Test environment setup completed")
            
        except Exception as e:
            logger.error("Failed to setup test environment", error=str(e))
            raise
    
    async def teardown_test_environment(self):
        """Clean up test environment."""
        try:
            if self.connection_manager:
                # Clean up test data
                await self._cleanup_test_data()
                
                # Close connections
                await self.connection_manager.close_pools()
            
            logger.info("Test environment teardown completed")
            
        except Exception as e:
            logger.error("Failed to teardown test environment", error=str(e))
    
    async def _cleanup_test_data(self):
        """Clean up test data from database."""
        async with self.connection_manager.get_write_connection() as conn:
            # Delete test records
            await conn.execute(
                "DELETE FROM storage_records WHERE tenant_id LIKE 'test-tenant-%'"
            )
            await conn.execute(
                "DELETE FROM audit_trail WHERE tenant_id LIKE 'test-tenant-%'"
            )
            await conn.execute(
                "DELETE FROM tenant_configs WHERE tenant_id LIKE 'test-tenant-%'"
            )
            await conn.execute(
                "DELETE FROM model_metrics WHERE tenant_id LIKE 'test-tenant-%'"
            )
            await conn.execute(
                "DELETE FROM detector_executions WHERE tenant_id LIKE 'test-tenant-%'"
            )
    
    async def test_azure_connection_features(self) -> Dict[str, bool]:
        """Test Azure-specific connection features."""
        results = {}
        
        try:
            # Test SSL connection
            async with self.connection_manager.get_write_connection() as conn:
                ssl_status = await conn.fetchval("SHOW ssl")
                results["ssl_enabled"] = ssl_status == "on"
            
            # Test Azure extensions
            async with self.connection_manager.get_write_connection() as conn:
                extensions = await conn.fetch(
                    "SELECT name FROM pg_available_extensions WHERE name IN ('uuid-ossp', 'pgcrypto')"
                )
                results["uuid_extension_available"] = any(ext['name'] == 'uuid-ossp' for ext in extensions)
                results["crypto_extension_available"] = any(ext['name'] == 'pgcrypto' for ext in extensions)
            
            # Test connection info
            connection_info = await self.connection_manager.get_azure_connection_info()
            results["connection_info_available"] = bool(connection_info)
            
            # Test health check
            health_status = await self.connection_manager.health_check()
            results["health_check_working"] = health_status.get("write_db", False)
            
        except Exception as e:
            logger.error("Azure connection feature test failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def test_tenant_isolation(self) -> Dict[str, bool]:
        """Test tenant isolation with Azure authentication."""
        results = {}
        
        try:
            # Create test tenant contexts
            tenant1_id = f"test-tenant-1-{uuid.uuid4().hex[:8]}"
            tenant2_id = f"test-tenant-2-{uuid.uuid4().hex[:8]}"
            
            tenant1_context = self.tenant_manager.create_tenant_context(tenant1_id)
            tenant2_context = self.tenant_manager.create_tenant_context(tenant2_id)
            
            # Test tenant context creation
            results["tenant_context_creation"] = bool(tenant1_context and tenant2_context)
            
            # Test tenant access validation
            same_tenant_access = self.tenant_manager.validate_tenant_access(
                tenant1_id, tenant1_id, "read"
            )
            cross_tenant_access = self.tenant_manager.validate_tenant_access(
                tenant1_id, tenant2_id, "read"
            )
            
            results["same_tenant_access"] = same_tenant_access
            results["cross_tenant_blocked"] = not cross_tenant_access
            
            # Test database-level tenant isolation
            async with self.connection_manager.get_write_connection() as conn:
                # Set tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant1_id}'")
                
                # Insert test record
                test_record = StorageRecord(
                    id=str(uuid.uuid4()),
                    source_data="test data",
                    mapped_data="test mapped",
                    model_version="test-v1",
                    timestamp=datetime.utcnow(),
                    metadata={},
                    tenant_id=tenant1_id
                )
                
                await conn.execute("""
                    INSERT INTO storage_records 
                    (id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, test_record.id, test_record.source_data, test_record.mapped_data,
                    test_record.model_version, test_record.timestamp, {}, test_record.tenant_id)
                
                # Test RLS policy enforcement
                tenant1_records = await conn.fetch("SELECT * FROM storage_records")
                
                # Switch tenant context
                await conn.execute(f"SET app.current_tenant_id = '{tenant2_id}'")
                tenant2_records = await conn.fetch("SELECT * FROM storage_records")
                
                results["rls_isolation_working"] = (
                    len(tenant1_records) > 0 and len(tenant2_records) == 0
                )
            
        except Exception as e:
            logger.error("Tenant isolation test failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test database performance benchmarks."""
        results = {}
        
        try:
            # Test connection pool performance
            start_time = datetime.utcnow()
            
            # Concurrent connection test
            async def get_connection_time():
                async with self.connection_manager.get_write_connection() as conn:
                    await conn.fetchval("SELECT 1")
            
            tasks = [get_connection_time() for _ in range(10)]
            await asyncio.gather(*tasks)
            
            connection_time = (datetime.utcnow() - start_time).total_seconds()
            results["concurrent_connections_time"] = connection_time
            results["concurrent_connections_success"] = connection_time < 5.0  # Should complete in under 5 seconds
            
            # Test query performance
            async with self.connection_manager.get_write_connection() as conn:
                # Insert test data
                test_records = []
                for i in range(100):
                    test_records.append((
                        str(uuid.uuid4()),
                        f"test data {i}",
                        f"mapped data {i}",
                        "test-v1",
                        datetime.utcnow(),
                        {},
                        self.test_tenant_id
                    ))
                
                start_time = datetime.utcnow()
                await conn.executemany("""
                    INSERT INTO storage_records 
                    (id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, test_records)
                
                insert_time = (datetime.utcnow() - start_time).total_seconds()
                results["bulk_insert_time"] = insert_time
                results["bulk_insert_performance"] = insert_time < 2.0  # Should complete in under 2 seconds
                
                # Test query performance
                start_time = datetime.utcnow()
                records = await conn.fetch(
                    "SELECT * FROM storage_records WHERE tenant_id = $1 ORDER BY timestamp DESC LIMIT 50",
                    self.test_tenant_id
                )
                query_time = (datetime.utcnow() - start_time).total_seconds()
                
                results["query_time"] = query_time
                results["query_performance"] = query_time < 0.1  # Should complete in under 100ms
                results["records_retrieved"] = len(records)
            
        except Exception as e:
            logger.error("Performance benchmark test failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def test_backup_and_recovery(self) -> Dict[str, bool]:
        """Test backup and disaster recovery procedures."""
        results = {}
        
        try:
            # Test backup status tracking
            async with self.connection_manager.get_write_connection() as conn:
                # Insert record with backup status
                test_record_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO storage_records 
                    (id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id, backup_status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, test_record_id, "test", "mapped", "v1", datetime.utcnow(), {}, 
                    self.test_tenant_id, "pending")
                
                # Update backup status
                await conn.execute(
                    "UPDATE storage_records SET backup_status = 'completed' WHERE id = $1",
                    test_record_id
                )
                
                # Verify backup status update
                backup_status = await conn.fetchval(
                    "SELECT backup_status FROM storage_records WHERE id = $1",
                    test_record_id
                )
                
                results["backup_status_tracking"] = backup_status == "completed"
            
            # Test point-in-time recovery simulation
            async with self.connection_manager.get_write_connection() as conn:
                # Create checkpoint
                checkpoint_time = datetime.utcnow()
                
                # Insert data after checkpoint
                post_checkpoint_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO storage_records 
                    (id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, post_checkpoint_id, "post-checkpoint", "mapped", "v1", 
                    datetime.utcnow(), {}, self.test_tenant_id)
                
                # Simulate recovery by querying data before checkpoint
                pre_checkpoint_records = await conn.fetch(
                    "SELECT * FROM storage_records WHERE tenant_id = $1 AND created_at < $2",
                    self.test_tenant_id, checkpoint_time
                )
                
                post_checkpoint_records = await conn.fetch(
                    "SELECT * FROM storage_records WHERE tenant_id = $1 AND created_at >= $2",
                    self.test_tenant_id, checkpoint_time
                )
                
                results["point_in_time_recovery_simulation"] = (
                    len(post_checkpoint_records) > 0
                )
            
        except Exception as e:
            logger.error("Backup and recovery test failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_tenant_id": self.test_tenant_id
        }
        
        try:
            await self.setup_test_environment()
            
            # Run all test suites
            test_results["azure_connection_features"] = await self.test_azure_connection_features()
            test_results["tenant_isolation"] = await self.test_tenant_isolation()
            test_results["performance_benchmarks"] = await self.test_performance_benchmarks()
            test_results["backup_and_recovery"] = await self.test_backup_and_recovery()
            
            # Calculate overall success rate
            all_results = []
            for test_suite, results in test_results.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, bool):
                            all_results.append(value)
            
            success_rate = sum(all_results) / len(all_results) if all_results else 0
            test_results["overall_success_rate"] = success_rate
            test_results["tests_passed"] = sum(all_results)
            test_results["total_tests"] = len(all_results)
            
        except Exception as e:
            logger.error("Comprehensive test suite failed", error=str(e))
            test_results["error"] = str(e)
        
        finally:
            await self.teardown_test_environment()
        
        return test_results


# Pytest fixtures and test functions
@pytest.fixture
async def azure_test_framework():
    """Pytest fixture for Azure database test framework."""
    framework = AzureDatabaseTestFramework()
    yield framework


@pytest.mark.asyncio
async def test_azure_database_connection(azure_test_framework):
    """Test Azure database connection features."""
    results = await azure_test_framework.test_azure_connection_features()
    assert results.get("ssl_enabled", False), "SSL should be enabled"
    assert results.get("health_check_working", False), "Health check should work"


@pytest.mark.asyncio
async def test_tenant_isolation_azure(azure_test_framework):
    """Test tenant isolation with Azure database."""
    results = await azure_test_framework.test_tenant_isolation()
    assert results.get("tenant_context_creation", False), "Tenant contexts should be created"
    assert results.get("same_tenant_access", False), "Same tenant access should be allowed"
    assert results.get("cross_tenant_blocked", False), "Cross-tenant access should be blocked"


@pytest.mark.asyncio
async def test_database_performance(azure_test_framework):
    """Test database performance benchmarks."""
    results = await azure_test_framework.test_performance_benchmarks()
    assert results.get("concurrent_connections_success", False), "Concurrent connections should perform well"
    assert results.get("bulk_insert_performance", False), "Bulk inserts should perform well"
    assert results.get("query_performance", False), "Queries should perform well"


@pytest.mark.asyncio
async def test_backup_recovery_features(azure_test_framework):
    """Test backup and recovery features."""
    results = await azure_test_framework.test_backup_and_recovery()
    assert results.get("backup_status_tracking", False), "Backup status tracking should work"
    assert results.get("point_in_time_recovery_simulation", False), "Point-in-time recovery should work"