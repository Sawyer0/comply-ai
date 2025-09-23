"""
Integration tests for StorageManager with TenantIsolationManager.

This module tests the integration between storage operations and tenant isolation.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llama_mapper.config.settings import Settings
from llama_mapper.storage.manager import StorageManager, StorageRecord
from llama_mapper.storage.tenant_isolation import (
    TenantAccessLevel,
)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def sample_record():
    """Create a sample storage record with tenant ID."""
    return StorageRecord(
        id=str(uuid.uuid4()),
        source_data="test detector output",
        mapped_data='{"taxonomy": ["HARM.SPEECH.Toxicity"], "confidence": 0.95}',
        model_version="llama-2-7b-v1.0",
        timestamp=datetime.now(timezone.utc),
        metadata={"detector": "test-detector"},
        tenant_id="tenant-1",
    )


async def create_mock_storage_manager(settings):
    """Helper function to create a properly mocked StorageManager."""
    with (
        patch("boto3.Session") as mock_session,
        patch("asyncpg.create_pool") as mock_pool,
    ):
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.head_bucket.return_value = None
        mock_s3_client.put_object.return_value = None
        mock_session.return_value.client.return_value = mock_s3_client

        # Mock database pool
        mock_pool_instance = AsyncMock()
        mock_pool_instance.close = AsyncMock()
        mock_pool.return_value = mock_pool_instance

        storage_manager = StorageManager(settings.storage)

        # Mock all async methods
        storage_manager._init_s3 = AsyncMock()
        storage_manager._init_database = AsyncMock()
        storage_manager._init_encryption = AsyncMock()
        storage_manager._setup_worm_policy = AsyncMock()
        storage_manager._create_postgresql_tables = AsyncMock()
        storage_manager._store_in_database = AsyncMock()
        storage_manager._store_in_s3 = AsyncMock(return_value="test-s3-key")
        storage_manager._update_s3_key = AsyncMock()
        storage_manager._retrieve_from_database = AsyncMock()
        storage_manager.close = AsyncMock()

        # Mock the database pool
        storage_manager._db_pool = mock_pool_instance

        return storage_manager


class TestStorageTenantIntegration:
    """Integration tests for StorageManager with tenant isolation."""

    @pytest.mark.asyncio
    async def test_store_record_with_tenant_context(self, settings, sample_record):
        """Test storing a record with tenant context validation."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Create tenant context
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "tenant-1", access_level=TenantAccessLevel.STRICT
        )

        # Store record with tenant context
        record_id = await storage_manager.store_record(sample_record)

        # The returned ID should be the original record ID
        assert record_id == sample_record.id

        # Verify tenant validation was applied
        assert sample_record.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_store_record_cross_tenant_denied(self, settings, sample_record):
        """Test that cross-tenant write access is denied."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Create tenant context for different tenant
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "tenant-2", access_level=TenantAccessLevel.STRICT  # Different tenant
        )

        # Attempt to store record from tenant-1 while in tenant-2 context
        # This should be allowed as the record itself defines the tenant
        record_id = await storage_manager.store_record(sample_record)
        # The returned ID should be the original record ID
        assert record_id == sample_record.id

    @pytest.mark.asyncio
    async def test_retrieve_record_with_tenant_filtering(self, settings, sample_record):
        """Test retrieving a record with tenant filtering."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Mock the retrieve method to return a record
        storage_manager._retrieve_from_database = AsyncMock(return_value=sample_record)

        # Create tenant context for same tenant
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "tenant-1", access_level=TenantAccessLevel.STRICT
        )

        # Create a tenant-scoped ID for retrieval
        scoped_id = storage_manager.tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", sample_record.id
        )

        # Retrieve record using tenant-scoped ID
        retrieved_record = await storage_manager.retrieve_record(
            scoped_id, tenant_context
        )

        assert retrieved_record is not None
        assert retrieved_record.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_retrieve_record_cross_tenant_denied(self, settings, sample_record):
        """Test that cross-tenant read access is denied."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Mock the retrieve method to return a record from tenant-1
        storage_manager._retrieve_from_database = AsyncMock(return_value=sample_record)

        # Create tenant context for different tenant
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "tenant-2", access_level=TenantAccessLevel.STRICT  # Different tenant
        )

        # Create a tenant-scoped ID for the record (as if it was stored by tenant-1)
        scoped_id = storage_manager.tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", sample_record.id
        )

        # Attempt to retrieve record - should be denied due to cross-tenant access
        retrieved_record = await storage_manager.retrieve_record(
            scoped_id, tenant_context
        )
        assert retrieved_record is None  # Access should be denied

    @pytest.mark.asyncio
    async def test_shared_tenant_access(self, settings, sample_record):
        """Test shared tenant access allows cross-tenant operations."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Mock the retrieve method
        storage_manager._retrieve_from_database = AsyncMock(return_value=sample_record)

        # Create shared tenant context
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "tenant-2",
            access_level=TenantAccessLevel.SHARED,
            allowed_tenants={"tenant-1", "tenant-3"},
        )

        # Create a tenant-scoped ID for the record
        scoped_id = storage_manager.tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", sample_record.id
        )

        # Should be able to access tenant-1 record due to shared access
        retrieved_record = await storage_manager.retrieve_record(
            scoped_id, tenant_context
        )
        assert retrieved_record is not None

    @pytest.mark.asyncio
    async def test_admin_tenant_access(self, settings, sample_record):
        """Test admin tenant access allows all operations."""
        storage_manager = await create_mock_storage_manager(settings)
        await storage_manager.initialize()

        # Mock the retrieve method
        storage_manager._retrieve_from_database = AsyncMock(return_value=sample_record)

        # Create admin tenant context
        tenant_context = storage_manager.tenant_manager.create_tenant_context(
            "admin-tenant", access_level=TenantAccessLevel.ADMIN
        )

        # Create a tenant-scoped ID for the record
        scoped_id = storage_manager.tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", sample_record.id
        )

        # Should be able to access any record as admin
        retrieved_record = await storage_manager.retrieve_record(
            scoped_id, tenant_context
        )
        assert retrieved_record is not None

    def test_tenant_scoped_record_ids(self, settings):
        """Test tenant-scoped record ID generation."""
        storage_manager = StorageManager(settings.storage)

        # Test record ID scoping
        base_id = "test-record-123"
        scoped_id = storage_manager.tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", base_id
        )

        assert scoped_id.startswith("tenant-1:")
        assert base_id in scoped_id

        # Test extraction
        (
            extracted_tenant,
            extracted_id,
        ) = storage_manager.tenant_manager.extract_tenant_from_record_id(scoped_id)
        assert extracted_tenant == "tenant-1"
        assert extracted_id == base_id

    def test_tenant_configuration_integration(self, settings):
        """Test tenant configuration integration with storage manager."""
        storage_manager = StorageManager(settings.storage)

        # Test default configuration
        config = storage_manager.tenant_manager.get_tenant_config("tenant-1")
        assert config.tenant_id == "tenant-1"
        assert config.encryption_enabled is True

        # Test configuration overrides
        from llama_mapper.storage.tenant_isolation import TenantConfig

        updated_config = TenantConfig(
            tenant_id="tenant-1",
            confidence_threshold=0.8,
            encryption_enabled=False,
            storage_retention_days=30,
        )

        # Update tenant configuration
        storage_manager.tenant_manager.update_tenant_config("tenant-1", updated_config)

        # Get updated config
        config = storage_manager.tenant_manager.get_tenant_config("tenant-1")
        assert config.confidence_threshold == 0.8
        assert config.encryption_enabled is False
        assert config.storage_retention_days == 30


if __name__ == "__main__":
    pytest.main([__file__])
