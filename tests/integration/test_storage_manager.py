"""
Tests for the StorageManager class.

This module tests the S3 and database integration functionality,
including WORM configuration, encryption, and lifecycle management.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_mapper.config.settings import Settings, StorageConfig
from llama_mapper.storage.manager import StorageBackend, StorageManager, StorageRecord


@pytest.fixture
def storage_config():
    """Create a test storage configuration."""
    return StorageConfig(
        s3_bucket="test-bucket",
        aws_region="us-east-1",
        storage_backend="postgresql",
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
        db_user="test_user",
        db_password="test_password",
        encryption_key="test-key-32-characters-long-12",
        retention_days=90,
    )


@pytest.fixture
def settings(storage_config):
    """Create test settings."""
    return Settings(storage=storage_config)


@pytest.fixture
def sample_record():
    """Create a sample storage record."""
    return StorageRecord(
        id=str(uuid.uuid4()),
        source_data="test detector output",
        mapped_data='{"taxonomy": ["HARM.SPEECH.Toxicity"], "confidence": 0.95}',
        model_version="llama-2-7b-v1.0",
        timestamp=datetime.now(timezone.utc),
        metadata={"detector": "test-detector", "tenant_id": "test-tenant"},
    )


class TestStorageManager:
    """Test cases for StorageManager."""

    @pytest.mark.asyncio
    async def test_initialization_postgresql(self, settings):
        """Test StorageManager initialization with PostgreSQL."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(
                StorageManager, "_create_postgresql_tables"
            ) as mock_create_tables,
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool
            mock_pool.return_value = AsyncMock()

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            assert storage_manager.backend == StorageBackend.POSTGRESQL
            mock_create_tables.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_clickhouse(self, storage_config):
        """Test StorageManager initialization with ClickHouse."""
        storage_config.storage_backend = "clickhouse"
        storage_config.db_port = 9000
        settings = Settings(storage=storage_config)

        with (
            patch("boto3.Session") as mock_session,
            patch("clickhouse_driver.Client") as mock_client,
            patch.object(
                StorageManager, "_create_clickhouse_tables"
            ) as mock_create_tables,
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock ClickHouse client
            mock_client.return_value.execute.return_value = None

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            assert storage_manager.backend == StorageBackend.CLICKHOUSE
            mock_create_tables.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_record_postgresql(self, settings, sample_record):
        """Test storing a record with PostgreSQL backend."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
            patch.object(StorageManager, "_init_encryption"),
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_s3_client.put_object.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool and connection
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = None
            mock_pool_instance = AsyncMock()
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.return_value = mock_pool_instance

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            # Store the record
            record_id = await storage_manager.store_record(sample_record)

            assert record_id == sample_record.id

            # Verify database insert was called
            mock_conn.execute.assert_called()

            # Verify S3 put_object was called
            mock_s3_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_record_from_database(self, settings, sample_record):
        """Test retrieving a record from the database."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
            patch.object(StorageManager, "_init_encryption"),
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool and connection
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": sample_record.id,
                "source_data": sample_record.source_data,
                "mapped_data": sample_record.mapped_data,
                "model_version": sample_record.model_version,
                "timestamp": sample_record.timestamp,
                "metadata": sample_record.metadata,
                "s3_key": "test-key",
                "encrypted": False,
            }
            mock_pool_instance = AsyncMock()
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.return_value = mock_pool_instance

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            # Retrieve the record
            retrieved_record = await storage_manager.retrieve_record(sample_record.id)

            assert retrieved_record is not None
            assert retrieved_record.id == sample_record.id
            assert retrieved_record.source_data == sample_record.source_data
            assert retrieved_record.mapped_data == sample_record.mapped_data

    @pytest.mark.asyncio
    async def test_cleanup_expired_records(self, settings):
        """Test cleanup of expired records."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
            patch.object(StorageManager, "_init_encryption"),
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool and connection
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = "DELETE 5"  # Simulate 5 records deleted
            mock_pool_instance = AsyncMock()
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.return_value = mock_pool_instance

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            # Run cleanup
            cleaned_count = await storage_manager.cleanup_expired_records()

            assert cleaned_count == 5
            mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_encryption_with_kms(self, storage_config):
        """Test encryption initialization with KMS."""
        storage_config.kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/test-key"
        settings = Settings(storage=storage_config)

        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
        ):
            # Mock KMS client
            mock_kms_client = MagicMock()
            mock_kms_client.generate_data_key.return_value = {
                "Plaintext": b"test-encryption-key-32-characters"
            }

            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None

            mock_session.return_value.client.side_effect = lambda service: {
                "s3": mock_s3_client,
                "kms": mock_kms_client,
            }[service]

            # Mock database pool
            mock_pool.return_value = AsyncMock()

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            # Verify KMS was called
            mock_kms_client.generate_data_key.assert_called_once()
            assert storage_manager._fernet is not None

    @pytest.mark.asyncio
    async def test_worm_policy_setup(self, settings):
        """Test WORM policy configuration."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
            patch.object(StorageManager, "_init_encryption"),
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_s3_client.get_object_lock_configuration.side_effect = Exception(
                "ObjectLockConfigurationNotFoundError"
            )
            mock_s3_client.put_object_lock_configuration.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool
            mock_pool.return_value = AsyncMock()

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()

            # Verify WORM policy was configured
            mock_s3_client.put_object_lock_configuration.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_s3_failure(self, settings):
        """Test error handling when S3 initialization fails."""
        with patch("boto3.Session") as mock_session:
            # Mock S3 client to raise an error
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.side_effect = Exception("S3 connection failed")
            mock_session.return_value.client.return_value = mock_s3_client

            storage_manager = StorageManager(settings.storage)

            with pytest.raises(Exception, match="S3 connection failed"):
                await storage_manager.initialize()

    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self, settings):
        """Test error handling when database initialization fails."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
        ):
            # Mock S3 client (successful)
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool to raise an error
            mock_pool.side_effect = Exception("Database connection failed")

            storage_manager = StorageManager(settings.storage)

            with pytest.raises(Exception, match="Database connection failed"):
                await storage_manager.initialize()

    @pytest.mark.asyncio
    async def test_close_connections(self, settings):
        """Test proper cleanup of connections."""
        with (
            patch("boto3.Session") as mock_session,
            patch("asyncpg.create_pool") as mock_pool,
            patch.object(StorageManager, "_create_postgresql_tables"),
            patch.object(StorageManager, "_init_encryption"),
        ):
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client

            # Mock database pool
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance

            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()
            await storage_manager.close()

            # Verify pool was closed
            mock_pool_instance.close.assert_called_once()


class TestStorageRecord:
    """Test cases for StorageRecord dataclass."""

    def test_storage_record_creation(self):
        """Test creating a StorageRecord."""
        record = StorageRecord(
            id="test-id",
            source_data="test source",
            mapped_data="test mapped",
            model_version="v1.0",
            timestamp=datetime.now(timezone.utc),
            metadata={"key": "value"},
        )

        assert record.id == "test-id"
        assert record.source_data == "test source"
        assert record.mapped_data == "test mapped"
        assert record.model_version == "v1.0"
        assert record.metadata == {"key": "value"}
        assert record.s3_key is None
        assert record.encrypted is False


if __name__ == "__main__":
    pytest.main([__file__])
