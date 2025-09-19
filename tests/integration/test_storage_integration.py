"""
Integration tests for StorageManager with real configuration.

This module tests the StorageManager with actual settings to ensure
proper integration with the configuration system.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from llama_mapper.config.settings import Settings
from llama_mapper.storage.manager import StorageManager, StorageRecord, StorageBackend


class TestStorageIntegration:
    """Integration tests for StorageManager."""
    
    def test_storage_manager_creation_with_settings(self):
        """Test creating StorageManager with default settings."""
        settings = Settings()
        storage_manager = StorageManager(settings.storage)
        
        assert storage_manager.settings == settings.storage
        assert storage_manager.backend == StorageBackend.POSTGRESQL  # Default backend
        assert storage_manager._s3_client is None  # Not initialized yet
        assert storage_manager._db_pool is None  # Not initialized yet
    
    def test_storage_config_validation(self):
        """Test that storage configuration is properly validated."""
        settings = Settings()
        
        # Check default values
        assert settings.storage.storage_backend == "postgresql"
        assert settings.storage.db_host == "localhost"
        assert settings.storage.db_port == 5432
        assert settings.storage.retention_days == 90
        assert settings.storage.s3_retention_years == 7
        assert settings.storage.aws_region == "us-east-1"
    
    def test_clickhouse_backend_selection(self):
        """Test selecting ClickHouse backend."""
        settings = Settings(storage__storage_backend="clickhouse")
        storage_manager = StorageManager(settings.storage)
        
        assert storage_manager.backend == StorageBackend.CLICKHOUSE
    
    @pytest.mark.asyncio
    async def test_storage_manager_initialization_mock(self):
        """Test StorageManager initialization with mocked dependencies."""
        settings = Settings()
        
        with patch('boto3.Session') as mock_session, \
             patch('asyncpg.create_pool') as mock_pool, \
             patch.object(StorageManager, '_create_postgresql_tables') as mock_create_tables, \
             patch.object(StorageManager, '_init_encryption') as mock_init_encryption, \
             patch.object(StorageManager, '_setup_worm_policy') as mock_worm_policy:
            
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.return_value = None
            mock_session.return_value.client.return_value = mock_s3_client
            
            # Mock database pool
            mock_pool.return_value = AsyncMock()
            
            storage_manager = StorageManager(settings.storage)
            await storage_manager.initialize()
            
            # Verify all initialization steps were called
            mock_create_tables.assert_called_once()
            mock_init_encryption.assert_called_once()
            mock_worm_policy.assert_called_once()
    
    def test_storage_record_structure(self):
        """Test StorageRecord structure matches requirements."""
        from datetime import datetime, timezone
        
        record = StorageRecord(
            id="test-record-123",
            source_data="detector output data",
            mapped_data='{"taxonomy": ["HARM.SPEECH.Toxicity"], "confidence": 0.95}',
            model_version="llama-3-8b-instruct-v1.0",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "tenant_id": "tenant-123",
                "detector_type": "deberta-toxicity",
                "taxonomy_hit": "HARM.SPEECH.Toxicity"
            }
        )
        
        # Verify all required fields are present
        assert record.id is not None
        assert record.source_data is not None
        assert record.mapped_data is not None
        assert record.model_version is not None
        assert record.timestamp is not None
        assert record.metadata is not None
        
        # Verify privacy-first approach - metadata only, no raw inputs
        assert "tenant_id" in record.metadata
        assert "detector_type" in record.metadata
        assert "taxonomy_hit" in record.metadata
    
    def test_settings_environment_override(self):
        """Test that environment variables can override storage settings."""
        import os
        
        # Set environment variables
        test_env = {
            "STORAGE__S3_BUCKET": "test-override-bucket",
            "STORAGE__DB_HOST": "test-db-host",
            "STORAGE__RETENTION_DAYS": "30"
        }
        
        with patch.dict(os.environ, test_env):
            settings = Settings()
            
            assert settings.storage.s3_bucket == "test-override-bucket"
            assert settings.storage.db_host == "test-db-host"
            assert settings.storage.retention_days == 30


if __name__ == "__main__":
    pytest.main([__file__])