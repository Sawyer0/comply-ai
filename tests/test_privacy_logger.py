"""
Tests for the PrivacyLogger class.

This module tests the privacy-first logging system that stores only
metadata and never persists raw detector inputs.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from llama_mapper.config.settings import Settings
from llama_mapper.storage.privacy_logger import (
    PrivacyLogger, PrivacyLogEntry, LogLevel, EventType
)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def privacy_logger(settings):
    """Create a PrivacyLogger instance."""
    return PrivacyLogger(settings)


class TestPrivacyLogEntry:
    """Test cases for PrivacyLogEntry."""
    
    def test_privacy_log_entry_creation(self):
        """Test creating a PrivacyLogEntry."""
        entry = PrivacyLogEntry(
            event_id="test-event-123",
            timestamp=datetime.now(timezone.utc),
            tenant_id="tenant-1",
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO,
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95
        )
        
        assert entry.event_id == "test-event-123"
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.MAPPING_SUCCESS
        assert entry.log_level == LogLevel.INFO
        assert entry.detector_type == "deberta-toxicity"
        assert entry.taxonomy_hit == "HARM.SPEECH.Toxicity"
        assert entry.confidence_score == 0.95
    
    def test_privacy_log_entry_to_dict(self):
        """Test converting PrivacyLogEntry to dictionary."""
        timestamp = datetime.now(timezone.utc)
        entry = PrivacyLogEntry(
            event_id="test-event-123",
            timestamp=timestamp,
            tenant_id="tenant-1",
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO
        )
        
        data = entry.to_dict()
        
        assert data['event_id'] == "test-event-123"
        assert data['timestamp'] == timestamp.isoformat()
        assert data['tenant_id'] == "tenant-1"
        assert data['event_type'] == "mapping_success"
        assert data['log_level'] == "info"
    
    def test_privacy_log_entry_to_json(self):
        """Test converting PrivacyLogEntry to JSON."""
        entry = PrivacyLogEntry(
            event_id="test-event-123",
            timestamp=datetime.now(timezone.utc),
            tenant_id="tenant-1",
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO
        )
        
        json_str = entry.to_json()
        
        assert "test-event-123" in json_str
        assert "tenant-1" in json_str
        assert "mapping_success" in json_str


class TestPrivacyLogger:
    """Test cases for PrivacyLogger."""
    
    def test_privacy_logger_initialization(self, privacy_logger):
        """Test PrivacyLogger initialization."""
        assert privacy_logger.settings is not None
        assert privacy_logger.privacy_mode is True  # Default setting
        assert privacy_logger.max_message_length == 500  # Default setting
        assert len(privacy_logger._log_entries) == 0
    
    def test_sanitize_metadata_privacy_mode(self, privacy_logger):
        """Test metadata sanitization in privacy mode."""
        metadata = {
            'detector_type': 'deberta-toxicity',
            'sensitive_data': 'this should be removed',
            'user_input': 'raw detector input',
            'confidence_threshold': 0.6,
            'very_long_string': 'x' * 1000
        }
        
        sanitized = privacy_logger._sanitize_metadata(metadata)
        
        # Should keep safe keys
        assert 'detector_type' in sanitized
        assert 'confidence_threshold' in sanitized
        
        # Should remove unsafe keys
        assert 'sensitive_data' not in sanitized
        assert 'user_input' not in sanitized
        
        # Should truncate long strings
        assert len(sanitized.get('very_long_string', '')) <= privacy_logger.max_message_length + 3  # +3 for "..."
    
    def test_sanitize_metadata_non_privacy_mode(self, settings):
        """Test metadata sanitization with privacy mode disabled."""
        settings.logging.privacy_mode = False
        privacy_logger = PrivacyLogger(settings)
        
        metadata = {
            'detector_type': 'deberta-toxicity',
            'sensitive_data': 'this should be kept',
            'user_input': 'raw detector input'
        }
        
        sanitized = privacy_logger._sanitize_metadata(metadata)
        
        # Should keep all keys when privacy mode is disabled
        assert sanitized == metadata
    
    @pytest.mark.asyncio
    async def test_log_mapping_success(self, privacy_logger):
        """Test logging a successful mapping operation."""
        event_id = await privacy_logger.log_mapping_success(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="llama-3-8b-v1.0",
            processing_time_ms=150,
            metadata={"framework": "SOC2"}
        )
        
        assert event_id is not None
        assert len(privacy_logger._log_entries) == 1
        
        entry = privacy_logger._log_entries[0]
        assert entry.event_id == event_id
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.MAPPING_SUCCESS
        assert entry.log_level == LogLevel.INFO
        assert entry.detector_type == "deberta-toxicity"
        assert entry.taxonomy_hit == "HARM.SPEECH.Toxicity"
        assert entry.confidence_score == 0.95
        assert entry.model_version == "llama-3-8b-v1.0"
        assert entry.processing_time_ms == 150
    
    @pytest.mark.asyncio
    async def test_log_mapping_failure(self, privacy_logger):
        """Test logging a failed mapping operation."""
        event_id = await privacy_logger.log_mapping_failure(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            error_code="VALIDATION_ERROR",
            error_category="schema",
            model_version="llama-3-8b-v1.0",
            processing_time_ms=75,
            metadata={"retry_count": 3}
        )
        
        assert event_id is not None
        assert len(privacy_logger._log_entries) == 1
        
        entry = privacy_logger._log_entries[0]
        assert entry.event_id == event_id
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.MAPPING_FAILURE
        assert entry.log_level == LogLevel.ERROR
        assert entry.detector_type == "deberta-toxicity"
        assert entry.error_code == "VALIDATION_ERROR"
        assert entry.error_category == "schema"
        assert entry.model_version == "llama-3-8b-v1.0"
        assert entry.processing_time_ms == 75
    
    @pytest.mark.asyncio
    async def test_log_confidence_fallback(self, privacy_logger):
        """Test logging confidence fallback."""
        event_id = await privacy_logger.log_confidence_fallback(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            original_confidence=0.45,
            threshold=0.6,
            fallback_taxonomy="OTHER.Unknown",
            model_version="llama-3-8b-v1.0",
            metadata={"fallback_reason": "low_confidence"}
        )
        
        assert event_id is not None
        assert len(privacy_logger._log_entries) == 1
        
        entry = privacy_logger._log_entries[0]
        assert entry.event_id == event_id
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.CONFIDENCE_FALLBACK
        assert entry.log_level == LogLevel.WARNING
        assert entry.detector_type == "deberta-toxicity"
        assert entry.taxonomy_hit == "OTHER.Unknown"
        assert entry.confidence_score == 0.45
        assert entry.model_version == "llama-3-8b-v1.0"
        assert entry.metadata['confidence_threshold'] == 0.6
        assert entry.metadata['fallback_used'] is True
    
    @pytest.mark.asyncio
    async def test_log_tenant_access_denied(self, privacy_logger):
        """Test logging tenant access denied."""
        event_id = await privacy_logger.log_tenant_access_denied(
            tenant_id="tenant-1",
            requested_resource="storage_records",
            access_level="strict",
            user_id="user-123",
            session_id="session-456",
            metadata={"ip_address": "192.168.1.1"}
        )
        
        assert event_id is not None
        assert len(privacy_logger._log_entries) == 1
        
        entry = privacy_logger._log_entries[0]
        assert entry.event_id == event_id
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.TENANT_ACCESS_DENIED
        assert entry.log_level == LogLevel.WARNING
        assert entry.user_id == "user-123"
        assert entry.session_id == "session-456"
        assert entry.metadata['requested_resource'] == "storage_records"
        assert entry.metadata['access_level'] == "strict"
    
    @pytest.mark.asyncio
    async def test_log_audit_trail(self, privacy_logger):
        """Test logging audit trail."""
        event_id = await privacy_logger.log_audit_trail(
            tenant_id="tenant-1",
            action="create",
            resource="mapping_record",
            user_id="user-123",
            session_id="session-456",
            request_id="req-789",
            metadata={"compliance_framework": "SOC2"}
        )
        
        assert event_id is not None
        assert len(privacy_logger._log_entries) == 1
        
        entry = privacy_logger._log_entries[0]
        assert entry.event_id == event_id
        assert entry.tenant_id == "tenant-1"
        assert entry.event_type == EventType.AUDIT_TRAIL
        assert entry.log_level == LogLevel.AUDIT
        assert entry.user_id == "user-123"
        assert entry.session_id == "session-456"
        assert entry.request_id == "req-789"
        assert entry.metadata['action'] == "create"
        assert entry.metadata['resource'] == "mapping_record"
    
    @pytest.mark.asyncio
    async def test_get_audit_trail(self, privacy_logger):
        """Test retrieving audit trail entries."""
        # Create some test entries
        await privacy_logger.log_mapping_success(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="llama-3-8b-v1.0"
        )
        
        await privacy_logger.log_mapping_failure(
            tenant_id="tenant-1",
            detector_type="openai-moderation",
            error_code="TIMEOUT",
            error_category="network",
            model_version="llama-3-8b-v1.0"
        )
        
        await privacy_logger.log_mapping_success(
            tenant_id="tenant-2",  # Different tenant
            detector_type="llama-guard",
            taxonomy_hit="PII.Contact.Email",
            confidence_score=0.88,
            model_version="llama-3-8b-v1.0"
        )
        
        # Retrieve audit trail for tenant-1
        entries = await privacy_logger.get_audit_trail(tenant_id="tenant-1")
        
        assert len(entries) == 2  # Only tenant-1 entries
        assert all(entry.tenant_id == "tenant-1" for entry in entries)
        
        # Test filtering by event type
        success_entries = await privacy_logger.get_audit_trail(
            tenant_id="tenant-1",
            event_types=[EventType.MAPPING_SUCCESS]
        )
        
        assert len(success_entries) == 1
        assert success_entries[0].event_type == EventType.MAPPING_SUCCESS
    
    @pytest.mark.asyncio
    async def test_get_compliance_report(self, privacy_logger):
        """Test generating compliance report."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        # Create some test entries
        await privacy_logger.log_mapping_success(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="llama-3-8b-v1.0"
        )
        
        await privacy_logger.log_mapping_success(
            tenant_id="tenant-1",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Obscenity",
            confidence_score=0.87,
            model_version="llama-3-8b-v1.0"
        )
        
        await privacy_logger.log_mapping_failure(
            tenant_id="tenant-1",
            detector_type="openai-moderation",
            error_code="TIMEOUT",
            error_category="network",
            model_version="llama-3-8b-v1.0"
        )
        
        # Generate compliance report
        report = await privacy_logger.get_compliance_report(
            tenant_id="tenant-1",
            start_time=start_time,
            end_time=end_time
        )
        
        assert report['tenant_id'] == "tenant-1"
        assert report['summary']['total_events'] == 3
        assert report['summary']['event_types']['mapping_success'] == 2
        assert report['summary']['event_types']['mapping_failure'] == 1
        assert report['summary']['detector_usage']['deberta-toxicity'] == 2
        assert report['summary']['detector_usage']['openai-moderation'] == 1
        assert report['summary']['taxonomy_distribution']['HARM.SPEECH.Toxicity'] == 1
        assert report['summary']['taxonomy_distribution']['HARM.SPEECH.Obscenity'] == 1
        assert report['summary']['error_distribution']['network'] == 1
        
        # Check that audit entries are included
        assert len(report['audit_entries']) == 3
    
    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self, privacy_logger):
        """Test cleaning up old log entries."""
        # Create entries with different timestamps
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        recent_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Mock timestamps for testing
        old_entry = PrivacyLogEntry(
            event_id="old-event",
            timestamp=old_time,
            tenant_id="tenant-1",
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO
        )
        
        recent_entry = PrivacyLogEntry(
            event_id="recent-event",
            timestamp=recent_time,
            tenant_id="tenant-1",
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO
        )
        
        privacy_logger._log_entries = [old_entry, recent_entry]
        
        # Clean up entries older than 90 days
        cleaned_count = await privacy_logger.cleanup_old_entries(retention_days=90)
        
        assert cleaned_count == 1
        assert len(privacy_logger._log_entries) == 1
        assert privacy_logger._log_entries[0].event_id == "recent-event"
    
    def test_generate_event_id(self, privacy_logger):
        """Test event ID generation."""
        event_id1 = privacy_logger._generate_event_id()
        event_id2 = privacy_logger._generate_event_id()
        
        assert event_id1 != event_id2
        assert len(event_id1) > 0
        assert len(event_id2) > 0


if __name__ == "__main__":
    pytest.main([__file__])