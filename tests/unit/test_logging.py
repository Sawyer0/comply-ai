"""Tests for privacy-first logging implementation."""

import logging
import tempfile
from pathlib import Path


from llama_mapper.logging import (
    MetadataOnlyProcessor,
    PrivacyFilter,
    get_logger,
    log_metadata_only,
    setup_logging,
)


class TestPrivacyFilter:
    """Test cases for PrivacyFilter."""

    def test_blocks_sensitive_keys(self):
        """Test that filter blocks records with sensitive keys."""
        privacy_filter = PrivacyFilter()

        # Create log record with sensitive key
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.raw_input = "sensitive data"

        # Should be blocked
        assert privacy_filter.filter(record) is False

    def test_allows_safe_records(self):
        """Test that filter allows records without sensitive data."""
        privacy_filter = PrivacyFilter()

        # Create safe log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Safe message",
            args=(),
            exc_info=None,
        )
        record.tenant_id = "tenant_123"
        record.detector_type = "deberta-toxicity"

        # Should be allowed
        assert privacy_filter.filter(record) is True

    def test_blocks_sensitive_message_content(self):
        """Test that filter blocks messages with sensitive content."""
        privacy_filter = PrivacyFilter()

        # Create record with sensitive message
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing raw_input from user",
            args=(),
            exc_info=None,
        )

        # Should be blocked
        assert privacy_filter.filter(record) is False


class TestMetadataOnlyProcessor:
    """Test cases for MetadataOnlyProcessor."""

    def test_filters_allowed_fields(self):
        """Test that processor only keeps allowed fields."""
        processor = MetadataOnlyProcessor()

        event_dict = {
            "tenant_id": "tenant_123",
            "detector_type": "deberta-toxicity",
            "taxonomy_label": "HARM.SPEECH.Toxicity",
            "confidence": 0.85,
            "raw_input": "sensitive data",  # Should be filtered out
            "user_content": "more sensitive data",  # Should be filtered out
            "event": "detector_mapping_completed",
        }

        filtered = processor(None, "info", event_dict)

        # Verify allowed fields are kept
        assert "tenant_id" in filtered
        assert "detector_type" in filtered
        assert "taxonomy_label" in filtered
        assert "confidence" in filtered
        assert "event" in filtered

        # Verify sensitive fields are removed
        assert "raw_input" not in filtered
        assert "user_content" not in filtered

    def test_sanitizes_event_messages(self):
        """Test that event messages are sanitized."""
        processor = MetadataOnlyProcessor()

        event_dict = {
            "event": "Processing email address from user input",
            "tenant_id": "tenant_123",
        }

        filtered = processor(None, "info", event_dict)

        # Event should be sanitized
        assert "EMAIL_REDACTED" in filtered["event"]


class TestLoggingSetup:
    """Test cases for logging setup."""

    def test_setup_logging_with_privacy_filter(self):
        """Test that logging setup enables privacy filtering."""
        # Clear any existing handlers first
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(log_level="INFO", log_format="json", enable_privacy_filter=True)

        logger = get_logger(__name__)

        # Verify logger is configured
        assert logger is not None

        # Test that privacy filter is working by checking root logger handlers
        handlers = root_logger.handlers

        # At least one handler should have PrivacyFilter
        has_privacy_filter = False
        for handler in handlers:
            for filter_obj in handler.filters:
                if isinstance(filter_obj, PrivacyFilter):
                    has_privacy_filter = True
                    break

        assert has_privacy_filter

    def test_setup_logging_with_file_output(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            setup_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
                enable_privacy_filter=True,
            )

            logger = get_logger(__name__)
            logger.info("Test message")

            # Verify log file was created
            assert log_file.exists()

    def test_log_metadata_only_function(self):
        """Test the log_metadata_only convenience function."""
        setup_logging(enable_privacy_filter=True)
        logger = get_logger(__name__)

        # This should work without errors
        log_metadata_only(
            logger,
            "detector_mapping_completed",
            tenant_id="tenant_123",
            detector_type="deberta-toxicity",
            taxonomy_label="HARM.SPEECH.Toxicity",
            confidence=0.85,
            fallback_used=False,
            schema_valid=True,
            latency_ms=120.5,
        )

        # Test with None values (should be filtered out)
        log_metadata_only(
            logger,
            "test_event",
            tenant_id=None,
            detector_type="test-detector",
            confidence=None,
        )
