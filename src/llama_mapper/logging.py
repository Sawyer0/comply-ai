"""
Privacy-first logging configuration for Llama Mapper.

Implements metadata-only logging that never persists raw detector inputs
to ensure compliance with data protection regulations.
"""

import logging
import logging.config
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import structlog
from structlog.types import FilteringBoundLogger


@dataclass
class MetadataEvent:
    """Data class for metadata logging events.

    Note: This dataclass has many fields (8/7) as it represents a comprehensive
    metadata event structure for compliance and audit logging purposes.
    """

    event: str
    tenant_id: Optional[str] = None
    detector_type: Optional[str] = None
    taxonomy_label: Optional[str] = None
    confidence: Optional[float] = None
    fallback_used: bool = False
    schema_valid: bool = True
    latency_ms: Optional[float] = None


class PrivacyFilter(logging.Filter):
    """
    Filter that prevents logging of sensitive data.

    Blocks any log records that might contain raw detector inputs
    or other sensitive information.

    Note: This class has only one public method (filter) as it's designed
    to be used as a single-purpose logging filter following the Python
    logging framework patterns.
    """

    SENSITIVE_KEYS = {
        "raw_input",
        "detector_input",
        "content",
        "text",
        "message_content",
        "user_input",
        "prompt",
        "response_text",
        "raw_output",
        "original_text",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to prevent sensitive data logging.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        # Check message for sensitive content
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg_lower = record.msg.lower()
            if any(key in msg_lower for key in self.SENSITIVE_KEYS):
                return False

        # Check extra fields for sensitive keys
        if hasattr(record, "__dict__"):
            for key in record.__dict__.keys():
                if key.lower() in self.SENSITIVE_KEYS:
                    return False

        return True


class MetadataOnlyProcessor:
    """
    Structlog processor that ensures only metadata is logged.

    Removes any fields that might contain sensitive data and
    ensures compliance with privacy-first logging requirements.

    Note: This class has only one public method (__call__) as it's designed
    to be used as a single-purpose structlog processor following the
    structlog framework patterns.
    """

    ALLOWED_FIELDS = {
        "timestamp",
        "level",
        "logger",
        "event",
        "tenant_id",
        "detector_type",
        "taxonomy_label",
        "confidence",
        "fallback_used",
        "schema_valid",
        "latency_ms",
        "request_id",
        "session_id",
        "user_id",
        "trace_id",
        "span_id",
        "version",
        "component",
        "operation",
        "status_code",
        "error_type",
        "error_code",
        "retry_count",
        "batch_size",
    }

    def __call__(
        self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process log event to ensure only metadata is included.

        Args:
            logger: Structlog logger instance
            method_name: Log method name (info, error, etc.)
            event_dict: Event dictionary to process

        Returns:
            Filtered event dictionary with only allowed fields
        """
        # Filter out non-allowed fields
        filtered_dict = {}
        for key, value in event_dict.items():
            if key in self.ALLOWED_FIELDS:
                if key == "event" and isinstance(value, str):
                    # Allow event messages but sanitize them
                    filtered_dict[key] = self._sanitize_message(value)
                else:
                    filtered_dict[key] = value

        return filtered_dict

    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize log message to remove potential sensitive content.

        Args:
            message: Original log message

        Returns:
            Sanitized message
        """
        # Replace potential sensitive patterns with placeholders
        sensitive_patterns = [
            ("email address", "[EMAIL_REDACTED]"),
            ("email", "[EMAIL_REDACTED]"),
            ("ssn", "[SSN_REDACTED]"),
            ("phone", "[PHONE_REDACTED]"),
            ("credit card", "[CC_REDACTED]"),
            ("password", "[PASSWORD_REDACTED]"),
            ("token", "[TOKEN_REDACTED]"),
            ("key", "[KEY_REDACTED]"),
        ]

        sanitized = message.lower()
        for pattern, replacement in sensitive_patterns:
            if pattern in sanitized:
                return f"Sensitive content detected: {replacement}"

        return message


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_privacy_filter: bool = True,
) -> None:
    """
    Set up privacy-first logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, console)
        log_file: Optional log file path
        enable_privacy_filter: Whether to enable privacy filtering
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if enable_privacy_filter:
        processors.append(MetadataOnlyProcessor())

    if log_format == "json":
        processors.extend(
            [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        )
    else:
        processors.extend([structlog.dev.ConsoleRenderer(colors=True)])

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLogger().getEffectiveLevel()
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if enable_privacy_filter:
        console_handler.addFilter(PrivacyFilter())

    if log_format == "json":
        console_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        if enable_privacy_filter:
            file_handler.addFilter(PrivacyFilter())

        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(message)s",
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> FilteringBoundLogger:
    """
    Get a privacy-aware structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return cast(FilteringBoundLogger, structlog.get_logger(name))


class MetadataLogger:
    """Helper class for logging metadata-only events."""

    def __init__(self, logger: FilteringBoundLogger):
        """Initialize metadata logger.

        Args:
            logger: Structlog logger instance
        """
        self.logger = logger

    def log_event(self, metadata_event: MetadataEvent, **kwargs: Any) -> None:
        """Log metadata-only information for audit and monitoring.

        This method ensures that only metadata is logged, never raw content.

        Args:
            metadata_event: Structured metadata event
            **kwargs: Additional metadata fields (will be filtered)
        """
        metadata = {
            "event": metadata_event.event,
            "tenant_id": metadata_event.tenant_id,
            "detector_type": metadata_event.detector_type,
            "taxonomy_label": metadata_event.taxonomy_label,
            "confidence": metadata_event.confidence,
            "fallback_used": metadata_event.fallback_used,
            "schema_valid": metadata_event.schema_valid,
            "latency_ms": metadata_event.latency_ms,
        }

        # Add additional metadata (will be filtered by MetadataOnlyProcessor)
        metadata.update(kwargs)

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        event_msg = str(metadata.pop("event", "event"))
        self.logger.info(event_msg, **metadata)

    def log(
        self,
        event: str,
        tenant_id: Optional[str] = None,
        detector_type: Optional[str] = None,
        taxonomy_label: Optional[str] = None,
        confidence: Optional[float] = None,
        fallback_used: bool = False,
        schema_valid: bool = True,
        latency_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log metadata-only information for audit and monitoring.

        This method ensures that only metadata is logged, never raw content.

        Args:
            event: Event description
            tenant_id: Tenant identifier
            detector_type: Type of detector used
            taxonomy_label: Canonical taxonomy label assigned
            confidence: Model confidence score
            fallback_used: Whether rule-based fallback was used
            schema_valid: Whether output passed schema validation
            latency_ms: Request latency in milliseconds
            **kwargs: Additional metadata fields (will be filtered)
        """
        metadata_event = MetadataEvent(
            event=event,
            tenant_id=tenant_id,
            detector_type=detector_type,
            taxonomy_label=taxonomy_label,
            confidence=confidence,
            fallback_used=fallback_used,
            schema_valid=schema_valid,
            latency_ms=latency_ms,
        )
        self.log_event(metadata_event, **kwargs)


def log_metadata_only(
    logger: FilteringBoundLogger,
    event: str,
    tenant_id: Optional[str] = None,
    detector_type: Optional[str] = None,
    taxonomy_label: Optional[str] = None,
    confidence: Optional[float] = None,
    fallback_used: bool = False,
    schema_valid: bool = True,
    latency_ms: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Log metadata-only information for audit and monitoring.

    This function ensures that only metadata is logged, never raw content.

    Args:
        logger: Structlog logger instance
        event: Event description
        tenant_id: Tenant identifier
        detector_type: Type of detector used
        taxonomy_label: Canonical taxonomy label assigned
        confidence: Model confidence score
        fallback_used: Whether rule-based fallback was used
        schema_valid: Whether output passed schema validation
        latency_ms: Request latency in milliseconds
        **kwargs: Additional metadata fields (will be filtered)
    """
    metadata_event = MetadataEvent(
        event=event,
        tenant_id=tenant_id,
        detector_type=detector_type,
        taxonomy_label=taxonomy_label,
        confidence=confidence,
        fallback_used=fallback_used,
        schema_valid=schema_valid,
        latency_ms=latency_ms,
    )
    metadata_logger = MetadataLogger(logger)
    metadata_logger.log_event(metadata_event, **kwargs)


# Example usage and testing functions
def test_privacy_filter() -> None:
    """Test privacy filter functionality."""
    setup_logging(enable_privacy_filter=True)
    logger = get_logger(__name__)

    # These should be logged (metadata only)
    metadata_logger = MetadataLogger(logger)
    metadata_logger.log(
        event="detector_mapping_completed",
        tenant_id="tenant_123",
        detector_type="deberta-toxicity",
        taxonomy_label="HARM.SPEECH.Toxicity",
        confidence=0.85,
        fallback_used=False,
        schema_valid=True,
        latency_ms=120.5,
    )

    # This should be filtered out
    logger.info("Processing raw input", raw_input="This is sensitive content")

    # This should be sanitized
    logger.info("Detected email pattern in content")


if __name__ == "__main__":
    test_privacy_filter()
