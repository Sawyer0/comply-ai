"""Privacy-first logging configuration with metadata-only approach."""

import logging
import sys
import json
import re
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import structlog
from structlog.stdlib import LoggerFactory


class PrivacyFilter(logging.Filter):
    """
    Filter to ensure privacy-first logging by removing sensitive data.
    
    This filter implements the metadata-only approach required by requirement 8.2:
    - Never logs raw detector inputs
    - Redacts potential PII patterns
    - Limits message length to prevent accidental data leakage
    """
    
    # Patterns that might contain sensitive data
    SENSITIVE_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card pattern
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address pattern
    ]
    
    def __init__(self, max_message_length: int = 500):
        """
        Initialize privacy filter.
        
        Args:
            max_message_length: Maximum allowed message length
        """
        super().__init__()
        self.max_message_length = max_message_length
        self.sensitive_regex = re.compile('|'.join(self.SENSITIVE_PATTERNS))
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record to ensure privacy compliance.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        # Redact sensitive patterns in message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.sensitive_regex.sub('[REDACTED]', record.msg)
            
            # Truncate long messages
            if len(record.msg) > self.max_message_length:
                record.msg = record.msg[:self.max_message_length] + '...[TRUNCATED]'
        
        # Redact sensitive data in extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if isinstance(value, str) and self.sensitive_regex.search(value):
                    setattr(record, key, '[REDACTED]')
        
        return True


class MetadataProcessor:
    """Structlog processor for metadata-only logging."""
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process log event to ensure only metadata is logged.
        
        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Event dictionary
            
        Returns:
            Processed event dictionary with only metadata
        """
        # Add standard metadata
        event_dict['timestamp'] = datetime.now(timezone.utc).isoformat()
        event_dict['level'] = method_name.upper()
        
        # Remove any fields that might contain raw data
        sensitive_keys = ['raw_input', 'detector_input', 'user_content', 'prompt', 'response_text']
        for key in sensitive_keys:
            if key in event_dict:
                event_dict[key] = '[REDACTED]'
        
        # Ensure tenant_id is present for audit trails
        if 'tenant_id' not in event_dict:
            event_dict['tenant_id'] = 'unknown'
        
        return event_dict


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    privacy_mode: bool = True,
    max_message_length: int = 500
) -> None:
    """
    Set up privacy-first logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format (json, text)
        privacy_mode: Enable privacy-first logging
        max_message_length: Maximum log message length
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if privacy_mode:
        processors.append(MetadataProcessor())
    
    if format_type.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Add privacy filter if enabled
    if privacy_mode:
        handler.addFilter(PrivacyFilter(max_message_length))
    
    # Set formatter
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Audit logging helpers
def log_mapping_event(
    logger: structlog.stdlib.BoundLogger,
    tenant_id: str,
    detector_type: str,
    canonical_label: str,
    confidence: float,
    fallback_used: bool = False
) -> None:
    """
    Log a mapping event with metadata only.
    
    Args:
        logger: Logger instance
        tenant_id: Tenant identifier
        detector_type: Type of detector used
        canonical_label: Resulting canonical label
        confidence: Mapping confidence score
        fallback_used: Whether fallback mapping was used
    """
    logger.info(
        "mapping_event",
        tenant_id=tenant_id,
        detector_type=detector_type,
        canonical_label=canonical_label,
        confidence=confidence,
        fallback_used=fallback_used,
        event_type="mapping"
    )


def log_validation_error(
    logger: structlog.stdlib.BoundLogger,
    tenant_id: str,
    detector_type: str,
    error_type: str,
    error_message: str
) -> None:
    """
    Log a validation error with metadata only.
    
    Args:
        logger: Logger instance
        tenant_id: Tenant identifier
        detector_type: Type of detector
        error_type: Type of validation error
        error_message: Error message (will be filtered for privacy)
    """
    logger.error(
        "validation_error",
        tenant_id=tenant_id,
        detector_type=detector_type,
        error_type=error_type,
        error_message=error_message,
        event_type="validation_error"
    )


def log_model_performance(
    logger: structlog.stdlib.BoundLogger,
    tenant_id: str,
    latency_ms: float,
    tokens_generated: int,
    memory_usage_mb: Optional[float] = None
) -> None:
    """
    Log model performance metrics.
    
    Args:
        logger: Logger instance
        tenant_id: Tenant identifier
        latency_ms: Request latency in milliseconds
        tokens_generated: Number of tokens generated
        memory_usage_mb: Memory usage in MB (optional)
    """
    logger.info(
        "model_performance",
        tenant_id=tenant_id,
        latency_ms=latency_ms,
        tokens_generated=tokens_generated,
        memory_usage_mb=memory_usage_mb,
        event_type="performance"
    )