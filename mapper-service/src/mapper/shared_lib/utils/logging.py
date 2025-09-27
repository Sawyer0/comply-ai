"""Shared logging utilities with privacy controls."""

import structlog
import logging.config
from typing import Any, Dict
from .correlation import get_correlation_id


def configure_logging(service_name: str, log_level: str = "INFO") -> None:
    """Configure structured logging for a service."""

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(colors=False),
                },
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": log_level,
                    "propagate": True,
                },
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_service_context,
            add_correlation_id,
            scrub_sensitive_data,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def add_service_context(logger, method_name, event_dict):
    """Add service context to log entries."""
    event_dict["service"] = logger.name
    return event_dict


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to all log entries."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def scrub_sensitive_data(logger, method_name, event_dict):
    """Scrub sensitive data from log entries."""
    sensitive_keys = {
        "password",
        "token",
        "api_key",
        "secret",
        "content",
        "raw_input",
        "raw_output",
        "pii",
        "personal_data",
    }

    def scrub_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        scrubbed = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys:
                scrubbed[key] = "[SCRUBBED]"
            elif isinstance(value, dict):
                scrubbed[key] = scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = [
                    scrub_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                scrubbed[key] = value
        return scrubbed

    return scrub_dict(event_dict)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)
