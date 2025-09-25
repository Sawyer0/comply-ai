"""
Integration module for shared components.

This module configures the mapper service to use shared components
from the comply-ai shared library.
"""

import sys
import os
from pathlib import Path

# Add root directory to Python path so we can import 'shared' module
# Current file: mapper-service/src/mapper/shared_integration.py
# Target: root directory containing shared/
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SHARED_DIR = ROOT_DIR / "shared"

if SHARED_DIR.exists() and str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
else:
    # Try alternative path resolution
    current_dir = Path(__file__).resolve().parent
    while current_dir.parent != current_dir:  # Stop at filesystem root
        shared_candidate = current_dir / "shared"
        if shared_candidate.exists():
            sys.path.insert(0, str(current_dir))
            break
        current_dir = current_dir.parent

# Import and configure shared components
from shared.utils.logging import configure_logging, get_logger
from shared.utils.correlation import set_correlation_id, get_correlation_id
from shared.utils.metrics import MetricsCollector, track_request_metrics
from shared.utils.circuit_breaker import CircuitBreaker
from shared.exceptions.base import (
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
    TimeoutError,
)

# Service configuration
SERVICE_NAME = "mapper-service"
logger = get_logger(__name__)


def initialize_shared_components():
    """Initialize shared components for the mapper service."""
    try:
        # Configure logging
        configure_logging(SERVICE_NAME)

        # Initialize metrics collector
        metrics_collector = MetricsCollector(SERVICE_NAME)

        logger.info("Shared components initialized successfully", service=SERVICE_NAME)

        return {"metrics_collector": metrics_collector, "logger": logger}

    except Exception as e:
        print(f"Failed to initialize shared components: {e}")
        raise


def get_shared_logger(name: str = None):
    """Get a configured logger instance."""
    return get_logger(name or SERVICE_NAME)


def get_shared_metrics():
    """Get the metrics collector instance."""
    return MetricsCollector(SERVICE_NAME)


# Export commonly used shared components
__all__ = [
    "initialize_shared_components",
    "get_shared_logger",
    "get_shared_metrics",
    "get_correlation_id",
    "set_correlation_id",
    "track_request_metrics",
    "CircuitBreaker",
    "BaseServiceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",
    "TimeoutError",
]
