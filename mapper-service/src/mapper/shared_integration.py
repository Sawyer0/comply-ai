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
from shared.utils.middleware import CorrelationMiddleware
from shared.database.connection_manager import get_service_db
# Service registry import 
try:
    from shared.database.service_registry import ServiceRegistry
except ImportError:
    # Fallback if service registry not available
    ServiceRegistry = None
from shared.clients.client_factory import ClientFactory
from shared.interfaces.tenant_isolation import (
    ITenantIsolationManager, 
    TenantContext,
    TenantAccessLevel,
    TenantConfig,
    TenantIsolationError,
)
from shared.interfaces.cost_monitoring import CostEvent, CostCategory
from shared.validation.auth import validate_api_key, get_tenant_from_api_key, check_api_key_permissions
from shared.validation.common_validators import validate_non_empty_string, validate_confidence_score
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
    """Initialize shared components for the mapper service.
    
    This function sets up all shared components including logging,
    metrics collection, database connections, and other cross-cutting
    concerns that are used throughout the mapper service.
    
    Returns:
        Dict[str, Any]: Dictionary containing initialized shared components
            - metrics_collector: Shared metrics collector instance
            - logger: Configured logger instance  
            - db_pool: Database connection pool (if configured)
            
    Raises:
        Exception: If any shared component fails to initialize
    """
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


def get_shared_database():
    """Get the database connection pool."""
    return get_service_db(SERVICE_NAME)


def get_shared_resilience_manager():
    """Get the resilience manager instance."""
    from .resilience.manager import ComprehensiveResilienceManager
    return ComprehensiveResilienceManager(get_shared_metrics())


def get_shared_service_clients():
    """Get the service client factory."""
    return ClientFactory()


def get_shared_tenant_manager():
    """Get the tenant isolation manager."""
    from .tenancy.shared_tenant_manager import get_shared_tenant_manager as get_mapper_tenant_manager
    return get_mapper_tenant_manager()


def get_shared_cost_monitor():
    """Get the cost monitoring instance."""
    from .cost.shared_cost_monitor import get_shared_cost_monitor as get_mapper_cost_monitor
    return get_mapper_cost_monitor()


# Export commonly used shared components
__all__ = [
    "initialize_shared_components",
    "get_shared_logger",
    "get_shared_metrics",
    "get_shared_database",
    "get_shared_resilience_manager",
    "get_shared_service_clients",
    "get_shared_tenant_manager",
    "get_shared_cost_monitor",
    "get_correlation_id",
    "set_correlation_id",
    "track_request_metrics",
    "CorrelationMiddleware",
    "ClientFactory",
    "TenantContext",
    "CostEvent",
    "CostCategory",
    "validate_api_key",
    "get_tenant_from_api_key",
    "check_api_key_permissions",
    "validate_non_empty_string",
    "validate_confidence_score",
    "CircuitBreaker",
    "BaseServiceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",
    "TimeoutError",
]
