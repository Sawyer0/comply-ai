"""
Integration module for shared components.

This module configures the detector orchestration service to use shared components
from the comply-ai shared library.
"""

import sys
from pathlib import Path
from typing import Optional

# Add root directory to Python path so we can import 'shared' module
# Current file: detector-orchestration/src/orchestration/shared_integration.py
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
from shared.utils.cache import (
    create_idempotency_cache,
    create_response_cache,
    IdempotencyCache,
    ResponseCache,
)
from shared.database.connection_manager import get_service_db
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

# Service-specific shared components initialization
def initialize_shared_components(service_name: str = "orchestration"):
    """Initialize all shared components for the orchestration service."""
    
    # Configure structured logging
    configure_logging(service_name)
    
    # Return initialized components
    return {
        "logger": get_shared_logger(),
        "metrics": get_shared_metrics(),
        "database": get_shared_database(),
        "resilience_manager": get_shared_resilience_manager(),
        "service_clients": get_shared_service_clients(),
        "tenant_manager": get_shared_tenant_manager(),
        "cost_monitor": get_shared_cost_monitor(),
        "idempotency_cache": get_shared_idempotency_cache(),
        "response_cache": get_shared_response_cache(),
    }


def get_shared_logger():
    """Get configured shared logger."""
    return get_logger(__name__)


def get_shared_metrics():
    """Get shared metrics collector."""
    return MetricsCollector("orchestration")


def get_shared_database():
    """Get shared database manager."""
    return get_service_db("orchestration")


def get_shared_resilience_manager():
    """Get shared resilience components."""
    return {
        "circuit_breaker": CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="orchestration_circuit_breaker"
        )
    }


def get_shared_service_clients():
    """Get shared service clients factory."""
    return ClientFactory()


def create_opa_client_with_config(**kwargs):
    """Create OPA client using global shared pattern."""
    from shared.clients.opa_client import create_opa_client_with_config as _create_opa_client
    return _create_opa_client(**kwargs)


# OPA Error classes
class OPAError(BaseServiceException):
    """OPA error exception."""
    pass


class OPAPolicyError(OPAError):
    """OPA policy error exception.""" 
    pass


def get_shared_tenant_manager():
    """Get shared tenant isolation manager."""
    # Import the actual implementation from mapper service
    from mapper.tenancy.shared_tenant_manager import SharedTenantManager
    return SharedTenantManager()


def get_shared_cost_monitor():
    """Get shared cost monitoring."""
    # Import the actual implementation from mapper service
    from mapper.tenancy.cost_tracker import CostTracker
    return CostTracker()


def get_shared_idempotency_cache():
    """Get shared idempotency cache."""
    # Use Redis in production, memory in development
    cache_backend = "redis"  # Change to "memory" for development
    return create_idempotency_cache(backend_type=cache_backend, ttl_seconds=3600)


def get_shared_response_cache():
    """Get shared response cache."""
    # Use Redis in production, memory in development
    cache_backend = "redis"  # Change to "memory" for development
    return create_response_cache(backend_type=cache_backend, ttl_seconds=1800)


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
    "get_shared_idempotency_cache",
    "get_shared_response_cache",
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
    "create_opa_client_with_config",
    "OPAError",
    "OPAPolicyError",
    # Cache utilities
    "create_idempotency_cache",
    "create_response_cache",
    "IdempotencyCache",
    "ResponseCache",
]
