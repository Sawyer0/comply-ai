"""
Integration module for shared components.

This module configures the analysis service to use shared components
from the comply-ai shared library.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

# Add root directory to Python path so we can import 'shared' module
# Current file: analysis-service/src/analysis/shared_integration.py
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
from shared.utils.caching import CacheManager
from shared.database.connection_manager import get_service_db
from shared.database.service_registry import ServiceRegistry  # Now implemented
from shared.clients.client_factory import ClientFactory
from shared.clients.analysis_client import AnalysisClient
from shared.interfaces.tenant_isolation import (
    ITenantIsolationManager, 
    TenantContext,
    TenantAccessLevel,
    TenantConfig,
    TenantIsolationError,
)
from shared.interfaces.cost_monitoring import CostEvent, CostCategory
from shared.interfaces.analysis import (
    AnalysisRequest as SharedAnalysisRequest,
    AnalysisResponse,
    CanonicalTaxonomyResult,
    QualityMetrics,
    PatternAnalysisResult,
    RiskScoringResult,
    ComplianceMappingResult,
    RAGInsights,
)
from shared.validation.auth import validate_api_key, get_tenant_from_api_key, check_api_key_permissions
from shared.validation.common_validators import validate_non_empty_string, validate_confidence_score
from shared.validation.decorators import validate_input, validate_output
from shared.exceptions.base import (
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
)

# Import analysis service resilience components
from .resilience import ComprehensiveResilienceManager
from .resilience.config import CircuitBreakerConfig, RetryConfig

# Service configuration
SERVICE_NAME = "analysis-service"
logger = get_logger(__name__)


def initialize_shared_components():
    """Initialize shared components for the analysis service.
    
    This function sets up all shared components including logging,
    metrics collection, database connections, service registry,
    and other cross-cutting concerns that are used throughout the analysis service.
    
    Returns:
        Dict[str, Any]: Dictionary containing initialized shared components
            - metrics_collector: Shared metrics collector instance
            - logger: Configured logger instance  
            - db_pool: Database connection pool
            - resilience_manager: Comprehensive resilience manager
            - service_registry: Service registry for inter-service communication
            - cache_manager: Distributed cache manager
            - client_factory: HTTP client factory
            
    Raises:
        Exception: If any shared component fails to initialize
    """
    try:
        # Configure logging with correlation ID support
        configure_logging(SERVICE_NAME)

        # Initialize metrics collector
        metrics_collector = MetricsCollector(SERVICE_NAME)

        # Initialize database connection pool
        db_pool = get_service_db(SERVICE_NAME)

        # Initialize service registry for service discovery
        service_registry = ServiceRegistry()
        asyncio.create_task(service_registry.register_service(
            SERVICE_NAME, 
            "http://localhost:8083",  # Analysis service default port
            {"status": "healthy", "version": "1.0.0"}
        ))

        # Initialize distributed cache manager
        cache_manager = CacheManager(
            redis_url="redis://localhost:6379/1",  # Analysis service cache DB
            default_ttl=3600,  # 1 hour default TTL
            key_prefix=f"{SERVICE_NAME}:"
        )

        # Initialize HTTP client factory with service discovery
        client_factory = ClientFactory()
        client_factory.set_service_registry(service_registry)

        # Initialize resilience manager with shared metrics
        resilience_manager = ComprehensiveResilienceManager(metrics_collector)

        logger.info("Shared components initialized successfully", 
                   service=SERVICE_NAME,
                   components=["logging", "metrics", "database", "service_registry", 
                             "cache", "clients", "resilience"])

        return {
            "metrics_collector": metrics_collector,
            "logger": logger,
            "db_pool": db_pool,
            "resilience_manager": resilience_manager,
            "service_registry": service_registry,
            "cache_manager": cache_manager,
            "client_factory": client_factory,
        }

    except Exception as e:
        logger.error("Failed to initialize shared components", error=str(e), service=SERVICE_NAME)
        raise


def get_shared_logger(name: Optional[str] = None):
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
    return ComprehensiveResilienceManager(get_shared_metrics())


def get_shared_service_clients():
    """Get the service client factory."""
    return ClientFactory()


def get_shared_tenant_manager():
    """Get the tenant isolation manager."""
    from .tenancy.shared_tenant_manager import get_shared_tenant_manager as get_analysis_tenant_manager
    return get_analysis_tenant_manager()


def get_shared_cost_monitor():
    """Get the cost monitoring instance."""
    from .cost.shared_cost_monitor import get_shared_cost_monitor as get_analysis_cost_monitor
    return get_analysis_cost_monitor()


def get_shared_cache_manager():
    """Get the cache manager instance."""
    return CacheManager(
        redis_url="redis://localhost:6379/1",
        default_ttl=3600,
        key_prefix=f"{SERVICE_NAME}:"
    )


def get_shared_service_registry():
    """Get the service registry instance."""
    return ServiceRegistry()


def get_shared_analysis_client():
    """Get the analysis client for inter-service communication."""
    client_factory = get_shared_service_clients()
    return client_factory.get_client("analysis-client", AnalysisClient)


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
    "get_shared_cache_manager",
    "get_shared_service_registry",
    "get_shared_analysis_client",
    "get_correlation_id",
    "set_correlation_id",
    "track_request_metrics",
    "CorrelationMiddleware",
    "ClientFactory",
    "AnalysisClient",
    "TenantContext",
    "CostEvent",
    "CostCategory",
    "validate_api_key",
    "get_tenant_from_api_key",
    "check_api_key_permissions",
    "validate_non_empty_string",
    "validate_confidence_score",
    "validate_input",
    "validate_output",
    "CircuitBreaker",
    "CacheManager",
    "ServiceRegistry",
    "SharedAnalysisRequest",
    "AnalysisResponse",
    "CanonicalTaxonomyResult",
    "QualityMetrics",
    "PatternAnalysisResult",
    "RiskScoringResult",
    "ComplianceMappingResult",
    "RAGInsights",
    "BaseServiceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",
]
