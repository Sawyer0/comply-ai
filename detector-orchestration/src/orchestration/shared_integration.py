"""Integration helpers that expose shared infrastructure to the orchestration service."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


def _ensure_shared_on_path() -> None:
    """Add the repository root to ``sys.path`` so ``shared`` imports succeed."""

    root = Path(__file__).resolve().parents[3]
    shared_dir = root / "shared"
    if shared_dir.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


try:  # pragma: no cover - import resolution depends on environment
    from shared.clients.client_factory import ClientFactory
    from shared.database.connection_manager import get_service_db
    from shared.exceptions.base import (
        AuthenticationError,
        AuthorizationError,
        BaseServiceException,
        ServiceUnavailableError,
        TimeoutError as SharedTimeoutError,
        ValidationError,
    )
    from shared.interfaces.cost_monitoring import CostCategory, CostEvent
    from shared.interfaces.tenant_isolation import (
        ITenantIsolationManager,
        TenantConfig,
        TenantContext,
        TenantIsolationError,
    )
    from shared.utils.cache import (
        IdempotencyCache,
        ResponseCache,
        create_idempotency_cache,
        create_response_cache,
    )
    from shared.utils.circuit_breaker import CircuitBreaker
    from shared.utils.correlation import get_correlation_id, set_correlation_id
    from shared.utils.logging import configure_logging, get_logger
    from shared.utils.middleware import CorrelationMiddleware
    from shared.utils.metrics import MetricsCollector, track_request_metrics
    from shared.validation.auth import (
        check_api_key_permissions,
        get_tenant_from_api_key,
        validate_api_key,
    )
    from shared.validation.common_validators import (
        validate_confidence_score,
        validate_non_empty_string,
    )
except ImportError:  # pragma: no cover - executed in tooling environments
    _ensure_shared_on_path()
    from shared.clients.client_factory import ClientFactory
    from shared.database.connection_manager import get_service_db
    from shared.exceptions.base import (
        AuthenticationError,
        AuthorizationError,
        BaseServiceException,
        ServiceUnavailableError,
        TimeoutError as SharedTimeoutError,
        ValidationError,
    )
    from shared.interfaces.cost_monitoring import CostCategory, CostEvent
    from shared.interfaces.tenant_isolation import (
        ITenantIsolationManager,
        TenantConfig,
        TenantContext,
        TenantIsolationError,
    )
    from shared.utils.cache import (
        IdempotencyCache,
        ResponseCache,
        create_idempotency_cache,
        create_response_cache,
    )
    from shared.utils.circuit_breaker import CircuitBreaker
    from shared.utils.correlation import get_correlation_id, set_correlation_id
    from shared.utils.logging import configure_logging, get_logger
    from shared.utils.middleware import CorrelationMiddleware
    from shared.utils.metrics import MetricsCollector, track_request_metrics
    from shared.validation.auth import (
        check_api_key_permissions,
        get_tenant_from_api_key,
        validate_api_key,
    )
    from shared.validation.common_validators import (
        validate_confidence_score,
        validate_non_empty_string,
    )


def initialize_shared_components(service_name: str = "orchestration") -> Dict[str, Any]:
    """Initialise logging, metrics, caches, and service clients for the service."""

    configure_logging(service_name)
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


def get_shared_logger() -> Any:
    """Return the configured shared logger."""

    return get_logger(__name__)


def get_shared_metrics() -> MetricsCollector:
    """Return a metrics collector namespaced for orchestration."""

    return MetricsCollector("orchestration")


def get_shared_database():
    """Return the orchestration database connection manager."""

    return get_service_db("orchestration")


def get_shared_resilience_manager() -> Dict[str, CircuitBreaker]:
    """Return the shared resilience primitives (currently a circuit breaker)."""

    return {
        "circuit_breaker": CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="orchestration_circuit_breaker",
        )
    }


def get_shared_service_clients() -> ClientFactory:
    """Return the shared client factory used to build downstream service clients."""

    return ClientFactory()


def create_opa_client_with_config(**kwargs):
    """Create an OPA client using the shared OPA client helper."""

    try:
        module = importlib.import_module("shared.clients.opa_client")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OPA client dependency is unavailable") from exc
    return module.create_opa_client_with_config(**kwargs)


def get_shared_tenant_manager():
    """Return the shared tenant isolation manager implementation."""

    try:
        module = importlib.import_module("mapper.tenancy.shared_tenant_manager")
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("Mapper tenant manager module is unavailable: %s", exc)
        raise RuntimeError("Mapper tenant manager module is unavailable") from exc
    return module.SharedTenantManager()


def get_shared_cost_monitor():
    """Return the shared cost monitoring helper."""

    try:
        module = importlib.import_module("mapper.tenancy.cost_tracker")
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("Mapper cost tracker module is unavailable: %s", exc)
        raise RuntimeError("Mapper cost tracker module is unavailable") from exc
    return module.CostTracker()


def get_shared_idempotency_cache() -> IdempotencyCache:
    """Return the idempotency cache (Redis in production, memory in development)."""

    cache_backend = "redis"
    redis_url = os.getenv("ORCHESTRATION_REDIS_URL", "redis://localhost:6379")
    return create_idempotency_cache(
        backend_type=cache_backend,
        ttl_seconds=3600,
        redis_url=redis_url,
    )


def get_shared_response_cache() -> ResponseCache:
    """Return the response cache (Redis in production, memory in development)."""

    cache_backend = "redis"
    redis_url = os.getenv("ORCHESTRATION_REDIS_URL", "redis://localhost:6379")
    return create_response_cache(
        backend_type=cache_backend,
        ttl_seconds=1800,
        redis_url=redis_url,
    )


class OPAError(BaseServiceException):
    """OPA error exception base class."""


class OPAPolicyError(OPAError):
    """OPA policy error exception."""


# re-export frequently used shared utilities for convenience
TenantIsolationManagerProtocol = ITenantIsolationManager
TenantIsolationContext = TenantContext
TenantIsolationConfig = TenantConfig
TenantIsolationFailure = TenantIsolationError
TimeoutError = SharedTimeoutError  # pylint: disable=redefined-builtin


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
    "TenantIsolationManagerProtocol",
    "TenantIsolationContext",
    "TenantIsolationConfig",
    "TenantIsolationFailure",
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
    "create_idempotency_cache",
    "create_response_cache",
    "IdempotencyCache",
    "ResponseCache",
]
