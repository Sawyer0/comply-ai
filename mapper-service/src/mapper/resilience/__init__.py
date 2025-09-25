"""
Resilience patterns for mapper service.

This package provides comprehensive resilience patterns including
retry logic, circuit breakers, bulkhead isolation, and failure handling
specifically designed for mapper service operations.
"""

from .bulkhead import BulkheadConfig, BulkheadIsolation, BulkheadManager
from .circuit_breaker import MapperCircuitBreaker, CircuitBreakerOpenException
from .config import CircuitBreakerConfig, RetryConfig
from .factory import MapperResilienceFactory, MapperResilienceManager
from .fallback import (
    MapperFallbackManager,
    FallbackConfig,
    FunctionFallbackStrategy,
    IFallbackStrategy,
    SimpleFallbackStrategy,
    with_fallback,
)
from .interfaces import (
    CircuitState,
    ICircuitBreaker,
    IResilienceManager,
    IResilienceMetricsCollector,
    IRetryManager,
    RetryStrategy,
)
from .manager import ComprehensiveResilienceManager
from .resilience_manager import ResilienceManager
from .retry_manager import MapperRetryManager

__all__ = [
    # Interfaces
    "CircuitState",
    "RetryStrategy",
    "ICircuitBreaker",
    "IRetryManager",
    "IResilienceManager",
    "IResilienceMetricsCollector",
    "IFallbackStrategy",
    # Implementations
    "MapperCircuitBreaker",
    "CircuitBreakerOpenException",
    "MapperRetryManager",
    "BulkheadIsolation",
    "BulkheadManager",
    "MapperFallbackManager",
    "SimpleFallbackStrategy",
    "FunctionFallbackStrategy",
    # Configuration
    "RetryConfig",
    "CircuitBreakerConfig",
    "BulkheadConfig",
    "FallbackConfig",
    # Factory
    "MapperResilienceFactory",
    "MapperResilienceManager",
    "ComprehensiveResilienceManager",
    "ResilienceManager",
    # Decorators
    "with_fallback",
]
