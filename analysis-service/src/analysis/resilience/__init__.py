"""
Resilience patterns for analysis service.

This package provides comprehensive resilience patterns including
retry logic, circuit breakers, bulkhead isolation, and failure handling
specifically designed for analysis service operations.
"""

from .bulkhead import BulkheadConfig, BulkheadIsolation, BulkheadManager
from .circuit_breaker import AnalysisCircuitBreaker, CircuitBreakerOpenException
from .config import CircuitBreakerConfig, RetryConfig
from .factory import AnalysisResilienceFactory, AnalysisResilienceManager
from .fallback import (
    AnalysisFallbackManager,
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
from .retry_manager import AnalysisRetryManager

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
    "AnalysisCircuitBreaker",
    "CircuitBreakerOpenException",
    "AnalysisRetryManager",
    "BulkheadIsolation",
    "BulkheadManager",
    "AnalysisFallbackManager",
    "SimpleFallbackStrategy",
    "FunctionFallbackStrategy",
    # Configuration
    "RetryConfig",
    "CircuitBreakerConfig",
    "BulkheadConfig",
    "FallbackConfig",
    # Factory
    "AnalysisResilienceFactory",
    "AnalysisResilienceManager",
    "ComprehensiveResilienceManager",
    # Decorators
    "with_fallback",
]
