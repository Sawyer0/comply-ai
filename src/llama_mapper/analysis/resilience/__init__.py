"""
Resilience package.

This package provides comprehensive resilience patterns including
retry logic, circuit breakers, and failure handling for distributed systems.
"""

from .circuit_breaker.implementation import CircuitBreaker, CircuitBreakerOpenException
from .config.circuit_breaker_config import CircuitBreakerConfig
from .config.retry_config import RetryConfig
from .factory import ResilienceFactory, ResilienceManager
from .interfaces import (
    CircuitState,
    ICircuitBreaker,
    IResilienceManager,
    IResilienceMetricsCollector,
    IRetryManager,
    RetryStrategy,
)
from .retry.implementation import RetryManager

__all__ = [
    # Interfaces
    "CircuitState",
    "RetryStrategy",
    "ICircuitBreaker",
    "IRetryManager",
    "IResilienceManager",
    "IResilienceMetricsCollector",
    # Implementations
    "CircuitBreaker",
    "CircuitBreakerOpenException",
    "RetryManager",
    # Configuration
    "RetryConfig",
    "CircuitBreakerConfig",
    # Factory
    "ResilienceFactory",
    "ResilienceManager",
]
