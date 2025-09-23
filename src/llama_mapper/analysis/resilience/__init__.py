"""
Resilience package.

This package provides comprehensive resilience patterns including
retry logic, circuit breakers, and failure handling for distributed systems.
"""

from .interfaces import (
    CircuitState, RetryStrategy,
    ICircuitBreaker, IRetryManager, IResilienceManager,
    IResilienceMetricsCollector
)
from .circuit_breaker.implementation import CircuitBreaker, CircuitBreakerOpenException
from .retry.implementation import RetryManager
from .config.retry_config import RetryConfig
from .config.circuit_breaker_config import CircuitBreakerConfig
from .factory import ResilienceFactory, ResilienceManager

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
