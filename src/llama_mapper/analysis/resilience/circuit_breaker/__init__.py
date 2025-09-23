"""Circuit breaker implementation package."""

from .implementation import CircuitBreaker, CircuitBreakerOpenException

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenException",
]
