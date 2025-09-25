"""
Interfaces for mapper service resilience components.

This module defines the core interfaces and abstract base classes
for retry logic, circuit breakers, and resilience patterns specific
to the mapper service domain.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class RetryStrategy(Enum):
    """Retry strategies."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


class ICircuitBreaker(ABC):
    """Interface for circuit breaker functionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Circuit breaker name."""
        ...

    @property
    @abstractmethod
    def state(self) -> CircuitState:
        """Current circuit state."""
        ...

    @abstractmethod
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        ...

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        ...

    @abstractmethod
    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        ...


class IRetryManager(ABC):
    """Interface for retry management."""

    @abstractmethod
    async def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        ...

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        ...


class IResilienceMetricsCollector(ABC):
    """Interface for resilience metrics collection."""

    @abstractmethod
    def record_circuit_breaker_state(self, state: CircuitState, service: str) -> None:
        """Record circuit breaker state changes."""
        ...

    @abstractmethod
    def record_retry_attempt(self, service: str, attempt: int, success: bool) -> None:
        """Record retry attempt metrics."""
        ...


class IResilienceManager(ABC):
    """Interface for resilience management."""

    @abstractmethod
    def create_circuit_breaker(
        self, name: str, config: Optional[Any] = None
    ) -> ICircuitBreaker:
        """Create a new circuit breaker."""
        ...

    @abstractmethod
    def get_circuit_breaker(self, name: str) -> Optional[ICircuitBreaker]:
        """Get circuit breaker by name."""
        ...

    @abstractmethod
    def create_retry_manager(
        self,
        name: str,
        config: Optional[Any] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
    ) -> IRetryManager:
        """Create a new retry manager."""
        ...

    @abstractmethod
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        ...

    @abstractmethod
    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        ...
