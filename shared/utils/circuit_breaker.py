"""Circuit breaker pattern implementation."""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional, Type, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Tuple[Type[Exception], ...] = (Exception,),
        name: Optional[str] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function through the circuit breaker."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit breaker %s transitioning to HALF_OPEN", self.name
                    )
                else:
                    logger.warning(
                        "Circuit breaker %s is OPEN, failing fast", self.name
                    )
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except self.expected_exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(
                    "Circuit breaker %s transitioning to CLOSED after successful call",
                    self.name,
                )
            self._failure_count = 0

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker %s transitioning to OPEN after %d failures",
                    self.name,
                    self._failure_count,
                )

    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker %s manually reset", self.name)


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Tuple[Type[Exception], ...] = (Exception,),
    name: Optional[str] = None,
):
    """Decorator for applying circuit breaker pattern."""

    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name,
        )

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                return await breaker.call(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                # For sync functions, we need to run the circuit breaker logic
                # in a way that doesn't require async
                if breaker.state == CircuitState.OPEN:
                    if breaker._should_attempt_reset():
                        breaker._state = CircuitState.HALF_OPEN
                        logger.info(
                            "Circuit breaker %s transitioning to HALF_OPEN",
                            breaker.name,
                        )
                    else:
                        logger.warning(
                            "Circuit breaker %s is OPEN, failing fast", breaker.name
                        )
                        raise CircuitBreakerError(
                            f"Circuit breaker {breaker.name} is OPEN"
                        )

                try:
                    result = func(*args, **kwargs)
                    # Success
                    if breaker.state == CircuitState.HALF_OPEN:
                        breaker._state = CircuitState.CLOSED
                        logger.info(
                            "Circuit breaker %s transitioning to CLOSED after successful call",
                            breaker.name,
                        )
                    breaker._failure_count = 0
                    return result

                except expected_exception as e:
                    # Failure
                    breaker._failure_count += 1
                    breaker._last_failure_time = time.time()

                    if breaker._failure_count >= failure_threshold:
                        breaker._state = CircuitState.OPEN
                        logger.warning(
                            "Circuit breaker %s transitioning to OPEN after %d failures",
                            breaker.name,
                            breaker._failure_count,
                        )
                    raise

            return sync_wrapper

    return decorator
