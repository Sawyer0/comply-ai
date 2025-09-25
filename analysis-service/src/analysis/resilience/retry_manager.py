"""
Retry logic implementation for analysis service.

This module provides the core retry implementation with
exponential backoff, jitter, and circuit breaker integration
specifically for analysis operations.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from .circuit_breaker import CircuitBreakerOpenException
from .config import RetryConfig
from .interfaces import IRetryManager, RetryStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AnalysisRetryManager(IRetryManager):
    """
    Retry manager for analysis operations with exponential backoff.

    Provides sophisticated retry logic with configurable strategies,
    jitter, and integration with circuit breakers for resilience.
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
    ):
        """
        Initialize retry manager.

        Args:
            config: Retry configuration
            circuit_breaker: Circuit breaker instance
            metrics_collector: Metrics collector for monitoring
        """
        self._config = config or RetryConfig()
        self._circuit_breaker = circuit_breaker
        self._metrics_collector = metrics_collector

        # Statistics
        self._total_attempts = 0
        self._total_retries = 0
        self._total_successes = 0
        self._total_failures = 0

        logger.info("Analysis retry manager initialized with config: %s", self._config)

    @staticmethod
    def with_retry(**kwargs):
        """
        Create a retry decorator with the given configuration.

        Args:
            **kwargs: Retry configuration parameters

        Returns:
            Decorator function that applies retry logic
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **func_kwargs):
                config = RetryConfig(**kwargs)
                retry_manager = AnalysisRetryManager(config)
                return await retry_manager.execute_with_retry(
                    func, *args, **func_kwargs
                )

            return wrapper

        return decorator

    async def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, self._config.max_attempts + 1):
            self._total_attempts += 1

            try:
                # Execute function (with circuit breaker if available)
                if self._circuit_breaker:
                    result = await self._circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs)

                # Record success
                self._total_successes += 1
                if attempt > 1:
                    self._total_retries += 1

                # Record metrics
                if self._metrics_collector:
                    self._metrics_collector.record_retry_attempt(
                        service=func.__name__, attempt=attempt, success=True
                    )

                logger.debug(
                    "Analysis function %s succeeded on attempt %s",
                    func.__name__,
                    attempt,
                )
                return result

            except CircuitBreakerOpenException:
                # Circuit breaker is open, don't retry
                self._total_failures += 1
                if self._metrics_collector:
                    self._metrics_collector.record_retry_attempt(
                        service=func.__name__, attempt=attempt, success=False
                    )
                raise

            except Exception as e:
                last_exception = e
                self._total_failures += 1

                # Record metrics
                if self._metrics_collector:
                    self._metrics_collector.record_retry_attempt(
                        service=func.__name__, attempt=attempt, success=False
                    )

                # Check if exception should be retried
                if not self._should_retry(e, attempt):
                    logger.warning(
                        "Analysis function %s failed with non-retryable exception: %s",
                        func.__name__,
                        e,
                    )
                    raise

                # Check if we should retry
                if attempt >= self._config.max_attempts:
                    logger.error(
                        "Analysis function %s failed after %d attempts: %s",
                        func.__name__,
                        attempt,
                        e,
                    )
                    raise

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    "Analysis function %s failed on attempt %d: %s. Retrying in %.2fs...",
                    func.__name__,
                    attempt,
                    e,
                    delay,
                )

                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

        raise RuntimeError("Retry logic completed without result or exception")

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should be retried."""
        # Check if we've exceeded max attempts
        if attempt >= self._config.max_attempts:
            return False

        # Check if exception is in non-retryable list
        if isinstance(exception, self._config.non_retryable_exceptions):
            return False

        # Check if exception is in retryable list
        if isinstance(exception, self._config.retryable_exceptions):
            return True

        # Default: don't retry
        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self._config.strategy == RetryStrategy.NO_RETRY:
            return 0.0

        if self._config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self._config.base_delay
        elif self._config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self._config.base_delay * attempt
        elif self._config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self._config.base_delay * (
                self._config.exponential_base ** (attempt - 1)
            )
        else:
            # Default to exponential backoff
            delay = self._config.base_delay * (
                self._config.exponential_base ** (attempt - 1)
            )

        # Apply maximum delay limit
        delay = min(delay, self._config.max_delay)

        # Add jitter if enabled
        if self._config.jitter_enabled:
            jitter_amount = delay * self._config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay + jitter)

        return delay

    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_attempts": self._total_attempts,
            "total_retries": self._total_retries,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": (
                self._total_successes / self._total_attempts
                if self._total_attempts > 0
                else 0
            ),
            "retry_rate": (
                self._total_retries / self._total_attempts
                if self._total_attempts > 0
                else 0
            ),
            "config": {
                "max_attempts": self._config.max_attempts,
                "base_delay": self._config.base_delay,
                "max_delay": self._config.max_delay,
                "exponential_base": self._config.exponential_base,
                "jitter_enabled": self._config.jitter_enabled,
                "strategy": self._config.strategy.value,
            },
        }
