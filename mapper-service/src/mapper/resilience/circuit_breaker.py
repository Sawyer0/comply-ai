"""
Circuit breaker implementation for mapper service.

This module provides the core circuit breaker implementation
with state management and failure detection specifically
tailored for mapper service operations.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from .config import CircuitBreakerConfig
from .interfaces import CircuitState, ICircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""


class MapperCircuitBreaker(ICircuitBreaker):
    """
    Circuit breaker implementation for mapper operations.

    Provides automatic failure detection and recovery for mapper service calls,
    preventing cascading failures and improving system resilience.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        metrics_collector: Optional[Any] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker (e.g., "model_server")
            config: Circuit breaker configuration
            metrics_collector: Metrics collector for monitoring
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._metrics_collector = metrics_collector

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._last_state_change = time.time()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_timeouts = 0

        logger.info(
            "Mapper circuit breaker '%s' initialized with config: %s",
            name,
            self._config,
        )

    @property
    def name(self) -> str:
        """Circuit breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        self._total_calls += 1

        # Check circuit state
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenException(
                    f"Mapper circuit breaker '{self._name}' is OPEN. "
                    f"Last failure: {time.time() - self._last_failure_time:.1f}s ago"
                )

        # Execute function with timeout
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self._config.timeout
            )

            # Handle success
            await self._handle_success()
            return result

        except asyncio.TimeoutError as e:
            await self._handle_failure(TimeoutError("Mapper call timed out"))
            raise e

        except Exception as e:
            # Check if this is an expected exception type
            if isinstance(e, self._config.expected_exception):
                await self._handle_failure(e)
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return time.time() - self._last_failure_time >= self._config.recovery_timeout

    async def _handle_success(self):
        """Handle successful call."""
        self._total_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

        # Record metrics
        if self._metrics_collector:
            self._metrics_collector.record_circuit_breaker_state("closed", self._name)

    async def _handle_failure(self, exception: Exception):
        """Handle failed call."""
        self._total_failures += 1
        self._last_failure_time = time.time()

        if isinstance(exception, TimeoutError):
            self._total_timeouts += 1

        if self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._transition_to_open()
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open state goes back to open
            self._transition_to_open()

        # Record metrics
        if self._metrics_collector:
            self._metrics_collector.record_circuit_breaker_state("open", self._name)

        logger.warning(
            "Mapper circuit breaker '%s' failure: %s. State: %s, Failures: %d",
            self._name,
            exception,
            self._state.value,
            self._failure_count,
        )

    def _transition_to_open(self):
        """Transition circuit to open state."""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._failure_count = 0
            self._success_count = 0
            self._last_state_change = time.time()

            logger.warning(
                "Mapper circuit breaker '%s' opened due to failures", self._name
            )

    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        if self._state != CircuitState.HALF_OPEN:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._last_state_change = time.time()

            logger.info(
                "Mapper circuit breaker '%s' transitioned to half-open", self._name
            )

    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_state_change = time.time()

            logger.info(
                "Mapper circuit breaker '%s' closed - service recovered", self._name
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        uptime = time.time() - self._last_state_change

        return {
            "name": self._name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_timeouts": self._total_timeouts,
            "failure_rate": (
                self._total_failures / self._total_calls if self._total_calls > 0 else 0
            ),
            "success_rate": (
                self._total_successes / self._total_calls
                if self._total_calls > 0
                else 0
            ),
            "last_failure_time": self._last_failure_time,
            "uptime_seconds": uptime,
            "config": {
                "failure_threshold": self._config.failure_threshold,
                "recovery_timeout": self._config.recovery_timeout,
                "success_threshold": self._config.success_threshold,
                "timeout": self._config.timeout,
            },
        }

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        self._transition_to_closed()
        logger.info("Mapper circuit breaker '%s' manually reset", self._name)

    def force_open(self):
        """Manually force circuit breaker to open state."""
        self._transition_to_open()
        logger.warning("Mapper circuit breaker '%s' manually opened", self._name)
