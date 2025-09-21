"""Circuit breaker implementation for fault tolerance.

This module provides circuit breaker functionality to prevent cascading failures
when downstream detectors are unavailable.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for detector fault tolerance."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60):
        """Initialize circuit breaker with failure threshold and recovery timeout."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure = 0.0

    def allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if we can transition to HALF_OPEN
            if (time.time() - self.last_failure) >= self.recovery_timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        """Record successful request, reset failure count and close circuit."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed request, increment failure count and potentially open circuit."""
        self.failure_count += 1
        self.last_failure = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    # pylint: disable=too-few-public-methods
    def __init__(self, failure_threshold: int, recovery_timeout_seconds: int):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._ft = failure_threshold
        self._rt = recovery_timeout_seconds

    def get(self, detector: str) -> CircuitBreaker:
        """Get circuit breaker for a specific detector."""
        br = self._breakers.get(detector)
        if not br:
            br = CircuitBreaker(self._ft, self._rt)
            self._breakers[detector] = br
        return br
