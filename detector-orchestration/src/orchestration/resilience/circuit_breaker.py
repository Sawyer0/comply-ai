"""Consolidated circuit breaker functionality.

This module consolidates the existing circuit breaker logic from the
scattered detector_orchestration module into the organized orchestration structure.
"""

import logging
from typing import Dict, Optional

# Import existing functionality to consolidate
from ...detector_orchestration.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerState,
)

logger = logging.getLogger(__name__)


class ConsolidatedCircuitBreakerManager:
    """Consolidated circuit breaker manager that wraps and enhances existing functionality."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60):
        # Use existing circuit breaker manager as the core
        self.manager = CircuitBreakerManager(
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
        )

        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

    def get_circuit_breaker(self, detector_id: str) -> CircuitBreaker:
        """Get circuit breaker for a specific detector."""

        breaker = self.manager.get(detector_id)

        logger.debug(
            "Retrieved circuit breaker for detector %s (state: %s)",
            detector_id,
            breaker.state.value,
            extra={
                "detector_id": detector_id,
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
            },
        )

        return breaker

    def get_all_states(self) -> Dict[str, CircuitBreakerState]:
        """Get states of all circuit breakers."""

        states = {}
        for detector_id, breaker in self.manager._breakers.items():
            states[detector_id] = breaker.state

        return states

    def reset_circuit_breaker(self, detector_id: str) -> bool:
        """Reset a specific circuit breaker."""

        if detector_id in self.manager._breakers:
            breaker = self.manager._breakers[detector_id]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0

            logger.info(
                "Circuit breaker reset for detector %s",
                detector_id,
                extra={"detector_id": detector_id},
            )

            return True

        return False

    def reset_all_circuit_breakers(self) -> int:
        """Reset all circuit breakers."""

        reset_count = 0
        for detector_id in list(self.manager._breakers.keys()):
            if self.reset_circuit_breaker(detector_id):
                reset_count += 1

        logger.info(
            "Reset %d circuit breakers", reset_count, extra={"reset_count": reset_count}
        )

        return reset_count

    def get_failure_counts(self) -> Dict[str, int]:
        """Get failure counts for all circuit breakers."""

        counts = {}
        for detector_id, breaker in self.manager._breakers.items():
            counts[detector_id] = breaker.failure_count

        return counts

    def is_detector_available(self, detector_id: str) -> bool:
        """Check if a detector is available (circuit breaker allows requests)."""

        breaker = self.get_circuit_breaker(detector_id)
        return breaker.allow_request()


class ConsolidatedCircuitBreaker:
    """Consolidated circuit breaker that wraps and enhances existing functionality."""

    def __init__(
        self,
        detector_id: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
    ):
        # Use existing circuit breaker as the core
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
        )

        self.detector_id = detector_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

    def allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker."""

        allowed = self.circuit_breaker.allow_request()

        if not allowed:
            logger.warning(
                "Circuit breaker blocking request for detector %s",
                self.detector_id,
                extra={
                    "detector_id": self.detector_id,
                    "state": self.circuit_breaker.state.value,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        return allowed

    def record_success(self) -> None:
        """Record successful request."""

        old_state = self.circuit_breaker.state
        self.circuit_breaker.record_success()

        if old_state != self.circuit_breaker.state:
            logger.info(
                "Circuit breaker state changed for detector %s: %s -> %s",
                self.detector_id,
                old_state.value,
                self.circuit_breaker.state.value,
                extra={
                    "detector_id": self.detector_id,
                    "old_state": old_state.value,
                    "new_state": self.circuit_breaker.state.value,
                },
            )

    def record_failure(self) -> None:
        """Record failed request."""

        old_state = self.circuit_breaker.state
        old_count = self.circuit_breaker.failure_count

        self.circuit_breaker.record_failure()

        logger.warning(
            "Circuit breaker recorded failure for detector %s (count: %d -> %d)",
            self.detector_id,
            old_count,
            self.circuit_breaker.failure_count,
            extra={
                "detector_id": self.detector_id,
                "failure_count": self.circuit_breaker.failure_count,
                "state": self.circuit_breaker.state.value,
            },
        )

        if old_state != self.circuit_breaker.state:
            logger.error(
                "Circuit breaker opened for detector %s after %d failures",
                self.detector_id,
                self.circuit_breaker.failure_count,
                extra={
                    "detector_id": self.detector_id,
                    "failure_count": self.circuit_breaker.failure_count,
                    "state": self.circuit_breaker.state.value,
                },
            )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.circuit_breaker.state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self.circuit_breaker.failure_count


# Re-export existing classes for consolidated access
__all__ = [
    "ConsolidatedCircuitBreakerManager",
    "ConsolidatedCircuitBreaker",
    "CircuitBreakerManager",  # Original for backward compatibility
    "CircuitBreaker",  # Original for backward compatibility
    "CircuitBreakerState",
]
