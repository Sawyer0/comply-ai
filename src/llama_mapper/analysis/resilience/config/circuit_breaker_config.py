"""
Circuit breaker configuration classes.

This module provides configuration classes for circuit breaker
functionality with validation and environment-specific defaults.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0
    expected_exception: tuple = (Exception,)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.failure_threshold < 1 or self.failure_threshold > 20:
            raise ValueError("failure_threshold must be between 1 and 20")

        if self.recovery_timeout < 10.0 or self.recovery_timeout > 300.0:
            raise ValueError("recovery_timeout must be between 10.0 and 300.0 seconds")

        if self.success_threshold < 1 or self.success_threshold > 10:
            raise ValueError("success_threshold must be between 1 and 10")

        if self.timeout < 5.0 or self.timeout > 120.0:
            raise ValueError("timeout must be between 5.0 and 120.0 seconds")

    @classmethod
    def from_settings(cls, settings) -> "CircuitBreakerConfig":
        """Create CircuitBreakerConfig from analysis settings."""
        return cls(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout,
            success_threshold=settings.circuit_breaker_success_threshold,
            timeout=settings.circuit_breaker_timeout,
            expected_exception=(Exception,),  # Default to all exceptions
        )
