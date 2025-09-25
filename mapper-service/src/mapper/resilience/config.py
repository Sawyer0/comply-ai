"""
Configuration classes for mapper service resilience patterns.

This module provides configuration classes for circuit breaker and retry
functionality with validation and environment-specific defaults.
"""

from dataclasses import dataclass
from typing import Tuple

from .interfaces import RetryStrategy


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
        """Create CircuitBreakerConfig from mapper settings."""
        return cls(
            failure_threshold=getattr(settings, "circuit_breaker_failure_threshold", 5),
            recovery_timeout=getattr(
                settings, "circuit_breaker_recovery_timeout", 60.0
            ),
            success_threshold=getattr(settings, "circuit_breaker_success_threshold", 3),
            timeout=getattr(settings, "circuit_breaker_timeout", 30.0),
            expected_exception=(Exception,),  # Default to all exceptions
        )


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True
    jitter_range: float = 0.1
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_attempts < 1 or self.max_attempts > 10:
            raise ValueError("max_attempts must be between 1 and 10")

        if self.base_delay < 0.1 or self.base_delay > 60.0:
            raise ValueError("base_delay must be between 0.1 and 60.0 seconds")

        if self.max_delay < 1.0 or self.max_delay > 300.0:
            raise ValueError("max_delay must be between 1.0 and 300.0 seconds")

        if self.exponential_base < 1.1 or self.exponential_base > 5.0:
            raise ValueError("exponential_base must be between 1.1 and 5.0")

        if self.jitter_range < 0.0 or self.jitter_range > 1.0:
            raise ValueError("jitter_range must be between 0.0 and 1.0")

    @classmethod
    def from_settings(cls, settings) -> "RetryConfig":
        """Create RetryConfig from mapper settings."""
        return cls(
            max_attempts=getattr(settings, "retry_max_attempts", 3),
            base_delay=getattr(settings, "retry_base_delay", 1.0),
            max_delay=getattr(settings, "retry_max_delay", 60.0),
            exponential_base=getattr(settings, "retry_exponential_base", 2.0),
            jitter_enabled=getattr(settings, "retry_jitter_enabled", True),
            jitter_range=getattr(settings, "retry_jitter_range", 0.1),
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,  # Default strategy
        )
