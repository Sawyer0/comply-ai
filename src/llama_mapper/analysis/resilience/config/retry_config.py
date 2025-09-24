"""
Retry configuration classes.

This module provides configuration classes for retry logic
with validation and environment-specific defaults.
"""

from dataclasses import dataclass
from typing import Tuple

from ..interfaces import RetryStrategy


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
        """Create RetryConfig from analysis settings."""
        return cls(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
            exponential_base=settings.retry_exponential_base,
            jitter_enabled=settings.retry_jitter_enabled,
            jitter_range=settings.retry_jitter_range,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,  # Default strategy
        )
