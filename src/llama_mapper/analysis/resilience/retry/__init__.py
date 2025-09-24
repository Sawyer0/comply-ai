"""Retry logic implementation package."""

from ..config.retry_config import RetryConfig
from ..interfaces import RetryStrategy
from .implementation import RetryManager

# Create aliases for backwards compatibility
RetryDecorator = RetryManager  # Alias for backwards compatibility
with_retry = RetryManager.with_retry

__all__ = [
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryDecorator",
    "with_retry",
]
