"""Rate limiting functionality following SRP.

This module provides ONLY rate limiting - controlling request rates per tenant/user.
Single Responsibility: Enforce rate limits to prevent abuse and ensure fair usage.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class RateLimitResult:
    """Result of rate limit check."""

    def __init__(
        self,
        allowed: bool,
        remaining: int,
        reset_time: datetime,
        retry_after: Optional[int] = None,
    ):
        self.allowed = allowed
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after  # seconds until next request allowed


class RateLimiter:
    """Rate limiter for controlling request rates.

    Single Responsibility: Enforce rate limits based on configurable strategies.
    Does NOT handle: authentication, authorization, metrics collection.
    """

    def __init__(
        self,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        cleanup_interval: int = 300,  # 5 minutes
    ):
        """Initialize rate limiter.

        Args:
            strategy: Rate limiting strategy to use
            cleanup_interval: Interval for cleaning up expired entries (seconds)
        """
        self.strategy = strategy
        self.cleanup_interval = cleanup_interval

        # Storage for rate limit data
        # Format: {key: {timestamp: count, ...}} for sliding window
        # Format: {key: (count, window_start)} for fixed window
        # Format: {key: (tokens, last_refill)} for token bucket
        self._rate_data: Dict[str, any] = {}

        # Last cleanup time
        self._last_cleanup = time.time()

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int, tenant_id: Optional[str] = None
    ) -> RateLimitResult:
        """Check if request is within rate limit.

        Args:
            key: Rate limit key (e.g., tenant_id, user_id, ip_address)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            tenant_id: Optional tenant ID for logging

        Returns:
            RateLimitResult indicating if request is allowed
        """
        correlation_id = get_correlation_id()
        current_time = time.time()

        try:
            # Perform cleanup if needed
            await self._cleanup_expired_entries(current_time)

            if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = self._check_sliding_window(
                    key, limit, window_seconds, current_time
                )
            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = self._check_fixed_window(
                    key, limit, window_seconds, current_time
                )
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = self._check_token_bucket(
                    key, limit, window_seconds, current_time
                )
            else:
                # Default to sliding window
                result = self._check_sliding_window(
                    key, limit, window_seconds, current_time
                )

            # Log rate limit check
            if not result.allowed:
                logger.warning(
                    "Rate limit exceeded for key %s (tenant: %s)",
                    key,
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "rate_limit_key": key,
                        "tenant_id": tenant_id,
                        "limit": limit,
                        "window_seconds": window_seconds,
                        "remaining": result.remaining,
                        "retry_after": result.retry_after,
                    },
                )
            else:
                logger.debug(
                    "Rate limit check passed for key %s (remaining: %d)",
                    key,
                    result.remaining,
                    extra={
                        "correlation_id": correlation_id,
                        "rate_limit_key": key,
                        "tenant_id": tenant_id,
                        "remaining": result.remaining,
                    },
                )

            return result

        except Exception as e:
            logger.error(
                "Rate limit check failed for key %s: %s",
                key,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "rate_limit_key": key,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            # On error, allow the request (fail open)
            return RateLimitResult(
                allowed=True,
                remaining=limit,
                reset_time=datetime.fromtimestamp(current_time + window_seconds),
            )

    def _check_sliding_window(
        self, key: str, limit: int, window_seconds: int, current_time: float
    ) -> RateLimitResult:
        """Check rate limit using sliding window strategy."""

        if key not in self._rate_data:
            self._rate_data[key] = {}

        window_data = self._rate_data[key]
        window_start = current_time - window_seconds

        # Remove expired entries
        expired_times = [t for t in window_data.keys() if t < window_start]
        for t in expired_times:
            del window_data[t]

        # Count current requests in window
        current_count = sum(window_data.values())

        if current_count >= limit:
            # Rate limit exceeded
            oldest_time = min(window_data.keys()) if window_data else current_time
            reset_time = datetime.fromtimestamp(oldest_time + window_seconds)
            retry_after = int(oldest_time + window_seconds - current_time)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(retry_after, 1),
            )

        # Add current request
        window_data[current_time] = window_data.get(current_time, 0) + 1
        remaining = limit - (current_count + 1)

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_time=datetime.fromtimestamp(current_time + window_seconds),
        )

    def _check_fixed_window(
        self, key: str, limit: int, window_seconds: int, current_time: float
    ) -> RateLimitResult:
        """Check rate limit using fixed window strategy."""

        window_start = int(current_time // window_seconds) * window_seconds

        if key not in self._rate_data:
            self._rate_data[key] = (0, window_start)

        count, stored_window_start = self._rate_data[key]

        # Reset if new window
        if stored_window_start < window_start:
            count = 0
            stored_window_start = window_start

        if count >= limit:
            # Rate limit exceeded
            reset_time = datetime.fromtimestamp(window_start + window_seconds)
            retry_after = int(window_start + window_seconds - current_time)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(retry_after, 1),
            )

        # Increment count
        self._rate_data[key] = (count + 1, stored_window_start)
        remaining = limit - (count + 1)

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_time=datetime.fromtimestamp(window_start + window_seconds),
        )

    def _check_token_bucket(
        self, key: str, limit: int, window_seconds: int, current_time: float
    ) -> RateLimitResult:
        """Check rate limit using token bucket strategy."""

        if key not in self._rate_data:
            self._rate_data[key] = (limit, current_time)

        tokens, last_refill = self._rate_data[key]

        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = int(time_elapsed * (limit / window_seconds))
        tokens = min(limit, tokens + tokens_to_add)

        if tokens < 1:
            # No tokens available
            time_until_token = (1.0 / (limit / window_seconds)) - (
                current_time - last_refill
            )
            retry_after = max(int(time_until_token), 1)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp(current_time + time_until_token),
                retry_after=retry_after,
            )

        # Consume one token
        self._rate_data[key] = (tokens - 1, current_time)

        return RateLimitResult(
            allowed=True,
            remaining=tokens - 1,
            reset_time=datetime.fromtimestamp(current_time + window_seconds),
        )

    async def _cleanup_expired_entries(self, current_time: float):
        """Clean up expired rate limit entries."""

        if current_time - self._last_cleanup < self.cleanup_interval:
            return

        try:
            keys_to_remove = []

            for key, data in self._rate_data.items():
                if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    # Remove if no recent activity
                    if isinstance(data, dict) and data:
                        latest_time = max(data.keys())
                        if current_time - latest_time > self.cleanup_interval:
                            keys_to_remove.append(key)
                    elif not data:
                        keys_to_remove.append(key)

                elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                    # Remove if window is old
                    if isinstance(data, tuple) and len(data) == 2:
                        _, window_start = data
                        if current_time - window_start > self.cleanup_interval:
                            keys_to_remove.append(key)

                elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    # Remove if not used recently
                    if isinstance(data, tuple) and len(data) == 2:
                        _, last_refill = data
                        if current_time - last_refill > self.cleanup_interval:
                            keys_to_remove.append(key)

            # Remove expired keys
            for key in keys_to_remove:
                del self._rate_data[key]

            self._last_cleanup = current_time

            if keys_to_remove:
                logger.debug(
                    "Cleaned up %d expired rate limit entries",
                    len(keys_to_remove),
                    extra={"correlation_id": get_correlation_id()},
                )

        except Exception as e:
            logger.error(
                "Failed to cleanup expired rate limit entries: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a specific key.

        Args:
            key: Rate limit key to reset

        Returns:
            True if reset successful, False if key not found
        """
        try:
            if key in self._rate_data:
                del self._rate_data[key]
                logger.info(
                    "Reset rate limit for key %s",
                    key,
                    extra={"correlation_id": get_correlation_id()},
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to reset rate limit for key %s: %s",
                key,
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )
            return False

    def get_rate_limit_status(self, key: str) -> Optional[Dict[str, any]]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Dictionary with rate limit status or None if key not found
        """
        try:
            if key not in self._rate_data:
                return None

            data = self._rate_data[key]
            current_time = time.time()

            if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                if isinstance(data, dict):
                    current_count = sum(data.values())
                    return {
                        "strategy": self.strategy.value,
                        "current_count": current_count,
                        "requests": list(data.keys()),
                    }

            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                if isinstance(data, tuple) and len(data) == 2:
                    count, window_start = data
                    return {
                        "strategy": self.strategy.value,
                        "current_count": count,
                        "window_start": window_start,
                        "window_remaining": window_start
                        + 3600
                        - current_time,  # Assuming 1 hour window
                    }

            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                if isinstance(data, tuple) and len(data) == 2:
                    tokens, last_refill = data
                    return {
                        "strategy": self.strategy.value,
                        "available_tokens": tokens,
                        "last_refill": last_refill,
                    }

            return {"strategy": self.strategy.value, "data": str(data)}

        except Exception as e:
            logger.error(
                "Failed to get rate limit status for key %s: %s",
                key,
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )
            return None

    def get_statistics(self) -> Dict[str, any]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            return {
                "strategy": self.strategy.value,
                "total_keys": len(self._rate_data),
                "cleanup_interval": self.cleanup_interval,
                "last_cleanup": self._last_cleanup,
                "memory_usage_keys": list(self._rate_data.keys())[
                    :10
                ],  # Sample of keys
            }
        except Exception as e:
            logger.error(
                "Failed to get rate limiter statistics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )
            return {}


# Export only the rate limiting functionality
__all__ = [
    "RateLimiter",
    "RateLimitResult",
    "RateLimitStrategy",
]
