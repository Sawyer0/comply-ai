"""
Rate limiting module for the Analysis Service.

Implements token bucket algorithm for rate limiting.
"""

import time
from typing import Any, Dict

import structlog

from .config import SecurityConfig
from .exceptions import RateLimitError

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="rate_limiter")
        self._rate_limit_buckets: Dict[str, Dict[str, Any]] = {}

    async def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """
        Check if request is within rate limits.

        Args:
            client_id: Client identifier (IP, user ID, etc.)
            endpoint: API endpoint being accessed

        Returns:
            True if within limits, False otherwise

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        bucket_key = f"{client_id}:{endpoint}"
        now = time.time()

        if bucket_key not in self._rate_limit_buckets:
            self._rate_limit_buckets[bucket_key] = {
                "tokens": self.config.rate_limit_burst_size,
                "last_refill": now,
            }

        bucket = self._rate_limit_buckets[bucket_key]

        # Refill tokens based on time elapsed
        time_elapsed = now - bucket["last_refill"]
        tokens_to_add = time_elapsed * (
            self.config.rate_limit_requests_per_minute / 60.0
        )
        bucket["tokens"] = min(
            self.config.rate_limit_burst_size, bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now

        # Check if request can be processed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        self.logger.warning(
            "Rate limit exceeded",
            client_id=client_id,
            endpoint=endpoint,
            tokens_remaining=bucket["tokens"],
        )
        raise RateLimitError(f"Rate limit exceeded for {client_id} on {endpoint}")

    async def get_rate_limit_status(
        self, client_id: str, endpoint: str
    ) -> Dict[str, Any]:
        """
        Get current rate limit status for client and endpoint.

        Args:
            client_id: Client identifier
            endpoint: API endpoint

        Returns:
            Rate limit status information
        """
        bucket_key = f"{client_id}:{endpoint}"

        if bucket_key not in self._rate_limit_buckets:
            return {
                "tokens_remaining": self.config.rate_limit_burst_size,
                "reset_time": time.time(),
                "limit": self.config.rate_limit_requests_per_minute,
            }

        bucket = self._rate_limit_buckets[bucket_key]

        return {
            "tokens_remaining": int(bucket["tokens"]),
            "reset_time": bucket["last_refill"] + 60,  # Reset every minute
            "limit": self.config.rate_limit_requests_per_minute,
        }

    def clear_rate_limits(self, client_id: str = None) -> None:
        """
        Clear rate limit buckets.

        Args:
            client_id: Optional client ID to clear specific buckets
        """
        if client_id:
            # Clear buckets for specific client
            keys_to_remove = [
                key
                for key in self._rate_limit_buckets.keys()
                if key.startswith(f"{client_id}:")
            ]
            for key in keys_to_remove:
                del self._rate_limit_buckets[key]
        else:
            # Clear all buckets
            self._rate_limit_buckets.clear()

        self.logger.info("Rate limit buckets cleared", client_id=client_id)
