"""Analysis service adapter for the shared rate limiting primitives."""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

from shared.security.rate_limiting import (
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitingService,
)

from .config import SecurityConfig
from .exceptions import RateLimitError

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Wraps the shared rate limiting service with analysis defaults."""

    def __init__(self, config: SecurityConfig, redis_client: Any = None):
        self.config = config
        self.logger = logger.bind(component="rate_limiter")

        default_configs: Dict[str, RateLimitConfig] = {
            "api_key": RateLimitConfig(
                requests_per_minute=config.rate_limit_requests_per_minute,
                burst_size=config.rate_limit_burst_size,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "ip": RateLimitConfig(
                requests_per_minute=60,
                burst_size=15,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "endpoint": RateLimitConfig(
                requests_per_minute=500,
                burst_size=100,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        }

        self._service = RateLimitingService(
            redis_client=redis_client,
            default_configs=default_configs,
        )

    async def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """Validate the request against configured limits."""

        result = await self._service.manager.check_multiple_limits(
            api_key_id=client_id or None,
            endpoint=endpoint,
        )

        if not result.allowed:
            self.logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                endpoint=endpoint,
                remaining=result.remaining,
                retry_after=result.retry_after,
            )
            raise RateLimitError(f"Rate limit exceeded for {client_id} on {endpoint}")

        return True

    async def get_rate_limit_status(self, client_id: str, endpoint: str) -> Dict[str, Any]:
        """Return the remaining allowance for the given client and endpoint."""

        result = await self._service.manager.check_multiple_limits(
            api_key_id=client_id or None,
            endpoint=endpoint,
        )

        api_key_config = self._service.manager.default_configs.get("api_key")
        limit = (
            api_key_config.requests_per_minute
            if api_key_config is not None
            else self.config.rate_limit_requests_per_minute
        )

        return {
            "tokens_remaining": result.remaining,
            "reset_time": result.reset_time.isoformat(),
            "limit": limit,
        }

    async def clear_rate_limits(self, client_id: Optional[str] = None) -> None:
        """Shared service handles expiry; explicit clearing is not yet supported."""

        self.logger.info(
            "Rate limit reset requested but not implemented",
            client_id=client_id,
        )

    async def get_metrics(self) -> Dict[str, Any]:
        """Expose configured defaults for observability."""

        defaults = {
            key: {
                "requests_per_minute": config.requests_per_minute,
                "burst_size": config.burst_size,
                "strategy": config.strategy.value,
            }
            for key, config in self._service.manager.default_configs.items()
        }

        return {"default_limits": defaults}
