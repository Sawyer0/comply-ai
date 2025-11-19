"""Mapper service adapter around shared rate limiting primitives."""

from __future__ import annotations

from typing import Dict, Optional

from shared.security.rate_limiting import (
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
    RateLimitingService as SharedRateLimitingService,
)

from .api_key_manager import APIKeyInfo


class RateLimitingService(SharedRateLimitingService):
    """Mapper-specific configuration for the shared rate limiting service."""

    def __init__(self, redis_client=None):
        default_configs: Dict[str, RateLimitConfig] = {
            "api_key": RateLimitConfig(
                requests_per_minute=1000,
                burst_size=100,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "ip": RateLimitConfig(
                requests_per_minute=100,
                burst_size=20,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "tenant": RateLimitConfig(
                requests_per_minute=5000,
                burst_size=500,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            # Endpoint defaults use the "endpoint:{path}" convention so the
            # shared manager can match them when evaluating a request.
            "endpoint:/api/v1/map": RateLimitConfig(
                requests_per_minute=2000,
                burst_size=200,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "endpoint:/api/v1/map/batch": RateLimitConfig(
                requests_per_minute=100,
                burst_size=10,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "endpoint:/api/v1/api-keys": RateLimitConfig(
                requests_per_minute=50,
                burst_size=10,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        }

        super().__init__(redis_client=redis_client, default_configs=default_configs)

    async def check_rate_limit(
        self, request, api_key_info: Optional[APIKeyInfo] = None
    ) -> RateLimitResult:
        """Compatibility wrapper for legacy call sites."""

        return await self.check_rate_limits(request, api_key_info)
