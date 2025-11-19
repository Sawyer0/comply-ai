"""Adapter around shared rate limiting primitives for orchestration service."""

from __future__ import annotations

from typing import Any, Dict, Optional

import logging

from shared.security.rate_limiting import (
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
    RateLimitingService as SharedRateLimitingService,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Wrap the shared rate limiting service with orchestration defaults."""

    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        tenant_limit: int = 120,
        window_seconds: int = 60,
        tenant_overrides: Optional[Dict[str, int]] = None,
    ) -> None:
        tenant_overrides = tenant_overrides or {}

        default_configs: Dict[str, RateLimitConfig] = {
            "tenant": RateLimitConfig(
                requests_per_minute=tenant_limit,
                burst_size=min(tenant_limit, max(tenant_limit // 10, 1)),
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                window_size_seconds=window_seconds,
            ),
        }

        for tenant_id, override in tenant_overrides.items():
            default_configs[f"tenant:{tenant_id}"] = RateLimitConfig(
                requests_per_minute=override,
                burst_size=min(override, max(override // 10, 1)),
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                window_size_seconds=window_seconds,
            )

        self._service = SharedRateLimitingService(
            redis_client=redis_client,
            default_configs=default_configs,
        )
        self.logger = logger.getChild("orchestration_rate_limiter")

    async def check_rate_limit(
        self,
        tenant_id: Optional[str],
        *,
        api_key_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> RateLimitResult:
        """Validate the incoming request against all configured limits."""

        result = await self._service.manager.check_multiple_limits(
            api_key_id=api_key_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            endpoint=endpoint,
        )

        if not result.allowed:
            self.logger.warning(
                "Rate limit exceeded",
                extra={
                    "tenant_id": tenant_id,
                    "endpoint": endpoint,
                    "remaining": result.remaining,
                    "retry_after": result.retry_after,
                },
            )

        return result

    async def get_rate_limit_status(self, tenant_id: str) -> Dict[str, Any]:
        """Return remaining allowance for observability endpoints."""

        status = await self._service.get_rate_limit_status(tenant_id=tenant_id)
        return status.get("tenant", {}) if status else {}

    async def reset_rate_limit(self, tenant_id: str) -> None:
        """Clearing limits is handled via TTL in shared service; log request."""

        self.logger.info("Rate limit reset requested", extra={"tenant_id": tenant_id})

    async def get_metrics(self) -> Dict[str, Any]:
        """Expose default configuration for metric exports."""

        defaults = {
            key: {
                "requests_per_minute": config.requests_per_minute,
                "burst_size": config.burst_size,
                "strategy": config.strategy.value,
                "window_size_seconds": config.window_size_seconds,
            }
            for key, config in self._service.manager.default_configs.items()
        }

        return {"default_limits": defaults}


__all__ = [
    "RateLimiter",
    "RateLimitResult",
    "RateLimitStrategy",
]
