"""
Rate Limiting Service for Mapper Service

Integrates rate limiting with the mapper service infrastructure
following Single Responsibility Principle.
"""

from typing import Dict, Optional

import structlog
from fastapi import Request

from .rate_limiter import (
    RateLimitManager,
    RateLimitStorage,
    MemoryRateLimitStorage,
    RedisRateLimitStorage,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
)
from .api_key_manager import APIKeyInfo

logger = structlog.get_logger(__name__)


class RateLimitingService:
    """
    Rate limiting service that integrates with mapper service components.

    Responsible for:
    - Initializing rate limiting infrastructure
    - Extracting rate limit keys from requests
    - Applying tenant-specific rate limits
    - Integrating with API key management
    """

    def __init__(self, redis_client=None):
        self.logger = logger.bind(component="rate_limiting_service")

        # Initialize storage backend
        if redis_client:
            self.storage = RedisRateLimitStorage(redis_client)
            self.logger.info("Initialized Redis rate limiting storage")
        else:
            self.storage = MemoryRateLimitStorage()
            self.logger.info("Initialized memory rate limiting storage")

        # Initialize rate limit manager
        self.rate_limit_manager = RateLimitManager(self.storage)

        # Configure default rate limits
        self._configure_default_limits()

    def _configure_default_limits(self) -> None:
        """Configure default rate limit settings."""
        # API key limits (per minute)
        self.rate_limit_manager.update_default_config(
            "api_key",
            RateLimitConfig(
                requests_per_minute=1000,
                burst_size=100,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
        )

        # IP-based limits (per minute)
        self.rate_limit_manager.update_default_config(
            "ip",
            RateLimitConfig(
                requests_per_minute=100,
                burst_size=20,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
        )

        # Tenant limits (per minute)
        self.rate_limit_manager.update_default_config(
            "tenant",
            RateLimitConfig(
                requests_per_minute=5000,
                burst_size=500,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
        )

        # Endpoint-specific limits (per minute)
        endpoint_configs = {
            "/api/v1/map": RateLimitConfig(
                requests_per_minute=2000,
                burst_size=200,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "/api/v1/map/batch": RateLimitConfig(
                requests_per_minute=100,
                burst_size=10,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "/api/v1/api-keys": RateLimitConfig(
                requests_per_minute=50,
                burst_size=10,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        }

        for endpoint, config in endpoint_configs.items():
            self.rate_limit_manager.update_default_config(
                f"endpoint:{endpoint}", config
            )

    async def check_rate_limits(
        self, request: Request, api_key_info: Optional[APIKeyInfo] = None
    ) -> RateLimitResult:
        """
        Check rate limits for an incoming request.

        Args:
            request: FastAPI request object
            api_key_info: API key information if authenticated

        Returns:
            Rate limit result
        """
        try:
            # Extract identifiers
            ip_address = self._extract_client_ip(request)
            endpoint = self._normalize_endpoint(request.url.path)

            # Get custom rate limits from API key
            custom_configs = {}
            if api_key_info:
                custom_configs = self._get_api_key_rate_limits(api_key_info)

            # Check all applicable rate limits
            result = await self.rate_limit_manager.check_multiple_limits(
                api_key_id=api_key_info.key_id if api_key_info else None,
                tenant_id=api_key_info.tenant_id if api_key_info else None,
                ip_address=ip_address,
                endpoint=endpoint,
                custom_configs=custom_configs,
            )

            # Log rate limit check
            if not result.allowed:
                self.logger.warning(
                    "Rate limit exceeded",
                    ip_address=ip_address,
                    endpoint=endpoint,
                    api_key_id=api_key_info.key_id if api_key_info else None,
                    tenant_id=api_key_info.tenant_id if api_key_info else None,
                    remaining=result.remaining,
                    reset_time=result.reset_time,
                )
            else:
                self.logger.debug(
                    "Rate limit check passed",
                    ip_address=ip_address,
                    endpoint=endpoint,
                    remaining=result.remaining,
                )

            return result

        except Exception as e:
            self.logger.error("Rate limit check failed", error=str(e))
            # Fail open - allow request if rate limiting fails
            from datetime import datetime, timedelta

            return RateLimitResult(
                allowed=True,
                remaining=100,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

    def _extract_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for rate limiting."""
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]

        # Normalize path parameters (e.g., /api/v1/api-keys/123 -> /api/v1/api-keys/{id})
        path_parts = path.split("/")
        normalized_parts = []

        for part in path_parts:
            if not part:
                continue

            # Check if part looks like an ID (UUID, number, etc.)
            if (len(part) == 36 and part.count("-") == 4) or part.isdigit():
                normalized_parts.append("{id}")
            else:
                normalized_parts.append(part)

        return "/" + "/".join(normalized_parts) if normalized_parts else "/"

    def _get_api_key_rate_limits(
        self, api_key_info: APIKeyInfo
    ) -> Dict[str, RateLimitConfig]:
        """Get custom rate limits from API key configuration."""
        custom_configs = {}

        # Use API key's rate limit if specified
        if api_key_info.rate_limit_per_minute:
            custom_configs["api_key"] = RateLimitConfig(
                requests_per_minute=api_key_info.rate_limit_per_minute,
                burst_size=min(api_key_info.rate_limit_per_minute // 10, 100),
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

        # Apply tenant-specific overrides based on permissions
        if "admin:system" in api_key_info.permissions:
            # Admin keys get higher limits
            custom_configs["api_key"] = RateLimitConfig(
                requests_per_minute=10000,
                burst_size=1000,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )
        elif "map:batch" in api_key_info.permissions:
            # Batch processing keys get higher limits for mapping
            custom_configs["endpoint:/api/v1/map"] = RateLimitConfig(
                requests_per_minute=5000,
                burst_size=500,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )
            custom_configs["endpoint:/api/v1/map/batch"] = RateLimitConfig(
                requests_per_minute=500,
                burst_size=50,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

        return custom_configs

    async def get_rate_limit_status(
        self,
        api_key_info: Optional[APIKeyInfo] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Get current rate limit status for debugging/monitoring.

        Args:
            api_key_info: API key information
            ip_address: Client IP address

        Returns:
            Rate limit status information
        """
        status = {}

        try:
            if api_key_info:
                # Check API key limits
                api_key_result = (
                    await self.rate_limit_manager.rate_limiter.check_rate_limit(
                        f"api_key:{api_key_info.key_id}",
                        self.rate_limit_manager.default_configs["api_key"],
                    )
                )
                status["api_key"] = {
                    "remaining": api_key_result.remaining,
                    "reset_time": api_key_result.reset_time.isoformat(),
                    "allowed": api_key_result.allowed,
                }

                # Check tenant limits
                tenant_result = (
                    await self.rate_limit_manager.rate_limiter.check_rate_limit(
                        f"tenant:{api_key_info.tenant_id}",
                        self.rate_limit_manager.default_configs["tenant"],
                    )
                )
                status["tenant"] = {
                    "remaining": tenant_result.remaining,
                    "reset_time": tenant_result.reset_time.isoformat(),
                    "allowed": tenant_result.allowed,
                }

            if ip_address:
                # Check IP limits
                ip_result = await self.rate_limit_manager.rate_limiter.check_rate_limit(
                    f"ip:{ip_address}", self.rate_limit_manager.default_configs["ip"]
                )
                status["ip"] = {
                    "remaining": ip_result.remaining,
                    "reset_time": ip_result.reset_time.isoformat(),
                    "allowed": ip_result.allowed,
                }

            return status

        except Exception as e:
            self.logger.error("Failed to get rate limit status", error=str(e))
            return {"error": str(e)}

    async def reset_rate_limits(
        self,
        api_key_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Reset rate limits for debugging/admin purposes.

        Args:
            api_key_id: API key to reset
            tenant_id: Tenant to reset
            ip_address: IP address to reset

        Returns:
            True if successful
        """
        try:
            # This would require implementing a reset method in the storage backend
            # For now, we'll log the request
            self.logger.info(
                "Rate limit reset requested",
                api_key_id=api_key_id,
                tenant_id=tenant_id,
                ip_address=ip_address,
            )

            # In a full implementation, this would clear the relevant keys from storage
            return True

        except Exception as e:
            self.logger.error("Failed to reset rate limits", error=str(e))
            return False
