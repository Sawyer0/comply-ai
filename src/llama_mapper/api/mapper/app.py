"""Application wiring for the Llama Mapper FastAPI service."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, cast

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from ...config.manager import ConfigManager
from ...monitoring.metrics_collector import MetricsCollector
from ...reporting.audit_trail import AuditTrailManager
from ...serving.fallback_mapper import FallbackMapper
from ...serving.json_validator import JSONValidator
from ...serving.model_server import ModelServer
from ...storage.manager import StorageManager
from ...versioning import VersionManager
from ..auth import IdempotencyCache, build_api_key_auth
from ..middleware.correlation import CorrelationMiddleware
from .routes import register_routes
from .service import MappingService
from ..demo import router as demo_router

logger = logging.getLogger(__name__)


class MapperAPI:
    """FastAPI application for the Llama Mapper service.

    Note: This class has many instance attributes (13) due to the complexity
    of the application setup. This is intentional and follows the pattern of
    a comprehensive application factory.
    """

    def _initialize_storage_manager(self) -> Optional[StorageManager]:
        """Initialize storage manager if configured."""
        storage_settings = getattr(self.config_manager, "storage", None)
        if storage_settings and getattr(storage_settings, "s3_bucket", None):
            try:
                return StorageManager(storage_settings)
            except (ImportError, AttributeError, ValueError, RuntimeError) as e:
                logger.warning(
                    "Storage manager initialization failed: %s", e, exc_info=True
                )
        return None

    def _configure_app(self) -> None:
        """Configure the FastAPI application instance."""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            del app  # unused
            try:
                if self.storage_manager:
                    try:
                        await self.storage_manager.initialize()
                        logger.info("Storage manager initialized")
                    except (ImportError, AttributeError, ValueError, RuntimeError):
                        logger.warning(
                            "Storage manager unavailable, continuing without persistence",
                            exc_info=True,
                        )
                        self.storage_manager = None
                yield
            finally:
                try:
                    close_fn = getattr(self.model_server, "close", None)
                    if close_fn:
                        if asyncio.iscoroutinefunction(close_fn):
                            await close_fn()
                        else:
                            close_fn()
                    logger.info("Shutdown complete: model server resources released")
                except (AttributeError, RuntimeError) as exc:
                    logger.warning(
                        "Shutdown cleanup encountered an error: %s", exc, exc_info=True
                    )
                if self.storage_manager:
                    try:
                        await self.storage_manager.close()
                    except (AttributeError, RuntimeError):
                        logger.warning(
                            "Failed to close storage manager cleanly", exc_info=True
                        )

        self.app = FastAPI(
            title="Llama Mapper API",
            description="AI safety detector output normalization service",
            version="1.0.0",
            lifespan=lifespan,
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",  # Development frontend
                "http://localhost:8080",  # Development dashboard
                "https://app.comply-ai.com",  # Production frontend
                "https://dashboard.comply-ai.com",  # Production dashboard
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Authorization", 
                "X-API-Key",
                "X-Tenant-ID",
                "X-Correlation-ID",
                "X-Request-ID"
            ],
        )

        # Add correlation ID middleware
        self.app.add_middleware(CorrelationMiddleware)

        @self.app.middleware("http")
        async def add_request_id(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            """Add request ID to each request for tracing."""
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

    def _configure_auth(self) -> None:
        """Configure authentication and API key handling."""
        api_key_header_name = "X-API-Key"
        try:
            api_key_header_name = cast(
                str,
                getattr(
                    getattr(self.config_manager, "security", None),
                    "api_key_header",
                    "X-API-Key",
                ),
            )
        except (AttributeError, TypeError):
            api_key_header_name = "X-API-Key"
        self._api_key_header_for_docs = APIKeyHeader(
            name=api_key_header_name, auto_error=False
        )
        self._auth_map_write = build_api_key_auth(
            self.config_manager, required_scopes=["map:write"]
        )

    def _configure_idempotency(self) -> None:
        """Configure idempotency cache with Redis fallback."""
        ttl_seconds = self._get_idempotency_ttl()
        backend_choice, redis_url, redis_prefix = self._get_idempotency_config()

        if backend_choice == "redis" and redis_url:
            self._setup_redis_idempotency(ttl_seconds, redis_url, redis_prefix)
        else:
            self._setup_memory_idempotency(ttl_seconds)

    def _get_idempotency_ttl(self) -> int:
        """Get idempotency cache TTL from configuration."""
        try:
            ttl_candidate = getattr(
                getattr(self.config_manager, "auth", None),
                "idempotency_cache_ttl_seconds",
                600,
            )
            return ttl_candidate if isinstance(ttl_candidate, int) else 600
        except (AttributeError, TypeError, ValueError):
            return 600

    def _get_idempotency_config(self) -> tuple[str, Optional[str], str]:
        """Get idempotency backend configuration."""
        try:
            auth_cfg = getattr(self.config_manager, "auth", None)
            backend_choice = getattr(auth_cfg, "idempotency_backend", "memory")
            redis_url = getattr(auth_cfg, "idempotency_redis_url", None)
            redis_prefix = getattr(auth_cfg, "idempotency_redis_prefix", "idem:mapper:")
            return backend_choice, redis_url, redis_prefix
        except (AttributeError, TypeError):
            return "memory", None, "idem:mapper:"

    def _setup_redis_idempotency(
        self, ttl_seconds: int, redis_url: str, redis_prefix: str
    ) -> None:
        """Set up Redis-based idempotency cache."""
        try:
            # Import inside try block since Redis is optional dependency
            from ..idempotency_redis import RedisIdempotencyCache

            redis_cache = RedisIdempotencyCache(
                redis_url=redis_url,
                key_prefix=str(redis_prefix),
                ttl_seconds=ttl_seconds,
            )
            if redis_cache.is_healthy():
                self._idempotency_cache = redis_cache
                self._record_redis_metrics(success=True)
            else:
                self._setup_memory_idempotency(ttl_seconds)
                self._record_redis_metrics(success=False, fallback=True)
        except (ImportError, AttributeError, ValueError, RuntimeError):
            self._setup_memory_idempotency(ttl_seconds)
            self._record_redis_metrics(success=False, fallback=True)

    def _setup_memory_idempotency(self, ttl_seconds: int) -> None:
        """Set up memory-based idempotency cache."""
        self._idempotency_cache = IdempotencyCache(ttl_seconds=ttl_seconds)

    def _record_redis_metrics(self, success: bool, fallback: bool = False) -> None:
        """Record Redis backend metrics."""
        try:
            redis_up_metric = self.metrics_collector.redis_backend_up
            fallback_metric = self.metrics_collector.redis_backend_fallback_total
            redis_up_metric.labels(component="idempotency").set(1 if success else 0)
            if fallback:
                fallback_metric.labels(component="idempotency").inc()
        except (AttributeError, RuntimeError):
            self.metrics_collector.set_gauge(
                "redis_idempotency_up", 1 if success else 0
            )
            if fallback:
                self.metrics_collector.increment_counter(
                    "redis_idempotency_fallback_total"
                )

    def _configure_services(self) -> None:
        """Configure business logic services."""
        self.service = MappingService(self)
        register_routes(self)
        
        # Add demo endpoints for investor demonstrations
        self.app.include_router(demo_router)

    def _configure_middleware(self) -> None:
        """Configure additional middleware."""
        # Import inside method since rate limiting is optional
        from ..rate_limit import RateLimitMiddleware

        self.app.add_middleware(
            RateLimitMiddleware,
            config_manager=self.config_manager,
            metrics_collector=self.metrics_collector,
        )

        @self.app.get("/openapi.yaml", include_in_schema=False)
        async def openapi_yaml() -> Response:
            """Serve OpenAPI specification in YAML format."""
            # Import inside function since yaml is optional for OpenAPI generation
            import yaml  # type: ignore

            yaml_text = yaml.safe_dump(self.app.openapi(), sort_keys=False)
            return Response(content=yaml_text, media_type="application/yaml")

    def __init__(
        self,
        model_server: ModelServer,
        json_validator: JSONValidator,
        fallback_mapper: FallbackMapper,
        config_manager: ConfigManager,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Initialize the Mapper API application.

        Args:
            model_server: Model server for inference operations
            json_validator: JSON schema validator
            fallback_mapper: Fallback mapping service
            config_manager: Configuration manager
            metrics_collector: Metrics collection service
        """
        self.model_server = model_server
        self.json_validator = json_validator
        self.fallback_mapper = fallback_mapper
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector

        # Initialize core services
        self.version_manager = VersionManager()
        self.audit_trail_manager = AuditTrailManager()
        self.storage_manager = self._initialize_storage_manager()
        self._idempotency_cache: Any = None  # Will be set in _configure_idempotency

        # Configure application
        self._configure_app()
        self._configure_auth()
        self._configure_idempotency()
        self._configure_services()
        self._configure_middleware()

    @property
    def api_key_header_for_docs(self) -> APIKeyHeader:
        """Get API key header configuration for documentation."""
        return self._api_key_header_for_docs

    @property
    def auth_dependency(self) -> Callable[..., Any]:
        """Get authentication dependency for mapping operations."""
        return self._auth_map_write

    @property
    def idempotency_cache(self) -> IdempotencyCache:
        """Get idempotency cache instance."""
        return self._idempotency_cache


def create_app(
    model_server: ModelServer,
    json_validator: JSONValidator,
    fallback_mapper: FallbackMapper,
    config_manager: ConfigManager,
    metrics_collector: MetricsCollector,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    This factory function requires many parameters (6/5) as it follows the
    application factory pattern where all dependencies are injected explicitly
    for better testability and configuration control.

    Args:
        model_server: Model server for inference operations
        json_validator: JSON schema validator
        fallback_mapper: Fallback mapping service
        config_manager: Configuration manager
        metrics_collector: Metrics collection service

    Returns:
        Configured FastAPI application instance
    """
    mapper = MapperAPI(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics_collector,
    )
    return mapper.app
