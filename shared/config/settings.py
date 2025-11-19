"""Unified configuration models and helpers for Comply-AI services."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration with connection pooling helpers."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="complyai")
    user: str = Field(default="postgres")
    password: str = Field(default="")
    ssl_mode: str = Field(default="prefer")
    max_connections: int = Field(default=20)
    min_connections: int = Field(default=5)

    @property
    def url(self) -> str:
        """Generate a PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def pool_config(self) -> Dict[str, Any]:
        """Connection pool tuning settings."""
        return {
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
        }


class SecurityConfig(BaseModel):
    """Authentication and authorization configuration."""

    api_keys: List[str] = Field(default_factory=list)
    require_api_key: bool = Field(default=False)
    enable_tenant_isolation: bool = Field(default=True)
    jwt_secret: Optional[str] = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")
    rate_limit_per_minute: int = Field(default=600)
    enable_waf: bool = Field(default=True)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value or []


class ObservabilityConfig(BaseModel):
    """Logging, tracing, and metrics configuration."""

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    enable_tracing: bool = Field(default=False)
    tracing_endpoint: Optional[str] = Field(default=None)
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    prometheus_gateway: Optional[str] = Field(default=None)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = (value or "INFO").upper()
        if level not in valid_levels:
            raise ValueError(f"log_level must be one of {sorted(valid_levels)}")
        return level


class ServiceConfig(BaseModel):
    """Server runtime configuration."""

    name: str = Field(default="service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    reload: bool = Field(default=False)
    workers: int = Field(default=1)
    api_prefix: str = Field(default="/api/v1")

    @field_validator("environment", mode="before")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        valid_envs = {"development", "staging", "production"}
        env_value = (value or "development").lower()
        if env_value not in valid_envs:
            raise ValueError(f"environment must be one of {sorted(valid_envs)}")
        return env_value


class SharedSettings(PydanticBaseSettings):
    """Primary configuration model shared by all services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Perform cross-field validation and emit helpful warnings."""

        if self.service.environment == "production":
            if self.service.debug:
                raise ValueError("debug must be False in production")
            if not self.security.require_api_key and not self.security.jwt_secret:
                logger.warning(
                    "Production environment without authentication configured. "
                    "Set REQUIRE_API_KEY=true or provide JWT_SECRET."
                )
            if self.observability.log_level == "DEBUG":
                logger.warning(
                    "DEBUG log level detected in production; consider INFO or higher."
                )

    def get_cors_config(self) -> Dict[str, Any]:
        """Return a CORS middleware configuration derived from settings."""

        if self.service.environment == "production":
            origins = [
                "https://app.comply-ai.com",
                "https://dashboard.comply-ai.com",
            ]
        else:
            origins = [
                "http://localhost:3000",
                "http://localhost:8080",
                "https://app.comply-ai.com",
                "https://dashboard.comply-ai.com",
            ]

        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-API-Key",
                "X-Tenant-ID",
                "X-Correlation-ID",
                "X-Request-ID",
            ],
        }

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Serialize settings for diagnostics (optionally masking secrets)."""

        result = self.model_dump(mode="json")
        if mask_secrets:
            if "password" in result.get("database", {}):
                result["database"]["password"] = "***MASKED***"
            if "jwt_secret" in result.get("security", {}):
                result["security"]["jwt_secret"] = "***MASKED***"
            if result.get("security", {}).get("api_keys"):
                count = len(result["security"]["api_keys"])
                result["security"]["api_keys"] = [f"***MASKED_{i}***" for i in range(count)]
        return result


def _build_settings_kwargs(service_name: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if service_name:
        data.setdefault("service", {})
        data["service"].setdefault("name", service_name)
    if overrides:
        data = {**data, **overrides}
    return data


@lru_cache()
def get_service_settings(service_name: str, **overrides: Any) -> SharedSettings:
    """Load shared settings for a specific service with optional overrides."""

    settings_kwargs = _build_settings_kwargs(service_name, overrides or None)
    return SharedSettings(**settings_kwargs)


class OrchestrationOptions(BaseModel):
    enable_health_monitoring: bool = Field(default=True)
    enable_service_discovery: bool = Field(default=True)
    enable_policy_management: bool = Field(default=True)
    health_check_interval: int = Field(default=30)
    service_ttl_minutes: int = Field(default=5)
    cache_backend: str = Field(default="memory")
    redis_url: Optional[str] = Field(default=None)
    redis_prefix: str = Field(default="orch")


class AnalysisOptions(BaseModel):
    enable_risk_scoring: bool = Field(default=True)
    enable_compliance_intelligence: bool = Field(default=True)
    enable_rag: bool = Field(default=True)
    model_timeout_seconds: int = Field(default=30)
    max_batch_size: int = Field(default=100)


class MapperOptions(BaseModel):
    model_name: str = Field(default="llama-3-8b-instruct")
    model_endpoint: Optional[str] = Field(default=None)
    serving_timeout_ms: int = Field(default=500)
    max_payload_kb: int = Field(default=64)
    reject_on_raw_content: bool = Field(default=True)
    idempotency_backend: str = Field(default="memory")
    idempotency_redis_url: Optional[str] = Field(default=None)


def get_orchestration_settings(**overrides: Any) -> tuple[SharedSettings, OrchestrationOptions]:
    """Convenience helper for the orchestration service."""

    shared = get_service_settings("detector-orchestration", **overrides)
    options = OrchestrationOptions(**overrides.get("orchestration", {})) if overrides else OrchestrationOptions()
    return shared, options


def get_analysis_settings(**overrides: Any) -> tuple[SharedSettings, AnalysisOptions]:
    """Convenience helper for the analysis service."""

    shared = get_service_settings("analysis-service", **overrides)
    options = AnalysisOptions(**overrides.get("analysis", {})) if overrides else AnalysisOptions()
    return shared, options


def get_mapper_settings(**overrides: Any) -> tuple[SharedSettings, MapperOptions]:
    """Convenience helper for the mapper service."""

    shared = get_service_settings("mapper-service", **overrides)
    options = MapperOptions(**overrides.get("mapper", {})) if overrides else MapperOptions()
    return shared, options


__all__ = [
    "SharedSettings",
    "BaseSettings",
    "DatabaseConfig",
    "SecurityConfig",
    "ObservabilityConfig",
    "ServiceConfig",
    "get_service_settings",
    "get_orchestration_settings",
    "get_analysis_settings",
    "get_mapper_settings",
    "OrchestrationOptions",
    "AnalysisOptions",
    "MapperOptions",
]


# Backwards compatibility alias for older imports that referenced BaseSettings.
BaseSettings = SharedSettings
