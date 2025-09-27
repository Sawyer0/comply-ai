"""Configuration management for the Detector Orchestration Service.

This module provides centralized configuration management following the
microservice architecture guidelines with environment-based overrides.
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProcessingMode(str, Enum):
    """Processing modes for orchestration."""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


class OrchestrationSettings(BaseSettings):
    """Main configuration for the Detector Orchestration Service."""

    # Service Configuration
    service_name: str = Field(default="detector-orchestration", env="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")

    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    enable_structured_logging: bool = Field(
        default=True, env="ENABLE_STRUCTURED_LOGGING"
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql://localhost:5432/orchestration", env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    database_timeout: int = Field(default=30, env="DATABASE_TIMEOUT")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")

    # Health Monitoring Configuration
    enable_health_monitoring: bool = Field(default=True, env="ENABLE_HEALTH_MONITORING")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    unhealthy_threshold: int = Field(default=3, env="UNHEALTHY_THRESHOLD")
    degraded_threshold: int = Field(default=2, env="DEGRADED_THRESHOLD")

    # Service Discovery Configuration
    enable_service_discovery: bool = Field(default=True, env="ENABLE_SERVICE_DISCOVERY")
    service_ttl_minutes: int = Field(default=30, env="SERVICE_TTL_MINUTES")
    discovery_cleanup_interval: int = Field(
        default=300, env="DISCOVERY_CLEANUP_INTERVAL"
    )

    # Policy Management Configuration
    enable_policy_management: bool = Field(default=True, env="ENABLE_POLICY_MANAGEMENT")
    opa_endpoint: Optional[str] = Field(default=None, env="OPA_ENDPOINT")
    policy_cache_ttl: int = Field(default=300, env="POLICY_CACHE_TTL")

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT"
    )

    # Rate Limiting Configuration
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    default_rate_limit: int = Field(default=100, env="DEFAULT_RATE_LIMIT")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # Security Configuration
    enable_waf: bool = Field(default=True, env="ENABLE_WAF")
    enable_input_sanitization: bool = Field(
        default=True, env="ENABLE_INPUT_SANITIZATION"
    )
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB

    # Async Job Processing Configuration
    max_concurrent_jobs: int = Field(default=10, env="MAX_CONCURRENT_JOBS")
    job_timeout: int = Field(default=300, env="JOB_TIMEOUT")
    job_cleanup_interval: int = Field(default=3600, env="JOB_CLEANUP_INTERVAL")

    # Monitoring Configuration
    enable_prometheus_metrics: bool = Field(
        default=True, env="ENABLE_PROMETHEUS_METRICS"
    )
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_distributed_tracing: bool = Field(
        default=True, env="ENABLE_DISTRIBUTED_TRACING"
    )
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080",
            "https://app.comply-ai.com",
            "https://dashboard.comply-ai.com",
        ],
        env="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # API Configuration
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    enable_api_docs: bool = Field(default=True, env="ENABLE_API_DOCS")
    api_docs_url: str = Field(default="/docs", env="API_DOCS_URL")

    # Tenant Configuration
    require_tenant_id: bool = Field(default=True, env="REQUIRE_TENANT_ID")
    default_tenant_id: Optional[str] = Field(default=None, env="DEFAULT_TENANT_ID")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        @staticmethod
        def get_env_file_path() -> str:
            """Get the environment file path."""
            return ".env"

        @staticmethod
        def get_encoding() -> str:
            """Get the environment file encoding."""
            return "utf-8"

    @validator("cors_origins", pre=True)
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v:
            raise ValueError("DATABASE_URL is required")
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return v

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_timeout,
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            "url": self.redis_url,
            "db": self.redis_db,
            "password": self.redis_password,
            "ssl": self.redis_ssl,
        }

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials,
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

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return str(self.environment).lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return str(self.environment).lower() == "development"

    def get_environment_variable(
        self, key: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Get environment variable value."""
        return os.getenv(key, default)


# Global settings instance
settings = OrchestrationSettings()
