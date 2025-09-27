"""
Configuration settings for Analysis Service.

This module provides configuration management for the Analysis Service.
Single Responsibility: Manage service configuration only.
"""

from typing import Optional, List, Dict, Any

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = Field(
        default="postgresql://analysis:password@localhost:5432/analysis_db",
        env="ANALYSIS_DATABASE_URL",
    )
    pool_min_size: int = Field(default=5, env="ANALYSIS_DB_POOL_MIN")
    pool_max_size: int = Field(default=20, env="ANALYSIS_DB_POOL_MAX")

    class Config:
        """Pydantic configuration for DatabaseSettings."""

        env_prefix = "ANALYSIS_DB_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_DB_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    url: str = Field(default="redis://localhost:6379/0", env="ANALYSIS_REDIS_URL")
    password: Optional[str] = Field(default=None, env="ANALYSIS_REDIS_PASSWORD")

    class Config:
        """Pydantic configuration for RedisSettings."""

        env_prefix = "ANALYSIS_REDIS_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_REDIS_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class PluginSettings(BaseSettings):
    """Plugin system configuration settings."""

    directories: List[str] = Field(
        default_factory=lambda: ["./plugins", "/opt/analysis/plugins"],
        env="ANALYSIS_PLUGIN_DIRECTORIES",
    )
    auto_discover: bool = Field(default=True, env="ANALYSIS_PLUGIN_AUTO_DISCOVER")
    max_execution_time_seconds: int = Field(
        default=300, env="ANALYSIS_PLUGIN_MAX_EXECUTION_TIME"
    )

    class Config:
        """Pydantic configuration for PluginSettings."""

        env_prefix = "ANALYSIS_PLUGIN_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_PLUGIN_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class TenancySettings(BaseSettings):
    """Multi-tenancy configuration settings."""

    default_quota_requests: int = Field(
        default=1000, env="ANALYSIS_DEFAULT_QUOTA_REQUESTS"
    )
    default_quota_batch_requests: int = Field(
        default=100, env="ANALYSIS_DEFAULT_QUOTA_BATCH_REQUESTS"
    )
    default_quota_storage_mb: int = Field(
        default=1024, env="ANALYSIS_DEFAULT_QUOTA_STORAGE_MB"
    )
    default_quota_cpu_minutes: int = Field(
        default=60, env="ANALYSIS_DEFAULT_QUOTA_CPU_MINUTES"
    )
    default_quota_ml_calls: int = Field(
        default=500, env="ANALYSIS_DEFAULT_QUOTA_ML_CALLS"
    )

    class Config:
        """Pydantic configuration for TenancySettings."""

        env_prefix = "ANALYSIS_TENANCY_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_TENANCY_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class AnalyticsSettings(BaseSettings):
    """Analytics configuration settings."""

    retention_days: int = Field(default=365, env="ANALYSIS_ANALYTICS_RETENTION_DAYS")
    aggregation_enabled: bool = Field(
        default=True, env="ANALYSIS_ANALYTICS_AGGREGATION_ENABLED"
    )
    weekly_aggregation_day: int = Field(
        default=0, env="ANALYSIS_ANALYTICS_WEEKLY_DAY"
    )  # 0 = Monday

    class Config:
        """Pydantic configuration for AnalyticsSettings."""

        env_prefix = "ANALYSIS_ANALYTICS_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_ANALYTICS_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class ModelSettings(BaseSettings):
    """Model configuration settings."""

    model_backend: str = Field(default="cpu", env="ANALYSIS_MODEL_BACKEND")
    model_path: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct", env="ANALYSIS_MODEL_PATH"
    )
    confidence_threshold: float = Field(
        default=0.7, env="ANALYSIS_CONFIDENCE_THRESHOLD"
    )
    enable_metrics: bool = Field(default=True, env="ANALYSIS_ENABLE_METRICS")

    # Generation configuration
    temperature: float = Field(default=0.1, env="ANALYSIS_MODEL_TEMPERATURE")
    top_p: float = Field(default=0.9, env="ANALYSIS_MODEL_TOP_P")
    max_new_tokens: int = Field(default=200, env="ANALYSIS_MODEL_MAX_TOKENS")

    class Config:
        """Pydantic configuration for ModelSettings."""

        env_prefix = "ANALYSIS_MODEL_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_MODEL_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class CORSSettings(BaseSettings):
    """CORS configuration settings for secure cross-origin requests."""

    # NEVER use ["*"] in production - this is a critical security vulnerability
    origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",  # Development frontend
            "http://localhost:8080",  # Development dashboard
            "https://app.comply-ai.com",  # Production frontend
            "https://dashboard.comply-ai.com",  # Production dashboard
        ],
        env="ANALYSIS_CORS_ORIGINS",
    )
    allow_credentials: bool = Field(default=True, env="ANALYSIS_CORS_ALLOW_CREDENTIALS")
    methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="ANALYSIS_CORS_METHODS",
    )
    headers: List[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Tenant-ID",
            "X-Correlation-ID",
            "X-Request-ID",
        ],
        env="ANALYSIS_CORS_HEADERS",
    )
    max_age: int = Field(default=3600, env="ANALYSIS_CORS_MAX_AGE")  # 1 hour

    class Config:
        """Pydantic configuration for CORSSettings."""

        env_prefix = "ANALYSIS_CORS_"

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_CORS_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


class Settings(BaseSettings):
    """Main settings for Analysis Service."""

    # Service settings
    service_name: str = Field(default="analysis", env="ANALYSIS_SERVICE_NAME")
    debug: bool = Field(default=False, env="ANALYSIS_DEBUG")
    log_level: str = Field(default="INFO", env="ANALYSIS_LOG_LEVEL")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    tenancy: TenancySettings = Field(default_factory=TenancySettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)

    @property
    def generation_config(self) -> Dict[str, Any]:
        """Get generation configuration for model server."""
        model_settings = (
            self.model if hasattr(self.model, "temperature") else ModelSettings()
        )
        return {
            "temperature": model_settings.temperature,
            "top_p": model_settings.top_p,
            "max_new_tokens": model_settings.max_new_tokens,
        }

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS allowed origins."""
        cors_settings = self.cors if hasattr(self.cors, "origins") else CORSSettings()
        return cors_settings.origins

    @property
    def cors_allow_credentials(self) -> bool:
        """Get CORS allow credentials setting."""
        cors_settings = (
            self.cors if hasattr(self.cors, "allow_credentials") else CORSSettings()
        )
        return cors_settings.allow_credentials

    @property
    def cors_allow_methods(self) -> List[str]:
        """Get CORS allowed methods."""
        cors_settings = self.cors if hasattr(self.cors, "methods") else CORSSettings()
        return cors_settings.methods

    @property
    def cors_allow_headers(self) -> List[str]:
        """Get CORS allowed headers."""
        cors_settings = self.cors if hasattr(self.cors, "headers") else CORSSettings()
        return cors_settings.headers

    class Config:
        """Pydantic configuration for main Settings."""

        env_prefix = "ANALYSIS_"
        case_sensitive = False

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "ANALYSIS_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False


# Global settings instance
settings = Settings()
