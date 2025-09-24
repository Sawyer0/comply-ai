"""
Configuration settings for the Analysis Module.

This module contains configuration classes and settings management
for the analysis module components.
"""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class AnalysisSettings(BaseSettings):
    """
    Analysis module specific settings.

    Extends the base application settings with analysis-specific
    configuration options.
    """

    # Model configuration
    analysis_model_path: str = Field(
        default="models/phi3-mini-3.8b", description="Path to the Phi-3 Mini model"
    )
    analysis_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for deterministic output",
    )
    analysis_confidence_cutoff: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Confidence threshold for fallback"
    )

    # Processing configuration
    max_concurrent_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent analysis requests"
    )
    request_timeout_seconds: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )
    batch_size_limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum batch size for batch requests"
    )

    # Cache configuration
    idempotency_cache_ttl_hours: int = Field(
        default=24, ge=1, le=168, description="Idempotency cache TTL in hours"
    )
    cache_cleanup_interval_minutes: int = Field(
        default=60, ge=1, le=1440, description="Cache cleanup interval in minutes"
    )

    # Quality evaluation configuration
    golden_dataset_path: Optional[str] = Field(
        default=None, description="Path to golden dataset for quality evaluation"
    )
    quality_evaluation_enabled: bool = Field(
        default=True, description="Enable quality evaluation"
    )
    drift_detection_enabled: bool = Field(
        default=True, description="Enable drift detection"
    )

    # OPA configuration
    opa_binary_path: Optional[str] = Field(
        default=None, description="Path to OPA binary (defaults to 'opa' in PATH)"
    )
    opa_validation_enabled: bool = Field(
        default=True, description="Enable OPA policy validation"
    )

    # Security configuration
    pii_redaction_enabled: bool = Field(
        default=True, description="Enable PII redaction"
    )
    security_headers_enabled: bool = Field(
        default=True, description="Enable security headers"
    )

    # Retry and Circuit Breaker configuration
    retry_max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum number of retry attempts"
    )
    retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay in seconds for retry attempts",
    )
    retry_max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds for retry attempts",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=5.0,
        description="Exponential backoff base for retry delays",
    )
    retry_jitter_enabled: bool = Field(
        default=True, description="Enable jitter in retry delays"
    )
    retry_jitter_range: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Jitter range for retry delays (0.0-1.0)",
    )

    # Circuit breaker configuration
    circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breaker for model server calls"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1, le=20, description="Number of failures before opening circuit"
    )
    circuit_breaker_recovery_timeout: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Time in seconds before attempting recovery",
    )
    circuit_breaker_success_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Successes needed to close circuit from half-open",
    )
    circuit_breaker_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout in seconds for individual calls",
    )

    # Monitoring configuration
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    detailed_logging: bool = Field(default=False, description="Enable detailed logging")

    # Schema configuration
    schema_validation_enabled: bool = Field(
        default=True, description="Enable schema validation"
    )
    schema_fallback_enabled: bool = Field(
        default=True, description="Enable schema validation fallback"
    )

    # CORS configuration
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "OPTIONS"], description="Allowed HTTP methods for CORS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"], description="Allowed headers for CORS"
    )

    # Rate limiting configuration
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Rate limit requests per minute per client",
    )
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")

    model_config = ConfigDict(env_prefix="ANALYSIS_", case_sensitive=False)


class AnalysisConfig:
    """
    Analysis module configuration container.

    Provides a centralized configuration container that combines
    application settings with analysis-specific settings.
    """

    def __init__(
        self,
        analysis_settings: Optional[AnalysisSettings] = None,
        base_settings: Optional[Any] = None,
    ):
        """
        Initialize the analysis configuration.

        Args:
            analysis_settings: Analysis-specific settings
            base_settings: Base application settings
        """
        self.analysis_settings = analysis_settings or AnalysisSettings()
        self.base_settings = base_settings

    @property
    def model_path(self) -> str:
        """Get model path."""
        return self.analysis_settings.analysis_model_path

    @property
    def temperature(self) -> float:
        """Get model temperature."""
        return self.analysis_settings.analysis_temperature

    @property
    def confidence_cutoff(self) -> float:
        """Get confidence cutoff."""
        return self.analysis_settings.analysis_confidence_cutoff

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent requests."""
        return self.analysis_settings.max_concurrent_requests

    @property
    def request_timeout(self) -> int:
        """Get request timeout."""
        return self.analysis_settings.request_timeout_seconds

    @property
    def batch_size_limit(self) -> int:
        """Get batch size limit."""
        return self.analysis_settings.batch_size_limit

    @property
    def cache_ttl_hours(self) -> int:
        """Get cache TTL in hours."""
        return self.analysis_settings.idempotency_cache_ttl_hours

    @property
    def golden_dataset_path(self) -> Optional[str]:
        """Get golden dataset path."""
        return self.analysis_settings.golden_dataset_path

    @property
    def opa_binary_path(self) -> Optional[str]:
        """Get OPA binary path."""
        return self.analysis_settings.opa_binary_path

    # Additional properties for CLI compatibility
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration as dict."""
        return {
            "model_name": self.analysis_settings.analysis_model_path,
            "temperature": self.analysis_settings.analysis_temperature,
            "confidence_threshold": self.analysis_settings.analysis_confidence_cutoff,
        }

    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration as dict."""
        return {
            "host": (
                getattr(self.base_settings, "host", "localhost")
                if self.base_settings
                else "localhost"
            ),
            "max_concurrent_requests": self.analysis_settings.max_concurrent_requests,
            "timeout_seconds": self.analysis_settings.request_timeout_seconds,
        }

    @property
    def quality(self) -> Dict[str, Any]:
        """Get quality configuration as dict."""
        return {
            "confidence_threshold": self.analysis_settings.analysis_confidence_cutoff,
            "evaluation_enabled": self.analysis_settings.quality_evaluation_enabled,
        }

    @property
    def is_quality_evaluation_enabled(self) -> bool:
        """Check if quality evaluation is enabled."""
        return self.analysis_settings.quality_evaluation_enabled

    @property
    def is_drift_detection_enabled(self) -> bool:
        """Check if drift detection is enabled."""
        return self.analysis_settings.drift_detection_enabled

    @property
    def is_opa_validation_enabled(self) -> bool:
        """Check if OPA validation is enabled."""
        return self.analysis_settings.opa_validation_enabled

    @property
    def is_pii_redaction_enabled(self) -> bool:
        """Check if PII redaction is enabled."""
        return self.analysis_settings.pii_redaction_enabled

    @property
    def is_security_headers_enabled(self) -> bool:
        """Check if security headers are enabled."""
        return self.analysis_settings.security_headers_enabled

    @property
    def is_metrics_enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self.analysis_settings.metrics_enabled

    @property
    def is_detailed_logging_enabled(self) -> bool:
        """Check if detailed logging is enabled."""
        return self.analysis_settings.detailed_logging

    @property
    def is_schema_validation_enabled(self) -> bool:
        """Check if schema validation is enabled."""
        return self.analysis_settings.schema_validation_enabled

    @property
    def is_schema_fallback_enabled(self) -> bool:
        """Check if schema fallback is enabled."""
        return self.analysis_settings.schema_fallback_enabled

    @property
    def analysis_model_path(self) -> str:
        """Get analysis model path."""
        return self.analysis_settings.analysis_model_path

    @property
    def analysis_temperature(self) -> float:
        """Get analysis temperature."""
        return self.analysis_settings.analysis_temperature

    @property
    def analysis_confidence_cutoff(self) -> float:
        """Get analysis confidence cutoff."""
        return self.analysis_settings.analysis_confidence_cutoff

    @property
    def max_concurrent_requests(self) -> int:
        """Get max concurrent requests."""
        return self.analysis_settings.max_concurrent_requests

    @property
    def request_timeout_seconds(self) -> int:
        """Get request timeout seconds."""
        return self.analysis_settings.request_timeout_seconds

    @property
    def cache_max_items(self) -> int:
        """Get cache max items."""
        return 1000  # Default value

    @property
    def idempotency_cache_ttl_hours(self) -> int:
        """Get idempotency cache TTL hours."""
        return self.analysis_settings.idempotency_cache_ttl_hours

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins from base settings."""
        if self.base_settings and hasattr(self.base_settings, "cors_origins"):
            return self.base_settings.cors_origins
        return ["*"]

    def get_database_url(self) -> Optional[str]:
        """Get database URL from base settings."""
        if self.base_settings and hasattr(self.base_settings, "database_url"):
            return self.base_settings.database_url
        return None

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL from base settings."""
        if self.base_settings and hasattr(self.base_settings, "redis_url"):
            return self.base_settings.redis_url
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "analysis_settings": self.analysis_settings.dict(),
            "base_settings": self.base_settings.dict() if self.base_settings else None,
        }

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        """Create configuration from environment variables."""
        analysis_settings = AnalysisSettings()
        return cls(analysis_settings=analysis_settings)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """Create configuration from dictionary."""
        analysis_settings = AnalysisSettings(**config_dict.get("analysis_settings", {}))
        return cls(analysis_settings=analysis_settings)

    @classmethod
    def from_config_manager(cls, config_manager) -> "AnalysisConfig":
        """Create configuration from ConfigManager."""
        # Extract analysis-specific settings from the main config
        config_dict = config_manager.get_config_dict()

        # Map main config settings to analysis settings
        analysis_settings_dict = {}

        # Model settings
        if "model" in config_dict:
            model_config = config_dict["model"]
            analysis_settings_dict["analysis_model_path"] = model_config.get(
                "name", "models/phi3-mini-3.8b"
            )
            analysis_settings_dict["analysis_temperature"] = model_config.get(
                "temperature", 0.1
            )
            analysis_settings_dict["analysis_confidence_cutoff"] = model_config.get(
                "confidence_threshold", 0.3
            )

        # API settings
        if "api" in config_dict:
            api_config = config_dict["api"]
            analysis_settings_dict["max_concurrent_requests"] = api_config.get(
                "max_concurrent_requests", 10
            )
            analysis_settings_dict["request_timeout_seconds"] = api_config.get(
                "timeout_seconds", 30
            )

        # Quality settings
        if "quality" in config_dict:
            quality_config = config_dict["quality"]
            analysis_settings_dict["quality_evaluation_enabled"] = quality_config.get(
                "evaluation_enabled", True
            )

        # Create analysis settings
        analysis_settings = AnalysisSettings(**analysis_settings_dict)

        # Use the config manager's settings as base settings
        return cls(analysis_settings=analysis_settings, base_settings=config_manager)
