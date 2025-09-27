"""
Configuration settings for the Mapper Service.

Single responsibility: Configuration management and validation.
"""

import os
from typing import Any, Dict, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class MapperSettings(BaseSettings):
    """
    Configuration settings for the Mapper Service.

    Single responsibility: Settings management only.
    """

    # Model configuration
    model_backend: str = Field(
        default="cpu", description="Model backend: vllm, tgi, or cpu"
    )
    model_path: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct", description="Path to model"
    )
    model_version: str = Field(default="1.0.0", description="Model version")

    # Generation configuration
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Generation temperature"
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    max_new_tokens: int = Field(
        default=200, ge=1, le=2048, description="Maximum new tokens"
    )

    # Taxonomy configuration
    taxonomy_version: str = Field(default="1.0.0", description="Taxonomy version")
    detector_configs_path: str = Field(
        default="config/detectors", description="Path to detector configs"
    )
    frameworks_path: str = Field(
        default="config/frameworks", description="Path to framework configs"
    )

    # Validation configuration
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence threshold"
    )
    enable_fallback: bool = Field(default=True, description="Enable fallback mapping")

    # Backend-specific configuration
    backend_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific arguments"
    )

    # Service configuration
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8002, ge=1, le=65535, description="Service port")
    workers: int = Field(default=1, ge=1, description="Number of workers")

    # CORS configuration
    allowed_origins: list[str] = Field(
        default=[
            "http://localhost:3000",  # Development frontend
            "http://localhost:8080",  # Development dashboard
            "https://app.comply-ai.com",  # Production frontend
            "https://dashboard.comply-ai.com",  # Production dashboard
        ],
        description="Allowed CORS origins",
    )

    # Database configuration (handled by environment variables in database_manager)
    database_url: Optional[str] = Field(
        default=None, description="Database URL (overrides individual settings)"
    )

    # Redis configuration
    redis_config: Optional[Dict[str, str]] = Field(
        default=None, description="Redis configuration for caching and rate limiting"
    )

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        """Pydantic configuration for MapperSettings."""

        env_prefix = "MAPPER_"
        case_sensitive = False

        @staticmethod
        def get_env_prefix() -> str:
            """Get environment variable prefix."""
            return "MAPPER_"

        @staticmethod
        def is_case_sensitive() -> bool:
            """Check if environment variables are case sensitive."""
            return False

    @validator("model_backend")
    @classmethod
    def validate_backend(cls, v):
        """Validate model backend selection."""
        valid_backends = ["vllm", "tgi", "cpu"]
        if v.lower() not in valid_backends:
            raise ValueError(f"Backend must be one of: {valid_backends}")
        return v.lower()

    @validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate logging level selection."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    def get_backend_kwargs(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        # Handle FieldInfo by getting the actual value
        backend_kwargs_value = (
            self.backend_kwargs if isinstance(self.backend_kwargs, dict) else {}
        )
        base_kwargs = backend_kwargs_value.copy()

        if self.model_backend == "vllm":
            base_kwargs.setdefault("tensor_parallel_size", 1)
            base_kwargs.setdefault("gpu_memory_utilization", 0.9)
        elif self.model_backend == "tgi":
            base_kwargs.setdefault("tgi_endpoint", "http://localhost:3000")
        elif self.model_backend == "cpu":
            base_kwargs.setdefault("device", "cpu")

        return base_kwargs

    def get_environment_variable(
        self, key: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Get environment variable value."""
        return os.getenv(key, default)
