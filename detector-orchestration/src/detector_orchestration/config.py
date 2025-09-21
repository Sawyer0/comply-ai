"""Configuration models for detector orchestration service.

This module defines all configuration schemas used by the orchestrator,
including SLA settings, rate limiting, and service discovery parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pathlib import Path

import yaml

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SLAConfig(BaseModel):
    """SLA configuration for request processing."""
    sync_request_sla_ms: int = 2000
    async_request_sla_ms: int = 30000
    mapper_timeout_budget_ms: int = 500
    sync_to_async_threshold_ms: int = 1500


class OrchestrationConfig(BaseModel):
    """Main configuration for detector orchestration."""
    max_concurrent_detectors: int = 10
    default_timeout_ms: int = 5000
    max_retries: int = 2
    sla: SLAConfig = SLAConfig()
    health_check_interval_seconds: int = 30
    unhealthy_threshold: int = 3
    response_cache_ttl_seconds: int = 300
    cache_enabled: bool = True
    cache_backend: str = "memory"  # memory | redis
    redis_url: str | None = None
    redis_prefix: str = "orch:"
    max_content_length: int = 50000
    # Secondary routing toggles
    secondary_on_coverage_below: bool = True
    secondary_min_coverage: float = 1.0
    # Retry policy toggles
    retry_on_timeouts: bool = True
    retry_on_failures: bool = True
    auto_map_results: bool = True
    mapper_endpoint: str = "http://localhost:8000/map"
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: int = 60
    # Policy/OPA
    policy_dir: str = "policies"
    opa_enabled: bool = False
    opa_url: str | None = None
    api_key_header: str = "X-API-Key"
    tenant_header: str = "X-Tenant-ID"
    # Simple rate limiting for external API
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_tenant_limit: int = 120
    # Optional per-tenant overrides for rate limiting (tenant_id -> limit per window)
    rate_limit_tenant_overrides: Dict[str, int] = {}
    # Incident / notification settings
    incident_notification_webhooks: List[str] = Field(
        default_factory=list,
        description="Webhook endpoints to notify on incidents",
    )
    event_history_limit: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Number of incident events to retain for dashboard queries",
    )


class DetectorEndpoint(BaseModel):
    """Configuration for a single detector endpoint."""
    name: str
    endpoint: str  # 'http://...' or 'builtin:toxicity' etc.
    timeout_ms: int = 3000
    max_retries: int = 1
    auth: Dict[str, Any] = {}
    weight: float = 1.0
    supported_content_types: List[str] = ["text"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_prefix="ORCH_", env_nested_delimiter="__")

    # Core
    environment: str = "dev"
    log_level: str = "INFO"
    config: OrchestrationConfig = OrchestrationConfig()
    api_keys: Dict[str, List[str]] = {}  # key -> scopes
    mapper_api_key: Optional[str] = None

    # Detectors registry (simple inline for now)
    detectors: Dict[str, DetectorEndpoint] = {
        "toxicity": DetectorEndpoint(name="toxicity", endpoint="builtin:toxicity"),
        # Alias for tests/compat with external naming
        "deberta-toxicity": DetectorEndpoint(
            name="deberta-toxicity", endpoint="builtin:toxicity"
        ),
        "regex-pii": DetectorEndpoint(
            name="regex-pii", endpoint="builtin:regex-pii", timeout_ms=1000
        ),
        "echo": DetectorEndpoint(name="echo", endpoint="builtin:echo", timeout_ms=500),
    }

    # Policy defaults
    required_detectors_default: List[str] = ["toxicity", "regex-pii"]

    def __init__(self, _env_file: Optional[str] = None, **values: Any) -> None:
        file_values: Dict[str, Any] = {}
        if _env_file:
            cfg_path = Path(_env_file)
            if cfg_path.exists():
                loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    file_values = loaded
        merged = {**file_values, **values}
        super().__init__(**merged)
