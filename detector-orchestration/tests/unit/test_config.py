"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from detector_orchestration.config import (
    OrchestrationConfig,
    SLAConfig,
    DetectorEndpoint,
    Settings,
)


class TestSLAConfig:
    def test_sla_config_defaults(self):
        """Test SLA configuration with default values."""
        sla = SLAConfig()

        assert sla.sync_request_sla_ms == 2000
        assert sla.async_request_sla_ms == 30000
        assert sla.mapper_timeout_budget_ms == 500
        assert sla.sync_to_async_threshold_ms == 1500

    def test_sla_config_custom_values(self):
        """Test SLA configuration with custom values."""
        sla = SLAConfig(
            sync_request_sla_ms=3000,
            async_request_sla_ms=45000,
            mapper_timeout_budget_ms=750,
            sync_to_async_threshold_ms=2000,
        )

        assert sla.sync_request_sla_ms == 3000
        assert sla.async_request_sla_ms == 45000
        assert sla.mapper_timeout_budget_ms == 750
        assert sla.sync_to_async_threshold_ms == 2000


class TestOrchestrationConfig:
    def test_orchestration_config_defaults(self):
        """Test orchestration configuration with default values."""
        config = OrchestrationConfig()

        assert config.max_concurrent_detectors == 10
        assert config.default_timeout_ms == 5000
        assert config.max_retries == 2
        assert config.health_check_interval_seconds == 30
        assert config.unhealthy_threshold == 3
        assert config.response_cache_ttl_seconds == 300
        assert config.cache_enabled is True
        assert config.cache_backend == "memory"
        assert config.redis_url is None
        assert config.redis_prefix == "orch:"
        assert config.max_content_length == 50000
        assert config.secondary_on_coverage_below is True
        assert config.secondary_min_coverage == 1.0
        assert config.retry_on_timeouts is True
        assert config.retry_on_failures is True
        assert config.auto_map_results is True
        assert config.mapper_endpoint == "http://localhost:8000/map"
        assert config.circuit_breaker_failure_threshold == 5
        assert config.circuit_breaker_recovery_timeout_seconds == 60
        assert config.policy_dir == "policies"
        assert config.opa_enabled is False
        assert config.opa_url is None
        assert config.api_key_header == "X-API-Key"
        assert config.tenant_header == "X-Tenant-ID"
        assert config.rate_limit_enabled is True
        assert config.rate_limit_window_seconds == 60
        assert config.rate_limit_tenant_limit == 120
        assert config.rate_limit_tenant_overrides == {}

    def test_orchestration_config_custom_values(self):
        """Test orchestration configuration with custom values."""
        config = OrchestrationConfig(
            max_concurrent_detectors=5,
            default_timeout_ms=3000,
            max_retries=1,
            response_cache_ttl_seconds=600,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_seconds=30,
            rate_limit_enabled=False,
            rate_limit_tenant_limit=60,
            rate_limit_tenant_overrides={"premium-tenant": 200},
        )

        assert config.max_concurrent_detectors == 5
        assert config.default_timeout_ms == 3000
        assert config.max_retries == 1
        assert config.response_cache_ttl_seconds == 600
        assert config.circuit_breaker_failure_threshold == 3
        assert config.circuit_breaker_recovery_timeout_seconds == 30
        assert config.rate_limit_enabled is False
        assert config.rate_limit_tenant_limit == 60
        assert config.rate_limit_tenant_overrides == {"premium-tenant": 200}


class TestDetectorEndpoint:
    def test_detector_endpoint_defaults(self):
        """Test detector endpoint with default values."""
        endpoint = DetectorEndpoint(name="toxicity", endpoint="builtin:toxicity")

        assert endpoint.name == "toxicity"
        assert endpoint.endpoint == "builtin:toxicity"
        assert endpoint.timeout_ms == 3000
        assert endpoint.max_retries == 1
        assert endpoint.auth == {}
        assert endpoint.weight == 1.0
        assert endpoint.supported_content_types == ["text"]

    def test_detector_endpoint_custom_values(self):
        """Test detector endpoint with custom values."""
        endpoint = DetectorEndpoint(
            name="regex-pii",
            endpoint="http://detector.example.com/pii",
            timeout_ms=2000,
            max_retries=2,
            auth={"token": "secret"},
            weight=0.8,
            supported_content_types=["text", "document"],
        )

        assert endpoint.name == "regex-pii"
        assert endpoint.endpoint == "http://detector.example.com/pii"
        assert endpoint.timeout_ms == 2000
        assert endpoint.max_retries == 2
        assert endpoint.auth == {"token": "secret"}
        assert endpoint.weight == 0.8
        assert endpoint.supported_content_types == ["text", "document"]


class TestSettings:
    def test_settings_defaults(self):
        """Test settings with default values."""
        settings = Settings()

        assert settings.environment == "dev"
        assert settings.log_level == "INFO"
        assert settings.config is not None
        assert settings.api_keys == {}
        assert settings.mapper_api_key is None
        assert "toxicity" in settings.detectors
        assert "regex-pii" in settings.detectors
        assert "echo" in settings.detectors
        assert settings.required_detectors_default == ["toxicity", "regex-pii"]

    def test_settings_custom_values(self):
        """Test settings with custom values."""
        config = OrchestrationConfig(
            max_concurrent_detectors=5,
            default_timeout_ms=3000,
            max_retries=1,
        )

        detectors = {
            "custom-detector": DetectorEndpoint(
                name="custom-detector",
                endpoint="builtin:custom",
                timeout_ms=1000,
            )
        }

        settings = Settings(
            environment="prod",
            log_level="DEBUG",
            config=config,
            api_keys={"test-key": ["read", "write"]},
            mapper_api_key="mapper-secret",
            detectors=detectors,
            required_detectors_default=["custom-detector"],
        )

        assert settings.environment == "prod"
        assert settings.log_level == "DEBUG"
        assert settings.config.max_concurrent_detectors == 5
        assert settings.config.default_timeout_ms == 3000
        assert settings.config.max_retries == 1
        assert settings.api_keys == {"test-key": ["read", "write"]}
        assert settings.mapper_api_key == "mapper-secret"
        assert settings.detectors == detectors
        assert settings.required_detectors_default == ["custom-detector"]

    def test_settings_environment_override(self):
        """Test settings with environment variable overrides."""
        import os

        # Set environment variables
        os.environ["ORCH_ENVIRONMENT"] = "test"
        os.environ["ORCH_LOG_LEVEL"] = "DEBUG"
        os.environ["ORCH_CONFIG__MAX_CONCURRENT_DETECTORS"] = "3"
        os.environ["ORCH_API_KEYS"] = '{"test-key": ["read"]}'

        try:
            settings = Settings()

            assert settings.environment == "test"
            assert settings.log_level == "DEBUG"
            assert settings.config.max_concurrent_detectors == 3
            assert settings.api_keys == {"test-key": ["read"]}
        finally:
            # Clean up environment variables
            del os.environ["ORCH_ENVIRONMENT"]
            del os.environ["ORCH_LOG_LEVEL"]
            del os.environ["ORCH_CONFIG__MAX_CONCURRENT_DETECTORS"]
            del os.environ["ORCH_API_KEYS"]

    def test_settings_yaml_config_file(self):
        """Test settings loading from YAML configuration file."""
        config_data = {
            "environment": "prod",
            "log_level": "INFO",
            "config": {
                "max_concurrent_detectors": 8,
                "default_timeout_ms": 4000,
                "max_retries": 3,
                "rate_limit_enabled": False,
            },
            "detectors": {
                "custom-toxicity": {
                    "name": "custom-toxicity",
                    "endpoint": "builtin:custom-toxicity",
                    "timeout_ms": 2500,
                    "max_retries": 2,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            settings = Settings(_env_file=config_path)

            assert settings.environment == "prod"
            assert settings.log_level == "INFO"
            assert settings.config.max_concurrent_detectors == 8
            assert settings.config.default_timeout_ms == 4000
            assert settings.config.max_retries == 3
            assert settings.config.rate_limit_enabled is False
            assert "custom-toxicity" in settings.detectors
            assert settings.detectors["custom-toxicity"].timeout_ms == 2500
            assert settings.detectors["custom-toxicity"].max_retries == 2
        finally:
            # Clean up the temporary file
            Path(config_path).unlink()
