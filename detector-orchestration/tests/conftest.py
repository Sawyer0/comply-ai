from __future__ import annotations

import json
from pathlib import Path

import sys

import pytest

# Ensure the detector_orchestration package is importable during tests
ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from detector_orchestration.config import OrchestrationConfig, Settings
from detector_orchestration.models import ContentType, Priority


@pytest.fixture(scope="session")
def tests_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture
def sample_config() -> OrchestrationConfig:
    """Provide a sample orchestration config for testing."""
    return OrchestrationConfig(
        max_concurrent_detectors=5,
        default_timeout_ms=3000,
        max_retries=1,
        response_cache_ttl_seconds=300,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout_seconds=30,
    )


@pytest.fixture
def sample_settings(sample_config: OrchestrationConfig) -> Settings:
    """Provide sample settings for testing."""
    return Settings(config=sample_config)


@pytest.fixture
def sample_request_data() -> dict:
    """Provide sample request data for testing."""
    return {
        "content": "This is a test content for analysis",
        "content_type": ContentType.TEXT,
        "tenant_id": "test-tenant",
        "policy_bundle": "default",
        "priority": Priority.NORMAL,
    }


@pytest.fixture(scope="session")
def test_fixtures(tests_root: Path) -> dict:
    """Load test fixtures from JSON files."""
    fixtures = {}
    fixtures_dir = tests_root / "fixtures"

    if fixtures_dir.exists():
        for fixture_file in fixtures_dir.glob("*.json"):
            with open(fixture_file, "r", encoding="utf-8") as f:
                fixtures[fixture_file.stem] = json.load(f)

    return fixtures
