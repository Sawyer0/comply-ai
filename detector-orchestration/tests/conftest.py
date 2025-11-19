from __future__ import annotations

import json
from pathlib import Path

import sys

import pytest

# Ensure the orchestration package (new stack) and shared library are importable during tests
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "src"
MONOREPO_ROOT = Path(__file__).resolve().parents[2]

for path in (MONOREPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    # Legacy package path used by older tests
    from detector_orchestration.config import OrchestrationConfig, Settings  # type: ignore[import]
    from detector_orchestration.models import ContentType, Priority  # type: ignore[import]
except ImportError:
    # Fallback for the refactored layout using the new orchestration package
    from dataclasses import dataclass
    from enum import Enum

    from orchestration.service.models import OrchestrationConfig  # type: ignore[no-redef]

    @dataclass
    class Settings:  # type: ignore[no-redef]
        """Minimal Settings stub for tests in the new layout."""

        config: OrchestrationConfig | None = None

        def __init__(self, config: OrchestrationConfig | None = None) -> None:
            self.config = config or OrchestrationConfig()

    class ContentType(str, Enum):  # type: ignore[no-redef]
        TEXT = "text"

    class Priority(str, Enum):  # type: ignore[no-redef]
        NORMAL = "normal"


@pytest.fixture(scope="session")
def tests_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture
def sample_config() -> OrchestrationConfig:
    """Provide a sample orchestration config for testing."""
    # The new orchestration config is intentionally minimal and focused on
    # core toggles and rate limiting. Tests that rely on additional knobs
    # should use explicit settings objects rather than this fixture.
    return OrchestrationConfig()


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
