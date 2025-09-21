"""FastAPI entrypoint for the detector orchestration service."""

from __future__ import annotations

from ..config import Settings
from ..service_factory import OrchestrationAppBuilder

# Expose a shared settings instance for tests to mutate at runtime
settings = Settings()

# Build the application via the service factory
_builder = OrchestrationAppBuilder(settings)
app = _builder.build()

# Convenience references used by integration tests
factory = _builder.factory
metrics = factory.metrics

__all__ = ["app", "settings", "factory", "metrics"]
