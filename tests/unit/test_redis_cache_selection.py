from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.llama_mapper.api.auth import IdempotencyCache
from src.llama_mapper.api.mapper import MapperAPI
from src.llama_mapper.api.rate_limit import (
    MemoryRateLimiterBackend,
    RateLimitMiddleware,
)
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector


class _DummyModelServer:
    pass


class _DummyJSONValidator:
    pass


class _DummyFallback:
    pass


def _dummy_asgi_app():
    async def app(scope, receive, send):
        return None

    return app


def test_mapper_idempotency_falls_back_when_redis_unavailable():
    # No redis library installed => backend should fall back
    cfg = ConfigManager()
    cfg.auth.idempotency_backend = "redis"  # type: ignore[attr-defined]
    cfg.auth.idempotency_redis_url = "redis://localhost:6379/0"  # type: ignore[attr-defined]

    metrics = MetricsCollector(enable_prometheus=False)
    api = MapperAPI(
        model_server=_DummyModelServer(),
        json_validator=_DummyJSONValidator(),
        fallback_mapper=_DummyFallback(),
        config_manager=cfg,
        metrics_collector=metrics,
    )
    # Access internal cache type; should be in-memory fallback
    assert isinstance(api._idempotency_cache, IdempotencyCache)  # type: ignore[attr-defined]


def test_rate_limit_backend_falls_back_to_memory_when_redis_unhealthy():
    cfg = ConfigManager()
    cfg.rate_limit.backend = "redis"  # type: ignore[attr-defined]
    cfg.auth.idempotency_redis_url = "redis://localhost:6379/0"  # type: ignore[attr-defined]
    metrics = MetricsCollector(enable_prometheus=False)

    m = RateLimitMiddleware(
        _dummy_asgi_app(), config_manager=cfg, metrics_collector=metrics
    )
    assert isinstance(m.backend, MemoryRateLimiterBackend)
