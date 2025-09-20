"""
Integration tests for rate limiting headers and 429 behavior.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.config.manager import RateLimitConfig, RateLimitEndpointConfig
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving import FallbackMapper, JSONValidator


@pytest.fixture
def mock_model_server():
    server = Mock()
    # Return valid JSON that matches schema structure expected by JSONValidator.parse_output
    server.generate_mapping = AsyncMock(
        return_value='{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.95}, "confidence": 0.95}'
    )
    server.health_check = AsyncMock(return_value=True)
    return server


@pytest.fixture
def mock_json_validator():
    validator = Mock(spec=JSONValidator)
    validator.validate.return_value = (True, None)

    class _Parsed:
        taxonomy = ["HARM.SPEECH.Toxicity"]
        scores = {"HARM.SPEECH.Toxicity": 0.95}
        confidence = 0.95
        provenance = None
        notes = None

    validator.parse_output.return_value = _Parsed()
    return validator


@pytest.fixture
def mock_fallback_mapper():
    mapper = Mock(spec=FallbackMapper)
    return mapper


@pytest.fixture
def rate_limited_config_manager():
    # Minimal mock ConfigManager with the attributes used by the app and middleware
    cfg = Mock()
    # Security headers
    sec = Mock()
    sec.api_key_header = "X-API-Key"
    sec.tenant_header = "X-Tenant-ID"
    cfg.security = sec
    # Auth disabled to avoid 401s in tests
    auth = Mock()
    auth.enabled = False
    cfg.auth = auth
    # Confidence config for mapper internals
    conf = Mock()
    conf.threshold = 0.6
    cfg.confidence = conf
    # Rate limit config: 2 requests per window for API key
    rl = RateLimitConfig(
        enabled=True,
        window_seconds=60,
        endpoints={
            "map": RateLimitEndpointConfig(api_key_limit=2, tenant_limit=2, ip_limit=2),
            "map_batch": RateLimitEndpointConfig(
                api_key_limit=2, tenant_limit=2, ip_limit=2
            ),
        },
    )
    cfg.rate_limit = rl
    return cfg


@pytest.fixture
def test_app(
    mock_model_server,
    mock_json_validator,
    mock_fallback_mapper,
    rate_limited_config_manager,
):
    metrics = MetricsCollector()
    app = create_app(
        model_server=mock_model_server,
        json_validator=mock_json_validator,
        fallback_mapper=mock_fallback_mapper,
        config_manager=rate_limited_config_manager,
        metrics_collector=metrics,
    )
    return TestClient(app)


def test_rate_limit_headers_and_429(test_app):
    headers = {"X-API-Key": "test-key"}
    payload = {"detector": "deberta-toxicity", "output": "toxic"}

    # First two requests should pass
    r1 = test_app.post("/map", json=payload, headers=headers)
    assert r1.status_code == 200
    assert "RateLimit-Limit" in r1.headers
    assert "RateLimit-Remaining" in r1.headers

    r2 = test_app.post("/map", json=payload, headers=headers)
    assert r2.status_code == 200
    assert "RateLimit-Remaining" in r2.headers

    # Third request should be rate-limited
    r3 = test_app.post("/map", json=payload, headers=headers)
    assert r3.status_code == 429
    assert r3.json()["detail"] == "Rate limit exceeded"
    assert r3.headers.get("RateLimit-Remaining") == "0"
    assert "Retry-After" in r3.headers


def test_rate_limit_headers_and_429_batch(test_app):
    headers = {"X-API-Key": "test-key"}
    batch_payload = {
        "requests": [
            {"detector": "deberta-toxicity", "output": "toxic"},
            {"detector": "regex-pii", "output": "email"},
        ]
    }

    # First two requests should pass
    r1 = test_app.post("/map/batch", json=batch_payload, headers=headers)
    assert r1.status_code == 200
    assert "RateLimit-Limit" in r1.headers
    assert "RateLimit-Remaining" in r1.headers
    assert "RateLimit-Reset" in r1.headers
    assert "X-RateLimit-Limit" in r1.headers
    assert "X-RateLimit-Remaining" in r1.headers

    r2 = test_app.post("/map/batch", json=batch_payload, headers=headers)
    assert r2.status_code == 200
    assert "RateLimit-Remaining" in r2.headers

    # Third request should be rate-limited
    r3 = test_app.post("/map/batch", json=batch_payload, headers=headers)
    assert r3.status_code == 429
    body = r3.json()
    assert body["detail"] == "Rate limit exceeded"
    assert r3.headers.get("RateLimit-Remaining") == "0"
    assert "Retry-After" in r3.headers
