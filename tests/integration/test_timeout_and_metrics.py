import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.api.models import MappingResponse
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving import FallbackMapper, JSONValidator


@pytest.fixture
def app_with_timeout():
    # Model server that sleeps longer than timeout
    async def _sleep_then_return(*args, **kwargs):
        await asyncio.sleep(0.05)
        return '{"taxonomy": ["OTHER.Unknown"], "scores": {"OTHER.Unknown": 0.0}, "confidence": 0.1}'

    mock_model_server = Mock()
    mock_model_server.generate_mapping = AsyncMock(side_effect=_sleep_then_return)

    mock_json_validator = Mock(spec=JSONValidator)
    mock_json_validator.validate.return_value = (True, None)
    mock_json_validator.parse_output.return_value = MappingResponse(
        taxonomy=["OTHER.Unknown"], scores={"OTHER.Unknown": 0.0}, confidence=0.1
    )

    mock_fallback = Mock(spec=FallbackMapper)

    mock_config = Mock(spec=ConfigManager)
    conf = Mock()
    conf.threshold = 0.6
    mock_config.confidence = conf
    serving = Mock()
    serving.mapper_timeout_ms = 10  # 10ms => force timeout
    serving.max_payload_kb = 64
    serving.reject_on_raw_content = True
    mock_config.serving = serving

    metrics = Mock(spec=MetricsCollector)

    app = create_app(
        model_server=mock_model_server,
        json_validator=mock_json_validator,
        fallback_mapper=mock_fallback,
        config_manager=mock_config,
        metrics_collector=metrics,
    )

    return TestClient(app), metrics


def test_timeout_returns_408_and_records_metric(app_with_timeout):
    client, metrics = app_with_timeout

    payload = {"detector": "test-detector", "output": "short"}
    resp = client.post("/map", json=payload)

    assert resp.status_code == 408
    body = resp.json()["detail"]
    assert body["error_code"] == "REQUEST_TIMEOUT"
    assert body["retryable"] is True

    # metrics assertions
    metrics.record_error.assert_any_call("REQUEST_TIMEOUT")
