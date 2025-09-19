from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.api.models import MappingResponse
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving import FallbackMapper, JSONValidator


@pytest.fixture
def test_app():
    # Mock model server (returns valid JSON)
    mock_model_server = Mock()
    mock_model_server.generate_mapping = AsyncMock(
        return_value='{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.95}, "confidence": 0.95}'
    )

    # Mock validator (accepts model output)
    mock_json_validator = Mock(spec=JSONValidator)
    mock_json_validator.validate.return_value = (True, None)
    mock_json_validator.parse_output.return_value = MappingResponse(
        taxonomy=["HARM.SPEECH.Toxicity"],
        scores={"HARM.SPEECH.Toxicity": 0.95},
        confidence=0.95,
    )

    # Mock fallback mapper
    mock_fallback = Mock(spec=FallbackMapper)

    # Mock config: leave auth disabled; confidence default
    mock_config = Mock(spec=ConfigManager)
    conf = Mock()
    conf.threshold = 0.6
    mock_config.confidence = conf

    metrics = Mock(spec=MetricsCollector)

    app = create_app(
        model_server=mock_model_server,
        json_validator=mock_json_validator,
        fallback_mapper=mock_fallback,
        config_manager=mock_config,
        metrics_collector=metrics,
    )

    return TestClient(app)


def test_map_rejects_raw_content_returns_canonical_error_body(test_app: TestClient):
    # output long free-text (>=2048 chars) triggers raw content heuristic
    long_text = "x" * 3000
    payload = {"detector": "test-detector", "output": long_text}

    resp = test_app.post("/map", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" in body
    detail = body["detail"]
    assert detail["error_code"] == "INVALID_REQUEST"
    assert detail["retryable"] is False
    assert "Raw content" in detail["message"]


def test_map_rejects_oversize_returns_canonical_error_body(test_app: TestClient):
    # Create payload > 64KB by using a large metadata blob
    big_blob = "a" * (70 * 1024)  # 70KB
    payload = {
        "detector": "test-detector",
        "output": "short",
        "metadata": {"blob": big_blob},
    }

    resp = test_app.post("/map", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    detail = body["detail"]
    assert detail["error_code"] == "INVALID_REQUEST"
    assert detail["retryable"] is False
    assert "Payload too large" in detail["message"]


def test_batch_raw_content_item_returns_canonical_error_body(test_app: TestClient):
    raw_item = {"detector": "test-detector", "output": "y" * 3000}
    payload = {"requests": [raw_item]}

    resp = test_app.post("/map/batch", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    detail = body["detail"]
    assert detail["error_code"] == "INVALID_REQUEST"
    assert detail["retryable"] is False
    assert "contains raw content" in detail["message"].lower()
