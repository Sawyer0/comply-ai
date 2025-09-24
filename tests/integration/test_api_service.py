"""
Tests for the FastAPI service layer.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.api.models import MappingResponse
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving import FallbackMapper, JSONValidator


@pytest.fixture
def mock_model_server():
    """Create a mock model server."""
    server = Mock()
    server.generate_mapping = AsyncMock(
        return_value='{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.95}, "confidence": 0.95}'
    )
    server.health_check = AsyncMock(return_value=True)
    return server


@pytest.fixture
def mock_json_validator():
    """Create a mock JSON validator."""
    validator = Mock(spec=JSONValidator)
    validator.validate.return_value = (True, None)
    validator.parse_output.return_value = MappingResponse(
        taxonomy=["HARM.SPEECH.Toxicity"],
        scores={"HARM.SPEECH.Toxicity": 0.95},
        confidence=0.95,
        notes=None,
        version_info=None,
    )
    return validator


@pytest.fixture
def mock_fallback_mapper():
    """Create a mock fallback mapper."""
    mapper = Mock(spec=FallbackMapper)
    mapper.map.return_value = MappingResponse(
        taxonomy=["OTHER.Unknown"],
        scores={"OTHER.Unknown": 0.0},
        confidence=0.0,
        notes="Fallback mapping used",
        version_info=None,
    )
    return mapper


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    config = Mock(spec=ConfigManager)
    # Mock the confidence config object
    confidence_config = Mock()
    confidence_config.threshold = 0.6
    config.confidence = confidence_config
    return config


@pytest.fixture
def mock_metrics_collector():
    """Create a mock metrics collector."""
    metrics = Mock(spec=MetricsCollector)
    metrics.increment_counter = Mock()
    metrics.record_histogram = Mock()
    return metrics


@pytest.fixture
def test_app(
    mock_model_server,
    mock_json_validator,
    mock_fallback_mapper,
    mock_config_manager,
    mock_metrics_collector,
):
    """Create a test FastAPI app with mocked dependencies."""
    app = create_app(
        model_server=mock_model_server,
        json_validator=mock_json_validator,
        fallback_mapper=mock_fallback_mapper,
        config_manager=mock_config_manager,
        metrics_collector=mock_metrics_collector,
    )
    return TestClient(app)


def test_health_endpoint(test_app):
    """Test the health check endpoint."""
    response = test_app.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_map_endpoint_success(test_app, mock_model_server, mock_json_validator):
    """Test successful mapping via the /map endpoint."""
    request_data = {
        "detector": "deberta-toxicity",
        "output": "toxic",
        "metadata": {"score": 0.95},
    }

    response = test_app.post("/map", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["taxonomy"] == ["HARM.SPEECH.Toxicity"]
    assert data["scores"]["HARM.SPEECH.Toxicity"] == 0.95
    assert data["confidence"] == 0.95

    # Verify model server was called
    mock_model_server.generate_mapping.assert_called_once()
    mock_json_validator.validate.assert_called_once()


def test_map_endpoint_fallback(
    test_app,
    mock_model_server,
    mock_json_validator,
    mock_fallback_mapper,
    mock_config_manager,
):
    """Test fallback mapping when model confidence is low."""
    # Configure low confidence response
    mock_json_validator.parse_output.return_value = MappingResponse(
        taxonomy=["HARM.SPEECH.Toxicity"],
        scores={"HARM.SPEECH.Toxicity": 0.95},
        confidence=0.3,  # Below threshold
        notes=None,
        version_info=None,
    )
    mock_config_manager.confidence.threshold = 0.6

    request_data = {"detector": "deberta-toxicity", "output": "toxic"}

    response = test_app.post("/map", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Should get fallback response
    assert data["taxonomy"] == ["OTHER.Unknown"]
    assert "rule-based mapping" in data["notes"].lower()

    # Verify fallback was called
    mock_fallback_mapper.map.assert_called_once_with(
        "deberta-toxicity", "toxic", reason="low_confidence"
    )


def test_batch_map_endpoint(test_app):
    """Test batch mapping endpoint."""
    request_data = {
        "requests": [
            {"detector": "deberta-toxicity", "output": "toxic"},
            {"detector": "openai-moderation", "output": "hate"},
        ]
    }

    response = test_app.post("/map/batch", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["taxonomy"] == ["HARM.SPEECH.Toxicity"]
    assert data["results"][1]["taxonomy"] == ["HARM.SPEECH.Toxicity"]


def test_invalid_request(test_app):
    """Test handling of invalid requests."""
    # Missing required fields
    request_data = {"detector": "test"}  # Missing 'output'

    response = test_app.post("/map", json=request_data)
    assert response.status_code == 422  # Validation error


def test_request_id_header(test_app):
    """Test that request ID is added to response headers."""
    request_data = {"detector": "deberta-toxicity", "output": "toxic"}

    response = test_app.post("/map", json=request_data)
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
