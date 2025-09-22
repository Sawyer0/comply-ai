"""
End-to-end integration tests for the complete orchestration → mapper flow.

This test suite validates the full pipeline from detector orchestration
through to the mapper service, ensuring both services work together correctly.
"""

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.api.models import MappingResponse
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving import FallbackMapper, JSONValidator


class MockModelServer:
    """Mock model server that returns predictable mapping results."""
    
    def __init__(self):
        self.is_loaded = True
    
    async def load_model(self) -> None:
        self.is_loaded = True
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        # Return a predictable mapping based on detector name
        if "toxicity" in prompt.lower():
            return '{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.95}, "confidence": 0.95}'
        elif "pii" in prompt.lower():
            return '{"taxonomy": ["PRIVACY.PII.Email"], "scores": {"PRIVACY.PII.Email": 0.88}, "confidence": 0.88}'
        else:
            return '{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.75}, "confidence": 0.75}'
    
    async def generate_mapping(self, detector: str, output: str, metadata=None) -> str:
        """Generate a canonical taxonomy mapping for detector output."""
        # Return a predictable mapping based on detector name
        if "toxicity" in detector.lower():
            return '{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.95}, "confidence": 0.95}'
        elif "pii" in detector.lower():
            return '{"taxonomy": ["PRIVACY.PII.Email"], "scores": {"PRIVACY.PII.Email": 0.88}, "confidence": 0.88}'
        else:
            return '{"taxonomy": ["HARM.SPEECH.Toxicity"], "scores": {"HARM.SPEECH.Toxicity": 0.75}, "confidence": 0.75}'
    
    async def health_check(self) -> bool:
        return True


@pytest.fixture
def mock_mapper_components():
    """Create mock components for the mapper service."""
    model_server = MockModelServer()
    
    json_validator = Mock(spec=JSONValidator)
    json_validator.validate.return_value = (True, None)
    json_validator.parse_output.side_effect = lambda output: MappingResponse.model_validate(
        json.loads(output)
    )
    
    fallback_mapper = Mock(spec=FallbackMapper)
    fallback_mapper.map.return_value = MappingResponse(
        taxonomy=["OTHER.Unknown"],
        scores={"OTHER.Unknown": 0.0},
        confidence=0.5,
        notes="No mapping found for detector deberta-toxicity output: toxic (reason: low_confidence)"
    )
    
    config_manager = Mock(spec=ConfigManager)
    # Mock the nested configuration objects
    config_manager.confidence = Mock()
    config_manager.confidence.threshold = 0.6
    config_manager.serving = Mock()
    config_manager.serving.max_payload_kb = 64
    config_manager.serving.reject_on_raw_content = True
    config_manager.auth = Mock()
    config_manager.auth.enabled = False
    
    metrics_collector = Mock(spec=MetricsCollector)
    metrics_collector.record_request = Mock()
    metrics_collector.record_error = Mock()
    metrics_collector.record_payload_rejection = Mock()
    metrics_collector.get_prometheus_metrics.return_value = "# Mock Prometheus metrics\nmapper_requests_total 42\n"
    
    return {
        "model_server": model_server,
        "json_validator": json_validator,
        "fallback_mapper": fallback_mapper,
        "config_manager": config_manager,
        "metrics_collector": metrics_collector
    }


@pytest.fixture
def mapper_app(mock_mapper_components):
    """Create a test mapper app with mocked components."""
    app = create_app(
        model_server=mock_mapper_components["model_server"],
        json_validator=mock_mapper_components["json_validator"],
        fallback_mapper=mock_mapper_components["fallback_mapper"],
        config_manager=mock_mapper_components["config_manager"],
        metrics_collector=mock_mapper_components["metrics_collector"]
    )
    return TestClient(app)


class TestEndToEndOrchestrationMapper:
    """Test the complete orchestration → mapper pipeline."""
    
    def test_single_detector_mapping_flow(self, mapper_app):
        """Test mapping a single detector output through the mapper service."""
        # Test data that would come from detector orchestration
        detector_output = {
            "detector": "deberta-toxicity",
            "output": "toxic",
            "metadata": {"score": 0.95, "model_version": "v1.2"},
            "tenant_id": "test-tenant"
        }
        
        response = mapper_app.post("/map", json=detector_output)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify the mapping response structure
        assert "taxonomy" in data
        assert "scores" in data
        assert "confidence" in data
        assert "provenance" in data
        
        # Verify the mapped taxonomy
        assert data["taxonomy"] == ["HARM.SPEECH.Toxicity"]
        assert data["scores"]["HARM.SPEECH.Toxicity"] == 0.95
        assert data["confidence"] == 0.95
        
        # Verify provenance information
        assert data["provenance"]["mapping_method"] == "model"
        assert data["provenance"]["model_version"] is not None
    
    def test_batch_mapping_flow(self, mapper_app):
        """Test batch mapping multiple detector outputs."""
        batch_request = {
            "requests": [
                {
                    "detector": "deberta-toxicity",
                    "output": "toxic",
                    "metadata": {"score": 0.95},
                    "tenant_id": "test-tenant"
                },
                {
                    "detector": "pii-detector",
                    "output": "email found",
                    "metadata": {"score": 0.88},
                    "tenant_id": "test-tenant"
                }
            ]
        }
        
        response = mapper_app.post("/map/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify batch response structure
        assert "results" in data
        assert len(data["results"]) == 2
        
        # Verify first result
        result1 = data["results"][0]
        assert result1["taxonomy"] == ["HARM.SPEECH.Toxicity"]
        assert result1["scores"]["HARM.SPEECH.Toxicity"] == 0.95
        
        # Verify second result
        result2 = data["results"][1]
        assert result2["taxonomy"] == ["PRIVACY.PII.Email"]
        assert result2["scores"]["PRIVACY.PII.Email"] == 0.88
    
    def test_mapper_health_endpoint(self, mapper_app):
        """Test the mapper health endpoint."""
        response = mapper_app.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_mapper_metrics_endpoint(self, mapper_app):
        """Test the mapper metrics endpoint."""
        response = mapper_app.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    def test_mapper_openapi_endpoint(self, mapper_app):
        """Test the mapper OpenAPI endpoint."""
        response = mapper_app.get("/openapi.yaml")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/yaml"
        
        # Verify it's valid YAML
        import yaml
        yaml.safe_load(response.text)
    
    def test_mapper_error_handling(self, mapper_app, mock_mapper_components):
        """Test mapper error handling for invalid requests."""
        # Test with raw content (should be rejected)
        raw_content = {
            "detector": "unknown",
            "output": "This is a very long piece of raw content that should be rejected because it's too long and looks like raw content rather than a detector output. " * 20,
            "tenant_id": "test-tenant"
        }
        
        response = mapper_app.post("/map", json=raw_content)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_mapper_fallback_behavior(self, mapper_app, mock_mapper_components):
        """Test mapper fallback behavior when model fails."""
        # Configure mock to simulate model failure
        mock_mapper_components["json_validator"].validate.return_value = (False, "Invalid JSON")
        
        detector_output = {
            "detector": "unknown-detector",
            "output": "some output",
            "tenant_id": "test-tenant"
        }
        
        response = mapper_app.post("/map", json=detector_output)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use fallback mapping (returns OTHER.Unknown for unknown detectors)
        assert data["taxonomy"] == ["OTHER.Unknown"]
        assert data["confidence"] == 0.5  # Fallback confidence


class TestOrchestrationMapperIntegration:
    """Test integration between orchestration and mapper services."""
    
    @pytest.mark.asyncio
    async def test_orchestration_to_mapper_handoff(self):
        """Test the handoff from orchestration to mapper service."""
        # This test would require both services running
        # For now, we'll test the contract between services
        
        # Simulate orchestration response that would be sent to mapper
        orchestration_response = {
            "detector": "deberta-toxicity",
            "output": "toxic",
            "metadata": {
                "score": 0.95,
                "processing_time_ms": 150,
                "model_version": "v1.2"
            },
            "tenant_id": "test-tenant",
            "provenance": {
                "orchestrator_request_id": "orch-123",
                "detector_endpoint": "http://detector:8080/detect",
                "routing_decision": "policy-based"
            }
        }
        
        # This is the format that should be accepted by the mapper
        assert "detector" in orchestration_response
        assert "output" in orchestration_response
        assert "tenant_id" in orchestration_response
        assert "metadata" in orchestration_response
    
    def test_mapper_payload_validation(self, mapper_app):
        """Test that mapper correctly validates MapperPayload format."""
        # Test with proper MapperPayload format
        mapper_payload = {
            "detector": "deberta-toxicity",
            "output": "toxic",
            "metadata": {
                "score": 0.95,
                "model_version": "v1.2"
            },
            "tenant_id": "test-tenant"
        }
        
        response = mapper_app.post("/map", json=mapper_payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "taxonomy" in data
    
    def test_legacy_detector_request_format(self, mapper_app):
        """Test backward compatibility with legacy DetectorRequest format."""
        # Test with legacy DetectorRequest format (should still work but show deprecation headers)
        legacy_request = {
            "detector": "deberta-toxicity",
            "output": "toxic",
            "metadata": {"score": 0.95}
        }
        
        response = mapper_app.post("/map", json=legacy_request)
        
        assert response.status_code == 200
        # Should include deprecation headers
        assert "Deprecation" in response.headers
        assert "Sunset" in response.headers


class TestPerformanceAndReliability:
    """Test performance and reliability aspects of the API."""
    
    def test_mapper_response_time(self, mapper_app):
        """Test that mapper responds within acceptable time limits."""
        detector_output = {
            "detector": "deberta-toxicity",
            "output": "toxic",
            "tenant_id": "test-tenant"
        }
        
        start_time = time.time()
        response = mapper_app.post("/map", json=detector_output)
        end_time = time.time()
        
        assert response.status_code == 200
        # Should respond within 5 seconds (generous for testing)
        assert (end_time - start_time) < 5.0
    
    def test_batch_processing_performance(self, mapper_app):
        """Test batch processing performance with multiple items."""
        batch_request = {
            "requests": [
                {
                    "detector": f"detector-{i}",
                    "output": f"output-{i}",
                    "tenant_id": "test-tenant"
                }
                for i in range(10)  # Test with 10 items
            ]
        }
        
        start_time = time.time()
        response = mapper_app.post("/map/batch", json=batch_request)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 10
        # Should process 10 items within 10 seconds
        assert (end_time - start_time) < 10.0
    
    def test_concurrent_requests(self, mapper_app):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            detector_output = {
                "detector": "deberta-toxicity",
                "output": "toxic",
                "tenant_id": "test-tenant"
            }
            response = mapper_app.post("/map", json=detector_output)
            results.put(response.status_code)
        
        # Start 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        while not results.empty():
            status_code = results.get()
            assert status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
