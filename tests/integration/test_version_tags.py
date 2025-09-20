"""Integration test to enforce version-tagged outputs in API responses.

This test spins up the FastAPI app with a stubbed ModelServer that forces
fallback mapping. It then verifies that:
- MappingResponse.notes contains a "versions:" tag
- MappingResponse.provenance.model is present (string)
"""

from pathlib import Path

from fastapi.testclient import TestClient

from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving.fallback_mapper import FallbackMapper
from src.llama_mapper.serving.json_validator import JSONValidator
from src.llama_mapper.serving.model_server import ModelServer


class _FailingModelServer(ModelServer):
    async def load_model(self) -> None:  # type: ignore[override]
        self.is_loaded = True

    async def generate_text(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        # Force failure so the API uses the fallback mapper
        raise RuntimeError("Stubbed model failure for test")

    async def health_check(self) -> bool:  # type: ignore[override]
        return True


def _build_app():
    schema_path = str(Path(".kiro/pillars-detectors/schema.json"))
    detectors_dir = str(Path(".kiro/pillars-detectors"))

    model_server = _FailingModelServer(model_path="stub")
    json_validator = JSONValidator(schema_path=schema_path)
    fallback_mapper = FallbackMapper(detector_configs_path=detectors_dir)
    config_manager = ConfigManager()
    metrics = MetricsCollector()

    return create_app(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics,
    )


def test_mapping_response_contains_version_tags():
    app = _build_app()
    client = TestClient(app)

    payload = {
        "detector": "deberta-toxicity",
        "output": "toxic",
        "tenant_id": "test-tenant",
    }

    resp = client.post("/map", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Verify notes contain version tag
    assert "notes" in data and isinstance(data["notes"], str)
    assert (
        "versions:" in data["notes"]
    ), f"Missing version tag in notes: {data['notes']}"

    # Verify provenance.model is present
    assert "provenance" in data and isinstance(data["provenance"], dict)
    assert "model" in data["provenance"], "provenance.model missing"
    assert isinstance(data["provenance"]["model"], str)
    assert (
        data["provenance"]["model"] != "unknown"
    ), "provenance.model must be a non-'unknown' version string"

    # Fallback mapping should map toxic -> HARM.SPEECH.Toxicity
    assert "taxonomy" in data and "HARM.SPEECH.Toxicity" in data["taxonomy"], data
