from fastapi import FastAPI

from ..api.mapper import create_app
from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector
from ..serving import (
    FallbackMapper,
    GenerationConfig,
    JSONValidator,
    create_model_server,
)


def create_rules_only_app() -> FastAPI:
    """
    Factory for a FastAPI app configured for rules-only mapping.

    - Builds ConfigManager from default config.yaml / env
    - Creates model_server without loading any model
    - Uses ./.kiro/pillars-detectors for schema and detector mappings
    - Returns the FastAPI app via create_app
    """
    config_manager = ConfigManager()

    # Force runtime into rules_only so mapper bypasses model generation path
    config_manager.serving = config_manager.serving.model_copy(
        update={"mode": "rules_only"}
    )

    # Generation config (only used if mode != rules_only)
    gen_cfg = GenerationConfig(
        temperature=config_manager.model.temperature,
        top_p=config_manager.model.top_p,
        max_new_tokens=config_manager.model.max_new_tokens,
    )

    # Create model server but DO NOT load model; in rules_only it won't be used
    model_server = create_model_server(
        backend=config_manager.serving.backend,
        model_path=config_manager.model.name,
        generation_config=gen_cfg,
        gpu_memory_utilization=getattr(
            config_manager.serving, "gpu_memory_utilization", 0.9
        ),
    )

    # Prefer local .kiro artifacts
    schema_path = ".kiro/pillars-detectors/schema.json"
    detectors_dir = ".kiro/pillars-detectors"

    json_validator = JSONValidator(schema_path=schema_path)
    fallback_mapper = FallbackMapper(detector_configs_path=detectors_dir)
    metrics = MetricsCollector()

    app = create_app(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics,
    )

    return app
