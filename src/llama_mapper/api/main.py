"""
Main entry point for running the FastAPI service.
"""

import asyncio
import logging

import uvicorn
from fastapi import FastAPI

from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector
from ..serving import (
    FallbackMapper,
    GenerationConfig,
    JSONValidator,
    create_model_server,
)
from .mapper import create_app

logger = logging.getLogger(__name__)


async def create_service() -> FastAPI:
    """Create and configure the FastAPI service with all dependencies."""

    # Load configuration
    config_manager = ConfigManager()

    # Create generation config
    generation_config = GenerationConfig(
        temperature=0.1,  # Low temperature for deterministic mapping
        top_p=0.9,
        max_new_tokens=200,
        do_sample=True,
    )

    # Create model server (defaults to vLLM if available, otherwise TGI)
    model_path = str(
        getattr(config_manager, "model_path", "meta-llama/Llama-3-8B-Instruct")
    )
    backend = str(getattr(config_manager, "serving_backend", "vllm"))

    model_server = create_model_server(
        backend=backend, model_path=model_path, generation_config=generation_config
    )

    # Load the model
    await model_server.load_model()

    # Create JSON validator
    schema_path = str(
        getattr(config_manager, "schema_path", "pillars-detectors/schema.json")
    )
    json_validator = JSONValidator(schema_path=schema_path)

    # Create fallback mapper
    detector_configs_path = str(
        getattr(config_manager, "detector_configs_path", "pillars-detectors")
    )
    fallback_mapper = FallbackMapper(detector_configs_path=detector_configs_path)

    # Create metrics collector
    metrics_collector = MetricsCollector()

    # Create the FastAPI app
    app = create_app(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics_collector,
    )

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # For development, we can use uvicorn directly
    if reload:
        uvicorn.run(
            "src.llama_mapper.api.main:create_service",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )
    else:
        # For production, create the app and run
        app = asyncio.run(create_service())
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Llama Mapper API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
