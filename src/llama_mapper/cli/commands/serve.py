"""Serving and training CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from ...api.mapper import create_app
from ...logging import get_logger
from ...monitoring.metrics_collector import MetricsCollector
from ...serving.fallback_mapper import FallbackMapper
from ...serving.json_validator import JSONValidator
from ...serving.model_server import GenerationConfig, create_model_server
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach server-related commands to the root CLI."""

    @main.command()
    @click.pass_context
    def train(ctx: click.Context) -> None:
        """Train the LoRA fine-tuned model."""
        config_manager = get_config_manager(ctx)
        logger = get_logger(__name__)

        logger.info(
            "Training command invoked",
            model_name=config_manager.model.name,
            lora_r=config_manager.model.lora_r,
            lora_alpha=config_manager.model.lora_alpha,
        )

        click.echo("Training not implemented yet")
        click.echo(f"Model: {config_manager.model.name}")
        click.echo(
            f"LoRA config: r={config_manager.model.lora_r}, α={config_manager.model.lora_alpha}"
        )

    @main.command()
    @click.option("--host", help="Host to bind to")
    @click.option("--port", type=int, help="Port to bind to")
    @click.pass_context
    def serve(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
        """Start the FastAPI server."""
        import uvicorn

        config_manager = get_config_manager(ctx)
        logger = get_logger(__name__)

        serve_host = host or config_manager.serving.host
        serve_port = port or config_manager.serving.port

        logger.info(
            "Serve command invoked",
            host=serve_host,
            port=serve_port,
            backend=config_manager.serving.backend,
        )

        try:
            schema_path = str(Path(".kiro/pillars-detectors/schema.json"))
            detectors_dir = str(Path(".kiro/pillars-detectors"))

            gen_cfg = GenerationConfig(
                temperature=config_manager.model.temperature,
                top_p=config_manager.model.top_p,
                max_new_tokens=config_manager.model.max_new_tokens,
            )
            model_server = create_model_server(
                backend=config_manager.serving.backend,
                model_path=config_manager.model.name,
                generation_config=gen_cfg,
                gpu_memory_utilization=config_manager.serving.gpu_memory_utilization,
            )
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
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize app", error=str(exc))
            raise SystemExit(1) from exc

        uvicorn.run(
            app,
            host=serve_host,
            port=serve_port,
            loop="uvloop",
            http="httptools",
            access_log=False,
            workers=config_manager.serving.workers,
            timeout_keep_alive=5,
        )
