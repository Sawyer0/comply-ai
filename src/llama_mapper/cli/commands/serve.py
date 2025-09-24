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
    @click.option("--model-type", type=click.Choice(["dual", "mapper", "analyst"]), default="dual", help="Type of model to train")
    @click.option("--output-dir", type=str, help="Output directory for training artifacts")
    @click.option("--config-file", type=str, help="Path to training configuration file")
    @click.option("--async", "async_execution", is_flag=True, help="Run training asynchronously")
    @click.pass_context
    def train(ctx: click.Context, model_type: str, output_dir: Optional[str], config_file: Optional[str], async_execution: bool) -> None:
        """Train models using the pipeline orchestration system."""
        import asyncio
        from pathlib import Path
        from ...pipeline import PipelineOrchestrator
        from ...pipeline.factories.training_pipeline_factory import TrainingPipelineFactory
        
        config_manager = get_config_manager(ctx)
        logger = get_logger(__name__)

        logger.info(
            "Training command invoked",
            model_type=model_type,
            output_dir=output_dir,
            async_execution=async_execution
        )

        # Set default output directory
        if not output_dir:
            output_dir = f"./training_output_{model_type}"
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        async def run_training():
            try:
                # Initialize orchestrator
                orchestrator = PipelineOrchestrator()
                
                # Create training pipeline
                factory = TrainingPipelineFactory()
                
                # Build training configuration
                training_config = {
                    "model_type": model_type,
                    "output_dir": output_dir,
                    "mapper_learning_rate": config_manager.model.lora_alpha / 10000,  # Convert from alpha
                    "mapper_lora_r": config_manager.model.lora_r,
                    "mapper_lora_alpha": config_manager.model.lora_alpha,
                    "mapper_epochs": 2,
                    "mapper_batch_size": 4,
                    "analyst_learning_rate": 1e-4,
                    "analyst_lora_r": 128,
                    "analyst_lora_alpha": 256,
                    "analyst_epochs": 2,
                    "analyst_batch_size": 8,
                    "training_datasets": ["hybrid_generated", "analysis_generated"]
                }
                
                # Load additional config from file if provided
                if config_file:
                    import json
                    with open(config_file, 'r') as f:
                        file_config = json.load(f)
                    training_config.update(file_config)
                
                # Create and register training pipeline
                pipeline = await factory.create_training_pipeline(model_type, training_config)
                await orchestrator.register_pipeline(pipeline)
                
                click.echo(f"🚀 Starting {model_type} model training...")
                click.echo(f"📁 Output directory: {output_dir}")
                click.echo(f"⚡ Async execution: {async_execution}")
                
                # Execute pipeline
                result_context = await orchestrator.execute_pipeline(
                    pipeline.name,
                    config=training_config,
                    async_execution=async_execution
                )
                
                if async_execution:
                    click.echo("✅ Training started asynchronously")
                    click.echo(f"📊 Monitor progress in: {output_dir}")
                else:
                    # Get results
                    training_results = result_context.artifacts.get("training_results", {})
                    evaluation_results = result_context.artifacts.get("evaluation_results", {})
                    deployment_results = result_context.artifacts.get("deployment_results", {})
                    
                    click.echo("✅ Training pipeline completed successfully!")
                    
                    # Display results summary
                    if training_results:
                        click.echo(f"\n📈 Training Results:")
                        if model_type == "dual":
                            click.echo(f"  Mapper: {training_results.get('mapper', {}).get('version', 'N/A')}")
                            click.echo(f"  Analyst: {training_results.get('analyst', {}).get('version', 'N/A')}")
                        else:
                            click.echo(f"  Version: {training_results.get('version', 'N/A')}")
                    
                    if evaluation_results:
                        click.echo(f"\n🎯 Evaluation Results:")
                        if model_type == "dual":
                            mapper_passed = evaluation_results.get("mapper", {}).get("passed_quality_gates", False)
                            analyst_passed = evaluation_results.get("analyst", {}).get("passed_quality_gates", False)
                            click.echo(f"  Mapper Quality Gates: {'✅ PASSED' if mapper_passed else '❌ FAILED'}")
                            click.echo(f"  Analyst Quality Gates: {'✅ PASSED' if analyst_passed else '❌ FAILED'}")
                        else:
                            passed = evaluation_results.get("passed_quality_gates", False)
                            click.echo(f"  Quality Gates: {'✅ PASSED' if passed else '❌ FAILED'}")
                    
                    if deployment_results:
                        status = deployment_results.get("overall_status", "unknown")
                        click.echo(f"\n🚀 Deployment Status: {status}")
                
                click.echo(f"\n📁 All artifacts saved to: {output_dir}")
                
            except Exception as e:
                logger.error("Training pipeline failed", error=str(e))
                click.echo(f"❌ Training failed: {str(e)}", err=True)
                raise click.ClickException(f"Training failed: {str(e)}")

        # Run the async training function
        try:
            asyncio.run(run_training())
        except KeyboardInterrupt:
            click.echo("\n⚠️  Training interrupted by user")
        except Exception as e:
            logger.error("Training command failed", error=str(e))
            raise

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
