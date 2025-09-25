"""
Model management CLI commands for Mapper Service.

Single Responsibility: Handle ML model operations via CLI.
Provides commands for model loading, management, and monitoring.
"""

import asyncio
import json
import sys
from typing import Optional

import click

from ..ml import ModelManager, ModelConfig, ModelBackend
from ..config.settings import MapperSettings


@click.group()
def model():
    """Model management - load, unload, and monitor ML models."""
    pass


@model.command()
@click.option("--model-id", "-m", required=True, help="Unique model identifier")
@click.option(
    "--model-path",
    "-p",
    required=True,
    help="Path to model (local path or HuggingFace model ID)",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["transformers", "vllm", "tgi", "onnx"]),
    default="transformers",
    help="Model backend to use",
)
@click.option(
    "--device",
    "-d",
    default="auto",
    help="Device to load model on (auto, cpu, cuda:0, etc.)",
)
@click.option("--max-length", type=int, default=2048, help="Maximum sequence length")
@click.option(
    "--torch-dtype",
    default="auto",
    help="PyTorch data type (auto, float16, float32, etc.)",
)
def load(
    model_id: str,
    model_path: str,
    backend: str,
    device: str,
    max_length: int,
    torch_dtype: str,
):
    """Load a model for inference.

    Examples:
        mapper model load -m llama-mapper -p ./models/llama-3-8b-mapper
        mapper model load -m deberta-classifier -p microsoft/deberta-v3-base -b transformers
    """

    async def _load():
        try:
            # Create model manager
            model_manager = ModelManager()

            # Create model configuration
            config = ModelConfig(
                model_id=model_id,
                model_path=model_path,
                backend=ModelBackend(backend),
                device=device,
                max_length=max_length,
                torch_dtype=torch_dtype,
            )

            click.echo(f"Loading model '{model_id}' from {model_path}...", err=True)
            click.echo(f"Backend: {backend}, Device: {device}", err=True)

            # Load model
            success = await model_manager.load_model(config)

            if success:
                click.echo(f"✓ Model '{model_id}' loaded successfully", err=True)

                # Show model info
                model_data = model_manager.get_model(model_id)
                if model_data:
                    click.echo(f"Model backend: {model_data['backend'].value}")

                    # Show metrics if available
                    metrics = model_manager.get_model_metrics(model_id)
                    if metrics:
                        click.echo(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
            else:
                click.echo(f"❌ Failed to load model '{model_id}'", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "model_manager" in locals():
                await model_manager.cleanup()

    asyncio.run(_load())


@model.command()
@click.option(
    "--model-id",
    "-m",
    help="Model ID to unload (if not specified, lists loaded models)",
)
def unload(model_id: Optional[str]):
    """Unload a model from memory.

    Examples:
        mapper model unload -m llama-mapper
        mapper model unload  # Lists loaded models
    """

    async def _unload():
        try:
            model_manager = ModelManager()

            if not model_id:
                # List loaded models
                loaded_models = model_manager.list_loaded_models()
                if not loaded_models:
                    click.echo("No models currently loaded")
                    return

                click.echo("Loaded models:")
                for mid in loaded_models:
                    status = model_manager.get_model_status(mid)
                    click.echo(f"  • {mid} ({status.value})")
                return

            click.echo(f"Unloading model '{model_id}'...", err=True)

            success = await model_manager.unload_model(model_id)

            if success:
                click.echo(f"✓ Model '{model_id}' unloaded successfully")
            else:
                click.echo(f"❌ Failed to unload model '{model_id}' (not found)")
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "model_manager" in locals():
                await model_manager.cleanup()

    asyncio.run(_unload())


@model.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def list(output_format: str):
    """List all loaded models and their status."""

    async def _list():
        try:
            model_manager = ModelManager()

            loaded_models = model_manager.list_loaded_models()

            if not loaded_models:
                click.echo("No models currently loaded")
                return

            if output_format == "json":
                models_info = {}
                for model_id in loaded_models:
                    status = model_manager.get_model_status(model_id)
                    metrics = model_manager.get_model_metrics(model_id)
                    model_data = model_manager.get_model(model_id)

                    models_info[model_id] = {
                        "status": status.value,
                        "backend": (
                            model_data["backend"].value if model_data else "unknown"
                        ),
                        "inference_count": metrics.inference_count if metrics else 0,
                        "avg_inference_time": (
                            metrics.avg_inference_time if metrics else 0.0
                        ),
                        "memory_usage_mb": metrics.memory_usage_mb if metrics else 0.0,
                    }

                click.echo(json.dumps(models_info, indent=2))

            else:  # table format
                click.echo("Loaded Models")
                click.echo("=" * 80)
                click.echo(
                    f"{'Model ID':<20} {'Status':<10} {'Backend':<12} {'Inferences':<12} {'Avg Time':<10} {'Memory':<10}"
                )
                click.echo("-" * 80)

                for model_id in loaded_models:
                    status = model_manager.get_model_status(model_id)
                    metrics = model_manager.get_model_metrics(model_id)
                    model_data = model_manager.get_model(model_id)

                    backend = model_data["backend"].value if model_data else "unknown"
                    inferences = metrics.inference_count if metrics else 0
                    avg_time = (
                        f"{metrics.avg_inference_time:.2f}s" if metrics else "0.00s"
                    )
                    memory = f"{metrics.memory_usage_mb:.1f}MB" if metrics else "0.0MB"

                    click.echo(
                        f"{model_id:<20} {status.value:<10} {backend:<12} {inferences:<12} {avg_time:<10} {memory:<10}"
                    )

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "model_manager" in locals():
                await model_manager.cleanup()

    asyncio.run(_list())


@model.command()
@click.option(
    "--model-id",
    "-m",
    help="Specific model to check (if not specified, checks all models)",
)
def health(model_id: Optional[str]):
    """Check model health and performance metrics."""

    async def _health():
        try:
            model_manager = ModelManager()

            health_status = await model_manager.health_check()

            if model_id:
                # Show specific model health
                if model_id not in health_status.get("models", {}):
                    click.echo(f"❌ Model '{model_id}' not found")
                    sys.exit(1)

                model_health = health_status["models"][model_id]
                click.echo(f"Model Health: {model_id}")
                click.echo("=" * 40)
                click.echo(f"Status: {model_health['status']}")
                click.echo(f"Backend: {model_health.get('backend', 'unknown')}")
                click.echo(f"Inferences: {model_health.get('inference_count', 0)}")
                click.echo(f"Errors: {model_health.get('error_count', 0)}")
                click.echo(
                    f"Avg Time: {model_health.get('avg_inference_time', 0):.3f}s"
                )
                click.echo(f"Memory: {model_health.get('memory_usage_mb', 0):.1f}MB")

                if model_health.get("error"):
                    click.echo(f"Last Error: {model_health['error']}")

            else:
                # Show overall health
                click.echo("Model Manager Health")
                click.echo("=" * 40)
                click.echo(
                    f"Overall Status: {'✓ Healthy' if health_status['healthy'] else '❌ Unhealthy'}"
                )
                click.echo(f"Loaded Models: {health_status['loaded_models']}")

                if health_status.get("models"):
                    click.echo("\nModel Status:")
                    for mid, mhealth in health_status["models"].items():
                        status_icon = "✓" if mhealth["status"] == "loaded" else "❌"
                        click.echo(f"  {status_icon} {mid}: {mhealth['status']}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "model_manager" in locals():
                await model_manager.cleanup()

    asyncio.run(_health())
