"""
CLI commands for training operations.

Single responsibility: Command-line interface for training management.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import click

from ..training.trainer import LoRATrainer, TrainingConfig
from ..training.checkpoint_manager import CheckpointManager
from ..training.model_loader import ModelLoader
from ..training.version_manager import ModelVersionManager, DeploymentManager, ModelType

logger = logging.getLogger(__name__)


@click.group()
def training_cli():
    """Training management commands."""
    pass


@training_cli.command()
@click.option("--train-data", required=True, help="Path to training data JSON file")
@click.option("--eval-data", help="Path to evaluation data JSON file")
@click.option(
    "--output-dir", default="./checkpoints", help="Output directory for checkpoints"
)
@click.option("--learning-rate", default=2e-4, type=float, help="Learning rate")
@click.option("--epochs", default=3, type=int, help="Number of training epochs")
@click.option("--batch-size", default=4, type=int, help="Per-device batch size")
@click.option("--lora-r", default=16, type=int, help="LoRA rank")
@click.option("--lora-alpha", default=32, type=int, help="LoRA alpha")
@click.option("--resume-from", help="Resume training from checkpoint")
def train(
    train_data: str,
    eval_data: Optional[str],
    output_dir: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    resume_from: Optional[str],
):
    """Start model training."""
    try:
        # Load training data
        with open(train_data, "r") as f:
            train_examples = json.load(f)

        eval_examples = None
        if eval_data:
            with open(eval_data, "r") as f:
                eval_examples = json.load(f)

        # Create training configuration
        config = TrainingConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Initialize trainer
        trainer = LoRATrainer(config)
        model_loader = ModelLoader()

        click.echo("Starting training...")

        # Start training
        metrics = trainer.train(
            train_examples=train_examples,
            eval_examples=eval_examples,
            model_loader=model_loader,
            resume_from_checkpoint=resume_from,
        )

        # Save checkpoint
        checkpoint_manager = CheckpointManager()
        version_id = checkpoint_manager.save_checkpoint(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            training_metrics=metrics,
            tags=["cli_training"],
        )

        click.echo(f"Training completed successfully!")
        click.echo(f"Version ID: {version_id}")
        click.echo(f"Final loss: {metrics.get('train_loss', 'N/A')}")

    except Exception as e:
        click.echo(f"Training failed: {str(e)}", err=True)
        raise click.Abort()


@training_cli.command()
@click.option("--tags", multiple=True, help="Filter by tags")
def list_versions(tags: List[str]):
    """List available model versions."""
    try:
        checkpoint_manager = CheckpointManager()
        versions = checkpoint_manager.list_versions(tags=list(tags) if tags else None)

        if not versions:
            click.echo("No versions found.")
            return

        click.echo("Available versions:")
        click.echo("-" * 80)

        for version in versions:
            click.echo(f"Version: {version.version}")
            click.echo(f"Created: {version.created_at}")
            click.echo(f"Path: {version.checkpoint_path}")

            # Show training metrics if available
            metrics = version.metadata.get("training_metrics", {})
            if metrics:
                click.echo(f"Train Loss: {metrics.get('train_loss', 'N/A')}")
                click.echo(f"Runtime: {metrics.get('train_runtime', 'N/A')}s")

            tags = version.metadata.get("tags", [])
            if tags:
                click.echo(f"Tags: {', '.join(tags)}")

            click.echo("-" * 80)

    except Exception as e:
        click.echo(f"Failed to list versions: {str(e)}", err=True)
        raise click.Abort()


@training_cli.command()
@click.argument("version_id")
@click.confirmation_option(prompt="Are you sure you want to delete this version?")
def delete_version(version_id: str):
    """Delete a model version."""
    try:
        checkpoint_manager = CheckpointManager()
        success = checkpoint_manager.delete_version(version_id)

        if success:
            click.echo(f"Version {version_id} deleted successfully.")
        else:
            click.echo(f"Version {version_id} not found.", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Failed to delete version: {str(e)}", err=True)
        raise click.Abort()


@training_cli.command()
@click.option("--model-name", required=True, help="Model name")
@click.option("--model-type", default="mapper", help="Model type (mapper/analyst)")
@click.option(
    "--training-datasets", multiple=True, required=True, help="Training datasets"
)
@click.option("--config-file", required=True, help="Training config JSON file")
@click.option("--metrics-file", required=True, help="Performance metrics JSON file")
@click.option(
    "--golden-metrics-file", required=True, help="Golden set metrics JSON file"
)
@click.option("--created-by", default="cli", help="Creator name")
def create_model(
    model_name: str,
    model_type: str,
    training_datasets: List[str],
    config_file: str,
    metrics_file: str,
    golden_metrics_file: str,
    created_by: str,
):
    """Create a new model version."""
    try:
        # Load configuration files
        with open(config_file, "r") as f:
            training_config = json.load(f)

        with open(metrics_file, "r") as f:
            performance_metrics = json.load(f)

        with open(golden_metrics_file, "r") as f:
            golden_set_metrics = json.load(f)

        # Create model version
        version_manager = ModelVersionManager()
        model_type_enum = (
            ModelType.MAPPER if model_type.lower() == "mapper" else ModelType.ANALYST
        )

        version = version_manager.create_model_version(
            model_name=model_name,
            model_type=model_type_enum,
            training_datasets=list(training_datasets),
            training_config=training_config,
            performance_metrics=performance_metrics,
            golden_set_metrics=golden_set_metrics,
            created_by=created_by,
        )

        click.echo(f"Model version created successfully!")
        click.echo(f"Model: {model_name}")
        click.echo(f"Version: {version}")

    except Exception as e:
        click.echo(f"Failed to create model version: {str(e)}", err=True)
        raise click.Abort()


@training_cli.command()
@click.argument("model_name")
def list_model_versions(model_name: str):
    """List versions for a specific model."""
    try:
        version_manager = ModelVersionManager()
        versions = version_manager.list_model_versions(model_name)

        if not versions:
            click.echo(f"No versions found for model: {model_name}")
            return

        click.echo(f"Versions for model: {model_name}")
        click.echo("-" * 40)

        for version in versions:
            click.echo(f"  {version}")

    except Exception as e:
        click.echo(f"Failed to list model versions: {str(e)}", err=True)
        raise click.Abort()


@training_cli.command()
@click.option("--model-name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version")
@click.option(
    "--canary-percentage", default=5.0, type=float, help="Canary traffic percentage"
)
@click.option("--baseline-version", help="Baseline version for comparison")
def deploy_canary(
    model_name: str,
    version: str,
    canary_percentage: float,
    baseline_version: Optional[str],
):
    """Deploy model as canary."""
    try:
        from ..training.version_manager import CanaryConfig

        version_manager = ModelVersionManager()
        deployment_manager = DeploymentManager(version_manager)

        canary_config = CanaryConfig(canary_percentage=canary_percentage)

        success = deployment_manager.deploy_canary(
            model_name=model_name,
            version=version,
            canary_config=canary_config,
            baseline_version=baseline_version,
        )

        if success:
            click.echo(f"Canary deployment successful!")
            click.echo(f"Model: {model_name}@{version}")
            click.echo(f"Traffic: {canary_percentage}%")
        else:
            click.echo("Canary deployment failed.", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Failed to deploy canary: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    training_cli()
