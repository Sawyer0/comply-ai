"""
Pipeline Management CLI Commands

Commands for managing and monitoring pipeline execution.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import click

from ...logging import get_logger
from ...pipeline import PipelineOrchestrator
from ...pipeline.factories import (
    TrainingPipelineFactory,
    EvaluationPipelineFactory,
    DeploymentPipelineFactory
)
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Register pipeline commands."""
    
    @main.group()
    def pipeline() -> None:
        """Pipeline orchestration and management commands."""
        pass
    
    @pipeline.command()
    @click.option("--model-type", type=click.Choice(["dual", "mapper", "analyst"]), default="dual")
    @click.option("--output-dir", type=str, help="Output directory for training artifacts")
    @click.option("--config-file", type=str, help="Path to training configuration file")
    @click.option("--async", "async_execution", is_flag=True, help="Run training asynchronously")
    @click.pass_context
    def train(ctx: click.Context, model_type: str, output_dir: Optional[str], config_file: Optional[str], async_execution: bool) -> None:
        """Execute training pipeline."""
        logger = get_logger(__name__)
        
        async def run_training():
            orchestrator = PipelineOrchestrator()
            factory = TrainingPipelineFactory()
            
            # Build configuration
            config = {
                "model_type": model_type,
                "output_dir": output_dir or f"./training_output_{model_type}",
            }
            
            if config_file:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            
            # Create and execute pipeline
            pipeline = await factory.create_training_pipeline(model_type, config)
            await orchestrator.register_pipeline(pipeline)
            
            click.echo(f"🚀 Starting {model_type} training pipeline...")
            
            result = await orchestrator.execute_pipeline(
                pipeline.name,
                config=config,
                async_execution=async_execution
            )
            
            if async_execution:
                click.echo("✅ Training started asynchronously")
            else:
                click.echo("✅ Training completed successfully!")
        
        try:
            asyncio.run(run_training())
        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise click.ClickException(f"Training failed: {str(e)}")
    
    @pipeline.command()
    @click.argument("model_name")
    @click.option("--model-path", type=str, help="Path to model checkpoint")
    @click.option("--test-data", type=str, help="Path to test data")
    @click.option("--output-dir", type=str, help="Output directory for evaluation results")
    @click.pass_context
    def evaluate(ctx: click.Context, model_name: str, model_path: Optional[str], test_data: Optional[str], output_dir: Optional[str]) -> None:
        """Execute evaluation pipeline."""
        logger = get_logger(__name__)
        
        async def run_evaluation():
            orchestrator = PipelineOrchestrator()
            factory = EvaluationPipelineFactory()
            
            config = {
                "model_name": model_name,
                "model_path": model_path,
                "test_data_path": test_data,
                "output_dir": output_dir or f"./evaluation_output_{model_name}"
            }
            
            pipeline = await factory.create_evaluation_pipeline(model_name, config)
            await orchestrator.register_pipeline(pipeline)
            
            click.echo(f"🎯 Starting evaluation for {model_name}...")
            
            await orchestrator.execute_pipeline(pipeline.name, config=config)
            
            click.echo("✅ Evaluation completed successfully!")
        
        try:
            asyncio.run(run_evaluation())
        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise click.ClickException(f"Evaluation failed: {str(e)}")
    
    @pipeline.command()
    @click.argument("model_name")
    @click.argument("version")
    @click.option("--environment", type=click.Choice(["staging", "production"]), default="staging")
    @click.option("--canary-percentage", type=float, default=5.0, help="Percentage of traffic for canary")
    @click.pass_context
    def deploy(ctx: click.Context, model_name: str, version: str, environment: str, canary_percentage: float) -> None:
        """Execute deployment pipeline."""
        logger = get_logger(__name__)
        
        async def run_deployment():
            orchestrator = PipelineOrchestrator()
            factory = DeploymentPipelineFactory()
            
            config = {
                "model_name": model_name,
                "model_version": version,
                "deployment_environment": environment,
                "canary_percentage": canary_percentage
            }
            
            pipeline = await factory.create_deployment_pipeline(model_name, version, config)
            await orchestrator.register_pipeline(pipeline)
            
            click.echo(f"🚀 Starting deployment for {model_name}@{version}...")
            
            await orchestrator.execute_pipeline(pipeline.name, config=config)
            
            click.echo("✅ Deployment completed successfully!")
        
        try:
            asyncio.run(run_deployment())
        except Exception as e:
            logger.error("Deployment failed", error=str(e))
            raise click.ClickException(f"Deployment failed: {str(e)}")
    
    @pipeline.command()
    def list() -> None:
        """List all registered pipelines."""
        async def list_pipelines():
            orchestrator = PipelineOrchestrator()
            pipelines = await orchestrator.list_pipelines()
            
            if not pipelines:
                click.echo("No pipelines registered")
                return
            
            click.echo("Registered Pipelines:")
            for pipeline_name in pipelines:
                click.echo(f"  • {pipeline_name}")
        
        asyncio.run(list_pipelines())
    
    @pipeline.command()
    @click.argument("pipeline_name")
    def status(pipeline_name: str) -> None:
        """Get status of a pipeline."""
        async def get_status():
            orchestrator = PipelineOrchestrator()
            status = await orchestrator.get_pipeline_status(pipeline_name)
            
            click.echo(f"Pipeline: {pipeline_name}")
            click.echo(f"Status: {status['status']}")
            click.echo(f"Active: {status['is_active']}")
        
        asyncio.run(get_status())
    
    @pipeline.command()
    @click.argument("pipeline_name")
    def metrics(pipeline_name: str) -> None:
        """Get metrics for a pipeline."""
        async def get_metrics():
            orchestrator = PipelineOrchestrator()
            metrics = await orchestrator.get_pipeline_metrics(pipeline_name)
            
            click.echo(f"Metrics for {pipeline_name}:")
            click.echo(json.dumps(metrics, indent=2))
        
        asyncio.run(get_metrics())
    
    @pipeline.command()
    def active() -> None:
        """List active pipeline executions."""
        async def list_active():
            orchestrator = PipelineOrchestrator()
            active = await orchestrator.list_active_executions()
            
            if not active:
                click.echo("No active pipeline executions")
                return
            
            click.echo("Active Pipeline Executions:")
            for pipeline_name in active:
                click.echo(f"  • {pipeline_name}")
        
        asyncio.run(list_active())