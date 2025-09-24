"""
Pipeline Orchestrator

High-level orchestrator that manages pipeline lifecycle and provides
a clean interface for pipeline execution and management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from .pipeline import Pipeline, PipelineContext
from .registry import PipelineRegistry
from ..config.pipeline_config import PipelineConfig
from ..exceptions import PipelineError, ConfigurationError
from ..monitoring.pipeline_monitor import PipelineMonitor

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """
    High-level orchestrator for managing pipeline execution.
    
    Provides:
    - Pipeline registration and discovery
    - Configuration management
    - Execution coordination
    - Monitoring and observability
    """
    
    def __init__(self, 
                 registry: Optional[PipelineRegistry] = None,
                 monitor: Optional[PipelineMonitor] = None):
        self.registry = registry or PipelineRegistry()
        self.monitor = monitor or PipelineMonitor()
        self.logger = logger.bind(component="orchestrator")
        
        # Track active pipeline executions
        self._active_executions: Dict[str, asyncio.Task] = {}
    
    async def register_pipeline(self, 
                               pipeline: Pipeline,
                               config: Optional[PipelineConfig] = None) -> None:
        """Register a pipeline with the orchestrator."""
        try:
            await self.registry.register_pipeline(pipeline, config)
            self.logger.info("Pipeline registered successfully", 
                           pipeline_name=pipeline.name)
        except Exception as e:
            self.logger.error("Failed to register pipeline",
                            pipeline_name=pipeline.name,
                            error=str(e))
            raise PipelineError(f"Failed to register pipeline {pipeline.name}: {str(e)}")
    
    async def execute_pipeline(self, 
                              pipeline_name: str,
                              config: Optional[Dict[str, Any]] = None,
                              context: Optional[PipelineContext] = None,
                              async_execution: bool = False) -> PipelineContext:
        """
        Execute a registered pipeline.
        
        Args:
            pipeline_name: Name of pipeline to execute
            config: Optional configuration overrides
            context: Optional initial context
            async_execution: If True, return immediately and execute in background
            
        Returns:
            Final pipeline context
        """
        pipeline = await self.registry.get_pipeline(pipeline_name)
        if not pipeline:
            raise PipelineError(f"Pipeline {pipeline_name} not found")
        
        pipeline_config = await self.registry.get_pipeline_config(pipeline_name)
        
        # Merge configurations
        final_config = {}
        if pipeline_config:
            final_config.update(pipeline_config.to_dict())
        if config:
            final_config.update(config)
        
        self.logger.info("Starting pipeline execution",
                        pipeline_name=pipeline_name,
                        async_execution=async_execution)
        
        if async_execution:
            # Execute in background
            task = asyncio.create_task(
                self._execute_pipeline_with_monitoring(pipeline, final_config, context)
            )
            self._active_executions[pipeline_name] = task
            
            # Return immediately with initial context
            return context or PipelineContext(
                pipeline_id=pipeline_name,
                run_id=f"{pipeline_name}_{datetime.now().isoformat()}",
                config=final_config
            )
        else:
            # Execute synchronously
            return await self._execute_pipeline_with_monitoring(pipeline, final_config, context)
    
    async def _execute_pipeline_with_monitoring(self,
                                               pipeline: Pipeline,
                                               config: Dict[str, Any],
                                               context: Optional[PipelineContext]) -> PipelineContext:
        """Execute pipeline with comprehensive monitoring."""
        try:
            # Set monitor on pipeline
            pipeline.monitor = self.monitor
            
            # Execute pipeline
            result_context = await pipeline.execute(config, context)
            
            self.logger.info("Pipeline execution completed successfully",
                           pipeline_name=pipeline.name)
            
            return result_context
            
        except Exception as e:
            self.logger.error("Pipeline execution failed",
                            pipeline_name=pipeline.name,
                            error=str(e))
            raise
        finally:
            # Clean up active execution tracking
            if pipeline.name in self._active_executions:
                del self._active_executions[pipeline.name]
    
    async def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get status of pipeline execution."""
        if pipeline_name in self._active_executions:
            task = self._active_executions[pipeline_name]
            return {
                "status": "running" if not task.done() else "completed",
                "pipeline_name": pipeline_name,
                "started_at": datetime.now().isoformat(),  # Would track actual start time
                "is_active": not task.done()
            }
        
        # Check registry for pipeline existence
        pipeline = await self.registry.get_pipeline(pipeline_name)
        if pipeline:
            return {
                "status": "registered",
                "pipeline_name": pipeline_name,
                "is_active": False
            }
        
        return {
            "status": "not_found",
            "pipeline_name": pipeline_name,
            "is_active": False
        }
    
    async def cancel_pipeline(self, pipeline_name: str) -> bool:
        """Cancel active pipeline execution."""
        if pipeline_name not in self._active_executions:
            return False
        
        task = self._active_executions[pipeline_name]
        if not task.done():
            task.cancel()
            self.logger.info("Pipeline execution cancelled", 
                           pipeline_name=pipeline_name)
            return True
        
        return False
    
    async def list_pipelines(self) -> List[str]:
        """List all registered pipelines."""
        return await self.registry.list_pipelines()
    
    async def list_active_executions(self) -> List[str]:
        """List currently active pipeline executions."""
        active = []
        for pipeline_name, task in self._active_executions.items():
            if not task.done():
                active.append(pipeline_name)
        return active
    
    async def get_pipeline_metrics(self, pipeline_name: str) -> Dict[str, Any]:
        """Get metrics for a pipeline."""
        return await self.monitor.get_pipeline_metrics(pipeline_name)
    
    async def validate_pipeline_config(self, 
                                      pipeline_name: str,
                                      config: Dict[str, Any]) -> List[str]:
        """
        Validate pipeline configuration.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        try:
            pipeline = await self.registry.get_pipeline(pipeline_name)
            if not pipeline:
                errors.append(f"Pipeline {pipeline_name} not found")
                return errors
            
            # Validate configuration against pipeline requirements
            pipeline_config = await self.registry.get_pipeline_config(pipeline_name)
            if pipeline_config:
                validation_errors = pipeline_config.validate(config)
                errors.extend(validation_errors)
            
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
        
        return errors
    
    async def create_training_pipeline(self, 
                                      model_type: str = "dual",
                                      config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create and register a training pipeline.
        
        Factory method for common training pipeline configurations.
        """
        from ..factories.training_pipeline_factory import TrainingPipelineFactory
        
        factory = TrainingPipelineFactory()
        pipeline = await factory.create_training_pipeline(model_type, config)
        
        await self.register_pipeline(pipeline)
        
        self.logger.info("Training pipeline created and registered",
                        pipeline_name=pipeline.name,
                        model_type=model_type)
        
        return pipeline.name
    
    async def create_evaluation_pipeline(self,
                                        model_name: str,
                                        config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create and register an evaluation pipeline.
        
        Factory method for model evaluation pipelines.
        """
        from ..factories.evaluation_pipeline_factory import EvaluationPipelineFactory
        
        factory = EvaluationPipelineFactory()
        pipeline = await factory.create_evaluation_pipeline(model_name, config)
        
        await self.register_pipeline(pipeline)
        
        self.logger.info("Evaluation pipeline created and registered",
                        pipeline_name=pipeline.name,
                        model_name=model_name)
        
        return pipeline.name
    
    async def create_deployment_pipeline(self,
                                        model_name: str,
                                        version: str,
                                        config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create and register a deployment pipeline.
        
        Factory method for model deployment pipelines.
        """
        from ..factories.deployment_pipeline_factory import DeploymentPipelineFactory
        
        factory = DeploymentPipelineFactory()
        pipeline = await factory.create_deployment_pipeline(model_name, version, config)
        
        await self.register_pipeline(pipeline)
        
        self.logger.info("Deployment pipeline created and registered",
                        pipeline_name=pipeline.name,
                        model_name=model_name,
                        version=version)
        
        return pipeline.name
    
    async def shutdown(self) -> None:
        """Gracefully shutdown orchestrator and cancel active executions."""
        self.logger.info("Shutting down pipeline orchestrator")
        
        # Cancel all active executions
        for pipeline_name in list(self._active_executions.keys()):
            await self.cancel_pipeline(pipeline_name)
        
        # Wait for cancellations to complete
        if self._active_executions:
            await asyncio.gather(
                *self._active_executions.values(),
                return_exceptions=True
            )
        
        self.logger.info("Pipeline orchestrator shutdown complete")