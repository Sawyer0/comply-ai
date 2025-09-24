"""
Core Pipeline Architecture

Clean, maintainable pipeline system with proper separation of concerns.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import structlog

from ..exceptions import PipelineError, StageError
from ..monitoring.pipeline_monitor import PipelineMonitor

logger = structlog.get_logger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """
    Immutable context passed between pipeline stages.
    
    Maintains state and artifacts throughout pipeline execution.
    """
    pipeline_id: str
    run_id: str
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def with_artifact(self, key: str, value: Any) -> PipelineContext:
        """Create new context with additional artifact (immutable)."""
        new_artifacts = self.artifacts.copy()
        new_artifacts[key] = value
        
        return PipelineContext(
            pipeline_id=self.pipeline_id,
            run_id=self.run_id,
            config=self.config,
            artifacts=new_artifacts,
            metadata=self.metadata,
            created_at=self.created_at
        )
    
    def with_metadata(self, key: str, value: Any) -> PipelineContext:
        """Create new context with additional metadata (immutable)."""
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        
        return PipelineContext(
            pipeline_id=self.pipeline_id,
            run_id=self.run_id,
            config=self.config,
            artifacts=self.artifacts,
            metadata=new_metadata,
            created_at=self.created_at
        )


@dataclass
class StageResult:
    """Result of stage execution."""
    stage_name: str
    status: StageStatus
    context: PipelineContext
    duration_seconds: float
    error: Optional[Exception] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Implements the Template Method pattern for consistent stage execution.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger.bind(stage=name)
        self._monitor: Optional[PipelineMonitor] = None
    
    def set_monitor(self, monitor: PipelineMonitor) -> None:
        """Set pipeline monitor for metrics collection."""
        self._monitor = monitor
    
    async def execute(self, context: PipelineContext) -> StageResult:
        """
        Execute stage with proper error handling and monitoring.
        
        Template method that handles common concerns:
        - Validation
        - Monitoring
        - Error handling
        - Logging
        """
        start_time = datetime.now()
        stage_logs = []
        
        try:
            self.logger.info("Starting stage execution", 
                           pipeline_id=context.pipeline_id,
                           run_id=context.run_id)
            
            # Pre-execution validation
            await self._validate_preconditions(context)
            
            # Execute stage-specific logic
            updated_context = await self._execute_stage(context)
            
            # Post-execution validation
            await self._validate_postconditions(updated_context)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                context=updated_context,
                duration_seconds=duration,
                logs=stage_logs,
                metrics=await self._collect_metrics(updated_context)
            )
            
            self.logger.info("Stage completed successfully",
                           duration_seconds=duration)
            
            if self._monitor:
                await self._monitor.record_stage_completion(self.name, duration, True)
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.error("Stage execution failed",
                            error=str(e),
                            duration_seconds=duration)
            
            if self._monitor:
                await self._monitor.record_stage_completion(self.name, duration, False)
            
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                context=context,
                duration_seconds=duration,
                error=StageError(f"Stage {self.name} failed: {str(e)}"),
                logs=stage_logs
            )
    
    @abstractmethod
    async def _execute_stage(self, context: PipelineContext) -> PipelineContext:
        """Execute stage-specific logic. Must be implemented by subclasses."""
        pass
    
    async def _validate_preconditions(self, context: PipelineContext) -> None:
        """Validate preconditions before stage execution."""
        # Default implementation - can be overridden
        pass
    
    async def _validate_postconditions(self, context: PipelineContext) -> None:
        """Validate postconditions after stage execution."""
        # Default implementation - can be overridden
        pass
    
    async def _collect_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        """Collect stage-specific metrics."""
        # Default implementation - can be overridden
        return {}
    
    def can_skip(self, context: PipelineContext) -> bool:
        """Determine if stage can be skipped based on context."""
        return False
    
    def get_dependencies(self) -> List[str]:
        """Get list of stage names this stage depends on."""
        return []


class Pipeline:
    """
    Pipeline orchestrates execution of multiple stages.
    
    Provides:
    - Stage dependency resolution
    - Parallel execution where possible
    - Error handling and recovery
    - Progress monitoring
    """
    
    def __init__(self, 
                 name: str,
                 stages: List[PipelineStage],
                 config: Optional[Dict[str, Any]] = None,
                 monitor: Optional[PipelineMonitor] = None):
        self.name = name
        self.stages = {stage.name: stage for stage in stages}
        self.config = config or {}
        self.monitor = monitor
        self.logger = logger.bind(pipeline=name)
        
        # Set monitor on all stages
        if monitor:
            for stage in stages:
                stage.set_monitor(monitor)
        
        # Validate stage dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that all stage dependencies exist."""
        for stage in self.stages.values():
            for dep in stage.get_dependencies():
                if dep not in self.stages:
                    raise PipelineError(
                        f"Stage {stage.name} depends on {dep} which doesn't exist"
                    )
    
    async def execute(self, 
                     config: Optional[Dict[str, Any]] = None,
                     context: Optional[PipelineContext] = None) -> PipelineContext:
        """
        Execute pipeline with dependency resolution and parallel execution.
        """
        pipeline_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        
        # Create or update context
        if context is None:
            context = PipelineContext(
                pipeline_id=pipeline_id,
                run_id=run_id,
                config={**self.config, **(config or {})}
            )
        
        self.logger.info("Starting pipeline execution",
                        pipeline_id=pipeline_id,
                        run_id=run_id,
                        stages=list(self.stages.keys()))
        
        if self.monitor:
            await self.monitor.record_pipeline_start(self.name, pipeline_id)
        
        try:
            # Execute stages in dependency order
            execution_plan = self._create_execution_plan()
            final_context = context
            
            for stage_batch in execution_plan:
                # Execute stages in parallel within each batch
                batch_results = await self._execute_stage_batch(stage_batch, final_context)
                
                # Update context with results from all stages in batch
                for result in batch_results:
                    if result.status == StageStatus.FAILED:
                        raise PipelineError(f"Pipeline failed at stage {result.stage_name}")
                    final_context = result.context
            
            self.logger.info("Pipeline completed successfully",
                           pipeline_id=pipeline_id)
            
            if self.monitor:
                await self.monitor.record_pipeline_completion(self.name, pipeline_id, True)
            
            return final_context
            
        except Exception as e:
            self.logger.error("Pipeline execution failed",
                            pipeline_id=pipeline_id,
                            error=str(e))
            
            if self.monitor:
                await self.monitor.record_pipeline_completion(self.name, pipeline_id, False)
            
            raise PipelineError(f"Pipeline {self.name} failed: {str(e)}")
    
    def _create_execution_plan(self) -> List[List[str]]:
        """
        Create execution plan with dependency resolution.
        
        Returns list of stage batches that can be executed in parallel.
        """
        # Topological sort with batching
        in_degree = {}
        graph = {}
        
        # Initialize graph
        for stage_name in self.stages:
            in_degree[stage_name] = 0
            graph[stage_name] = []
        
        # Build dependency graph
        for stage in self.stages.values():
            for dep in stage.get_dependencies():
                graph[dep].append(stage.name)
                in_degree[stage.name] += 1
        
        # Create execution batches
        execution_plan = []
        remaining_stages = set(self.stages.keys())
        
        while remaining_stages:
            # Find stages with no dependencies
            ready_stages = [
                stage for stage in remaining_stages 
                if in_degree[stage] == 0
            ]
            
            if not ready_stages:
                raise PipelineError("Circular dependency detected in pipeline stages")
            
            execution_plan.append(ready_stages)
            
            # Remove ready stages and update dependencies
            for stage in ready_stages:
                remaining_stages.remove(stage)
                for dependent in graph[stage]:
                    in_degree[dependent] -= 1
        
        return execution_plan
    
    async def _execute_stage_batch(self, 
                                  stage_names: List[str], 
                                  context: PipelineContext) -> List[StageResult]:
        """Execute a batch of stages in parallel."""
        tasks = []
        
        for stage_name in stage_names:
            stage = self.stages[stage_name]
            
            # Check if stage can be skipped
            if stage.can_skip(context):
                self.logger.info("Skipping stage", stage=stage_name)
                continue
            
            task = asyncio.create_task(stage.execute(context))
            tasks.append(task)
        
        if not tasks:
            return []
        
        # Wait for all stages in batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        stage_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed stage result
                stage_results.append(StageResult(
                    stage_name=stage_names[i],
                    status=StageStatus.FAILED,
                    context=context,
                    duration_seconds=0,
                    error=result
                ))
            else:
                stage_results.append(result)
        
        return stage_results
    
    def add_stage(self, stage: PipelineStage) -> None:
        """Add stage to pipeline."""
        self.stages[stage.name] = stage
        if self.monitor:
            stage.set_monitor(self.monitor)
        self._validate_dependencies()
    
    def remove_stage(self, stage_name: str) -> None:
        """Remove stage from pipeline."""
        if stage_name in self.stages:
            del self.stages[stage_name]
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        return self.stages.get(stage_name)
    
    def list_stages(self) -> List[str]:
        """List all stage names."""
        return list(self.stages.keys())