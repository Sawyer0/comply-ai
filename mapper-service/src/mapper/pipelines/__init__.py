"""
Pipeline management for the mapper service.

This module provides comprehensive pipeline capabilities including
deployment pipelines and optimization pipelines.
"""

from .deployment import (
    PipelineExecutor,
    PipelineConfig,
    PipelineExecution,
    PipelineStage,
    ValidationGate,
    PipelineStatus,
    StageStatus,
    GateType,
    StageExecution,
    GateValidator,
    get_pipeline_executor,
)

from .optimization import (
    OptimizationPipeline,
    OptimizationConfig,
    OptimizationExecution,
    OptimizationType,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationConstraint,
    OptimizationResult,
    MetricType,
    MetricsCollector,
    OptimizationStrategy,
    BayesianOptimizationStrategy,
    get_optimization_pipeline,
)

__all__ = [
    # Deployment pipelines
    "PipelineExecutor",
    "PipelineConfig",
    "PipelineExecution",
    "PipelineStage",
    "ValidationGate",
    "PipelineStatus",
    "StageStatus",
    "GateType",
    "StageExecution",
    "GateValidator",
    "get_pipeline_executor",
    # Optimization pipelines
    "OptimizationPipeline",
    "OptimizationConfig",
    "OptimizationExecution",
    "OptimizationType",
    "OptimizationStatus",
    "OptimizationTarget",
    "OptimizationConstraint",
    "OptimizationResult",
    "MetricType",
    "MetricsCollector",
    "OptimizationStrategy",
    "BayesianOptimizationStrategy",
    "get_optimization_pipeline",
]
