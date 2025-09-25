"""
Optimization pipelines for the mapper service.

This module provides optimization pipeline capabilities including
model optimization, performance tuning, and resource optimization.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel
import json
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization."""

    MODEL_PERFORMANCE = "model_performance"
    RESOURCE_UTILIZATION = "resource_utilization"
    COST_OPTIMIZATION = "cost_optimization"
    LATENCY_OPTIMIZATION = "latency_optimization"
    THROUGHPUT_OPTIMIZATION = "throughput_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    ACCURACY_OPTIMIZATION = "accuracy_optimization"


class OptimizationStatus(Enum):
    """Optimization pipeline status."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Types of metrics to optimize."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ACCURACY = "accuracy"
    COST = "cost"
    ERROR_RATE = "error_rate"


@dataclass
class OptimizationTarget:
    """Optimization target configuration."""

    metric: MetricType
    target_value: float
    current_value: Optional[float] = None
    improvement_threshold: float = 0.1  # 10% improvement
    weight: float = 1.0  # Relative importance


@dataclass
class OptimizationConstraint:
    """Optimization constraint."""

    metric: MetricType
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    hard_constraint: bool = True  # If false, treated as soft constraint


@dataclass
class OptimizationConfig:
    """Configuration for optimization pipeline."""

    optimization_id: str
    name: str
    description: str
    optimization_type: OptimizationType

    # Targets and constraints
    targets: List[OptimizationTarget] = field(default_factory=list)
    constraints: List[OptimizationConstraint] = field(default_factory=list)

    # Optimization parameters
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    timeout_minutes: int = 120

    # Search strategy
    search_strategy: str = "bayesian"  # bayesian, grid, random
    search_space: Dict[str, Any] = field(default_factory=dict)

    # Validation configuration
    validation_samples: int = 1000
    validation_timeout_minutes: int = 30

    # Rollback configuration
    auto_rollback_on_regression: bool = True
    regression_threshold: float = 0.05  # 5% regression

    # Callbacks
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None


class OptimizationResult(BaseModel):
    """Result of optimization attempt."""

    iteration: int
    parameters: Dict[str, Any]
    metrics: Dict[MetricType, float]
    score: float
    improvement: float
    validation_passed: bool
    timestamp: datetime


@dataclass
class OptimizationExecution:
    """Optimization pipeline execution."""

    execution_id: str
    config: OptimizationConfig
    status: OptimizationStatus = OptimizationStatus.PENDING

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Results
    results: List[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    baseline_metrics: Dict[MetricType, float] = field(default_factory=dict)

    # Progress tracking
    current_iteration: int = 0
    convergence_history: List[float] = field(default_factory=list)

    # Error tracking
    error_message: Optional[str] = None
    rollback_executed: bool = False


class MetricsCollector:
    """Collects metrics for optimization."""

    def __init__(self):
        self._collectors: Dict[MetricType, Callable] = {}
        self._register_default_collectors()

    def register_collector(self, metric_type: MetricType, collector: Callable):
        """Register a metrics collector."""
        self._collectors[metric_type] = collector
        logger.debug(f"Registered metrics collector: {metric_type.value}")

    async def collect_metrics(
        self, parameters: Dict[str, Any], metric_types: List[MetricType]
    ) -> Dict[MetricType, float]:
        """Collect specified metrics."""
        metrics = {}

        for metric_type in metric_types:
            collector = self._collectors.get(metric_type)
            if collector:
                try:
                    value = await collector(parameters)
                    metrics[metric_type] = float(value)
                except Exception as e:
                    logger.error(f"Failed to collect metric {metric_type.value}: {e}")
                    metrics[metric_type] = float("inf")  # Worst possible value
            else:
                logger.warning(
                    f"No collector registered for metric: {metric_type.value}"
                )
                metrics[metric_type] = float("inf")

        return metrics

    def _register_default_collectors(self):
        """Register default metrics collectors."""

        async def latency_collector(parameters: Dict[str, Any]) -> float:
            """Collect latency metrics."""
            # Simulate latency measurement
            await asyncio.sleep(0.1)

            # In a real implementation, this would measure actual latency
            base_latency = 100.0  # ms

            # Simulate parameter effects on latency
            batch_size = parameters.get("batch_size", 1)
            model_size = parameters.get("model_size", "medium")

            latency = base_latency
            if batch_size > 1:
                latency *= 0.8  # Batching improves latency per item

            if model_size == "large":
                latency *= 1.5
            elif model_size == "small":
                latency *= 0.7

            return latency

        async def throughput_collector(parameters: Dict[str, Any]) -> float:
            """Collect throughput metrics."""
            # Simulate throughput measurement
            await asyncio.sleep(0.1)

            # In a real implementation, this would measure actual throughput
            base_throughput = 100.0  # requests/second

            batch_size = parameters.get("batch_size", 1)
            parallel_workers = parameters.get("parallel_workers", 1)

            throughput = base_throughput * batch_size * parallel_workers * 0.8

            return throughput

        async def memory_collector(parameters: Dict[str, Any]) -> float:
            """Collect memory usage metrics."""
            # Simulate memory measurement
            await asyncio.sleep(0.05)

            base_memory = 1024.0  # MB

            model_size = parameters.get("model_size", "medium")
            batch_size = parameters.get("batch_size", 1)

            memory = base_memory
            if model_size == "large":
                memory *= 2.0
            elif model_size == "small":
                memory *= 0.5

            memory *= batch_size * 0.8  # Sublinear scaling

            return memory

        async def accuracy_collector(parameters: Dict[str, Any]) -> float:
            """Collect accuracy metrics."""
            # Simulate accuracy measurement
            await asyncio.sleep(0.2)

            # In a real implementation, this would run validation tests
            base_accuracy = 0.85

            model_size = parameters.get("model_size", "medium")
            learning_rate = parameters.get("learning_rate", 0.001)

            accuracy = base_accuracy
            if model_size == "large":
                accuracy += 0.05
            elif model_size == "small":
                accuracy -= 0.03

            # Learning rate effects
            if learning_rate > 0.01:
                accuracy -= 0.02  # Too high learning rate
            elif learning_rate < 0.0001:
                accuracy -= 0.01  # Too low learning rate

            return min(1.0, max(0.0, accuracy))

        self._collectors[MetricType.LATENCY] = latency_collector
        self._collectors[MetricType.THROUGHPUT] = throughput_collector
        self._collectors[MetricType.MEMORY_USAGE] = memory_collector
        self._collectors[MetricType.ACCURACY] = accuracy_collector


class OptimizationStrategy:
    """Base class for optimization strategies."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.search_space = config.search_space

    async def suggest_parameters(
        self, iteration: int, previous_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Suggest next parameters to try."""
        raise NotImplementedError

    def calculate_score(self, metrics: Dict[MetricType, float]) -> float:
        """Calculate optimization score from metrics."""
        score = 0.0
        total_weight = 0.0

        for target in self.config.targets:
            if target.metric in metrics:
                metric_value = metrics[target.metric]

                # Calculate improvement (lower is better for latency, memory, etc.)
                if target.metric in [
                    MetricType.LATENCY,
                    MetricType.MEMORY_USAGE,
                    MetricType.CPU_USAGE,
                    MetricType.COST,
                    MetricType.ERROR_RATE,
                ]:
                    # Lower is better
                    if target.current_value:
                        improvement = (
                            target.current_value - metric_value
                        ) / target.current_value
                    else:
                        improvement = 1.0 / (1.0 + metric_value)  # Normalize
                else:
                    # Higher is better (accuracy, throughput)
                    if target.current_value:
                        improvement = (
                            metric_value - target.current_value
                        ) / target.current_value
                    else:
                        improvement = metric_value

                score += improvement * target.weight
                total_weight += target.weight

        return score / total_weight if total_weight > 0 else 0.0

    def check_constraints(
        self, metrics: Dict[MetricType, float]
    ) -> Tuple[bool, List[str]]:
        """Check if metrics satisfy constraints."""
        violations = []

        for constraint in self.config.constraints:
            if constraint.metric not in metrics:
                continue

            metric_value = metrics[constraint.metric]

            if constraint.max_value is not None and metric_value > constraint.max_value:
                violations.append(f"{constraint.metric.value} > {constraint.max_value}")

            if constraint.min_value is not None and metric_value < constraint.min_value:
                violations.append(f"{constraint.metric.value} < {constraint.min_value}")

        # Only hard constraints cause failure
        hard_violations = [
            v for v, c in zip(violations, self.config.constraints) if c.hard_constraint
        ]

        return len(hard_violations) == 0, violations


class BayesianOptimizationStrategy(OptimizationStrategy):
    """Bayesian optimization strategy."""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.gp_model = None  # Gaussian Process model

    async def suggest_parameters(
        self, iteration: int, previous_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Suggest parameters using Bayesian optimization."""
        if iteration == 0 or len(previous_results) < 3:
            # Random exploration for first few iterations
            return self._random_parameters()

        # Use Bayesian optimization (simplified implementation)
        return self._bayesian_suggest(previous_results)

    def _random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within search space."""
        parameters = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "float":
                min_val = param_config.get("min", 0.0)
                max_val = param_config.get("max", 1.0)
                parameters[param_name] = np.random.uniform(min_val, max_val)

            elif param_type == "int":
                min_val = param_config.get("min", 1)
                max_val = param_config.get("max", 10)
                parameters[param_name] = np.random.randint(min_val, max_val + 1)

            elif param_type == "choice":
                choices = param_config.get("choices", [])
                if choices:
                    parameters[param_name] = np.random.choice(choices)

        return parameters

    def _bayesian_suggest(
        self, previous_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Suggest parameters using Bayesian optimization (simplified)."""
        # In a real implementation, this would use a proper Gaussian Process
        # For now, we'll use a simple heuristic based on best results

        # Find best result
        best_result = max(previous_results, key=lambda r: r.score)

        # Add some noise to best parameters
        parameters = {}
        for param_name, param_value in best_result.parameters.items():
            param_config = self.search_space.get(param_name, {})
            param_type = param_config.get("type", "float")

            if param_type == "float":
                noise_scale = (
                    param_config.get("max", 1.0) - param_config.get("min", 0.0)
                ) * 0.1
                noise = np.random.normal(0, noise_scale)
                new_value = param_value + noise

                # Clip to bounds
                min_val = param_config.get("min", 0.0)
                max_val = param_config.get("max", 1.0)
                parameters[param_name] = np.clip(new_value, min_val, max_val)

            elif param_type == "int":
                # Small random change
                change = np.random.randint(-2, 3)
                new_value = param_value + change

                min_val = param_config.get("min", 1)
                max_val = param_config.get("max", 10)
                parameters[param_name] = np.clip(new_value, min_val, max_val)

            else:
                parameters[param_name] = param_value

        return parameters


class OptimizationPipeline:
    """Main optimization pipeline executor."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self._active_executions: Dict[str, OptimizationExecution] = {}
        self._strategies: Dict[str, type] = {
            "bayesian": BayesianOptimizationStrategy,
            "random": OptimizationStrategy,  # Base class does random search
        }

    def register_strategy(self, name: str, strategy_class: type):
        """Register an optimization strategy."""
        self._strategies[name] = strategy_class
        logger.debug(f"Registered optimization strategy: {name}")

    async def execute_optimization(
        self, config: OptimizationConfig
    ) -> OptimizationExecution:
        """Execute an optimization pipeline."""
        execution_id = f"{config.optimization_id}_{datetime.utcnow().timestamp()}"

        execution = OptimizationExecution(
            execution_id=execution_id,
            config=config,
            status=OptimizationStatus.ANALYZING,
            start_time=datetime.utcnow(),
        )

        self._active_executions[execution_id] = execution

        try:
            # Collect baseline metrics
            await self._collect_baseline_metrics(execution)

            # Initialize optimization strategy
            strategy_class = self._strategies.get(
                config.search_strategy, OptimizationStrategy
            )
            strategy = strategy_class(config)

            # Execute optimization loop
            execution.status = OptimizationStatus.OPTIMIZING
            await self._optimization_loop(execution, strategy)

            # Validate best result
            execution.status = OptimizationStatus.VALIDATING
            await self._validate_best_result(execution)

            execution.status = OptimizationStatus.COMPLETED

            # Execute completion callback
            if config.completion_callback:
                await config.completion_callback(execution)

            logger.info(f"Optimization completed: {execution_id}")

        except Exception as e:
            logger.error(f"Optimization failed: {execution_id}: {e}")
            execution.status = OptimizationStatus.FAILED
            execution.error_message = str(e)

        finally:
            execution.end_time = datetime.utcnow()
            if execution.start_time:
                execution.duration_minutes = (
                    execution.end_time - execution.start_time
                ).total_seconds() / 60

        return execution

    async def _collect_baseline_metrics(self, execution: OptimizationExecution):
        """Collect baseline metrics before optimization."""
        # Use default parameters as baseline
        default_params = {}
        for param_name, param_config in execution.config.search_space.items():
            default_params[param_name] = param_config.get("default")

        # Collect all target metrics
        target_metrics = [target.metric for target in execution.config.targets]
        baseline_metrics = await self.metrics_collector.collect_metrics(
            default_params, target_metrics
        )

        execution.baseline_metrics = baseline_metrics

        # Update target current values
        for target in execution.config.targets:
            if target.metric in baseline_metrics:
                target.current_value = baseline_metrics[target.metric]

        logger.info(f"Baseline metrics collected: {baseline_metrics}")

    async def _optimization_loop(
        self, execution: OptimizationExecution, strategy: OptimizationStrategy
    ):
        """Main optimization loop."""
        convergence_window = 10

        for iteration in range(execution.config.max_iterations):
            execution.current_iteration = iteration

            # Suggest parameters
            parameters = await strategy.suggest_parameters(iteration, execution.results)

            # Collect metrics
            target_metrics = [target.metric for target in execution.config.targets]
            constraint_metrics = [
                constraint.metric for constraint in execution.config.constraints
            ]
            all_metrics = list(set(target_metrics + constraint_metrics))

            metrics = await self.metrics_collector.collect_metrics(
                parameters, all_metrics
            )

            # Check constraints
            constraints_satisfied, violations = strategy.check_constraints(metrics)

            # Calculate score
            score = strategy.calculate_score(metrics)

            # Calculate improvement over baseline
            improvement = 0.0
            if execution.baseline_metrics:
                baseline_score = strategy.calculate_score(execution.baseline_metrics)
                improvement = (
                    (score - baseline_score) / abs(baseline_score)
                    if baseline_score != 0
                    else score
                )

            # Create result
            result = OptimizationResult(
                iteration=iteration,
                parameters=parameters,
                metrics=metrics,
                score=score,
                improvement=improvement,
                validation_passed=constraints_satisfied,
                timestamp=datetime.utcnow(),
            )

            execution.results.append(result)

            # Update best result
            if constraints_satisfied and (
                execution.best_result is None or score > execution.best_result.score
            ):
                execution.best_result = result

            # Track convergence
            execution.convergence_history.append(score)

            # Check convergence
            if len(execution.convergence_history) >= convergence_window:
                recent_scores = execution.convergence_history[-convergence_window:]
                score_variance = np.var(recent_scores)

                if score_variance < execution.config.convergence_threshold:
                    logger.info(
                        f"Optimization converged after {iteration + 1} iterations"
                    )
                    break

            # Execute progress callback
            if execution.config.progress_callback:
                await execution.config.progress_callback(execution, result)

            logger.debug(
                f"Optimization iteration {iteration}: score={score:.4f}, "
                f"improvement={improvement:.4f}, constraints_ok={constraints_satisfied}"
            )

    async def _validate_best_result(self, execution: OptimizationExecution):
        """Validate the best optimization result."""
        if not execution.best_result:
            raise RuntimeError("No valid optimization result found")

        # Run validation with more samples
        validation_metrics = []

        for _ in range(execution.config.validation_samples):
            target_metrics = [target.metric for target in execution.config.targets]
            metrics = await self.metrics_collector.collect_metrics(
                execution.best_result.parameters, target_metrics
            )
            validation_metrics.append(metrics)

        # Calculate average metrics
        avg_metrics = {}
        for metric_type in validation_metrics[0].keys():
            values = [m[metric_type] for m in validation_metrics]
            avg_metrics[metric_type] = np.mean(values)

        # Check for regression
        baseline_score = execution.best_result.score
        validation_score = execution.config.__class__(
            optimization_id="validation",
            name="validation",
            description="validation",
            optimization_type=execution.config.optimization_type,
            targets=execution.config.targets,
        )

        strategy = OptimizationStrategy(validation_score)
        validation_score = strategy.calculate_score(avg_metrics)

        regression = (
            (baseline_score - validation_score) / baseline_score
            if baseline_score != 0
            else 0
        )

        if (
            execution.config.auto_rollback_on_regression
            and regression > execution.config.regression_threshold
        ):
            execution.rollback_executed = True
            raise RuntimeError(
                f"Validation failed: {regression:.2%} regression detected"
            )

        logger.info(
            f"Validation passed: average score={validation_score:.4f}, "
            f"regression={regression:.2%}"
        )

    def get_execution_status(
        self, execution_id: str
    ) -> Optional[OptimizationExecution]:
        """Get status of an optimization execution."""
        return self._active_executions.get(execution_id)

    def list_active_executions(self) -> List[OptimizationExecution]:
        """List all active optimization executions."""
        return list(self._active_executions.values())

    async def cancel_optimization(self, execution_id: str) -> bool:
        """Cancel an optimization execution."""
        execution = self._active_executions.get(execution_id)
        if not execution:
            return False

        execution.status = OptimizationStatus.CANCELLED
        execution.end_time = datetime.utcnow()

        logger.info(f"Optimization cancelled: {execution_id}")
        return True


# Global optimization pipeline instance
_optimization_pipeline: Optional[OptimizationPipeline] = None


def get_optimization_pipeline() -> OptimizationPipeline:
    """Get the global optimization pipeline instance."""
    global _optimization_pipeline
    if _optimization_pipeline is None:
        _optimization_pipeline = OptimizationPipeline()
    return _optimization_pipeline
