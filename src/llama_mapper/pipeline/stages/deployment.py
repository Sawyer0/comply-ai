"""
Deployment Pipeline Stage

Orchestrates model deployment using existing production infrastructure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..core.pipeline import PipelineStage, PipelineContext
from ..exceptions import StageError
from ...production.model_versioning import (
    ModelVersionManager, 
    CanaryConfig, 
    CanaryEvaluator,
    KPIMetrics
)

logger = structlog.get_logger(__name__)


class DeploymentStage(PipelineStage):
    """
    Deployment stage that orchestrates model deployment.
    
    Handles:
    - Pre-deployment validation
    - Canary deployment
    - KPI monitoring
    - Production promotion
    """
    
    def __init__(self, name: str = "deployment", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.version_manager = ModelVersionManager()
        self.canary_evaluator = CanaryEvaluator(self.version_manager)
    
    async def _validate_preconditions(self, context: PipelineContext) -> None:
        """Validate that evaluation results are available and models passed quality gates."""
        evaluation_results = context.artifacts.get("evaluation_results")
        if not evaluation_results:
            raise StageError("Evaluation results not found in context")
        
        # Check quality gates
        model_type = evaluation_results["model_type"]
        
        if model_type == "dual":
            mapper_passed = evaluation_results.get("mapper", {}).get("passed_quality_gates", False)
            analyst_passed = evaluation_results.get("analyst", {}).get("passed_quality_gates", False)
            
            if not (mapper_passed and analyst_passed):
                raise StageError("Models failed quality gates - deployment not allowed")
        else:
            if not evaluation_results.get("passed_quality_gates", False):
                raise StageError("Model failed quality gates - deployment not allowed")
        
        # Validate required configuration
        required_keys = ["deployment_environment"]
        for key in required_keys:
            if key not in context.config:
                raise StageError(f"Required configuration key missing: {key}")
    
    async def _execute_stage(self, context: PipelineContext) -> PipelineContext:
        """Execute deployment stage."""
        evaluation_results = context.artifacts["evaluation_results"]
        training_results = context.artifacts["training_results"]
        deployment_env = context.config["deployment_environment"]
        
        self.logger.info("Starting deployment stage",
                        model_type=evaluation_results.get("model_type"),
                        environment=deployment_env)
        
        deployment_results = {}
        
        model_type = evaluation_results["model_type"]
        
        if model_type == "dual":
            # Deploy both models
            mapper_deployment = await self._deploy_model(
                training_results["mapper"],
                evaluation_results["mapper"],
                context
            )
            analyst_deployment = await self._deploy_model(
                training_results["analyst"],
                evaluation_results["analyst"],
                context
            )
            
            deployment_results = {
                "mapper": mapper_deployment,
                "analyst": analyst_deployment,
                "model_type": "dual"
            }
        else:
            # Deploy single model
            deployment_results = await self._deploy_model(
                training_results,
                evaluation_results,
                context
            )
            deployment_results["model_type"] = model_type
        
        # Monitor deployment health
        await self._monitor_deployment_health(deployment_results, context)
        
        self.logger.info("Deployment stage completed",
                        deployment_status=self._get_overall_deployment_status(deployment_results))
        
        return context.with_artifact("deployment_results", deployment_results)
    
    async def _deploy_model(self, 
                           training_results: Dict[str, Any],
                           evaluation_results: Dict[str, Any],
                           context: PipelineContext) -> Dict[str, Any]:
        """Deploy a single model."""
        model_name = training_results["model_name"]
        version = training_results["version"]
        deployment_env = context.config["deployment_environment"]
        
        self.logger.info("Deploying model",
                        model_name=model_name,
                        version=version,
                        environment=deployment_env)
        
        # Step 1: Pre-deployment validation
        validation_results = await self._pre_deployment_validation(
            model_name, version, evaluation_results
        )
        
        if not validation_results["passed"]:
            raise StageError(f"Pre-deployment validation failed: {validation_results['errors']}")
        
        # Step 2: Canary deployment
        canary_results = await self._deploy_canary(model_name, version, context)
        
        # Step 3: Canary evaluation
        canary_evaluation = await self._evaluate_canary(model_name, version, context)
        
        # Step 4: Production promotion (if canary successful)
        promotion_results = None
        if canary_evaluation["should_promote"]:
            promotion_results = await self._promote_to_production(model_name, version, canary_evaluation)
        
        return {
            "model_name": model_name,
            "version": version,
            "deployment_environment": deployment_env,
            "pre_deployment_validation": validation_results,
            "canary_deployment": canary_results,
            "canary_evaluation": canary_evaluation,
            "production_promotion": promotion_results,
            "deployment_status": "promoted" if promotion_results else "canary"
        }
    
    async def _pre_deployment_validation(self, 
                                        model_name: str,
                                        version: str,
                                        evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pre-deployment validation."""
        self.logger.info("Running pre-deployment validation",
                        model_name=model_name,
                        version=version)
        
        validation_errors = []
        
        # Check model files exist
        checkpoint_path = Path(evaluation_results.get("checkpoint_path", ""))
        if not checkpoint_path.exists():
            validation_errors.append(f"Model checkpoint not found: {checkpoint_path}")
        
        # Check evaluation metrics meet minimum thresholds
        metrics = evaluation_results.get("evaluation_metrics", {}).get("overall_metrics", {})
        
        min_thresholds = {
            "accuracy": 0.80,
            "f1_score": 0.75,
            "precision": 0.75,
            "recall": 0.70
        }
        
        for metric, min_value in min_thresholds.items():
            actual_value = metrics.get(metric, 0)
            if actual_value < min_value:
                validation_errors.append(
                    f"{metric} {actual_value:.3f} below minimum threshold {min_value}"
                )
        
        # Check model version exists in registry
        loop = asyncio.get_event_loop()
        model_info = await loop.run_in_executor(
            None,
            self.version_manager.get_model_info,
            model_name,
            version
        )
        
        if not model_info:
            validation_errors.append(f"Model {model_name}@{version} not found in registry")
        
        passed = len(validation_errors) == 0
        
        self.logger.info("Pre-deployment validation completed",
                        model_name=model_name,
                        passed=passed,
                        error_count=len(validation_errors))
        
        return {
            "passed": passed,
            "errors": validation_errors,
            "validation_timestamp": asyncio.get_event_loop().time()
        }
    
    async def _deploy_canary(self, 
                            model_name: str,
                            version: str,
                            context: PipelineContext) -> Dict[str, Any]:
        """Deploy model as canary."""
        canary_percentage = context.config.get("canary_percentage", 5.0)
        evaluation_duration = context.config.get("canary_evaluation_duration", 3600)  # 1 hour
        
        self.logger.info("Deploying canary",
                        model_name=model_name,
                        version=version,
                        percentage=canary_percentage)
        
        # Create canary configuration
        canary_config = CanaryConfig(
            canary_percentage=canary_percentage,
            evaluation_duration=evaluation_duration,
            auto_promote=context.config.get("auto_promote", False)
        )
        
        # Deploy canary using version manager
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self.version_manager.deploy_canary,
            model_name,
            version,
            canary_config,
            context.config.get("baseline_version")
        )
        
        if not success:
            raise StageError(f"Failed to deploy canary for {model_name}@{version}")
        
        # Simulate canary deployment infrastructure setup
        # In a real implementation, this would:
        # 1. Update Kubernetes deployments
        # 2. Configure traffic routing
        # 3. Set up monitoring
        
        return {
            "success": success,
            "canary_percentage": canary_percentage,
            "evaluation_duration": evaluation_duration,
            "deployment_timestamp": asyncio.get_event_loop().time()
        }
    
    async def _evaluate_canary(self, 
                              model_name: str,
                              version: str,
                              context: PipelineContext) -> Dict[str, Any]:
        """Evaluate canary deployment performance."""
        baseline_version = context.config.get("baseline_version")
        evaluation_duration = context.config.get("canary_evaluation_duration", 3600)
        
        self.logger.info("Evaluating canary performance",
                        model_name=model_name,
                        version=version,
                        baseline_version=baseline_version)
        
        # Run canary evaluation
        loop = asyncio.get_event_loop()
        evaluation_result = await loop.run_in_executor(
            None,
            self.canary_evaluator.evaluate_canary,
            model_name,
            version,
            baseline_version or "baseline",
            evaluation_duration
        )
        
        # Determine if canary should be promoted
        should_promote = await loop.run_in_executor(
            None,
            self.canary_evaluator.should_promote_canary,
            model_name,
            version
        )
        
        evaluation_result["should_promote"] = should_promote
        
        self.logger.info("Canary evaluation completed",
                        model_name=model_name,
                        should_promote=should_promote,
                        confidence=evaluation_result.get("confidence", 0))
        
        return evaluation_result
    
    async def _promote_to_production(self, 
                                    model_name: str,
                                    version: str,
                                    canary_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Promote canary to production."""
        self.logger.info("Promoting canary to production",
                        model_name=model_name,
                        version=version)
        
        # Create KPI metrics from canary evaluation
        kpi_comparison = canary_evaluation.get("kpi_comparison", {})
        
        kpi_metrics = KPIMetrics(
            p95_latency=kpi_comparison.get("p95_latency", {}).get("canary", 100.0),
            schema_pass_rate=kpi_comparison.get("schema_pass_rate", {}).get("canary", 0.95),
            f1_score=kpi_comparison.get("f1_score", {}).get("canary", 0.90),
            cache_hit_rate=0.85,  # Would be measured from actual deployment
            error_rate=kpi_comparison.get("error_rate", {}).get("canary", 0.01),
            throughput=100.0,  # Would be measured from actual deployment
            timestamp=""  # Will be set by __post_init__
        )
        
        # Promote to production using version manager
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self.version_manager.promote_to_production,
            model_name,
            version,
            kpi_metrics
        )
        
        if not success:
            raise StageError(f"Failed to promote {model_name}@{version} to production")
        
        # Simulate production deployment infrastructure updates
        # In a real implementation, this would:
        # 1. Update Kubernetes deployments to 100% traffic
        # 2. Update load balancer configurations
        # 3. Update monitoring and alerting
        
        self.logger.info("Production promotion completed",
                        model_name=model_name,
                        version=version)
        
        return {
            "success": success,
            "promotion_timestamp": asyncio.get_event_loop().time(),
            "kpi_metrics": kpi_metrics.__dict__
        }
    
    async def _monitor_deployment_health(self, 
                                        deployment_results: Dict[str, Any],
                                        context: PipelineContext) -> None:
        """Monitor deployment health after completion."""
        model_type = deployment_results["model_type"]
        
        self.logger.info("Monitoring deployment health", model_type=model_type)
        
        # In a real implementation, this would:
        # 1. Set up health checks
        # 2. Configure alerting
        # 3. Monitor key metrics
        # 4. Set up automated rollback triggers
        
        # For now, we'll simulate health monitoring
        await asyncio.sleep(1)  # Simulate monitoring setup time
        
        if model_type == "dual":
            mapper_status = deployment_results["mapper"]["deployment_status"]
            analyst_status = deployment_results["analyst"]["deployment_status"]
            
            self.logger.info("Dual model deployment health",
                           mapper_status=mapper_status,
                           analyst_status=analyst_status)
        else:
            status = deployment_results["deployment_status"]
            self.logger.info("Single model deployment health", status=status)
    
    def _get_overall_deployment_status(self, deployment_results: Dict[str, Any]) -> str:
        """Get overall deployment status."""
        model_type = deployment_results["model_type"]
        
        if model_type == "dual":
            mapper_status = deployment_results["mapper"]["deployment_status"]
            analyst_status = deployment_results["analyst"]["deployment_status"]
            
            if mapper_status == "promoted" and analyst_status == "promoted":
                return "fully_promoted"
            elif mapper_status == "canary" or analyst_status == "canary":
                return "partial_canary"
            else:
                return "unknown"
        else:
            return deployment_results["deployment_status"]
    
    async def _collect_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        """Collect deployment stage metrics."""
        deployment_results = context.artifacts.get("deployment_results", {})
        
        metrics = {
            "model_type": deployment_results.get("model_type"),
            "models_deployed": 0,
            "canary_deployments": 0,
            "production_promotions": 0,
            "overall_status": self._get_overall_deployment_status(deployment_results)
        }
        
        if deployment_results.get("model_type") == "dual":
            metrics["models_deployed"] = 2
            
            for model_key in ["mapper", "analyst"]:
                model_result = deployment_results.get(model_key, {})
                if model_result.get("canary_deployment", {}).get("success"):
                    metrics["canary_deployments"] += 1
                if model_result.get("production_promotion", {}).get("success"):
                    metrics["production_promotions"] += 1
        else:
            metrics["models_deployed"] = 1
            if deployment_results.get("canary_deployment", {}).get("success"):
                metrics["canary_deployments"] = 1
            if deployment_results.get("production_promotion", {}).get("success"):
                metrics["production_promotions"] = 1
        
        return metrics
    
    def get_dependencies(self) -> List[str]:
        """Deployment depends on evaluation."""
        return ["evaluation"]