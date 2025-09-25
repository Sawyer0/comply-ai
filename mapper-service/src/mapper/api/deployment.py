"""
API endpoints for deployment management.

This module provides REST API endpoints for managing deployments,
feature flags, and optimization pipelines.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Import shared components
from shared.interfaces.base import BaseRequest, BaseResponse
from shared.utils.logging import get_logger
from shared.utils.correlation import get_correlation_id
from shared.utils.metrics import track_request_metrics
from shared.exceptions.base import ValidationError, ServiceUnavailableError

from ..deployment import (
    get_deployment_manager,
    get_canary_controller,
    get_blue_green_controller,
    get_feature_flag_manager,
    CanaryConfig,
    BlueGreenConfig,
    EvaluationContext,
    FlagType,
)
from ..pipelines import get_pipeline_executor, get_optimization_pipeline
from ..plugins import get_plugin_manager
from ..tenancy import get_tenant_service

logger = get_logger(__name__)

router = APIRouter(prefix="/deployment", tags=["deployment"])


# Authentication dependency
async def get_current_user(api_key: str = Depends(lambda: "dummy_key")):
    """Get current authenticated user - placeholder implementation."""
    # In a real implementation, this would validate the API key
    # and return user information
    return {"user_id": "system", "tenant_id": "default"}


# Authorization dependency
async def require_admin_access(user: dict = Depends(get_current_user)):
    """Require admin access for management operations."""
    # In a real implementation, this would check user permissions
    if not user.get("is_admin", True):  # Default to True for now
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# Request/Response Models
class PluginStatusResponse(BaseResponse):
    """Plugin status response."""

    plugin_id: str
    name: str
    version: str
    plugin_type: str
    enabled: bool


class CanaryDeploymentRequest(BaseRequest):
    """Canary deployment request."""

    deployment_id: str
    service_name: str
    canary_version: str
    stable_version: str
    initial_traffic_percent: float = 5.0
    max_traffic_percent: float = 100.0
    traffic_increment: float = 25.0
    promotion_interval_minutes: int = 10


class BlueGreenDeploymentRequest(BaseModel):
    """Blue-green deployment request."""

    deployment_id: str
    service_name: str
    new_version: str
    current_version: str
    validation_tests: List[str] = Field(default=["health_check", "smoke_test"])
    validation_timeout_minutes: int = Field(default=30, gt=0)


class FeatureFlagRequest(BaseModel):
    """Feature flag creation/update request."""

    flag_id: str
    name: str
    description: str
    flag_type: str = Field(default="boolean")
    default_value: Any
    enabled: bool = Field(default=False)
    current_value: Optional[Any] = None


class FlagEvaluationRequest(BaseModel):
    """Feature flag evaluation request."""

    flag_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    user_attributes: Dict[str, Any] = Field(default_factory=dict)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    """Optimization pipeline request."""

    optimization_id: str
    name: str
    description: str
    optimization_type: str
    max_iterations: int = Field(default=100, gt=0)
    timeout_minutes: int = Field(default=120, gt=0)
    search_strategy: str = Field(default="bayesian")


# Plugin Management Endpoints
@router.get("/plugins", response_model=List[PluginStatusResponse])
async def list_plugins():
    """List all registered plugins."""
    plugin_manager = get_plugin_manager()
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin manager not available")

    plugins = plugin_manager.get_available_plugins()

    return [
        PluginStatusResponse(
            plugin_id=plugin.plugin_id,
            name=plugin.name,
            version=plugin.version,
            plugin_type=plugin.plugin_type.value,
            enabled=True,  # All registered plugins are considered enabled
        )
        for plugin in plugins
    ]


@router.get("/plugins/{plugin_id}")
async def get_plugin_details(plugin_id: str):
    """Get detailed information about a specific plugin."""
    plugin_manager = get_plugin_manager()
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin manager not available")

    plugin = plugin_manager.registry.get_plugin(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")

    metadata = plugin.get_metadata()

    return {
        "plugin_id": metadata.plugin_id,
        "name": metadata.name,
        "version": metadata.version,
        "author": metadata.author,
        "description": metadata.description,
        "plugin_type": metadata.plugin_type.value,
        "capabilities": [
            {
                "name": cap.name,
                "version": cap.version,
                "description": cap.description,
                "supported_formats": cap.supported_formats,
            }
            for cap in metadata.capabilities
        ],
        "dependencies": metadata.dependencies,
        "config_schema": metadata.config_schema,
    }


# Canary Deployment Endpoints
@router.post("/canary")
@track_request_metrics("start_canary_deployment")
async def start_canary_deployment(
    request: CanaryDeploymentRequest, current_user: dict = Depends(require_admin_access)
):
    """Start a new canary deployment."""
    correlation_id = get_correlation_id()
    logger.info(
        "Starting canary deployment",
        deployment_id=request.deployment_id,
        correlation_id=correlation_id,
    )

    # Validate request
    if request.initial_traffic_percent < 0 or request.initial_traffic_percent > 100:
        raise ValidationError("Initial traffic percent must be between 0 and 100")

    canary_controller = get_canary_controller()
    if not canary_controller:
        raise ServiceUnavailableError("Canary controller not available")

    try:
        config = CanaryConfig(
            deployment_id=request.deployment_id,
            service_name=request.service_name,
            canary_version=request.canary_version,
            stable_version=request.stable_version,
            initial_traffic_percent=request.initial_traffic_percent,
            max_traffic_percent=request.max_traffic_percent,
            traffic_increment=request.traffic_increment,
            promotion_interval_minutes=request.promotion_interval_minutes,
        )

        deployment = await canary_controller.start_canary_deployment(config)

        return {
            "deployment_id": deployment.config.deployment_id,
            "status": deployment.status.value,
            "current_traffic_percent": deployment.current_traffic_percent,
            "start_time": deployment.start_time,
        }

    except ServiceUnavailableError as e:
        logger.error(
            "Canary controller unavailable", error=str(e), correlation_id=correlation_id
        )
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValidationError as e:
        logger.error(
            "Canary deployment validation failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Failed to start canary deployment",
            error=str(e),
            correlation_id=correlation_id,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/canary/{deployment_id}")
async def get_canary_status(deployment_id: str):
    """Get status of a canary deployment."""
    canary_controller = get_canary_controller()

    deployment = canary_controller.get_deployment_status(deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=404, detail=f"Canary deployment {deployment_id} not found"
        )

    return {
        "deployment_id": deployment.config.deployment_id,
        "status": deployment.status.value,
        "current_traffic_percent": deployment.current_traffic_percent,
        "start_time": deployment.start_time,
        "end_time": deployment.end_time,
        "failure_count": deployment.failure_count,
        "promotion_count": deployment.promotion_count,
        "health_checks": len(deployment.health_checks),
        "last_error": deployment.last_error,
        "rollback_reason": deployment.rollback_reason,
    }


@router.post("/canary/{deployment_id}/promote")
async def promote_canary(deployment_id: str):
    """Promote canary deployment to full traffic."""
    canary_controller = get_canary_controller()

    try:
        success = await canary_controller.promote_canary(deployment_id)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to promote canary deployment"
            )

        return {"message": f"Canary deployment {deployment_id} promotion started"}

    except Exception as e:
        logger.error(f"Failed to promote canary deployment {deployment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/canary/{deployment_id}/rollback")
async def rollback_canary(deployment_id: str, reason: str = "Manual rollback"):
    """Rollback canary deployment."""
    canary_controller = get_canary_controller()

    try:
        success = await canary_controller.rollback_canary(deployment_id, reason)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to rollback canary deployment"
            )

        return {"message": f"Canary deployment {deployment_id} rollback initiated"}

    except Exception as e:
        logger.error(f"Failed to rollback canary deployment {deployment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Blue-Green Deployment Endpoints
@router.post("/blue-green")
async def start_blue_green_deployment(request: BlueGreenDeploymentRequest):
    """Start a new blue-green deployment."""
    blue_green_controller = get_blue_green_controller()

    try:
        config = BlueGreenConfig(
            deployment_id=request.deployment_id,
            service_name=request.service_name,
            new_version=request.new_version,
            current_version=request.current_version,
            validation_tests=request.validation_tests,
            validation_timeout_minutes=request.validation_timeout_minutes,
        )

        deployment = await blue_green_controller.start_blue_green_deployment(config)

        return {
            "deployment_id": deployment.config.deployment_id,
            "status": deployment.status.value,
            "active_environment": deployment.active_environment.value,
            "target_environment": deployment.target_environment.value,
            "start_time": deployment.start_time,
        }

    except Exception as e:
        logger.error(f"Failed to start blue-green deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/blue-green/{deployment_id}")
async def get_blue_green_status(deployment_id: str):
    """Get status of a blue-green deployment."""
    blue_green_controller = get_blue_green_controller()

    deployment = blue_green_controller.get_deployment_status(deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=404, detail=f"Blue-green deployment {deployment_id} not found"
        )

    return {
        "deployment_id": deployment.config.deployment_id,
        "status": deployment.status.value,
        "active_environment": deployment.active_environment.value,
        "target_environment": deployment.target_environment.value,
        "start_time": deployment.start_time,
        "switch_time": deployment.switch_time,
        "end_time": deployment.end_time,
        "validation_results": [
            {
                "test_name": result.test_name,
                "passed": result.passed,
                "duration_ms": result.duration_ms,
                "error_message": result.error_message,
            }
            for result in deployment.validation_results
        ],
        "validation_success_rate": deployment.validation_success_rate,
        "last_error": deployment.last_error,
        "rollback_reason": deployment.rollback_reason,
    }


# Feature Flag Endpoints
@router.post("/feature-flags")
async def create_feature_flag(request: FeatureFlagRequest):
    """Create a new feature flag."""
    feature_flag_manager = get_feature_flag_manager()
    if not feature_flag_manager:
        raise HTTPException(
            status_code=503, detail="Feature flag manager not available"
        )

    try:
        flag_type = FlagType(request.flag_type)

        flag = feature_flag_manager.create_flag(
            flag_id=request.flag_id,
            name=request.name,
            description=request.description,
            flag_type=flag_type,
            default_value=request.default_value,
            enabled=request.enabled,
            current_value=request.current_value,
        )

        return {
            "flag_id": flag.flag_id,
            "name": flag.name,
            "enabled": flag.enabled,
            "status": flag.status.value,
            "created_at": flag.created_at,
        }

    except Exception as e:
        logger.error(f"Failed to create feature flag: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/feature-flags")
async def list_feature_flags():
    """List all feature flags."""
    feature_flag_manager = get_feature_flag_manager()
    if not feature_flag_manager:
        raise HTTPException(
            status_code=503, detail="Feature flag manager not available"
        )

    return feature_flag_manager.list_flags()


@router.get("/feature-flags/{flag_id}")
async def get_feature_flag(flag_id: str):
    """Get detailed information about a feature flag."""
    feature_flag_manager = get_feature_flag_manager()
    if not feature_flag_manager:
        raise HTTPException(
            status_code=503, detail="Feature flag manager not available"
        )

    flag_status = feature_flag_manager.get_flag_status(flag_id)
    if not flag_status:
        raise HTTPException(status_code=404, detail=f"Feature flag {flag_id} not found")

    return flag_status


@router.post("/feature-flags/{flag_id}/evaluate")
async def evaluate_feature_flag(flag_id: str, request: FlagEvaluationRequest):
    """Evaluate a feature flag for given context."""
    feature_flag_manager = get_feature_flag_manager()
    if not feature_flag_manager:
        raise HTTPException(
            status_code=503, detail="Feature flag manager not available"
        )

    try:
        context = EvaluationContext(
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            user_attributes=request.user_attributes,
            custom_attributes=request.custom_attributes,
        )

        result = await feature_flag_manager.evaluate_flag(flag_id, context)

        return {
            "flag_id": result.flag_id,
            "value": result.value,
            "variant": result.variant,
            "reason": result.reason,
            "evaluation_time_ms": result.evaluation_time_ms,
        }

    except Exception as e:
        logger.error(f"Failed to evaluate feature flag {flag_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/feature-flags/{flag_id}")
async def update_feature_flag(flag_id: str, updates: Dict[str, Any]):
    """Update a feature flag."""
    feature_flag_manager = get_feature_flag_manager()
    if not feature_flag_manager:
        raise HTTPException(
            status_code=503, detail="Feature flag manager not available"
        )

    try:
        flag = feature_flag_manager.update_flag(flag_id, **updates)
        if not flag:
            raise HTTPException(
                status_code=404, detail=f"Feature flag {flag_id} not found"
            )

        return {
            "flag_id": flag.flag_id,
            "name": flag.name,
            "enabled": flag.enabled,
            "status": flag.status.value,
            "updated_at": flag.updated_at,
        }

    except Exception as e:
        logger.error(f"Failed to update feature flag {flag_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Pipeline Endpoints
@router.get("/pipelines")
async def list_pipeline_executions():
    """List active pipeline executions."""
    pipeline_executor = get_pipeline_executor()

    executions = pipeline_executor.list_active_executions()

    return [
        {
            "execution_id": exec.execution_id,
            "pipeline_id": exec.pipeline_id,
            "status": exec.status.value,
            "start_time": exec.start_time,
            "duration_minutes": exec.duration_minutes,
            "success_rate": exec.success_rate,
            "failed_stages": exec.failed_stages,
        }
        for exec in executions
    ]


@router.get("/pipelines/{execution_id}")
async def get_pipeline_execution(execution_id: str):
    """Get detailed status of a pipeline execution."""
    pipeline_executor = get_pipeline_executor()

    execution = pipeline_executor.get_execution_status(execution_id)
    if not execution:
        raise HTTPException(
            status_code=404, detail=f"Pipeline execution {execution_id} not found"
        )

    return {
        "execution_id": execution.execution_id,
        "pipeline_id": execution.pipeline_id,
        "status": execution.status.value,
        "start_time": execution.start_time,
        "end_time": execution.end_time,
        "duration_minutes": execution.duration_minutes,
        "success_rate": execution.success_rate,
        "failed_stages": execution.failed_stages,
        "rollback_executed": execution.rollback_executed,
        "rollback_reason": execution.rollback_reason,
        "stage_executions": {
            stage_id: {
                "status": stage_exec.status.value,
                "start_time": stage_exec.start_time,
                "end_time": stage_exec.end_time,
                "duration_minutes": stage_exec.duration_minutes,
                "attempt_count": stage_exec.attempt_count,
                "error_message": stage_exec.error_message,
            }
            for stage_id, stage_exec in execution.stage_executions.items()
        },
    }


# Optimization Endpoints
@router.get("/optimization")
async def list_optimization_executions():
    """List active optimization executions."""
    optimization_pipeline = get_optimization_pipeline()

    executions = optimization_pipeline.list_active_executions()

    return [
        {
            "execution_id": exec.execution_id,
            "optimization_id": exec.config.optimization_id,
            "status": exec.status.value,
            "start_time": exec.start_time,
            "duration_minutes": exec.duration_minutes,
            "current_iteration": exec.current_iteration,
            "best_score": exec.best_result.score if exec.best_result else None,
        }
        for exec in executions
    ]


@router.get("/optimization/{execution_id}")
async def get_optimization_execution(execution_id: str):
    """Get detailed status of an optimization execution."""
    optimization_pipeline = get_optimization_pipeline()

    execution = optimization_pipeline.get_execution_status(execution_id)
    if not execution:
        raise HTTPException(
            status_code=404, detail=f"Optimization execution {execution_id} not found"
        )

    return {
        "execution_id": execution.execution_id,
        "optimization_id": execution.config.optimization_id,
        "status": execution.status.value,
        "start_time": execution.start_time,
        "end_time": execution.end_time,
        "duration_minutes": execution.duration_minutes,
        "current_iteration": execution.current_iteration,
        "max_iterations": execution.config.max_iterations,
        "baseline_metrics": execution.baseline_metrics,
        "best_result": (
            {
                "iteration": execution.best_result.iteration,
                "parameters": execution.best_result.parameters,
                "metrics": execution.best_result.metrics,
                "score": execution.best_result.score,
                "improvement": execution.best_result.improvement,
            }
            if execution.best_result
            else None
        ),
        "convergence_history": execution.convergence_history,
        "error_message": execution.error_message,
        "rollback_executed": execution.rollback_executed,
    }


@router.post("/optimization/{execution_id}/cancel")
async def cancel_optimization(execution_id: str):
    """Cancel an optimization execution."""
    optimization_pipeline = get_optimization_pipeline()

    try:
        success = await optimization_pipeline.cancel_optimization(execution_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization execution {execution_id} not found",
            )

        return {"message": f"Optimization execution {execution_id} cancelled"}

    except Exception as e:
        logger.error(f"Failed to cancel optimization {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Status and Health Endpoints
@router.get("/status")
async def get_deployment_status():
    """Get comprehensive deployment system status."""
    deployment_manager = get_deployment_manager()
    if not deployment_manager:
        raise HTTPException(status_code=503, detail="Deployment manager not available")

    return deployment_manager.get_status()


@router.get("/health")
async def get_deployment_health():
    """Get deployment system health check."""
    deployment_manager = get_deployment_manager()
    if not deployment_manager:
        raise HTTPException(status_code=503, detail="Deployment manager not available")

    health = await deployment_manager.health_check()

    if not health["healthy"]:
        raise HTTPException(status_code=503, detail="Deployment system unhealthy")

    return health


# Tenant Management Endpoints
class TenantRequest(BaseModel):
    """Tenant creation/update request."""

    tenant_id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant display name")
    description: Optional[str] = Field(None, description="Tenant description")
    settings: Dict[str, Any] = Field(
        default_factory=dict, description="Tenant-specific settings"
    )
    enabled: bool = Field(default=True, description="Whether tenant is active")


class TenantResponse(BaseResponse):
    """Tenant response model."""

    tenant_id: str
    name: str
    description: Optional[str]
    settings: Dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


class TenantUsageResponse(BaseResponse):
    """Tenant usage statistics response."""

    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    total_cost: float
    usage_by_detector: Dict[str, int]
    usage_by_framework: Dict[str, int]


@router.post("/tenants", response_model=TenantResponse)
@track_request_metrics("create_tenant")
async def create_tenant(
    request: TenantRequest, current_user: dict = Depends(require_admin_access)
):
    """Create a new tenant."""
    correlation_id = get_correlation_id()
    logger.info(
        "Creating tenant", tenant_id=request.tenant_id, correlation_id=correlation_id
    )

    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise ServiceUnavailableError("Tenant service not available")

        tenant = await tenant_service.create_tenant(
            tenant_id=request.tenant_id,
            name=request.name,
            description=request.description,
            settings=request.settings,
            enabled=request.enabled,
        )

        return TenantResponse(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            description=tenant.description,
            settings=tenant.settings,
            enabled=tenant.enabled,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
        )

    except ServiceUnavailableError as e:
        logger.error(
            "Tenant service unavailable", error=str(e), correlation_id=correlation_id
        )
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValidationError as e:
        logger.error(
            "Tenant validation failed", error=str(e), correlation_id=correlation_id
        )
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Failed to create tenant", error=str(e), correlation_id=correlation_id
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/tenants", response_model=List[TenantResponse])
@track_request_metrics("list_tenants")
async def list_tenants(
    enabled: Optional[bool] = None,
    limit: int = Field(default=50, le=1000),
    offset: int = Field(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """List all tenants with optional filtering."""
    correlation_id = get_correlation_id()
    logger.info(
        "Listing tenants",
        enabled=enabled,
        limit=limit,
        offset=offset,
        correlation_id=correlation_id,
    )

    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise ServiceUnavailableError("Tenant service not available")

        tenants = await tenant_service.list_tenants(
            enabled=enabled, limit=limit, offset=offset
        )

        return [
            TenantResponse(
                tenant_id=tenant.tenant_id,
                name=tenant.name,
                description=tenant.description,
                settings=tenant.settings,
                enabled=tenant.enabled,
                created_at=tenant.created_at,
                updated_at=tenant.updated_at,
            )
            for tenant in tenants
        ]

    except Exception as e:
        logger.error(f"Failed to list tenants: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/tenants/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: str):
    """Get tenant details."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        tenant = await tenant_service.get_tenant(tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

        return TenantResponse(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            description=tenant.description,
            settings=tenant.settings,
            enabled=tenant.enabled,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/tenants/{tenant_id}", response_model=TenantResponse)
async def update_tenant(tenant_id: str, request: TenantRequest):
    """Update tenant configuration."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        tenant = await tenant_service.update_tenant(
            tenant_id=tenant_id,
            name=request.name,
            description=request.description,
            settings=request.settings,
            enabled=request.enabled,
        )

        if not tenant:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

        return TenantResponse(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            description=tenant.description,
            settings=tenant.settings,
            enabled=tenant.enabled,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(
    tenant_id: str, current_user: dict = Depends(require_admin_access)
):
    """Delete a tenant."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        success = await tenant_service.delete_tenant(tenant_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

        return {"message": f"Tenant {tenant_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/tenants/{tenant_id}/usage", response_model=TenantUsageResponse)
async def get_tenant_usage(
    tenant_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Get tenant usage statistics."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date.replace(day=end_date.day - 30)

        usage = await tenant_service.get_tenant_usage(
            tenant_id=tenant_id, start_date=start_date, end_date=end_date
        )

        if not usage:
            raise HTTPException(
                status_code=404, detail=f"Usage data for tenant {tenant_id} not found"
            )

        return TenantUsageResponse(
            tenant_id=usage.tenant_id,
            period_start=usage.period_start,
            period_end=usage.period_end,
            total_requests=usage.total_requests,
            successful_requests=usage.successful_requests,
            failed_requests=usage.failed_requests,
            avg_response_time_ms=usage.avg_response_time_ms,
            total_cost=usage.total_cost,
            usage_by_detector=usage.usage_by_detector,
            usage_by_framework=usage.usage_by_framework,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/tenants/{tenant_id}/enable")
async def enable_tenant(tenant_id: str):
    """Enable a tenant."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        success = await tenant_service.enable_tenant(tenant_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

        return {"message": f"Tenant {tenant_id} enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/tenants/{tenant_id}/disable")
async def disable_tenant(tenant_id: str):
    """Disable a tenant."""
    try:
        tenant_service = get_tenant_service()
        if not tenant_service:
            raise HTTPException(status_code=503, detail="Tenant service not available")

        success = await tenant_service.disable_tenant(tenant_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

        return {"message": f"Tenant {tenant_id} disabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
