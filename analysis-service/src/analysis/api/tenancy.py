"""
Tenancy API endpoints for Analysis Service.

This module provides REST API endpoints for tenant management, configuration,
resource quotas, and analytics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import logging

from ..tenancy import (
    TenantManager,
    TenantConfiguration,
    TenantRequest,
    QuotaRequest,
    TenantAnalytics,
    ResourceType,
)
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenancy", tags=["tenancy"])


# Import dependency injection
from ..dependencies import get_tenant_manager


@router.post("/tenants/{tenant_id}", response_model=Dict[str, Any])
async def create_tenant(
    tenant_id: str,
    request: TenantRequest,
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Create a new tenant configuration.

    Args:
        tenant_id: Unique tenant identifier
        request: Tenant creation request
        tenant_manager: Tenant manager instance

    Returns:
        Created tenant configuration
    """
    try:
        logger.info(
            "Creating tenant", tenant_id=tenant_id, correlation_id=get_correlation_id()
        )

        # Check if tenant already exists
        existing_config = await tenant_manager.get_tenant_config(tenant_id)
        if existing_config:
            raise HTTPException(status_code=409, detail="Tenant already exists")

        # Create tenant
        tenant_config = await tenant_manager.create_tenant(tenant_id, request)

        logger.info("Tenant created successfully", tenant_id=tenant_id)

        return {
            "tenant_id": tenant_config.tenant_id,
            "name": tenant_config.name,
            "status": tenant_config.status.value,
            "created_at": tenant_config.created_at.isoformat(),
            "message": "Tenant created successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create tenant", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to create tenant: {str(e)}"
        )


@router.get("/tenants/{tenant_id}", response_model=Dict[str, Any])
async def get_tenant(
    tenant_id: str, tenant_manager: TenantManager = Depends(get_tenant_manager)
):
    """
    Get tenant configuration.

    Args:
        tenant_id: Tenant identifier
        tenant_manager: Tenant manager instance

    Returns:
        Tenant configuration
    """
    try:
        logger.info(
            "Getting tenant config",
            tenant_id=tenant_id,
            correlation_id=get_correlation_id(),
        )

        tenant_config = await tenant_manager.get_tenant_config(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        return {
            "tenant_id": tenant_config.tenant_id,
            "name": tenant_config.name,
            "status": tenant_config.status.value,
            "configuration": {
                "default_confidence_threshold": tenant_config.default_confidence_threshold,
                "enable_ml_analysis": tenant_config.enable_ml_analysis,
                "enable_statistical_analysis": tenant_config.enable_statistical_analysis,
                "enable_pattern_recognition": tenant_config.enable_pattern_recognition,
                "quality_alert_threshold": tenant_config.quality_alert_threshold,
                "enable_quality_monitoring": tenant_config.enable_quality_monitoring,
                "enable_content_scrubbing": tenant_config.enable_content_scrubbing,
                "log_level": tenant_config.log_level,
                "custom_engines": tenant_config.custom_engines,
                "preferred_frameworks": tenant_config.preferred_frameworks,
            },
            "quotas": {
                rt.value: {
                    "limit": quota.limit,
                    "current_usage": quota.current_usage,
                    "remaining": quota.remaining(),
                    "period_hours": quota.period_hours,
                    "reset_at": quota.reset_at.isoformat() if quota.reset_at else None,
                }
                for rt, quota in tenant_config.quotas.items()
            },
            "created_at": tenant_config.created_at.isoformat(),
            "updated_at": tenant_config.updated_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant", tenant_id=tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tenant: {str(e)}")


@router.put("/tenants/{tenant_id}", response_model=Dict[str, Any])
async def update_tenant(
    tenant_id: str,
    updates: Dict[str, Any],
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Update tenant configuration.

    Args:
        tenant_id: Tenant identifier
        updates: Configuration updates
        tenant_manager: Tenant manager instance

    Returns:
        Updated tenant configuration
    """
    try:
        logger.info(
            "Updating tenant config",
            tenant_id=tenant_id,
            correlation_id=get_correlation_id(),
        )

        tenant_config = await tenant_manager.update_tenant_config(tenant_id, updates)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        logger.info("Tenant updated successfully", tenant_id=tenant_id)

        return {
            "tenant_id": tenant_config.tenant_id,
            "name": tenant_config.name,
            "status": tenant_config.status.value,
            "updated_at": tenant_config.updated_at.isoformat(),
            "message": "Tenant updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update tenant", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to update tenant: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/quotas", response_model=Dict[str, Any])
async def set_resource_quota(
    tenant_id: str,
    quota_request: QuotaRequest,
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Set resource quota for a tenant.

    Args:
        tenant_id: Tenant identifier
        quota_request: Quota configuration
        tenant_manager: Tenant manager instance

    Returns:
        Success message
    """
    try:
        logger.info(
            "Setting resource quota",
            tenant_id=tenant_id,
            resource_type=quota_request.resource_type,
            correlation_id=get_correlation_id(),
        )

        success = await tenant_manager.set_resource_quota(tenant_id, quota_request)
        if not success:
            raise HTTPException(
                status_code=404, detail="Tenant not found or quota update failed"
            )

        logger.info("Resource quota set successfully", tenant_id=tenant_id)

        return {
            "tenant_id": tenant_id,
            "resource_type": quota_request.resource_type,
            "limit": quota_request.limit,
            "period_hours": quota_request.period_hours,
            "message": "Resource quota set successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set resource quota", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to set resource quota: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/quotas", response_model=Dict[str, Any])
async def get_resource_quotas(
    tenant_id: str, tenant_manager: TenantManager = Depends(get_tenant_manager)
):
    """
    Get resource quotas for a tenant.

    Args:
        tenant_id: Tenant identifier
        tenant_manager: Tenant manager instance

    Returns:
        Resource quotas
    """
    try:
        logger.info(
            "Getting resource quotas",
            tenant_id=tenant_id,
            correlation_id=get_correlation_id(),
        )

        tenant_config = await tenant_manager.get_tenant_config(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        quotas = {}
        for resource_type, quota in tenant_config.quotas.items():
            # Reset quota if expired
            quota.reset_if_expired()

            quotas[resource_type.value] = {
                "limit": quota.limit,
                "current_usage": quota.current_usage,
                "remaining": quota.remaining(),
                "period_hours": quota.period_hours,
                "reset_at": quota.reset_at.isoformat() if quota.reset_at else None,
                "is_exceeded": quota.is_exceeded(),
            }

        return {"tenant_id": tenant_id, "quotas": quotas}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get resource quotas", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get resource quotas: {str(e)}"
        )


@router.post(
    "/tenants/{tenant_id}/quotas/{resource_type}/consume", response_model=Dict[str, Any]
)
async def consume_quota(
    tenant_id: str,
    resource_type: str,
    amount: int = Query(1, ge=1, description="Amount to consume"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Consume tenant resource quota.

    Args:
        tenant_id: Tenant identifier
        resource_type: Resource type to consume
        amount: Amount to consume
        tenant_manager: Tenant manager instance

    Returns:
        Consumption result
    """
    try:
        logger.info(
            "Consuming quota",
            tenant_id=tenant_id,
            resource_type=resource_type,
            amount=amount,
            correlation_id=get_correlation_id(),
        )

        # Validate resource type
        try:
            rt = ResourceType(resource_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid resource type: {resource_type}"
            )

        # Check quota availability
        can_consume = await tenant_manager.check_quota(tenant_id, rt, amount)
        if not can_consume:
            raise HTTPException(status_code=429, detail="Quota exceeded")

        # Consume quota
        success = await tenant_manager.consume_quota(tenant_id, rt, amount)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to consume quota")

        # Get updated quota info
        tenant_config = await tenant_manager.get_tenant_config(tenant_id)
        quota = tenant_config.quotas.get(rt)

        return {
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "consumed": amount,
            "remaining": quota.remaining() if quota else 0,
            "message": "Quota consumed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to consume quota", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to consume quota: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/analytics", response_model=Dict[str, Any])
async def get_tenant_analytics(
    tenant_id: str,
    period_start: Optional[datetime] = Query(
        None, description="Analytics period start"
    ),
    period_end: Optional[datetime] = Query(None, description="Analytics period end"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Get tenant analytics for a specific period.

    Args:
        tenant_id: Tenant identifier
        period_start: Analytics period start (defaults to 24 hours ago)
        period_end: Analytics period end (defaults to now)
        tenant_manager: Tenant manager instance

    Returns:
        Tenant analytics
    """
    try:
        logger.info(
            "Getting tenant analytics",
            tenant_id=tenant_id,
            correlation_id=get_correlation_id(),
        )

        # Default to last 24 hours if not specified
        if not period_end:
            period_end = datetime.now(timezone.utc)
        if not period_start:
            period_start = period_end - timedelta(hours=24)

        analytics = await tenant_manager.get_tenant_analytics(
            tenant_id, period_start, period_end
        )

        if not analytics:
            # Return empty analytics if none found
            analytics = TenantAnalytics(
                tenant_id=tenant_id, period_start=period_start, period_end=period_end
            )

        return {
            "tenant_id": analytics.tenant_id,
            "period_start": analytics.period_start.isoformat(),
            "period_end": analytics.period_end.isoformat(),
            "metrics": {
                "requests": {
                    "total": analytics.total_requests,
                    "successful": analytics.successful_requests,
                    "failed": analytics.failed_requests,
                    "success_rate": (
                        analytics.successful_requests / max(1, analytics.total_requests)
                    )
                    * 100,
                },
                "performance": {
                    "avg_response_time_ms": analytics.avg_response_time_ms,
                    "p95_response_time_ms": analytics.p95_response_time_ms,
                },
                "quality": {
                    "avg_confidence_score": analytics.avg_confidence_score,
                    "low_confidence_count": analytics.low_confidence_count,
                },
                "resources": {
                    "cpu_minutes_used": analytics.cpu_minutes_used,
                    "storage_mb_used": analytics.storage_mb_used,
                    "ml_inference_calls": analytics.ml_inference_calls,
                },
                "analysis_breakdown": {
                    "pattern_recognition": analytics.pattern_recognition_count,
                    "risk_scoring": analytics.risk_scoring_count,
                    "compliance_mapping": analytics.compliance_mapping_count,
                },
                "framework_usage": analytics.framework_usage,
                "error_breakdown": analytics.error_types,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get tenant analytics", tenant_id=tenant_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get tenant analytics: {str(e)}"
        )


@router.get("/tenants", response_model=List[Dict[str, Any]])
async def list_tenants(tenant_manager: TenantManager = Depends(get_tenant_manager)):
    """
    List all tenant configurations.

    Args:
        tenant_manager: Tenant manager instance

    Returns:
        List of tenant configurations
    """
    try:
        logger.info("Listing tenants", correlation_id=get_correlation_id())

        tenants = await tenant_manager.list_tenants()

        return [
            {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "status": tenant.status.value,
                "created_at": tenant.created_at.isoformat(),
                "updated_at": tenant.updated_at.isoformat(),
                "quota_summary": {
                    rt.value: {
                        "limit": quota.limit,
                        "current_usage": quota.current_usage,
                        "is_exceeded": quota.is_exceeded(),
                    }
                    for rt, quota in tenant.quotas.items()
                },
            }
            for tenant in tenants
        ]

    except Exception as e:
        logger.error("Failed to list tenants", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tenants: {str(e)}")


@router.post("/tenants/{tenant_id}/analytics/update", response_model=Dict[str, Any])
async def update_tenant_analytics(
    tenant_id: str,
    metrics: Dict[str, Any],
    tenant_manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Update tenant analytics with new metrics.

    Args:
        tenant_id: Tenant identifier
        metrics: Metrics to update
        tenant_manager: Tenant manager instance

    Returns:
        Update result
    """
    try:
        logger.info(
            "Updating tenant analytics",
            tenant_id=tenant_id,
            correlation_id=get_correlation_id(),
        )

        success = await tenant_manager.update_analytics(tenant_id, metrics)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update analytics")

        return {
            "tenant_id": tenant_id,
            "updated_metrics": list(metrics.keys()),
            "message": "Analytics updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to update tenant analytics", tenant_id=tenant_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to update tenant analytics: {str(e)}"
        )
