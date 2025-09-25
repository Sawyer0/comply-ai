"""
Risk Scoring API endpoints for Analysis Service.

This module provides REST API endpoints specifically for risk scoring operations,
following SRP by focusing only on risk scoring API concerns.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Dict, Any, Optional
import logging
from uuid import uuid4

from ..dependencies import (
    get_tenant_manager,
    get_risk_scoring_db_manager,
    get_risk_scoring_validator,
)
from ..tenancy import TenantManager
from ..engines.risk_scoring.database import RiskScoringDatabaseManager
from ..engines.risk_scoring.validator import RiskScoringValidator
from ..schemas.domain_models import SecurityFinding, RiskLevel
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk-scoring", tags=["risk-scoring"])


@router.post("/validate-findings", response_model=Dict[str, Any])
async def validate_security_findings(
    findings_data: List[Dict[str, Any]],
    x_tenant_id: str = Header(..., description="Tenant ID"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    risk_validator: RiskScoringValidator = Depends(get_risk_scoring_validator),
):
    """
    Validate security findings for risk scoring.

    Args:
        findings_data: List of security findings to validate
        x_tenant_id: Tenant identifier
        tenant_manager: Tenant manager instance
        risk_validator: Risk scoring validator instance

    Returns:
        Validation results
    """
    try:
        logger.info(
            "Validating security findings for risk scoring",
            tenant_id=x_tenant_id,
            findings_count=len(findings_data),
            correlation_id=get_correlation_id(),
        )

        # Check tenant exists
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Convert to SecurityFinding objects
        findings = []
        for finding_data in findings_data:
            try:
                finding = SecurityFinding(
                    finding_id=finding_data.get("finding_id", str(uuid4())),
                    detector_id=finding_data.get("detector_id", "unknown"),
                    severity=RiskLevel(finding_data.get("severity", "medium")),
                    category=finding_data.get("category", "security"),
                    description=finding_data.get("description", ""),
                    confidence=finding_data.get("confidence", 0.5),
                    metadata=finding_data.get("metadata", {}),
                )
                findings.append(finding)
            except Exception as e:
                logger.warning(
                    "Failed to parse finding", finding_data=finding_data, error=str(e)
                )

        # Validate findings
        validation_result = risk_validator.validate_security_findings(findings)

        logger.info(
            "Security findings validation completed",
            tenant_id=x_tenant_id,
            is_valid=validation_result.is_valid,
            errors_count=len(validation_result.errors),
            warnings_count=len(validation_result.warnings),
        )

        return {
            "validation_id": get_correlation_id(),
            "tenant_id": x_tenant_id,
            "findings_count": len(findings),
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "summary": {
                "total_findings": len(findings),
                "valid_findings": len(findings) - len(validation_result.errors),
                "validation_passed": validation_result.is_valid,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Security findings validation failed", tenant_id=x_tenant_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/statistics/{tenant_id}", response_model=Dict[str, Any])
async def get_risk_scoring_statistics(
    tenant_id: str,
    x_tenant_id: str = Header(..., description="Tenant ID"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    risk_scoring_db: RiskScoringDatabaseManager = Depends(get_risk_scoring_db_manager),
):
    """
    Get risk scoring statistics for a tenant.

    Args:
        tenant_id: Tenant ID to get statistics for
        x_tenant_id: Request tenant ID (must match tenant_id)
        tenant_manager: Tenant manager instance
        risk_scoring_db: Risk scoring database manager

    Returns:
        Risk scoring statistics
    """
    try:
        # Verify tenant access
        if tenant_id != x_tenant_id:
            raise HTTPException(
                status_code=403, detail="Cannot access statistics for different tenant"
            )

        # Check tenant exists
        tenant_config = await tenant_manager.get_tenant_config(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Get statistics
        statistics = await risk_scoring_db.get_tenant_risk_statistics(tenant_id)

        logger.info(
            "Risk scoring statistics retrieved",
            tenant_id=tenant_id,
            total_risk_scores=statistics.get("total_risk_scores", 0),
        )

        return {
            "tenant_id": tenant_id,
            "statistics": statistics,
            "retrieved_at": get_correlation_id(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get risk scoring statistics", tenant_id=tenant_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )
