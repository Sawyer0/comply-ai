"""
Analysis API endpoints for Analysis Service.

This module provides REST API endpoints for analysis operations including
pattern recognition, risk scoring, and compliance intelligence.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Dict, Any, Optional
import logging
import hashlib
import time

from ..dependencies import (
    get_tenant_manager,
    get_analytics_manager,
    get_plugin_manager,
    get_risk_scoring_db_manager,
    get_privacy_db_manager,
    get_risk_scoring_validator,
    get_privacy_validator,
)
from ..tenancy import TenantManager, AnalyticsManager, ResourceType
from ..plugins import PluginManager, AnalysisRequest, AnalysisResult, PluginType
from ..engines.risk_scoring.database import RiskScoringDatabaseManager
from ..privacy.database import PrivacyDatabaseManager
from ..engines.risk_scoring.validator import RiskScoringValidator
from ..privacy.privacy_validator import PrivacyValidator
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_content(
    request_data: Dict[str, Any],
    x_tenant_id: str = Header(..., description="Tenant ID"),
    x_analysis_type: str = Header(
        default="pattern_recognition", description="Analysis type"
    ),
    x_framework: Optional[str] = Header(None, description="Compliance framework"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    analytics_manager: AnalyticsManager = Depends(get_analytics_manager),
    plugin_manager: PluginManager = Depends(get_plugin_manager),
    risk_scoring_db: RiskScoringDatabaseManager = Depends(get_risk_scoring_db_manager),
    privacy_db: PrivacyDatabaseManager = Depends(get_privacy_db_manager),
    risk_validator: RiskScoringValidator = Depends(get_risk_scoring_validator),
    privacy_validator: PrivacyValidator = Depends(get_privacy_validator),
):
    """
    Perform analysis on content using specified analysis type.

    Args:
        request_data: Analysis request data
        x_tenant_id: Tenant identifier
        x_analysis_type: Type of analysis to perform
        x_framework: Optional compliance framework
        tenant_manager: Tenant manager instance
        analytics_manager: Analytics manager instance
        plugin_manager: Plugin manager instance

    Returns:
        Analysis results
    """
    start_time = time.time()

    try:
        logger.info(
            "Starting analysis",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            correlation_id=get_correlation_id(),
        )

        # Get tenant configuration
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Check resource quota
        can_consume = await tenant_manager.check_quota(
            x_tenant_id, ResourceType.ANALYSIS_REQUESTS, 1
        )
        if not can_consume:
            raise HTTPException(status_code=429, detail="Analysis quota exceeded")

        # Validate analysis type is enabled for tenant
        if (
            x_analysis_type == "pattern_recognition"
            and not tenant_config.enable_pattern_recognition
        ):
            raise HTTPException(
                status_code=403, detail="Pattern recognition disabled for tenant"
            )

        # Create content hash for privacy (don't log raw content)
        content_str = str(request_data.get("content", ""))
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Validate privacy compliance
        privacy_result = privacy_validator.validate_data_privacy(request_data)
        if not privacy_result.is_compliant:
            logger.warning(
                "Privacy validation failed",
                tenant_id=x_tenant_id,
                violations=privacy_result.violations,
                correlation_id=get_correlation_id(),
            )
            # Use sanitized data if available
            if privacy_result.sanitized_data:
                request_data = privacy_result.sanitized_data

        # Create analysis request
        analysis_request = AnalysisRequest(
            request_id=get_correlation_id(),
            tenant_id=x_tenant_id,
            content_hash=content_hash,
            metadata=request_data.get("metadata", {}),
            analysis_type=x_analysis_type,
            confidence_threshold=tenant_config.default_confidence_threshold,
            framework=x_framework,
            custom_config=request_data.get("config", {}),
        )

        # Find appropriate plugin for analysis type
        plugin_name = f"builtin_{x_analysis_type}"

        # Execute analysis
        result = await plugin_manager.execute_analysis(plugin_name, analysis_request)

        if not result:
            raise HTTPException(status_code=500, detail="Analysis execution failed")

        # Perform risk scoring if analysis type is risk_scoring
        risk_score_data = None
        if x_analysis_type == "risk_scoring" and result.result_data.get("findings"):
            try:
                from ..schemas.domain_models import SecurityFinding, RiskLevel
                from uuid import uuid4

                # Convert result findings to SecurityFinding objects for validation
                findings = []
                for finding_data in result.result_data.get("findings", []):
                    finding = SecurityFinding(
                        finding_id=finding_data.get("id", str(uuid4())),
                        detector_id=finding_data.get("detector_id", "unknown"),
                        severity=RiskLevel(finding_data.get("severity", "medium")),
                        category=finding_data.get("category", "security"),
                        description=finding_data.get("description", ""),
                        confidence=finding_data.get("confidence", 0.5),
                        metadata=finding_data.get("metadata", {}),
                    )
                    findings.append(finding)

                # Validate findings for risk scoring
                validation_result = risk_validator.validate_security_findings(findings)
                if validation_result.is_valid:
                    # Save risk scoring data to database using real database manager
                    from ..engines.risk_scoring.database import RiskScoringDatabaseManager
                    
                    risk_db_manager = RiskScoringDatabaseManager()
                    await risk_db_manager.initialize()
                    
                    # Create comprehensive risk score data
                    risk_score_data = {
                        "tenant_id": x_tenant_id,
                        "analysis_id": result.analysis_id,
                        "composite_score": result.confidence,
                        "technical_risk": result.technical_risk,
                        "business_risk": result.business_risk,
                        "regulatory_risk": result.regulatory_risk,
                        "temporal_risk": result.temporal_risk,
                        "risk_level": result.risk_level,
                        "findings_count": len(findings),
                        "validation_passed": True,
                        "created_at": datetime.utcnow(),
                        "correlation_id": get_correlation_id(),
                    }
                    
                    # Save to database
                    await risk_db_manager.save_risk_score(risk_score_data)
                    
                    logger.info(
                        "Risk scoring data saved to database",
                        tenant_id=x_tenant_id,
                        analysis_id=result.analysis_id,
                        composite_score=result.confidence,
                        findings_count=len(findings),
                    )
                else:
                    logger.warning(
                        "Risk scoring validation failed",
                        tenant_id=x_tenant_id,
                        errors=validation_result.errors,
                    )

            except Exception as e:
                logger.error(
                    "Risk scoring integration failed",
                    tenant_id=x_tenant_id,
                    error=str(e),
                )

        # Consume quota
        await tenant_manager.consume_quota(
            x_tenant_id, ResourceType.ANALYSIS_REQUESTS, 1
        )

        # Record analytics
        processing_time = (time.time() - start_time) * 1000
        await analytics_manager.record_request_metrics(
            x_tenant_id,
            {
                "request_count": 1,
                "success": True,
                "response_time_ms": processing_time,
                "confidence": result.confidence,
                "analysis_type": x_analysis_type,
                "framework": x_framework,
            },
        )

        logger.info(
            "Analysis completed successfully",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            confidence=result.confidence,
            processing_time_ms=processing_time,
        )

        # Prepare response
        response_data = {
            "request_id": result.request_id,
            "tenant_id": x_tenant_id,
            "analysis_type": x_analysis_type,
            "confidence": result.confidence,
            "results": result.result_data,
            "processing_time_ms": result.processing_time_ms,
            "framework": x_framework,
            "metadata": result.metadata,
        }

        # Add privacy validation info (without sensitive details)
        if not privacy_result.is_compliant:
            response_data["privacy_status"] = {
                "compliant": False,
                "violations_count": len(privacy_result.violations),
                "warnings_count": len(privacy_result.warnings),
                "data_sanitized": privacy_result.sanitized_data is not None,
            }

        # Add risk scoring data if available
        if risk_score_data:
            response_data["risk_scoring"] = risk_score_data

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        # Record failed analytics
        processing_time = (time.time() - start_time) * 1000
        await analytics_manager.record_request_metrics(
            x_tenant_id,
            {
                "request_count": 1,
                "success": False,
                "response_time_ms": processing_time,
                "analysis_type": x_analysis_type,
                "error_type": type(e).__name__,
            },
        )

        logger.error(
            "Analysis failed",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch-analyze", response_model=Dict[str, Any])
async def batch_analyze_content(
    batch_request: Dict[str, Any],
    x_tenant_id: str = Header(..., description="Tenant ID"),
    x_analysis_type: str = Header(
        default="pattern_recognition", description="Analysis type"
    ),
    x_framework: Optional[str] = Header(None, description="Compliance framework"),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    analytics_manager: AnalyticsManager = Depends(get_analytics_manager),
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Perform batch analysis on multiple content items.

    Args:
        batch_request: Batch analysis request data
        x_tenant_id: Tenant identifier
        x_analysis_type: Type of analysis to perform
        x_framework: Optional compliance framework
        tenant_manager: Tenant manager instance
        analytics_manager: Analytics manager instance
        plugin_manager: Plugin manager instance

    Returns:
        Batch analysis results
    """
    start_time = time.time()

    try:
        requests_data = batch_request.get("requests", [])
        if not requests_data:
            raise HTTPException(status_code=400, detail="No requests provided")

        if len(requests_data) > 100:  # Configurable limit
            raise HTTPException(
                status_code=400, detail="Too many requests in batch (max 100)"
            )

        logger.info(
            "Starting batch analysis",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            batch_size=len(requests_data),
            correlation_id=get_correlation_id(),
        )

        # Get tenant configuration
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Check batch quota
        can_consume = await tenant_manager.check_quota(
            x_tenant_id, ResourceType.BATCH_REQUESTS, 1
        )
        if not can_consume:
            raise HTTPException(status_code=429, detail="Batch analysis quota exceeded")

        # Check individual request quota
        can_consume_requests = await tenant_manager.check_quota(
            x_tenant_id, ResourceType.ANALYSIS_REQUESTS, len(requests_data)
        )
        if not can_consume_requests:
            raise HTTPException(
                status_code=429, detail="Analysis requests quota exceeded"
            )

        # Create analysis requests
        analysis_requests = []
        for i, request_data in enumerate(requests_data):
            content_str = str(request_data.get("content", ""))
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()

            analysis_request = AnalysisRequest(
                request_id=f"{get_correlation_id()}-{i}",
                tenant_id=x_tenant_id,
                content_hash=content_hash,
                metadata=request_data.get("metadata", {}),
                analysis_type=x_analysis_type,
                confidence_threshold=tenant_config.default_confidence_threshold,
                framework=x_framework,
                custom_config=request_data.get("config", {}),
            )
            analysis_requests.append(analysis_request)

        # Find appropriate plugin for analysis type
        plugin_name = f"builtin_{x_analysis_type}"

        # Execute batch analysis
        results = await plugin_manager.execute_batch_analysis(
            plugin_name, analysis_requests
        )

        if not results:
            raise HTTPException(
                status_code=500, detail="Batch analysis execution failed"
            )

        # Consume quotas
        await tenant_manager.consume_quota(x_tenant_id, ResourceType.BATCH_REQUESTS, 1)
        await tenant_manager.consume_quota(
            x_tenant_id, ResourceType.ANALYSIS_REQUESTS, len(requests_data)
        )

        # Calculate summary statistics
        successful_results = [r for r in results if not r.errors]
        failed_results = [r for r in results if r.errors]

        avg_confidence = 0.0
        if successful_results:
            avg_confidence = sum(r.confidence for r in successful_results) / len(
                successful_results
            )

        processing_time = (time.time() - start_time) * 1000

        # Record analytics
        await analytics_manager.record_request_metrics(
            x_tenant_id,
            {
                "request_count": len(requests_data),
                "success": len(failed_results) == 0,
                "response_time_ms": processing_time,
                "confidence": avg_confidence,
                "analysis_type": x_analysis_type,
                "framework": x_framework,
            },
        )

        logger.info(
            "Batch analysis completed",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            total_requests=len(requests_data),
            successful_count=len(successful_results),
            failed_count=len(failed_results),
            avg_confidence=avg_confidence,
            processing_time_ms=processing_time,
        )

        return {
            "batch_id": get_correlation_id(),
            "tenant_id": x_tenant_id,
            "analysis_type": x_analysis_type,
            "total_requests": len(requests_data),
            "successful_count": len(successful_results),
            "failed_count": len(failed_results),
            "avg_confidence": avg_confidence,
            "processing_time_ms": processing_time,
            "results": [
                {
                    "request_id": result.request_id,
                    "confidence": result.confidence,
                    "results": result.result_data,
                    "processing_time_ms": result.processing_time_ms,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
                for result in results
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        # Record failed analytics
        processing_time = (time.time() - start_time) * 1000
        await analytics_manager.record_request_metrics(
            x_tenant_id,
            {
                "request_count": len(batch_request.get("requests", [])),
                "success": False,
                "response_time_ms": processing_time,
                "analysis_type": x_analysis_type,
                "error_type": type(e).__name__,
            },
        )

        logger.error(
            "Batch analysis failed",
            tenant_id=x_tenant_id,
            analysis_type=x_analysis_type,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/types", response_model=List[str])
async def get_analysis_types(
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Get available analysis types.

    Args:
        plugin_manager: Plugin manager instance

    Returns:
        List of available analysis types
    """
    try:
        registry = plugin_manager.get_registry()
        analysis_plugins = registry.get_plugins_by_type(PluginType.ANALYSIS_ENGINE)

        analysis_types = []
        for plugin in analysis_plugins:
            # Find plugin name
            plugin_name = None
            for name, p in registry.plugins.items():
                if p == plugin:
                    plugin_name = name
                    break

            if plugin_name and hasattr(plugin, "get_supported_analysis_types"):
                types = plugin.get_supported_analysis_types()
                analysis_types.extend(types)

        return list(set(analysis_types))  # Remove duplicates

    except Exception as e:
        logger.error("Failed to get analysis types", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get analysis types: {str(e)}"
        )


@router.get("/frameworks", response_model=List[str])
async def get_supported_frameworks():
    """
    Get supported compliance frameworks.

    Returns:
        List of supported frameworks
    """
    return ["SOC2", "ISO27001", "HIPAA", "GDPR", "PCI-DSS", "NIST"]
