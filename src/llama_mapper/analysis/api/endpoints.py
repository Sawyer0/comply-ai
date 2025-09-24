"""
API endpoints for the Analysis Module.

This module contains the FastAPI endpoint implementations that use
the application services for business logic.
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from ..application.dto import (
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
)
from ..infrastructure.auth import APIKeyManager, APIKeyScope
from .auth_middleware import APIKeyAuthDependency
from .dependencies import (
    AnalysisServiceDep,
    BatchAnalysisServiceDep,
)
from .metrics import metrics_collector

logger = logging.getLogger(__name__)


class AnalysisEndpoints:
    """
    Analysis API endpoints.

    Provides REST API endpoints for analysis operations using
    the application service layer.
    """

    def __init__(self, api_key_manager: Optional[APIKeyManager] = None):
        """
        Initialize the analysis endpoints.

        Args:
            api_key_manager: API key manager for authentication
        """
        self.api_key_manager = api_key_manager
        self.router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
        self._register_endpoints()

    def _register_endpoints(self):
        """Register all endpoints with the router."""

        @self.router.post("/analyze", response_model=AnalysisResponseDTO)
        async def analyze_metrics(
            request: dict,
            analysis_service: AnalysisServiceDep,
            auth: dict = (
                Depends(
                    APIKeyAuthDependency(self.api_key_manager, {APIKeyScope.ANALYZE})
                )
                if self.api_key_manager
                else None
            ),
        ):
            """
            Analyze structured metrics for insights and remediations.

            Args:
                request: Analysis request DTO
                analysis_service: Analysis application service
                auth: Authentication info (API key validated)

            Returns:
                Analysis response DTO
            """
            try:
                # Authentication is handled by the dependency
                # Rate limiting is handled by the middleware

                # Extract the request DTO from the wrapper
                request_dto = AnalysisRequestDTO(**request["request"])

                # Process analysis request
                response = await analysis_service.analyze_metrics(request_dto)

                # Record metrics
                metrics_collector.record_confidence(
                    confidence=response.confidence,
                    analysis_type=getattr(response, "analysis_type", "unknown"),
                    env=request_dto.env,
                    tenant=request_dto.tenant,
                )

                # Check for coverage gaps
                has_coverage_gap = (
                    response.reason and "coverage gap" in response.reason.lower()
                )
                metrics_collector.record_coverage_gap(
                    tenant=request_dto.tenant,
                    env=request_dto.env,
                    has_gap=has_coverage_gap,
                )

                return response

            except ValueError as e:
                logger.warning("Validation error in analyze endpoint: %s", e)
                metrics_collector.record_error("validation_error", "/analyze")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Analysis failed in analyze endpoint: %s", e)
                metrics_collector.record_error("processing_error", "/analyze")
                raise HTTPException(
                    status_code=500, detail="Analysis processing failed"
                )

        @self.router.post("/analyze/batch", response_model=BatchAnalysisResponseDTO)
        async def analyze_metrics_batch(
            batch_request: BatchAnalysisRequestDTO,
            batch_analysis_service: BatchAnalysisServiceDep,
            idempotency_key: str = Header(..., alias="Idempotency-Key"),
            auth: dict = (
                Depends(
                    APIKeyAuthDependency(self.api_key_manager, {APIKeyScope.ANALYZE, APIKeyScope.BATCH_ANALYZE})
                )
                if self.api_key_manager
                else None
            ),
        ):
            """
            Batch analyze structured metrics.

            Args:
                batch_request: Batch analysis request DTO
                idempotency_key: Idempotency key for caching
                batch_analysis_service: Batch analysis application service
                auth: Authentication info (API key validated)

            Returns:
                Batch analysis response DTO
            """
            try:
                # Authentication is handled by the dependency
                # Extract tenant from auth if available
                tenant_id = auth.get("tenant_id", "unknown") if auth else "unknown"

                # Rate limiting for batch operations:
                # The RateLimitMiddleware should be configured with different limits for batch endpoints:
                # - Regular /analyze: 60 requests/min
                # - Batch /analyze/batch: 10 requests/min (more restrictive)
                # This is handled at the middleware level, not in the endpoint logic
                
                # Process batch request
                response = await batch_analysis_service.process_batch(
                    batch_request, idempotency_key
                )

                # Record batch metrics
                metrics_collector.record_request(
                    endpoint="/analyze/batch",
                    status="success",
                    tenant=tenant_id,
                    duration=0.0,  # Duration would be measured by middleware
                    analysis_type="batch",
                )

                return response

            except ValueError as e:
                logger.warning("Validation error in batch analyze endpoint: %s", e)
                metrics_collector.record_error("validation_error", "/analyze/batch")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error("Batch analysis failed in batch analyze endpoint: %s", e)
                metrics_collector.record_error("processing_error", "/analyze/batch")
                raise HTTPException(
                    status_code=500, detail="Batch analysis processing failed"
                )

        @self.router.get("/health")
        async def health_check():
            """
            Health check endpoint.

            Returns:
                Health status
            """
            return {
                "status": "healthy",
                "service": "analysis",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
            }

        @self.router.get("/metrics")
        async def get_metrics():
            """
            Prometheus metrics endpoint.

            Returns:
                Prometheus metrics in Prometheus format
            """
            from .metrics import get_metrics_response
            return get_metrics_response()

        @self.router.post("/admin/cache/cleanup")
        async def cleanup_cache(
            analysis_service: AnalysisServiceDep,
            auth: dict = (
                Depends(
                    APIKeyAuthDependency(self.api_key_manager, {APIKeyScope.ADMIN})
                )
                if self.api_key_manager
                else None
            ),
        ):
            """
            Clean up expired cache entries. Requires ADMIN scope.

            Args:
                analysis_service: Analysis application service
                auth: Authentication info (API key validated with ADMIN scope)

            Returns:
                Cleanup results
            """
            try:
                # Authentication is handled by the dependency
                # Only users with ADMIN scope can access this endpoint
                
                import datetime
                cleaned_count = await analysis_service.cleanup_cache()
                return {
                    "cleaned_entries": cleaned_count,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "tenant_id": auth.get("tenant_id", "unknown") if auth else "unknown"
                }
            except Exception as e:
                logger.error("Cache cleanup failed: %s", e)
                raise HTTPException(status_code=500, detail="Cache cleanup failed")

        @self.router.get("/quality/evaluation")
        async def get_quality_evaluation(analysis_service: AnalysisServiceDep):
            """
            Get quality evaluation summary.

            Args:
                analysis_service: Analysis application service

            Returns:
                Quality evaluation summary
            """
            try:
                summary = analysis_service.get_evaluation_summary()
                return summary
            except Exception as e:
                logger.error("Quality evaluation failed: %s", e)
                raise HTTPException(status_code=500, detail="Quality evaluation failed")

        @self.router.post("/quality/evaluate")
        async def evaluate_quality(
            examples: list[tuple[AnalysisRequestDTO, AnalysisResponseDTO]],
            analysis_service: AnalysisServiceDep,
        ):
            """
            Evaluate quality of analysis outputs.

            Args:
                examples: List of (request, response) DTO tuples
                analysis_service: Analysis application service

            Returns:
                Quality evaluation metrics
            """
            try:
                metrics = await analysis_service.evaluate_quality(examples)
                return metrics
            except Exception as e:
                logger.error("Quality evaluation failed: %s", e)
                raise HTTPException(status_code=500, detail="Quality evaluation failed")
