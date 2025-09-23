"""
API endpoints for the Analysis Module.

This module contains the FastAPI endpoint implementations that use
the application services for business logic.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse

from ..application.dto import (
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
    HealthCheckDTO,
)
from ..application.services import (
    AnalysisApplicationService,
    BatchAnalysisApplicationService,
)
from ..infrastructure.auth import APIKeyManager, APIKeyScope
from .dependencies import AnalysisServiceDep, BatchAnalysisServiceDep, get_batch_analysis_service
from .auth_middleware import APIKeyAuthDependency

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
            request: AnalysisRequestDTO,
            analysis_service: AnalysisServiceDep,
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ANALYZE}
            ))
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
                
                # Process analysis request
                response = await analysis_service.analyze_metrics(request)
                
                return response
                
            except ValueError as e:
                logger.warning(f"Validation error in analyze endpoint: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Analysis failed in analyze endpoint: {e}")
                raise HTTPException(status_code=500, detail="Analysis processing failed")
        
        @self.router.post("/analyze/batch", response_model=BatchAnalysisResponseDTO)
        async def analyze_metrics_batch(
            request: BatchAnalysisRequestDTO,
            batch_analysis_service: BatchAnalysisServiceDep,
            idempotency_key: str = Header(..., alias="Idempotency-Key"),
            x_api_key: str = Header(..., alias="X-API-Key"),
            x_tenant_id: str = Header(..., alias="X-Tenant-ID")
        ):
            """
            Batch analyze structured metrics.
            
            Args:
                request: Batch analysis request DTO
                idempotency_key: Idempotency key for caching
                batch_analysis_service: Batch analysis application service
                x_api_key: API key for authentication
                x_tenant_id: Tenant ID for multi-tenancy
                
            Returns:
                Batch analysis response DTO
            """
            try:
                # TODO: Add authentication and authorization
                # await auth_service.authenticate(x_api_key, x_tenant_id)
                
                # TODO: Add rate limiting (more restrictive for batch)
                # await rate_limiter.check_rate_limit(x_tenant_id, "analyze_batch")
                
                # Process batch request
                response = await batch_analysis_service.process_batch(request, idempotency_key)
                
                return response
                
            except ValueError as e:
                logger.warning(f"Validation error in batch analyze endpoint: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Batch analysis failed in batch analyze endpoint: {e}")
                raise HTTPException(status_code=500, detail="Batch analysis processing failed")
        
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
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
        @self.router.get("/metrics")
        async def get_metrics():
            """
            Prometheus metrics endpoint.
            
            Returns:
                Prometheus metrics
            """
            # TODO: Implement metrics collection
            return {"message": "Metrics endpoint not yet implemented"}
        
        @self.router.post("/admin/cache/cleanup")
        async def cleanup_cache(
            analysis_service: AnalysisServiceDep,
            x_api_key: str = Header(..., alias="X-API-Key"),
            x_tenant_id: str = Header(..., alias="X-Tenant-ID")
        ):
            """
            Clean up expired cache entries.
            
            Args:
                analysis_service: Analysis application service
                x_api_key: API key for authentication
                x_tenant_id: Tenant ID for multi-tenancy
                
            Returns:
                Cleanup results
            """
            try:
                # TODO: Add authentication and authorization
                # await auth_service.authenticate(x_api_key, x_tenant_id)
                
                cleaned_count = await analysis_service.cleanup_cache()
                return {
                    "cleaned_entries": cleaned_count,
                    "timestamp": "2024-01-01T00:00:00Z"  # This should be dynamic
                }
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                raise HTTPException(status_code=500, detail="Cache cleanup failed")
        
        @self.router.get("/quality/evaluation")
        async def get_quality_evaluation(
            analysis_service: AnalysisServiceDep
        ):
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
                logger.error(f"Quality evaluation failed: {e}")
                raise HTTPException(status_code=500, detail="Quality evaluation failed")
        
        @self.router.post("/quality/evaluate")
        async def evaluate_quality(
            examples: list[tuple[AnalysisRequestDTO, AnalysisResponseDTO]],
            analysis_service: AnalysisServiceDep
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
                logger.error(f"Quality evaluation failed: {e}")
                raise HTTPException(status_code=500, detail="Quality evaluation failed")
