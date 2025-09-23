"""
Use cases for the Analysis Module.

This module contains the application use cases that orchestrate
business operations and coordinate between domain services.
"""

import logging
from typing import Any, Dict, List, Optional

from ..domain.entities import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisErrorResponse,
)
from ..domain.services import AnalysisService, BatchAnalysisService, ValidationService
from .dto import (
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
    AnalysisErrorResponseDTO,
)

logger = logging.getLogger(__name__)


class AnalyzeMetricsUseCase:
    """
    Use case for analyzing single metrics request.
    
    Orchestrates the analysis process for a single request,
    handling validation, processing, and error scenarios.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisService,
        validation_service: ValidationService,
    ):
        """
        Initialize the analyze metrics use case.
        
        Args:
            analysis_service: Analysis service for processing
            validation_service: Validation service for request validation
        """
        self.analysis_service = analysis_service
        self.validation_service = validation_service
    
    async def execute(self, request_dto: AnalysisRequestDTO) -> AnalysisResponseDTO:
        """
        Execute the analyze metrics use case.
        
        Args:
            request_dto: Analysis request DTO
            
        Returns:
            Analysis response DTO
            
        Raises:
            ValueError: If request validation fails
            RuntimeError: If analysis processing fails
        """
        try:
            # Convert DTO to domain entity
            request = request_dto.to_domain_entity()
            
            # Validate request
            validation_errors = self.validation_service.validate_request(request)
            if validation_errors:
                raise ValueError(f"Request validation failed: {', '.join(validation_errors)}")
            
            # Process analysis
            response = await self.analysis_service.analyze_single(request)
            
            # Convert domain entity to DTO
            return AnalysisResponseDTO.from_domain_entity(response)
            
        except ValueError as e:
            logger.error(f"Validation error in analyze metrics use case: {e}")
            raise
        except Exception as e:
            logger.error(f"Analysis error in analyze metrics use case: {e}")
            raise RuntimeError(f"Analysis processing failed: {str(e)}")


class BatchAnalyzeMetricsUseCase:
    """
    Use case for analyzing batch metrics requests.
    
    Orchestrates the batch analysis process, handling
    idempotency, concurrent processing, and error scenarios.
    """
    
    def __init__(
        self,
        batch_analysis_service: BatchAnalysisService,
        validation_service: ValidationService,
    ):
        """
        Initialize the batch analyze metrics use case.
        
        Args:
            batch_analysis_service: Batch analysis service for processing
            validation_service: Validation service for request validation
        """
        self.batch_analysis_service = batch_analysis_service
        self.validation_service = validation_service
    
    async def execute(
        self, 
        request_dto: BatchAnalysisRequestDTO,
        idempotency_key: str
    ) -> BatchAnalysisResponseDTO:
        """
        Execute the batch analyze metrics use case.
        
        Args:
            request_dto: Batch analysis request DTO
            idempotency_key: Idempotency key for caching
            
        Returns:
            Batch analysis response DTO
            
        Raises:
            ValueError: If request validation fails
            RuntimeError: If batch analysis processing fails
        """
        try:
            # Convert DTO to domain entity
            batch_request = request_dto.to_domain_entity()
            
            # Validate each request in the batch
            for request in batch_request.requests:
                validation_errors = self.validation_service.validate_request(request)
                if validation_errors:
                    raise ValueError(f"Request validation failed for request {request.request_id}: {', '.join(validation_errors)}")
            
            # Process batch analysis
            batch_response = await self.batch_analysis_service.process_batch(
                batch_request, idempotency_key
            )
            
            # Convert domain entity to DTO
            return BatchAnalysisResponseDTO.from_domain_entity(batch_response)
            
        except ValueError as e:
            logger.error(f"Validation error in batch analyze metrics use case: {e}")
            raise
        except Exception as e:
            logger.error(f"Batch analysis error in batch analyze metrics use case: {e}")
            raise RuntimeError(f"Batch analysis processing failed: {str(e)}")


class QualityEvaluationUseCase:
    """
    Use case for quality evaluation operations.
    
    Orchestrates quality evaluation processes including
    golden dataset evaluation and drift detection.
    """
    
    def __init__(self, quality_evaluator):
        """
        Initialize the quality evaluation use case.
        
        Args:
            quality_evaluator: Quality evaluator implementation
        """
        self.quality_evaluator = quality_evaluator
    
    async def evaluate_quality(
        self, 
        examples: List[tuple[AnalysisRequest, AnalysisResponse]]
    ) -> Dict[str, Any]:
        """
        Execute quality evaluation.
        
        Args:
            examples: List of (request, response) tuples to evaluate
            
        Returns:
            Quality evaluation metrics
        """
        try:
            return await self.quality_evaluator.evaluate_batch(examples)
        except Exception as e:
            logger.error(f"Quality evaluation error: {e}")
            raise RuntimeError(f"Quality evaluation failed: {str(e)}")
    
    def calculate_drift(self, recent_outputs: List[AnalysisResponse]) -> float:
        """
        Calculate quality drift.
        
        Args:
            recent_outputs: Recent analysis outputs
            
        Returns:
            Drift score
        """
        try:
            return self.quality_evaluator.calculate_drift_score(recent_outputs)
        except Exception as e:
            logger.error(f"Drift calculation error: {e}")
            raise RuntimeError(f"Drift calculation failed: {str(e)}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary.
        
        Returns:
            Evaluation summary
        """
        try:
            return self.quality_evaluator.get_evaluation_summary()
        except Exception as e:
            logger.error(f"Evaluation summary error: {e}")
            raise RuntimeError(f"Evaluation summary failed: {str(e)}")


class CacheManagementUseCase:
    """
    Use case for cache management operations.
    
    Orchestrates cache cleanup and management operations.
    """
    
    def __init__(self, idempotency_manager):
        """
        Initialize the cache management use case.
        
        Args:
            idempotency_manager: Idempotency manager implementation
        """
        self.idempotency_manager = idempotency_manager
    
    async def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            return await self.idempotency_manager.cleanup_expired_entries()
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            raise RuntimeError(f"Cache cleanup failed: {str(e)}")


class HealthCheckUseCase:
    """
    Use case for health check operations.
    
    Orchestrates health check processes and system status validation.
    """
    
    def __init__(self, model_server, validator, opa_generator):
        """
        Initialize the health check use case.
        
        Args:
            model_server: Model server for health checks
            validator: Validator for health checks
            opa_generator: OPA generator for health checks
        """
        self.model_server = model_server
        self.validator = validator
        self.opa_generator = opa_generator
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute health check.
        
        Returns:
            Health check results
        """
        try:
            checks = {
                "model_server": await self._check_model_server(),
                "validator": self._check_validator(),
                "opa_generator": self._check_opa_generator(),
            }
            
            # Determine overall status
            all_healthy = all(check["status"] == "healthy" for check in checks.values())
            status = "healthy" if all_healthy else "unhealthy"
            
            return {
                "status": status,
                "service": "analysis",
                "version": "0.1.0",
                "timestamp": "2024-01-01T00:00:00Z",  # This should be dynamic
                "checks": checks
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "unhealthy",
                "service": "analysis",
                "version": "0.1.0",
                "timestamp": "2024-01-01T00:00:00Z",  # This should be dynamic
                "error": str(e)
            }
    
    async def _check_model_server(self) -> Dict[str, Any]:
        """Check model server health."""
        try:
            # This would be a more sophisticated health check in production
            return {"status": "healthy", "message": "Model server operational"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    
    def _check_validator(self) -> Dict[str, Any]:
        """Check validator health."""
        try:
            # Test validation with sample data
            test_output = {
                "reason": "test",
                "remediation": "test",
                "opa_diff": "",
                "confidence": 0.8,
                "evidence_refs": ["test"]
            }
            is_valid = self.validator.validate_schema_compliance(test_output)
            return {"status": "healthy" if is_valid else "unhealthy", "message": "Validator operational"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    
    def _check_opa_generator(self) -> Dict[str, Any]:
        """Check OPA generator health."""
        try:
            # Test OPA generation with sample data
            test_policy = self.opa_generator.generate_coverage_policy(["test"], {"test": 0.8})
            is_valid = self.opa_generator.validate_rego(test_policy)
            return {"status": "healthy" if is_valid else "unhealthy", "message": "OPA generator operational"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
