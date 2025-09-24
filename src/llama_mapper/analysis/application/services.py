"""
Application services for the Analysis Module.

This module contains application services that coordinate between
use cases and provide higher-level orchestration.
"""

import logging
from typing import Any, Dict, List

from ..domain.entities import VersionInfo
from .dto import (
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
    HealthCheckDTO,
)
from .use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
    QualityEvaluationUseCase,
)

logger = logging.getLogger(__name__)


class AnalysisApplicationService:
    """
    Application service for analysis operations.

    Provides high-level orchestration for analysis use cases
    and coordinates between different application components.
    """

    def __init__(
        self,
        analyze_metrics_use_case: AnalyzeMetricsUseCase,
        batch_analyze_metrics_use_case: BatchAnalyzeMetricsUseCase,
        quality_evaluation_use_case: QualityEvaluationUseCase,
        cache_management_use_case: CacheManagementUseCase,
        health_check_use_case: HealthCheckUseCase,
    ):
        """
        Initialize the analysis application service.

        Args:
            analyze_metrics_use_case: Use case for single analysis
            batch_analyze_metrics_use_case: Use case for batch analysis
            quality_evaluation_use_case: Use case for quality evaluation
            cache_management_use_case: Use case for cache management
            health_check_use_case: Use case for health checks
        """
        self.analyze_metrics_use_case = analyze_metrics_use_case
        self.batch_analyze_metrics_use_case = batch_analyze_metrics_use_case
        self.quality_evaluation_use_case = quality_evaluation_use_case
        self.cache_management_use_case = cache_management_use_case
        self.health_check_use_case = health_check_use_case

    async def analyze_metrics(
        self, request_dto: AnalysisRequestDTO
    ) -> AnalysisResponseDTO:
        """
        Analyze metrics for a single request.

        Args:
            request_dto: Analysis request DTO

        Returns:
            Analysis response DTO
        """
        try:
            return await self.analyze_metrics_use_case.execute(request_dto)
        except Exception as e:
            logger.error("Analysis application service error: %s", e)
            raise

    async def batch_analyze_metrics(
        self, request_dto: BatchAnalysisRequestDTO, idempotency_key: str
    ) -> BatchAnalysisResponseDTO:
        """
        Analyze metrics for a batch of requests.

        Args:
            request_dto: Batch analysis request DTO
            idempotency_key: Idempotency key for caching

        Returns:
            Batch analysis response DTO
        """
        try:
            return await self.batch_analyze_metrics_use_case.execute(
                request_dto, idempotency_key
            )
        except Exception as e:
            logger.error("Batch analysis application service error: %s", e)
            raise

    async def evaluate_quality(
        self, examples: List[tuple[AnalysisRequestDTO, AnalysisResponseDTO]]
    ) -> Dict[str, Any]:
        """
        Evaluate quality of analysis outputs.

        Args:
            examples: List of (request, response) DTO tuples

        Returns:
            Quality evaluation metrics
        """
        try:
            # Convert DTOs to domain entities for evaluation
            domain_examples = []
            for req_dto, resp_dto in examples:
                domain_examples.append(
                    (req_dto.to_domain_entity(), resp_dto.to_domain_entity())
                )

            return await self.quality_evaluation_use_case.evaluate_quality(
                domain_examples
            )
        except Exception as e:
            logger.error("Quality evaluation application service error: %s", e)
            raise

    def calculate_drift(self, recent_outputs: List[AnalysisResponseDTO]) -> float:
        """
        Calculate quality drift over time.

        Args:
            recent_outputs: Recent analysis response DTOs

        Returns:
            Drift score
        """
        try:
            # Convert DTOs to domain entities for drift calculation
            domain_outputs = [
                resp_dto.to_domain_entity() for resp_dto in recent_outputs
            ]
            return self.quality_evaluation_use_case.calculate_drift(domain_outputs)
        except Exception as e:
            logger.error("Drift calculation application service error: %s", e)
            raise

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary.

        Returns:
            Evaluation summary
        """
        try:
            return self.quality_evaluation_use_case.get_evaluation_summary()
        except Exception as e:
            logger.error("Evaluation summary application service error: %s", e)
            raise

    async def cleanup_cache(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        try:
            return await self.cache_management_use_case.cleanup_expired_cache()
        except Exception as e:
            logger.error("Cache cleanup application service error: %s", e)
            raise

    async def health_check(self) -> HealthCheckDTO:
        """
        Perform health check.

        Returns:
            Health check DTO
        """
        try:
            health_data = await self.health_check_use_case.execute()
            return HealthCheckDTO(**health_data)
        except Exception as e:
            logger.error("Health check application service error: %s", e)
            raise


class BatchAnalysisApplicationService:
    """
    Application service for batch analysis operations.

    Provides specialized orchestration for batch processing
    scenarios and handles batch-specific concerns.
    """

    def __init__(
        self,
        batch_analyze_metrics_use_case: BatchAnalyzeMetricsUseCase,
        cache_management_use_case: CacheManagementUseCase,
    ):
        """
        Initialize the batch analysis application service.

        Args:
            batch_analyze_metrics_use_case: Use case for batch analysis
            cache_management_use_case: Use case for cache management
        """
        self.batch_analyze_metrics_use_case = batch_analyze_metrics_use_case
        self.cache_management_use_case = cache_management_use_case

    async def process_batch(
        self, request_dto: BatchAnalysisRequestDTO, idempotency_key: str
    ) -> BatchAnalysisResponseDTO:
        """
        Process batch analysis request.

        Args:
            request_dto: Batch analysis request DTO
            idempotency_key: Idempotency key for caching

        Returns:
            Batch analysis response DTO
        """
        try:
            return await self.batch_analyze_metrics_use_case.execute(
                request_dto, idempotency_key
            )
        except Exception as e:
            logger.error("Batch processing application service error: %s", e)
            raise

    async def cleanup_batch_cache(self) -> int:
        """
        Clean up batch-related cache entries.

        Returns:
            Number of entries cleaned up
        """
        try:
            return await self.cache_management_use_case.cleanup_expired_cache()
        except Exception as e:
            logger.error("Batch cache cleanup application service error: %s", e)
            raise
