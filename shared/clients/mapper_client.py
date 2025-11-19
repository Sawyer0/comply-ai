"""HTTP client for Mapper Service."""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from ..exceptions.base import (
    AuthenticationError,
    BaseServiceException,
    ServiceUnavailableError,
    ValidationError,
)
from ..interfaces.mapper import (
    BatchMappingRequest,
    BatchMappingResponse,
    DeploymentExperiment,
    ExperimentRequest,
    FeatureFlag,
    FrameworkConfig,
    MappingMode,
    MappingRequest,
    MappingResponse,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
    ModelVersion,
    Taxonomy,
    TaxonomyRequest,
    TrainingJob,
    TrainingJobRequest,
    TrainingJobResponse,
    ValidationRequest,
    ValidationResult,
)
from ..interfaces.analysis import AnalysisRequest, AnalysisType
from ..interfaces.orchestration import OrchestrationResponse
from .analysis_client import AnalysisServiceClient, create_analysis_client
from ..utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class MapperServiceClient:
    """HTTP client for the Mapper Service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        api_key: Optional[str] = None,
        timeout: float = 120.0,  # Longer timeout for model operations
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers=self._get_default_headers(),
        )

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "llama-mapper-client/1.0.0",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def _get_request_headers(
        self, tenant_id: str, correlation_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Get headers for a specific request."""
        headers = {}
        headers["X-Tenant-ID"] = tenant_id

        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        else:
            headers["X-Correlation-ID"] = get_correlation_id()

        return headers

    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError:
            error_data: Dict[str, Any] = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text}

            # Normalize common fields for type safety
            message_val = error_data.get("message")
            message = str(message_val) if message_val is not None else f"HTTP {response.status_code}"

            error_code_val = error_data.get("error_code")
            error_code = str(error_code_val) if error_code_val is not None else "UNKNOWN_ERROR"

            correlation_id = error_data.get("correlation_id")

            details_raw = error_data.get("details")
            details = details_raw if isinstance(details_raw, dict) else None

            if response.status_code == 400:
                raise ValidationError(
                    message if message else "Bad request",
                    error_code=error_code or "BAD_REQUEST",
                    details=details,
                    correlation_id=correlation_id,
                )
            if response.status_code == 401:
                raise AuthenticationError(
                    message if message else "Unauthorized",
                    error_code=error_code or "UNAUTHORIZED",
                    correlation_id=correlation_id,
                )
            if response.status_code >= 500:
                raise ServiceUnavailableError(
                    message if message else "Service unavailable",
                    error_code=error_code or "SERVICE_UNAVAILABLE",
                    correlation_id=correlation_id,
                )
            raise BaseServiceException(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            response = await self.client.get("/health")
            return await self._handle_response(response)
        except Exception as e:
            logger.error("Health check failed: %s", e)
            raise

    async def map_content(
        self,
        request: MappingRequest,
        tenant_id: str,
        correlation_id: Optional[str] = None,
    ) -> MappingResponse:
        """Perform core mapping functionality."""
        try:
            headers = self._get_request_headers(tenant_id, correlation_id)

            response = await self.client.post(
                "/api/v1/map", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return MappingResponse(**data)

        except Exception as e:
            logger.error(
                "Mapping failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "target_frameworks": request.target_frameworks,
                },
            )
            raise

    async def analyze_and_map_from_orchestration(
        self,
        orchestration_response: OrchestrationResponse,
        tenant_id: str,
        target_frameworks: List[str],
        analysis_types: Optional[List[AnalysisType]] = None,
        correlation_id: Optional[str] = None,
        analysis_client: Optional[AnalysisServiceClient] = None,
    ) -> MappingResponse:
        """Run the canonical orchestration -> analysis -> mapping flow.

        This helper:
        1. Builds a shared AnalysisRequest from an OrchestrationResponse
           (which carries canonical_outputs).
        2. Calls the analysis-service /api/v1/analyze endpoint to obtain a
           shared AnalysisResponse with canonical_results populated.
        3. Builds a shared MappingRequest using that AnalysisResponse and
           the requested target frameworks.
        4. Calls the mapper-service /api/v1/map endpoint and returns the
           shared MappingResponse.
        """

        own_client = False
        if analysis_client is None:
            analysis_client = create_analysis_client()
            own_client = True

        try:
            # Default to risk + compliance analysis if not specified
            if not analysis_types:
                analysis_types = [
                    AnalysisType.RISK,
                    AnalysisType.COMPLIANCE,
                ]

            # Shared AnalysisRequest only carries orchestration_response,
            # analysis_types, frameworks, include_recommendations, context.
            analysis_request = AnalysisRequest(
                orchestration_response=orchestration_response,
                analysis_types=analysis_types,
                frameworks=target_frameworks,
                include_recommendations=True,
                context=None,
            )

            analysis_response = await analysis_client.analyze_content(
                request=analysis_request,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Shared MappingRequest expects an AnalysisResponse plus frameworks
            mapping_request = MappingRequest(
                analysis_response=analysis_response,
                target_frameworks=target_frameworks,
                mapping_mode=MappingMode.STANDARD,
                include_validation=True,
                context=None,
            )

            return await self.map_content(
                request=mapping_request,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

        finally:
            if own_client:
                await analysis_client.close()

    async def batch_map(
        self,
        request: BatchMappingRequest,
        tenant_id: str,
    ) -> BatchMappingResponse:
        """Perform batch mapping operations."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/batch-map", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return BatchMappingResponse(**data)

        except Exception as e:
            logger.error(
                "Batch mapping failed: %s",
                e,
                extra={"tenant_id": tenant_id, "batch_size": len(request.requests)},
            )
            raise

    async def validate_mapping(
        self,
        request: ValidationRequest,
        tenant_id: str,
    ) -> ValidationResult:
        """Validate input/output data."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/validate", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return ValidationResult(**data)

        except Exception as e:
            logger.error(
                "Validation failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "validation_type": request.validation_type,
                },
            )
            raise

    async def list_models(
        self,
        tenant_id: str,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelVersion]:
        """List available models."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if model_type:
                params["model_type"] = model_type
            if status:
                params["status"] = status

            response = await self.client.get(
                "/api/v1/models", params=params, headers=headers
            )

            data = await self._handle_response(response)
            return [ModelVersion.parse_obj(item) for item in data]

        except Exception as e:
            logger.error("List models failed: %s", e, extra={"tenant_id": tenant_id})
            raise

    async def deploy_model(
        self,
        model_id: str,
        request: ModelDeploymentRequest,
        tenant_id: str,
    ) -> ModelDeploymentResponse:
        """Deploy model version."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                f"/api/v1/models/{model_id}/deploy",
                json=request.dict(),
                headers=headers,
            )

            data = await self._handle_response(response)
            return ModelDeploymentResponse(**data)

        except Exception as e:
            logger.error(
                "Model deployment failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "model_id": model_id,
                    "deployment_type": request.deployment_type,
                },
            )
            raise

    async def submit_training_job(
        self,
        request: TrainingJobRequest,
        tenant_id: str,
    ) -> TrainingJobResponse:
        """Submit training job."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/training/jobs", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return TrainingJobResponse(**data)

        except Exception as e:
            logger.error(
                "Training job submission failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "job_name": request.job_name,
                    "training_type": request.training_type,
                },
            )
            raise

    async def list_training_jobs(
        self,
        tenant_id: str,
        status: Optional[str] = None,
    ) -> List[TrainingJob]:
        """List training jobs."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if status:
                params["status"] = status

            response = await self.client.get(
                "/api/v1/training/jobs", params=params, headers=headers
            )

            data = await self._handle_response(response)
            return [TrainingJob.parse_obj(item) for item in data]

        except Exception as e:
            logger.error(
                "List training jobs failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def get_training_job_status(
        self,
        job_id: str,
        tenant_id: str,
    ) -> TrainingJob:
        """Get training job status."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.get(
                f"/api/v1/training/jobs/{job_id}", headers=headers
            )

            data = await self._handle_response(response)
            return TrainingJob(**data)

        except Exception as e:
            logger.error(
                "Get training job status failed: %s",
                e,
                extra={"tenant_id": tenant_id, "job_id": job_id},
            )
            raise

    async def create_experiment(
        self,
        request: ExperimentRequest,
        tenant_id: str,
    ) -> DeploymentExperiment:
        """Create A/B testing experiment."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/experiments", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return DeploymentExperiment(**data)

        except Exception as e:
            logger.error(
                "Experiment creation failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "experiment_name": request.experiment_name,
                },
            )
            raise

    async def list_experiments(
        self,
        tenant_id: str,
        status: Optional[str] = None,
    ) -> List[DeploymentExperiment]:
        """List experiments."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if status:
                params["status"] = status

            response = await self.client.get(
                "/api/v1/experiments", params=params, headers=headers
            )

            data = await self._handle_response(response)
            return [DeploymentExperiment.parse_obj(item) for item in data]

        except Exception as e:
            logger.error(
                "List experiments failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def list_taxonomies(
        self,
        tenant_id: str,
        is_active: Optional[bool] = None,
    ) -> List[Taxonomy]:
        """List taxonomies."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if is_active is not None:
                params["is_active"] = is_active

            response = await self.client.get(
                "/api/v1/taxonomies", params=params, headers=headers
            )

            data = await self._handle_response(response)
            return [Taxonomy.parse_obj(item) for item in data]

        except Exception as e:
            logger.error(
                "List taxonomies failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def create_taxonomy(
        self,
        request: TaxonomyRequest,
        tenant_id: str,
    ) -> Taxonomy:
        """Create taxonomy."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/taxonomies", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return Taxonomy(**data)

        except Exception as e:
            logger.error(
                "Taxonomy creation failed: %s",
                e,
                extra={
                    "tenant_id": tenant_id,
                    "taxonomy_name": request.taxonomy_name,
                },
            )
            raise

    async def list_frameworks(
        self,
        tenant_id: str,
        is_active: Optional[bool] = None,
    ) -> List[FrameworkConfig]:
        """List framework configurations."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if is_active is not None:
                params["is_active"] = is_active

            response = await self.client.get(
                "/api/v1/frameworks", params=params, headers=headers
            )

            data = await self._handle_response(response)
            return [FrameworkConfig.parse_obj(item) for item in data]

        except Exception as e:
            logger.error(
                "List frameworks failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def get_cost_metrics(
        self,
        tenant_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get cost metrics."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()

            response = await self.client.get(
                "/api/v1/cost/metrics", params=params, headers=headers
            )

            data = await self._handle_response(response)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []

        except Exception as e:
            logger.error(
                "Get cost metrics failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def list_feature_flags(
        self,
        tenant_id: str,
    ) -> List[FeatureFlag]:
        """List feature flags."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.get("/api/v1/feature-flags", headers=headers)

            data = await self._handle_response(response)
            return [FeatureFlag.parse_obj(item) for item in data]

        except Exception as e:
            logger.error(
                "List feature flags failed: %s", e, extra={"tenant_id": tenant_id}
            )
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience function for creating client instances
def create_mapper_client(
    base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> MapperServiceClient:
    """Create a mapper service client with default configuration."""
    if base_url is None:
        base_url = os.getenv("MAPPER_SERVICE_URL", "http://localhost:8002")

    if api_key is None:
        api_key = os.getenv("MAPPER_API_KEY")

    return MapperServiceClient(base_url=base_url, api_key=api_key, **kwargs)
