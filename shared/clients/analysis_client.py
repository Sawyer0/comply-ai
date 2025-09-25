"""HTTP client for Analysis Service."""

import httpx
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, date
import uuid

from ..interfaces.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    PatternAnalysisResult,
    RiskScoringResult,
    ComplianceMappingResult,
    RAGInsights,
    QualityMetrics,
)
from ..utils.correlation import get_correlation_id
from ..exceptions.base import (
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


class AnalysisServiceClient:
    """HTTP client for the Analysis Service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        api_key: Optional[str] = None,
        timeout: float = 60.0,  # Longer timeout for analysis operations
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
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text}

            if response.status_code == 400:
                raise ValidationError(
                    error_data.get("message", "Bad request"),
                    error_code=error_data.get("error_code"),
                    details=error_data.get("details"),
                    correlation_id=error_data.get("correlation_id"),
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    error_data.get("message", "Unauthorized"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )
            elif response.status_code >= 500:
                raise ServiceUnavailableError(
                    error_data.get("message", "Service unavailable"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )
            else:
                raise BaseServiceException(
                    error_data.get("message", f"HTTP {response.status_code}"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            response = await self.client.get("/health")
            return await self._handle_response(response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise

    async def analyze_content(
        self,
        request: AnalysisRequest,
        tenant_id: str,
        correlation_id: Optional[str] = None,
    ) -> AnalysisResponse:
        """Perform comprehensive analysis."""
        try:
            headers = self._get_request_headers(tenant_id, correlation_id)

            response = await self.client.post(
                "/api/v1/analyze", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return AnalysisResponse(**data)

        except Exception as e:
            logger.error(
                f"Analysis failed: {e}",
                extra={
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "analysis_types": request.analysis_types,
                },
            )
            raise

    async def analyze_patterns(
        self,
        detector_results: List[Dict[str, Any]],
        tenant_id: str,
        time_window_hours: int = 24,
    ) -> PatternAnalysisResult:
        """Perform pattern recognition analysis."""
        try:
            headers = self._get_request_headers(tenant_id)

            request_data = {
                "detector_results": detector_results,
                "time_window_hours": time_window_hours,
            }

            response = await self.client.post(
                "/api/v1/patterns", json=request_data, headers=headers
            )

            data = await self._handle_response(response)
            return PatternAnalysisResult(**data)

        except Exception as e:
            logger.error(
                f"Pattern analysis failed: {e}",
                extra={"tenant_id": tenant_id, "detector_count": len(detector_results)},
            )
            raise

    async def calculate_risk_score(
        self,
        canonical_results: List[Dict[str, Any]],
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskScoringResult:
        """Calculate risk scores."""
        try:
            headers = self._get_request_headers(tenant_id)

            request_data = {
                "canonical_results": canonical_results,
                "context": context or {},
            }

            response = await self.client.post(
                "/api/v1/risk-score", json=request_data, headers=headers
            )

            data = await self._handle_response(response)
            return RiskScoringResult(**data)

        except Exception as e:
            logger.error(
                f"Risk scoring failed: {e}",
                extra={"tenant_id": tenant_id, "results_count": len(canonical_results)},
            )
            raise

    async def map_compliance(
        self,
        canonical_results: List[Dict[str, Any]],
        frameworks: List[str],
        tenant_id: str,
    ) -> List[ComplianceMappingResult]:
        """Map to compliance frameworks."""
        try:
            headers = self._get_request_headers(tenant_id)

            request_data = {
                "canonical_results": canonical_results,
                "frameworks": frameworks,
            }

            response = await self.client.post(
                "/api/v1/compliance", json=request_data, headers=headers
            )

            data = await self._handle_response(response)
            return [ComplianceMappingResult(**item) for item in data]

        except Exception as e:
            logger.error(
                f"Compliance mapping failed: {e}",
                extra={"tenant_id": tenant_id, "frameworks": frameworks},
            )
            raise

    async def rag_query(
        self,
        query_text: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 5,
    ) -> RAGInsights:
        """Perform RAG-enhanced query."""
        try:
            headers = self._get_request_headers(tenant_id)

            request_data = {
                "query_text": query_text,
                "context": context or {},
                "max_results": max_results,
            }

            response = await self.client.post(
                "/api/v1/rag/query", json=request_data, headers=headers
            )

            data = await self._handle_response(response)
            return RAGInsights(**data)

        except Exception as e:
            logger.error(
                f"RAG query failed: {e}",
                extra={"tenant_id": tenant_id, "query_length": len(query_text)},
            )
            raise

    async def get_quality_metrics(
        self,
        tenant_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        metric_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get quality metrics."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            if metric_type:
                params["metric_type"] = metric_type

            response = await self.client.get(
                "/api/v1/quality/metrics", params=params, headers=headers
            )

            return await self._handle_response(response)

        except Exception as e:
            logger.error(
                f"Get quality metrics failed: {e}", extra={"tenant_id": tenant_id}
            )
            raise

    async def get_quality_alerts(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get quality alerts."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if status:
                params["status"] = status
            if severity:
                params["severity"] = severity

            response = await self.client.get(
                "/api/v1/quality/alerts", params=params, headers=headers
            )

            return await self._handle_response(response)

        except Exception as e:
            logger.error(
                f"Get quality alerts failed: {e}", extra={"tenant_id": tenant_id}
            )
            raise

    async def list_models(
        self,
        tenant_id: str,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List ML models."""
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

            return await self._handle_response(response)

        except Exception as e:
            logger.error(f"List models failed: {e}", extra={"tenant_id": tenant_id})
            raise

    async def get_weekly_evaluations(
        self,
        tenant_id: str,
        week: Optional[date] = None,
        model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get weekly evaluations."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if week:
                params["week"] = week.isoformat()
            if model_version:
                params["model_version"] = model_version

            response = await self.client.get(
                "/api/v1/evaluations/weekly", params=params, headers=headers
            )

            return await self._handle_response(response)

        except Exception as e:
            logger.error(
                f"Get weekly evaluations failed: {e}", extra={"tenant_id": tenant_id}
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
def create_analysis_client(
    base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> AnalysisServiceClient:
    """Create an analysis service client with default configuration."""
    import os

    if base_url is None:
        base_url = os.getenv("ANALYSIS_SERVICE_URL", "http://localhost:8001")

    if api_key is None:
        api_key = os.getenv("ANALYSIS_API_KEY")

    return AnalysisServiceClient(base_url=base_url, api_key=api_key, **kwargs)
