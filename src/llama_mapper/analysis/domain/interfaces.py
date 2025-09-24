"""
Domain interfaces for the Analysis Module.

This module defines the contracts and interfaces that domain services
and infrastructure components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .entities import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    VersionInfo,
)


class IModelServer(ABC):
    """Interface for model server implementations."""

    @property
    @abstractmethod
    def version_info(self) -> VersionInfo:
        """
        Get version information for the model server.

        Returns:
            VersionInfo object containing version information
        """
        pass

    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Generate analysis using the model.

        Args:
            request: Analysis request with structured metrics

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    async def analyze_batch(
        self, requests: List[AnalysisRequest], idempotency_key: str
    ) -> List[Dict[str, Any]]:
        """
        Process batch requests with idempotency support.

        Args:
            requests: List of analysis requests
            idempotency_key: Idempotency key for caching

        Returns:
            List of analysis results
        """
        pass


class IValidator(ABC):
    """Interface for validation implementations."""

    @abstractmethod
    def validate_and_fallback(
        self, model_output: Dict[str, Any], request: AnalysisRequest
    ) -> Dict[str, Any]:
        """
        Validate output and fallback to templates on failure.

        Args:
            model_output: Model output to validate
            request: Original analysis request

        Returns:
            Validated output or template fallback
        """
        pass

    @abstractmethod
    def validate_schema_compliance(self, output: Dict[str, Any]) -> bool:
        """
        Check if output complies with schema without raising exceptions.

        Args:
            output: Output to validate

        Returns:
            True if compliant, False otherwise
        """
        pass

    @abstractmethod
    def get_validation_errors(self, output: Dict[str, Any]) -> List[str]:
        """
        Get list of validation errors for output.

        Args:
            output: Output to validate

        Returns:
            List of error messages
        """
        pass


class ITemplateProvider(ABC):
    """Interface for template provider implementations."""

    @abstractmethod
    def get_template_response(
        self,
        request: AnalysisRequest,
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get template response for the specified analysis type.

        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback

        Returns:
            Template response dictionary
        """
        pass

    @abstractmethod
    def select_analysis_type(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select the most appropriate analysis type based on request content.

        Args:
            request: Analysis request

        Returns:
            Analysis type to use
        """
        pass


class IOPAGenerator(ABC):
    """Interface for OPA policy generator implementations."""

    @abstractmethod
    def validate_rego(self, rego_snippet: str) -> bool:
        """
        Validate Rego syntax using OPA compiler.

        Args:
            rego_snippet: Rego policy snippet to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def generate_coverage_policy(
        self, required_detectors: List[str], required_coverage: Dict[str, float]
    ) -> str:
        """
        Generate OPA policy for coverage violations.

        Args:
            required_detectors: List of required detector names
            required_coverage: Required coverage per detector

        Returns:
            OPA/Rego policy string
        """
        pass

    @abstractmethod
    def generate_threshold_policy(self, detector: str, new_threshold: float) -> str:
        """
        Generate OPA policy for threshold adjustments.

        Args:
            detector: Detector name
            new_threshold: New threshold value

        Returns:
            OPA/Rego policy string
        """
        pass


class ISecurityValidator(ABC):
    """Interface for security validation implementations."""

    @abstractmethod
    def validate_response_security(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and redact PII from analysis response.

        Args:
            response: Analysis response dictionary

        Returns:
            Response with PII redacted
        """
        pass

    @abstractmethod
    def validate_request_security(self, request_data: Dict[str, Any]) -> bool:
        """
        Validate request data for security issues.

        Args:
            request_data: Request data to validate

        Returns:
            True if secure, False otherwise
        """
        pass

    @abstractmethod
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers for API responses.

        Returns:
            Dictionary of security headers
        """
        pass


class IIdempotencyManager(ABC):
    """Interface for idempotency management implementations."""

    @abstractmethod
    async def get_cached_response(
        self, idempotency_key: str, request_data: Dict[str, Any]
    ) -> Optional[BatchAnalysisResponse]:
        """
        Get cached response for idempotency key.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data

        Returns:
            Cached response if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    async def cache_response(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any],
        response: BatchAnalysisResponse,
    ) -> None:
        """
        Cache response for idempotency key.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data
            response: Response to cache
        """
        pass

    @abstractmethod
    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        pass


class IQualityEvaluator(ABC):
    """Interface for quality evaluation implementations."""

    @abstractmethod
    async def evaluate_batch(
        self, examples: List[tuple[AnalysisRequest, AnalysisResponse]]
    ) -> Dict[str, Any]:
        """
        Evaluate batch of examples against golden dataset.

        Args:
            examples: List of (request, response) tuples to evaluate

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def calculate_drift_score(self, recent_outputs: List[AnalysisResponse]) -> float:
        """
        Calculate quality drift over time.

        Args:
            recent_outputs: Recent analysis outputs

        Returns:
            Drift score (0.0-1.0, higher is worse)
        """
        pass

    @abstractmethod
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation history.

        Returns:
            Evaluation summary
        """
        pass


class IReportGenerator(ABC):
    """Interface for report generation implementations."""

    @abstractmethod
    def generate_report(
        self,
        report_data: Any,
        format_type: str,
        tenant_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Generate a report in the specified format.

        Args:
            report_data: Data to include in the report
            format_type: Output format (PDF, CSV, or JSON)
            tenant_id: Optional tenant ID for multi-tenancy
            requested_by: Optional user who requested the report
            report_type: Optional report type identifier

        Returns:
            Report content as bytes (PDF), string (CSV), or dict (JSON)
        """
        pass


class IAlertingSystem(ABC):
    """Interface for alerting system implementations."""

    @abstractmethod
    async def send_evaluation_notification(
        self,
        recipient: str,
        report_path: str,
        quality_metrics: Any,
        alerts: List[str],
    ) -> bool:
        """
        Send evaluation notification to a recipient.

        Args:
            recipient: Email address or notification target
            report_path: Path to the evaluation report
            quality_metrics: Quality metrics from the evaluation
            alerts: List of alert messages

        Returns:
            True if notification sent successfully
        """
        pass


class IStorageBackend(ABC):
    """Interface for storage backend implementations."""

    @abstractmethod
    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        pass

    @abstractmethod
    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        pass

    @abstractmethod
    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        pass

    @abstractmethod
    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """
        Save evaluation report.

        Args:
            tenant_id: Tenant ID
            content: Report content
            report_type: Type of report

        Returns:
            Path to saved report
        """
        pass

    @abstractmethod
    async def get_evaluation_schedules(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        pass
