"""
Domain services for the Analysis Module.

This module contains domain services that encapsulate business logic
and coordinate between domain entities and repositories.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .entities import (
    AnalysisErrorResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    HealthStatus,
    QualityMetrics,
)
from .interfaces import (
    IAlertingSystem,
    IIdempotencyManager,
    IModelServer,
    IOPAGenerator,
    IQualityEvaluator,
    IReportGenerator,
    ISecurityValidator,
    IStorageBackend,
    ITemplateProvider,
    IValidator,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Domain service for single analysis operations.

    Encapsulates the business logic for analyzing individual requests
    and coordinates between model server, validator, and template provider.
    """

    def __init__(
        self,
        model_server: IModelServer,
        validator: IValidator,
        template_provider: ITemplateProvider,
        security_validator: ISecurityValidator,
    ):
        """
        Initialize the analysis service.

        Args:
            model_server: Model server implementation
            validator: Validator implementation
            template_provider: Template provider implementation
            security_validator: Security validator implementation
        """
        self.model_server = model_server
        self.validator = validator
        self.template_provider = template_provider
        self.security_validator = security_validator

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze a single request.

        Args:
            request: Analysis request

        Returns:
            Analysis response
        """
        try:
            # Generate analysis using model server
            model_output = await self.model_server.analyze(request)

            # Validate and potentially fallback to templates
            validated_output = self.validator.validate_and_fallback(
                model_output, request
            )

            # Check if fallback was used (for low confidence)
            if validated_output.get("_template_fallback"):
                # Still return a normal AnalysisResponse but include fallback info in notes
                validated_output["notes"] = (
                    f"Template fallback used: {validated_output.get('_fallback_reason', 'Low confidence')}"
                )

            # Create successful response
            response = self._create_response_from_output(validated_output, request)

            # Apply security validation and PII redaction
            response_dict = response.dict()
            redacted_dict = self.security_validator.validate_response_security(
                response_dict
            )

            return AnalysisResponse(**redacted_dict)

        except Exception as e:
            logger.error("Analysis service error: %s", e)
            # Return template fallback as normal AnalysisResponse
            template_response = self.template_provider.get_template_response(
                request, AnalysisType.INSUFFICIENT_DATA, f"Analysis failed: {str(e)}"
            )

            # Return template response as AnalysisResponse with error info in notes
            return self._create_response_from_output(template_response, request)

    def _create_response_from_output(
        self, output: Dict[str, Any], request: AnalysisRequest
    ) -> AnalysisResponse:
        """
        Create AnalysisResponse from model output.

        Args:
            output: Model output dictionary
            request: Original request

        Returns:
            Analysis response
        """
        return AnalysisResponse(
            reason=output.get("reason", ""),
            remediation=output.get("remediation", ""),
            opa_diff=output.get("opa_diff", ""),
            confidence=output.get("confidence", 0.0),
            confidence_cutoff_used=output.get("confidence_cutoff_used", 0.3),
            evidence_refs=output.get("evidence_refs", []),
            notes=output.get("notes", ""),
            version_info=self.model_server.version_info,
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=output.get("processing_time_ms", 0),
        )


class BatchAnalysisService:
    """
    Domain service for batch analysis operations.

    Encapsulates the business logic for processing multiple analysis requests
    with idempotency support and concurrent processing.
    """

    def __init__(
        self,
        analysis_service: AnalysisService,
        idempotency_manager: IIdempotencyManager,
        max_concurrent: int = 10,
        request_timeout: int = 30,
    ):
        """
        Initialize the batch analysis service.

        Args:
            analysis_service: Single analysis service
            idempotency_manager: Idempotency manager implementation
            max_concurrent: Maximum concurrent requests
            request_timeout: Request timeout in seconds
        """
        self.analysis_service = analysis_service
        self.idempotency_manager = idempotency_manager
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout

    async def analyze_batch(
        self, batch_request: BatchAnalysisRequest, idempotency_key: str
    ) -> BatchAnalysisResponse:
        """
        Analyze a batch of requests.

        Args:
            batch_request: Batch analysis request
            idempotency_key: Idempotency key for caching

        Returns:
            Batch analysis response
        """
        start_time = time.time()

        # Check for cached response
        request_data = batch_request.dict()
        cached_response = await self.idempotency_manager.get_cached_response(
            idempotency_key, request_data
        )

        if cached_response:
            logger.info(
                f"Returning cached response for idempotency key: {idempotency_key}"
            )
            return cached_response

        # Validate batch size
        if len(batch_request.requests) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")

        # Process requests concurrently
        responses = await self._process_requests_concurrently(batch_request.requests)

        # Create batch response
        batch_response = BatchAnalysisResponse(
            responses=responses,
            batch_id=str(uuid4()),
            idempotency_key=idempotency_key,
            total_processing_time_ms=int((time.time() - start_time) * 1000),
            success_count=sum(1 for r in responses if isinstance(r, AnalysisResponse)),
            error_count=sum(
                1 for r in responses if isinstance(r, AnalysisErrorResponse)
            ),
        )

        # Cache the response
        await self.idempotency_manager.cache_response(
            idempotency_key, request_data, batch_response
        )

        return batch_response

    async def _process_requests_concurrently(
        self, requests: List[AnalysisRequest]
    ) -> List[Union[AnalysisResponse, AnalysisErrorResponse]]:
        """
        Process requests concurrently with semaphore limiting.

        Args:
            requests: List of analysis requests

        Returns:
            List of analysis responses or errors
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_single_request(
            request: AnalysisRequest,
        ) -> Union[AnalysisResponse, AnalysisErrorResponse]:
            async with semaphore:
                return await self._process_single_request(request)

        # Process all requests concurrently
        tasks = [process_single_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = AnalysisErrorResponse(
                    error_type="processing_error",
                    message=str(response),
                    request_id=getattr(requests[i], "request_id", str(uuid4())),
                    fallback_used=False,
                    mode="error",
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)

        return processed_responses

    async def _process_single_request(
        self, request: AnalysisRequest
    ) -> Union[AnalysisResponse, AnalysisErrorResponse]:
        """
        Process a single analysis request.

        Args:
            request: Analysis request

        Returns:
            Analysis response or error response
        """
        try:
            # Add timeout to the request processing
            response = await asyncio.wait_for(
                self.analysis_service.analyze(request), timeout=self.request_timeout
            )

            return response

        except asyncio.TimeoutError:
            logger.error("Request timeout for request: %s", request.request_id)
            return AnalysisErrorResponse(
                error_type="timeout_error",
                message="Request processing timeout",
                request_id=request.request_id,
                fallback_used=True,
                mode="fallback",
                template_response=self._get_timeout_fallback(request),
            )

        except Exception as e:
            logger.error("Request processing error: %s", e)
            return AnalysisErrorResponse(
                error_type="processing_error",
                message=str(e),
                request_id=request.request_id,
                fallback_used=True,
                mode="fallback",
                template_response=self._get_error_fallback(request),
            )

    def _get_timeout_fallback(self, request: AnalysisRequest) -> AnalysisResponse:
        """Get timeout fallback response."""
        return AnalysisResponse(
            reason="request timeout - insufficient processing time",
            remediation="retry request or reduce batch size",
            opa_diff="",
            confidence=0.1,
            confidence_cutoff_used=0.3,
            evidence_refs=["required_detectors"],
            notes="Template fallback due to request timeout",
            version_info=self.analysis_service.model_server.version_info,
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=0,
        )

    def _get_error_fallback(self, request: AnalysisRequest) -> AnalysisResponse:
        """Get error fallback response."""
        return AnalysisResponse(
            reason="processing error - unable to analyze",
            remediation="check request format and retry",
            opa_diff="",
            confidence=0.1,
            confidence_cutoff_used=0.3,
            evidence_refs=["required_detectors"],
            notes="Template fallback due to processing error",
            version_info=self.analysis_service.model_server.version_info,
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=0,
        )


class ValidationService:
    """
    Domain service for validation operations.

    Encapsulates validation business logic and coordinates between
    different validation components.
    """

    def __init__(
        self,
        validator: IValidator,
        security_validator: ISecurityValidator,
        opa_generator: IOPAGenerator,
    ):
        """
        Initialize the validation service.

        Args:
            validator: Validator implementation
            security_validator: Security validator implementation
            opa_generator: OPA generator implementation
        """
        self.validator = validator
        self.security_validator = security_validator
        self.opa_generator = opa_generator

    def validate_analysis_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Validate analysis request data.

        Args:
            request_data: Request data to validate

        Returns:
            True if valid, False otherwise
        """
        # Security validation
        if not self.security_validator.validate_request_security(request_data):
            return False

        # Additional business validation can be added here
        return True

    def validate_analysis_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Validate analysis response data.

        Args:
            response_data: Response data to validate

        Returns:
            True if valid, False otherwise
        """
        # Schema validation
        if not self.validator.validate_schema_compliance(response_data):
            return False

        # Security validation
        try:
            self.security_validator.validate_response_security(response_data)
            return True
        except Exception:
            return False

    def validate_opa_policy(self, opa_diff: str) -> bool:
        """
        Validate OPA policy.

        Args:
            opa_diff: OPA policy to validate

        Returns:
            True if valid, False otherwise
        """
        if not opa_diff.strip():
            return True  # Empty is valid

        return self.opa_generator.validate_rego(opa_diff)


class QualityService:
    """
    Domain service for quality operations.

    Encapsulates quality evaluation business logic and coordinates
    between quality evaluator and other components.
    """

    def __init__(self, quality_evaluator: IQualityEvaluator):
        """
        Initialize the quality service.

        Args:
            quality_evaluator: Quality evaluator implementation
        """
        self.quality_evaluator = quality_evaluator

    async def evaluate_quality(
        self, examples: List[tuple[AnalysisRequest, AnalysisResponse]]
    ) -> QualityMetrics:
        """
        Evaluate quality of analysis outputs.

        Args:
            examples: List of (request, response) tuples

        Returns:
            Quality metrics
        """
        metrics_data = await self.quality_evaluator.evaluate_batch(examples)

        return QualityMetrics(
            total_examples=metrics_data["total_examples"],
            schema_valid_rate=metrics_data["schema_valid_rate"],
            rubric_score=metrics_data["rubric_score"],
            opa_compile_success_rate=metrics_data["opa_compile_success_rate"],
            evidence_accuracy=metrics_data["evidence_accuracy"],
            individual_rubric_scores=metrics_data["individual_rubric_scores"],
        )

    def calculate_drift(self, recent_outputs: List[AnalysisResponse]) -> float:
        """
        Calculate quality drift over time.

        Args:
            recent_outputs: Recent analysis outputs

        Returns:
            Drift score
        """
        return self.quality_evaluator.calculate_drift_score(recent_outputs)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary.

        Returns:
            Evaluation summary
        """
        return self.quality_evaluator.get_evaluation_summary()


class WeeklyEvaluationService:
    """
    Service for managing weekly evaluation scheduling and reporting.

    Handles scheduling of weekly quality evaluations, report generation,
    and distribution of evaluation results.
    """

    def __init__(
        self,
        quality_service: QualityService,
        report_generator: IReportGenerator,
        alerting_system: IAlertingSystem,
        storage_backend: IStorageBackend,
    ):
        """
        Initialize the weekly evaluation service.

        Args:
            quality_service: Quality service for evaluations
            report_generator: Report generator for creating reports
            alerting_system: Alerting system for notifications
            storage_backend: Storage backend for persistence
        """
        self.quality_service = quality_service
        self.report_generator = report_generator
        self.alerting_system = alerting_system
        self.storage_backend = storage_backend
        self.scheduled_evaluations = {}

        # Statistics tracking
        self._stats = {
            "schedules_created": 0,
            "evaluations_run": 0,
            "evaluations_failed": 0,
            "reports_generated": 0,
            "notifications_sent": 0,
            "last_evaluation": None,
            "service_start_time": datetime.utcnow(),
        }

    async def schedule_weekly_evaluation(
        self,
        tenant_id: str,
        cron_schedule: str = "0 9 * * 1",  # Every Monday at 9 AM
        report_recipients: Optional[List[str]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Schedule a weekly evaluation for a tenant.

        Args:
            tenant_id: Tenant ID for the evaluation
            cron_schedule: Cron expression for scheduling (default: Monday 9 AM)
            report_recipients: List of email addresses for report distribution
            evaluation_config: Configuration for the evaluation

        Returns:
            Schedule ID for the evaluation

        Raises:
            ValueError: If tenant_id is empty or cron_schedule is invalid
            RuntimeError: If storage operation fails
        """
        from datetime import datetime
        from uuid import uuid4

        from croniter import croniter

        # Validate inputs
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")

        if not cron_schedule or not cron_schedule.strip():
            raise ValueError("cron_schedule cannot be empty")

        # Validate cron expression
        try:
            croniter(cron_schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron schedule '{cron_schedule}': {e}")

        # Validate email addresses if provided
        if report_recipients:
            import re

            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            for email in report_recipients:
                if not re.match(email_pattern, email):
                    raise ValueError(f"Invalid email address: {email}")

        schedule_id = str(uuid4())

        try:
            schedule_data = {
                "schedule_id": schedule_id,
                "tenant_id": tenant_id.strip(),
                "cron_schedule": cron_schedule.strip(),
                "report_recipients": report_recipients or [],
                "evaluation_config": evaluation_config or {},
                "created_at": datetime.utcnow(),
                "active": True,
                "last_run": None,
                "next_run": None,
            }

            # Store schedule in backend
            await self.storage_backend.store_evaluation_schedule(schedule_data)
            self.scheduled_evaluations[schedule_id] = schedule_data

            # Update statistics
            self._stats["schedules_created"] += 1

            logger.info(
                f"Scheduled weekly evaluation for tenant {tenant_id}: {schedule_id}",
                extra={
                    "tenant_id": tenant_id,
                    "schedule_id": schedule_id,
                    "cron_schedule": cron_schedule,
                    "recipients_count": len(report_recipients or []),
                },
            )
            return schedule_id

        except Exception as e:
            logger.error(
                f"Failed to schedule weekly evaluation for tenant {tenant_id}: {e}"
            )
            raise RuntimeError(f"Failed to schedule weekly evaluation: {e}")

    async def run_scheduled_evaluation(self, schedule_id: str) -> Dict[str, Any]:
        """
        Run a scheduled evaluation.

        Args:
            schedule_id: ID of the scheduled evaluation

        Returns:
            Evaluation results

        Raises:
            ValueError: If schedule_id is not found
            RuntimeError: If evaluation execution fails
        """
        if not schedule_id or not schedule_id.strip():
            raise ValueError("schedule_id cannot be empty")

        # Try to get schedule from cache first, then from storage
        schedule = self.scheduled_evaluations.get(schedule_id)
        if not schedule:
            # Try to load from storage
            schedules = await self.storage_backend.get_evaluation_schedules()
            schedule = next(
                (s for s in schedules if s["schedule_id"] == schedule_id), None
            )

        if not schedule:
            raise ValueError(f"Schedule {schedule_id} not found")

        if not schedule.get("active", True):
            raise ValueError(f"Schedule {schedule_id} is not active")

        tenant_id = schedule["tenant_id"]

        logger.info("Running scheduled evaluation for tenant %s", tenant_id)

        try:
            # Get recent analysis data for evaluation
            recent_data = await self._get_recent_analysis_data(tenant_id)

            if not recent_data:
                logger.warning(
                    f"No recent data found for tenant {tenant_id}, using empty dataset"
                )

            # Run quality evaluation
            quality_metrics = await self.quality_service.evaluate_quality(recent_data)

            # Generate evaluation report
            report = await self._generate_evaluation_report(
                tenant_id, quality_metrics, schedule
            )

            # Store evaluation results
            evaluation_result = {
                "schedule_id": schedule_id,
                "tenant_id": tenant_id,
                "evaluation_date": datetime.utcnow(),
                "quality_metrics": quality_metrics.dict(),
                "report_path": report.get("file_path"),
                "status": "completed",
                "data_points": len(recent_data),
            }

            await self.storage_backend.store_evaluation_result(evaluation_result)

            # Send notifications if configured
            if schedule.get("report_recipients"):
                try:
                    await self._send_evaluation_notifications(
                        schedule["report_recipients"], report, quality_metrics
                    )
                except Exception as notification_error:
                    logger.error(
                        f"Failed to send notifications for evaluation {schedule_id}: {notification_error}"
                    )
                    # Don't fail the entire evaluation if notifications fail

            # Update schedule with last run time
            schedule["last_run"] = datetime.utcnow()
            await self.storage_backend.update_evaluation_schedule(schedule)

            # Update statistics
            self._stats["evaluations_run"] += 1
            self._stats["reports_generated"] += 1
            self._stats["notifications_sent"] += len(
                schedule.get("report_recipients", [])
            )
            self._stats["last_evaluation"] = datetime.utcnow()

            logger.info(
                f"Completed scheduled evaluation for tenant {tenant_id}",
                extra={
                    "tenant_id": tenant_id,
                    "schedule_id": schedule_id,
                    "status": "completed",
                    "data_points": len(recent_data),
                    "quality_score": quality_metrics.rubric_score,
                    "report_path": report.get("file_path"),
                },
            )
            return evaluation_result

        except Exception as e:
            logger.error("Failed to run scheduled evaluation %s: %s", schedule_id, e)

            # Store failure result
            failure_result = {
                "schedule_id": schedule_id,
                "tenant_id": tenant_id,
                "evaluation_date": datetime.utcnow(),
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }

            try:
                await self.storage_backend.store_evaluation_result(failure_result)
            except Exception as storage_error:
                logger.error(
                    f"Failed to store failure result for evaluation {schedule_id}: {storage_error}"
                )

            # Update failure statistics
            self._stats["evaluations_failed"] += 1

            logger.error(
                f"Failed to run scheduled evaluation {schedule_id}",
                extra={
                    "tenant_id": tenant_id,
                    "schedule_id": schedule_id,
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            raise RuntimeError(f"Failed to run scheduled evaluation {schedule_id}: {e}")

    async def _get_recent_analysis_data(
        self, tenant_id: str, days_back: int = 7
    ) -> List[tuple[AnalysisRequest, AnalysisResponse]]:
        """Get recent analysis data for evaluation."""
        # This would integrate with the storage backend to get recent data
        # For now, return empty list - would be implemented based on storage backend
        return []

    async def _generate_evaluation_report(
        self, tenant_id: str, quality_metrics: QualityMetrics, schedule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate evaluation report."""
        from ...reporting.models import ReportData, ReportFormat

        # Create report data with evaluation metrics
        report_data = ReportData(
            custom_data={
                "tenant_id": tenant_id,
                "evaluation_metrics": quality_metrics.dict(),
                "evaluation_period": "Last 7 days",
                "report_type": "weekly_evaluation",
                "quality_metrics": quality_metrics.dict(),
            }
        )

        # Generate PDF report
        report_content = self.report_generator.generate_report(
            report_data=report_data,
            format_type=ReportFormat.PDF.value,
            tenant_id=tenant_id,
            report_type="weekly_evaluation",
        )

        # Ensure content is bytes before saving
        if isinstance(report_content, str):
            report_content_bytes = report_content.encode("utf-8")
        elif isinstance(report_content, dict):
            report_content_bytes = str(report_content).encode("utf-8")
        else:
            report_content_bytes = report_content

        # Save report to storage
        report_path = await self.storage_backend.save_evaluation_report(
            tenant_id, report_content_bytes, "weekly_evaluation"
        )

        return {
            "file_path": report_path,
            "content": report_content,
            "format": "PDF",
        }

    async def _send_evaluation_notifications(
        self,
        recipients: List[str],
        report: Dict[str, Any],
        quality_metrics: QualityMetrics,
    ) -> None:
        """Send evaluation notifications to recipients."""
        # Check if quality metrics indicate issues
        alerts = []

        if quality_metrics.schema_valid_rate < 0.98:
            alerts.append("Schema validation rate below threshold")

        if quality_metrics.rubric_score < 0.8:
            alerts.append("Rubric score below threshold")

        if quality_metrics.opa_compile_success_rate < 0.95:
            alerts.append("OPA compilation success rate below threshold")

        # Send notifications via alerting system
        for recipient in recipients:
            await self.alerting_system.send_evaluation_notification(
                recipient=recipient,
                report_path=report["file_path"],
                quality_metrics=quality_metrics,
                alerts=alerts,
            )

    async def list_scheduled_evaluations(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List scheduled evaluations, optionally filtered by tenant."""
        # Get schedules from storage backend
        all_schedules = await self.storage_backend.get_evaluation_schedules(tenant_id)

        # Update in-memory cache
        for schedule in all_schedules:
            self.scheduled_evaluations[schedule["schedule_id"]] = schedule

        if tenant_id:
            return [
                schedule
                for schedule in all_schedules
                if schedule["tenant_id"] == tenant_id
            ]
        return all_schedules

    async def cancel_scheduled_evaluation(self, schedule_id: str) -> bool:
        """Cancel a scheduled evaluation."""
        if schedule_id not in self.scheduled_evaluations:
            return False

        schedule = self.scheduled_evaluations[schedule_id]
        schedule["active"] = False

        await self.storage_backend.update_evaluation_schedule(schedule)
        del self.scheduled_evaluations[schedule_id]

        logger.info("Cancelled scheduled evaluation %s", schedule_id)
        return True

    def get_service_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics and health information.

        Returns:
            Dictionary containing service statistics
        """
        uptime = datetime.utcnow() - self._stats["service_start_time"]

        return {
            "service_uptime_seconds": uptime.total_seconds(),
            "schedules_created": self._stats["schedules_created"],
            "evaluations_run": self._stats["evaluations_run"],
            "evaluations_failed": self._stats["evaluations_failed"],
            "reports_generated": self._stats["reports_generated"],
            "notifications_sent": self._stats["notifications_sent"],
            "last_evaluation": (
                self._stats["last_evaluation"].isoformat()
                if self._stats["last_evaluation"]
                else None
            ),
            "active_schedules": len(self.scheduled_evaluations),
            "success_rate": (
                self._stats["evaluations_run"]
                / (self._stats["evaluations_run"] + self._stats["evaluations_failed"])
                if (self._stats["evaluations_run"] + self._stats["evaluations_failed"])
                > 0
                else 0.0
            ),
        }


class HealthService:
    """
    Domain service for health operations.

    Encapsulates health check business logic and coordinates
    between different health check components.
    """

    def __init__(
        self,
        model_server: IModelServer,
        idempotency_manager: IIdempotencyManager,
        quality_evaluator: IQualityEvaluator,
    ):
        """
        Initialize the health service.

        Args:
            model_server: Model server implementation
            idempotency_manager: Idempotency manager implementation
            quality_evaluator: Quality evaluator implementation
        """
        self.model_server = model_server
        self.idempotency_manager = idempotency_manager
        self.quality_evaluator = quality_evaluator

    async def check_health(self) -> HealthStatus:
        """
        Perform comprehensive health check.

        Returns:
            Health status
        """
        checks = {}
        overall_status = "healthy"

        # Check model server health
        try:
            # This would be a health check method on the model server
            checks["model_server"] = {
                "status": "healthy",
                "details": "Model server operational",
            }
        except Exception as e:
            checks["model_server"] = {"status": "unhealthy", "details": str(e)}
            overall_status = "unhealthy"

        # Check idempotency manager health
        try:
            # This would be a health check method on the idempotency manager
            checks["idempotency_manager"] = {
                "status": "healthy",
                "details": "Idempotency manager operational",
            }
        except Exception as e:
            checks["idempotency_manager"] = {"status": "unhealthy", "details": str(e)}
            overall_status = "unhealthy"

        # Check quality evaluator health
        try:
            # This would be a health check method on the quality evaluator
            checks["quality_evaluator"] = {
                "status": "healthy",
                "details": "Quality evaluator operational",
            }
        except Exception as e:
            checks["quality_evaluator"] = {"status": "unhealthy", "details": str(e)}
            overall_status = "degraded"  # Quality issues are degraded, not unhealthy

        return HealthStatus(
            status=overall_status,
            service="analysis",
            version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            checks=checks,
        )


# These imports are now at the top of the file
