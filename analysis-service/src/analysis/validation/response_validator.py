"""
Response validation for Analysis Service.

Validates complete analysis responses before returning to clients.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from ..shared_integration import get_shared_logger

    logger = get_shared_logger(__name__)
except ImportError:
    # Fallback to standard logging if shared integration is not available
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""

    is_valid: bool
    validated_response: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any]


class ResponseValidator:
    """Validates complete analysis responses."""

    def __init__(self):
        self.logger = logger.bind(component="response_validator")

        # Define expected response structure for different analysis types
        self.response_schemas = {
            "risk_assessment": {
                "required": ["risk_score", "risk_factors", "recommendations"],
                "optional": ["mitigation_steps", "controls", "timeline"],
            },
            "pattern_analysis": {
                "required": ["patterns", "confidence", "analysis_summary"],
                "optional": ["trends", "anomalies", "predictions"],
            },
            "compliance_mapping": {
                "required": ["framework", "mappings", "compliance_score"],
                "optional": ["gaps", "recommendations", "controls"],
            },
            "rag_analysis": {
                "required": ["insights", "relevant_regulations", "confidence"],
                "optional": ["remediation_steps", "risk_context"],
            },
        }

    def validate_analysis_response(
        self,
        response: Dict[str, Any],
        analysis_type: str,
        request_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a complete analysis response.

        Args:
            response: Analysis response to validate
            analysis_type: Type of analysis performed
            request_id: Optional request ID for tracking

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_response = response.copy()
        performance_metrics = {}

        try:
            # Validate basic response structure
            basic_validation = self._validate_basic_structure(response)
            errors.extend(basic_validation["errors"])
            warnings.extend(basic_validation["warnings"])

            # Validate analysis-type specific structure
            if analysis_type in self.response_schemas:
                type_validation = self._validate_analysis_type_structure(
                    response, analysis_type
                )
                errors.extend(type_validation["errors"])
                warnings.extend(type_validation["warnings"])
            else:
                warnings.append(f"Unknown analysis type: {analysis_type}")

            # Validate performance metrics
            perf_validation = self._validate_performance_metrics(response)
            errors.extend(perf_validation["errors"])
            warnings.extend(perf_validation["warnings"])
            performance_metrics = perf_validation["metrics"]

            # Validate data quality
            quality_validation = self._validate_data_quality(response)
            warnings.extend(quality_validation)

            # Validate business rules
            business_validation = self._validate_business_rules(response, analysis_type)
            warnings.extend(business_validation)

            return ValidationResult(
                is_valid=len(errors) == 0,
                validated_response=validated_response,
                errors=errors,
                warnings=warnings,
                performance_metrics=performance_metrics,
                metadata={
                    "analysis_type": analysis_type,
                    "request_id": request_id,
                    "validation_timestamp": self._get_timestamp(),
                },
            )

        except Exception as e:
            self.logger.error("Response validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                validated_response=response,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                performance_metrics={},
                metadata={},
            )

    def _validate_basic_structure(
        self, response: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate basic response structure."""
        errors = []
        warnings = []

        # Check for required top-level fields
        required_fields = ["status", "results"]
        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        # Validate status
        status = response.get("status")
        if status:
            valid_statuses = ["completed", "partial", "failed", "timeout"]
            if status not in valid_statuses:
                errors.append(f"Invalid status: {status}")

        # Validate results structure
        results = response.get("results")
        if results is not None:
            if not isinstance(results, (dict, list)):
                errors.append("results must be a dictionary or list")
            elif isinstance(results, list) and len(results) == 0:
                warnings.append("results list is empty")
            elif isinstance(results, dict) and len(results) == 0:
                warnings.append("results dictionary is empty")

        # Check for error information if status indicates failure
        if status in ["failed", "timeout"] and "error" not in response:
            warnings.append("Failed status but no error information provided")

        return {"errors": errors, "warnings": warnings}

    def _validate_analysis_type_structure(
        self, response: Dict[str, Any], analysis_type: str
    ) -> Dict[str, List[str]]:
        """Validate analysis-type specific structure."""
        errors = []
        warnings = []

        schema = self.response_schemas.get(analysis_type, {})
        required_fields = schema.get("required", [])
        optional_fields = schema.get("optional", [])

        results = response.get("results", {})
        if not isinstance(results, dict):
            return {
                "errors": ["results must be a dictionary for structure validation"],
                "warnings": [],
            }

        # Check required fields
        for field in required_fields:
            if field not in results:
                errors.append(f"Missing required field in results: {field}")

        # Check for unexpected fields
        all_expected_fields = set(required_fields + optional_fields)
        actual_fields = set(results.keys())
        unexpected_fields = actual_fields - all_expected_fields

        if unexpected_fields:
            warnings.append(f"Unexpected fields in results: {list(unexpected_fields)}")

        return {"errors": errors, "warnings": warnings}

    def _validate_performance_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance metrics in response."""
        errors = []
        warnings = []
        metrics = {}

        # Extract performance metrics
        processing_time = response.get("processing_time_ms")
        if processing_time is not None:
            if not isinstance(processing_time, (int, float)):
                errors.append("processing_time_ms must be a number")
            elif processing_time < 0:
                errors.append("processing_time_ms cannot be negative")
            elif processing_time > 300000:  # 5 minutes
                warnings.append("Very long processing time detected")
            else:
                metrics["processing_time_ms"] = processing_time

        # Check for confidence scores
        results = response.get("results", {})
        if isinstance(results, dict):
            confidence = results.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    errors.append("confidence must be a number")
                elif not (0 <= confidence <= 1):
                    errors.append("confidence must be between 0 and 1")
                elif confidence < 0.5:
                    warnings.append("Low confidence score in results")
                else:
                    metrics["confidence"] = confidence

        return {"errors": errors, "warnings": warnings, "metrics": metrics}

    def _validate_data_quality(self, response: Dict[str, Any]) -> List[str]:
        """Validate data quality aspects of the response."""
        warnings = []

        results = response.get("results", {})
        if not isinstance(results, dict):
            return warnings

        # Check for empty or null values in critical fields
        critical_fields = ["risk_score", "compliance_score", "confidence"]
        for field in critical_fields:
            value = results.get(field)
            if value is None:
                continue
            elif isinstance(value, str) and not value.strip():
                warnings.append(f"Empty string value for {field}")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                warnings.append(f"Empty collection for {field}")

        # Check for reasonable value ranges
        risk_score = results.get("risk_score")
        if isinstance(risk_score, dict):
            overall_risk = risk_score.get("overall_risk_score")
            if isinstance(overall_risk, (int, float)):
                if overall_risk == 0:
                    warnings.append(
                        "Zero overall risk score may indicate incomplete analysis"
                    )
                elif overall_risk == 1:
                    warnings.append("Maximum risk score detected - verify accuracy")

        return warnings

    def _validate_business_rules(
        self, response: Dict[str, Any], analysis_type: str
    ) -> List[str]:
        """Validate business logic rules."""
        warnings = []

        results = response.get("results", {})
        if not isinstance(results, dict):
            return warnings

        # Risk assessment specific rules
        if analysis_type == "risk_assessment":
            risk_score = results.get("risk_score", {})
            recommendations = results.get("recommendations", [])

            # High risk should have recommendations
            if isinstance(risk_score, dict):
                overall_risk = risk_score.get("overall_risk_score", 0)
                if overall_risk > 0.7 and len(recommendations) == 0:
                    warnings.append(
                        "High risk detected but no recommendations provided"
                    )

        # Compliance mapping specific rules
        elif analysis_type == "compliance_mapping":
            compliance_score = results.get("compliance_score", 1)
            gaps = results.get("gaps", [])

            # Low compliance should have identified gaps
            if compliance_score < 0.7 and len(gaps) == 0:
                warnings.append("Low compliance score but no gaps identified")

        # Pattern analysis specific rules
        elif analysis_type == "pattern_analysis":
            patterns = results.get("patterns", {})
            confidence = results.get("confidence", 0)

            # High confidence should have detected patterns
            if confidence > 0.8 and not patterns:
                warnings.append("High confidence but no patterns detected")

        return warnings

    def validate_batch_response(
        self, responses: List[Dict[str, Any]], analysis_types: List[str]
    ) -> ValidationResult:
        """
        Validate a batch of analysis responses.

        Args:
            responses: List of analysis responses
            analysis_types: Corresponding analysis types

        Returns:
            Aggregated validation result
        """
        all_errors = []
        all_warnings = []
        validated_responses = []
        all_metrics = {}

        try:
            if len(responses) != len(analysis_types):
                all_errors.append("Mismatch between responses and analysis types count")
                return ValidationResult(
                    is_valid=False,
                    validated_response={"responses": responses},
                    errors=all_errors,
                    warnings=all_warnings,
                    performance_metrics={},
                    metadata={},
                )

            # Validate each response
            for i, (response, analysis_type) in enumerate(
                zip(responses, analysis_types)
            ):
                validation = self.validate_analysis_response(
                    response, analysis_type, request_id=f"batch_{i}"
                )

                validated_responses.append(validation.validated_response)

                # Collect errors and warnings with context
                for error in validation.errors:
                    all_errors.append(f"Response {i}: {error}")

                for warning in validation.warnings:
                    all_warnings.append(f"Response {i}: {warning}")

                # Aggregate metrics
                for key, value in validation.performance_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

            # Calculate aggregate metrics
            aggregate_metrics = {}
            for key, values in all_metrics.items():
                if values:
                    aggregate_metrics[f"avg_{key}"] = sum(values) / len(values)
                    aggregate_metrics[f"max_{key}"] = max(values)
                    aggregate_metrics[f"min_{key}"] = min(values)

            return ValidationResult(
                is_valid=len(all_errors) == 0,
                validated_response={"responses": validated_responses},
                errors=all_errors,
                warnings=all_warnings,
                performance_metrics=aggregate_metrics,
                metadata={
                    "batch_size": len(responses),
                    "validation_timestamp": self._get_timestamp(),
                },
            )

        except Exception as e:
            self.logger.error("Batch validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                validated_response={"responses": responses},
                errors=[f"Batch validation error: {str(e)}"],
                warnings=[],
                performance_metrics={},
                metadata={},
            )

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.utcnow().isoformat()
