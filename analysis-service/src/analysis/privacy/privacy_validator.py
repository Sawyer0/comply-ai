"""
Privacy validator for analysis service.

This module provides privacy-specific validation to ensure compliance
with privacy regulations and data protection requirements.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrivacyValidationResult:
    """Result of privacy validation."""

    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None


class PrivacyValidator:
    """
    Privacy validator for ensuring data protection compliance.

    Focuses specifically on privacy validation concerns following SRP.
    """

    def __init__(self):
        # PII patterns for detection
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        # Sensitive data keywords
        self.sensitive_keywords = {
            "password",
            "secret",
            "token",
            "key",
            "credential",
            "private",
            "confidential",
            "personal",
            "sensitive",
        }

    def validate_data_privacy(self, data: Any) -> PrivacyValidationResult:
        """
        Validate data for privacy compliance.

        Args:
            data: Data to validate for privacy compliance

        Returns:
            Privacy validation result
        """
        violations = []
        warnings = []

        try:
            # Convert data to string for pattern matching
            data_str = str(data)

            # Check for PII patterns
            pii_violations = self._detect_pii_patterns(data_str)
            violations.extend(pii_violations)

            # Check for sensitive keywords
            keyword_warnings = self._detect_sensitive_keywords(data_str)
            warnings.extend(keyword_warnings)

            # Check data structure for privacy concerns
            if isinstance(data, dict):
                struct_violations = self._validate_data_structure(data)
                violations.extend(struct_violations)

            is_compliant = len(violations) == 0

            return PrivacyValidationResult(
                is_compliant=is_compliant,
                violations=violations,
                warnings=warnings,
                sanitized_data=self._sanitize_data(data) if not is_compliant else None,
            )

        except Exception as e:
            logger.error("Privacy validation failed: %s", e)
            return PrivacyValidationResult(
                is_compliant=False,
                violations=[f"Privacy validation error: {e}"],
                warnings=[],
            )

    def _detect_pii_patterns(self, text: str) -> List[str]:
        """Detect PII patterns in text."""
        violations = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                violations.append(
                    f"Detected {pii_type} pattern: {len(matches)} instances"
                )

        return violations

    def _detect_sensitive_keywords(self, text: str) -> List[str]:
        """Detect sensitive keywords in text."""
        warnings = []
        text_lower = text.lower()

        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                warnings.append(f"Sensitive keyword detected: {keyword}")

        return warnings

    def _validate_data_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate data structure for privacy concerns."""
        violations = []

        # Check for common PII field names
        pii_fields = {
            "email",
            "phone",
            "ssn",
            "social_security_number",
            "credit_card",
            "password",
            "personal_info",
        }

        for field_name in data.keys():
            if field_name.lower() in pii_fields:
                violations.append(f"PII field detected: {field_name}")

        return violations

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data by removing/masking PII."""
        if isinstance(data, str):
            sanitized = data
            # Mask PII patterns
            for pii_type, pattern in self.pii_patterns.items():
                sanitized = re.sub(
                    pattern,
                    f"[MASKED_{pii_type.upper()}]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
            return sanitized

        elif isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key.lower() in {"email", "phone", "ssn", "password"}:
                    sanitized[key] = "[MASKED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized

        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]

        else:
            return data

    def validate_retention_compliance(
        self, data_age_days: int, data_type: str
    ) -> PrivacyValidationResult:
        """
        Validate data retention compliance.

        Args:
            data_age_days: Age of data in days
            data_type: Type of data for retention policy

        Returns:
            Retention compliance validation result
        """
        violations = []
        warnings = []

        # Define retention policies (in days)
        retention_policies = {
            "analysis_results": 365,  # 1 year
            "training_data": 730,  # 2 years
            "audit_logs": 2555,  # 7 years
            "personal_data": 90,  # 3 months
            "default": 365,
        }

        max_retention = retention_policies.get(data_type, retention_policies["default"])

        if data_age_days > max_retention:
            violations.append(
                f"Data retention violation: {data_type} data is {data_age_days} days old, max allowed is {max_retention} days"
            )
        elif data_age_days > (max_retention * 0.8):  # 80% of max retention
            warnings.append(
                f"Data approaching retention limit: {data_age_days}/{max_retention} days"
            )

        return PrivacyValidationResult(
            is_compliant=len(violations) == 0, violations=violations, warnings=warnings
        )
