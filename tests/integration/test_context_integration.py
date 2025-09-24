"""
Integration tests for Enhanced Context Manager with real-world scenarios.

Tests the complete context system with realistic data and edge cases.
"""

import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.llama_mapper.context.enhanced_context_manager import (
    ApplicationContext,
    BusinessContext,
    ContextAwareCache,
    ContextAwarePromptBuilder,
    ContextManager,
    DetectorContext,
    EnhancedContext,
    HistoricalContext,
    PolicyContext,
    TenantContext,
)


class TestRealWorldScenarios:
    """Test real-world compliance scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()
        self.prompt_builder = ContextAwarePromptBuilder(self.context_manager)

    def test_healthcare_pii_detection_scenario(self):
        """Test healthcare PII detection with HIPAA compliance."""
        # Real healthcare scenario
        mapper_payload = {
            "detector": "pii-detector-v2",
            "output": "SSN detected: 123-45-6789 in patient record",
            "tenant_id": "mayo_clinic_tenant",
            "metadata": {
                "policy_bundle": "healthcare",
                "app_name": "epic-patient-portal",
                "route": "/api/patient/records",
                "environment": "prod",
                "user_role": "physician",
                "session_id": "sess_abc123",
                "request_source": "api",
                "detector_type": "pii",
                "confidence_threshold": 0.95,
                "coverage_achieved": 0.98,
                "contributing_detectors": ["pii-detector-v2", "ssn-detector"],
                "aggregation_method": "highest_confidence",
                "processing_time_ms": 120,
            },
        }

        tenant_context = {
            "industry": "healthcare",
            "compliance_requirements": ["HIPAA", "GDPR", "HITECH"],
            "risk_tolerance": "strict",
            "data_classification": "confidential",
            "jurisdiction": ["US", "EU"],
            "custom_taxonomy": {
                "PII.Health.PatientSSN": "Patient Social Security Number",
                "PII.Health.MedicalRecord": "Medical record identifier",
            },
            "custom_policies": {"hipaa_strict_mode": True, "require_audit_log": True},
            "compliance_contacts": ["compliance@mayoclinic.org"],
            "escalation_rules": {
                "high_risk_pii": "immediate_notification",
                "hipaa_violation": "legal_team_escalation",
            },
        }

        historical_data = {
            "similar_detections": [
                {
                    "type": "ssn",
                    "confidence": 0.98,
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "type": "ssn",
                    "confidence": 0.95,
                    "timestamp": "2024-01-14T14:20:00Z",
                },
            ],
            "false_positive_rate": 0.05,  # Very low for healthcare
            "detection_trends": {
                "ssn_detections": "increasing",
                "hipaa_compliance": "stable",
            },
            "previous_mappings": [
                {"taxonomy": ["PII.Health.PatientSSN"], "confidence": 0.98},
                {"taxonomy": ["PII.Health.MedicalRecord"], "confidence": 0.95},
            ],
            "confidence_history": [0.98, 0.95, 0.97, 0.96],
        }

        # Build enhanced context
        enhanced_context = self.context_manager.build_mapper_context(
            mapper_payload, tenant_context, historical_data
        )

        # Verify healthcare-specific context
        assert enhanced_context.business.industry == "healthcare"
        assert "HIPAA" in enhanced_context.business.compliance_requirements
        assert enhanced_context.business.risk_tolerance == "strict"
        assert enhanced_context.business.data_classification == "confidential"

        assert enhanced_context.application.app_name == "epic-patient-portal"
        assert enhanced_context.application.user_role == "physician"
        assert enhanced_context.application.environment == "prod"

        assert enhanced_context.policy.policy_bundle == "healthcare"
        assert "HIPAA" in enhanced_context.policy.applicable_frameworks
        assert enhanced_context.policy.enforcement_level == "strict"
        assert "quarterly_audit" in enhanced_context.policy.audit_requirements
        assert "breach_notification" in enhanced_context.policy.reporting_obligations

        assert enhanced_context.historical is not None
        assert enhanced_context.historical.false_positive_rate == 0.05
        assert len(enhanced_context.historical.similar_detections) == 2
        assert "ssn_detections" in enhanced_context.historical.detection_trends

        assert enhanced_context.tenant is not None
        assert enhanced_context.tenant.tenant_id == "mayo_clinic_tenant"
        assert "PII.Health.PatientSSN" in enhanced_context.tenant.custom_taxonomy
        assert enhanced_context.tenant.custom_policies["hipaa_strict_mode"] is True

        assert enhanced_context.detector is not None
        assert enhanced_context.detector.detector_id == "pii-detector-v2"
        assert enhanced_context.detector.confidence_threshold == 0.95
        assert enhanced_context.detector.coverage_achieved == 0.98

        # Build context-aware prompt
        prompt = self.prompt_builder.build_mapper_prompt(
            mapper_payload["output"], enhanced_context
        )

        # Verify prompt contains healthcare-specific instructions
        assert "healthcare" in prompt.lower()
        assert "HIPAA" in prompt
        assert "strict" in prompt.lower()
        assert "SSN detected: 123-45-6789 in patient record" in prompt
        assert "Apply strict compliance mapping with detailed taxonomy" in prompt
        assert "High false positive rate" not in prompt  # 5% is low
        assert "context_notes" in prompt

    def test_financial_aml_detection_scenario(self):
        """Test financial AML detection with SOX and PCI-DSS compliance."""
        mapper_payload = {
            "detector": "aml-detector-pro",
            "output": "Suspicious transaction pattern: $50,000 transfer to high-risk jurisdiction",
            "tenant_id": "chase_bank_tenant",
            "metadata": {
                "policy_bundle": "financial",
                "app_name": "chase-mobile-banking",
                "route": "/api/transactions/transfer",
                "environment": "prod",
                "user_role": "customer",
                "session_id": "sess_def456",
                "request_source": "mobile_app",
                "detector_type": "aml",
                "confidence_threshold": 0.85,
                "coverage_achieved": 0.92,
                "contributing_detectors": ["aml-detector-pro", "risk-scorer"],
                "aggregation_method": "weighted_average",
                "processing_time_ms": 200,
            },
        }

        tenant_context = {
            "industry": "financial_services",
            "compliance_requirements": ["SOX", "PCI-DSS", "GDPR", "AML"],
            "risk_tolerance": "strict",
            "data_classification": "restricted",
            "jurisdiction": ["US", "EU", "UK"],
            "custom_taxonomy": {
                "FINANCIAL.AML.SuspiciousTransaction": "Suspicious transaction pattern",
                "FINANCIAL.RISK.HighRiskJurisdiction": "High-risk jurisdiction transfer",
            },
            "custom_policies": {
                "aml_strict_mode": True,
                "require_kyc_verification": True,
                "auto_freeze_threshold": 10000,
            },
            "compliance_contacts": ["aml@chase.com", "compliance@chase.com"],
            "escalation_rules": {
                "suspicious_transaction": "immediate_review",
                "high_risk_jurisdiction": "aml_team_escalation",
                "large_amount": "senior_analyst_review",
            },
        }

        historical_data = {
            "similar_detections": [
                {"type": "suspicious_transaction", "confidence": 0.88, "amount": 25000},
                {
                    "type": "high_risk_jurisdiction",
                    "confidence": 0.92,
                    "jurisdiction": "high_risk",
                },
            ],
            "false_positive_rate": 0.12,  # Moderate for AML
            "detection_trends": {
                "aml_detections": "increasing",
                "high_risk_jurisdictions": "stable",
                "transaction_volumes": "increasing",
            },
            "previous_mappings": [
                {
                    "taxonomy": ["FINANCIAL.AML.SuspiciousTransaction"],
                    "confidence": 0.88,
                },
                {
                    "taxonomy": ["FINANCIAL.RISK.HighRiskJurisdiction"],
                    "confidence": 0.92,
                },
            ],
            "confidence_history": [0.88, 0.92, 0.85, 0.90],
        }

        # Build enhanced context
        enhanced_context = self.context_manager.build_mapper_context(
            mapper_payload, tenant_context, historical_data
        )

        # Verify financial-specific context
        assert enhanced_context.business.industry == "financial_services"
        assert "SOX" in enhanced_context.business.compliance_requirements
        assert "PCI-DSS" in enhanced_context.business.compliance_requirements
        assert enhanced_context.business.risk_tolerance == "strict"
        assert enhanced_context.business.data_classification == "restricted"

        assert enhanced_context.policy.policy_bundle == "financial"
        assert "SOX" in enhanced_context.policy.applicable_frameworks
        assert "PCI-DSS" in enhanced_context.policy.applicable_frameworks
        assert enhanced_context.policy.enforcement_level == "strict"
        assert "annual_audit" in enhanced_context.policy.audit_requirements
        assert "regulatory_reporting" in enhanced_context.policy.reporting_obligations

        assert enhanced_context.historical is not None
        assert enhanced_context.historical.false_positive_rate == 0.12
        assert "aml_detections" in enhanced_context.historical.detection_trends

        assert enhanced_context.tenant is not None
        assert (
            "FINANCIAL.AML.SuspiciousTransaction"
            in enhanced_context.tenant.custom_taxonomy
        )
        assert enhanced_context.tenant.custom_policies["aml_strict_mode"] is True

        assert enhanced_context.detector is not None
        assert enhanced_context.detector.detector_id == "aml-detector-pro"
        assert enhanced_context.detector.confidence_threshold == 0.85

        # Build context-aware prompt
        prompt = self.prompt_builder.build_mapper_prompt(
            mapper_payload["output"], enhanced_context
        )

        # Verify prompt contains financial-specific instructions
        assert "financial_services" in prompt.lower()
        assert "SOX" in prompt
        assert "PCI-DSS" in prompt
        assert "strict" in prompt.lower()
        assert "Suspicious transaction pattern" in prompt
        assert "Apply strict compliance mapping with detailed taxonomy" in prompt
        assert "context_notes" in prompt

    def test_tech_company_gdpr_scenario(self):
        """Test technology company GDPR compliance scenario."""
        analysis_request = {
            "tenant": "google_cloud_tenant",
            "app": "google-analytics",
            "route": "/api/user-data",
            "env": "prod",
            "policy_bundle": "default",
            "required_detectors": ["pii-detector", "consent-tracker"],
            "observed_coverage": {"pii-detector": 0.95, "consent-tracker": 0.88},
            "required_coverage": {"pii-detector": 0.98, "consent-tracker": 0.95},
            "detector_errors": {"consent-tracker": {"error": "timeout", "count": 5}},
            "high_sev_hits": [
                {"type": "pii", "severity": "high", "data": "email addresses"},
                {"type": "consent", "severity": "medium", "data": "missing consent"},
            ],
            "false_positive_bands": [
                {"type": "false_positive", "detector": "pii-detector", "count": 12}
            ],
        }

        tenant_context = {
            "industry": "technology",
            "compliance_requirements": ["GDPR", "CCPA", "LGPD"],
            "risk_tolerance": "medium",
            "data_classification": "internal",
            "jurisdiction": ["EU", "US", "BR"],
        }

        # Build enhanced context
        enhanced_context = self.context_manager.build_analyst_context(
            analysis_request, tenant_context
        )

        # Verify tech-specific context
        assert enhanced_context.business.industry == "technology"
        assert "GDPR" in enhanced_context.business.compliance_requirements
        assert "CCPA" in enhanced_context.business.compliance_requirements
        assert enhanced_context.business.risk_tolerance == "medium"
        assert enhanced_context.business.data_classification == "internal"

        assert enhanced_context.application.app_name == "google-analytics"
        assert enhanced_context.application.environment == "prod"

        assert enhanced_context.policy.policy_bundle == "default"
        assert "GDPR" in enhanced_context.policy.applicable_frameworks
        assert "CCPA" in enhanced_context.policy.applicable_frameworks
        assert enhanced_context.policy.enforcement_level == "moderate"

        assert enhanced_context.historical is not None
        assert len(enhanced_context.historical.similar_detections) == 2
        assert "coverage_gaps" in enhanced_context.historical.detection_trends
        assert "detector_errors" in enhanced_context.historical.detection_trends

        # Build context-aware prompt
        prompt = self.prompt_builder.build_analyst_prompt(
            "GDPR compliance analysis for user data processing", enhanced_context
        )

        # Verify prompt contains tech-specific instructions
        assert "technology" in prompt.lower()
        assert "GDPR" in prompt
        assert "CCPA" in prompt
        assert "medium" in prompt.lower()
        assert "GDPR compliance analysis for user data processing" in prompt
        assert "analysis_type" in prompt
        assert "risk_level" in prompt
        assert "recommendations" in prompt


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()
        self.prompt_builder = ContextAwarePromptBuilder(self.context_manager)

    def test_empty_metadata(self):
        """Test handling of empty metadata."""
        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {},
        }

        enhanced_context = self.context_manager.build_mapper_context(mapper_payload)

        # Should use defaults
        assert enhanced_context.application.app_name == "unknown"
        assert enhanced_context.application.route == "unknown"
        assert enhanced_context.application.environment == "prod"
        assert enhanced_context.detector.detector_type == "unknown"
        assert enhanced_context.detector.confidence_threshold == 0.7

    def test_missing_tenant_context(self):
        """Test handling of missing tenant context."""
        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {"policy_bundle": "healthcare"},
        }

        enhanced_context = self.context_manager.build_mapper_context(mapper_payload)

        # Should use defaults
        assert enhanced_context.business.industry == "technology"
        assert enhanced_context.business.compliance_requirements == ["GDPR", "CCPA"]
        assert enhanced_context.business.risk_tolerance == "medium"
        assert enhanced_context.tenant is None

    def test_unknown_policy_bundle(self):
        """Test handling of unknown policy bundle."""
        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {"policy_bundle": "unknown_bundle"},
        }

        enhanced_context = self.context_manager.build_mapper_context(mapper_payload)

        # Should fall back to default
        assert enhanced_context.policy.policy_bundle == "unknown_bundle"
        assert enhanced_context.policy.applicable_frameworks == ["GDPR", "CCPA"]
        assert enhanced_context.policy.enforcement_level == "moderate"

    def test_high_false_positive_rate(self):
        """Test handling of high false positive rate."""
        historical_data = {
            "similar_detections": [{"type": "test"}],
            "false_positive_rate": 0.35,  # High false positive rate
            "detection_trends": {"test": "increasing"},
            "previous_mappings": [],
            "confidence_history": [],
        }

        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {},
        }

        enhanced_context = self.context_manager.build_mapper_context(
            mapper_payload, None, historical_data
        )

        # Build prompt and verify high false positive rate handling
        prompt = self.prompt_builder.build_mapper_prompt(
            mapper_payload["output"], enhanced_context
        )

        assert "High false positive rate - apply conservative mapping" in prompt
        assert "35.0%" in prompt

    def test_low_false_positive_rate(self):
        """Test handling of low false positive rate."""
        historical_data = {
            "similar_detections": [{"type": "test"}],
            "false_positive_rate": 0.02,  # Very low false positive rate
            "detection_trends": {"test": "stable"},
            "previous_mappings": [],
            "confidence_history": [],
        }

        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {},
        }

        enhanced_context = self.context_manager.build_mapper_context(
            mapper_payload, None, historical_data
        )

        # Build prompt and verify low false positive rate handling
        prompt = self.prompt_builder.build_mapper_prompt(
            mapper_payload["output"], enhanced_context
        )

        assert "High false positive rate" not in prompt
        assert "2.0%" in prompt

    def test_malformed_historical_data(self):
        """Test handling of malformed historical data."""
        # Missing required fields
        historical_data = {
            "similar_detections": [{"type": "test"}]
            # Missing other required fields
        }

        mapper_payload = {
            "detector": "test-detector",
            "output": "test output",
            "tenant_id": "test_tenant",
            "metadata": {},
        }

        enhanced_context = self.context_manager.build_mapper_context(
            mapper_payload, None, historical_data
        )

        # Should handle gracefully with defaults
        assert enhanced_context.historical is not None
        assert enhanced_context.historical.false_positive_rate == 0.1  # Default
        assert enhanced_context.historical.detection_trends == {}  # Default
        assert enhanced_context.historical.previous_mappings == []  # Default
        assert enhanced_context.historical.confidence_history == []  # Default


class TestContextAwareCacheIntegration:
    """Test context-aware caching integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_cache = Mock()
        self.context_cache = ContextAwareCache(self.mock_base_cache)
        self.context_manager = ContextManager()

    def test_cache_key_differentiation(self):
        """Test that different contexts generate different cache keys."""
        # Healthcare context
        healthcare_context = self.context_manager.build_mapper_context(
            {
                "detector": "pii-detector",
                "output": "email detected: test@example.com",
                "tenant_id": "healthcare_tenant",
                "metadata": {"policy_bundle": "healthcare"},
            }
        )

        # Financial context
        financial_context = self.context_manager.build_mapper_context(
            {
                "detector": "pii-detector",
                "output": "email detected: test@example.com",
                "tenant_id": "financial_tenant",
                "metadata": {"policy_bundle": "financial"},
            }
        )

        # Same detector output, different contexts
        healthcare_key = self.context_cache.get_cache_key(
            "email detected: test@example.com", healthcare_context, "medium"
        )
        financial_key = self.context_cache.get_cache_key(
            "email detected: test@example.com", financial_context, "medium"
        )

        # Should be different due to different industry and policy bundle
        assert healthcare_key != financial_key
        assert "healthcare" in healthcare_key
        assert "financial" in financial_key

    def test_cache_sensitivity_levels(self):
        """Test different cache sensitivity levels."""
        context = self.context_manager.build_mapper_context(
            {
                "detector": "pii-detector",
                "output": "email detected: test@example.com",
                "tenant_id": "test_tenant",
                "metadata": {
                    "policy_bundle": "healthcare",
                    "app_name": "patient-portal",
                    "environment": "prod",
                },
            }
        )

        # Different sensitivity levels
        high_key = self.context_cache.get_cache_key(
            "email detected: test@example.com", context, "high"
        )
        medium_key = self.context_cache.get_cache_key(
            "email detected: test@example.com", context, "medium"
        )
        low_key = self.context_cache.get_cache_key(
            "email detected: test@example.com", context, "low"
        )

        # High sensitivity should include more context
        assert len(high_key) > len(medium_key)
        assert len(medium_key) > len(low_key)

        # All should include the base detector output
        assert "email detected: test@example.com" in high_key
        assert "email detected: test@example.com" in medium_key
        assert "email detected: test@example.com" in low_key

        # High should include environment, medium and low should not
        assert "prod" in high_key
        assert "prod" not in medium_key
        assert "prod" not in low_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
