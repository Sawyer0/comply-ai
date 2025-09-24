"""
Comprehensive tests for Enhanced Context Manager.

Tests all context types, builders, caching, and edge cases.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from src.llama_mapper.context.enhanced_context_manager import (
    ApplicationContext,
    BusinessContext,
    ContextAwareCache,
    ContextAwarePromptBuilder,
    ContextManager,
    ContextType,
    DetectorContext,
    EnhancedContext,
    HistoricalContext,
    PolicyContext,
    TenantContext,
)


class TestBusinessContext:
    """Test BusinessContext dataclass."""

    def test_business_context_creation(self):
        """Test basic business context creation."""
        context = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA", "GDPR"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US", "EU"],
        )

        assert context.industry == "healthcare"
        assert context.compliance_requirements == ["HIPAA", "GDPR"]
        assert context.risk_tolerance == "strict"
        assert context.data_classification == "confidential"
        assert context.jurisdiction == ["US", "EU"]

    def test_business_context_validation(self):
        """Test business context with different industry types."""
        industries = ["healthcare", "financial_services", "technology", "retail"]

        for industry in industries:
            context = BusinessContext(
                industry=industry,
                compliance_requirements=["GDPR"],
                risk_tolerance="medium",
                data_classification="internal",
                jurisdiction=["US"],
            )
            assert context.industry == industry


class TestApplicationContext:
    """Test ApplicationContext dataclass."""

    def test_application_context_creation(self):
        """Test basic application context creation."""
        context = ApplicationContext(
            app_name="patient-portal",
            route="/api/patient-data",
            environment="prod",
            user_role="admin",
            session_id="session_123",
            request_source="api",
        )

        assert context.app_name == "patient-portal"
        assert context.route == "/api/patient-data"
        assert context.environment == "prod"
        assert context.user_role == "admin"
        assert context.session_id == "session_123"
        assert context.request_source == "api"

    def test_application_context_optional_fields(self):
        """Test application context with optional fields."""
        context = ApplicationContext(
            app_name="test-app", route="/test", environment="dev"
        )

        assert context.app_name == "test-app"
        assert context.route == "/test"
        assert context.environment == "dev"
        assert context.user_role is None
        assert context.session_id is None
        assert context.request_source is None


class TestPolicyContext:
    """Test PolicyContext dataclass."""

    def test_policy_context_creation(self):
        """Test basic policy context creation."""
        context = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA", "GDPR"],
            enforcement_level="strict",
            audit_requirements=["quarterly_audit"],
            reporting_obligations=["breach_notification"],
        )

        assert context.policy_bundle == "healthcare"
        assert context.applicable_frameworks == ["HIPAA", "GDPR"]
        assert context.enforcement_level == "strict"
        assert context.audit_requirements == ["quarterly_audit"]
        assert context.reporting_obligations == ["breach_notification"]


class TestHistoricalContext:
    """Test HistoricalContext dataclass."""

    def test_historical_context_creation(self):
        """Test basic historical context creation."""
        context = HistoricalContext(
            similar_detections=[{"type": "pii", "confidence": 0.9}],
            false_positive_rate=0.15,
            detection_trends={"pii": "increasing"},
            previous_mappings=[{"taxonomy": ["PII.Email"]}],
            confidence_history=[0.8, 0.9, 0.85],
        )

        assert len(context.similar_detections) == 1
        assert context.false_positive_rate == 0.15
        assert context.detection_trends == {"pii": "increasing"}
        assert len(context.previous_mappings) == 1
        assert context.confidence_history == [0.8, 0.9, 0.85]


class TestTenantContext:
    """Test TenantContext dataclass."""

    def test_tenant_context_creation(self):
        """Test basic tenant context creation."""
        context = TenantContext(
            tenant_id="tenant_123",
            custom_taxonomy={"PII.Health": "Health-specific PII"},
            custom_policies={"strict_mode": True},
            compliance_contacts=["compliance@company.com"],
            escalation_rules={"high_risk": "immediate"},
        )

        assert context.tenant_id == "tenant_123"
        assert context.custom_taxonomy == {"PII.Health": "Health-specific PII"}
        assert context.custom_policies == {"strict_mode": True}
        assert context.compliance_contacts == ["compliance@company.com"]
        assert context.escalation_rules == {"high_risk": "immediate"}

    def test_tenant_context_optional_fields(self):
        """Test tenant context with optional fields."""
        context = TenantContext(tenant_id="tenant_456")

        assert context.tenant_id == "tenant_456"
        assert context.custom_taxonomy is None
        assert context.custom_policies is None
        assert context.compliance_contacts is None
        assert context.escalation_rules is None


class TestDetectorContext:
    """Test DetectorContext dataclass."""

    def test_detector_context_creation(self):
        """Test basic detector context creation."""
        context = DetectorContext(
            detector_id="pii-detector",
            detector_type="pii",
            confidence_threshold=0.8,
            coverage_achieved=0.95,
            contributing_detectors=["pii-detector", "email-detector"],
            aggregation_method="highest_confidence",
            processing_time_ms=150,
        )

        assert context.detector_id == "pii-detector"
        assert context.detector_type == "pii"
        assert context.confidence_threshold == 0.8
        assert context.coverage_achieved == 0.95
        assert context.contributing_detectors == ["pii-detector", "email-detector"]
        assert context.aggregation_method == "highest_confidence"
        assert context.processing_time_ms == 150


class TestEnhancedContext:
    """Test EnhancedContext dataclass."""

    def test_enhanced_context_creation(self):
        """Test basic enhanced context creation."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="test-app", route="/test", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        assert context.business == business
        assert context.application == application
        assert context.policy == policy
        assert context.historical is None
        assert context.tenant is None
        assert context.detector is None
        assert context.timestamp is not None
        assert isinstance(context.timestamp, str)

    def test_enhanced_context_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated."""
        business = BusinessContext(
            industry="tech",
            compliance_requirements=["GDPR"],
            risk_tolerance="medium",
            data_classification="internal",
            jurisdiction=["EU"],
        )
        application = ApplicationContext(
            app_name="app", route="/route", environment="dev"
        )
        policy = PolicyContext(
            policy_bundle="default",
            applicable_frameworks=["GDPR"],
            enforcement_level="moderate",
            audit_requirements=["monthly"],
            reporting_obligations=["incident"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        # Should be a valid ISO timestamp
        datetime.fromisoformat(context.timestamp.replace("Z", "+00:00"))


class TestContextManager:
    """Test ContextManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()

    def test_context_manager_initialization(self):
        """Test context manager initialization."""
        assert self.context_manager.logger is not None
        assert self.context_manager.context_cache == {}
        assert self.context_manager.context_history == []

    def test_build_business_context_default(self):
        """Test building default business context."""
        context = self.context_manager._build_business_context("tenant_123", None)

        assert context.industry == "technology"
        assert context.compliance_requirements == ["GDPR", "CCPA"]
        assert context.risk_tolerance == "medium"
        assert context.data_classification == "internal"
        assert context.jurisdiction == ["US", "EU"]

    def test_build_business_context_custom(self):
        """Test building custom business context."""
        tenant_context = {
            "industry": "healthcare",
            "compliance_requirements": ["HIPAA", "GDPR"],
            "risk_tolerance": "strict",
            "data_classification": "confidential",
            "jurisdiction": ["US"],
        }

        context = self.context_manager._build_business_context(
            "tenant_123", tenant_context
        )

        assert context.industry == "healthcare"
        assert context.compliance_requirements == ["HIPAA", "GDPR"]
        assert context.risk_tolerance == "strict"
        assert context.data_classification == "confidential"
        assert context.jurisdiction == ["US"]

    def test_build_application_context(self):
        """Test building application context."""
        payload = {
            "metadata": {
                "app_name": "patient-portal",
                "route": "/api/patient-data",
                "environment": "prod",
                "user_role": "admin",
                "session_id": "session_123",
                "request_source": "api",
            }
        }

        context = self.context_manager._build_application_context(payload)

        assert context.app_name == "patient-portal"
        assert context.route == "/api/patient-data"
        assert context.environment == "prod"
        assert context.user_role == "admin"
        assert context.session_id == "session_123"
        assert context.request_source == "api"

    def test_build_application_context_defaults(self):
        """Test building application context with defaults."""
        payload = {"metadata": {}}

        context = self.context_manager._build_application_context(payload)

        assert context.app_name == "unknown"
        assert context.route == "unknown"
        assert context.environment == "prod"
        assert context.user_role is None
        assert context.session_id is None
        assert context.request_source is None

    def test_build_policy_context_default(self):
        """Test building default policy context."""
        context = self.context_manager._build_policy_context(None)

        assert context.policy_bundle == "default"
        assert context.applicable_frameworks == ["GDPR", "CCPA"]
        assert context.enforcement_level == "moderate"
        assert context.audit_requirements == ["monthly_review"]
        assert context.reporting_obligations == ["incident_reporting"]

    def test_build_policy_context_healthcare(self):
        """Test building healthcare policy context."""
        context = self.context_manager._build_policy_context("healthcare")

        assert context.policy_bundle == "healthcare"
        assert context.applicable_frameworks == ["HIPAA", "GDPR"]
        assert context.enforcement_level == "strict"
        assert "quarterly_audit" in context.audit_requirements
        assert "breach_notification" in context.reporting_obligations

    def test_build_policy_context_financial(self):
        """Test building financial policy context."""
        context = self.context_manager._build_policy_context("financial")

        assert context.policy_bundle == "financial"
        assert context.applicable_frameworks == ["SOX", "PCI-DSS", "GDPR"]
        assert context.enforcement_level == "strict"
        assert "annual_audit" in context.audit_requirements
        assert "regulatory_reporting" in context.reporting_obligations

    def test_build_policy_context_unknown(self):
        """Test building policy context for unknown bundle."""
        context = self.context_manager._build_policy_context("unknown_bundle")

        # Should fall back to default
        assert context.policy_bundle == "unknown_bundle"
        assert context.applicable_frameworks == ["GDPR", "CCPA"]
        assert context.enforcement_level == "moderate"

    def test_build_historical_context_none(self):
        """Test building historical context with no data."""
        context = self.context_manager._build_historical_context(
            "tenant_123", "detector", None
        )

        assert context is None

    def test_build_historical_context_with_data(self):
        """Test building historical context with data."""
        historical_data = {
            "similar_detections": [{"type": "pii", "confidence": 0.9}],
            "false_positive_rate": 0.15,
            "detection_trends": {"pii": "increasing"},
            "previous_mappings": [{"taxonomy": ["PII.Email"]}],
            "confidence_history": [0.8, 0.9, 0.85],
        }

        context = self.context_manager._build_historical_context(
            "tenant_123", "detector", historical_data
        )

        assert context is not None
        assert len(context.similar_detections) == 1
        assert context.false_positive_rate == 0.15
        assert context.detection_trends == {"pii": "increasing"}
        assert len(context.previous_mappings) == 1
        assert context.confidence_history == [0.8, 0.9, 0.85]

    def test_build_analysis_historical_context(self):
        """Test building analysis historical context."""
        analysis_request = {
            "high_sev_hits": [{"type": "pii", "severity": "high"}],
            "false_positive_bands": [{"type": "false_positive"}],
            "observed_coverage": {"pii": 0.8},
            "detector_errors": {"pii": "timeout"},
        }

        context = self.context_manager._build_analysis_historical_context(
            analysis_request
        )

        assert context is not None
        assert len(context.similar_detections) == 1
        assert context.false_positive_rate == 1.0  # 1 false positive / 1 high sev hit
        assert "coverage_gaps" in context.detection_trends
        assert "detector_errors" in context.detection_trends

    def test_build_analysis_historical_context_empty(self):
        """Test building analysis historical context with empty data."""
        analysis_request = {}

        context = self.context_manager._build_analysis_historical_context(
            analysis_request
        )

        assert context is None

    def test_build_tenant_context_none(self):
        """Test building tenant context with no data."""
        context = self.context_manager._build_tenant_context("tenant_123", None)

        assert context is None

    def test_build_tenant_context_with_data(self):
        """Test building tenant context with data."""
        tenant_context = {
            "custom_taxonomy": {"PII.Health": "Health PII"},
            "custom_policies": {"strict_mode": True},
            "compliance_contacts": ["compliance@company.com"],
            "escalation_rules": {"high_risk": "immediate"},
        }

        context = self.context_manager._build_tenant_context(
            "tenant_123", tenant_context
        )

        assert context is not None
        assert context.tenant_id == "tenant_123"
        assert context.custom_taxonomy == {"PII.Health": "Health PII"}
        assert context.custom_policies == {"strict_mode": True}
        assert context.compliance_contacts == ["compliance@company.com"]
        assert context.escalation_rules == {"high_risk": "immediate"}

    def test_build_detector_context(self):
        """Test building detector context."""
        metadata = {
            "detector_type": "pii",
            "confidence_threshold": 0.8,
            "coverage_achieved": 0.95,
            "contributing_detectors": ["pii-detector", "email-detector"],
            "aggregation_method": "highest_confidence",
            "processing_time_ms": 150,
        }

        context = self.context_manager._build_detector_context("pii-detector", metadata)

        assert context.detector_id == "pii-detector"
        assert context.detector_type == "pii"
        assert context.confidence_threshold == 0.8
        assert context.coverage_achieved == 0.95
        assert context.contributing_detectors == ["pii-detector", "email-detector"]
        assert context.aggregation_method == "highest_confidence"
        assert context.processing_time_ms == 150

    def test_build_detector_context_defaults(self):
        """Test building detector context with defaults."""
        context = self.context_manager._build_detector_context("detector", {})

        assert context.detector_id == "detector"
        assert context.detector_type == "unknown"
        assert context.confidence_threshold == 0.7
        assert context.coverage_achieved == 1.0
        assert context.contributing_detectors == ["detector"]
        assert context.aggregation_method == "highest_confidence"
        assert context.processing_time_ms == 0

    def test_build_mapper_context(self):
        """Test building complete mapper context."""
        mapper_payload = {
            "detector": "pii-detector",
            "output": "email detected",
            "tenant_id": "tenant_123",
            "metadata": {
                "policy_bundle": "healthcare",
                "app_name": "patient-portal",
                "route": "/api/patient-data",
                "environment": "prod",
            },
        }

        tenant_context = {
            "industry": "healthcare",
            "compliance_requirements": ["HIPAA", "GDPR"],
            "risk_tolerance": "strict",
        }

        historical_data = {
            "similar_detections": [{"type": "pii"}],
            "false_positive_rate": 0.1,
        }

        context = self.context_manager.build_mapper_context(
            mapper_payload, tenant_context, historical_data
        )

        assert isinstance(context, EnhancedContext)
        assert context.business.industry == "healthcare"
        assert context.application.app_name == "patient-portal"
        assert context.policy.policy_bundle == "healthcare"
        assert context.historical is not None
        assert context.tenant is not None
        assert context.detector is not None

    def test_build_analyst_context(self):
        """Test building complete analyst context."""
        analysis_request = {
            "tenant": "tenant_123",
            "app": "patient-portal",
            "route": "/api/patient-data",
            "env": "prod",
            "policy_bundle": "healthcare",
            "high_sev_hits": [{"type": "pii"}],
            "false_positive_bands": [{"type": "false_positive"}],
        }

        tenant_context = {
            "industry": "healthcare",
            "compliance_requirements": ["HIPAA", "GDPR"],
        }

        context = self.context_manager.build_analyst_context(
            analysis_request, tenant_context
        )

        assert isinstance(context, EnhancedContext)
        assert context.business.industry == "healthcare"
        assert context.application.app_name == "patient-portal"
        assert context.policy.policy_bundle == "healthcare"
        assert context.historical is not None


class TestContextAwarePromptBuilder:
    """Test ContextAwarePromptBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()
        self.prompt_builder = ContextAwarePromptBuilder(self.context_manager)

    def test_build_context_summary(self):
        """Test building context summary."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        summary = self.prompt_builder._build_context_summary(context)

        assert "Industry: healthcare" in summary
        assert "Environment: prod" in summary
        assert "Frameworks: HIPAA" in summary
        assert "Risk Tolerance: strict" in summary

    def test_build_context_summary_with_optional_fields(self):
        """Test building context summary with optional fields."""
        business = BusinessContext(
            industry="tech",
            compliance_requirements=["GDPR"],
            risk_tolerance="medium",
            data_classification="internal",
            jurisdiction=["EU"],
        )
        application = ApplicationContext(
            app_name="app", route="/route", environment="dev"
        )
        policy = PolicyContext(
            policy_bundle="default",
            applicable_frameworks=["GDPR"],
            enforcement_level="moderate",
            audit_requirements=["monthly"],
            reporting_obligations=["incident"],
        )
        tenant = TenantContext(tenant_id="tenant_123")
        detector = DetectorContext(
            detector_id="detector",
            detector_type="pii",
            confidence_threshold=0.8,
            coverage_achieved=0.9,
            contributing_detectors=["detector"],
            aggregation_method="highest_confidence",
            processing_time_ms=100,
        )

        context = EnhancedContext(
            business=business,
            application=application,
            policy=policy,
            tenant=tenant,
            detector=detector,
        )

        summary = self.prompt_builder._build_context_summary(context)

        assert "Tenant: tenant_123" in summary
        assert "Detector: detector" in summary

    def test_build_business_instructions(self):
        """Test building business instructions."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA", "GDPR"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )

        instructions = self.prompt_builder._build_business_instructions(business)

        assert "Industry: healthcare" in instructions
        assert "Compliance Requirements: HIPAA, GDPR" in instructions
        assert "Risk Tolerance: strict" in instructions
        assert "Data Classification: confidential" in instructions

    def test_build_business_instructions_risk_tolerance(self):
        """Test business instructions with different risk tolerances."""
        # Low risk tolerance
        business_low = BusinessContext(
            industry="tech",
            compliance_requirements=["GDPR"],
            risk_tolerance="low",
            data_classification="internal",
            jurisdiction=["EU"],
        )

        instructions_low = self.prompt_builder._build_business_instructions(
            business_low
        )
        assert (
            "Apply conservative mapping with higher confidence thresholds"
            in instructions_low
        )

        # High risk tolerance
        business_high = BusinessContext(
            industry="tech",
            compliance_requirements=["GDPR"],
            risk_tolerance="high",
            data_classification="internal",
            jurisdiction=["EU"],
        )

        instructions_high = self.prompt_builder._build_business_instructions(
            business_high
        )
        assert (
            "Apply aggressive mapping with lower confidence thresholds"
            in instructions_high
        )

    def test_build_policy_instructions(self):
        """Test building policy instructions."""
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA", "GDPR"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        instructions = self.prompt_builder._build_policy_instructions(policy)

        assert "Policy Bundle: healthcare" in instructions
        assert "Applicable Frameworks: HIPAA, GDPR" in instructions
        assert "Enforcement Level: strict" in instructions
        assert "Apply strict compliance mapping with detailed taxonomy" in instructions

    def test_build_policy_instructions_enforcement_levels(self):
        """Test policy instructions with different enforcement levels."""
        # Strict enforcement
        policy_strict = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        instructions_strict = self.prompt_builder._build_policy_instructions(
            policy_strict
        )
        assert (
            "Apply strict compliance mapping with detailed taxonomy"
            in instructions_strict
        )

        # Lenient enforcement
        policy_lenient = PolicyContext(
            policy_bundle="default",
            applicable_frameworks=["GDPR"],
            enforcement_level="lenient",
            audit_requirements=["monthly"],
            reporting_obligations=["incident"],
        )

        instructions_lenient = self.prompt_builder._build_policy_instructions(
            policy_lenient
        )
        assert (
            "Apply flexible compliance mapping with broader categories"
            in instructions_lenient
        )

    def test_build_historical_instructions_none(self):
        """Test building historical instructions with no historical context."""
        instructions = self.prompt_builder._build_historical_instructions(None)

        assert instructions == "No historical context available"

    def test_build_historical_instructions_with_data(self):
        """Test building historical instructions with data."""
        historical = HistoricalContext(
            similar_detections=[{"type": "pii"}],
            false_positive_rate=0.25,  # High false positive rate
            detection_trends={"pii": "increasing"},
            previous_mappings=[],
            confidence_history=[],
        )

        instructions = self.prompt_builder._build_historical_instructions(historical)

        assert "False Positive Rate: 25.0%" in instructions
        assert "Similar Detections: 1" in instructions
        assert "High false positive rate - apply conservative mapping" in instructions
        assert "Trends:" in instructions

    def test_build_mapper_prompt(self):
        """Test building complete mapper prompt."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        prompt = self.prompt_builder.build_mapper_prompt(
            "email detected: test@example.com", context
        )

        assert (
            "You are a compliance taxonomy mapper with rich context awareness" in prompt
        )
        assert "CONTEXT SUMMARY:" in prompt
        assert "BUSINESS CONTEXT:" in prompt
        assert "POLICY CONTEXT:" in prompt
        assert "HISTORICAL CONTEXT:" in prompt
        assert "DETECTOR OUTPUT:" in prompt
        assert "email detected: test@example.com" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "Required JSON format:" in prompt
        assert "taxonomy" in prompt
        assert "scores" in prompt
        assert "confidence" in prompt
        assert "context_notes" in prompt

    def test_build_analyst_prompt(self):
        """Test building complete analyst prompt."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        prompt = self.prompt_builder.build_analyst_prompt(
            "PII detected in patient records", context
        )

        assert "You are a compliance analyst with rich context awareness" in prompt
        assert "CONTEXT SUMMARY:" in prompt
        assert "BUSINESS CONTEXT:" in prompt
        assert "POLICY CONTEXT:" in prompt
        assert "ANALYSIS SCENARIO:" in prompt
        assert "PII detected in patient records" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "Required JSON format:" in prompt
        assert "analysis_type" in prompt
        assert "risk_level" in prompt
        assert "recommendations" in prompt
        assert "reason" in prompt
        assert "compliance_frameworks" in prompt
        assert "audit_implications" in prompt


class TestContextAwareCache:
    """Test ContextAwareCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_cache = Mock()
        self.context_cache = ContextAwareCache(self.mock_base_cache)
        self.context_manager = ContextManager()

    def test_get_cache_key_high_sensitivity(self):
        """Test cache key generation with high sensitivity."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        cache_key = self.context_cache.get_cache_key("email detected", context, "high")

        assert "email detected" in cache_key
        assert "healthcare" in cache_key
        assert "strict" in cache_key
        assert "healthcare" in cache_key  # policy bundle
        assert "prod" in cache_key

    def test_get_cache_key_medium_sensitivity(self):
        """Test cache key generation with medium sensitivity."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        cache_key = self.context_cache.get_cache_key(
            "email detected", context, "medium"
        )

        assert "email detected" in cache_key
        assert "healthcare" in cache_key  # industry
        assert "healthcare" in cache_key  # policy bundle
        assert "strict" not in cache_key  # not included in medium sensitivity
        assert "prod" not in cache_key  # not included in medium sensitivity

    def test_get_cache_key_low_sensitivity(self):
        """Test cache key generation with low sensitivity."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        cache_key = self.context_cache.get_cache_key("email detected", context, "low")

        assert "email detected" in cache_key
        assert "healthcare" in cache_key  # only industry included
        assert "strict" not in cache_key
        assert "prod" not in cache_key

    def test_get_cached_result(self):
        """Test getting cached result."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        expected_result = {"taxonomy": ["PII.Email"], "confidence": 0.9}
        self.mock_base_cache.get.return_value = expected_result

        result = self.context_cache.get("email detected", context, "medium")

        assert result == expected_result
        self.mock_base_cache.get.assert_called_once()

    def test_put_cached_result(self):
        """Test putting cached result."""
        business = BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA"],
            risk_tolerance="strict",
            data_classification="confidential",
            jurisdiction=["US"],
        )
        application = ApplicationContext(
            app_name="patient-portal", route="/api/patient-data", environment="prod"
        )
        policy = PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA"],
            enforcement_level="strict",
            audit_requirements=["quarterly"],
            reporting_obligations=["breach"],
        )

        context = EnhancedContext(
            business=business, application=application, policy=policy
        )

        mapping_result = {"taxonomy": ["PII.Email"], "confidence": 0.9}

        self.context_cache.put(
            "email detected", context, mapping_result, "medium", 3600
        )

        self.mock_base_cache.put.assert_called_once()
        # Verify the call was made with the correct arguments
        call_args = self.mock_base_cache.put.call_args
        assert call_args[0][1] == mapping_result  # mapping_result
        assert call_args[0][2] == 3600  # ttl


class TestIntegration:
    """Integration tests for the complete context system."""

    def test_complete_mapper_workflow(self):
        """Test complete mapper workflow with context."""
        context_manager = ContextManager()
        prompt_builder = ContextAwarePromptBuilder(context_manager)

        # Sample mapper payload
        mapper_payload = {
            "detector": "pii-detector",
            "output": "email address detected: john@company.com",
            "tenant_id": "healthcare_tenant",
            "metadata": {
                "policy_bundle": "healthcare",
                "app_name": "patient-portal",
                "route": "/api/patient-data",
                "environment": "prod",
            },
        }

        # Tenant context
        tenant_context = {
            "industry": "healthcare",
            "compliance_requirements": ["HIPAA", "GDPR"],
            "risk_tolerance": "strict",
            "data_classification": "confidential",
        }

        # Historical data
        historical_data = {
            "similar_detections": [{"type": "email", "confidence": 0.9}],
            "false_positive_rate": 0.1,
            "detection_trends": {"email": "stable"},
        }

        # Build enhanced context
        enhanced_context = context_manager.build_mapper_context(
            mapper_payload, tenant_context, historical_data
        )

        # Build context-aware prompt
        prompt = prompt_builder.build_mapper_prompt(
            mapper_payload["output"], enhanced_context
        )

        # Verify the complete workflow
        assert isinstance(enhanced_context, EnhancedContext)
        assert enhanced_context.business.industry == "healthcare"
        assert enhanced_context.policy.policy_bundle == "healthcare"
        assert enhanced_context.historical is not None
        assert enhanced_context.historical.false_positive_rate == 0.1

        assert "healthcare" in prompt
        assert "HIPAA" in prompt
        assert "strict" in prompt
        assert "email address detected: john@company.com" in prompt
        assert "context_notes" in prompt

    def test_complete_analyst_workflow(self):
        """Test complete analyst workflow with context."""
        context_manager = ContextManager()
        prompt_builder = ContextAwarePromptBuilder(context_manager)

        # Sample analysis request
        analysis_request = {
            "tenant": "financial_tenant",
            "app": "trading-platform",
            "route": "/api/transactions",
            "env": "prod",
            "policy_bundle": "financial",
            "high_sev_hits": [{"type": "pii", "severity": "high"}],
            "false_positive_bands": [{"type": "false_positive"}],
        }

        # Tenant context
        tenant_context = {
            "industry": "financial_services",
            "compliance_requirements": ["SOX", "PCI-DSS", "GDPR"],
            "risk_tolerance": "strict",
        }

        # Build enhanced context
        enhanced_context = context_manager.build_analyst_context(
            analysis_request, tenant_context
        )

        # Build context-aware prompt
        prompt = prompt_builder.build_analyst_prompt(
            "High-risk PII detected in transaction logs", enhanced_context
        )

        # Verify the complete workflow
        assert isinstance(enhanced_context, EnhancedContext)
        assert enhanced_context.business.industry == "financial_services"
        assert enhanced_context.policy.policy_bundle == "financial"
        assert "SOX" in enhanced_context.policy.applicable_frameworks
        assert enhanced_context.historical is not None

        assert "financial_services" in prompt
        assert "SOX" in prompt
        assert "strict" in prompt
        assert "High-risk PII detected in transaction logs" in prompt
        assert "analysis_type" in prompt
        assert "risk_level" in prompt
        assert "recommendations" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
