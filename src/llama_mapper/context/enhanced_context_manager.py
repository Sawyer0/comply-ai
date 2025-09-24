"""
Enhanced Context Management for Compliance AI Models

Provides rich, structured context for both Mapper and Analyst models including:
- Business context (industry, compliance requirements)
- Application context (app/route/environment)
- Policy context (specific frameworks, regulations)
- Historical context (previous detections, patterns)
- Tenant-specific context (custom configurations)
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class ContextType(Enum):
    """Types of context information."""

    BUSINESS = "business"
    APPLICATION = "application"
    POLICY = "policy"
    HISTORICAL = "historical"
    TENANT = "tenant"
    DETECTOR = "detector"


@dataclass
class BusinessContext:
    """Business and industry context."""

    industry: str  # "financial_services", "healthcare", "technology", "retail"
    compliance_requirements: List[str]  # ["GDPR", "HIPAA", "SOX", "PCI-DSS"]
    risk_tolerance: str  # "low", "medium", "high"
    data_classification: str  # "public", "internal", "confidential", "restricted"
    jurisdiction: List[str]  # ["US", "EU", "CA", "UK"]


@dataclass
class ApplicationContext:
    """Application and system context."""

    app_name: str
    route: str
    environment: str  # "dev", "stage", "prod"
    user_role: Optional[str] = None
    session_id: Optional[str] = None
    request_source: Optional[str] = None  # "api", "ui", "batch", "webhook"


@dataclass
class PolicyContext:
    """Policy and regulatory context."""

    policy_bundle: str
    applicable_frameworks: List[str]
    enforcement_level: str  # "strict", "moderate", "lenient"
    audit_requirements: List[str]
    reporting_obligations: List[str]


@dataclass
class HistoricalContext:
    """Historical patterns and context."""

    similar_detections: List[Dict[str, Any]]
    false_positive_rate: float
    detection_trends: Dict[str, Any]
    previous_mappings: List[Dict[str, Any]]
    confidence_history: List[float]


@dataclass
class TenantContext:
    """Tenant-specific context."""

    tenant_id: str
    custom_taxonomy: Optional[Dict[str, Any]] = None
    custom_policies: Optional[Dict[str, Any]] = None
    compliance_contacts: Optional[List[str]] = None
    escalation_rules: Optional[Dict[str, Any]] = None


@dataclass
class DetectorContext:
    """Detector-specific context."""

    detector_id: str
    detector_type: str
    confidence_threshold: float
    coverage_achieved: float
    contributing_detectors: List[str]
    aggregation_method: str
    processing_time_ms: int


@dataclass
class EnhancedContext:
    """Complete enhanced context for model inference."""

    business: BusinessContext
    application: ApplicationContext
    policy: PolicyContext
    historical: Optional[HistoricalContext] = None
    tenant: Optional[TenantContext] = None
    detector: Optional[DetectorContext] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ContextManager:
    """Manages enhanced context for model inference."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.context_cache = {}
        self.context_history = []

    def build_mapper_context(
        self,
        mapper_payload: Dict[str, Any],
        tenant_context: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
    ) -> EnhancedContext:
        """Build enhanced context for Mapper inference."""

        # Extract basic context from payload
        tenant_id = mapper_payload.get("tenant_id", "unknown")
        detector = mapper_payload.get("detector", "unknown")
        metadata = mapper_payload.get("metadata", {})

        # Build business context (would come from tenant configuration)
        business_context = self._build_business_context(tenant_id, tenant_context)

        # Build application context (would come from request metadata)
        application_context = self._build_application_context(mapper_payload)

        # Build policy context (would come from policy bundle)
        policy_context = self._build_policy_context(metadata.get("policy_bundle"))

        # Build historical context
        historical_context = self._build_historical_context(
            tenant_id, detector, historical_data
        )

        # Build tenant context
        tenant_ctx = self._build_tenant_context(tenant_id, tenant_context)

        # Build detector context
        detector_context = self._build_detector_context(detector, metadata)

        return EnhancedContext(
            business=business_context,
            application=application_context,
            policy=policy_context,
            historical=historical_context,
            tenant=tenant_ctx,
            detector=detector_context,
        )

    def build_analyst_context(
        self,
        analysis_request: Dict[str, Any],
        tenant_context: Optional[Dict[str, Any]] = None,
    ) -> EnhancedContext:
        """Build enhanced context for Analyst inference."""

        # Build business context
        business_context = self._build_business_context(
            analysis_request.get("tenant", "unknown"), tenant_context
        )

        # Build application context
        application_context = ApplicationContext(
            app_name=analysis_request.get("app", "unknown"),
            route=analysis_request.get("route", "unknown"),
            environment=analysis_request.get("env", "prod"),
        )

        # Build policy context
        policy_context = self._build_policy_context(
            analysis_request.get("policy_bundle")
        )

        # Build historical context for analysis
        historical_context = self._build_analysis_historical_context(analysis_request)

        return EnhancedContext(
            business=business_context,
            application=application_context,
            policy=policy_context,
            historical=historical_context,
        )

    def _build_business_context(
        self, tenant_id: str, tenant_context: Optional[Dict[str, Any]]
    ) -> BusinessContext:
        """Build business context from tenant information."""

        # Default business context (would be enriched from tenant config)
        default_context = BusinessContext(
            industry="technology",
            compliance_requirements=["GDPR", "CCPA"],
            risk_tolerance="medium",
            data_classification="internal",
            jurisdiction=["US", "EU"],
        )

        if tenant_context:
            # Override with tenant-specific context
            return BusinessContext(
                industry=tenant_context.get("industry", default_context.industry),
                compliance_requirements=tenant_context.get(
                    "compliance_requirements", default_context.compliance_requirements
                ),
                risk_tolerance=tenant_context.get(
                    "risk_tolerance", default_context.risk_tolerance
                ),
                data_classification=tenant_context.get(
                    "data_classification", default_context.data_classification
                ),
                jurisdiction=tenant_context.get(
                    "jurisdiction", default_context.jurisdiction
                ),
            )

        return default_context

    def _build_application_context(self, payload: Dict[str, Any]) -> ApplicationContext:
        """Build application context from request payload."""

        # Extract from metadata or headers
        metadata = payload.get("metadata", {})

        return ApplicationContext(
            app_name=metadata.get("app_name", "unknown"),
            route=metadata.get("route", "unknown"),
            environment=metadata.get("environment", "prod"),
            user_role=metadata.get("user_role"),
            session_id=metadata.get("session_id"),
            request_source=metadata.get("request_source"),
        )

    def _build_policy_context(self, policy_bundle: Optional[str]) -> PolicyContext:
        """Build policy context from policy bundle."""

        if not policy_bundle:
            return PolicyContext(
                policy_bundle="default",
                applicable_frameworks=["GDPR", "CCPA"],
                enforcement_level="moderate",
                audit_requirements=["monthly_review"],
                reporting_obligations=["incident_reporting"],
            )

        # Map policy bundles to frameworks (would be from policy service)
        policy_mapping = {
            "healthcare": {
                "applicable_frameworks": ["HIPAA", "GDPR"],
                "enforcement_level": "strict",
                "audit_requirements": ["quarterly_audit", "incident_review"],
                "reporting_obligations": [
                    "breach_notification",
                    "compliance_reporting",
                ],
            },
            "financial": {
                "applicable_frameworks": ["SOX", "PCI-DSS", "GDPR"],
                "enforcement_level": "strict",
                "audit_requirements": ["annual_audit", "quarterly_review"],
                "reporting_obligations": ["regulatory_reporting", "incident_reporting"],
            },
            "default": {
                "applicable_frameworks": ["GDPR", "CCPA"],
                "enforcement_level": "moderate",
                "audit_requirements": ["monthly_review"],
                "reporting_obligations": ["incident_reporting"],
            },
        }

        policy_config = policy_mapping.get(policy_bundle, policy_mapping["default"])

        return PolicyContext(
            policy_bundle=policy_bundle,
            applicable_frameworks=policy_config["applicable_frameworks"],
            enforcement_level=policy_config["enforcement_level"],
            audit_requirements=policy_config["audit_requirements"],
            reporting_obligations=policy_config["reporting_obligations"],
        )

    def _build_historical_context(
        self, tenant_id: str, detector: str, historical_data: Optional[Dict[str, Any]]
    ) -> Optional[HistoricalContext]:
        """Build historical context from past detections."""

        if not historical_data:
            return None

        return HistoricalContext(
            similar_detections=historical_data.get("similar_detections", []),
            false_positive_rate=historical_data.get("false_positive_rate", 0.1),
            detection_trends=historical_data.get("detection_trends", {}),
            previous_mappings=historical_data.get("previous_mappings", []),
            confidence_history=historical_data.get("confidence_history", []),
        )

    def _build_analysis_historical_context(
        self, analysis_request: Dict[str, Any]
    ) -> Optional[HistoricalContext]:
        """Build historical context for analysis requests."""

        # Extract historical patterns from analysis request
        high_sev_hits = analysis_request.get("high_sev_hits", [])
        false_positive_bands = analysis_request.get("false_positive_bands", [])

        if not high_sev_hits and not false_positive_bands:
            return None

        return HistoricalContext(
            similar_detections=high_sev_hits,
            false_positive_rate=len(false_positive_bands) / max(len(high_sev_hits), 1),
            detection_trends={
                "coverage_gaps": analysis_request.get("observed_coverage", {}),
                "detector_errors": analysis_request.get("detector_errors", {}),
            },
            previous_mappings=[],
            confidence_history=[],
        )

    def _build_tenant_context(
        self, tenant_id: str, tenant_context: Optional[Dict[str, Any]]
    ) -> Optional[TenantContext]:
        """Build tenant-specific context."""

        if not tenant_context:
            return None

        return TenantContext(
            tenant_id=tenant_id,
            custom_taxonomy=tenant_context.get("custom_taxonomy"),
            custom_policies=tenant_context.get("custom_policies"),
            compliance_contacts=tenant_context.get("compliance_contacts"),
            escalation_rules=tenant_context.get("escalation_rules"),
        )

    def _build_detector_context(
        self, detector: str, metadata: Dict[str, Any]
    ) -> DetectorContext:
        """Build detector-specific context."""

        return DetectorContext(
            detector_id=detector,
            detector_type=metadata.get("detector_type", "unknown"),
            confidence_threshold=metadata.get("confidence_threshold", 0.7),
            coverage_achieved=metadata.get("coverage_achieved", 1.0),
            contributing_detectors=metadata.get("contributing_detectors", [detector]),
            aggregation_method=metadata.get("aggregation_method", "highest_confidence"),
            processing_time_ms=metadata.get("processing_time_ms", 0),
        )


class ContextAwarePromptBuilder:
    """Builds context-aware prompts for model inference."""

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def build_mapper_prompt(
        self, detector_output: str, enhanced_context: EnhancedContext
    ) -> str:
        """Build context-aware prompt for Mapper."""

        # Build context summary
        context_summary = self._build_context_summary(enhanced_context)

        # Build business-specific instructions
        business_instructions = self._build_business_instructions(
            enhanced_context.business
        )

        # Build policy-specific instructions
        policy_instructions = self._build_policy_instructions(enhanced_context.policy)

        # Build historical context
        historical_context = self._build_historical_instructions(
            enhanced_context.historical
        )

        prompt = f"""You are a compliance taxonomy mapper with rich context awareness.

CONTEXT SUMMARY:
{context_summary}

BUSINESS CONTEXT:
{business_instructions}

POLICY CONTEXT:
{policy_instructions}

HISTORICAL CONTEXT:
{historical_context}

DETECTOR OUTPUT:
{detector_output}

INSTRUCTIONS:
- Map to canonical taxonomy considering all context above
- Apply business-specific compliance requirements
- Consider historical patterns and false positive rates
- Provide confidence score based on context alignment

Required JSON format:
{{
  "taxonomy": ["CATEGORY.Subcategory.Item"],
  "scores": {{"CATEGORY.Subcategory.Item": 0.95}},
  "confidence": 0.95,
  "context_notes": "Brief context consideration notes"
}}"""

        return prompt

    def build_analyst_prompt(
        self, analysis_scenario: str, enhanced_context: EnhancedContext
    ) -> str:
        """Build context-aware prompt for Analyst."""

        # Build context summary
        context_summary = self._build_context_summary(enhanced_context)

        # Build business-specific analysis instructions
        business_instructions = self._build_business_analysis_instructions(
            enhanced_context.business
        )

        # Build policy-specific analysis instructions
        policy_instructions = self._build_policy_analysis_instructions(
            enhanced_context.policy
        )

        prompt = f"""You are a compliance analyst with rich context awareness.

CONTEXT SUMMARY:
{context_summary}

BUSINESS CONTEXT:
{business_instructions}

POLICY CONTEXT:
{policy_instructions}

ANALYSIS SCENARIO:
{analysis_scenario}

INSTRUCTIONS:
- Provide analysis considering business context and risk tolerance
- Apply relevant compliance frameworks and enforcement levels
- Consider audit requirements and reporting obligations
- Provide actionable recommendations

Required JSON format:
{{
  "analysis_type": "compliance_analysis",
  "risk_level": "medium",
  "recommendations": ["actionable recommendation"],
  "reason": "Context-aware reasoning (max 120 chars)",
  "compliance_frameworks": ["GDPR", "CCPA"],
  "audit_implications": ["audit requirement"]
}}"""

        return prompt

    def _build_context_summary(self, context: EnhancedContext) -> str:
        """Build concise context summary."""
        summary_parts = [
            f"Industry: {context.business.industry}",
            f"Environment: {context.application.environment}",
            f"Frameworks: {', '.join(context.policy.applicable_frameworks)}",
            f"Risk Tolerance: {context.business.risk_tolerance}",
        ]

        if context.tenant:
            summary_parts.append(f"Tenant: {context.tenant.tenant_id}")

        if context.detector:
            summary_parts.append(f"Detector: {context.detector.detector_id}")

        return "\n".join(summary_parts)

    def _build_business_instructions(self, business: BusinessContext) -> str:
        """Build business-specific mapping instructions."""
        instructions = [
            f"Industry: {business.industry}",
            f"Compliance Requirements: {', '.join(business.compliance_requirements)}",
            f"Risk Tolerance: {business.risk_tolerance}",
            f"Data Classification: {business.data_classification}",
        ]

        if business.risk_tolerance == "low":
            instructions.append(
                "Apply conservative mapping with higher confidence thresholds"
            )
        elif business.risk_tolerance == "high":
            instructions.append(
                "Apply aggressive mapping with lower confidence thresholds"
            )

        return "\n".join(instructions)

    def _build_policy_instructions(self, policy: PolicyContext) -> str:
        """Build policy-specific mapping instructions."""
        instructions = [
            f"Policy Bundle: {policy.policy_bundle}",
            f"Applicable Frameworks: {', '.join(policy.applicable_frameworks)}",
            f"Enforcement Level: {policy.enforcement_level}",
        ]

        if policy.enforcement_level == "strict":
            instructions.append(
                "Apply strict compliance mapping with detailed taxonomy"
            )
        elif policy.enforcement_level == "lenient":
            instructions.append(
                "Apply flexible compliance mapping with broader categories"
            )

        return "\n".join(instructions)

    def _build_historical_instructions(
        self, historical: Optional[HistoricalContext]
    ) -> str:
        """Build historical context instructions."""
        if not historical:
            return "No historical context available"

        instructions = [
            f"False Positive Rate: {historical.false_positive_rate:.1%}",
            f"Similar Detections: {len(historical.similar_detections)}",
        ]

        if historical.false_positive_rate > 0.2:
            instructions.append("High false positive rate - apply conservative mapping")

        if historical.detection_trends:
            instructions.append(
                f"Trends: {json.dumps(historical.detection_trends, indent=2)}"
            )

        return "\n".join(instructions)

    def _build_business_analysis_instructions(self, business: BusinessContext) -> str:
        """Build business-specific analysis instructions."""
        instructions = [
            f"Industry: {business.industry}",
            f"Compliance Requirements: {', '.join(business.compliance_requirements)}",
            f"Risk Tolerance: {business.risk_tolerance}",
        ]

        if business.industry == "healthcare":
            instructions.append("Focus on HIPAA compliance and patient data protection")
        elif business.industry == "financial_services":
            instructions.append("Focus on SOX, PCI-DSS, and financial data protection")

        return "\n".join(instructions)

    def _build_policy_analysis_instructions(self, policy: PolicyContext) -> str:
        """Build policy-specific analysis instructions."""
        instructions = [
            f"Enforcement Level: {policy.enforcement_level}",
            f"Audit Requirements: {', '.join(policy.audit_requirements)}",
            f"Reporting Obligations: {', '.join(policy.reporting_obligations)}",
        ]

        if policy.enforcement_level == "strict":
            instructions.append(
                "Provide detailed analysis with specific remediation steps"
            )

        return "\n".join(instructions)


class ContextAwareCache:
    """Context-aware caching that considers business and policy context."""

    def __init__(self, base_cache):
        self.base_cache = base_cache
        self.context_manager = ContextManager()

    def get_cache_key(
        self,
        detector_output: str,
        enhanced_context: EnhancedContext,
        context_sensitivity: str = "medium",
    ) -> str:
        """Generate context-aware cache key."""

        # Base key from detector output
        base_key = detector_output

        # Add context-sensitive components based on sensitivity level
        if context_sensitivity == "high":
            # Include all context components
            context_components = [
                enhanced_context.business.industry,
                enhanced_context.business.risk_tolerance,
                enhanced_context.policy.policy_bundle,
                enhanced_context.application.environment,
            ]
        elif context_sensitivity == "medium":
            # Include key context components
            context_components = [
                enhanced_context.business.industry,
                enhanced_context.policy.policy_bundle,
            ]
        else:  # low
            # Minimal context
            context_components = [enhanced_context.business.industry]

        # Create context-aware key
        context_key = "|".join(context_components)
        return f"{base_key}|{context_key}"

    def get(
        self,
        detector_output: str,
        enhanced_context: EnhancedContext,
        context_sensitivity: str = "medium",
    ) -> Optional[Dict[str, Any]]:
        """Get cached result with context awareness."""
        cache_key = self.get_cache_key(
            detector_output, enhanced_context, context_sensitivity
        )
        return self.base_cache.get(cache_key)

    def put(
        self,
        detector_output: str,
        enhanced_context: EnhancedContext,
        mapping_result: Dict[str, Any],
        context_sensitivity: str = "medium",
        ttl: Optional[int] = None,
    ) -> None:
        """Put result in cache with context awareness."""
        cache_key = self.get_cache_key(
            detector_output, enhanced_context, context_sensitivity
        )
        self.base_cache.put(cache_key, mapping_result, ttl)


# Example usage and testing
if __name__ == "__main__":
    # Create context manager
    context_manager = ContextManager()

    # Sample mapper payload
    mapper_payload = {
        "detector": "pii-detector",
        "output": "email address detected: john@company.com",
        "tenant_id": "tenant_123",
        "metadata": {
            "policy_bundle": "healthcare",
            "app_name": "patient-portal",
            "route": "/api/patient-data",
            "environment": "prod",
        },
    }

    # Build enhanced context
    enhanced_context = context_manager.build_mapper_context(mapper_payload)

    print("Enhanced Context Built:")
    print(f"  Business: {enhanced_context.business.industry}")
    print(f"  Application: {enhanced_context.application.app_name}")
    print(f"  Policy: {enhanced_context.policy.policy_bundle}")
    print(f"  Frameworks: {enhanced_context.policy.applicable_frameworks}")

    # Build context-aware prompt
    prompt_builder = ContextAwarePromptBuilder(context_manager)
    prompt = prompt_builder.build_mapper_prompt(
        mapper_payload["output"], enhanced_context
    )

    print(f"\nContext-Aware Prompt Generated:")
    print(f"  Length: {len(prompt)} characters")
    print(f"  Includes: Business, Policy, Historical context")

    print(f"\n🎉 Enhanced Context Management Ready!")
    print(f"  - Rich business context (industry, compliance, risk tolerance)")
    print(f"  - Application context (app, route, environment)")
    print(f"  - Policy context (frameworks, enforcement, audit requirements)")
    print(f"  - Historical context (patterns, false positives, trends)")
    print(f"  - Context-aware caching and prompts")
