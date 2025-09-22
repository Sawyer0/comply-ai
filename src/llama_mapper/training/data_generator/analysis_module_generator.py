"""Enhanced training data generator for Analysis Module (Phi-3 Mini)."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from .models import TrainingExample

logger = logging.getLogger(__name__)


class AnalysisModuleDataGenerator:
    """Generates training data for the Analysis Module (Phi-3 Mini) compliance analysis."""

    def __init__(
        self,
        confidence_range: Tuple[float, float] = (0.7, 0.95),
        random_seed: Optional[int] = None,
    ):
        self.confidence_range = confidence_range

        if random_seed is not None:
            random.seed(random_seed)

        self._analysis_scenarios = self._create_analysis_scenarios()

    def generate_coverage_gap_examples(self, num_examples: int = 200) -> List[TrainingExample]:
        """Generate coverage gap analysis training examples."""
        logger.info("Generating %s coverage gap analysis examples...", num_examples)

        examples: List[TrainingExample] = []
        scenarios = self._analysis_scenarios["coverage_gaps"]

        for _ in range(min(num_examples, len(scenarios))):
            scenario = random.choice(scenarios)
            confidence = random.uniform(*self.confidence_range)

            # Create structured input for analysis
            analysis_input = {
                "period": "2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
                "tenant": scenario["tenant"],
                "app": scenario["app"],
                "route": scenario["route"],
                "required_detectors": scenario["required_detectors"],
                "observed_coverage": scenario["observed_coverage"],
                "required_coverage": scenario["required_coverage"],
                "detector_errors": scenario.get("detector_errors", {}),
                "high_sev_hits": scenario.get("high_sev_hits", []),
                "false_positive_bands": scenario.get("false_positive_bands", []),
                "policy_bundle": scenario.get("policy_bundle", "riskpack-1.4.0"),
                "env": scenario.get("env", "prod"),
            }

            # Create expected analysis output
            analysis_output = {
                "reason": scenario["reason"],
                "remediation": scenario["remediation"],
                "opa_diff": scenario.get("opa_diff", ""),
                "confidence": confidence,
                "evidence_refs": scenario.get("evidence_refs", []),
                "notes": scenario.get("notes", ""),
            }

            instruction = self._create_coverage_gap_instruction(analysis_input)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=json.dumps(analysis_output, ensure_ascii=False),
                    metadata={
                        "analysis_type": "coverage_gap",
                        "tenant": scenario["tenant"],
                        "app": scenario["app"],
                        "route": scenario["route"],
                        "detectors_affected": len(scenario["required_detectors"]),
                        "coverage_gap_severity": scenario.get("severity", "medium"),
                    },
                )
            )

        logger.info("Generated %s coverage gap analysis examples", len(examples))
        return examples

    def generate_incident_anomaly_examples(self, num_examples: int = 150) -> List[TrainingExample]:
        """Generate incident and anomaly summary training examples."""
        logger.info("Generating %s incident/anomaly analysis examples...", num_examples)

        examples: List[TrainingExample] = []
        scenarios = self._analysis_scenarios["incidents_anomalies"]

        for _ in range(min(num_examples, len(scenarios))):
            scenario = random.choice(scenarios)
            confidence = random.uniform(*self.confidence_range)

            # Create structured input for analysis
            analysis_input = {
                "period": "2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
                "tenant": scenario["tenant"],
                "app": scenario["app"],
                "route": scenario["route"],
                "required_detectors": scenario["required_detectors"],
                "observed_coverage": scenario["observed_coverage"],
                "required_coverage": scenario["required_coverage"],
                "detector_errors": scenario.get("detector_errors", {}),
                "high_sev_hits": scenario.get("high_sev_hits", []),
                "false_positive_bands": scenario.get("false_positive_bands", []),
                "policy_bundle": scenario.get("policy_bundle", "riskpack-1.4.0"),
                "env": scenario.get("env", "prod"),
            }

            # Create expected analysis output
            analysis_output = {
                "reason": scenario["reason"],
                "remediation": scenario["remediation"],
                "opa_diff": scenario.get("opa_diff", ""),
                "confidence": confidence,
                "evidence_refs": scenario.get("evidence_refs", []),
                "notes": scenario.get("notes", ""),
            }

            instruction = self._create_incident_anomaly_instruction(analysis_input)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=json.dumps(analysis_output, ensure_ascii=False),
                    metadata={
                        "analysis_type": "incident_anomaly",
                        "tenant": scenario["tenant"],
                        "app": scenario["app"],
                        "route": scenario["route"],
                        "incident_type": scenario.get("incident_type", "unknown"),
                        "severity": scenario.get("severity", "medium"),
                    },
                )
            )

        logger.info("Generated %s incident/anomaly analysis examples", len(examples))
        return examples

    def generate_threshold_tuning_examples(self, num_examples: int = 100) -> List[TrainingExample]:
        """Generate threshold tuning suggestion training examples."""
        logger.info("Generating %s threshold tuning examples...", num_examples)

        examples: List[TrainingExample] = []
        scenarios = self._analysis_scenarios["threshold_tuning"]

        for _ in range(min(num_examples, len(scenarios))):
            scenario = random.choice(scenarios)
            confidence = random.uniform(*self.confidence_range)

            # Create structured input for analysis
            analysis_input = {
                "period": "2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
                "tenant": scenario["tenant"],
                "app": scenario["app"],
                "route": scenario["route"],
                "required_detectors": scenario["required_detectors"],
                "observed_coverage": scenario["observed_coverage"],
                "required_coverage": scenario["required_coverage"],
                "detector_errors": scenario.get("detector_errors", {}),
                "high_sev_hits": scenario.get("high_sev_hits", []),
                "false_positive_bands": scenario.get("false_positive_bands", []),
                "policy_bundle": scenario.get("policy_bundle", "riskpack-1.4.0"),
                "env": scenario.get("env", "prod"),
            }

            # Create expected analysis output
            analysis_output = {
                "reason": scenario["reason"],
                "remediation": scenario["remediation"],
                "opa_diff": scenario.get("opa_diff", ""),
                "confidence": confidence,
                "evidence_refs": scenario.get("evidence_refs", []),
                "notes": scenario.get("notes", ""),
            }

            instruction = self._create_threshold_tuning_instruction(analysis_input)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=json.dumps(analysis_output, ensure_ascii=False),
                    metadata={
                        "analysis_type": "threshold_tuning",
                        "tenant": scenario["tenant"],
                        "app": scenario["app"],
                        "route": scenario["route"],
                        "tuning_type": scenario.get("tuning_type", "threshold_adjustment"),
                        "fp_rate": scenario.get("fp_rate", 0.0),
                    },
                )
            )

        logger.info("Generated %s threshold tuning examples", len(examples))
        return examples

    def generate_opa_policy_examples(self, num_examples: int = 100) -> List[TrainingExample]:
        """Generate OPA policy generation training examples."""
        logger.info("Generating %s OPA policy examples...", num_examples)

        examples: List[TrainingExample] = []
        scenarios = self._analysis_scenarios["opa_policies"]

        for _ in range(min(num_examples, len(scenarios))):
            scenario = random.choice(scenarios)
            confidence = random.uniform(*self.confidence_range)

            # Create structured input for analysis
            analysis_input = {
                "period": "2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
                "tenant": scenario["tenant"],
                "app": scenario["app"],
                "route": scenario["route"],
                "required_detectors": scenario["required_detectors"],
                "observed_coverage": scenario["observed_coverage"],
                "required_coverage": scenario["required_coverage"],
                "detector_errors": scenario.get("detector_errors", {}),
                "high_sev_hits": scenario.get("high_sev_hits", []),
                "false_positive_bands": scenario.get("false_positive_bands", []),
                "policy_bundle": scenario.get("policy_bundle", "riskpack-1.4.0"),
                "env": scenario.get("env", "prod"),
            }

            # Create expected analysis output
            analysis_output = {
                "reason": scenario["reason"],
                "remediation": scenario["remediation"],
                "opa_diff": scenario.get("opa_diff", ""),
                "confidence": confidence,
                "evidence_refs": scenario.get("evidence_refs", []),
                "notes": scenario.get("notes", ""),
            }

            instruction = self._create_opa_policy_instruction(analysis_input)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=json.dumps(analysis_output, ensure_ascii=False),
                    metadata={
                        "analysis_type": "opa_policy",
                        "tenant": scenario["tenant"],
                        "app": scenario["app"],
                        "route": scenario["route"],
                        "policy_type": scenario.get("policy_type", "coverage_requirement"),
                        "has_opa_diff": bool(scenario.get("opa_diff")),
                    },
                )
            )

        logger.info("Generated %s OPA policy examples", len(examples))
        return examples

    def generate_balanced_analysis_set(
        self,
        target_examples_per_category: int = 100,
        include_coverage_gaps: bool = True,
        include_incidents: bool = True,
        include_threshold_tuning: bool = True,
        include_opa_policies: bool = True,
    ) -> List[TrainingExample]:
        """Generate a balanced analysis training set."""
        logger.info("Generating balanced analysis training set...")

        all_examples: List[TrainingExample] = []

        if include_coverage_gaps:
            all_examples.extend(
                self.generate_coverage_gap_examples(target_examples_per_category)
            )

        if include_incidents:
            all_examples.extend(
                self.generate_incident_anomaly_examples(target_examples_per_category)
            )

        if include_threshold_tuning:
            all_examples.extend(
                self.generate_threshold_tuning_examples(target_examples_per_category)
            )

        if include_opa_policies:
            all_examples.extend(
                self.generate_opa_policy_examples(target_examples_per_category)
            )

        random.shuffle(all_examples)

        logger.info("Generated %s balanced analysis examples", len(all_examples))
        return all_examples

    def _create_analysis_scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create realistic analysis scenarios for training."""
        return {
            "coverage_gaps": [
                {
                    "tenant": "enterprise-client",
                    "app": "user-management",
                    "route": "/api/users/*",
                    "required_detectors": ["pii", "jailbreak", "toxicity"],
                    "observed_coverage": {"pii": 0.95, "jailbreak": 0.0, "toxicity": 0.88},
                    "required_coverage": {"pii": 0.95, "jailbreak": 0.95, "toxicity": 0.90},
                    "detector_errors": {"jailbreak": {"5xx": 142}},
                    "reason": "Jailbreak detector down causing coverage gap",
                    "remediation": "Restart jailbreak detector service",
                    "opa_diff": "",
                    "evidence_refs": ["jailbreak_5xx_errors"],
                    "notes": "High severity gap in security coverage",
                    "severity": "high",
                },
                {
                    "tenant": "fintech-startup",
                    "app": "payment-processing",
                    "route": "/api/payments/*",
                    "required_detectors": ["pii", "fraud", "compliance"],
                    "observed_coverage": {"pii": 0.78, "fraud": 0.92, "compliance": 0.65},
                    "required_coverage": {"pii": 0.95, "fraud": 0.95, "compliance": 0.90},
                    "reason": "PII and compliance detectors below threshold",
                    "remediation": "Increase PII detector sensitivity and add compliance checks",
                    "opa_diff": "",
                    "evidence_refs": ["pii_coverage", "compliance_coverage"],
                    "notes": "Multiple coverage gaps in critical payment flow",
                    "severity": "critical",
                },
                {
                    "tenant": "healthcare-provider",
                    "app": "patient-portal",
                    "route": "/api/patients/*",
                    "required_detectors": ["hipaa", "pii", "access-control"],
                    "observed_coverage": {"hipaa": 0.99, "pii": 0.85, "access-control": 0.70},
                    "required_coverage": {"hipaa": 0.99, "pii": 0.95, "access-control": 0.90},
                    "reason": "Access control detector coverage insufficient",
                    "remediation": "Deploy additional access control monitoring",
                    "opa_diff": "",
                    "evidence_refs": ["access_control_coverage"],
                    "notes": "HIPAA compliance at risk due to access control gap",
                    "severity": "high",
                },
            ],
            "incidents_anomalies": [
                {
                    "tenant": "e-commerce-platform",
                    "app": "checkout",
                    "route": "/api/checkout/*",
                    "required_detectors": ["fraud", "pii", "payment"],
                    "observed_coverage": {"fraud": 0.95, "pii": 0.95, "payment": 0.95},
                    "required_coverage": {"fraud": 0.95, "pii": 0.95, "payment": 0.95},
                    "detector_errors": {},
                    "high_sev_hits": [
                        {"taxonomy": "FRAUD.PaymentFraud", "count": 45, "p95_score": 0.92},
                        {"taxonomy": "PII.Identifier.CreditCard", "count": 12, "p95_score": 0.88},
                    ],
                    "reason": "Payment fraud spike detected in checkout flow",
                    "remediation": "Increase fraud detection sensitivity and review payment patterns",
                    "opa_diff": "",
                    "evidence_refs": ["fraud_hits", "pii_hits"],
                    "notes": "Unusual fraud activity pattern detected",
                    "incident_type": "fraud_spike",
                    "severity": "high",
                },
                {
                    "tenant": "social-media-app",
                    "app": "content-moderation",
                    "route": "/api/posts/*",
                    "required_detectors": ["toxicity", "hate-speech", "spam"],
                    "observed_coverage": {"toxicity": 0.90, "hate-speech": 0.85, "spam": 0.95},
                    "required_coverage": {"toxicity": 0.90, "hate-speech": 0.90, "spam": 0.95},
                    "detector_errors": {"hate-speech": {"5xx": 23, "time_buckets": ["2024-01-15T14:00:00Z"]}},
                    "high_sev_hits": [
                        {"taxonomy": "CONTENT.Toxicity.Hate", "count": 156, "p95_score": 0.94},
                    ],
                    "reason": "Hate speech detector errors causing missed content",
                    "remediation": "Fix hate speech detector and review missed content",
                    "opa_diff": "",
                    "evidence_refs": ["hate_speech_errors", "toxicity_hits"],
                    "notes": "Detector reliability issues affecting content moderation",
                    "incident_type": "detector_failure",
                    "severity": "medium",
                },
            ],
            "threshold_tuning": [
                {
                    "tenant": "banking-app",
                    "app": "account-management",
                    "route": "/api/accounts/*",
                    "required_detectors": ["pii", "fraud", "compliance"],
                    "observed_coverage": {"pii": 0.95, "fraud": 0.95, "compliance": 0.95},
                    "required_coverage": {"pii": 0.95, "fraud": 0.95, "compliance": 0.95},
                    "false_positive_bands": [
                        {"detector": "pii", "score_min": 0.6, "score_max": 0.7, "fp_rate": 0.72},
                        {"detector": "fraud", "score_min": 0.5, "score_max": 0.6, "fp_rate": 0.45},
                    ],
                    "reason": "PII detector producing high false positive rate",
                    "remediation": "Increase PII detector threshold to 0.75",
                    "opa_diff": 'package riskpack\n\npii_threshold = 0.75',
                    "evidence_refs": ["pii_fp_band"],
                    "notes": "Threshold adjustment needed to reduce noise",
                    "tuning_type": "threshold_increase",
                    "fp_rate": 0.72,
                },
                {
                    "tenant": "healthcare-portal",
                    "app": "patient-data",
                    "route": "/api/patients/*",
                    "required_detectors": ["hipaa", "pii", "access-control"],
                    "observed_coverage": {"hipaa": 0.99, "pii": 0.95, "access-control": 0.90},
                    "required_coverage": {"hipaa": 0.99, "pii": 0.95, "access-control": 0.90},
                    "false_positive_bands": [
                        {"detector": "access-control", "score_min": 0.8, "score_max": 0.9, "fp_rate": 0.15},
                    ],
                    "reason": "Access control detector threshold too high",
                    "remediation": "Lower access control threshold to 0.75 for better coverage",
                    "opa_diff": 'package riskpack\n\naccess_control_threshold = 0.75',
                    "evidence_refs": ["access_control_fp_band"],
                    "notes": "Threshold too conservative, missing legitimate access violations",
                    "tuning_type": "threshold_decrease",
                    "fp_rate": 0.15,
                },
            ],
            "opa_policies": [
                {
                    "tenant": "fintech-startup",
                    "app": "payment-processing",
                    "route": "/api/payments/*",
                    "required_detectors": ["fraud", "pii", "compliance"],
                    "observed_coverage": {"fraud": 0.85, "pii": 0.90, "compliance": 0.80},
                    "required_coverage": {"fraud": 0.95, "pii": 0.95, "compliance": 0.90},
                    "reason": "Payment processing requires higher coverage thresholds",
                    "remediation": "Update policy to require 95% coverage for all detectors",
                    "opa_diff": 'package riskpack\n\npayment_coverage_requirements = {\n    "fraud": 0.95,\n    "pii": 0.95,\n    "compliance": 0.90\n}',
                    "evidence_refs": ["coverage_gaps"],
                    "notes": "Policy update to meet regulatory requirements",
                    "policy_type": "coverage_requirement",
                },
                {
                    "tenant": "healthcare-provider",
                    "app": "patient-portal",
                    "route": "/api/patients/*",
                    "required_detectors": ["hipaa", "pii", "access-control"],
                    "observed_coverage": {"hipaa": 0.99, "pii": 0.95, "access-control": 0.90},
                    "required_coverage": {"hipaa": 0.99, "pii": 0.95, "access-control": 0.90},
                    "reason": "HIPAA compliance requires specific detector configuration",
                    "remediation": "Add HIPAA-specific detector requirements to policy",
                    "opa_diff": 'package riskpack\n\nhipaa_requirements = {\n    "hipaa": 0.99,\n    "pii": 0.95,\n    "access_control": 0.90\n}\n\nhipaa_compliance_check = {\n    "required": hipaa_requirements,\n    "enforcement": "strict"\n}',
                    "evidence_refs": ["hipaa_compliance"],
                    "notes": "HIPAA-specific policy configuration",
                    "policy_type": "compliance_requirement",
                },
            ],
        }

    def _create_coverage_gap_instruction(self, analysis_input: Dict[str, Any]) -> str:
        """Create instruction for coverage gap analysis."""
        return (
            f"Analyze the following compliance metrics and provide coverage gap analysis:\n\n"
            f"Period: {analysis_input['period']}\n"
            f"Tenant: {analysis_input['tenant']}\n"
            f"App: {analysis_input['app']}\n"
            f"Route: {analysis_input['route']}\n"
            f"Required Detectors: {analysis_input['required_detectors']}\n"
            f"Observed Coverage: {analysis_input['observed_coverage']}\n"
            f"Required Coverage: {analysis_input['required_coverage']}\n"
            f"Detector Errors: {analysis_input.get('detector_errors', {})}\n"
            f"High Severity Hits: {analysis_input.get('high_sev_hits', [])}\n"
            f"False Positive Bands: {analysis_input.get('false_positive_bands', [])}\n"
            f"Policy Bundle: {analysis_input.get('policy_bundle', 'riskpack-1.4.0')}\n"
            f"Environment: {analysis_input.get('env', 'prod')}\n\n"
            f"Provide analysis with reason (≤20 words), remediation (≤20 words), "
            f"OPA diff (if applicable), confidence score, evidence references, and notes."
        )

    def _create_incident_anomaly_instruction(self, analysis_input: Dict[str, Any]) -> str:
        """Create instruction for incident/anomaly analysis."""
        return (
            f"Analyze the following compliance metrics and provide incident/anomaly summary:\n\n"
            f"Period: {analysis_input['period']}\n"
            f"Tenant: {analysis_input['tenant']}\n"
            f"App: {analysis_input['app']}\n"
            f"Route: {analysis_input['route']}\n"
            f"Required Detectors: {analysis_input['required_detectors']}\n"
            f"Observed Coverage: {analysis_input['observed_coverage']}\n"
            f"Required Coverage: {analysis_input['required_coverage']}\n"
            f"Detector Errors: {analysis_input.get('detector_errors', {})}\n"
            f"High Severity Hits: {analysis_input.get('high_sev_hits', [])}\n"
            f"False Positive Bands: {analysis_input.get('false_positive_bands', [])}\n"
            f"Policy Bundle: {analysis_input.get('policy_bundle', 'riskpack-1.4.0')}\n"
            f"Environment: {analysis_input.get('env', 'prod')}\n\n"
            f"Provide analysis with reason (≤20 words), remediation (≤20 words), "
            f"OPA diff (if applicable), confidence score, evidence references, and notes."
        )

    def _create_threshold_tuning_instruction(self, analysis_input: Dict[str, Any]) -> str:
        """Create instruction for threshold tuning analysis."""
        return (
            f"Analyze the following compliance metrics and provide threshold tuning suggestions:\n\n"
            f"Period: {analysis_input['period']}\n"
            f"Tenant: {analysis_input['tenant']}\n"
            f"App: {analysis_input['app']}\n"
            f"Route: {analysis_input['route']}\n"
            f"Required Detectors: {analysis_input['required_detectors']}\n"
            f"Observed Coverage: {analysis_input['observed_coverage']}\n"
            f"Required Coverage: {analysis_input['required_coverage']}\n"
            f"Detector Errors: {analysis_input.get('detector_errors', {})}\n"
            f"High Severity Hits: {analysis_input.get('high_sev_hits', [])}\n"
            f"False Positive Bands: {analysis_input.get('false_positive_bands', [])}\n"
            f"Policy Bundle: {analysis_input.get('policy_bundle', 'riskpack-1.4.0')}\n"
            f"Environment: {analysis_input.get('env', 'prod')}\n\n"
            f"Provide analysis with reason (≤20 words), remediation (≤20 words), "
            f"OPA diff (if applicable), confidence score, evidence references, and notes."
        )

    def _create_opa_policy_instruction(self, analysis_input: Dict[str, Any]) -> str:
        """Create instruction for OPA policy generation."""
        return (
            f"Analyze the following compliance metrics and generate OPA policy recommendations:\n\n"
            f"Period: {analysis_input['period']}\n"
            f"Tenant: {analysis_input['tenant']}\n"
            f"App: {analysis_input['app']}\n"
            f"Route: {analysis_input['route']}\n"
            f"Required Detectors: {analysis_input['required_detectors']}\n"
            f"Observed Coverage: {analysis_input['observed_coverage']}\n"
            f"Required Coverage: {analysis_input['required_coverage']}\n"
            f"Detector Errors: {analysis_input.get('detector_errors', {})}\n"
            f"High Severity Hits: {analysis_input.get('high_sev_hits', [])}\n"
            f"False Positive Bands: {analysis_input.get('false_positive_bands', [])}\n"
            f"Policy Bundle: {analysis_input.get('policy_bundle', 'riskpack-1.4.0')}\n"
            f"Environment: {analysis_input.get('env', 'prod')}\n\n"
            f"Provide analysis with reason (≤20 words), remediation (≤20 words), "
            f"OPA diff (if applicable), confidence score, evidence references, and notes."
        )

    def get_analysis_statistics(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Get statistics about analysis training examples."""
        stats: Dict[str, Any] = {
            "total_examples": len(examples),
            "analysis_types": {},
            "tenants": {},
            "apps": {},
            "severity_levels": {},
            "confidence_stats": {"min": float("inf"), "max": float("-inf"), "avg": 0.0},
            "opa_policy_stats": {
                "total_with_opa": 0,
                "opa_policy_types": {},
            },
        }

        confidences: List[float] = []

        for example in examples:
            metadata = example.metadata
            analysis_type = metadata.get("analysis_type", "unknown")
            tenant = metadata.get("tenant", "unknown")
            app = metadata.get("app", "unknown")
            severity = metadata.get("severity", "unknown")

            stats["analysis_types"][analysis_type] = (
                stats["analysis_types"].get(analysis_type, 0) + 1
            )
            stats["tenants"][tenant] = (
                stats["tenants"].get(tenant, 0) + 1
            )
            stats["apps"][app] = (
                stats["apps"].get(app, 0) + 1
            )
            stats["severity_levels"][severity] = (
                stats["severity_levels"].get(severity, 0) + 1
            )

            # Check for OPA policies
            if metadata.get("has_opa_diff", False):
                stats["opa_policy_stats"]["total_with_opa"] += 1
                policy_type = metadata.get("policy_type", "unknown")
                stats["opa_policy_stats"]["opa_policy_types"][policy_type] = (
                    stats["opa_policy_stats"]["opa_policy_types"].get(policy_type, 0) + 1
                )

            try:
                response_data = json.loads(example.response)
                confidence = response_data.get("confidence", 0.0)
                confidences.append(confidence)
            except (json.JSONDecodeError, KeyError):
                pass

        if confidences:
            stats["confidence_stats"]["min"] = min(confidences)
            stats["confidence_stats"]["max"] = max(confidences)
            stats["confidence_stats"]["avg"] = sum(confidences) / len(confidences)

        return stats


__all__ = ["AnalysisModuleDataGenerator"]
