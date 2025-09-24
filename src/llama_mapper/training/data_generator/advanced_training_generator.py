"""Advanced training data generator with missing high-value data sources."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from .models import TrainingExample

logger = logging.getLogger(__name__)


class AdvancedTrainingDataGenerator:
    """Generates advanced training data with missing high-value sources."""

    def __init__(
        self,
        confidence_range: Tuple[float, float] = (0.7, 0.95),
        random_seed: Optional[int] = None,
    ):
        self.confidence_range = confidence_range

        if random_seed is not None:
            random.seed(random_seed)

        self._advanced_scenarios = self._create_advanced_scenarios()

    def generate_fda_enforcement_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate FDA enforcement action examples."""
        logger.info("Generating %s FDA enforcement examples...", num_examples)

        examples: List[TrainingExample] = []
        fda_cases = self._advanced_scenarios["fda_enforcement"]

        for _ in range(num_examples):
            case = random.choice(fda_cases)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_fda_instruction(case)
            response = self._create_fda_response(case, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "fda_enforcement",
                        "case_id": case["case_id"],
                        "violation_type": case["violation_type"],
                        "product_type": case["product_type"],
                        "enforcement_action": case["enforcement_action"],
                        "fine_amount": case.get("fine_amount"),
                        "regulatory_framework": "FDA",
                    },
                )
            )

        logger.info("Generated %s FDA enforcement examples", len(examples))
        return examples

    def generate_aml_compliance_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate Anti-Money Laundering compliance examples."""
        logger.info("Generating %s AML compliance examples...", num_examples)

        examples: List[TrainingExample] = []
        aml_cases = self._advanced_scenarios["aml_compliance"]

        for _ in range(num_examples):
            case = random.choice(aml_cases)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_aml_instruction(case)
            response = self._create_aml_response(case, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "aml_compliance",
                        "case_id": case["case_id"],
                        "violation_type": case["violation_type"],
                        "transaction_type": case["transaction_type"],
                        "suspicious_activity": case["suspicious_activity"],
                        "regulatory_framework": "AML",
                    },
                )
            )

        logger.info("Generated %s AML compliance examples", len(examples))
        return examples

    def generate_audit_findings_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate audit findings examples."""
        logger.info("Generating %s audit findings examples...", num_examples)

        examples: List[TrainingExample] = []
        audit_cases = self._advanced_scenarios["audit_findings"]

        for _ in range(num_examples):
            case = random.choice(audit_cases)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_audit_instruction(case)
            response = self._create_audit_response(case, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "audit_finding",
                        "case_id": case["case_id"],
                        "finding_type": case["finding_type"],
                        "severity": case["severity"],
                        "control_deficiency": case["control_deficiency"],
                        "audit_framework": case["audit_framework"],
                    },
                )
            )

        logger.info("Generated %s audit findings examples", len(examples))
        return examples

    def generate_legal_reasoning_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate legal reasoning examples."""
        logger.info("Generating %s legal reasoning examples...", num_examples)

        examples: List[TrainingExample] = []
        legal_cases = self._advanced_scenarios["legal_reasoning"]

        for _ in range(num_examples):
            case = random.choice(legal_cases)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_legal_reasoning_instruction(case)
            response = self._create_legal_reasoning_response(case, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "legal_reasoning",
                        "case_id": case["case_id"],
                        "reasoning_type": case["reasoning_type"],
                        "legal_framework": case["legal_framework"],
                        "complexity_level": case["complexity_level"],
                        "chain_of_thought": case.get("chain_of_thought", False),
                    },
                )
            )

        logger.info("Generated %s legal reasoning examples", len(examples))
        return examples

    def generate_few_shot_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate few-shot learning examples."""
        logger.info("Generating %s few-shot examples...", num_examples)

        examples: List[TrainingExample] = []
        few_shot_templates = self._advanced_scenarios["few_shot_templates"]

        for _ in range(num_examples):
            template = random.choice(few_shot_templates)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_few_shot_instruction(template)
            response = self._create_few_shot_response(template, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "few_shot_learning",
                        "template_type": template["template_type"],
                        "num_examples": template["num_examples"],
                        "reasoning_chain": template.get("reasoning_chain", False),
                        "domain": template["domain"],
                    },
                )
            )

        logger.info("Generated %s few-shot examples", len(examples))
        return examples

    def generate_chain_of_thought_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate chain-of-thought reasoning examples."""
        logger.info("Generating %s chain-of-thought examples...", num_examples)

        examples: List[TrainingExample] = []
        cot_scenarios = self._advanced_scenarios["chain_of_thought"]

        for _ in range(num_examples):
            scenario = random.choice(cot_scenarios)
            confidence = random.uniform(*self.confidence_range)

            instruction = self._create_chain_of_thought_instruction(scenario)
            response = self._create_chain_of_thought_response(scenario, confidence)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response,
                    metadata={
                        "example_type": "chain_of_thought",
                        "scenario_type": scenario["scenario_type"],
                        "reasoning_steps": len(scenario["reasoning_steps"]),
                        "complexity_level": scenario["complexity_level"],
                        "domain": scenario["domain"],
                    },
                )
            )

        logger.info("Generated %s chain-of-thought examples", len(examples))
        return examples

    def _create_advanced_scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create advanced training scenarios."""
        return {
            "fda_enforcement": [
                {
                    "case_id": "FDA-2023-001",
                    "violation_type": "drug_safety_violation",
                    "product_type": "pharmaceutical",
                    "enforcement_action": "Warning Letter",
                    "case_summary": "Company failed to report adverse drug reactions within required timeframe",
                    "fine_amount": 5000000,
                    "canonical_label": "COMPLIANCE.FDA.DrugSafety.AdverseEventReporting",
                },
                {
                    "case_id": "FDA-2023-002",
                    "violation_type": "medical_device_violation",
                    "product_type": "medical_device",
                    "enforcement_action": "Import Alert",
                    "case_summary": "Medical device manufacturer failed to comply with quality system regulations",
                    "fine_amount": 2000000,
                    "canonical_label": "COMPLIANCE.FDA.MedicalDevice.QualitySystem",
                },
                {
                    "case_id": "FDA-2023-003",
                    "violation_type": "food_safety_violation",
                    "product_type": "food",
                    "enforcement_action": "Seizure",
                    "case_summary": "Food manufacturer failed to implement proper sanitation procedures",
                    "fine_amount": 1000000,
                    "canonical_label": "COMPLIANCE.FDA.FoodSafety.Sanitation",
                },
            ],
            "aml_compliance": [
                {
                    "case_id": "AML-2023-001",
                    "violation_type": "suspicious_transaction_reporting",
                    "transaction_type": "wire_transfer",
                    "suspicious_activity": "Large cash deposits followed by immediate wire transfers to high-risk jurisdictions",
                    "case_summary": "Bank failed to file suspicious activity reports for high-risk transactions",
                    "canonical_label": "COMPLIANCE.AML.SuspiciousActivity.Reporting",
                },
                {
                    "case_id": "AML-2023-002",
                    "violation_type": "customer_due_diligence",
                    "transaction_type": "account_opening",
                    "suspicious_activity": "Customer provided false identification documents",
                    "case_summary": "Financial institution failed to verify customer identity and beneficial ownership",
                    "canonical_label": "COMPLIANCE.AML.CustomerDueDiligence.Verification",
                },
                {
                    "case_id": "AML-2023-003",
                    "violation_type": "transaction_monitoring",
                    "transaction_type": "cryptocurrency",
                    "suspicious_activity": "Multiple small transactions to avoid reporting thresholds",
                    "case_summary": "Cryptocurrency exchange failed to monitor transactions for structuring patterns",
                    "canonical_label": "COMPLIANCE.AML.TransactionMonitoring.Structuring",
                },
            ],
            "audit_findings": [
                {
                    "case_id": "AUDIT-2023-001",
                    "finding_type": "control_deficiency",
                    "severity": "high",
                    "control_deficiency": "Segregation of duties violation in financial reporting",
                    "case_summary": "Same employee responsible for recording and approving financial transactions",
                    "audit_framework": "SOX",
                    "canonical_label": "COMPLIANCE.SOX.InternalControls.SegregationOfDuties",
                },
                {
                    "case_id": "AUDIT-2023-002",
                    "finding_type": "compliance_violation",
                    "severity": "medium",
                    "control_deficiency": "Inadequate data backup and recovery procedures",
                    "case_summary": "Company lacks proper backup procedures for critical business data",
                    "audit_framework": "SOC2",
                    "canonical_label": "COMPLIANCE.SOC2.Availability.DataBackup",
                },
                {
                    "case_id": "AUDIT-2023-003",
                    "finding_type": "security_weakness",
                    "severity": "critical",
                    "control_deficiency": "Weak access controls for privileged accounts",
                    "case_summary": "Administrative accounts lack multi-factor authentication",
                    "audit_framework": "ISO27001",
                    "canonical_label": "COMPLIANCE.ISO27001.AccessControl.PrivilegedAccounts",
                },
            ],
            "legal_reasoning": [
                {
                    "case_id": "LEGAL-2023-001",
                    "reasoning_type": "causal_analysis",
                    "legal_framework": "GDPR",
                    "complexity_level": "high",
                    "case_summary": "Analyze the causal chain leading to a data breach and determine liability",
                    "reasoning_steps": [
                        "Identify the initial security vulnerability",
                        "Trace the exploitation pathway",
                        "Determine if reasonable security measures were in place",
                        "Assess whether the breach was foreseeable",
                        "Evaluate the organization's response to the breach",
                    ],
                    "canonical_label": "COMPLIANCE.GDPR.DataBreach.LiabilityAnalysis",
                },
                {
                    "case_id": "LEGAL-2023-002",
                    "reasoning_type": "regulatory_interpretation",
                    "legal_framework": "HIPAA",
                    "complexity_level": "medium",
                    "case_summary": "Interpret HIPAA requirements for cloud service providers",
                    "reasoning_steps": [
                        "Identify the specific HIPAA provisions applicable",
                        "Analyze the cloud service provider's role",
                        "Determine if a business associate agreement is required",
                        "Assess compliance with administrative, physical, and technical safeguards",
                    ],
                    "canonical_label": "COMPLIANCE.HIPAA.CloudServices.BusinessAssociate",
                },
            ],
            "few_shot_templates": [
                {
                    "template_type": "classification_with_examples",
                    "num_examples": 3,
                    "domain": "compliance",
                    "examples": [
                        "SEC insider trading violation → COMPLIANCE.SEC.InsiderTrading.Rule10b5",
                        "GDPR data processing violation → COMPLIANCE.GDPR.DataProcessing.Article6",
                        "HIPAA security incident → COMPLIANCE.HIPAA.SecurityIncident.164.308",
                    ],
                    "target_scenario": "FDA drug safety violation",
                    "expected_output": "COMPLIANCE.FDA.DrugSafety.AdverseEventReporting",
                },
                {
                    "template_type": "reasoning_with_examples",
                    "num_examples": 2,
                    "domain": "compliance_analysis",
                    "reasoning_chain": True,
                    "examples": [
                        "Coverage gap: Detector down → Reason: Service failure → Remediation: Restart service",
                        "Threshold issue: High false positives → Reason: Threshold too low → Remediation: Increase threshold",
                    ],
                    "target_scenario": "Incident analysis: Fraud spike detected",
                    "expected_output": "Reason: Unusual activity pattern → Remediation: Increase detection sensitivity",
                },
            ],
            "chain_of_thought": [
                {
                    "scenario_type": "multi_step_compliance_analysis",
                    "complexity_level": "high",
                    "domain": "compliance",
                    "scenario": "Multi-jurisdictional data breach affecting EU and US customers",
                    "reasoning_steps": [
                        "Step 1: Identify applicable jurisdictions (EU: GDPR, US: CCPA)",
                        "Step 2: Determine breach notification requirements for each jurisdiction",
                        "Step 3: Analyze timing requirements (GDPR: 72 hours, CCPA: varies by state)",
                        "Step 4: Assess data subject rights and notification obligations",
                        "Step 5: Evaluate potential penalties and regulatory actions",
                        "Step 6: Develop coordinated response strategy",
                    ],
                    "canonical_labels": [
                        "COMPLIANCE.GDPR.DataBreach.Article33",
                        "COMPLIANCE.CCPA.DataBreach.Section1798.150",
                    ],
                },
                {
                    "scenario_type": "threshold_optimization_analysis",
                    "complexity_level": "medium",
                    "domain": "compliance_analysis",
                    "scenario": "PII detector producing high false positive rate",
                    "reasoning_steps": [
                        "Step 1: Analyze false positive patterns and frequency",
                        "Step 2: Identify root causes (threshold too low, training data issues)",
                        "Step 3: Evaluate impact on compliance coverage",
                        "Step 4: Consider alternative approaches (ensemble methods, feature engineering)",
                        "Step 5: Test threshold adjustments on validation data",
                        "Step 6: Implement gradual threshold increase with monitoring",
                    ],
                    "canonical_labels": [
                        "COMPLIANCE.Analysis.ThresholdOptimization.PII"
                    ],
                },
            ],
        }

    def _create_fda_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for FDA enforcement case."""
        return (
            f"Map the following FDA enforcement action to canonical taxonomy:\n"
            f"Case: {case['case_summary']}\n"
            f"Violation Type: {case['violation_type']}\n"
            f"Product Type: {case['product_type']}\n"
            f"Enforcement Action: {case['enforcement_action']}"
        )

    def _create_fda_response(self, case: Dict[str, Any], confidence: float) -> str:
        """Create response for FDA enforcement case."""
        return json.dumps(
            {
                "taxonomy": [case["canonical_label"]],
                "scores": {case["canonical_label"]: confidence},
                "confidence": confidence,
                "notes": f"FDA enforcement action: {case['case_summary']}",
                "provenance": {
                    "detector": "fda-compliance-audit",
                    "detector_version": "advanced-v1",
                    "source": "FDA Enforcement Actions",
                    "case_id": case["case_id"],
                },
            },
            ensure_ascii=False,
        )

    def _create_aml_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for AML compliance case."""
        return (
            f"Map the following AML compliance violation to canonical taxonomy:\n"
            f"Case: {case['case_summary']}\n"
            f"Violation Type: {case['violation_type']}\n"
            f"Transaction Type: {case['transaction_type']}\n"
            f"Suspicious Activity: {case['suspicious_activity']}"
        )

    def _create_aml_response(self, case: Dict[str, Any], confidence: float) -> str:
        """Create response for AML compliance case."""
        return json.dumps(
            {
                "taxonomy": [case["canonical_label"]],
                "scores": {case["canonical_label"]: confidence},
                "confidence": confidence,
                "notes": f"AML compliance violation: {case['case_summary']}",
                "provenance": {
                    "detector": "aml-compliance-audit",
                    "detector_version": "advanced-v1",
                    "source": "AML Enforcement Actions",
                    "case_id": case["case_id"],
                },
            },
            ensure_ascii=False,
        )

    def _create_audit_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for audit finding case."""
        return (
            f"Map the following audit finding to canonical taxonomy:\n"
            f"Finding: {case['case_summary']}\n"
            f"Finding Type: {case['finding_type']}\n"
            f"Severity: {case['severity']}\n"
            f"Control Deficiency: {case['control_deficiency']}\n"
            f"Audit Framework: {case['audit_framework']}"
        )

    def _create_audit_response(self, case: Dict[str, Any], confidence: float) -> str:
        """Create response for audit finding case."""
        return json.dumps(
            {
                "taxonomy": [case["canonical_label"]],
                "scores": {case["canonical_label"]: confidence},
                "confidence": confidence,
                "notes": f"Audit finding: {case['case_summary']}",
                "provenance": {
                    "detector": "audit-compliance-audit",
                    "detector_version": "advanced-v1",
                    "source": "Audit Findings Database",
                    "case_id": case["case_id"],
                },
            },
            ensure_ascii=False,
        )

    def _create_legal_reasoning_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for legal reasoning case."""
        return (
            f"Analyze the following legal scenario using step-by-step reasoning:\n"
            f"Scenario: {case['case_summary']}\n"
            f"Legal Framework: {case['legal_framework']}\n"
            f"Reasoning Type: {case['reasoning_type']}\n\n"
            f"Provide a detailed analysis with reasoning steps."
        )

    def _create_legal_reasoning_response(
        self, case: Dict[str, Any], confidence: float
    ) -> str:
        """Create response for legal reasoning case."""
        reasoning_steps = case.get("reasoning_steps", [])
        reasoning_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)]
        )

        return json.dumps(
            {
                "taxonomy": [case["canonical_label"]],
                "scores": {case["canonical_label"]: confidence},
                "confidence": confidence,
                "reasoning_steps": reasoning_steps,
                "reasoning_text": reasoning_text,
                "notes": f"Legal reasoning analysis: {case['case_summary']}",
                "provenance": {
                    "detector": "legal-reasoning-audit",
                    "detector_version": "advanced-v1",
                    "source": "Legal Reasoning Database",
                    "case_id": case["case_id"],
                },
            },
            ensure_ascii=False,
        )

    def _create_few_shot_instruction(self, template: Dict[str, Any]) -> str:
        """Create few-shot learning instruction."""
        examples = template["examples"]
        examples_text = "\n".join(
            [f"Example {i+1}: {example}" for i, example in enumerate(examples)]
        )

        return (
            f"Based on the following examples, classify the new scenario:\n\n"
            f"{examples_text}\n\n"
            f"New scenario: {template['target_scenario']}\n"
            f"Expected output: {template['expected_output']}"
        )

    def _create_few_shot_response(
        self, template: Dict[str, Any], confidence: float
    ) -> str:
        """Create few-shot learning response."""
        return json.dumps(
            {
                "taxonomy": [template["expected_output"]],
                "scores": {template["expected_output"]: confidence},
                "confidence": confidence,
                "reasoning": f"Based on {template['num_examples']} examples, this scenario matches the pattern",
                "notes": f"Few-shot learning classification: {template['target_scenario']}",
                "provenance": {
                    "detector": "few-shot-classifier",
                    "detector_version": "advanced-v1",
                    "source": "Few-Shot Learning Templates",
                    "template_type": template["template_type"],
                },
            },
            ensure_ascii=False,
        )

    def _create_chain_of_thought_instruction(self, scenario: Dict[str, Any]) -> str:
        """Create chain-of-thought instruction."""
        return (
            f"Analyze the following compliance scenario step by step:\n"
            f"Scenario: {scenario['scenario']}\n"
            f"Domain: {scenario['domain']}\n"
            f"Complexity: {scenario['complexity_level']}\n\n"
            f"Provide a detailed step-by-step analysis with reasoning."
        )

    def _create_chain_of_thought_response(
        self, scenario: Dict[str, Any], confidence: float
    ) -> str:
        """Create chain-of-thought response."""
        reasoning_steps = scenario["reasoning_steps"]
        reasoning_text = "\n".join([f"{step}" for step in reasoning_steps])

        return json.dumps(
            {
                "taxonomy": scenario["canonical_labels"],
                "scores": {label: confidence for label in scenario["canonical_labels"]},
                "confidence": confidence,
                "reasoning_steps": reasoning_steps,
                "reasoning_text": reasoning_text,
                "notes": f"Chain-of-thought analysis: {scenario['scenario']}",
                "provenance": {
                    "detector": "chain-of-thought-analyzer",
                    "detector_version": "advanced-v1",
                    "source": "Chain-of-Thought Scenarios",
                    "scenario_type": scenario["scenario_type"],
                },
            },
            ensure_ascii=False,
        )


__all__ = ["AdvancedTrainingDataGenerator"]
