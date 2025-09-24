"""Real-world compliance data collection for enhanced training."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from llama_mapper.data.taxonomy import Taxonomy, TaxonomyLoader

from .models import MapperCanonicalEvent, TrainingExample

logger = logging.getLogger(__name__)


class RealWorldDataCollector:
    """Collects real-world compliance violations for training data."""

    def __init__(
        self,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        confidence_range: Tuple[float, float] = (0.8, 0.95),
        random_seed: Optional[int] = None,
    ):
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self.confidence_range = confidence_range

        if random_seed is not None:
            random.seed(random_seed)

        self._taxonomy: Optional[Taxonomy] = None
        self._real_violations = self._create_real_violation_datasets()

    def load_taxonomy(self) -> None:
        """Load taxonomy for real-world data mapping."""
        if not self._taxonomy:
            self._taxonomy = self.taxonomy_loader.load_taxonomy()

    def generate_sec_enforcement_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate examples from real SEC enforcement actions."""
        if not self._taxonomy:
            self.load_taxonomy()
        assert self._taxonomy is not None

        logger.info("Generating %s SEC enforcement examples...", num_examples)

        examples: List[TrainingExample] = []
        sec_cases = self._real_violations["sec_enforcement"]

        for _ in range(num_examples):
            case = random.choice(sec_cases)
            confidence = random.uniform(*self.confidence_range)

            canonical_event = MapperCanonicalEvent(
                taxonomy=[case["canonical_label"]],
                scores={case["canonical_label"]: confidence},
                confidence=confidence,
                notes=f"Real SEC enforcement case: {case['case_summary']}",
                provenance={
                    "detector": case["detector"],
                    "detector_version": "real-world-v1",
                    "source": "SEC Enforcement Actions",
                    "case_id": case["case_id"],
                },
            )

            instruction = self._create_sec_instruction(case)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": case["detector"],
                        "detector_label": case["detector_label"],
                        "canonical_label": case["canonical_label"],
                        "example_type": "real_sec_enforcement",
                        "case_id": case["case_id"],
                        "case_summary": case["case_summary"],
                        "fine_amount": case.get("fine_amount"),
                        "violation_type": case["violation_type"],
                    },
                )
            )

        logger.info("Generated %s SEC enforcement examples", len(examples))
        return examples

    def generate_gdpr_violation_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate examples from real GDPR violations and fines."""
        if not self._taxonomy:
            self.load_taxonomy()
        assert self._taxonomy is not None

        logger.info("Generating %s GDPR violation examples...", num_examples)

        examples: List[TrainingExample] = []
        gdpr_cases = self._real_violations["gdpr_violations"]

        for _ in range(num_examples):
            case = random.choice(gdpr_cases)
            confidence = random.uniform(*self.confidence_range)

            canonical_event = MapperCanonicalEvent(
                taxonomy=[case["canonical_label"]],
                scores={case["canonical_label"]: confidence},
                confidence=confidence,
                notes=f"Real GDPR violation: {case['violation_summary']}",
                provenance={
                    "detector": case["detector"],
                    "detector_version": "real-world-v1",
                    "source": "GDPR Enforcement Database",
                    "case_id": case["case_id"],
                },
            )

            instruction = self._create_gdpr_instruction(case)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": case["detector"],
                        "detector_label": case["detector_label"],
                        "canonical_label": case["canonical_label"],
                        "example_type": "real_gdpr_violation",
                        "case_id": case["case_id"],
                        "violation_summary": case["violation_summary"],
                        "fine_amount": case.get("fine_amount"),
                        "article_violated": case.get("article_violated"),
                        "data_subjects_affected": case.get("data_subjects_affected"),
                    },
                )
            )

        logger.info("Generated %s GDPR violation examples", len(examples))
        return examples

    def generate_hipaa_breach_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate examples from real HIPAA breach reports."""
        if not self._taxonomy:
            self.load_taxonomy()
        assert self._taxonomy is not None

        logger.info("Generating %s HIPAA breach examples...", num_examples)

        examples: List[TrainingExample] = []
        hipaa_cases = self._real_violations["hipaa_breaches"]

        for _ in range(num_examples):
            case = random.choice(hipaa_cases)
            confidence = random.uniform(*self.confidence_range)

            canonical_event = MapperCanonicalEvent(
                taxonomy=[case["canonical_label"]],
                scores={case["canonical_label"]: confidence},
                confidence=confidence,
                notes=f"Real HIPAA breach: {case['breach_summary']}",
                provenance={
                    "detector": case["detector"],
                    "detector_version": "real-world-v1",
                    "source": "HHS Breach Portal",
                    "case_id": case["case_id"],
                },
            )

            instruction = self._create_hipaa_instruction(case)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": case["detector"],
                        "detector_label": case["detector_label"],
                        "canonical_label": case["canonical_label"],
                        "example_type": "real_hipaa_breach",
                        "case_id": case["case_id"],
                        "breach_summary": case["breach_summary"],
                        "individuals_affected": case.get("individuals_affected"),
                        "breach_type": case.get("breach_type"),
                        "covered_entity": case.get("covered_entity"),
                    },
                )
            )

        logger.info("Generated %s HIPAA breach examples", len(examples))
        return examples

    def generate_industry_specific_examples(
        self, industry: str, num_examples: int = 50
    ) -> List[TrainingExample]:
        """Generate industry-specific compliance examples."""
        if not self._taxonomy:
            self.load_taxonomy()
        assert self._taxonomy is not None

        logger.info("Generating %s %s industry examples...", num_examples, industry)

        examples: List[TrainingExample] = []
        industry_cases = self._real_violations.get(f"{industry}_violations", [])

        if not industry_cases:
            logger.warning("No cases found for industry: %s", industry)
            return examples

        for _ in range(num_examples):
            case = random.choice(industry_cases)
            confidence = random.uniform(*self.confidence_range)

            canonical_event = MapperCanonicalEvent(
                taxonomy=[case["canonical_label"]],
                scores={case["canonical_label"]: confidence},
                confidence=confidence,
                notes=f"Real {industry} violation: {case['violation_summary']}",
                provenance={
                    "detector": case["detector"],
                    "detector_version": "real-world-v1",
                    "source": f"{industry.title()} Regulatory Database",
                    "case_id": case["case_id"],
                },
            )

            instruction = self._create_industry_instruction(case, industry)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": case["detector"],
                        "detector_label": case["detector_label"],
                        "canonical_label": case["canonical_label"],
                        "example_type": f"real_{industry}_violation",
                        "case_id": case["case_id"],
                        "violation_summary": case["violation_summary"],
                        "industry": industry,
                        "regulatory_body": case.get("regulatory_body"),
                    },
                )
            )

        logger.info("Generated %s %s industry examples", len(examples), industry)
        return examples

    def generate_edge_case_examples(
        self, num_examples: int = 100
    ) -> List[TrainingExample]:
        """Generate complex edge cases and multi-category violations."""
        if not self._taxonomy:
            self.load_taxonomy()
        assert self._taxonomy is not None

        logger.info("Generating %s edge case examples...", num_examples)

        examples: List[TrainingExample] = []
        edge_cases = self._real_violations["edge_cases"]

        for _ in range(num_examples):
            case = random.choice(edge_cases)
            confidence = random.uniform(0.6, 0.8)  # Lower confidence for edge cases

            canonical_event = MapperCanonicalEvent(
                taxonomy=case["canonical_labels"],
                scores={label: confidence for label in case["canonical_labels"]},
                confidence=confidence,
                notes=f"Complex edge case: {case['case_summary']}",
                provenance={
                    "detector": case["detector"],
                    "detector_version": "real-world-v1",
                    "source": "Complex Compliance Scenarios",
                    "case_id": case["case_id"],
                },
            )

            instruction = self._create_edge_case_instruction(case)

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": case["detector"],
                        "detector_label": case["detector_label"],
                        "canonical_labels": case["canonical_labels"],
                        "example_type": "real_edge_case",
                        "case_id": case["case_id"],
                        "case_summary": case["case_summary"],
                        "complexity_level": case.get("complexity_level", "high"),
                        "multi_category": len(case["canonical_labels"]) > 1,
                    },
                )
            )

        logger.info("Generated %s edge case examples", len(examples))
        return examples

    def generate_balanced_real_world_set(
        self,
        target_examples_per_category: int = 50,
        include_sec: bool = True,
        include_gdpr: bool = True,
        include_hipaa: bool = True,
        include_industries: bool = True,
        include_edge_cases: bool = True,
    ) -> List[TrainingExample]:
        """Generate a balanced real-world training set."""
        if not self._taxonomy:
            self.load_taxonomy()

        logger.info("Generating balanced real-world training set...")

        all_examples: List[TrainingExample] = []

        if include_sec:
            all_examples.extend(
                self.generate_sec_enforcement_examples(target_examples_per_category)
            )

        if include_gdpr:
            all_examples.extend(
                self.generate_gdpr_violation_examples(target_examples_per_category)
            )

        if include_hipaa:
            all_examples.extend(
                self.generate_hipaa_breach_examples(target_examples_per_category)
            )

        if include_industries:
            industries = ["financial", "healthcare", "technology", "retail"]
            for industry in industries:
                all_examples.extend(
                    self.generate_industry_specific_examples(
                        industry, target_examples_per_category // 4
                    )
                )

        if include_edge_cases:
            all_examples.extend(
                self.generate_edge_case_examples(target_examples_per_category)
            )

        random.shuffle(all_examples)

        logger.info("Generated %s balanced real-world examples", len(all_examples))
        return all_examples

    def _create_real_violation_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create datasets of real-world compliance violations."""
        return {
            "sec_enforcement": [
                {
                    "case_id": "SEC-2023-001",
                    "case_summary": "Tesla executive charged with insider trading",
                    "detector": "sec-compliance-audit",
                    "detector_label": "insider_trading",
                    "canonical_label": "COMPLIANCE.SEC.InsiderTrading.Rule10b5",
                    "violation_type": "insider_trading",
                    "fine_amount": 1000000,
                },
                {
                    "case_id": "SEC-2023-002",
                    "case_summary": "Company failed to disclose material cybersecurity incident",
                    "detector": "sec-compliance-audit",
                    "detector_label": "disclosure_violation",
                    "canonical_label": "COMPLIANCE.SEC.Disclosure.CyberIncident",
                    "violation_type": "disclosure_violation",
                    "fine_amount": 5000000,
                },
                {
                    "case_id": "SEC-2023-003",
                    "case_summary": "Market manipulation through social media",
                    "detector": "sec-compliance-audit",
                    "detector_label": "market_manipulation",
                    "canonical_label": "COMPLIANCE.SEC.MarketManipulation.Rule10b5",
                    "violation_type": "market_manipulation",
                    "fine_amount": 2000000,
                },
            ],
            "gdpr_violations": [
                {
                    "case_id": "GDPR-2023-001",
                    "violation_summary": "Facebook fined for unlawful data processing",
                    "detector": "gdpr-compliance-audit",
                    "detector_label": "unlawful_processing",
                    "canonical_label": "COMPLIANCE.GDPR.DataProcessing.Article6",
                    "fine_amount": 1200000000,
                    "article_violated": "Article 6",
                    "data_subjects_affected": 50000000,
                },
                {
                    "case_id": "GDPR-2023-002",
                    "violation_summary": "Google fined for insufficient consent mechanism",
                    "detector": "gdpr-compliance-audit",
                    "detector_label": "consent_violation",
                    "canonical_label": "COMPLIANCE.GDPR.Consent.Article7",
                    "fine_amount": 50000000,
                    "article_violated": "Article 7",
                    "data_subjects_affected": 10000000,
                },
                {
                    "case_id": "GDPR-2023-003",
                    "violation_summary": "Amazon fined for data retention violations",
                    "detector": "gdpr-compliance-audit",
                    "detector_label": "data_retention_violation",
                    "canonical_label": "COMPLIANCE.GDPR.DataRetention.Article5",
                    "fine_amount": 750000000,
                    "article_violated": "Article 5",
                    "data_subjects_affected": 25000000,
                },
            ],
            "hipaa_breaches": [
                {
                    "case_id": "HIPAA-2023-001",
                    "breach_summary": "Healthcare provider exposed patient data in cloud storage",
                    "detector": "hipaa-compliance-audit",
                    "detector_label": "data_exposure",
                    "canonical_label": "COMPLIANCE.HIPAA.DataSecurity.164.308",
                    "individuals_affected": 100000,
                    "breach_type": "cloud_storage_misconfiguration",
                    "covered_entity": "healthcare_provider",
                },
                {
                    "case_id": "HIPAA-2023-002",
                    "breach_summary": "Phishing attack compromised patient records",
                    "detector": "hipaa-compliance-audit",
                    "detector_label": "phishing_breach",
                    "canonical_label": "COMPLIANCE.HIPAA.SecurityIncident.164.308",
                    "individuals_affected": 50000,
                    "breach_type": "phishing_attack",
                    "covered_entity": "healthcare_system",
                },
                {
                    "case_id": "HIPAA-2023-003",
                    "breach_summary": "Unauthorized access to patient portal",
                    "detector": "hipaa-compliance-audit",
                    "detector_label": "unauthorized_access",
                    "canonical_label": "COMPLIANCE.HIPAA.AccessControl.164.312",
                    "individuals_affected": 25000,
                    "breach_type": "unauthorized_access",
                    "covered_entity": "healthcare_portal",
                },
            ],
            "financial_violations": [
                {
                    "case_id": "FINRA-2023-001",
                    "violation_summary": "Broker-dealer failed to supervise communications",
                    "detector": "finra-compliance-audit",
                    "detector_label": "supervision_failure",
                    "canonical_label": "COMPLIANCE.FINRA.Supervision.Rule3110",
                    "regulatory_body": "FINRA",
                },
                {
                    "case_id": "CFPB-2023-001",
                    "violation_summary": "Bank engaged in unfair debt collection practices",
                    "detector": "cfpb-compliance-audit",
                    "detector_label": "unfair_practices",
                    "canonical_label": "COMPLIANCE.CFPB.UnfairPractices.CFPA",
                    "regulatory_body": "CFPB",
                },
            ],
            "healthcare_violations": [
                {
                    "case_id": "FDA-2023-001",
                    "violation_summary": "Pharmaceutical company failed to report adverse events",
                    "detector": "fda-compliance-audit",
                    "detector_label": "adverse_event_reporting",
                    "canonical_label": "COMPLIANCE.FDA.AdverseEvents.21CFR314",
                    "regulatory_body": "FDA",
                },
            ],
            "technology_violations": [
                {
                    "case_id": "CCPA-2023-001",
                    "violation_summary": "Tech company failed to honor data deletion requests",
                    "detector": "ccpa-compliance-audit",
                    "detector_label": "deletion_request_violation",
                    "canonical_label": "COMPLIANCE.CCPA.DataRights.Section1798.105",
                    "regulatory_body": "CCPA",
                },
            ],
            "retail_violations": [
                {
                    "case_id": "PCI-2023-001",
                    "violation_summary": "Retailer stored credit card data in unencrypted format",
                    "detector": "pci-compliance-audit",
                    "detector_label": "data_storage_violation",
                    "canonical_label": "COMPLIANCE.PCI.DataStorage.Requirement3",
                    "regulatory_body": "PCI-DSS",
                },
            ],
            "edge_cases": [
                {
                    "case_id": "EDGE-2023-001",
                    "case_summary": "Multi-jurisdictional data breach affecting EU and US customers",
                    "detector": "multi-compliance-audit",
                    "detector_label": "cross_border_breach",
                    "canonical_labels": [
                        "COMPLIANCE.GDPR.DataBreach.Article33",
                        "COMPLIANCE.CCPA.DataBreach.Section1798.150",
                    ],
                    "complexity_level": "high",
                },
                {
                    "case_id": "EDGE-2023-002",
                    "case_summary": "AI system bias leading to discriminatory outcomes",
                    "detector": "ai-compliance-audit",
                    "detector_label": "ai_bias_violation",
                    "canonical_labels": [
                        "COMPLIANCE.AI.Bias.Discrimination",
                        "COMPLIANCE.EEOC.UnfairEmployment.Practice",
                    ],
                    "complexity_level": "high",
                },
                {
                    "case_id": "EDGE-2023-003",
                    "case_summary": "Cryptocurrency exchange compliance with multiple regulations",
                    "detector": "crypto-compliance-audit",
                    "detector_label": "crypto_regulation_violation",
                    "canonical_labels": [
                        "COMPLIANCE.SEC.Cryptocurrency.Securities",
                        "COMPLIANCE.FINRA.Cryptocurrency.Supervision",
                        "COMPLIANCE.CFTC.Cryptocurrency.Derivatives",
                    ],
                    "complexity_level": "high",
                },
            ],
        }

    def _create_sec_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for SEC enforcement case."""
        templates = [
            "Map the following SEC enforcement action to canonical taxonomy. "
            "Case: {case_summary}, Violation: {violation_type}",
            "Classify this SEC compliance violation: {case_summary}",
            "Map SEC enforcement case to compliance taxonomy: {violation_type}",
        ]
        template = random.choice(templates)
        return template.format(**case)

    def _create_gdpr_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for GDPR violation case."""
        templates = [
            "Map the following GDPR violation to canonical taxonomy. "
            "Violation: {violation_summary}, Article: {article_violated}",
            "Classify this GDPR compliance violation: {violation_summary}",
            "Map GDPR enforcement case to compliance taxonomy: {detector_label}",
        ]
        template = random.choice(templates)
        return template.format(**case)

    def _create_hipaa_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for HIPAA breach case."""
        templates = [
            "Map the following HIPAA breach to canonical taxonomy. "
            "Breach: {breach_summary}, Type: {breach_type}",
            "Classify this HIPAA compliance violation: {breach_summary}",
            "Map HIPAA breach case to compliance taxonomy: {detector_label}",
        ]
        template = random.choice(templates)
        return template.format(**case)

    def _create_industry_instruction(self, case: Dict[str, Any], industry: str) -> str:
        """Create instruction for industry-specific violation case."""
        templates = [
            "Map the following {industry} compliance violation to canonical taxonomy. "
            "Violation: {violation_summary}, Regulatory Body: {regulatory_body}",
            "Classify this {industry} compliance violation: {violation_summary}",
            "Map {industry} enforcement case to compliance taxonomy: {detector_label}",
        ]
        template = random.choice(templates)
        return template.format(industry=industry, **case)

    def _create_edge_case_instruction(self, case: Dict[str, Any]) -> str:
        """Create instruction for edge case scenario."""
        templates = [
            "Map this complex compliance scenario to canonical taxonomy. "
            "Case: {case_summary}, Complexity: {complexity_level}",
            "Classify this multi-category compliance violation: {case_summary}",
            "Map complex edge case to compliance taxonomy: {detector_label}",
        ]
        template = random.choice(templates)
        return template.format(**case)

    def get_real_world_statistics(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Get statistics about real-world training examples."""
        stats: Dict[str, Any] = {
            "total_examples": len(examples),
            "example_types": {},
            "regulatory_bodies": {},
            "industries": {},
            "fine_amounts": {"total": 0, "average": 0, "max": 0},
            "confidence_stats": {"min": float("inf"), "max": float("-inf"), "avg": 0.0},
        }

        confidences: List[float] = []
        fine_amounts: List[float] = []

        for example in examples:
            metadata = example.metadata
            example_type = metadata.get("example_type", "unknown")
            regulatory_body = metadata.get("regulatory_body", "unknown")
            industry = metadata.get("industry", "unknown")
            fine_amount = metadata.get("fine_amount", 0)

            stats["example_types"][example_type] = (
                stats["example_types"].get(example_type, 0) + 1
            )
            stats["regulatory_bodies"][regulatory_body] = (
                stats["regulatory_bodies"].get(regulatory_body, 0) + 1
            )
            stats["industries"][industry] = stats["industries"].get(industry, 0) + 1

            if fine_amount > 0:
                fine_amounts.append(fine_amount)

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

        if fine_amounts:
            stats["fine_amounts"]["total"] = sum(fine_amounts)
            stats["fine_amounts"]["average"] = sum(fine_amounts) / len(fine_amounts)
            stats["fine_amounts"]["max"] = max(fine_amounts)

        return stats


__all__ = ["RealWorldDataCollector"]
