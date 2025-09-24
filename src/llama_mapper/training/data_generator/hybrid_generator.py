"""Hybrid training data generator combining real-world and synthetic data."""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from llama_mapper.data.taxonomy import Taxonomy, TaxonomyLoader

from .models import MapperCanonicalEvent, TrainingExample
from .real_world_collector import RealWorldDataCollector
from .synthetic import SyntheticDataGenerator

logger = logging.getLogger(__name__)


class HybridTrainingDataGenerator:
    """Generates hybrid training datasets combining real-world and synthetic data."""

    def __init__(
        self,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        real_world_ratio: float = 0.6,
        synthetic_ratio: float = 0.4,
        random_seed: Optional[int] = None,
    ):
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self.real_world_ratio = real_world_ratio
        self.synthetic_ratio = synthetic_ratio

        if random_seed is not None:
            random.seed(random_seed)

        self._taxonomy: Optional[Taxonomy] = None
        self.synthetic_generator = SyntheticDataGenerator(taxonomy_loader)
        self.real_world_collector = RealWorldDataCollector(taxonomy_loader)

    def load_taxonomy(self) -> None:
        """Load taxonomy for hybrid data generation."""
        if not self._taxonomy:
            self._taxonomy = self.taxonomy_loader.load_taxonomy()
            self.synthetic_generator.load_taxonomy()
            self.real_world_collector.load_taxonomy()

    def generate_hybrid_training_set(
        self,
        total_examples: int = 2500,
        real_world_categories: Optional[List[str]] = None,
        synthetic_categories: Optional[List[str]] = None,
        balance_categories: bool = True,
    ) -> List[TrainingExample]:
        """Generate a hybrid training set with real-world and synthetic data."""
        if not self._taxonomy:
            self.load_taxonomy()

        logger.info(
            "Generating hybrid training set with %s total examples...", total_examples
        )

        # Calculate target examples for each type
        real_world_target = int(total_examples * self.real_world_ratio)
        synthetic_target = int(total_examples * self.synthetic_ratio)

        logger.info(
            "Target distribution: %s real-world, %s synthetic",
            real_world_target,
            synthetic_target,
        )

        # Generate real-world examples
        real_world_examples = self._generate_real_world_portion(
            real_world_target, real_world_categories
        )

        # Generate synthetic examples
        synthetic_examples = self._generate_synthetic_portion(
            synthetic_target, synthetic_categories
        )

        # Combine and balance
        all_examples = real_world_examples + synthetic_examples

        if balance_categories:
            all_examples = self._balance_categories(all_examples)

        # Shuffle final dataset
        random.shuffle(all_examples)

        logger.info("Generated hybrid training set with %s examples", len(all_examples))
        return all_examples

    def generate_enhanced_synthetic_data(
        self,
        num_examples: int = 1000,
        use_real_patterns: bool = True,
        include_edge_cases: bool = True,
    ) -> List[TrainingExample]:
        """Generate enhanced synthetic data using real-world patterns."""
        if not self._taxonomy:
            self.load_taxonomy()

        logger.info(
            "Generating enhanced synthetic data with %s examples...", num_examples
        )

        examples: List[TrainingExample] = []

        # Base synthetic examples
        base_examples = self.synthetic_generator.generate_balanced_training_set(
            target_examples_per_category=num_examples // 4
        )
        examples.extend(base_examples)

        if use_real_patterns:
            # Enhanced synthetic examples using real patterns
            enhanced_examples = self._generate_pattern_based_synthetic(
                num_examples // 2
            )
            examples.extend(enhanced_examples)

        if include_edge_cases:
            # Synthetic edge cases
            edge_examples = self._generate_synthetic_edge_cases(num_examples // 4)
            examples.extend(edge_examples)

        random.shuffle(examples)

        logger.info("Generated %s enhanced synthetic examples", len(examples))
        return examples

    def generate_industry_specific_hybrid(
        self,
        industry: str,
        total_examples: int = 500,
        real_world_ratio: float = 0.7,
    ) -> List[TrainingExample]:
        """Generate industry-specific hybrid training data."""
        if not self._taxonomy:
            self.load_taxonomy()

        logger.info(
            "Generating %s industry-specific hybrid data with %s examples...",
            industry,
            total_examples,
        )

        real_world_target = int(total_examples * real_world_ratio)
        synthetic_target = total_examples - real_world_target

        # Industry-specific real-world examples
        real_examples = self.real_world_collector.generate_industry_specific_examples(
            industry, real_world_target
        )

        # Industry-specific synthetic examples
        synthetic_examples = self._generate_industry_synthetic_examples(
            industry, synthetic_target
        )

        all_examples = real_examples + synthetic_examples
        random.shuffle(all_examples)

        logger.info("Generated %s %s industry examples", len(all_examples), industry)
        return all_examples

    def generate_edge_case_hybrid(
        self,
        total_examples: int = 300,
        real_world_ratio: float = 0.8,
    ) -> List[TrainingExample]:
        """Generate hybrid edge case training data."""
        if not self._taxonomy:
            self.load_taxonomy()

        logger.info(
            "Generating edge case hybrid data with %s examples...", total_examples
        )

        real_world_target = int(total_examples * real_world_ratio)
        synthetic_target = total_examples - real_world_target

        # Real-world edge cases
        real_edge_examples = self.real_world_collector.generate_edge_case_examples(
            real_world_target
        )

        # Synthetic edge cases
        synthetic_edge_examples = self._generate_synthetic_edge_cases(synthetic_target)

        all_examples = real_edge_examples + synthetic_edge_examples
        random.shuffle(all_examples)

        logger.info("Generated %s edge case examples", len(all_examples))
        return all_examples

    def _generate_real_world_portion(
        self, target_examples: int, categories: Optional[List[str]]
    ) -> List[TrainingExample]:
        """Generate real-world portion of training data."""
        if categories is None:
            categories = ["sec", "gdpr", "hipaa", "industries", "edge_cases"]

        examples: List[TrainingExample] = []
        examples_per_category = target_examples // len(categories)

        if "sec" in categories:
            sec_examples = self.real_world_collector.generate_sec_enforcement_examples(
                examples_per_category
            )
            examples.extend(sec_examples)

        if "gdpr" in categories:
            gdpr_examples = self.real_world_collector.generate_gdpr_violation_examples(
                examples_per_category
            )
            examples.extend(gdpr_examples)

        if "hipaa" in categories:
            hipaa_examples = self.real_world_collector.generate_hipaa_breach_examples(
                examples_per_category
            )
            examples.extend(hipaa_examples)

        if "industries" in categories:
            industries = ["financial", "healthcare", "technology", "retail"]
            industry_examples_per = examples_per_category // len(industries)
            for industry in industries:
                industry_examples = (
                    self.real_world_collector.generate_industry_specific_examples(
                        industry, industry_examples_per
                    )
                )
                examples.extend(industry_examples)

        if "edge_cases" in categories:
            edge_examples = self.real_world_collector.generate_edge_case_examples(
                examples_per_category
            )
            examples.extend(edge_examples)

        return examples

    def _generate_synthetic_portion(
        self, target_examples: int, categories: Optional[List[str]]
    ) -> List[TrainingExample]:
        """Generate synthetic portion of training data."""
        if categories is None:
            categories = ["pii", "prompt_injection", "jailbreak", "augmented"]

        examples: List[TrainingExample] = []
        examples_per_category = target_examples // len(categories)

        if "pii" in categories:
            pii_examples = self.synthetic_generator.generate_synthetic_pii_examples(
                examples_per_category
            )
            examples.extend(pii_examples)

        if "prompt_injection" in categories:
            injection_examples = (
                self.synthetic_generator.generate_synthetic_prompt_injection_examples(
                    examples_per_category
                )
            )
            examples.extend(injection_examples)

        if "jailbreak" in categories:
            jailbreak_examples = (
                self.synthetic_generator.generate_synthetic_jailbreak_examples(
                    examples_per_category
                )
            )
            examples.extend(jailbreak_examples)

        if "augmented" in categories:
            augmented_examples = self._generate_augmented_synthetic_examples(
                examples_per_category
            )
            examples.extend(augmented_examples)

        return examples

    def _generate_pattern_based_synthetic(
        self, num_examples: int
    ) -> List[TrainingExample]:
        """Generate synthetic examples using real-world patterns."""
        examples: List[TrainingExample] = []

        # Get real-world patterns
        real_examples = self.real_world_collector.generate_balanced_real_world_set(50)
        real_patterns = self._extract_patterns_from_real_data(real_examples)

        for _ in range(num_examples):
            # Select a real pattern
            pattern = random.choice(real_patterns)

            # Generate synthetic variation
            synthetic_example = self._create_synthetic_variation(pattern)
            examples.append(synthetic_example)

        return examples

    def _generate_synthetic_edge_cases(
        self, num_examples: int
    ) -> List[TrainingExample]:
        """Generate synthetic edge cases."""
        examples: List[TrainingExample] = []

        edge_case_templates = [
            {
                "instruction": "Map complex multi-category violation: PII exposure via prompt injection",
                "detector_output": "pii_leak + prompt_injection",
                "canonical_labels": [
                    "PII.Contact.Email",
                    "PROMPT_INJECTION.DataExfiltration",
                ],
                "confidence": 0.75,
                "notes": "Synthetic multi-category violation",
            },
            {
                "instruction": "Map regulatory conflict: GDPR vs SOX requirements",
                "detector_output": "regulatory_conflict",
                "canonical_labels": [
                    "COMPLIANCE.GDPR.DataRetention.Article5",
                    "COMPLIANCE.SOX.RecordRetention.Section802",
                ],
                "confidence": 0.65,
                "notes": "Synthetic regulatory conflict scenario",
            },
            {
                "instruction": "Map low-confidence edge case: ambiguous compliance violation",
                "detector_output": "ambiguous_violation",
                "canonical_labels": ["COMPLIANCE.OTHER.Ambiguous"],
                "confidence": 0.55,
                "notes": "Synthetic low-confidence edge case",
            },
        ]

        for _ in range(num_examples):
            template = random.choice(edge_case_templates)

            canonical_event = MapperCanonicalEvent(
                taxonomy=template["canonical_labels"],
                scores={
                    label: template["confidence"]
                    for label in template["canonical_labels"]
                },
                confidence=template["confidence"],
                notes=template["notes"],
                provenance={
                    "detector": "synthetic-edge-generator",
                    "detector_version": "hybrid-v1",
                    "source": "Synthetic Edge Case Generation",
                },
            )

            examples.append(
                TrainingExample(
                    instruction=template["instruction"],
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": "synthetic-edge-generator",
                        "detector_label": "edge_case",
                        "canonical_labels": template["canonical_labels"],
                        "example_type": "synthetic_edge_case",
                        "complexity_level": "high",
                        "multi_category": len(template["canonical_labels"]) > 1,
                    },
                )
            )

        return examples

    def _generate_industry_synthetic_examples(
        self, industry: str, num_examples: int
    ) -> List[TrainingExample]:
        """Generate industry-specific synthetic examples."""
        examples: List[TrainingExample] = []

        industry_templates = {
            "financial": {
                "detectors": ["finra-audit", "sec-compliance", "cfpb-check"],
                "violations": [
                    "insider_trading",
                    "market_manipulation",
                    "consumer_harm",
                ],
                "canonical_labels": [
                    "COMPLIANCE.FINRA.Supervision.Rule3110",
                    "COMPLIANCE.SEC.InsiderTrading.Rule10b5",
                    "COMPLIANCE.CFPB.UnfairPractices.CFPA",
                ],
            },
            "healthcare": {
                "detectors": ["hipaa-audit", "fda-compliance", "clinical-trial"],
                "violations": ["phi_breach", "consent_violation", "data_retention"],
                "canonical_labels": [
                    "COMPLIANCE.HIPAA.DataSecurity.164.308",
                    "COMPLIANCE.FDA.AdverseEvents.21CFR314",
                    "COMPLIANCE.HIPAA.Consent.164.508",
                ],
            },
            "technology": {
                "detectors": ["gdpr-audit", "ccpa-compliance", "ai-bias-check"],
                "violations": [
                    "data_processing",
                    "consent_violation",
                    "algorithmic_bias",
                ],
                "canonical_labels": [
                    "COMPLIANCE.GDPR.DataProcessing.Article6",
                    "COMPLIANCE.CCPA.DataRights.Section1798.105",
                    "COMPLIANCE.AI.Bias.Discrimination",
                ],
            },
            "retail": {
                "detectors": ["pci-audit", "consumer-protection", "data-privacy"],
                "violations": ["data_storage", "consumer_harm", "privacy_violation"],
                "canonical_labels": [
                    "COMPLIANCE.PCI.DataStorage.Requirement3",
                    "COMPLIANCE.CFPB.UnfairPractices.CFPA",
                    "COMPLIANCE.CCPA.DataRights.Section1798.105",
                ],
            },
        }

        if industry not in industry_templates:
            logger.warning("No templates found for industry: %s", industry)
            return examples

        template = industry_templates[industry]

        for _ in range(num_examples):
            detector = random.choice(template["detectors"])
            violation = random.choice(template["violations"])
            canonical_label = random.choice(template["canonical_labels"])
            confidence = random.uniform(0.7, 0.9)

            canonical_event = MapperCanonicalEvent(
                taxonomy=[canonical_label],
                scores={canonical_label: confidence},
                confidence=confidence,
                notes=f"Synthetic {industry} compliance violation: {violation}",
                provenance={
                    "detector": detector,
                    "detector_version": "synthetic-industry-v1",
                    "source": f"Synthetic {industry.title()} Compliance",
                },
            )

            instruction = (
                f"Map {industry} compliance violation: {detector} detected {violation}"
            )

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=canonical_event.to_json(),
                    metadata={
                        "detector": detector,
                        "detector_label": violation,
                        "canonical_label": canonical_label,
                        "example_type": f"synthetic_{industry}_violation",
                        "industry": industry,
                    },
                )
            )

        return examples

    def _generate_augmented_synthetic_examples(
        self, num_examples: int
    ) -> List[TrainingExample]:
        """Generate augmented synthetic examples with variations."""
        examples: List[TrainingExample] = []

        # Get base synthetic examples
        base_examples = self.synthetic_generator.generate_balanced_training_set(50)

        for _ in range(num_examples):
            base_example = random.choice(base_examples)

            # Create variations
            variations = [
                self._add_negation_variation(base_example),
                self._add_tense_variation(base_example),
                self._add_formality_variation(base_example),
                self._add_context_variation(base_example),
            ]

            examples.extend(variations)

        return examples[:num_examples]  # Trim to target size

    def _extract_patterns_from_real_data(
        self, real_examples: List[TrainingExample]
    ) -> List[Dict[str, Any]]:
        """Extract patterns from real-world data for synthetic generation."""
        patterns = []

        for example in real_examples:
            try:
                response_data = json.loads(example.response)
                pattern = {
                    "instruction_template": example.instruction,
                    "canonical_labels": response_data.get("taxonomy", []),
                    "confidence_range": (0.7, 0.95),
                    "metadata": example.metadata,
                }
                patterns.append(pattern)
            except (json.JSONDecodeError, KeyError):
                continue

        return patterns

    def _create_synthetic_variation(self, pattern: Dict[str, Any]) -> TrainingExample:
        """Create synthetic variation from real-world pattern."""
        # Generate variation of the pattern
        confidence = random.uniform(*pattern["confidence_range"])
        canonical_label = random.choice(pattern["canonical_labels"])

        canonical_event = MapperCanonicalEvent(
            taxonomy=[canonical_label],
            scores={canonical_label: confidence},
            confidence=confidence,
            notes=f"Synthetic variation of real-world pattern",
            provenance={
                "detector": "pattern-based-synthetic",
                "detector_version": "hybrid-v1",
                "source": "Real-World Pattern Variation",
            },
        )

        # Create instruction variation
        instruction_variations = [
            pattern["instruction_template"],
            f"Classify compliance violation: {canonical_label}",
            f"Map to canonical taxonomy: {canonical_label}",
        ]
        instruction = random.choice(instruction_variations)

        return TrainingExample(
            instruction=instruction,
            response=canonical_event.to_json(),
            metadata={
                **pattern["metadata"],
                "example_type": "pattern_based_synthetic",
                "source_pattern": "real_world_variation",
            },
        )

    def _add_negation_variation(self, example: TrainingExample) -> TrainingExample:
        """Add negation variation to example."""
        # Simple negation variation
        negated_instruction = example.instruction.replace("Map", "Do not map").replace(
            "Classify", "Do not classify"
        )

        return TrainingExample(
            instruction=negated_instruction,
            response=example.response,
            metadata={
                **example.metadata,
                "variation_type": "negation",
            },
        )

    def _add_tense_variation(self, example: TrainingExample) -> TrainingExample:
        """Add tense variation to example."""
        # Simple tense variation
        tense_variations = [
            example.instruction.replace("Map", "Mapped"),
            example.instruction.replace("Classify", "Classified"),
            example.instruction.replace("Transform", "Transformed"),
        ]

        return TrainingExample(
            instruction=random.choice(tense_variations),
            response=example.response,
            metadata={
                **example.metadata,
                "variation_type": "tense",
            },
        )

    def _add_formality_variation(self, example: TrainingExample) -> TrainingExample:
        """Add formality variation to example."""
        # Simple formality variation
        formal_instruction = example.instruction.replace("Map", "Please map").replace(
            "Classify", "Please classify"
        )

        return TrainingExample(
            instruction=formal_instruction,
            response=example.response,
            metadata={
                **example.metadata,
                "variation_type": "formality",
            },
        )

    def _add_context_variation(self, example: TrainingExample) -> TrainingExample:
        """Add context variation to example."""
        # Add business context
        context_prefixes = [
            "In a financial services context: ",
            "For healthcare compliance: ",
            "In a technology company: ",
            "For retail operations: ",
        ]

        contextual_instruction = random.choice(context_prefixes) + example.instruction

        return TrainingExample(
            instruction=contextual_instruction,
            response=example.response,
            metadata={
                **example.metadata,
                "variation_type": "context",
            },
        )

    def _balance_categories(
        self, examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """Balance categories in the training set."""
        # Group examples by canonical label
        label_groups: Dict[str, List[TrainingExample]] = {}

        for example in examples:
            try:
                response_data = json.loads(example.response)
                canonical_labels = response_data.get("taxonomy", [])

                for label in canonical_labels:
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(example)
            except (json.JSONDecodeError, KeyError):
                continue

        # Balance groups (limit to max examples per label)
        max_examples_per_label = 50
        balanced_examples = []

        for label, group_examples in label_groups.items():
            if len(group_examples) > max_examples_per_label:
                # Randomly sample to balance
                balanced_examples.extend(
                    random.sample(group_examples, max_examples_per_label)
                )
            else:
                balanced_examples.extend(group_examples)

        return balanced_examples

    def get_hybrid_statistics(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Get comprehensive statistics about hybrid training examples."""
        stats: Dict[str, Any] = {
            "total_examples": len(examples),
            "real_world_examples": 0,
            "synthetic_examples": 0,
            "example_types": {},
            "industries": {},
            "regulatory_bodies": {},
            "confidence_stats": {"min": float("inf"), "max": float("-inf"), "avg": 0.0},
            "data_quality_metrics": {
                "real_world_ratio": 0.0,
                "synthetic_ratio": 0.0,
                "edge_case_ratio": 0.0,
                "multi_category_ratio": 0.0,
            },
        }

        confidences: List[float] = []
        edge_cases = 0
        multi_category = 0

        for example in examples:
            metadata = example.metadata
            example_type = metadata.get("example_type", "unknown")
            industry = metadata.get("industry", "unknown")
            regulatory_body = metadata.get("regulatory_body", "unknown")

            # Count real vs synthetic
            if "real_" in example_type:
                stats["real_world_examples"] += 1
            elif "synthetic_" in example_type:
                stats["synthetic_examples"] += 1

            # Count edge cases
            if "edge_case" in example_type:
                edge_cases += 1

            # Count multi-category
            if metadata.get("multi_category", False):
                multi_category += 1

            stats["example_types"][example_type] = (
                stats["example_types"].get(example_type, 0) + 1
            )
            stats["industries"][industry] = stats["industries"].get(industry, 0) + 1
            stats["regulatory_bodies"][regulatory_body] = (
                stats["regulatory_bodies"].get(regulatory_body, 0) + 1
            )

            try:
                response_data = json.loads(example.response)
                confidence = response_data.get("confidence", 0.0)
                confidences.append(confidence)
            except (json.JSONDecodeError, KeyError):
                pass

        # Calculate ratios
        total = len(examples)
        if total > 0:
            stats["data_quality_metrics"]["real_world_ratio"] = (
                stats["real_world_examples"] / total
            )
            stats["data_quality_metrics"]["synthetic_ratio"] = (
                stats["synthetic_examples"] / total
            )
            stats["data_quality_metrics"]["edge_case_ratio"] = edge_cases / total
            stats["data_quality_metrics"]["multi_category_ratio"] = (
                multi_category / total
            )

        if confidences:
            stats["confidence_stats"]["min"] = min(confidences)
            stats["confidence_stats"]["max"] = max(confidences)
            stats["confidence_stats"]["avg"] = sum(confidences) / len(confidences)

        return stats


__all__ = ["HybridTrainingDataGenerator"]
