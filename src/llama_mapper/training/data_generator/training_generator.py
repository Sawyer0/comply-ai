"""Generation of instruction-following training examples from detector mappings."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_mapper.data.detectors import DetectorConfigLoader, DetectorMapping
from llama_mapper.data.taxonomy import Taxonomy, TaxonomyLoader

from .models import MapperCanonicalEvent, TrainingExample

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates instruction-following training examples from detector mappings."""

    def __init__(
        self,
        detector_loader: Optional[DetectorConfigLoader] = None,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        confidence_range: Tuple[float, float] = (0.7, 0.95),
        random_seed: Optional[int] = None,
    ):
        self.detector_loader = detector_loader or DetectorConfigLoader()
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self.confidence_range = confidence_range

        if random_seed is not None:
            random.seed(random_seed)

        self._taxonomy: Optional[Taxonomy] = None
        self._detector_mappings: Dict[str, DetectorMapping] = {}
        self._instruction_templates = self._create_instruction_templates()

    def load_data(self) -> None:
        """Load taxonomy and detector configurations."""
        logger.info("Loading taxonomy and detector configurations...")

        self._taxonomy = self.taxonomy_loader.load_taxonomy()
        self._detector_mappings = self.detector_loader.load_all_detector_configs()

        logger.info(
            "Loaded taxonomy with %s labels", len(self._taxonomy.get_all_labels())
        )
        logger.info("Loaded %s detector configurations", len(self._detector_mappings))

    def generate_training_examples(
        self,
        examples_per_mapping: int = 3,
        include_multi_label: bool = True,
        include_variations: bool = True,
    ) -> List[TrainingExample]:
        """Generate training examples from all detector mappings."""
        if not self._taxonomy or not self._detector_mappings:
            self.load_data()

        logger.info(
            "Generating training examples with %s examples per mapping...",
            examples_per_mapping,
        )

        all_examples: List[TrainingExample] = []

        for detector_name, detector_mapping in self._detector_mappings.items():
            logger.debug("Generating examples for detector: %s", detector_name)

            for detector_label, canonical_label in detector_mapping.maps.items():
                examples = self._generate_examples_for_mapping(
                    detector_name=detector_name,
                    detector_label=detector_label,
                    canonical_label=canonical_label,
                    num_examples=examples_per_mapping,
                    include_variations=include_variations,
                )
                all_examples.extend(examples)

        if include_multi_label:
            multi_label_examples = self._generate_multi_label_examples()
            all_examples.extend(multi_label_examples)

        logger.info("Generated %s total training examples", len(all_examples))
        return all_examples

    def _generate_examples_for_mapping(
        self,
        detector_name: str,
        detector_label: str,
        canonical_label: str,
        num_examples: int = 3,
        include_variations: bool = True,
    ) -> List[TrainingExample]:
        """Generate training examples for a specific detector mapping."""
        examples: List[TrainingExample] = []

        assert self._taxonomy is not None
        taxonomy_label = self._taxonomy.get_label_by_name(canonical_label)
        if not taxonomy_label:
            logger.warning("Canonical label not found in taxonomy: %s", canonical_label)
            return examples

        base_confidence = random.uniform(*self.confidence_range)
        canonical_event = MapperCanonicalEvent(
            taxonomy=[canonical_label],
            scores={canonical_label: base_confidence},
            confidence=base_confidence,
            provenance={
                "detector": detector_name,
                "detector_version": self._detector_mappings[detector_name].version,
            },
        )

        for _ in range(num_examples):
            instruction = random.choice(self._instruction_templates).format(
                detector=detector_name,
                detector_output=detector_label,
                output=canonical_label,
            )

            variation_factor = random.uniform(0.9, 1.1)
            confidence = min(
                max(base_confidence * variation_factor, self.confidence_range[0]),
                self.confidence_range[1],
            )

            scores = {canonical_label: confidence}
            if (
                include_variations
                and hasattr(taxonomy_label, "related_labels")
                and taxonomy_label.related_labels
            ):  # type: ignore[attr-defined]
                related_label = random.choice(
                    list(taxonomy_label.related_labels)
                )  # type: ignore[attr-defined]
                scores[related_label] = max(confidence - 0.1, 0.5)

            response = MapperCanonicalEvent(
                taxonomy=list(scores.keys()),
                scores=scores,
                confidence=confidence,
                provenance=canonical_event.provenance,
            )

            metadata = {
                "detector": detector_name,
                "detector_label": detector_label,
                "canonical_label": canonical_label,
                "example_type": "single_label",
                "confidence": confidence,
            }

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response.to_json(),
                    metadata=metadata,
                )
            )

        return examples

    def _generate_multi_label_examples(self) -> List[TrainingExample]:
        """Generate examples that require mapping to multiple canonical labels."""
        examples: List[TrainingExample] = []
        multi_label_candidates = self._identify_multi_label_scenarios()

        for scenario in multi_label_candidates:
            detector_name = scenario["detector"]
            detector_output = scenario["detector_output"]
            canonical_labels = scenario["canonical_labels"]

            instruction = (
                f"The detector {detector_name} produced a complex output involving "
                f"multiple categories for '{detector_output}'. Map it to the canonical taxonomy."
            )

            scores = {label: random.uniform(0.6, 0.9) for label in canonical_labels}
            confidence = sum(scores.values()) / len(scores)

            response = MapperCanonicalEvent(
                taxonomy=canonical_labels,
                scores=scores,
                confidence=confidence,
                provenance={
                    "detector": detector_name,
                    "detector_version": self._detector_mappings[detector_name].version,
                    "scenario": "multi_label",
                },
            )

            metadata = {
                "detector": detector_name,
                "detector_label": detector_output,
                "canonical_label": canonical_labels,
                "example_type": "multi_label",
                "confidence": confidence,
            }

            examples.append(
                TrainingExample(
                    instruction=instruction,
                    response=response.to_json(),
                    metadata=metadata,
                )
            )

        return examples

    def _identify_multi_label_scenarios(self) -> List[Dict[str, Any]]:
        """Identify scenarios where a detector output maps to multiple canonical labels."""
        scenarios: List[Dict[str, Any]] = []

        assert self._detector_mappings is not None
        assert self._taxonomy is not None

        for detector_name, mapping in self._detector_mappings.items():
            reverse_mapping: Dict[str, List[str]] = {}

            for detector_output, canonical_label in mapping.maps.items():
                reverse_mapping.setdefault(canonical_label, []).append(detector_output)

            for canonical_label, detector_outputs in reverse_mapping.items():
                if len(detector_outputs) > 1:
                    taxonomy_label = self._taxonomy.get_label_by_name(canonical_label)
                    related_labels = (
                        getattr(taxonomy_label, "related_labels", set())
                        if taxonomy_label
                        else set()  # type: ignore[attr-defined]
                    )

                    scenarios.append(
                        {
                            "detector": detector_name,
                            "canonical_labels": [canonical_label, *related_labels],
                            "detector_output": random.choice(detector_outputs),
                        }
                    )

        return scenarios

    def _could_be_related(self, label_one: str, label_two: str) -> bool:
        """Check if two taxonomy labels could be related based on hierarchy."""
        if label_one == label_two:
            return True

        if label_one.startswith(label_two.split(".")[0]):
            return True

        if label_two.startswith(label_one.split(".")[0]):
            return True

        return False

    def _create_instruction_templates(self) -> List[str]:
        """Create instruction templates for detector mapping scenarios."""
        return [
            "Transform detector output to taxonomy format. Source: {detector}, "
            "Label: {detector_output}",
            "Map detector result to canonical taxonomy: {detector} → {output}",
            "Convert to canonical format: {detector} output '{detector_output}'",
            "Standardize this detection: {detector} found '{output}'",
            "Canonical mapping for: {detector} detector output " "'{detector_output}'",
            "Normalize to taxonomy: {detector} result = {output}",
            "Map to canonical labels: {detector} detected '{detector_output}'",
        ]

    def save_training_data(
        self,
        examples: List[TrainingExample],
        output_path: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save training examples to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving %s training examples to %s", len(examples), output_path)

        if format.lower() == "jsonl":
            with open(output_path, "w", encoding="utf-8") as file:
                for example in examples:
                    file.write(example.to_json() + "\n")
        elif format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(
                    [example.to_dict() for example in examples],
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            raise ValueError("Unsupported format: %s. Use 'jsonl' or 'json'" % format)

        logger.info("Training data saved successfully to %s", output_path)

    def get_generation_statistics(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Get statistics about the generated training examples."""
        stats: Dict[str, Any] = {
            "total_examples": len(examples),
            "detectors": set(),
            "canonical_labels": set(),
            "example_types": {},
            "detector_distribution": {},
            "label_distribution": {},
            "confidence_stats": {"min": float("inf"), "max": float("-inf"), "avg": 0.0},
        }

        confidences: List[float] = []

        for example in examples:
            metadata = example.metadata
            detector = metadata.get("detector", "unknown")
            example_type = metadata.get("example_type", "unknown")

            stats["detectors"].add(detector)
            stats["detector_distribution"][detector] = (
                stats["detector_distribution"].get(detector, 0) + 1
            )
            stats["example_types"][example_type] = (
                stats["example_types"].get(example_type, 0) + 1
            )

            try:
                response_data = json.loads(example.response)
                taxonomy_labels = response_data.get("taxonomy", [])
                confidence = response_data.get("confidence", 0.0)

                for label in taxonomy_labels:
                    stats["canonical_labels"].add(label)
                    stats["label_distribution"][label] = (
                        stats["label_distribution"].get(label, 0) + 1
                    )

                confidences.append(confidence)

            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to parse response for statistics: %s", exc)

        if confidences:
            stats["confidence_stats"]["min"] = min(confidences)
            stats["confidence_stats"]["max"] = max(confidences)
            stats["confidence_stats"]["avg"] = sum(confidences) / len(confidences)

        stats["detectors"] = sorted(list(stats["detectors"]))
        stats["canonical_labels"] = sorted(list(stats["canonical_labels"]))

        return stats

    def validate_training_examples(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Validate training examples for quality and consistency."""
        if not self._taxonomy:
            self._taxonomy = self.taxonomy_loader.load_taxonomy()
        assert self._taxonomy is not None

        validation_report: Dict[str, Any] = {
            "total_examples": len(examples),
            "valid_examples": 0,
            "errors": [],
            "warnings": [],
            "schema_validation": {"valid": 0, "invalid": 0, "errors": []},
        }

        for index, example in enumerate(examples):
            try:
                response_data = json.loads(example.response)

                required_fields = ["taxonomy", "scores", "confidence"]
                missing_fields = [
                    field for field in required_fields if field not in response_data
                ]

                if missing_fields:
                    validation_report["errors"].append(
                        "Example %s: Missing required fields: %s"
                        % (index, missing_fields)
                    )
                    continue

                taxonomy_labels = response_data["taxonomy"]
                for label in taxonomy_labels:
                    if not self._taxonomy.validate_label_name(label):
                        validation_report["errors"].append(
                            "Example %s: Invalid taxonomy label: %s" % (index, label)
                        )

                scores = response_data["scores"]
                for label, score in scores.items():
                    if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                        validation_report["errors"].append(
                            "Example %s: Invalid score for %s: %s"
                            % (index, label, score)
                        )

                confidence = response_data["confidence"]
                if not isinstance(confidence, (int, float)) or not (
                    0 <= confidence <= 1
                ):
                    validation_report["errors"].append(
                        "Example %s: Invalid confidence: %s" % (index, confidence)
                    )

                if set(taxonomy_labels) != set(scores.keys()):
                    validation_report["warnings"].append(
                        "Example %s: Mismatch between taxonomy labels and scores"
                        % index
                    )

                validation_report["valid_examples"] += 1
                validation_report["schema_validation"]["valid"] += 1

            except json.JSONDecodeError as exc:
                validation_report["errors"].append(
                    "Example %s: Invalid JSON in response: %s" % (index, exc)
                )
                validation_report["schema_validation"]["invalid"] += 1
                validation_report["schema_validation"]["errors"].append(str(exc))
            except Exception as exc:  # pragma: no cover - defensive logging
                validation_report["errors"].append(
                    "Example %s: Validation error: %s" % (index, exc)
                )

        return validation_report


__all__ = ["TrainingDataGenerator"]
