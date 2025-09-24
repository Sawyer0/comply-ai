#!/usr/bin/env python3
"""
Validate the new recommended datasets for dual-model training.

This script uses our existing DatasetValidator to test the new recommended datasets
from Hugging Face and validates their compatibility with our training pipeline.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from llama_mapper.data.detectors import DetectorConfigLoader
from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.training.data_generator import DatasetValidator
from llama_mapper.training.data_generator.models import TrainingExample

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecommendedDatasetValidator:
    """Validates new recommended datasets for dual-model training."""

    def __init__(self):
        """Initialize validator with dependencies."""
        self.taxonomy_loader = TaxonomyLoader(
            taxonomy_path=".kiro/pillars-detectors/taxonomy.yaml"
        )
        self.detector_loader = DetectorConfigLoader(".kiro/pillars-detectors")
        self.validator = DatasetValidator(
            taxonomy_loader=self.taxonomy_loader,
            detector_loader=self.detector_loader,
            schema_path=".kiro/pillars-detectors/schema.json",
        )

        # Load dependencies
        self.validator.load_dependencies()

        # Recommended datasets to validate
        self.recommended_datasets = {
            "pii_enhanced": {
                "source": "ai4privacy/pii-masking-43k",
                "description": "Enhanced PII detection with 43k examples",
                "use_case": "mapper_training",
                "expected_size": 43000,
            },
            "legal_bench": {
                "source": "nguha/legalbench",
                "description": "Legal reasoning tasks for compliance analysis",
                "use_case": "analyst_training",
                "expected_size": 1000,
            },
            "gdpr_complete": {
                "source": "AndreaSimeri/GDPR",
                "description": "Complete GDPR regulation text",
                "use_case": "analyst_training",
                "expected_size": 1000,
            },
            "policy_qa": {
                "source": "qa4pc/QA4PC",
                "description": "Policy compliance Q&A",
                "use_case": "analyst_training",
                "expected_size": 2000,
            },
            "security_attacks": {
                "source": "ibm-research/AttaQ",
                "description": "Attack pattern detection",
                "use_case": "mapper_training",
                "expected_size": 10000,
            },
            "content_moderation": {
                "source": "allenai/wildguardmix",
                "description": "Content toxicity detection",
                "use_case": "mapper_training",
                "expected_size": 5000,
            },
        }

    def load_and_validate_dataset(
        self, dataset_name: str, dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load and validate a specific recommended dataset."""
        logger.info(
            "Loading and validating %s: %s", dataset_name, dataset_info["description"]
        )

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_info["source"], split="train")
            logger.info(
                "Loaded %s examples from %s", len(dataset), dataset_info["source"]
            )

            # Convert to our training format
            training_examples = self._convert_to_training_format(
                dataset, dataset_name, dataset_info
            )
            logger.info("Converted to %s training examples", len(training_examples))

            # Validate the dataset
            validation_report = self.validator.validate_training_dataset(
                training_examples
            )

            # Add dataset-specific metadata
            validation_report["dataset_metadata"] = {
                "name": dataset_name,
                "source": dataset_info["source"],
                "description": dataset_info["description"],
                "use_case": dataset_info["use_case"],
                "expected_size": dataset_info["expected_size"],
                "actual_size": len(training_examples),
                "conversion_success": True,
            }

            return validation_report

        except Exception as e:
            logger.error("Failed to load/validate %s: %s", dataset_name, e)
            return {
                "dataset_metadata": {
                    "name": dataset_name,
                    "source": dataset_info["source"],
                    "description": dataset_info["description"],
                    "use_case": dataset_info["use_case"],
                    "expected_size": dataset_info["expected_size"],
                    "actual_size": 0,
                    "conversion_success": False,
                    "error": str(e),
                },
                "validation_error": str(e),
            }

    def _convert_to_training_format(
        self, dataset, dataset_name: str, dataset_info: Dict[str, Any]
    ) -> List[TrainingExample]:
        """Convert Hugging Face dataset to our training format."""
        training_examples = []

        for i, example in enumerate(dataset):
            try:
                # Convert based on dataset type
                if dataset_name == "pii_enhanced":
                    training_example = self._convert_pii_example(example, i)
                elif dataset_name == "legal_bench":
                    training_example = self._convert_legal_example(example, i)
                elif dataset_name == "gdpr_complete":
                    training_example = self._convert_gdpr_example(example, i)
                elif dataset_name == "policy_qa":
                    training_example = self._convert_policy_qa_example(example, i)
                elif dataset_name == "security_attacks":
                    training_example = self._convert_security_example(example, i)
                elif dataset_name == "content_moderation":
                    training_example = self._convert_content_example(example, i)
                else:
                    training_example = self._convert_generic_example(
                        example, i, dataset_name
                    )

                if training_example:
                    training_examples.append(training_example)

            except Exception as e:
                logger.warning(
                    "Failed to convert example %s from %s: %s", i, dataset_name, e
                )
                continue

        return training_examples

    def _convert_pii_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert PII detection example to training format."""
        text = example.get("text", "")
        labels = example.get("labels", [])

        # Create instruction
        instruction = f"Map this PII detection to compliance taxonomy: {text[:200]}..."

        # Create response with PII taxonomy
        taxonomy = []
        scores = {}

        for label in labels:
            if "email" in str(label).lower():
                taxonomy.append("PII.Contact.Email")
                scores["PII.Contact.Email"] = 0.95
            elif "phone" in str(label).lower():
                taxonomy.append("PII.Contact.Phone")
                scores["PII.Contact.Phone"] = 0.90
            elif "ssn" in str(label).lower() or "social" in str(label).lower():
                taxonomy.append("PII.Identifier.SSN")
                scores["PII.Identifier.SSN"] = 0.95
            elif "credit" in str(label).lower() or "card" in str(label).lower():
                taxonomy.append("PII.Identifier.CreditCard")
                scores["PII.Identifier.CreditCard"] = 0.90
            else:
                taxonomy.append("PII.Other")
                scores["PII.Other"] = 0.85

        if not taxonomy:
            taxonomy = ["PII.Other"]
            scores = {"PII.Other": 0.80}

        response = json.dumps(
            {
                "taxonomy": taxonomy,
                "scores": scores,
                "confidence": 0.85,
                "provenance": {
                    "detector": "regex-pii",
                    "source": "ai4privacy/pii-masking-43k",
                    "example_index": index,
                },
                "notes": "PII detection from enhanced dataset",
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "ai4privacy/pii-masking-43k",
                "type": "pii_detection",
                "example_index": index,
                "use_case": "mapper_training",
            },
        )

    def _convert_legal_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert legal reasoning example to training format."""
        question = example.get("question", "")
        answer = example.get("answer", "")

        instruction = f"Analyze this legal compliance scenario: {question}"

        response = json.dumps(
            {
                "risk_assessment": {
                    "severity": "MEDIUM",
                    "compliance_frameworks": ["GDPR", "CCPA"],
                    "affected_articles": ["GDPR-6", "GDPR-32"],
                },
                "remediation_steps": [
                    "1. Review legal basis for processing",
                    "2. Implement appropriate safeguards",
                    "3. Ensure data subject rights",
                ],
                "audit_evidence": {
                    "required_documentation": ["Legal Analysis", "Risk Assessment"],
                    "compliance_gaps": ["Legal basis documentation"],
                },
                "confidence": 0.88,
                "provenance": {"source": "nguha/legalbench", "example_index": index},
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "nguha/legalbench",
                "type": "legal_reasoning",
                "example_index": index,
                "use_case": "analyst_training",
            },
        )

    def _convert_gdpr_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert GDPR example to training format."""
        text = example.get("text", "")

        instruction = (
            f"Provide compliance analysis for this GDPR requirement: {text[:200]}..."
        )

        response = json.dumps(
            {
                "risk_assessment": {
                    "severity": "HIGH",
                    "compliance_frameworks": ["GDPR"],
                    "affected_articles": ["GDPR-5", "GDPR-6", "GDPR-32"],
                },
                "remediation_steps": [
                    "1. Implement data minimization principles",
                    "2. Establish lawful basis for processing",
                    "3. Implement appropriate technical measures",
                ],
                "confidence": 0.92,
                "provenance": {"source": "AndreaSimeri/GDPR", "example_index": index},
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "AndreaSimeri/GDPR",
                "type": "gdpr_analysis",
                "example_index": index,
                "use_case": "analyst_training",
            },
        )

    def _convert_policy_qa_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert policy Q&A example to training format."""
        question = example.get("question", "")
        answer = example.get("answer", "")

        instruction = f"Answer this compliance policy question: {question}"

        response = json.dumps(
            {
                "compliance_answer": answer,
                "risk_assessment": {
                    "severity": "MEDIUM",
                    "compliance_frameworks": ["SOC-2", "ISO-27001"],
                },
                "confidence": 0.85,
                "provenance": {"source": "qa4pc/QA4PC", "example_index": index},
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "qa4pc/QA4PC",
                "type": "policy_qa",
                "example_index": index,
                "use_case": "analyst_training",
            },
        )

    def _convert_security_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert security attack example to training format."""
        text = example.get("text", "")
        label = example.get("label", "")

        instruction = (
            f"Map this security attack to compliance taxonomy: {text[:200]}..."
        )

        # Map security attacks to compliance taxonomy
        taxonomy = []
        scores = {}

        if "injection" in str(label).lower():
            taxonomy = ["PROMPT_INJECTION.Other"]
            scores = {"PROMPT_INJECTION.Other": 0.90}
        elif "violence" in str(label).lower():
            taxonomy = ["HARM.VIOLENCE.Physical"]
            scores = {"HARM.VIOLENCE.Physical": 0.85}
        elif "hate" in str(label).lower():
            taxonomy = ["HARM.SPEECH.Hate.Other"]
            scores = {"HARM.SPEECH.Hate.Other": 0.88}
        else:
            taxonomy = ["HARM.VIOLENCE.Other"]
            scores = {"HARM.VIOLENCE.Other": 0.80}

        response = json.dumps(
            {
                "taxonomy": taxonomy,
                "scores": scores,
                "confidence": 0.85,
                "provenance": {
                    "detector": "llama-guard",
                    "source": "ibm-research/AttaQ",
                    "example_index": index,
                },
                "notes": "Security attack pattern detection",
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "ibm-research/AttaQ",
                "type": "security_attack",
                "example_index": index,
                "use_case": "mapper_training",
            },
        )

    def _convert_content_example(
        self, example: Dict[str, Any], index: int
    ) -> TrainingExample:
        """Convert content moderation example to training format."""
        text = example.get("text", "")
        labels = example.get("labels", [])

        instruction = f"Map this content to compliance taxonomy: {text[:200]}..."

        taxonomy = []
        scores = {}

        for label in labels:
            if "toxicity" in str(label).lower():
                taxonomy.append("HARM.SPEECH.Toxicity")
                scores["HARM.SPEECH.Toxicity"] = 0.90
            elif "hate" in str(label).lower():
                taxonomy.append("HARM.SPEECH.Hate.Other")
                scores["HARM.SPEECH.Hate.Other"] = 0.85
            elif "violence" in str(label).lower():
                taxonomy.append("HARM.VIOLENCE.Physical")
                scores["HARM.VIOLENCE.Physical"] = 0.88

        if not taxonomy:
            taxonomy = ["HARM.SPEECH.Toxicity"]
            scores = {"HARM.SPEECH.Toxicity": 0.80}

        response = json.dumps(
            {
                "taxonomy": taxonomy,
                "scores": scores,
                "confidence": 0.85,
                "provenance": {
                    "detector": "deberta-toxicity",
                    "source": "allenai/wildguardmix",
                    "example_index": index,
                },
                "notes": "Content moderation detection",
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": "allenai/wildguardmix",
                "type": "content_moderation",
                "example_index": index,
                "use_case": "mapper_training",
            },
        )

    def _convert_generic_example(
        self, example: Dict[str, Any], index: int, dataset_name: str
    ) -> TrainingExample:
        """Convert generic example to training format."""
        text = str(example.get("text", ""))

        instruction = f"Analyze this compliance scenario: {text[:200]}..."

        response = json.dumps(
            {
                "taxonomy": ["OTHER.Unknown"],
                "scores": {"OTHER.Unknown": 0.70},
                "confidence": 0.70,
                "provenance": {"source": dataset_name, "example_index": index},
                "notes": "Generic compliance analysis",
            }
        )

        return TrainingExample(
            instruction=instruction,
            response=response,
            metadata={
                "source": dataset_name,
                "type": "generic",
                "example_index": index,
                "use_case": "mapper_training",
            },
        )

    def validate_all_recommended_datasets(self) -> Dict[str, Any]:
        """Validate all recommended datasets."""
        logger.info("Starting validation of all recommended datasets...")

        validation_results = {}
        summary_stats = {
            "total_datasets": len(self.recommended_datasets),
            "successful_validations": 0,
            "failed_validations": 0,
            "total_examples": 0,
            "mapper_examples": 0,
            "analyst_examples": 0,
        }

        for dataset_name, dataset_info in self.recommended_datasets.items():
            logger.info("Validating %s...", dataset_name)

            validation_report = self.load_and_validate_dataset(
                dataset_name, dataset_info
            )
            validation_results[dataset_name] = validation_report

            # Update summary stats
            if validation_report.get("dataset_metadata", {}).get(
                "conversion_success", False
            ):
                summary_stats["successful_validations"] += 1
                total_examples = validation_report.get("dataset_metadata", {}).get(
                    "actual_size", 0
                )
                summary_stats["total_examples"] += total_examples

                use_case = validation_report.get("dataset_metadata", {}).get(
                    "use_case", ""
                )
                if use_case == "mapper_training":
                    summary_stats["mapper_examples"] += total_examples
                elif use_case == "analyst_training":
                    summary_stats["analyst_examples"] += total_examples
            else:
                summary_stats["failed_validations"] += 1

        # Add summary to results
        validation_results["summary"] = summary_stats

        return validation_results

    def export_validation_results(self, results: Dict[str, Any], output_path: str):
        """Export validation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Validation results exported to %s", output_path)


def main():
    """Main entry point for dataset validation."""
    logger.info("Starting validation of recommended datasets...")

    try:
        # Initialize validator
        validator = RecommendedDatasetValidator()

        # Validate all datasets
        results = validator.validate_all_recommended_datasets()

        # Print summary
        summary = results["summary"]
        logger.info("Validation Summary:")
        logger.info("  Total datasets: %s", summary["total_datasets"])
        logger.info("  Successful validations: %s", summary["successful_validations"])
        logger.info("  Failed validations: %s", summary["failed_validations"])
        logger.info("  Total examples: %s", summary["total_examples"])
        logger.info("  Mapper training examples: %s", summary["mapper_examples"])
        logger.info("  Analyst training examples: %s", summary["analyst_examples"])

        # Export results
        validator.export_validation_results(
            results, "llm/llm-reports/recommended_datasets_validation.json"
        )

        # Print detailed results for each dataset
        logger.info("\nDetailed Results:")
        for dataset_name, report in results.items():
            if dataset_name == "summary":
                continue

            metadata = report.get("dataset_metadata", {})
            if metadata.get("conversion_success", False):
                overall_score = report.get("overall_score", 0)
                logger.info(
                    "  %s: ✅ %.1f/100 score, %s examples",
                    dataset_name,
                    overall_score,
                    metadata["actual_size"],
                )
            else:
                error = metadata.get("error", "Unknown error")
                logger.info("  %s: ❌ Failed - %s", dataset_name, error)

        logger.info("Dataset validation completed successfully!")
        return 0

    except Exception as e:
        logger.error("Dataset validation failed: %s", e)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
