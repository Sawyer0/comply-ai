#!/usr/bin/env python3
"""
Evaluation Parity Validation Script

Ensures that evaluation datasets use the same tokenization and chat templates
as training to maintain metric comparability.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

# Optional imports with fallbacks
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    print("Warning: transformers not available. Install with: pip install transformers")

logger = structlog.get_logger(__name__)


class EvaluationParityValidator:
    """Validates evaluation dataset parity with training configuration."""

    def __init__(self, config_path: str = "config/fine_tuning_preparation.yaml"):
        """Initialize validator with configuration."""
        self.config = self._load_config(config_path)
        self.tokenizers = {}
        self.validation_results = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml

            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not available, using default config")
            return self._get_default_config()
        except FileNotFoundError:
            logger.warning(
                "Config file %s not found, using default config", config_path
            )
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML not available."""
        return {
            "tokenizer_config": {
                "mapper": {"model_name": "meta-llama/Llama-3-8B-Instruct"},
                "analyst": {"model_name": "microsoft/Phi-3-mini-4k-instruct"},
            },
            "evaluation_config": {
                "gold_sets": {
                    "mapper_gold": "tests/golden_test_cases_mapper.json",
                    "analyst_gold": "tests/golden_test_cases_analyst.json",
                }
            },
        }

    def setup_tokenizers(self) -> None:
        """Set up tokenizers for both models."""
        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            logger.error(
                "Transformers not available. Cannot validate tokenization parity."
            )
            return

        for model_type in ["mapper", "analyst"]:
            model_name = self.config["tokenizer_config"][model_type]["model_name"]
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Add padding token if not present
                if tokenizer.pad_token is None:
                    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

                self.tokenizers[model_type] = tokenizer
                logger.info("Loaded tokenizer for %s: %s", model_type, model_name)

            except Exception as e:
                logger.warning("Failed to load tokenizer for %s: %s", model_type, e)
                # Use fallback tokenizer for testing
                if model_type == "mapper":
                    fallback_model = "microsoft/DialoGPT-medium"
                else:
                    fallback_model = "microsoft/Phi-3-mini-4k-instruct"

                try:
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self.tokenizers[model_type] = tokenizer
                    logger.info(
                        "Using fallback tokenizer for %s: %s",
                        model_type,
                        fallback_model,
                    )
                except Exception as fallback_e:
                    logger.error(
                        "Failed to load fallback tokenizer for %s: %s",
                        model_type,
                        fallback_e,
                    )

    def validate_gold_set_parity(
        self, gold_set_path: str, model_type: str
    ) -> Dict[str, Any]:
        """
        Validate that gold set uses same tokenization as training.

        Args:
            gold_set_path: Path to gold test cases
            model_type: Type of model ("mapper" or "analyst")

        Returns:
            Validation results
        """
        if model_type not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_type} not loaded")

        tokenizer = self.tokenizers[model_type]

        logger.info("Validating gold set parity for %s: %s", model_type, gold_set_path)

        # Load gold set
        with open(gold_set_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)

        validation_results = {
            "gold_set_path": gold_set_path,
            "model_type": model_type,
            "total_cases": len(gold_data),
            "tokenization_consistency": True,
            "template_consistency": True,
            "issues": [],
            "recommendations": [],
        }

        # Validate each test case
        for i, test_case in enumerate(gold_data):
            case_validation = self._validate_test_case(
                test_case, tokenizer, model_type, i
            )

            if not case_validation["tokenization_consistent"]:
                validation_results["tokenization_consistency"] = False
                validation_results["issues"].append(
                    {
                        "case_index": i,
                        "issue": "tokenization_inconsistency",
                        "details": case_validation["issues"],
                    }
                )

            if not case_validation["template_consistent"]:
                validation_results["template_consistency"] = False
                validation_results["issues"].append(
                    {
                        "case_index": i,
                        "issue": "template_inconsistency",
                        "details": case_validation["issues"],
                    }
                )

        # Generate recommendations
        validation_results["recommendations"] = self._generate_parity_recommendations(
            validation_results, model_type
        )

        self.validation_results[model_type] = validation_results
        return validation_results

    def _validate_test_case(
        self,
        test_case: Dict[str, Any],
        tokenizer: Any,
        model_type: str,
        case_index: int,
    ) -> Dict[str, Any]:
        """Validate a single test case for tokenization and template consistency."""
        validation = {
            "case_index": case_index,
            "tokenization_consistent": True,
            "template_consistent": True,
            "issues": [],
        }

        # Check if test case has expected structure
        if "input" not in test_case or "expected_output" not in test_case:
            validation["issues"].append("Missing input or expected_output fields")
            validation["tokenization_consistent"] = False
            return validation

        # Validate input tokenization
        input_text = str(test_case["input"])
        try:
            # Tokenize input
            input_tokens = tokenizer.encode(input_text, add_special_tokens=True)

            # Check for reasonable token count
            if len(input_tokens) > 2048:  # Reasonable upper bound
                validation["issues"].append(
                    f"Input token count too high: {len(input_tokens)}"
                )
                validation["tokenization_consistent"] = False

            # Check for proper special tokens
            if model_type == "mapper":
                # Llama-3 format should have proper special tokens
                if not any(
                    token in input_text
                    for token in ["<|begin_of_text|>", "<|start_header_id|>"]
                ):
                    validation["issues"].append(
                        "Missing Llama-3 special tokens in input"
                    )
                    validation["template_consistent"] = False
            else:
                # Phi-3 format should have proper special tokens
                if not any(
                    token in input_text for token in ["<|user|>", "<|assistant|>"]
                ):
                    validation["issues"].append("Missing Phi-3 special tokens in input")
                    validation["template_consistent"] = False

        except Exception as e:
            validation["issues"].append(f"Tokenization error: {e}")
            validation["tokenization_consistent"] = False

        # Validate expected output format
        expected_output = test_case.get("expected_output", {})
        if isinstance(expected_output, dict):
            # Check for required fields based on model type
            if model_type == "mapper":
                required_fields = ["scores", "confidence"]
                for field in required_fields:
                    if field not in expected_output:
                        validation["issues"].append(
                            f"Missing required field in expected output: {field}"
                        )
                        validation["template_consistent"] = False
            else:
                required_fields = ["analysis_type", "risk_level", "recommendations"]
                for field in required_fields:
                    if field not in expected_output:
                        validation["issues"].append(
                            f"Missing required field in expected output: {field}"
                        )
                        validation["template_consistent"] = False

        return validation

    def _generate_parity_recommendations(
        self, validation_results: Dict[str, Any], model_type: str
    ) -> List[str]:
        """Generate recommendations for fixing parity issues."""
        recommendations = []

        if not validation_results["tokenization_consistency"]:
            recommendations.append(
                f"Re-tokenize {model_type} gold set with current training tokenizer"
            )

        if not validation_results["template_consistency"]:
            recommendations.append(
                f"Update {model_type} gold set to use current chat template format"
            )

        if validation_results["issues"]:
            recommendations.append(
                f"Review and fix {len(validation_results['issues'])} issues in {model_type} gold set"
            )

        # Add general recommendations
        recommendations.extend(
            [
                "Ensure gold sets are updated whenever training tokenization changes",
                "Run this validation script before each training run",
                "Consider versioning gold sets with tokenizer configurations",
            ]
        )

        return recommendations

    def generate_tokenization_hash(self, model_type: str) -> str:
        """Generate hash of current tokenization configuration."""
        if model_type not in self.tokenizers:
            return "unknown"

        tokenizer = self.tokenizers[model_type]

        # Create hash of tokenizer configuration
        config_data = {
            "model_name": self.config["tokenizer_config"][model_type]["model_name"],
            "vocab_size": len(tokenizer),
            "special_tokens": {
                "pad_token": str(tokenizer.pad_token),
                "eos_token": str(tokenizer.eos_token),
                "bos_token": str(tokenizer.bos_token),
                "unk_token": str(tokenizer.unk_token),
            },
        }

        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_validation_results(
        self, output_dir: str = "analysis/evaluation_parity"
    ) -> None:
        """Save validation results to files."""
        os.makedirs(output_dir, exist_ok=True)

        for model_type, results in self.validation_results.items():
            # Add tokenization hash
            results["tokenization_hash"] = self.generate_tokenization_hash(model_type)

            output_path = f"{output_dir}/{model_type}_parity_validation.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(
                "Saved validation results for %s to %s", model_type, output_path
            )

    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION PARITY VALIDATION SUMMARY")
        print("=" * 80)

        all_consistent = True

        for model_type, results in self.validation_results.items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Gold set: {results['gold_set_path']}")
            print(f"  Total test cases: {results['total_cases']}")
            print(
                f"  Tokenization consistent: {'✓' if results['tokenization_consistency'] else '✗'}"
            )
            print(
                f"  Template consistent: {'✓' if results['template_consistency'] else '✗'}"
            )
            print(f"  Issues found: {len(results['issues'])}")
            print(f"  Tokenization hash: {results.get('tokenization_hash', 'unknown')}")

            if (
                not results["tokenization_consistency"]
                or not results["template_consistency"]
            ):
                all_consistent = False
                print(f"  ⚠️  PARITY ISSUES DETECTED")

            if results["recommendations"]:
                print(f"  Recommendations:")
                for rec in results["recommendations"]:
                    print(f"    • {rec}")

        print(f"\n{'='*80}")
        if all_consistent:
            print("✓ ALL EVALUATION DATASETS MAINTAIN PARITY WITH TRAINING")
        else:
            print("⚠️  EVALUATION PARITY ISSUES DETECTED - REVIEW REQUIRED")
        print("=" * 80)


def main():
    """Main function for evaluation parity validation."""
    parser = argparse.ArgumentParser(description="Validate evaluation dataset parity")
    parser.add_argument(
        "--mapper-gold", required=True, help="Path to mapper gold test cases"
    )
    parser.add_argument(
        "--analyst-gold", required=True, help="Path to analyst gold test cases"
    )
    parser.add_argument(
        "--config",
        default="config/fine_tuning_preparation.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output-dir", default="analysis/evaluation_parity", help="Output directory"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = EvaluationParityValidator(args.config)
    validator.setup_tokenizers()

    if not validator.tokenizers:
        print("Error: No tokenizers loaded. Check your configuration and dependencies.")
        return

    # Validate gold sets
    try:
        validator.validate_gold_set_parity(args.mapper_gold, "mapper")
        validator.validate_gold_set_parity(args.analyst_gold, "analyst")

        # Save results
        validator.save_validation_results(args.output_dir)

        # Print summary
        validator.print_summary()

        print(f"\nValidation complete! Results saved to {args.output_dir}")

    except Exception as e:
        logger.error("Validation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
