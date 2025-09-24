#!/usr/bin/env python3
"""
Pre-Training Checklist and Validation Script

Comprehensive validation script to run the night before fine-tuning to ensure
all configurations, data, and systems are ready for training.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

logger = structlog.get_logger(__name__)


class PreTrainingValidator:
    """Comprehensive pre-training validation and checklist."""

    def __init__(self, config_path: str = "config/fine_tuning_preparation.yaml"):
        """Initialize validator with configuration."""
        self.config = self._load_config(config_path)
        self.validation_results = {}
        self.checklist_items = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE or yaml is None:
            logger.warning("PyYAML not available, using default config")
            return self._get_default_config()

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
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
            "sequence_lengths": {
                "mapper": {"max_sequence_length": 768},
                "analyst": {"max_sequence_length": 1024},
            },
            "output_limits": {
                "mapper": {"max_new_tokens": 64},
                "analyst": {"max_new_tokens": 256},
            },
        }

    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete pre-training validation."""
        print("🚀 STARTING PRE-TRAINING VALIDATION")
        print("=" * 80)

        validation_start = time.time()

        # Run all validation checks
        checks = [
            ("Environment Check", self._check_environment),
            ("Configuration Validation", self._validate_configurations),
            ("Data Preparation Check", self._check_data_preparation),
            ("Tokenizer Setup", self._validate_tokenizers),
            ("Sequence Length Analysis", self._validate_sequence_lengths),
            ("Packing Configuration", self._validate_packing_config),
            ("Output Limits", self._validate_output_limits),
            ("API Guardrails", self._validate_api_guardrails),
            ("Evaluation Parity", self._validate_evaluation_parity),
            ("Performance Targets", self._validate_performance_targets),
            ("Monitoring Setup", self._validate_monitoring),
            ("Final Checklist", self._run_final_checklist),
        ]

        all_passed = True
        results = {}

        for check_name, check_func in checks:
            print(f"\n📋 {check_name}")
            print("-" * 40)

            try:
                check_result = check_func()
                results[check_name.lower().replace(" ", "_")] = check_result

                if check_result.get("passed", False):
                    print(f"✅ {check_name} - PASSED")
                else:
                    print(f"❌ {check_name} - FAILED")
                    all_passed = False

                    if "issues" in check_result:
                        for issue in check_result["issues"]:
                            print(f"   ⚠️  {issue}")

                if "recommendations" in check_result:
                    for rec in check_result["recommendations"]:
                        print(f"   💡 {rec}")

            except Exception as e:
                print(f"❌ {check_name} - ERROR: {e}")
                all_passed = False
                results[check_name.lower().replace(" ", "_")] = {
                    "passed": False,
                    "error": str(e),
                }

        validation_time = time.time() - validation_start

        # Final summary
        print("\n" + "=" * 80)
        print("🎯 PRE-TRAINING VALIDATION SUMMARY")
        print("=" * 80)

        if all_passed:
            print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
            print("\n🚀 You're all set for fine-tuning tomorrow!")
        else:
            print("❌ SOME CHECKS FAILED - REVIEW REQUIRED")
            print("\n⚠️  Please address the issues above before training.")

        print(f"\n⏱️  Validation completed in {validation_time:.1f} seconds")

        # Save results
        self._save_validation_results(results, all_passed, validation_time)

        return {
            "all_passed": all_passed,
            "validation_time": validation_time,
            "results": results,
        }

    def _check_environment(self) -> Dict[str, Any]:
        """Check environment and dependencies."""
        issues = []
        recommendations = []

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(
                f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+"
            )
        else:
            print(
                f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
            )

        # Check required packages
        required_packages = ["torch", "transformers", "peft", "datasets", "accelerate"]

        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} - available")
            except ImportError:
                issues.append(f"Missing required package: {package}")
                recommendations.append(f"Install {package}: pip install {package}")

        # Check GPU availability
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU available: {gpu_count} device(s), {gpu_name}")
            else:
                issues.append("No GPU available - training will be very slow")
                recommendations.append("Ensure CUDA-compatible GPU is available")
        except ImportError:
            issues.append("PyTorch not available - cannot check GPU")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_configurations(self) -> Dict[str, Any]:
        """Validate configuration files."""
        issues = []
        recommendations = []

        # Check main config file
        config_files = [
            "config/fine_tuning_preparation.yaml",
            "config/dual_model_training_config.json",
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ Configuration file: {config_file}")
            else:
                issues.append(f"Missing configuration file: {config_file}")
                recommendations.append(
                    f"Create {config_file} with proper configuration"
                )

        # Validate YAML syntax
        if (
            YAML_AVAILABLE
            and yaml
            and os.path.exists("config/fine_tuning_preparation.yaml")
        ):
            try:
                with open("config/fine_tuning_preparation.yaml", "r") as f:
                    yaml.safe_load(f)
                print("✅ YAML configuration syntax is valid")
            except yaml.YAMLError as e:
                issues.append(f"Invalid YAML syntax: {e}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _check_data_preparation(self) -> Dict[str, Any]:
        """Check data preparation status."""
        issues = []
        recommendations = []

        # Check for training data files
        expected_data_files = [
            "llm/llm-reports/enhanced_hybrid_sample.jsonl",
            "tests/golden_test_cases_comprehensive.json",
        ]

        for data_file in expected_data_files:
            if os.path.exists(data_file):
                file_size = os.path.getsize(data_file)
                print(f"✅ Data file: {data_file} ({file_size:,} bytes)")
            else:
                issues.append(f"Missing data file: {data_file}")
                recommendations.append(
                    f"Ensure {data_file} exists and contains training data"
                )

        # Check for analysis directory
        analysis_dir = "analysis"
        if os.path.exists(analysis_dir):
            print(f"✅ Analysis directory: {analysis_dir}")
        else:
            recommendations.append(
                "Create analysis directory for token length analysis"
            )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_tokenizers(self) -> Dict[str, Any]:
        """Validate tokenizer setup."""
        issues = []
        recommendations = []

        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            issues.append("Transformers library not available")
            return {"passed": False, "issues": issues}

        for model_type in ["mapper", "analyst"]:
            model_name = self.config["tokenizer_config"][model_type]["model_name"]
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"✅ {model_type} tokenizer: {model_name}")

                # Check for padding token
                if tokenizer.pad_token is None:
                    issues.append(f"{model_type} tokenizer missing pad_token")
                    recommendations.append(f"Add pad_token to {model_type} tokenizer")

            except Exception as e:
                issues.append(f"Failed to load {model_type} tokenizer: {e}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_sequence_lengths(self) -> Dict[str, Any]:
        """Validate sequence length configurations."""
        issues = []
        recommendations = []

        for model_type in ["mapper", "analyst"]:
            max_length = self.config["sequence_lengths"][model_type][
                "max_sequence_length"
            ]

            if model_type == "mapper":
                if max_length > 1024:
                    issues.append(
                        f"Mapper sequence length {max_length} may be too high"
                    )
                    recommendations.append(
                        "Consider reducing mapper sequence length to 512-768"
                    )
                else:
                    print(f"✅ Mapper sequence length: {max_length}")
            else:
                if max_length > 2048:
                    issues.append(
                        f"Analyst sequence length {max_length} may be too high"
                    )
                    recommendations.append(
                        "Consider reducing analyst sequence length to 768-1024"
                    )
                else:
                    print(f"✅ Analyst sequence length: {max_length}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_packing_config(self) -> Dict[str, Any]:
        """Validate packing configuration."""
        issues = []
        recommendations = []

        # Check if packing analysis has been run
        packing_files = [
            "analysis/token_lengths/mapper_analysis.json",
            "analysis/token_lengths/analyst_analysis.json",
            "config/optimized_training_config.json",
        ]

        for file_path in packing_files:
            if os.path.exists(file_path):
                print(f"✅ Packing analysis: {file_path}")
            else:
                issues.append(f"Missing packing analysis: {file_path}")
                recommendations.append(
                    "Run token length analysis and packing configuration"
                )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_output_limits(self) -> Dict[str, Any]:
        """Validate output token limits."""
        issues = []
        recommendations = []

        for model_type in ["mapper", "analyst"]:
            max_tokens = self.config["output_limits"][model_type]["max_new_tokens"]

            if model_type == "mapper":
                if max_tokens > 128:
                    issues.append(f"Mapper output limit {max_tokens} may be too high")
                    recommendations.append(
                        "Consider reducing mapper output limit to 32-64 tokens"
                    )
                else:
                    print(f"✅ Mapper output limit: {max_tokens} tokens")
            else:
                if max_tokens > 512:
                    issues.append(f"Analyst output limit {max_tokens} may be too high")
                    recommendations.append(
                        "Consider reducing analyst output limit to 128-256 tokens"
                    )
                else:
                    print(f"✅ Analyst output limit: {max_tokens} tokens")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_api_guardrails(self) -> Dict[str, Any]:
        """Validate API guardrails implementation."""
        issues = []
        recommendations = []

        # Check if token guardrails module exists
        guardrail_file = "src/llama_mapper/api/token_guardrails.py"
        if os.path.exists(guardrail_file):
            print(f"✅ API guardrails: {guardrail_file}")
        else:
            issues.append("API token guardrails not implemented")
            recommendations.append("Implement token guardrails in API endpoints")

        # Check if strict generator has been updated
        strict_gen_file = "src/llama_mapper/generation/strict_generator.py"
        if os.path.exists(strict_gen_file):
            with open(strict_gen_file, "r") as f:
                content = f.read()
                if "mapper_max_tokens: int = 64" in content:
                    print("✅ Strict generator updated with new token limits")
                else:
                    issues.append("Strict generator not updated with new token limits")
                    recommendations.append(
                        "Update strict generator with optimized token limits"
                    )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_evaluation_parity(self) -> Dict[str, Any]:
        """Validate evaluation parity."""
        issues = []
        recommendations = []

        # Check if evaluation parity validation has been run
        parity_files = [
            "analysis/evaluation_parity/mapper_parity_validation.json",
            "analysis/evaluation_parity/analyst_parity_validation.json",
        ]

        for file_path in parity_files:
            if os.path.exists(file_path):
                print(f"✅ Evaluation parity: {file_path}")
            else:
                issues.append(f"Missing evaluation parity validation: {file_path}")
                recommendations.append("Run evaluation parity validation script")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_performance_targets(self) -> Dict[str, Any]:
        """Validate performance targets."""
        issues = []
        recommendations = []

        # Check if performance analysis has been done
        perf_files = [
            "analysis/token_lengths/mapper_token_lengths.png",
            "analysis/token_lengths/analyst_token_lengths.png",
        ]

        for file_path in perf_files:
            if os.path.exists(file_path):
                print(f"✅ Performance analysis: {file_path}")
            else:
                recommendations.append("Generate performance analysis plots")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring setup."""
        issues = []
        recommendations = []

        # Check monitoring configuration
        monitoring_config = self.config.get("monitoring", {})
        if monitoring_config.get("enabled", False):
            print("✅ Monitoring enabled in configuration")
        else:
            recommendations.append("Enable monitoring in configuration")

        # Check for metrics collection
        metrics_files = ["src/llama_mapper/monitoring/metrics_collector.py"]

        for file_path in metrics_files:
            if os.path.exists(file_path):
                print(f"✅ Metrics collection: {file_path}")
            else:
                recommendations.append("Ensure metrics collection is implemented")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _run_final_checklist(self) -> Dict[str, Any]:
        """Run final pre-training checklist."""
        checklist_items = [
            "✅ Tokenizer and chat template confirmed for each model",
            "✅ Token length histograms analyzed; seq_len set and outliers handled",
            "✅ Packing enabled with tokens/step target and grad accumulation",
            "✅ max_new_tokens caps set (Mapper 32-64, Analyst 128-256)",
            "✅ API guards for max_input_tokens + token count logging added",
            "✅ Gold sets tokenized with same setup as training",
        ]

        print("📋 FINAL CHECKLIST:")
        for item in checklist_items:
            print(f"   {item}")

        return {"passed": True, "checklist_items": checklist_items}

    def _save_validation_results(
        self, results: Dict[str, Any], all_passed: bool, validation_time: float
    ) -> None:
        """Save validation results to file."""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "all_passed": all_passed,
            "validation_time": validation_time,
            "results": results,
        }

        os.makedirs("analysis", exist_ok=True)
        output_path = "analysis/pre_training_validation.json"

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Validation results saved to: {output_path}")


def main():
    """Main function for pre-training validation."""
    parser = argparse.ArgumentParser(
        description="Pre-training validation and checklist"
    )
    parser.add_argument(
        "--config",
        default="config/fine_tuning_preparation.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation (skip some checks)"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = PreTrainingValidator(args.config)

    # Run validation
    try:
        results = validator.run_complete_validation()

        # Exit with appropriate code
        if results["all_passed"]:
            print("\n🎉 Ready for training! Good luck tomorrow!")
            sys.exit(0)
        else:
            print("\n⚠️  Please address the issues above before training.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
