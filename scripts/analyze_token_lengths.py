#!/usr/bin/env python3
"""
Token Length Analysis Script for Fine-Tuning Preparation

Analyzes training data token lengths to determine optimal sequence lengths,
identifies outliers, and generates packing recommendations.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    print("Warning: transformers not available. Install with: pip install transformers")

logger = structlog.get_logger(__name__)


class TokenLengthAnalyzer:
    """Analyzes token lengths in training datasets for fine-tuning optimization."""

    def __init__(self, config_path: str = "config/fine_tuning_preparation.yaml"):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.tokenizers = {}
        self.results = {}

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
            "sequence_lengths": {
                "mapper": {"max_sequence_length": 768, "target_95th_percentile": 512},
                "analyst": {"max_sequence_length": 1024, "target_95th_percentile": 768},
            },
            "tokenizer_config": {
                "mapper": {"model_name": "meta-llama/Llama-3-8B-Instruct"},
                "analyst": {"model_name": "microsoft/Phi-3-mini-4k-instruct"},
            },
        }

    def setup_tokenizers(self) -> None:
        """Set up tokenizers for both models."""
        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            logger.error("Transformers not available. Cannot analyze token lengths.")
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

    def analyze_dataset(self, dataset_path: str, model_type: str) -> Dict[str, Any]:
        """
        Analyze token lengths in a dataset.

        Args:
            dataset_path: Path to dataset file (JSONL format)
            model_type: Type of model ("mapper" or "analyst")

        Returns:
            Analysis results including percentiles and recommendations
        """
        if model_type not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_type} not loaded")

        tokenizer = self.tokenizers[model_type]
        lengths = []
        examples = []

        logger.info("Analyzing %s for %s", dataset_path, model_type)

        # Load and analyze dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())

                    # Create prompt based on model type
                    if model_type == "mapper":
                        prompt = self._create_mapper_prompt(example)
                    else:
                        prompt = self._create_analyst_prompt(example)

                    # Tokenize and get length
                    tokens = tokenizer.encode(prompt, add_special_tokens=True)
                    length = len(tokens)

                    lengths.append(length)
                    examples.append(
                        {
                            "line": line_num,
                            "length": length,
                            "instruction": example.get("instruction", "")[:100] + "...",
                            "response": example.get("response", "")[:100] + "...",
                        }
                    )

                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON on line %s: %s", line_num, e)
                except Exception as e:
                    logger.warning("Error processing line %s: %s", line_num, e)

        # Calculate statistics
        lengths_array = np.array(lengths)
        stats = {
            "total_examples": len(lengths),
            "mean_length": float(np.mean(lengths_array)),
            "median_length": float(np.median(lengths_array)),
            "std_length": float(np.std(lengths_array)),
            "min_length": int(np.min(lengths_array)),
            "max_length": int(np.max(lengths_array)),
            "percentiles": {
                "50": float(np.percentile(lengths_array, 50)),
                "75": float(np.percentile(lengths_array, 75)),
                "90": float(np.percentile(lengths_array, 90)),
                "95": float(np.percentile(lengths_array, 95)),
                "99": float(np.percentile(lengths_array, 99)),
            },
        }

        # Identify outliers
        p95 = stats["percentiles"]["95"]
        outliers = [ex for ex in examples if ex["length"] > p95 * 1.5]

        # Generate recommendations
        recommendations = self._generate_recommendations(stats, model_type)

        result = {
            "dataset_path": dataset_path,
            "model_type": model_type,
            "statistics": stats,
            "outliers": outliers[:10],  # Top 10 outliers
            "recommendations": recommendations,
            "length_distribution": dict(Counter(lengths)),
        }

        self.results[model_type] = result
        return result

    def _create_mapper_prompt(self, example: Dict[str, str]) -> str:
        """Create mapper prompt format."""
        instruction = example.get("instruction", "")
        response = example.get("response", "")

        # Use Llama-3 format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a compliance taxonomy mapper. Map detector outputs to canonical taxonomy labels. Output only valid JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        return prompt

    def _create_analyst_prompt(self, example: Dict[str, str]) -> str:
        """Create analyst prompt format."""
        instruction = example.get("instruction", "")
        response = example.get("response", "")

        # Use Phi-3 format
        prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
        return prompt

    def _generate_recommendations(
        self, stats: Dict[str, Any], model_type: str
    ) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        p95 = stats["percentiles"]["95"]
        p99 = stats["percentiles"]["99"]

        # Get target from config
        target_p95 = self.config["sequence_lengths"][model_type][
            "target_95th_percentile"
        ]

        recommendations = {
            "recommended_max_length": int(
                p95 * 1.1
            ),  # 10% buffer above 95th percentile
            "target_95th_percentile": target_p95,
            "outlier_threshold": int(p99),
            "packing_efficiency": "high" if p95 < target_p95 else "medium",
            "memory_impact": "low" if p95 < target_p95 else "high",
            "truncation_risk": "low" if p95 < target_p95 else "medium",
        }

        # Specific recommendations
        if p95 > target_p95:
            recommendations["action_required"] = (
                "Consider reducing sequence length or implementing chunking"
            )
            recommendations["chunking_recommended"] = True
        else:
            recommendations["action_required"] = "Current length is optimal"
            recommendations["chunking_recommended"] = False

        return recommendations

    def generate_histograms(self, output_dir: str = "analysis/token_lengths") -> None:
        """Generate histograms for token length distributions."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping histogram generation.")
            return

        os.makedirs(output_dir, exist_ok=True)

        for model_type, result in self.results.items():
            lengths = list(result["length_distribution"].keys())
            counts = list(result["length_distribution"].values())

            plt.figure(figsize=(12, 6))
            plt.hist(lengths, bins=50, weights=counts, alpha=0.7, edgecolor="black")
            plt.axvline(
                result["statistics"]["percentiles"]["95"],
                color="red",
                linestyle="--",
                label=f'95th percentile: {result["statistics"]["percentiles"]["95"]:.0f}',
            )
            plt.axvline(
                result["recommendations"]["recommended_max_length"],
                color="green",
                linestyle="--",
                label=f'Recommended max: {result["recommendations"]["recommended_max_length"]}',
            )

            plt.xlabel("Token Length")
            plt.ylabel("Frequency")
            plt.title(f"Token Length Distribution - {model_type.title()} Model")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(
                f"{output_dir}/{model_type}_token_lengths.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logger.info(
                "Generated histogram for %s at %s/%s_token_lengths.png",
                model_type,
                output_dir,
                model_type,
            )

    def save_results(self, output_dir: str = "analysis/token_lengths") -> None:
        """Save analysis results to JSON files."""
        os.makedirs(output_dir, exist_ok=True)

        for model_type, result in self.results.items():
            output_path = f"{output_dir}/{model_type}_analysis.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Saved analysis results for %s to %s", model_type, output_path)

    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("TOKEN LENGTH ANALYSIS SUMMARY")
        print("=" * 80)

        for model_type, result in self.results.items():
            stats = result["statistics"]
            rec = result["recommendations"]

            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Total examples: {stats['total_examples']:,}")
            print(f"  Mean length: {stats['mean_length']:.1f} tokens")
            print(f"  95th percentile: {stats['percentiles']['95']:.1f} tokens")
            print(f"  99th percentile: {stats['percentiles']['99']:.1f} tokens")
            print(f"  Max length: {stats['max_length']} tokens")
            print(f"  Recommended max length: {rec['recommended_max_length']} tokens")
            print(f"  Action required: {rec['action_required']}")
            print(f"  Packing efficiency: {rec['packing_efficiency']}")

            if rec["chunking_recommended"]:
                print(
                    f"  ⚠️  CHUNKING RECOMMENDED for sequences > {rec['outlier_threshold']} tokens"
                )


def main():
    """Main function for token length analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze token lengths for fine-tuning preparation"
    )
    parser.add_argument(
        "--mapper-data", required=True, help="Path to mapper training data (JSONL)"
    )
    parser.add_argument(
        "--analyst-data", required=True, help="Path to analyst training data (JSONL)"
    )
    parser.add_argument(
        "--config",
        default="config/fine_tuning_preparation.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output-dir", default="analysis/token_lengths", help="Output directory"
    )
    parser.add_argument(
        "--generate-plots", action="store_true", help="Generate histograms"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TokenLengthAnalyzer(args.config)
    analyzer.setup_tokenizers()

    if not analyzer.tokenizers:
        print("Error: No tokenizers loaded. Check your configuration and dependencies.")
        return

    # Analyze datasets
    try:
        analyzer.analyze_dataset(args.mapper_data, "mapper")
        analyzer.analyze_dataset(args.analyst_data, "analyst")

        # Generate outputs
        analyzer.save_results(args.output_dir)
        if args.generate_plots:
            analyzer.generate_histograms(args.output_dir)

        # Print summary
        analyzer.print_summary()

        print(f"\nAnalysis complete! Results saved to {args.output_dir}")

    except Exception as e:
        logger.error("Analysis failed: %s", e)
        raise


if __name__ == "__main__":
    main()
