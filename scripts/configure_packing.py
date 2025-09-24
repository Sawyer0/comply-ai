#!/usr/bin/env python3
"""
Packing Configuration Script for Fine-Tuning

Configures packing strategies and tokens-per-update budgets for optimal
training efficiency and memory usage.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PackingConfigurator:
    """Configures packing strategies for efficient training."""

    def __init__(self, config_path: str = "config/fine_tuning_preparation.yaml"):
        """Initialize packer with configuration."""
        self.config = self._load_config(config_path)
        self.packing_strategies = {}

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
            "packing_config": {
                "mapper": {
                    "target_tokens_per_step": 48000,
                    "batch_size": 4,
                    "gradient_accumulation": 8,
                },
                "analyst": {
                    "target_tokens_per_step": 24000,
                    "batch_size": 8,
                    "gradient_accumulation": 4,
                },
            }
        }

    def calculate_optimal_packing(
        self, model_type: str, token_lengths: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate optimal packing configuration for a model type.

        Args:
            model_type: Type of model ("mapper" or "analyst")
            token_lengths: List of token lengths from analysis

        Returns:
            Optimal packing configuration
        """
        target_tokens = self.config["packing_config"][model_type][
            "target_tokens_per_step"
        ]
        batch_size = self.config["packing_config"][model_type]["batch_size"]
        grad_accum = self.config["packing_config"][model_type]["gradient_accumulation"]

        # Calculate effective batch size
        effective_batch_size = batch_size * grad_accum

        # Calculate target tokens per sample
        target_tokens_per_sample = target_tokens // effective_batch_size

        # Analyze length distribution for packing
        lengths_array = np.array(token_lengths)
        mean_length = np.mean(lengths_array)
        median_length = np.median(lengths_array)
        p95_length = np.percentile(lengths_array, 95)

        # Determine packing strategy
        if p95_length <= target_tokens_per_sample * 0.8:
            # Most samples fit comfortably - use simple packing
            strategy = "simple_packing"
            packing_efficiency = 0.9
        elif p95_length <= target_tokens_per_sample:
            # Some samples are close to limit - use length bucketing
            strategy = "length_bucketing"
            packing_efficiency = 0.85
        else:
            # Many samples exceed limit - use chunking
            strategy = "chunking"
            packing_efficiency = 0.75

        # Calculate memory requirements
        max_seq_len = int(p95_length * 1.1)  # 10% buffer
        memory_per_sample = self._estimate_memory_usage(model_type, max_seq_len)
        total_memory = memory_per_sample * effective_batch_size

        # Generate recommendations
        recommendations = self._generate_packing_recommendations(
            model_type, strategy, packing_efficiency, total_memory, target_tokens
        )

        config = {
            "model_type": model_type,
            "strategy": strategy,
            "target_tokens_per_step": target_tokens,
            "batch_size": batch_size,
            "gradient_accumulation": grad_accum,
            "effective_batch_size": effective_batch_size,
            "target_tokens_per_sample": target_tokens_per_sample,
            "max_sequence_length": max_seq_len,
            "packing_efficiency": packing_efficiency,
            "estimated_memory_gb": total_memory,
            "recommendations": recommendations,
            "length_statistics": {
                "mean": float(mean_length),
                "median": float(median_length),
                "p95": float(p95_length),
            },
        }

        self.packing_strategies[model_type] = config
        return config

    def _estimate_memory_usage(self, model_type: str, seq_len: int) -> float:
        """Estimate memory usage per sample in GB."""
        # Rough estimates based on model size and sequence length
        if model_type == "mapper":
            # Llama-3-8B with LoRA
            base_memory = 8.0  # GB for model weights
            sequence_memory = seq_len * 0.0001  # Rough estimate
            return base_memory + sequence_memory
        else:
            # Phi-3-Mini with LoRA
            base_memory = 2.0  # GB for model weights
            sequence_memory = seq_len * 0.00005  # Rough estimate
            return base_memory + sequence_memory

    def _generate_packing_recommendations(
        self,
        model_type: str,
        strategy: str,
        efficiency: float,
        memory_gb: float,
        target_tokens: int,
    ) -> Dict[str, Any]:
        """Generate specific recommendations for packing configuration."""
        recommendations = {
            "strategy": strategy,
            "efficiency_rating": (
                "high"
                if efficiency > 0.85
                else "medium" if efficiency > 0.75 else "low"
            ),
            "memory_requirement": f"{memory_gb:.1f}GB",
            "gpu_recommendation": self._get_gpu_recommendation(memory_gb),
            "optimization_suggestions": [],
        }

        # Add specific suggestions based on strategy
        if strategy == "simple_packing":
            recommendations["optimization_suggestions"].append(
                "Use standard padding - most samples fit within target length"
            )
        elif strategy == "length_bucketing":
            recommendations["optimization_suggestions"].append(
                "Group samples by similar lengths to reduce padding waste"
            )
            recommendations["optimization_suggestions"].append(
                "Consider dynamic batching for better efficiency"
            )
        elif strategy == "chunking":
            recommendations["optimization_suggestions"].append(
                "Implement sequence chunking for long samples"
            )
            recommendations["optimization_suggestions"].append(
                "Consider reducing max_sequence_length to improve efficiency"
            )

        # Memory-based suggestions
        if memory_gb > 20:
            recommendations["optimization_suggestions"].append(
                "Consider reducing batch size or using gradient checkpointing"
            )
        elif memory_gb < 8:
            recommendations["optimization_suggestions"].append(
                "Can increase batch size for better throughput"
            )

        # Token efficiency suggestions
        if efficiency < 0.8:
            recommendations["optimization_suggestions"].append(
                "Consider implementing dynamic sequence length based on actual content"
            )

        return recommendations

    def _get_gpu_recommendation(self, memory_gb: float) -> str:
        """Get GPU recommendation based on memory requirements."""
        if memory_gb <= 8:
            return "T4 (16GB) or A10G (24GB)"
        elif memory_gb <= 16:
            return "A10G (24GB) or V100 (32GB)"
        elif memory_gb <= 24:
            return "A100 (40GB) or V100 (32GB)"
        else:
            return "A100 (80GB) or multiple GPUs"

    def generate_training_config(
        self, output_path: str = "config/optimized_training_config.json"
    ) -> None:
        """Generate optimized training configuration with packing settings."""
        config = {
            "experiment_name": "comply-ai-optimized-dual-model",
            "description": "Optimized training configuration with packing and token limits",
            "models": {},
            "training": {},
            "packing": {},
            "performance_targets": {},
        }

        for model_type, packing_config in self.packing_strategies.items():
            # Model configuration
            if model_type == "mapper":
                config["models"]["mapper"] = {
                    "base_model": "meta-llama/Llama-3-8B-Instruct",
                    "max_sequence_length": packing_config["max_sequence_length"],
                    "lora": {
                        "r": 256,
                        "alpha": 512,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        "dropout": 0.1,
                    },
                }
            else:
                config["models"]["analyst"] = {
                    "base_model": "microsoft/Phi-3-mini-4k-instruct",
                    "max_sequence_length": packing_config["max_sequence_length"],
                    "lora": {
                        "r": 128,
                        "alpha": 256,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "dropout": 0.1,
                    },
                }

            # Training configuration
            config["training"][model_type] = {
                "batch_size": packing_config["batch_size"],
                "gradient_accumulation": packing_config["gradient_accumulation"],
                "effective_batch_size": packing_config["effective_batch_size"],
                "target_tokens_per_step": packing_config["target_tokens_per_step"],
                "max_sequence_length": packing_config["max_sequence_length"],
                "learning_rate": 5e-5 if model_type == "mapper" else 1e-4,
                "epochs": 3 if model_type == "mapper" else 2,
                "optimization": {
                    "use_gradient_checkpointing": True,
                    "use_flash_attention": True,
                    "mixed_precision": "fp16",
                },
            }

            # Packing configuration
            config["packing"][model_type] = {
                "strategy": packing_config["strategy"],
                "efficiency": packing_config["packing_efficiency"],
                "estimated_memory_gb": packing_config["estimated_memory_gb"],
                "recommendations": packing_config["recommendations"],
            }

            # Performance targets
            config["performance_targets"][model_type] = {
                "target_throughput": packing_config["target_tokens_per_step"]
                / 60,  # tokens per second
                "memory_limit_gb": packing_config["estimated_memory_gb"]
                * 1.2,  # 20% buffer
                "gpu_recommendation": packing_config["recommendations"][
                    "gpu_recommendation"
                ],
            }

        # Save configuration
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Generated optimized training configuration: %s", output_path)

    def print_summary(self) -> None:
        """Print packing configuration summary."""
        print("\n" + "=" * 80)
        print("PACKING CONFIGURATION SUMMARY")
        print("=" * 80)

        for model_type, config in self.packing_strategies.items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Strategy: {config['strategy']}")
            print(f"  Target tokens/step: {config['target_tokens_per_step']:,}")
            print(f"  Batch size: {config['batch_size']}")
            print(f"  Gradient accumulation: {config['gradient_accumulation']}")
            print(f"  Effective batch size: {config['effective_batch_size']}")
            print(f"  Max sequence length: {config['max_sequence_length']}")
            print(f"  Packing efficiency: {config['packing_efficiency']:.1%}")
            print(f"  Estimated memory: {config['estimated_memory_gb']:.1f}GB")
            print(
                f"  GPU recommendation: {config['recommendations']['gpu_recommendation']}"
            )

            if config["recommendations"]["optimization_suggestions"]:
                print(f"  Optimization suggestions:")
                for suggestion in config["recommendations"]["optimization_suggestions"]:
                    print(f"    • {suggestion}")


def main():
    """Main function for packing configuration."""
    parser = argparse.ArgumentParser(description="Configure packing for fine-tuning")
    parser.add_argument(
        "--mapper-analysis", required=True, help="Path to mapper token analysis JSON"
    )
    parser.add_argument(
        "--analyst-analysis", required=True, help="Path to analyst token analysis JSON"
    )
    parser.add_argument(
        "--config",
        default="config/fine_tuning_preparation.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output",
        default="config/optimized_training_config.json",
        help="Output configuration file",
    )

    args = parser.parse_args()

    # Initialize configurator
    configurator = PackingConfigurator(args.config)

    # Load analysis results
    with open(args.mapper_analysis, "r") as f:
        mapper_analysis = json.load(f)

    with open(args.analyst_analysis, "r") as f:
        analyst_analysis = json.load(f)

    # Extract token lengths (simulate from distribution)
    mapper_lengths = []
    for length, count in mapper_analysis["length_distribution"].items():
        mapper_lengths.extend([int(length)] * count)

    analyst_lengths = []
    for length, count in analyst_analysis["length_distribution"].items():
        analyst_lengths.extend([int(length)] * count)

    # Calculate optimal packing
    configurator.calculate_optimal_packing("mapper", mapper_lengths)
    configurator.calculate_optimal_packing("analyst", analyst_lengths)

    # Generate configuration
    configurator.generate_training_config(args.output)

    # Print summary
    configurator.print_summary()

    print(f"\nPacking configuration complete! Optimized config saved to {args.output}")


if __name__ == "__main__":
    main()
