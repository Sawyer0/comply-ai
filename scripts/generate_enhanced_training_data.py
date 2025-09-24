#!/usr/bin/env python3
"""Generate enhanced training data with real-world and synthetic examples."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.training.data_generator import (
    HybridTrainingDataGenerator,
    RealWorldDataCollector,
    SyntheticDataGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_hybrid_training_data(
    output_file: str,
    total_examples: int = 2500,
    real_world_ratio: float = 0.6,
    synthetic_ratio: float = 0.4,
    include_industries: bool = True,
    include_edge_cases: bool = True,
) -> Dict[str, Any]:
    """Generate hybrid training data with real-world and synthetic examples."""
    logger.info("Starting hybrid training data generation...")

    # Initialize generators
    taxonomy_loader = TaxonomyLoader(
        taxonomy_path=".kiro/pillars-detectors/taxonomy.yaml"
    )
    hybrid_generator = HybridTrainingDataGenerator(
        taxonomy_loader=taxonomy_loader,
        real_world_ratio=real_world_ratio,
        synthetic_ratio=synthetic_ratio,
    )

    # Generate hybrid training set
    examples = hybrid_generator.generate_hybrid_training_set(
        total_examples=total_examples,
        real_world_categories=(
            ["sec", "gdpr", "hipaa", "industries", "edge_cases"]
            if include_edge_cases
            else ["sec", "gdpr", "hipaa", "industries"]
        ),
        synthetic_categories=["pii", "prompt_injection", "jailbreak", "augmented"],
        balance_categories=True,
    )

    # Get statistics
    stats = hybrid_generator.get_hybrid_statistics(examples)

    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    {
                        "instruction": example.instruction,
                        "response": example.response,
                        "metadata": example.metadata,
                    }
                )
                + "\n"
            )

    logger.info("Generated %s hybrid training examples", len(examples))
    logger.info("Saved to: %s", output_path)

    return stats


def generate_industry_specific_data(
    industry: str,
    output_file: str,
    total_examples: int = 500,
    real_world_ratio: float = 0.7,
) -> Dict[str, Any]:
    """Generate industry-specific hybrid training data."""
    logger.info("Generating %s industry-specific training data...", industry)

    # Initialize generators
    taxonomy_loader = TaxonomyLoader(
        taxonomy_path=".kiro/pillars-detectors/taxonomy.yaml"
    )
    hybrid_generator = HybridTrainingDataGenerator(taxonomy_loader=taxonomy_loader)

    # Generate industry-specific examples
    examples = hybrid_generator.generate_industry_specific_hybrid(
        industry=industry,
        total_examples=total_examples,
        real_world_ratio=real_world_ratio,
    )

    # Get statistics
    stats = hybrid_generator.get_hybrid_statistics(examples)

    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    {
                        "instruction": example.instruction,
                        "response": example.response,
                        "metadata": example.metadata,
                    }
                )
                + "\n"
            )

    logger.info("Generated %s %s industry examples", len(examples), industry)
    logger.info("Saved to: %s", output_path)

    return stats


def generate_edge_case_data(
    output_file: str,
    total_examples: int = 300,
    real_world_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Generate edge case hybrid training data."""
    logger.info("Generating edge case training data...")

    # Initialize generators
    taxonomy_loader = TaxonomyLoader(
        taxonomy_path=".kiro/pillars-detectors/taxonomy.yaml"
    )
    hybrid_generator = HybridTrainingDataGenerator(taxonomy_loader=taxonomy_loader)

    # Generate edge case examples
    examples = hybrid_generator.generate_edge_case_hybrid(
        total_examples=total_examples,
        real_world_ratio=real_world_ratio,
    )

    # Get statistics
    stats = hybrid_generator.get_hybrid_statistics(examples)

    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    {
                        "instruction": example.instruction,
                        "response": example.response,
                        "metadata": example.metadata,
                    }
                )
                + "\n"
            )

    logger.info("Generated %s edge case examples", len(examples))
    logger.info("Saved to: %s", output_path)

    return stats


def generate_enhanced_synthetic_data(
    output_file: str,
    total_examples: int = 1000,
    use_real_patterns: bool = True,
    include_edge_cases: bool = True,
) -> Dict[str, Any]:
    """Generate enhanced synthetic training data."""
    logger.info("Generating enhanced synthetic training data...")

    # Initialize generators
    taxonomy_loader = TaxonomyLoader(
        taxonomy_path=".kiro/pillars-detectors/taxonomy.yaml"
    )
    hybrid_generator = HybridTrainingDataGenerator(taxonomy_loader=taxonomy_loader)

    # Generate enhanced synthetic examples
    examples = hybrid_generator.generate_enhanced_synthetic_data(
        num_examples=total_examples,
        use_real_patterns=use_real_patterns,
        include_edge_cases=include_edge_cases,
    )

    # Get statistics
    stats = hybrid_generator.get_hybrid_statistics(examples)

    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    {
                        "instruction": example.instruction,
                        "response": example.response,
                        "metadata": example.metadata,
                    }
                )
                + "\n"
            )

    logger.info("Generated %s enhanced synthetic examples", len(examples))
    logger.info("Saved to: %s", output_path)

    return stats


def main():
    """Main function for training data generation."""
    parser = argparse.ArgumentParser(description="Generate enhanced training data")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "industry", "edge-cases", "synthetic"],
        default="hybrid",
        help="Generation mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm/llm-reports/enhanced_training_data.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=2500,
        help="Total number of examples to generate",
    )
    parser.add_argument(
        "--real-world-ratio",
        type=float,
        default=0.6,
        help="Ratio of real-world examples",
    )
    parser.add_argument(
        "--industry",
        type=str,
        choices=["financial", "healthcare", "technology", "retail"],
        default="financial",
        help="Industry for industry-specific generation",
    )
    parser.add_argument(
        "--use-real-patterns",
        action="store_true",
        help="Use real-world patterns for synthetic generation",
    )
    parser.add_argument(
        "--include-edge-cases",
        action="store_true",
        default=True,
        help="Include edge cases in generation",
    )
    parser.add_argument(
        "--stats-output", type=str, help="Output file for generation statistics"
    )

    args = parser.parse_args()

    try:
        if args.mode == "hybrid":
            stats = generate_hybrid_training_data(
                output_file=args.output,
                total_examples=args.total_examples,
                real_world_ratio=args.real_world_ratio,
                synthetic_ratio=1.0 - args.real_world_ratio,
                include_edge_cases=args.include_edge_cases,
            )
        elif args.mode == "industry":
            stats = generate_industry_specific_data(
                industry=args.industry,
                output_file=args.output,
                total_examples=args.total_examples,
                real_world_ratio=args.real_world_ratio,
            )
        elif args.mode == "edge-cases":
            stats = generate_edge_case_data(
                output_file=args.output,
                total_examples=args.total_examples,
                real_world_ratio=args.real_world_ratio,
            )
        elif args.mode == "synthetic":
            stats = generate_enhanced_synthetic_data(
                output_file=args.output,
                total_examples=args.total_examples,
                use_real_patterns=args.use_real_patterns,
                include_edge_cases=args.include_edge_cases,
            )

        # Print statistics
        logger.info("Generation Statistics:")
        logger.info("Total examples: %s", stats["total_examples"])
        logger.info("Real-world examples: %s", stats["real_world_examples"])
        logger.info("Synthetic examples: %s", stats["synthetic_examples"])
        logger.info(
            "Real-world ratio: %.2f", stats["data_quality_metrics"]["real_world_ratio"]
        )
        logger.info(
            "Synthetic ratio: %.2f", stats["data_quality_metrics"]["synthetic_ratio"]
        )
        logger.info(
            "Edge case ratio: %.2f", stats["data_quality_metrics"]["edge_case_ratio"]
        )
        logger.info(
            "Multi-category ratio: %.2f",
            stats["data_quality_metrics"]["multi_category_ratio"],
        )
        logger.info("Average confidence: %.3f", stats["confidence_stats"]["avg"])

        # Save statistics if requested
        if args.stats_output:
            stats_path = Path(args.stats_output)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info("Statistics saved to: %s", stats_path)

        logger.info("Training data generation completed successfully!")

    except Exception as e:
        logger.error("Error generating training data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
