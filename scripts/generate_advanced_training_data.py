#!/usr/bin/env python3
"""Generate advanced training data with missing high-value sources."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.training.data_generator import AdvancedTrainingDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_advanced_training_data(
    output_file: str,
    total_examples: int = 1000,
    include_fda: bool = True,
    include_aml: bool = True,
    include_audit: bool = True,
    include_legal_reasoning: bool = True,
    include_few_shot: bool = True,
    include_chain_of_thought: bool = True,
) -> Dict[str, Any]:
    """Generate advanced training data with missing high-value sources."""
    logger.info("Starting advanced training data generation...")

    # Initialize generator
    generator = AdvancedTrainingDataGenerator()

    all_examples = []

    # Generate different types of advanced examples
    if include_fda:
        fda_examples = generator.generate_fda_enforcement_examples(total_examples // 6)
        all_examples.extend(fda_examples)
        logger.info("Generated %s FDA enforcement examples", len(fda_examples))

    if include_aml:
        aml_examples = generator.generate_aml_compliance_examples(total_examples // 6)
        all_examples.extend(aml_examples)
        logger.info("Generated %s AML compliance examples", len(aml_examples))

    if include_audit:
        audit_examples = generator.generate_audit_findings_examples(total_examples // 6)
        all_examples.extend(audit_examples)
        logger.info("Generated %s audit findings examples", len(audit_examples))

    if include_legal_reasoning:
        legal_examples = generator.generate_legal_reasoning_examples(
            total_examples // 6
        )
        all_examples.extend(legal_examples)
        logger.info("Generated %s legal reasoning examples", len(legal_examples))

    if include_few_shot:
        few_shot_examples = generator.generate_few_shot_examples(total_examples // 6)
        all_examples.extend(few_shot_examples)
        logger.info("Generated %s few-shot examples", len(few_shot_examples))

    if include_chain_of_thought:
        cot_examples = generator.generate_chain_of_thought_examples(total_examples // 6)
        all_examples.extend(cot_examples)
        logger.info("Generated %s chain-of-thought examples", len(cot_examples))

    # Get statistics
    stats = {
        "total_examples": len(all_examples),
        "example_types": {},
        "domains": {},
        "complexity_levels": {},
        "reasoning_types": {},
        "average_confidence": 0.0,
    }

    confidences = []
    for example in all_examples:
        metadata = example.metadata
        example_type = metadata.get("example_type", "unknown")
        domain = metadata.get("domain", "unknown")
        complexity = metadata.get("complexity_level", "unknown")
        reasoning_type = metadata.get("reasoning_type", "unknown")

        stats["example_types"][example_type] = (
            stats["example_types"].get(example_type, 0) + 1
        )
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        stats["complexity_levels"][complexity] = (
            stats["complexity_levels"].get(complexity, 0) + 1
        )
        stats["reasoning_types"][reasoning_type] = (
            stats["reasoning_types"].get(reasoning_type, 0) + 1
        )

        try:
            response_data = json.loads(example.response)
            confidence = response_data.get("confidence", 0.0)
            confidences.append(confidence)
        except (json.JSONDecodeError, KeyError):
            pass

    if confidences:
        stats["average_confidence"] = sum(confidences) / len(confidences)

    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in all_examples:
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

    logger.info("Generated %s advanced training examples", len(all_examples))
    logger.info("Saved to: %s", output_path)

    return stats


def main():
    """Main function for advanced training data generation."""
    parser = argparse.ArgumentParser(description="Generate advanced training data")
    parser.add_argument(
        "--output",
        type=str,
        default="llm/llm-reports/advanced_training_data.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=1000,
        help="Total number of examples to generate",
    )
    parser.add_argument(
        "--include-fda",
        action="store_true",
        default=True,
        help="Include FDA enforcement examples",
    )
    parser.add_argument(
        "--include-aml",
        action="store_true",
        default=True,
        help="Include AML compliance examples",
    )
    parser.add_argument(
        "--include-audit",
        action="store_true",
        default=True,
        help="Include audit findings examples",
    )
    parser.add_argument(
        "--include-legal-reasoning",
        action="store_true",
        default=True,
        help="Include legal reasoning examples",
    )
    parser.add_argument(
        "--include-few-shot",
        action="store_true",
        default=True,
        help="Include few-shot learning examples",
    )
    parser.add_argument(
        "--include-chain-of-thought",
        action="store_true",
        default=True,
        help="Include chain-of-thought examples",
    )
    parser.add_argument(
        "--stats-output", type=str, help="Output file for generation statistics"
    )

    args = parser.parse_args()

    try:
        stats = generate_advanced_training_data(
            output_file=args.output,
            total_examples=args.total_examples,
            include_fda=args.include_fda,
            include_aml=args.include_aml,
            include_audit=args.include_audit,
            include_legal_reasoning=args.include_legal_reasoning,
            include_few_shot=args.include_few_shot,
            include_chain_of_thought=args.include_chain_of_thought,
        )

        # Print statistics
        logger.info("Generation Statistics:")
        logger.info("Total examples: %s", stats["total_examples"])
        logger.info("Example types: %s", stats["example_types"])
        logger.info("Domains: %s", stats["domains"])
        logger.info("Complexity levels: %s", stats["complexity_levels"])
        logger.info("Reasoning types: %s", stats["reasoning_types"])
        logger.info("Average confidence: %.3f", stats["average_confidence"])

        # Save statistics if requested
        if args.stats_output:
            stats_path = Path(args.stats_output)
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info("Statistics saved to: %s", stats_path)

        logger.info("Advanced training data generation completed successfully!")

    except Exception as e:
        logger.error("Error generating training data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
