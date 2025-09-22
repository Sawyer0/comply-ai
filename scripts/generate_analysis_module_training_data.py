#!/usr/bin/env python3
"""Generate enhanced training data for Analysis Module (Phi-3 Mini)."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.training.data_generator import AnalysisModuleDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_analysis_module_training_data(
    output_file: str,
    total_examples: int = 1000,
    include_coverage_gaps: bool = True,
    include_incidents: bool = True,
    include_threshold_tuning: bool = True,
    include_opa_policies: bool = True,
) -> Dict[str, Any]:
    """Generate training data for Analysis Module."""
    logger.info("Starting Analysis Module training data generation...")
    
    # Initialize generator
    generator = AnalysisModuleDataGenerator()
    
    # Generate balanced training set
    examples = generator.generate_balanced_analysis_set(
        target_examples_per_category=total_examples // 4,
        include_coverage_gaps=include_coverage_gaps,
        include_incidents=include_incidents,
        include_threshold_tuning=include_threshold_tuning,
        include_opa_policies=include_opa_policies,
    )
    
    # Get statistics
    stats = generator.get_analysis_statistics(examples)
    
    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "instruction": example.instruction,
                "response": example.response,
                "metadata": example.metadata,
            }) + "\n")
    
    logger.info("Generated %s Analysis Module training examples", len(examples))
    logger.info("Saved to: %s", output_path)
    
    return stats


def generate_coverage_gap_data(
    output_file: str,
    total_examples: int = 200,
) -> Dict[str, Any]:
    """Generate coverage gap analysis training data."""
    logger.info("Generating coverage gap analysis training data...")
    
    # Initialize generator
    generator = AnalysisModuleDataGenerator()
    
    # Generate coverage gap examples
    examples = generator.generate_coverage_gap_examples(total_examples)
    
    # Get statistics
    stats = generator.get_analysis_statistics(examples)
    
    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "instruction": example.instruction,
                "response": example.response,
                "metadata": example.metadata,
            }) + "\n")
    
    logger.info("Generated %s coverage gap examples", len(examples))
    logger.info("Saved to: %s", output_path)
    
    return stats


def generate_incident_anomaly_data(
    output_file: str,
    total_examples: int = 150,
) -> Dict[str, Any]:
    """Generate incident/anomaly analysis training data."""
    logger.info("Generating incident/anomaly analysis training data...")
    
    # Initialize generator
    generator = AnalysisModuleDataGenerator()
    
    # Generate incident/anomaly examples
    examples = generator.generate_incident_anomaly_examples(total_examples)
    
    # Get statistics
    stats = generator.get_analysis_statistics(examples)
    
    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "instruction": example.instruction,
                "response": example.response,
                "metadata": example.metadata,
            }) + "\n")
    
    logger.info("Generated %s incident/anomaly examples", len(examples))
    logger.info("Saved to: %s", output_path)
    
    return stats


def generate_threshold_tuning_data(
    output_file: str,
    total_examples: int = 100,
) -> Dict[str, Any]:
    """Generate threshold tuning training data."""
    logger.info("Generating threshold tuning training data...")
    
    # Initialize generator
    generator = AnalysisModuleDataGenerator()
    
    # Generate threshold tuning examples
    examples = generator.generate_threshold_tuning_examples(total_examples)
    
    # Get statistics
    stats = generator.get_analysis_statistics(examples)
    
    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "instruction": example.instruction,
                "response": example.response,
                "metadata": example.metadata,
            }) + "\n")
    
    logger.info("Generated %s threshold tuning examples", len(examples))
    logger.info("Saved to: %s", output_path)
    
    return stats


def generate_opa_policy_data(
    output_file: str,
    total_examples: int = 100,
) -> Dict[str, Any]:
    """Generate OPA policy training data."""
    logger.info("Generating OPA policy training data...")
    
    # Initialize generator
    generator = AnalysisModuleDataGenerator()
    
    # Generate OPA policy examples
    examples = generator.generate_opa_policy_examples(total_examples)
    
    # Get statistics
    stats = generator.get_analysis_statistics(examples)
    
    # Save examples
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "instruction": example.instruction,
                "response": example.response,
                "metadata": example.metadata,
            }) + "\n")
    
    logger.info("Generated %s OPA policy examples", len(examples))
    logger.info("Saved to: %s", output_path)
    
    return stats


def main():
    """Main function for Analysis Module training data generation."""
    parser = argparse.ArgumentParser(description="Generate Analysis Module training data")
    parser.add_argument(
        "--mode",
        choices=["all", "coverage-gaps", "incidents", "threshold-tuning", "opa-policies"],
        default="all",
        help="Generation mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm/llm-reports/analysis_module_training_data.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=1000,
        help="Total number of examples to generate"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        help="Output file for generation statistics"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "all":
            stats = generate_analysis_module_training_data(
                output_file=args.output,
                total_examples=args.total_examples,
            )
        elif args.mode == "coverage-gaps":
            stats = generate_coverage_gap_data(
                output_file=args.output,
                total_examples=args.total_examples,
            )
        elif args.mode == "incidents":
            stats = generate_incident_anomaly_data(
                output_file=args.output,
                total_examples=args.total_examples,
            )
        elif args.mode == "threshold-tuning":
            stats = generate_threshold_tuning_data(
                output_file=args.output,
                total_examples=args.total_examples,
            )
        elif args.mode == "opa-policies":
            stats = generate_opa_policy_data(
                output_file=args.output,
                total_examples=args.total_examples,
            )
        
        # Print statistics
        logger.info("Generation Statistics:")
        logger.info("Total examples: %s", stats["total_examples"])
        logger.info("Analysis types: %s", stats["analysis_types"])
        logger.info("Tenants: %s", stats["tenants"])
        logger.info("Apps: %s", stats["apps"])
        logger.info("Severity levels: %s", stats["severity_levels"])
        logger.info("OPA policies: %s", stats["opa_policy_stats"]["total_with_opa"])
        logger.info("Average confidence: %.3f", stats["confidence_stats"]["avg"])
        
        # Save statistics if requested
        if args.stats_output:
            stats_path = Path(args.stats_output)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info("Statistics saved to: %s", stats_path)
        
        logger.info("Analysis Module training data generation completed successfully!")
        
    except Exception as e:
        logger.error("Error generating training data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
