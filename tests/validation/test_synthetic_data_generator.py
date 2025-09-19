#!/usr/bin/env python3
"""
Test script for SyntheticDataGenerator to verify implementation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.training.data_generator import SyntheticDataGenerator


def main():
    """Test the SyntheticDataGenerator implementation."""
    print("Testing SyntheticDataGenerator...")

    try:
        # Initialize components
        taxonomy_loader = TaxonomyLoader("pillars-detectors/taxonomy.yaml")

        # Create generator
        generator = SyntheticDataGenerator(
            taxonomy_loader=taxonomy_loader, random_seed=42  # For reproducible results
        )

        # Load taxonomy
        print("Loading taxonomy...")
        generator.load_taxonomy()

        # Test PII generation
        print("Generating synthetic PII examples...")
        pii_examples = generator.generate_synthetic_pii_examples(num_examples=20)
        print(f"Generated {len(pii_examples)} PII examples")

        # Show sample PII examples
        print("\nSample PII examples:")
        for i, example in enumerate(pii_examples[:3]):
            print(f"\nPII Example {i+1}:")
            print(f"Instruction: {example.instruction}")
            print(f"Response: {example.response}")
            print(f"Metadata: {example.metadata}")

        # Test jailbreak generation
        print("\nGenerating synthetic jailbreak examples...")
        jailbreak_examples = generator.generate_synthetic_jailbreak_examples(
            num_examples=10
        )
        print(f"Generated {len(jailbreak_examples)} jailbreak examples")

        # Show sample jailbreak example
        if jailbreak_examples:
            print("\nSample jailbreak example:")
            example = jailbreak_examples[0]
            print(f"Instruction: {example.instruction}")
            print(f"Response: {example.response}")
            print(
                f"Synthetic prompt: {example.metadata.get('synthetic_prompt', 'N/A')}"
            )

        # Test prompt injection generation
        print("\nGenerating synthetic prompt injection examples...")
        injection_examples = generator.generate_synthetic_prompt_injection_examples(
            num_examples=15
        )
        print(f"Generated {len(injection_examples)} prompt injection examples")

        # Show sample injection example
        if injection_examples:
            print("\nSample prompt injection example:")
            example = injection_examples[0]
            print(f"Instruction: {example.instruction}")
            print(f"Response: {example.response}")
            print(
                f"Synthetic prompt: {example.metadata.get('synthetic_prompt', 'N/A')}"
            )

        # Test balanced training set generation
        print("\nGenerating balanced synthetic training set...")
        balanced_examples = generator.generate_balanced_training_set(
            target_examples_per_category=5,
            include_pii=True,
            include_jailbreak=True,
            include_prompt_injection=True,
        )
        print(f"Generated {len(balanced_examples)} balanced examples")

        # Get statistics
        print("\nSynthetic data statistics:")
        all_synthetic_examples = pii_examples + jailbreak_examples + injection_examples
        stats = generator.get_synthetic_statistics(all_synthetic_examples)
        print(json.dumps(stats, indent=2, default=str))

        # Save sample synthetic data
        output_path = Path("sample_synthetic_data.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in balanced_examples[:20]:  # Save first 20 examples
                f.write(example.to_json() + "\n")

        print(f"\nSaved sample synthetic data to {output_path}")

        print("\n✅ SyntheticDataGenerator test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
