#!/usr/bin/env python3
"""
Test script for TrainingDataGenerator to verify implementation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_mapper.training.data_generator import TrainingDataGenerator
from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.data.detectors import DetectorConfigLoader


def main():
    """Test the TrainingDataGenerator implementation."""
    print("Testing TrainingDataGenerator...")
    
    try:
        # Initialize components
        taxonomy_loader = TaxonomyLoader("pillars-detectors/taxonomy.yaml")
        detector_loader = DetectorConfigLoader("pillars-detectors")
        
        # Create generator
        generator = TrainingDataGenerator(
            detector_loader=detector_loader,
            taxonomy_loader=taxonomy_loader,
            random_seed=42  # For reproducible results
        )
        
        # Load data
        print("Loading taxonomy and detector configurations...")
        generator.load_data()
        
        # Generate training examples
        print("Generating training examples...")
        examples = generator.generate_training_examples(
            examples_per_mapping=2,  # Fewer examples for testing
            include_multi_label=True,
            include_variations=True
        )
        
        print(f"Generated {len(examples)} training examples")
        
        # Show some examples
        print("\nSample training examples:")
        for i, example in enumerate(examples[:3]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {example.instruction}")
            print(f"Response: {example.response}")
            print(f"Metadata: {example.metadata}")
        
        # Get statistics
        print("\nGeneration statistics:")
        stats = generator.get_generation_statistics(examples)
        print(json.dumps(stats, indent=2, default=str))
        
        # Validate examples
        print("\nValidating training examples...")
        validation_report = generator.validate_training_examples(examples)
        print(f"Valid examples: {validation_report['valid_examples']}/{validation_report['total_examples']}")
        
        if validation_report['errors']:
            print(f"Errors found: {len(validation_report['errors'])}")
            for error in validation_report['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        if validation_report['warnings']:
            print(f"Warnings: {len(validation_report['warnings'])}")
            for warning in validation_report['warnings'][:3]:  # Show first 3 warnings
                print(f"  - {warning}")
        
        # Save sample data
        output_path = Path("sample_training_data.jsonl")
        generator.save_training_data(examples[:10], output_path, format="jsonl")
        print(f"\nSaved sample training data to {output_path}")
        
        print("\n✅ TrainingDataGenerator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())