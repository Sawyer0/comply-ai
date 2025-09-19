#!/usr/bin/env python3
"""
Comprehensive test script for the complete dataset preparation pipeline.
Tests TrainingDataGenerator, SyntheticDataGenerator, and DatasetValidator.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_mapper.training.data_generator import (
    TrainingDataGenerator, 
    SyntheticDataGenerator, 
    DatasetValidator
)
from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.data.detectors import DetectorConfigLoader


def main():
    """Test the complete dataset preparation pipeline."""
    print("Testing Complete Dataset Preparation Pipeline...")
    print("=" * 60)
    
    try:
        # Initialize components
        taxonomy_loader = TaxonomyLoader("pillars-detectors/taxonomy.yaml")
        detector_loader = DetectorConfigLoader("pillars-detectors")
        
        # Test 1: TrainingDataGenerator
        print("\n1. Testing TrainingDataGenerator...")
        training_generator = TrainingDataGenerator(
            detector_loader=detector_loader,
            taxonomy_loader=taxonomy_loader,
            random_seed=42
        )
        
        training_generator.load_data()
        training_examples = training_generator.generate_training_examples(
            examples_per_mapping=2,
            include_multi_label=True,
            include_variations=True
        )
        
        print(f"   ✅ Generated {len(training_examples)} training examples")
        
        # Test 2: SyntheticDataGenerator
        print("\n2. Testing SyntheticDataGenerator...")
        synthetic_generator = SyntheticDataGenerator(
            taxonomy_loader=taxonomy_loader,
            random_seed=42
        )
        
        synthetic_generator.load_taxonomy()
        
        # Generate different types of synthetic data
        pii_examples = synthetic_generator.generate_synthetic_pii_examples(num_examples=30)
        jailbreak_examples = synthetic_generator.generate_synthetic_jailbreak_examples(num_examples=15)
        injection_examples = synthetic_generator.generate_synthetic_prompt_injection_examples(num_examples=20)
        
        print(f"   ✅ Generated {len(pii_examples)} PII examples")
        print(f"   ✅ Generated {len(jailbreak_examples)} jailbreak examples")
        print(f"   ✅ Generated {len(injection_examples)} prompt injection examples")
        
        # Generate balanced synthetic set
        balanced_examples = synthetic_generator.generate_balanced_training_set(
            target_examples_per_category=10,
            include_pii=True,
            include_jailbreak=True,
            include_prompt_injection=True
        )
        
        print(f"   ✅ Generated {len(balanced_examples)} balanced synthetic examples")
        
        # Test 3: DatasetValidator
        print("\n3. Testing DatasetValidator...")
        validator = DatasetValidator(
            taxonomy_loader=taxonomy_loader,
            detector_loader=detector_loader,
            schema_path="pillars-detectors/schema.json"
        )
        
        validator.load_dependencies()
        
        # Combine all examples for validation
        all_examples = training_examples + balanced_examples
        print(f"   Validating {len(all_examples)} total examples...")
        
        # Comprehensive validation
        validation_report = validator.validate_training_dataset(all_examples)
        
        print(f"   ✅ Validation completed with overall score: {validation_report['overall_score']:.2f}/100")
        
        # Test 4: Show detailed results
        print("\n4. Detailed Results...")
        print("-" * 40)
        
        # Training data statistics
        training_stats = training_generator.get_generation_statistics(training_examples)
        print(f"Training Examples Statistics:")
        print(f"  - Total examples: {training_stats['total_examples']}")
        print(f"  - Detectors covered: {len(training_stats['detectors'])}")
        print(f"  - Canonical labels: {len(training_stats['canonical_labels'])}")
        print(f"  - Example types: {training_stats['example_types']}")
        
        # Synthetic data statistics
        synthetic_stats = synthetic_generator.get_synthetic_statistics(balanced_examples)
        print(f"\nSynthetic Examples Statistics:")
        print(f"  - Total examples: {synthetic_stats['total_examples']}")
        print(f"  - Example types: {synthetic_stats['example_types']}")
        print(f"  - Canonical labels: {len(synthetic_stats['canonical_labels'])}")
        
        # Validation results summary
        print(f"\nValidation Results Summary:")
        format_val = validation_report["format_validation"]
        taxonomy_val = validation_report["taxonomy_validation"]
        coverage = validation_report["coverage_analysis"]["coverage_statistics"]
        
        print(f"  - Format validation: {format_val['valid_examples']}/{format_val['valid_examples'] + format_val['invalid_examples']} valid")
        print(f"  - Taxonomy validation: {taxonomy_val['valid_labels']} valid labels, {taxonomy_val['invalid_labels']} invalid")
        print(f"  - Detector coverage: {coverage['detector_coverage_percentage']:.1f}%")
        print(f"  - Taxonomy coverage: {coverage['taxonomy_coverage_percentage']:.1f}%")
        
        # Show recommendations
        recommendations = validation_report["recommendations"]
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n✅ No recommendations - dataset quality is good!")
        
        # Test 5: Save sample outputs
        print("\n5. Saving Sample Outputs...")
        
        # Save sample training data
        with open("sample_complete_training_data.jsonl", 'w', encoding='utf-8') as f:
            for example in all_examples[:50]:  # Save first 50 examples
                f.write(example.to_json() + '\n')
        print("   ✅ Saved sample training data to sample_complete_training_data.jsonl")
        
        # Save validation report
        validator.export_validation_report(
            validation_report, 
            "validation_report.json", 
            format="json"
        )
        print("   ✅ Saved validation report to validation_report.json")
        
        validator.export_validation_report(
            validation_report, 
            "validation_report.txt", 
            format="txt"
        )
        print("   ✅ Saved validation report to validation_report.txt")
        
        # Test 6: Individual example validation
        print("\n6. Testing Individual Example Validation...")
        if all_examples:
            sample_example = all_examples[0]
            single_validation = validator.validate_single_example(sample_example)
            
            if single_validation["valid"]:
                print("   ✅ Sample example validation passed")
            else:
                print(f"   ❌ Sample example validation failed: {single_validation['errors']}")
        
        print("\n" + "=" * 60)
        print("🎉 Complete Dataset Preparation Pipeline Test PASSED!")
        print("=" * 60)
        
        # Final summary
        print(f"\nFinal Summary:")
        print(f"  📊 Total examples generated: {len(all_examples)}")
        print(f"  🎯 Training examples: {len(training_examples)}")
        print(f"  🔧 Synthetic examples: {len(balanced_examples)}")
        print(f"  📈 Overall validation score: {validation_report['overall_score']:.2f}/100")
        print(f"  🏷️  Taxonomy labels covered: {coverage['covered_taxonomy_labels']}/{coverage['total_taxonomy_labels']}")
        print(f"  🔍 Detectors covered: {coverage['covered_detectors']}/{coverage['total_detectors']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())