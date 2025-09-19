#!/usr/bin/env python3
"""Simple test to verify imports work correctly."""

try:
    from src.llama_mapper.serving.confidence_evaluator import ConfidenceEvaluator
    print("✓ ConfidenceEvaluator import successful")
except Exception as e:
    print(f"✗ ConfidenceEvaluator import failed: {e}")

try:
    from src.llama_mapper.serving.fallback_mapper import FallbackMapper
    print("✓ FallbackMapper import successful")
except Exception as e:
    print(f"✗ FallbackMapper import failed: {e}")

try:
    # Test basic initialization
    evaluator = ConfidenceEvaluator()
    print(f"✓ ConfidenceEvaluator initialized with threshold: {evaluator.threshold}")
except Exception as e:
    print(f"✗ ConfidenceEvaluator initialization failed: {e}")

try:
    # Test basic fallback mapper (will fail gracefully if no detector configs)
    mapper = FallbackMapper("nonexistent-path")
    print(f"✓ FallbackMapper initialized with {len(mapper.detector_mappings)} mappings")
except Exception as e:
    print(f"✗ FallbackMapper initialization failed: {e}")

print("Import test completed!")