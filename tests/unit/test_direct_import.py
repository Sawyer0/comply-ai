#!/usr/bin/env python3
"""Test direct imports without going through __init__.py files."""

import sys
sys.path.insert(0, 'src')

print("Testing direct imports...")

try:
    # Import directly without going through serving.__init__
    import llama_mapper.serving.fallback_mapper as fallback_module
    FallbackMapper = fallback_module.FallbackMapper
    print("✓ FallbackMapper direct import successful")
    
    # Test initialization
    mapper = FallbackMapper("nonexistent-path")
    print(f"✓ FallbackMapper initialized with {len(mapper.detector_mappings)} mappings")
except Exception as e:
    print(f"✗ FallbackMapper direct import failed: {e}")

try:
    # Import confidence evaluator directly
    import llama_mapper.serving.confidence_evaluator as conf_module
    ConfidenceEvaluator = conf_module.ConfidenceEvaluator
    print("✓ ConfidenceEvaluator direct import successful")
    
    # Test initialization
    evaluator = ConfidenceEvaluator()
    print(f"✓ ConfidenceEvaluator initialized with threshold: {evaluator.threshold}")
except Exception as e:
    print(f"✗ ConfidenceEvaluator direct import failed: {e}")

print("Direct import test completed!")