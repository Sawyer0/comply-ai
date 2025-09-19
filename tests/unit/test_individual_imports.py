#!/usr/bin/env python3
"""Test individual imports to isolate circular dependency."""

print("Testing individual imports...")

try:

    print("✓ ConfigManager import successful")
except Exception as e:
    print(f"✗ ConfigManager import failed: {e}")

try:

    print("✓ MetricsCollector import successful")
except Exception as e:
    print(f"✗ MetricsCollector import failed: {e}")

try:

    print("✓ API models import successful")
except Exception as e:
    print(f"✗ API models import failed: {e}")

try:

    print("✓ FallbackMapper import successful")
except Exception as e:
    print(f"✗ FallbackMapper import failed: {e}")

try:

    print("✓ ConfidenceEvaluator import successful")
except Exception as e:
    print(f"✗ ConfidenceEvaluator import failed: {e}")

print("Individual import test completed!")
