#!/usr/bin/env python3
"""Test individual imports to isolate circular dependency."""

print("Testing individual imports...")

try:
    from src.llama_mapper.config.manager import ConfigManager
    print("✓ ConfigManager import successful")
except Exception as e:
    print(f"✗ ConfigManager import failed: {e}")

try:
    from src.llama_mapper.monitoring.metrics_collector import get_metrics_collector
    print("✓ MetricsCollector import successful")
except Exception as e:
    print(f"✗ MetricsCollector import failed: {e}")

try:
    from src.llama_mapper.api.models import MappingResponse, Provenance
    print("✓ API models import successful")
except Exception as e:
    print(f"✗ API models import failed: {e}")

try:
    from src.llama_mapper.serving.fallback_mapper import FallbackMapper
    print("✓ FallbackMapper import successful")
except Exception as e:
    print(f"✗ FallbackMapper import failed: {e}")

try:
    from src.llama_mapper.serving.confidence_evaluator import ConfidenceEvaluator
    print("✓ ConfidenceEvaluator import successful")
except Exception as e:
    print(f"✗ ConfidenceEvaluator import failed: {e}")

print("Individual import test completed!")