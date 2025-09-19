#!/usr/bin/env python3
"""Simple test."""

import sys
sys.path.insert(0, 'src')

try:
    print("Importing fallback_mapper module...")
    import llama_mapper.serving.fallback_mapper
    print("✓ Module imported successfully")
    
    print("Getting FallbackMapper class...")
    FallbackMapper = llama_mapper.serving.fallback_mapper.FallbackMapper
    print("✓ Class retrieved successfully")
    
    print("Creating instance...")
    mapper = FallbackMapper("nonexistent")
    print("✓ Instance created successfully")
    
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()