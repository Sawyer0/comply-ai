"""
Analysis Service

This package provides comprehensive analysis capabilities including
pattern recognition, risk scoring, compliance intelligence, and more.
"""

__version__ = "1.0.0"

# Initialize shared components integration
from .shared_integration import initialize_shared_components

# Initialize shared components when the module is imported
try:
    _shared_components = initialize_shared_components()
except Exception as e:
    print(f"Warning: Failed to initialize shared components: {e}")
    _shared_components = None
