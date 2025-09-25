"""
ML Infrastructure Module

This module provides ML model serving infrastructure for the Analysis Service,
including Phi-3 model serving, embeddings management, and model optimization.

Components:
- Model Server (vLLM/TGI/CPU backends)
- Embeddings Management
- Model Optimization and Caching
- Fallback Mechanisms
"""

from .model_server import ModelServer, ModelBackend
from .embeddings import EmbeddingsManager
from .optimization import ModelOptimizer
from .fallback import MLFallbackManager

__all__ = [
    "ModelServer",
    "ModelBackend",
    "EmbeddingsManager",
    "ModelOptimizer",
    "MLFallbackManager",
]
