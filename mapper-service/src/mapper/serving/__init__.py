"""
Model serving components for the Mapper Service.

This module consolidates model serving functionality with support for:
- vLLM for GPU deployment
- TGI (Text Generation Inference) for CPU deployment
- CPU fallback for lightweight deployment
- Confidence evaluation and calibration
"""

from .model_server import (
    GenerationConfig,
    ModelServer,
    TGIModelServer,
    VLLMModelServer,
    CPUModelServer,
    create_model_server,
)
from .fallback_mapper import FallbackMapper
from .confidence_evaluator import ConfidenceEvaluator, ConfidenceCalibrator

__all__ = [
    "ModelServer",
    "VLLMModelServer",
    "TGIModelServer",
    "CPUModelServer",
    "GenerationConfig",
    "create_model_server",
    "FallbackMapper",
    "ConfidenceEvaluator",
    "ConfidenceCalibrator",
]
