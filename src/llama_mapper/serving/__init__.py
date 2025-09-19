"""
Model serving components for the Llama Mapper.

Note: Heavy, optional dependencies (e.g., torch) are guarded to allow
lightweight import contexts (e.g., OpenAPI export) without full stack.
"""
from .model_server import ModelServer, VLLMModelServer, TGIModelServer, GenerationConfig, create_model_server
from .json_validator import JSONValidator
from .fallback_mapper import FallbackMapper

# Guard heavy imports (torch-dependent) for optional use
try:  # pragma: no cover - import guard
    from .confidence_evaluator import ConfidenceEvaluator, ConfidenceCalibrator
except Exception:  # ImportError or any torch-related error
    ConfidenceEvaluator = None  # type: ignore
    ConfidenceCalibrator = None  # type: ignore

__all__ = [
    "ModelServer",
    "VLLMModelServer", 
    "TGIModelServer",
    "GenerationConfig",
    "create_model_server",
    "JSONValidator",
    "FallbackMapper",
    "ConfidenceEvaluator",
    "ConfidenceCalibrator"
]
