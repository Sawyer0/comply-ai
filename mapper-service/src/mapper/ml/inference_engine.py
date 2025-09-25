"""
Inference Engine for Mapper Service

Single Responsibility: Handle ML model inference requests and response processing.
Coordinates with ModelManager for model access and handles inference optimization.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

import structlog

from .model_manager import ModelManager, ModelConfig, ModelBackend

logger = structlog.get_logger(__name__)


@dataclass
class InferenceRequest:
    """Inference request data."""

    model_id: str
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    stop_sequences: Optional[List[str]] = None
    return_full_text: bool = False


@dataclass
class InferenceResponse:
    """Inference response data."""

    model_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    inference_time: float
    success: bool
    error: Optional[str] = None


class InferenceEngine:
    """
    Inference Engine for ML model inference.

    Single Responsibility: Execute inference requests and optimize performance.

    This class handles:
    - Inference request processing
    - Response generation and formatting
    - Performance optimization (batching, caching)
    - Error handling and fallbacks
    - Inference metrics collection
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logger.bind(component="inference_engine")

        # Inference optimization settings
        self.batch_size = 1  # Start with single inference
        self.max_batch_wait_time = 0.1  # 100ms max wait for batching
        self.inference_cache: Dict[str, InferenceResponse] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute inference request.

        Single Responsibility: Process a single inference request.
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.inference_cache:
                cached_response = self.inference_cache[cache_key]
                self.logger.debug(
                    "Cache hit", model_id=request.model_id, cache_key=cache_key
                )
                return cached_response

            # Get model
            model_data = self.model_manager.get_model(request.model_id)
            if not model_data:
                return InferenceResponse(
                    model_id=request.model_id,
                    generated_text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    inference_time=0.0,
                    success=False,
                    error=f"Model {request.model_id} not loaded",
                )

            # Execute inference based on backend
            if model_data["backend"] == ModelBackend.TRANSFORMERS:
                response = await self._infer_transformers(request, model_data)
            elif model_data["backend"] == ModelBackend.VLLM:
                response = await self._infer_vllm(request, model_data)
            else:
                raise ValueError(f"Unsupported backend: {model_data['backend']}")

            # Update metrics
            inference_time = time.time() - start_time
            response.inference_time = inference_time

            metrics = self.model_manager.get_model_metrics(request.model_id)
            if metrics:
                metrics.update_inference(inference_time, response.success)

            # Cache successful responses
            if response.success:
                self.inference_cache[cache_key] = response

            return response

        except (RuntimeError, ValueError, ImportError, OSError) as e:
            inference_time = time.time() - start_time
            self.logger.error(
                "Inference failed",
                model_id=request.model_id,
                error=str(e),
                inference_time=inference_time,
            )

            # Update error metrics
            metrics = self.model_manager.get_model_metrics(request.model_id)
            if metrics:
                metrics.update_inference(inference_time, success=False)

            return InferenceResponse(
                model_id=request.model_id,
                generated_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                inference_time=inference_time,
                success=False,
                error=str(e),
            )

    async def _infer_transformers(
        self, request: InferenceRequest, model_data: Dict[str, Any]
    ) -> InferenceResponse:
        """Execute inference using Transformers backend."""

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        if request.return_full_text:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Only return the new tokens
            new_tokens = outputs[0][prompt_tokens:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        completion_tokens = outputs[0].shape[0] - prompt_tokens
        total_tokens = outputs[0].shape[0]

        return InferenceResponse(
            model_id=request.model_id,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            inference_time=0.0,  # Will be set by caller
            success=True,
        )

    async def _infer_vllm(
        self, request: InferenceRequest, model_data: Dict[str, Any]
    ) -> InferenceResponse:
        """Execute inference using vLLM backend."""

        try:
            from vllm import SamplingParams

            model = model_data["model"]

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
                stop=request.stop_sequences,
            )

            # Generate
            outputs = model.generate([request.prompt], sampling_params)

            if not outputs:
                raise ValueError("No output generated")

            output = outputs[0]
            generated_text = output.outputs[0].text

            # vLLM doesn't provide token counts directly, estimate them
            prompt_tokens = len(request.prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(generated_text.split()) * 1.3
            total_tokens = prompt_tokens + completion_tokens

            return InferenceResponse(
                model_id=request.model_id,
                generated_text=generated_text,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                inference_time=0.0,  # Will be set by caller
                success=True,
            )

        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

    async def batch_infer(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """
        Execute batch inference for multiple requests.

        Single Responsibility: Optimize inference through batching.
        """
        if not requests:
            return []

        # Group requests by model_id
        model_groups: Dict[str, List[InferenceRequest]] = {}
        for request in requests:
            if request.model_id not in model_groups:
                model_groups[request.model_id] = []
            model_groups[request.model_id].append(request)

        # Process each model group
        all_responses = []
        for model_id, model_requests in model_groups.items():
            model_responses = await self._batch_infer_single_model(model_requests)
            all_responses.extend(model_responses)

        return all_responses

    async def _batch_infer_single_model(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Execute batch inference for a single model."""

        # For now, process sequentially
        # TODO: Implement true batching for supported backends
        responses = []
        for request in requests:
            response = await self.infer(request)
            responses.append(response)

        return responses

    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for inference request."""

        cache_data = {
            "model_id": request.model_id,
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.do_sample,
            "stop_sequences": request.stop_sequences,
        }

        import hashlib
        import json

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear inference cache."""
        self.inference_cache.clear()
        self.logger.info("Inference cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.inference_cache),
            "cache_ttl": self.cache_ttl,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on inference engine."""

        return {
            "healthy": True,
            "cache_size": len(self.inference_cache),
            "batch_size": self.batch_size,
            "max_batch_wait_time": self.max_batch_wait_time,
        }
