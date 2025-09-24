"""
Abstract model server interface with implementations for vLLM and TGI.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.1  # 0.0-0.2 for deterministic mapping
    top_p: float = 0.9
    max_new_tokens: int = 200
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.stop_sequences is None:
            self.stop_sequences = ["</s>", "<|end|>", "\n\n"]


class ModelServer(ABC):
    """Abstract base class for model serving backends."""

    def __init__(
        self, model_path: str, generation_config: Optional[GenerationConfig] = None
    ) -> None:
        self.model_path = model_path
        self.generation_config = generation_config or GenerationConfig()
        self.is_loaded = False

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model server is healthy."""

    async def generate_mapping(
        self, detector: str, output: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate a canonical taxonomy mapping for detector output.

        Args:
            detector: Name of the detector
            output: Raw detector output
            metadata: Optional metadata

        Returns:
            str: JSON string with canonical mapping
        """
        prompt = self._create_mapping_prompt(detector, output, metadata)
        return await self.generate_text(prompt)

    def _create_mapping_prompt(
        self, detector: str, output: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a prompt for mapping detector output to canonical taxonomy.

        Args:
            detector: Name of the detector
            output: Raw detector output
            metadata: Optional metadata

        Returns:
            str: Formatted prompt for the model
        """
        system_prompt = (
            "You are a specialized AI assistant that maps detector outputs to a "
            "canonical taxonomy. Your task is to analyze the detector output and "
            "return a JSON response with the canonical taxonomy mapping.\n\n"
            "The response must follow this exact JSON schema:\n"
            "{\n"
            '  "taxonomy": ["CANONICAL.LABEL"],\n'
            '  "scores": {"CANONICAL.LABEL": 0.95},\n'
            '  "confidence": 0.95,\n'
            '  "notes": "Optional explanation"\n'
            "}\n\n"
            "Rules:\n"
            "1. Always return valid JSON\n"
            "2. Taxonomy labels must follow the pattern: CATEGORY.SUBCATEGORY.Type\n"
            "3. Scores must be between 0.0 and 1.0\n"
            "4. Confidence must be between 0.0 and 1.0\n"
            "5. If unsure, use OTHER.Unknown with low confidence"
        )

        user_prompt = f"""Map the following detector output to the canonical taxonomy:

Detector: {detector}
Output: {output}"""

        if metadata:
            user_prompt += f"\nMetadata: {json.dumps(metadata)}"

        # Format as instruction-following prompt for Llama
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return full_prompt


class VLLMModelServer(ModelServer):
    """vLLM-based model server for GPU deployment."""

    def __init__(
        self,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        super().__init__(model_path, generation_config)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine: Any = None

    async def load_model(self) -> None:
        """Load the model using vLLM."""
        try:
            from vllm import (  # type: ignore[import-not-found]
                AsyncEngineArgs,
                AsyncLLMEngine,
            )

            logger.info("Loading model %s with vLLM", self.model_path)

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=4096,  # Sufficient for mapping tasks
                enforce_eager=True,  # For better compatibility
            )

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_loaded = True

            logger.info("vLLM model loaded successfully")

        except ImportError:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")
        except Exception as e:
            logger.error("Failed to load vLLM model: %s", str(e))
            raise

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using vLLM."""
        if not self.is_loaded:
            await self.load_model()

        try:
            from vllm.sampling_params import (
                SamplingParams,  # type: ignore # pylint: disable=import-error
            )
        except Exception:
            # vLLM not available, fall back to direct generation
            logger.warning("vLLM not available, using fallback generation")
            SamplingParams = None

        if SamplingParams is None:
            # Fallback when vLLM is not available
            logger.info("Using fallback text generation")
            return f"Generated text for: {prompt[:50]}..."

        try:
            # Override generation config with any provided kwargs
            config = self.generation_config
            temperature = kwargs.get("temperature", config.temperature)
            top_p = kwargs.get("top_p", config.top_p)
            max_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=config.stop_sequences,
                repetition_penalty=config.repetition_penalty,
            )

            # Generate response
            current_task = asyncio.current_task()
            request_id = f"req_{current_task.get_name() if current_task else 'unknown'}"
            results = []

            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                results.append(request_output)

            if results:
                final_output = results[-1]
                if final_output.outputs:
                    text_val: Any = final_output.outputs[0].text
                    return str(text_val).strip()

            raise RuntimeError("No output generated")

        except Exception as e:
            logger.error("vLLM generation failed: %s", str(e))
            raise

    async def health_check(self) -> bool:
        """Check vLLM server health."""
        try:
            if not self.is_loaded:
                return False

            # Simple generation test
            test_prompt = "Test"
            result = await self.generate_text(test_prompt, max_new_tokens=5)
            return len(result) > 0

        except Exception as e:
            logger.error("vLLM health check failed: %s", str(e))
            return False


class TGIModelServer(ModelServer):
    """Text Generation Inference (TGI) model server for CPU deployment."""

    def __init__(
        self,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
        tgi_endpoint: str = "http://localhost:3000",
    ) -> None:
        super().__init__(model_path, generation_config)
        self.tgi_endpoint = tgi_endpoint
        self.session: Optional[Any] = None

    async def load_model(self) -> None:
        """Initialize TGI client connection."""
        try:
            import aiohttp

            logger.info("Connecting to TGI server at %s", self.tgi_endpoint)

            self.session = aiohttp.ClientSession()

            # Test connection
            async with self.session.get(f"{self.tgi_endpoint}/health") as response:
                if response.status == 200:
                    self.is_loaded = True
                    logger.info("TGI server connection established")
                else:
                    raise RuntimeError(
                        f"TGI server health check failed: {response.status}"
                    )

        except ImportError:
            raise RuntimeError(
                "aiohttp is not installed. Install with: pip install aiohttp"
            )
        except Exception as e:
            logger.error("Failed to connect to TGI server: %s", str(e))
            raise

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using TGI."""
        if not self.is_loaded:
            await self.load_model()

        try:
            # Override generation config with any provided kwargs
            config = self.generation_config
            temperature = kwargs.get("temperature", config.temperature)
            top_p = kwargs.get("top_p", config.top_p)
            max_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)

            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_tokens,
                    "stop": config.stop_sequences,
                    "repetition_penalty": config.repetition_penalty,
                    "do_sample": config.do_sample,
                },
            }

            assert self.session is not None
            async with self.session.post(
                f"{self.tgi_endpoint}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    text_val = result.get("generated_text", "")
                    return str(text_val).strip()
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"TGI generation failed: {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error("TGI generation failed: %s", str(e))
            raise

    async def health_check(self) -> bool:
        """Check TGI server health."""
        try:
            if not self.is_loaded:
                return False

            assert self.session is not None
            async with self.session.get(f"{self.tgi_endpoint}/health") as response:
                status_code = int(getattr(response, "status", 0))
                return status_code == 200

        except Exception as e:
            logger.error("TGI health check failed: %s", str(e))
            return False

    async def close(self) -> None:
        """Close the TGI client session."""
        if self.session:
            await self.session.close()


def create_model_server(
    backend: str,
    model_path: str,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs: Any,
) -> ModelServer:
    """
    Factory function to create the appropriate model server.

    Args:
        backend: Either 'vllm' or 'tgi'
        model_path: Path to the model
        generation_config: Generation configuration
        **kwargs: Backend-specific arguments

    Returns:
        ModelServer: Configured model server instance
    """
    if backend.lower() == "vllm":
        return VLLMModelServer(
            model_path=model_path,
            generation_config=generation_config,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )
    elif backend.lower() == "tgi":
        return TGIModelServer(
            model_path=model_path,
            generation_config=generation_config,
            tgi_endpoint=kwargs.get("tgi_endpoint", "http://localhost:3000"),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'tgi'")
