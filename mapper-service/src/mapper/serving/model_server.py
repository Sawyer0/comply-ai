"""
Unified model server interface with implementations for vLLM, TGI, and CPU backends.

This consolidates the model serving functionality from the original llama-mapper
implementation with enhanced error handling and fallback mechanisms.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import optional dependencies at module level with proper error handling
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.sampling_params import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    SamplingParams = None
    VLLM_AVAILABLE = False

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False

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
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")

        try:
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

        except Exception as e:
            logger.error("Failed to load vLLM model: %s", str(e))
            raise RuntimeError(f"Failed to load vLLM model: {str(e)}") from e

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using vLLM."""
        if not self.is_loaded:
            await self.load_model()

        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available")

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
                    return str(final_output.outputs[0].text).strip()

            raise RuntimeError("No output generated")

        except Exception as e:
            logger.error("vLLM generation failed: %s", str(e))
            raise RuntimeError(f"vLLM generation failed: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check vLLM server health."""
        try:
            if not self.is_loaded:
                return False

            # Simple generation test
            test_prompt = "Test"
            result = await self.generate_text(test_prompt, max_new_tokens=5)
            return len(result) > 0

        except (RuntimeError, ValueError, OSError) as e:
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
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is not installed. Install with: pip install aiohttp"
            )

        try:
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

        except Exception as e:
            logger.error("Failed to connect to TGI server: %s", str(e))
            raise RuntimeError(f"Failed to connect to TGI server: {str(e)}") from e

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
                    return str(result.get("generated_text", "")).strip()

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
                return response.status == 200

        except (RuntimeError, ValueError, OSError, aiohttp.ClientError) as e:
            logger.error("TGI health check failed: %s", str(e))
            return False

    async def close(self) -> None:
        """Close the TGI client session."""
        if self.session:
            await self.session.close()


class CPUModelServer(ModelServer):
    """CPU-based model server using transformers library."""

    def __init__(
        self,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_path, generation_config)
        self.device = device
        self.model: Any = None
        self.tokenizer: Any = None

    async def load_model(self) -> None:
        """Load the model using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is not installed. Install with: pip install transformers torch"
            )

        try:
            logger.info("Loading model %s with transformers (CPU)", self.model_path)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=self.device,
                trust_remote_code=True,
            )

            self.is_loaded = True
            logger.info("CPU model loaded successfully")

        except Exception as e:
            logger.error("Failed to load CPU model: %s", str(e))
            raise RuntimeError(f"Failed to load CPU model: {str(e)}") from e

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using transformers."""
        if not self.is_loaded:
            await self.load_model()

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not available")

        try:
            # Override generation config with any provided kwargs
            config = self.generation_config
            temperature = kwargs.get("temperature", config.temperature)
            top_p = kwargs.get("top_p", config.top_p)
            max_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error("CPU generation failed: %s", str(e))
            raise RuntimeError(f"CPU generation failed: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check CPU model health."""
        try:
            if not self.is_loaded:
                return False

            # Simple generation test
            test_prompt = "Test"
            result = await self.generate_text(test_prompt, max_new_tokens=5)
            return len(result) > 0

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("CPU model health check failed: %s", str(e))
            return False


def create_model_server(
    backend: str,
    model_path: str,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs: Any,
) -> ModelServer:
    """
    Factory function to create the appropriate model server.

    Args:
        backend: Either 'vllm', 'tgi', or 'cpu'
        model_path: Path to the model
        generation_config: Generation configuration
        **kwargs: Backend-specific arguments

    Returns:
        ModelServer: Configured model server instance
    """
    backend_lower = backend.lower()

    if backend_lower == "vllm":
        return VLLMModelServer(
            model_path=model_path,
            generation_config=generation_config,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )
    if backend_lower == "tgi":
        return TGIModelServer(
            model_path=model_path,
            generation_config=generation_config,
            tgi_endpoint=kwargs.get("tgi_endpoint", "http://localhost:3000"),
        )
    if backend_lower == "cpu":
        return CPUModelServer(
            model_path=model_path,
            generation_config=generation_config,
            device=kwargs.get("device", "cpu"),
        )

    raise ValueError(f"Unsupported backend: {backend}. Use 'vllm', 'tgi', or 'cpu'")
