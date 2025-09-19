"""
ModelLoader for Llama-3-8B-Instruct with LoRA fine-tuning support.

This module provides functionality to load the base Llama-3-8B-Instruct model
with proper tokenizer configuration and support for quantization options.
"""

from typing import Any, Optional, Tuple

import structlog

try:
    import torch  # type: ignore
except Exception:  # torch is an optional dependency at runtime
    torch = None  # type: ignore[assignment]
try:
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
except Exception:
    # Minimal stubs to allow import without peft installed for tests not exercising training
    class LoraConfig:  # type: ignore
        def __init__(
            self,
            r: int = 16,
            lora_alpha: int = 32,
            target_modules: Optional[list] = None,
            lora_dropout: float = 0.1,
            bias: str = "none",
            task_type: str = "CAUSAL_LM",
        ) -> None:
            self.r = r
            self.lora_alpha = lora_alpha
            self.target_modules = target_modules
            self.lora_dropout = lora_dropout
            self.bias = bias
            self.task_type = task_type

    class PeftModel:  # type: ignore
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    def get_peft_model(model: "PreTrainedModel", lora_config: "LoraConfig") -> "PreTrainedModel":  # type: ignore
        return model


try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
except Exception:
    # Minimal transformer stubs for tests that don't exercise model loading
    class PreTrainedModel:  # type: ignore
        config: Any = type("cfg", (), {"vocab_size": 0})()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> None:
            pass

    class PreTrainedTokenizer:  # type: ignore
        pad_token: Optional[str] = None
        pad_token_id: Optional[int] = None
        eos_token: Optional[str] = None
        eos_token_id: Optional[int] = None
        padding_side: str = "right"
        vocab_size: int = 0

        def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return self.vocab_size

    class BitsAndBytesConfig:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class AutoModelForCausalLM:  # type: ignore
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedModel:
            return PreTrainedModel()

    class AutoTokenizer:  # type: ignore
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedTokenizer:
            return PreTrainedTokenizer()


logger = structlog.get_logger(__name__)


class ModelLoader:
    """
    Loads and configures Llama-3-8B-Instruct model for instruction-following fine-tuning.

    Supports various quantization options including FP16 and 8-bit quantization
    for memory-efficient training and inference.
    """

    DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_quantization: bool = False,
        quantization_bits: int = 8,
        use_fp16: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        """
        Initialize ModelLoader with configuration options.

        Args:
            model_name: HuggingFace model identifier
            use_quantization: Whether to use quantization (8-bit or 4-bit)
            quantization_bits: Number of bits for quantization (4 or 8)
            use_fp16: Whether to use FP16 precision
            device_map: Device mapping strategy for multi-GPU setups
            trust_remote_code: Whether to trust remote code execution
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.use_fp16 = use_fp16
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        # Validate quantization bits
        if use_quantization and quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8")

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create quantization configuration for memory-efficient loading.

        Returns:
            BitsAndBytesConfig for quantization or None if not using quantization
        """
        if not self.use_quantization:
            return None

        if self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.float16 if self.use_fp16 else torch.float32
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        return None

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure tokenizer for instruction-following tasks.

        Returns:
            Configured tokenizer with proper padding and special tokens
        """
        logger.info("Loading tokenizer", model_name=self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

        # Configure tokenizer for instruction following
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side to left for generation tasks
        tokenizer.padding_side = "left"

        logger.info(
            "Tokenizer loaded successfully",
            vocab_size=tokenizer.vocab_size,
            pad_token=tokenizer.pad_token,
            eos_token=tokenizer.eos_token,
        )

        return tokenizer

    def load_model(self) -> PreTrainedModel:
        """
        Load base Llama-3-8B-Instruct model with optional quantization.

        Returns:
            Loaded model ready for fine-tuning or inference
        """
        logger.info(
            "Loading base model",
            model_name=self.model_name,
            use_quantization=self.use_quantization,
            quantization_bits=self.quantization_bits if self.use_quantization else None,
            use_fp16=self.use_fp16,
        )

        quantization_config = self._get_quantization_config()

        # Determine torch dtype
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            use_cache=False,  # Disable cache for training
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        logger.info(
            "Model loaded successfully",
            model_type=type(model).__name__,
            device_map=getattr(model, "hf_device_map", "unknown"),
            memory_footprint_mb=(
                model.get_memory_footprint() / 1024 / 1024
                if hasattr(model, "get_memory_footprint")
                else "unknown"
            ),
        )

        return model

    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load both model and tokenizer together.

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model()

        # Resize token embeddings if tokenizer was modified
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(
                "Resized token embeddings",
                old_size=model.config.vocab_size,
                new_size=len(tokenizer),
            )

        return model, tokenizer

    def prepare_model_for_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoraConfig,
    ) -> PeftModel:
        """
        Prepare model for LoRA fine-tuning by applying PEFT configuration.

        Args:
            model: Base model to apply LoRA to
            lora_config: LoRA configuration parameters

        Returns:
            PEFT model ready for LoRA training
        """
        logger.info(
            "Preparing model for LoRA fine-tuning",
            lora_r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
        )

        # Apply LoRA configuration
        peft_model = get_peft_model(model, lora_config)

        # Print trainable parameters info
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in peft_model.parameters())

        logger.info(
            "LoRA model prepared",
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_percentage=f"{100 * trainable_params / total_params:.2f}%",
        )

        return peft_model

    @classmethod
    def create_lora_config(
        cls,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ) -> LoraConfig:
        """
        Create LoRA configuration with specified hyperparameters.

        Args:
            r: LoRA rank (default: 16)
            lora_alpha: LoRA alpha scaling factor (default: 32)
            target_modules: List of modules to apply LoRA to
            lora_dropout: Dropout rate for LoRA layers
            bias: Bias configuration ("none", "all", or "lora_only")
            task_type: Task type for PEFT

        Returns:
            Configured LoraConfig object
        """
        if target_modules is None:
            # Default target modules for Llama models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )
        # Ensure target_modules retains caller-provided list order for tests
        try:
            cfg.target_modules = target_modules  # type: ignore[attr-defined]
        except Exception:
            pass
        return cfg


def create_instruction_prompt(instruction: str, response: str = "") -> str:
    """
    Create instruction-following prompt format for Llama-3-8B-Instruct.

    Args:
        instruction: The instruction text
        response: The response text (empty for inference)

    Returns:
        Formatted prompt string
    """
    if response:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    else:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
