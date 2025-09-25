"""
Model loader for training infrastructure.

Single responsibility: Load and configure models for training.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and configures models for training.

    Single responsibility: Model loading and configuration.
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
        Initialize ModelLoader.

        Args:
            model_name: HuggingFace model identifier
            use_quantization: Whether to use quantization
            quantization_bits: Number of bits for quantization (4 or 8)
            use_fp16: Whether to use FP16 precision
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
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

    def _get_quantization_config(self) -> Optional[Any]:
        """Create quantization configuration."""
        if not self.use_quantization:
            return None

        try:
            from transformers import BitsAndBytesConfig
            import torch

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
        except ImportError:
            logger.warning("BitsAndBytesConfig not available, skipping quantization")
            return None

        return None

    def load_tokenizer(self) -> Any:
        """Load and configure tokenizer."""
        logger.info("Loading tokenizer", model_name=self.model_name)

        try:
            from transformers import AutoTokenizer

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

        except ImportError:
            raise RuntimeError("transformers library not available")

    def load_model(self) -> Any:
        """Load base model with optional quantization."""
        logger.info(
            "Loading base model",
            model_name=self.model_name,
            use_quantization=self.use_quantization,
            quantization_bits=self.quantization_bits if self.use_quantization else None,
            use_fp16=self.use_fp16,
        )

        try:
            from transformers import AutoModelForCausalLM
            import torch

            quantization_config = self._get_quantization_config()
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
            )

            return model

        except ImportError:
            raise RuntimeError("transformers library not available")

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load both model and tokenizer together."""
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

    def prepare_model_for_lora(self, model: Any, lora_config: Any) -> Any:
        """Prepare model for LoRA fine-tuning."""
        logger.info(
            "Preparing model for LoRA fine-tuning",
            lora_r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
        )

        try:
            from peft import get_peft_model

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

        except ImportError:
            logger.warning("PEFT not available, returning base model")
            return model

    @classmethod
    def create_lora_config(
        cls,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
    ) -> Any:
        """Create LoRA configuration."""
        if target_modules is None:
            # Default target modules for Llama models
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        try:
            from peft import LoraConfig

            return LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=task_type,
            )
        except ImportError:
            logger.warning("PEFT not available, returning None")
            return None
