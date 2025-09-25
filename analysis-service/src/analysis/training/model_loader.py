"""
Model loading utilities for Analysis Service.

Handles loading of trained Phi-3-Mini models for inference.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..shared_integration import get_shared_logger

logger = get_shared_logger(__name__)


class ModelLoader:
    """Utility class for loading trained models."""

    def __init__(self):
        self.logger = logger.bind(component="model_loader")
        self._loaded_models: Dict[str, Tuple[Any, Any]] = {}

    def load_model(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        cache_model: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Load a trained model and tokenizer.

        Args:
            model_path: Path to the saved model
            device_map: Device mapping strategy
            torch_dtype: Torch data type for model
            trust_remote_code: Whether to trust remote code
            cache_model: Whether to cache the loaded model

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache first
        if cache_model and model_path in self._loaded_models:
            self.logger.info("Loading model from cache", model_path=model_path)
            return self._loaded_models[model_path]

        try:
            self.logger.info("Loading model from disk", model_path=model_path)

            # Verify model path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )

            # Set to evaluation mode
            model.eval()

            # Cache if requested
            if cache_model:
                self._loaded_models[model_path] = (model, tokenizer)

            self.logger.info(
                "Model loaded successfully",
                model_path=model_path,
                model_type=type(model).__name__,
                vocab_size=tokenizer.vocab_size,
            )

            return model, tokenizer

        except Exception as e:
            self.logger.error(
                "Failed to load model", model_path=model_path, error=str(e)
            )
            raise

    def load_base_model(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ) -> Tuple[Any, Any]:
        """
        Load the base Phi-3-Mini model.

        Args:
            model_name: Name of the base model
            device_map: Device mapping strategy
            torch_dtype: Torch data type for model

        Returns:
            Tuple of (model, tokenizer)
        """
        return self.load_model(
            model_path=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_model=True,
        )

    def unload_model(self, model_path: str) -> None:
        """
        Unload a cached model to free memory.

        Args:
            model_path: Path of the model to unload
        """
        if model_path in self._loaded_models:
            del self._loaded_models[model_path]

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Model unloaded from cache", model_path=model_path)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._loaded_models.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Model cache cleared")

    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved model.

        Args:
            model_path: Path to the model

        Returns:
            Model information dictionary or None if not found
        """
        try:
            # Check for metadata file
            metadata_path = os.path.join(model_path, "version_metadata.json")
            if os.path.exists(metadata_path):
                import json

                with open(metadata_path, "r") as f:
                    return json.load(f)

            # Check for config file
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json

                with open(config_path, "r") as f:
                    config = json.load(f)

                return {
                    "model_type": config.get("model_type"),
                    "vocab_size": config.get("vocab_size"),
                    "hidden_size": config.get("hidden_size"),
                    "num_attention_heads": config.get("num_attention_heads"),
                    "num_hidden_layers": config.get("num_hidden_layers"),
                }

            return None

        except Exception as e:
            self.logger.warning(
                "Failed to get model info", model_path=model_path, error=str(e)
            )
            return None

    def validate_model(self, model_path: str) -> bool:
        """
        Validate that a model can be loaded successfully.

        Args:
            model_path: Path to the model

        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Try to load the model
            model, tokenizer = self.load_model(
                model_path, cache_model=False  # Don't cache validation loads
            )

            # Basic validation - try a simple forward pass
            test_input = tokenizer(
                "Test input for validation",
                return_tensors="pt",
                max_length=100,
                truncation=True,
            )

            with torch.no_grad():
                outputs = model(**test_input)

            # Check output shape
            if outputs.logits.shape[-1] != tokenizer.vocab_size:
                self.logger.warning(
                    "Model output dimension mismatch",
                    expected=tokenizer.vocab_size,
                    actual=outputs.logits.shape[-1],
                )
                return False

            self.logger.info("Model validation successful", model_path=model_path)
            return True

        except Exception as e:
            self.logger.error(
                "Model validation failed", model_path=model_path, error=str(e)
            )
            return False

    def list_cached_models(self) -> List[str]:
        """Get list of currently cached models."""
        return list(self._loaded_models.keys())

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {}

        if torch.cuda.is_available():
            memory_info["cuda_allocated"] = torch.cuda.memory_allocated()
            memory_info["cuda_reserved"] = torch.cuda.memory_reserved()
            memory_info["cuda_max_allocated"] = torch.cuda.max_memory_allocated()

        memory_info["cached_models"] = len(self._loaded_models)

        return memory_info
