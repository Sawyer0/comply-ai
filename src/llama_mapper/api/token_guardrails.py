"""
Token Guardrails for API Endpoints

Implements input token limits and guardrails to prevent excessive token usage
and ensure consistent performance across the API.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import structlog

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Model types for token guardrails."""

    MAPPER = "mapper"
    ANALYST = "analyst"


@dataclass
class TokenGuardrailConfig:
    """Configuration for token guardrails."""

    # Input token limits
    mapper_max_input_tokens: int = 1024
    analyst_max_input_tokens: int = 2048

    # Fallback strategies
    mapper_fallback_strategy: str = "rules_based"  # rules_based, chunked, reject
    analyst_fallback_strategy: str = "chunked_analysis"  # chunked_analysis, reject

    # Logging configuration
    log_token_counts: bool = True
    log_content: bool = False  # Only log counts, not actual content

    # Performance thresholds
    warning_threshold_ratio: float = 0.8  # Warn at 80% of limit
    error_threshold_ratio: float = 1.0  # Error at 100% of limit


class TokenGuardrailError(Exception):
    """Exception raised when token limits are exceeded."""

    def __init__(self, message: str, token_count: int, limit: int, model_type: str):
        super().__init__(message)
        self.token_count = token_count
        self.limit = limit
        self.model_type = model_type


class TokenGuardrail:
    """Token guardrail implementation for API endpoints."""

    def __init__(self, config: TokenGuardrailConfig):
        """Initialize token guardrail with configuration."""
        self.config = config
        self.tokenizers = {}
        self._setup_tokenizers()

    def _setup_tokenizers(self) -> None:
        """Set up tokenizers for both models."""
        if AutoTokenizer is None:
            logger.warning(
                "Transformers not available. Token counting will be approximate."
            )
            return

        try:
            # Load mapper tokenizer
            mapper_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3-8B-Instruct"
            )
            if mapper_tokenizer.pad_token is None:
                mapper_tokenizer.pad_token = mapper_tokenizer.eos_token
            self.tokenizers[ModelType.MAPPER] = mapper_tokenizer

            # Load analyst tokenizer
            analyst_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct"
            )
            if analyst_tokenizer.pad_token is None:
                analyst_tokenizer.pad_token = analyst_tokenizer.eos_token
            self.tokenizers[ModelType.ANALYST] = analyst_tokenizer

            logger.info("Token guardrail tokenizers loaded successfully")

        except Exception as e:
            logger.error("Failed to load tokenizers: %s", e)
            self.tokenizers = {}

    def count_tokens(self, text: str, model_type: ModelType) -> int:
        """
        Count tokens in text for a specific model type.

        Args:
            text: Text to count tokens for
            model_type: Type of model (mapper or analyst)

        Returns:
            Number of tokens
        """
        if model_type in self.tokenizers:
            try:
                tokens = self.tokenizers[model_type].encode(
                    text, add_special_tokens=True
                )
                return len(tokens)
            except Exception as e:
                logger.warning("Token counting failed for %s: %s", model_type, e)
                # Fallback to approximate counting
                return self._approximate_token_count(text)
        else:
            # Fallback to approximate counting
            return self._approximate_token_count(text)

    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count when tokenizer is not available."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def validate_input_tokens(
        self, input_data: Dict[str, Any], model_type: ModelType
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input token count and apply guardrails.

        Args:
            input_data: Input data to validate
            model_type: Type of model

        Returns:
            Tuple of (is_valid, validation_result)
        """
        # Extract text content for token counting
        text_content = self._extract_text_content(input_data, model_type)

        # Count tokens
        token_count = self.count_tokens(text_content, model_type)

        # Get limits
        if model_type == ModelType.MAPPER:
            max_tokens = self.config.mapper_max_input_tokens
            fallback_strategy = self.config.mapper_fallback_strategy
        else:
            max_tokens = self.config.analyst_max_input_tokens
            fallback_strategy = self.config.analyst_fallback_strategy

        # Check thresholds
        warning_threshold = int(max_tokens * self.config.warning_threshold_ratio)
        error_threshold = int(max_tokens * self.config.error_threshold_ratio)

        validation_result = {
            "token_count": token_count,
            "max_tokens": max_tokens,
            "model_type": model_type.value,
            "within_limits": token_count <= max_tokens,
            "warning_threshold_exceeded": token_count > warning_threshold,
            "error_threshold_exceeded": token_count > error_threshold,
            "fallback_strategy": (
                fallback_strategy if token_count > max_tokens else None
            ),
        }

        # Log token usage
        if self.config.log_token_counts:
            self._log_token_usage(
                token_count, max_tokens, model_type, validation_result
            )

        # Handle exceeded limits
        if token_count > max_tokens:
            if fallback_strategy == "reject":
                raise TokenGuardrailError(
                    f"Input token count {token_count} exceeds limit {max_tokens} for {model_type.value}",
                    token_count,
                    max_tokens,
                    model_type.value,
                )
            else:
                # Apply fallback strategy
                validation_result["fallback_applied"] = True
                validation_result["fallback_result"] = self._apply_fallback_strategy(
                    input_data, model_type, fallback_strategy, token_count, max_tokens
                )

        return validation_result["within_limits"], validation_result

    def _extract_text_content(
        self, input_data: Dict[str, Any], model_type: ModelType
    ) -> str:
        """Extract text content from input data for token counting."""
        if model_type == ModelType.MAPPER:
            # For mapper, combine detector outputs and context
            content_parts = []

            if "detector_outputs" in input_data:
                content_parts.append(str(input_data["detector_outputs"]))

            if "context" in input_data:
                content_parts.append(str(input_data["context"]))

            if "metadata" in input_data:
                content_parts.append(str(input_data["metadata"]))

            return " ".join(content_parts)

        else:  # ANALYST
            # For analyst, combine all relevant fields
            content_parts = []

            if "content" in input_data:
                content_parts.append(str(input_data["content"]))

            if "mapping_result" in input_data:
                content_parts.append(str(input_data["mapping_result"]))

            if "context" in input_data:
                content_parts.append(str(input_data["context"]))

            if "requirements" in input_data:
                content_parts.append(str(input_data["requirements"]))

            return " ".join(content_parts)

    def _apply_fallback_strategy(
        self,
        input_data: Dict[str, Any],
        model_type: ModelType,
        strategy: str,
        token_count: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Apply fallback strategy when token limits are exceeded."""
        if strategy == "rules_based":
            return self._apply_rules_based_fallback(input_data, model_type)

        elif strategy == "chunked_analysis":
            return self._apply_chunked_analysis_fallback(
                input_data, model_type, max_tokens
            )

        elif strategy == "chunked":
            return self._apply_chunked_fallback(input_data, model_type, max_tokens)

        else:
            logger.warning("Unknown fallback strategy: %s", strategy)
            return {"error": f"Unknown fallback strategy: {strategy}"}

    def _apply_rules_based_fallback(
        self, input_data: Dict[str, Any], model_type: ModelType
    ) -> Dict[str, Any]:
        """Apply rules-based fallback for mapper when token limit exceeded."""
        logger.info("Applying rules-based fallback for %s", model_type.value)

        # Simple rules-based mapping as fallback
        fallback_result = {
            "mapping_type": "rules_based_fallback",
            "scores": {},
            "confidence": 0.5,  # Lower confidence for fallback
            "fallback_reason": "input_token_limit_exceeded",
        }

        # Basic rule-based mapping logic
        if "detector_outputs" in input_data:
            detector_outputs = input_data["detector_outputs"]
            if isinstance(detector_outputs, dict):
                for detector, result in detector_outputs.items():
                    if isinstance(result, dict) and "category" in result:
                        category = result["category"]
                        confidence = result.get("confidence", 0.5)
                        fallback_result["scores"][f"Rules.{category}"] = confidence

        return fallback_result

    def _apply_chunked_analysis_fallback(
        self, input_data: Dict[str, Any], model_type: ModelType, max_tokens: int
    ) -> Dict[str, Any]:
        """Apply chunked analysis fallback for analyst when token limit exceeded."""
        logger.info("Applying chunked analysis fallback for %s", model_type.value)

        # Split content into chunks and analyze separately
        content = str(input_data.get("content", ""))
        chunk_size = max_tokens // 4  # Rough chunk size

        chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]

        return {
            "analysis_type": "chunked_analysis",
            "chunks_analyzed": len(chunks),
            "risk_level": "medium",  # Conservative assessment
            "recommendations": [
                "Content too large for full analysis - review in chunks"
            ],
            "reason": "input_token_limit_exceeded",
            "fallback_applied": True,
        }

    def _apply_chunked_fallback(
        self, input_data: Dict[str, Any], model_type: ModelType, max_tokens: int
    ) -> Dict[str, Any]:
        """Apply general chunked fallback."""
        logger.info("Applying chunked fallback for %s", model_type.value)

        return {
            "processing_type": "chunked",
            "reason": "input_token_limit_exceeded",
            "max_tokens": max_tokens,
            "fallback_applied": True,
        }

    def _log_token_usage(
        self,
        token_count: int,
        max_tokens: int,
        model_type: ModelType,
        validation_result: Dict[str, Any],
    ) -> None:
        """Log token usage statistics."""
        log_data = {
            "token_count": token_count,
            "max_tokens": max_tokens,
            "model_type": model_type.value,
            "utilization_ratio": token_count / max_tokens,
            "within_limits": validation_result["within_limits"],
            "warning_threshold_exceeded": validation_result[
                "warning_threshold_exceeded"
            ],
            "error_threshold_exceeded": validation_result["error_threshold_exceeded"],
        }

        if validation_result["error_threshold_exceeded"]:
            logger.warning("Token limit exceeded", **log_data)
        elif validation_result["warning_threshold_exceeded"]:
            logger.info("Token usage approaching limit", **log_data)
        else:
            logger.debug("Token usage within limits", **log_data)


def create_token_guardrail(
    config: Optional[TokenGuardrailConfig] = None,
) -> TokenGuardrail:
    """Create a token guardrail instance with default or custom configuration."""
    if config is None:
        config = TokenGuardrailConfig()

    return TokenGuardrail(config)


# FastAPI dependency for token guardrails
def get_token_guardrail() -> TokenGuardrail:
    """FastAPI dependency to get token guardrail instance."""
    return create_token_guardrail()
