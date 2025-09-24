"""
Strict Output Generation with Reject Sampling

System prompts that ban rambling; hard cap max_new_tokens (Mapper very small; Analyst concise).
Add reject-sampling: if schema validator fails, one retry with a terse "JSON-only" system reminder.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# import torch  # External dependency
# from transformers import AutoTokenizer, AutoModelForCausalLM  # External dependency


class ModelType(Enum):
    """Model types for strict generation."""

    MAPPER = "mapper"
    ANALYST = "analyst"


@dataclass
class StrictGenerationConfig:
    """Configuration for strict output generation."""

    # Token limits (optimized for fine-tuning)
    mapper_max_tokens: int = 64  # 32-64 tokens for tiny JSON output
    analyst_max_tokens: int = 256  # 128-256 tokens for concise reasoning
    mapper_min_tokens: int = 8
    analyst_min_tokens: int = 32

    # Generation parameters
    temperature: float = 0.1  # Low temperature for deterministic outputs
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Reject sampling
    max_retries: int = 1
    retry_temperature: float = 0.05  # Even lower temperature for retry

    # Schema validation
    strict_schema: bool = True
    fail_on_schema_error: bool = True


class StrictSystemPrompts:
    """System prompts that ban rambling and enforce strict outputs."""

    MAPPER_PROMPT = """You are a compliance taxonomy mapper. Your ONLY job is to map detector outputs to canonical taxonomy labels.

RULES:
- Output ONLY valid JSON
- NO explanations, NO rambling, NO prose
- Keep responses under 64 tokens
- Use exact taxonomy format: "CATEGORY.Subcategory.Item"
- Confidence must be 0.0-1.0

Required JSON format:
{
  "taxonomy": ["PII.Contact.Email"],
  "scores": {"PII.Contact.Email": 0.95},
  "confidence": 0.95
}

JSON ONLY. NO OTHER TEXT."""

    ANALYST_PROMPT = """You are a compliance analyst. Your ONLY job is to provide structured compliance analysis.

RULES:
- Output ONLY valid JSON
- NO explanations, NO rambling, NO prose
- Keep responses under 256 tokens
- Use exact analysis types and risk levels
- Reason field max 120 characters

Required JSON format:
{
  "analysis_type": "privacy_risk",
  "risk_level": "medium",
  "recommendations": ["Implement log sanitization"],
  "reason": "PII detected in logs (max 120 chars)"
}

JSON ONLY. NO OTHER TEXT."""

    RETRY_PROMPT = """JSON-ONLY response required. Fix schema errors.

RULES:
- Output ONLY valid JSON
- NO explanations, NO rambling, NO prose
- Follow exact schema requirements
- Keep responses concise

JSON ONLY. NO OTHER TEXT."""


class StrictOutputGenerator:
    """Generates strict outputs with reject sampling for schema compliance."""

    def __init__(self, model, tokenizer, config: StrictGenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompts = StrictSystemPrompts()

    def generate_mapper_output(
        self, detector_output: str, context: str = ""
    ) -> Dict[str, Any]:
        """Generate strict Mapper output with reject sampling."""
        return self._generate_with_reject_sampling(
            ModelType.MAPPER, detector_output, context, self.config.mapper_max_tokens
        )

    def generate_analyst_output(
        self, compliance_scenario: str, context: str = ""
    ) -> Dict[str, Any]:
        """Generate strict Analyst output with reject sampling."""
        return self._generate_with_reject_sampling(
            ModelType.ANALYST,
            compliance_scenario,
            context,
            self.config.analyst_max_tokens,
        )

    def _generate_with_reject_sampling(
        self, model_type: ModelType, input_text: str, context: str, max_tokens: int
    ) -> Dict[str, Any]:
        """Generate output with reject sampling for schema compliance."""

        # First attempt
        result = self._generate_single_attempt(
            model_type,
            input_text,
            context,
            max_tokens,
            temperature=self.config.temperature,
        )

        # Validate schema
        validation_result = self._validate_output(model_type, result["generated_text"])

        if validation_result["valid"]:
            return {
                "success": True,
                "output": validation_result["parsed_output"],
                "generated_text": result["generated_text"],
                "attempts": 1,
                "validation_errors": [],
            }

        # Retry with stricter parameters if validation failed
        if self.config.max_retries > 0:
            retry_result = self._generate_single_attempt(
                model_type,
                input_text,
                context,
                max_tokens,
                temperature=self.config.retry_temperature,
                use_retry_prompt=True,
            )

            retry_validation = self._validate_output(
                model_type, retry_result["generated_text"]
            )

            if retry_validation["valid"]:
                return {
                    "success": True,
                    "output": retry_validation["parsed_output"],
                    "generated_text": retry_result["generated_text"],
                    "attempts": 2,
                    "validation_errors": validation_result["errors"],
                }

        # Both attempts failed
        return {
            "success": False,
            "output": None,
            "generated_text": result["generated_text"],
            "attempts": 2,
            "validation_errors": validation_result["errors"],
        }

    def _generate_single_attempt(
        self,
        model_type: ModelType,
        input_text: str,
        context: str,
        max_tokens: int,
        temperature: float,
        use_retry_prompt: bool = False,
    ) -> Dict[str, Any]:
        """Generate a single attempt with given parameters."""

        # Create prompt
        if use_retry_prompt:
            system_prompt = self.system_prompts.RETRY_PROMPT
        else:
            system_prompt = (
                self.system_prompts.MAPPER_PROMPT
                if model_type == ModelType.MAPPER
                else self.system_prompts.ANALYST_PROMPT
            )

        # Format input
        if context:
            formatted_input = f"Context: {context}\n\nInput: {input_text}"
        else:
            formatted_input = f"Input: {input_text}"

        # Create full prompt
        full_prompt = f"{system_prompt}\n\n{formatted_input}\n\nResponse:"

        # Tokenize
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        # Generate
        # with torch.no_grad():  # External dependency
        # Simplified for now - would need torch for actual implementation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        return {
            "generated_text": generated_text,
            "prompt": full_prompt,
            "temperature": temperature,
        }

    def _validate_output(
        self, model_type: ModelType, generated_text: str
    ) -> Dict[str, Any]:
        """Validate generated output against schema."""
        try:
            # Try to parse JSON
            parsed_output = json.loads(generated_text)

            # Basic schema validation
            if model_type == ModelType.MAPPER:
                return self._validate_mapper_schema(parsed_output)
            else:
                return self._validate_analyst_schema(parsed_output)

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON decode error: {str(e)}"],
                "parsed_output": None,
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "parsed_output": None,
            }

    def _validate_mapper_schema(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Mapper output schema."""
        errors = []

        # Check required fields
        required_fields = ["taxonomy", "scores", "confidence"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")

        # Check taxonomy format
        if "taxonomy" in output:
            taxonomy = output["taxonomy"]
            if not isinstance(taxonomy, list):
                errors.append("Taxonomy must be a list")
            else:
                for item in taxonomy:
                    if not isinstance(item, str) or not re.match(
                        r"^[A-Z]+\.[A-Za-z]+(\.[A-Za-z]+)*$", item
                    ):
                        errors.append(f"Invalid taxonomy format: {item}")

        # Check scores format
        if "scores" in output:
            scores = output["scores"]
            if not isinstance(scores, dict):
                errors.append("Scores must be a dictionary")
            else:
                for key, value in scores.items():
                    if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                        errors.append(f"Invalid score for {key}: {value}")

        # Check confidence
        if "confidence" in output:
            confidence = output["confidence"]
            if not isinstance(confidence, (int, float)) or not (
                0.0 <= confidence <= 1.0
            ):
                errors.append(f"Invalid confidence: {confidence}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "parsed_output": output if len(errors) == 0 else None,
        }

    def _validate_analyst_schema(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Analyst output schema."""
        errors = []

        # Check required fields
        required_fields = ["analysis_type", "risk_level", "recommendations"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")

        # Check analysis_type
        if "analysis_type" in output:
            valid_types = [
                "GDPR_Compliance_Analysis",
                "Legal_Compliance_Analysis",
                "Policy_Compliance_Analysis",
                "Stakeholder_Engagement_Analysis",
                "Multi_Framework_Compliance_Analysis",
                "policy_violation",
                "privacy_risk",
                "security_risk",
                "compliance_gap",
            ]
            if output["analysis_type"] not in valid_types:
                errors.append(f"Invalid analysis_type: {output['analysis_type']}")

        # Check risk_level
        if "risk_level" in output:
            valid_levels = ["low", "medium", "high", "critical"]
            if output["risk_level"] not in valid_levels:
                errors.append(f"Invalid risk_level: {output['risk_level']}")

        # Check recommendations
        if "recommendations" in output:
            recommendations = output["recommendations"]
            if not isinstance(recommendations, list):
                errors.append("Recommendations must be a list")
            elif len(recommendations) == 0:
                errors.append("Recommendations cannot be empty")

        # Check reason length
        if "reason" in output:
            reason = output["reason"]
            if isinstance(reason, str) and len(reason) > 120:
                errors.append(f"Reason too long: {len(reason)} > 120 characters")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "parsed_output": output if len(errors) == 0 else None,
        }


class StrictGenerationPipeline:
    """Complete pipeline for strict output generation with monitoring."""

    def __init__(self, model, tokenizer, config: StrictGenerationConfig):
        self.generator = StrictOutputGenerator(model, tokenizer, config)
        self.config = config
        self.generation_stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "retry_attempts": 0,
            "schema_errors": defaultdict(int),
        }

    def process_mapper_request(
        self, detector_output: str, context: str = ""
    ) -> Dict[str, Any]:
        """Process a Mapper request with strict generation."""
        self.generation_stats["total_attempts"] += 1

        result = self.generator.generate_mapper_output(detector_output, context)

        if result["success"]:
            self.generation_stats["successful_generations"] += 1
        else:
            self.generation_stats["failed_generations"] += 1
            for error in result["validation_errors"]:
                self.generation_stats["schema_errors"][error] += 1

        if result["attempts"] > 1:
            self.generation_stats["retry_attempts"] += 1

        return result

    def process_analyst_request(
        self, compliance_scenario: str, context: str = ""
    ) -> Dict[str, Any]:
        """Process an Analyst request with strict generation."""
        self.generation_stats["total_attempts"] += 1

        result = self.generator.generate_analyst_output(compliance_scenario, context)

        if result["success"]:
            self.generation_stats["successful_generations"] += 1
        else:
            self.generation_stats["failed_generations"] += 1
            for error in result["validation_errors"]:
                self.generation_stats["schema_errors"][error] += 1

        if result["attempts"] > 1:
            self.generation_stats["retry_attempts"] += 1

        return result

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        total = self.generation_stats["total_attempts"]
        if total == 0:
            return self.generation_stats

        return {
            **self.generation_stats,
            "success_rate": self.generation_stats["successful_generations"] / total,
            "retry_rate": self.generation_stats["retry_attempts"] / total,
            "failure_rate": self.generation_stats["failed_generations"] / total,
        }

    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "retry_attempts": 0,
            "schema_errors": defaultdict(int),
        }


# Example usage and testing
if __name__ == "__main__":
    # This would be used with actual model loading
    print("🔥 Strict Output Generation System")
    print("  - Mapper max tokens: 50")
    print("  - Analyst max tokens: 200")
    print("  - Reject sampling: 1 retry")
    print("  - Schema validation: Strict")
    print("  - System prompts: Anti-rambling")

    # Example configuration
    config = StrictGenerationConfig(
        mapper_max_tokens=50, analyst_max_tokens=200, temperature=0.1, max_retries=1
    )

    print(f"\n📋 Configuration:")
    print(f"  Mapper max tokens: {config.mapper_max_tokens}")
    print(f"  Analyst max tokens: {config.analyst_max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Strict schema: {config.strict_schema}")
