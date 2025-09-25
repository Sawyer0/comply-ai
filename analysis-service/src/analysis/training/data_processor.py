"""
Data processing for Analysis Service training.

Handles preparation of training data for risk assessment and compliance analysis models.

Note: This module requires additional ML dependencies. Install with:
pip install -r requirements-training.txt
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..shared_integration import get_shared_logger

# Initialize variables before try block
TRAINING_DEPENDENCIES_AVAILABLE = False
_MISSING_DEPENDENCY_ERROR = "Unknown import error"

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    TRAINING_DEPENDENCIES_AVAILABLE = True
    _MISSING_DEPENDENCY_ERROR = None
except ImportError as e:
    TRAINING_DEPENDENCIES_AVAILABLE = False
    _MISSING_DEPENDENCY_ERROR = str(e)

logger = get_shared_logger(__name__)


# Conditional base class to handle missing dependencies
if TRAINING_DEPENDENCIES_AVAILABLE:
    _DatasetBase = Dataset
else:
    _DatasetBase = object


class RiskAssessmentDataset(_DatasetBase):
    """Dataset for risk assessment and compliance analysis training."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,  # Remove type hint to avoid import issues
        max_length: int = 4096,
        prompt_template: Optional[str] = None,
    ):
        if not TRAINING_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                f"Training dependencies not available: {_MISSING_DEPENDENCY_ERROR}. "
                "Please install with: pip install -r requirements-training.txt"
            )

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._default_prompt_template()

        # Tokenize all examples
        self.tokenized_data = self._tokenize_data()

    def _default_prompt_template(self) -> str:
        """Default prompt template for risk assessment training."""
        return """<|system|>
You are a compliance and risk assessment expert. Analyze the provided security findings and provide detailed risk assessment with remediation guidance.

<|user|>
Context: {context}
Security Findings: {findings}
Compliance Framework: {framework}

Please provide a comprehensive risk assessment including:
1. Overall risk level (low/medium/high/critical)
2. Technical risk factors
3. Business impact assessment
4. Regulatory compliance implications
5. Specific remediation steps
6. Recommended controls

<|assistant|>
{response}"""

    def _tokenize_data(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize all training examples."""
        tokenized_examples = []

        for example in self.data:
            # Format the prompt
            prompt = self.prompt_template.format(
                context=example.get("context", ""),
                findings=json.dumps(example.get("findings", {}), indent=2),
                framework=example.get("framework", "General"),
                response=example["response"],
            )

            # Tokenize
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            tokenized_examples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze().clone(),
                }
            )

        return tokenized_examples

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.tokenized_data[idx]


class AnalysisDataProcessor:
    """Data processor for analysis service training data."""

    def __init__(self, tokenizer):  # Remove type hint to avoid import issues
        if not TRAINING_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                f"Training dependencies not available: {_MISSING_DEPENDENCY_ERROR}. "
                "Please install with: pip install -r requirements-training.txt"
            )

        self.tokenizer = tokenizer
        self.logger = logger.bind(component="analysis_data_processor")

    def prepare_risk_assessment_data(
        self, raw_data: List[Dict[str, Any]], validation_split: float = 0.2
    ) -> Tuple[RiskAssessmentDataset, RiskAssessmentDataset]:
        """
        Prepare risk assessment training data.

        Args:
            raw_data: Raw training examples
            validation_split: Fraction of data to use for validation

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Validate and clean data
        cleaned_data = self._validate_and_clean_data(raw_data)

        # Split data
        split_idx = int(len(cleaned_data) * (1 - validation_split))
        train_data = cleaned_data[:split_idx]
        val_data = cleaned_data[split_idx:]

        # Create datasets
        train_dataset = RiskAssessmentDataset(
            train_data, self.tokenizer, max_length=4096
        )

        val_dataset = RiskAssessmentDataset(val_data, self.tokenizer, max_length=4096)

        self.logger.info(
            "Data preparation complete",
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
        )

        return train_dataset, val_dataset

    def _validate_and_clean_data(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and clean training data."""
        cleaned_data = []

        for i, example in enumerate(raw_data):
            try:
                # Required fields
                if not all(key in example for key in ["findings", "response"]):
                    self.logger.warning(
                        f"Skipping example {i}: missing required fields"
                    )
                    continue

                # Validate findings structure
                findings = example["findings"]
                if not isinstance(findings, (dict, list)):
                    self.logger.warning(
                        f"Skipping example {i}: invalid findings format"
                    )
                    continue

                # Validate response
                response = example["response"]
                if not isinstance(response, str) or len(response.strip()) == 0:
                    self.logger.warning(f"Skipping example {i}: invalid response")
                    continue

                # Clean and normalize
                cleaned_example = {
                    "context": example.get("context", ""),
                    "findings": findings,
                    "framework": example.get("framework", "General"),
                    "response": response.strip(),
                }

                cleaned_data.append(cleaned_example)

            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Error processing example {i}", error=str(e))
                continue
            except Exception as e:
                self.logger.error(
                    f"Unexpected error processing example {i}", error=str(e)
                )
                continue

        self.logger.info(
            "Data validation complete",
            original_count=len(raw_data),
            cleaned_count=len(cleaned_data),
            filtered_count=len(raw_data) - len(cleaned_data),
        )

        return cleaned_data

    def create_compliance_training_data(
        self,
        detector_outputs: List[Dict[str, Any]],
        risk_assessments: List[Dict[str, Any]],
        frameworks: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create training data from detector outputs and risk assessments.

        Args:
            detector_outputs: List of detector output examples
            risk_assessments: Corresponding risk assessments
            frameworks: List of compliance frameworks to include

        Returns:
            Formatted training examples
        """
        if frameworks is None:
            frameworks = ["SOC2", "ISO27001", "HIPAA", "GDPR"]

        training_examples = []

        for detector_output, risk_assessment in zip(detector_outputs, risk_assessments):
            for framework in frameworks:
                # Create framework-specific training example
                example = {
                    "context": f"Compliance analysis for {framework}",
                    "findings": detector_output,
                    "framework": framework,
                    "response": self._format_risk_assessment_response(
                        risk_assessment, framework
                    ),
                }
                training_examples.append(example)

        return training_examples

    def _format_risk_assessment_response(
        self, risk_assessment: Dict[str, Any], framework: str
    ) -> str:
        """Format risk assessment as training response."""
        response_parts = []

        # Overall risk level
        risk_level = risk_assessment.get("risk_level", "medium")
        response_parts.append(f"**Overall Risk Level:** {risk_level.upper()}")

        # Technical risk factors
        technical_risks = risk_assessment.get("technical_risks", [])
        if technical_risks:
            response_parts.append("**Technical Risk Factors:**")
            for risk in technical_risks:
                response_parts.append(f"- {risk}")

        # Business impact
        business_impact = risk_assessment.get("business_impact", "")
        if business_impact:
            response_parts.append(f"**Business Impact:** {business_impact}")

        # Regulatory implications
        regulatory_impact = risk_assessment.get("regulatory_impact", "")
        if regulatory_impact:
            response_parts.append(
                f"**Regulatory Compliance ({framework}):** {regulatory_impact}"
            )

        # Remediation steps
        remediation_steps = risk_assessment.get("remediation_steps", [])
        if remediation_steps:
            response_parts.append("**Remediation Steps:**")
            for i, step in enumerate(remediation_steps, 1):
                response_parts.append(f"{i}. {step}")

        # Recommended controls
        controls = risk_assessment.get("recommended_controls", [])
        if controls:
            response_parts.append("**Recommended Controls:**")
            for control in controls:
                response_parts.append(f"- {control}")

        return "\n\n".join(response_parts)

    def load_training_data_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.logger.info(f"Loaded training data from {file_path}", count=len(data))
            return data

        except FileNotFoundError as e:
            self.logger.error(f"Training data file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Invalid JSON in training data file: {file_path}", error=str(e)
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to load training data from {file_path}", error=str(e)
            )
            raise

    def save_training_data_to_file(
        self, data: List[Dict[str, Any]], file_path: str
    ) -> None:
        """Save training data to JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved training data to {file_path}", count=len(data))

        except PermissionError as e:
            self.logger.error(f"Permission denied saving to {file_path}")
            raise
        except OSError as e:
            self.logger.error(
                f"OS error saving training data to {file_path}", error=str(e)
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to save training data to {file_path}", error=str(e)
            )
            raise
