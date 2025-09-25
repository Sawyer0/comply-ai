"""
Validation components for the Analysis Service.

This module consolidates validation functionality including:
- Input validation and sanitization for risk assessment
- Output validation against analysis schemas
- Business logic validation for compliance analysis
- Response format validation for risk assessments
"""

# Import with error handling
try:
    from .input_validator import InputValidator, InputValidationResult

    _INPUT_VALIDATOR_AVAILABLE = True
except ImportError as e:
    InputValidator = None
    InputValidationResult = None
    _INPUT_VALIDATOR_AVAILABLE = False

try:
    from .output_validator import OutputValidator, OutputValidationResult

    _OUTPUT_VALIDATOR_AVAILABLE = True
except ImportError as e:
    OutputValidator = None
    OutputValidationResult = None
    _OUTPUT_VALIDATOR_AVAILABLE = False

try:
    from .response_validator import ResponseValidator, ValidationResult

    _RESPONSE_VALIDATOR_AVAILABLE = True
except ImportError as e:
    ResponseValidator = None
    ValidationResult = None
    _RESPONSE_VALIDATOR_AVAILABLE = False

try:
    from .schema_validator import (
        SchemaValidator,
        AnalysisSchemaValidator,
        SchemaValidationConfig,
        validate_analysis_output,
        AnalysisType,
    )

    _SCHEMA_VALIDATOR_AVAILABLE = True
except ImportError as e:
    SchemaValidator = None
    AnalysisSchemaValidator = None
    SchemaValidationConfig = None
    validate_analysis_output = None
    AnalysisType = None
    _SCHEMA_VALIDATOR_AVAILABLE = False

# Build __all__ list dynamically based on what was successfully imported
__all__ = []

if _INPUT_VALIDATOR_AVAILABLE:
    __all__.extend(["InputValidator", "InputValidationResult"])

if _OUTPUT_VALIDATOR_AVAILABLE:
    __all__.extend(["OutputValidator", "OutputValidationResult"])

if _RESPONSE_VALIDATOR_AVAILABLE:
    __all__.extend(["ResponseValidator", "ValidationResult"])

if _SCHEMA_VALIDATOR_AVAILABLE:
    __all__.extend(
        [
            "SchemaValidator",
            "AnalysisSchemaValidator",
            "SchemaValidationConfig",
            "validate_analysis_output",
            "AnalysisType",
        ]
    )
