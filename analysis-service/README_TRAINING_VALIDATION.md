# Analysis Service - Training & Validation Infrastructure

This document describes the training and validation infrastructure for the Analysis Service, which uses Phi-3-Mini models for context-aware risk assessments and compliance analysis.

## Training Infrastructure (`src/analysis/training/`)

### Core Components

1. **Trainer (`trainer.py`)**
   - `Phi3Trainer`: Main trainer class for Phi-3-Mini models
   - `TrainingConfig`: Configuration dataclass for training parameters
   - Supports LoRA fine-tuning for efficient training
   - Handles model setup, training, evaluation, and saving

2. **Data Processing (`data_processor.py`)**
   - `AnalysisDataProcessor`: Processes training data for risk assessment
   - `RiskAssessmentDataset`: PyTorch dataset for risk assessment training
   - Handles data validation, cleaning, and formatting
   - Supports multiple compliance frameworks (SOC2, ISO27001, HIPAA, etc.)

3. **Checkpoint Management (`checkpoint_manager.py`)**
   - `CheckpointManager`: Manages model checkpoints and versions
   - `ModelVersion`: Metadata structure for model versions
   - Handles saving, loading, and cleanup of model checkpoints
   - Tracks performance metrics and training configurations

4. **Model Loading (`model_loader.py`)**
   - `ModelLoader`: Utility for loading trained models
   - Supports caching and memory management
   - Handles base model and fine-tuned model loading
   - Includes model validation and health checks

5. **Version Management (`version_manager.py`)**
   - `ModelVersionManager`: Manages model versions and registry
   - `DeploymentManager`: Handles model deployments and A/B testing
   - `DeploymentConfig`: Configuration for deployment strategies
   - Supports blue-green, canary, and rolling deployments

### Training Data Format

```python
# Risk Assessment Training Example
{
    "context": "Compliance analysis for SOC2",
    "findings": {
        "detector_type": "presidio",
        "findings": [
            {
                "type": "PERSON",
                "confidence": 0.95,
                "location": "field_name"
            }
        ]
    },
    "framework": "SOC2",
    "response": """**Overall Risk Level:** MEDIUM

**Technical Risk Factors:**
- PII exposure in unencrypted field
- Lack of access controls

**Business Impact:** Potential compliance violation

**Regulatory Compliance (SOC2):** Violates CC6.1 access control requirements

**Remediation Steps:**
1. Implement field-level encryption
2. Add access controls and audit logging
3. Update data handling procedures

**Recommended Controls:**
- Encryption at rest and in transit
- Role-based access control (RBAC)
- Regular compliance audits"""
}
```

### Usage Example

```python
from analysis.training import Phi3Trainer, TrainingConfig, AnalysisDataProcessor

# Setup training configuration
config = TrainingConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=5,
    lora_r=16,
    lora_alpha=32
)

# Initialize trainer
trainer = Phi3Trainer(config)

# Process training data
data_processor = AnalysisDataProcessor(trainer.tokenizer)
train_dataset, val_dataset = data_processor.prepare_risk_assessment_data(
    raw_training_data,
    validation_split=0.2
)

# Train the model
results = trainer.train(train_dataset, val_dataset)

# Save the trained model
trainer.save_model(
    "checkpoints/phi3-risk-assessment-v1.0",
    metadata={
        "training_data_version": "v1.0",
        "performance_metrics": results["metrics"]
    }
)
```

## Validation Infrastructure (`src/analysis/validation/`)

### Core Components

1. **Input Validator (`input_validator.py`)**
   - `InputValidator`: Validates and sanitizes analysis inputs
   - `InputValidationResult`: Structured validation results
   - Supports multiple analysis types (risk assessment, pattern analysis, etc.)
   - Handles framework validation and data sanitization

2. **Output Validator (`output_validator.py`)**
   - `OutputValidator`: Validates analysis outputs against schemas
   - `OutputValidationResult`: Structured output validation results
   - Validates canonical results, risk scores, compliance mappings
   - Checks consistency across multiple outputs

3. **Response Validator (`response_validator.py`)**
   - `ResponseValidator`: Validates complete analysis responses
   - `ValidationResult`: Comprehensive validation results
   - Validates response structure, performance metrics, data quality
   - Supports batch response validation

4. **Schema Validator (`schema_validator.py`)**
   - `SchemaValidator`: Base JSON schema validator
   - `AnalysisSchemaValidator`: Analysis-specific schema validation
   - `AnalysisType`: Enum for supported analysis types
   - Predefined schemas for different analysis types

### Validation Types

1. **Input Validation**
   - Required field validation
   - Data type and format validation
   - Framework and analysis type validation
   - Input sanitization and length limits

2. **Output Validation**
   - Schema compliance validation
   - Business rule validation
   - Confidence score validation
   - Consistency checks across outputs

3. **Response Validation**
   - Complete response structure validation
   - Performance metrics validation
   - Data quality assessment
   - Business logic rule validation

### Usage Example

```python
from analysis.validation import (
    InputValidator, 
    OutputValidator, 
    ResponseValidator,
    AnalysisType
)

# Validate input
input_validator = InputValidator()
input_result = input_validator.validate_analysis_request({
    "tenant_id": "acme-corp",
    "analysis_types": ["risk_assessment"],
    "frameworks": ["SOC2"],
    "findings": {...},
    "context": "Security analysis for compliance"
})

if not input_result.is_valid:
    print("Input validation errors:", input_result.errors)

# Validate output
output_validator = OutputValidator()
output_result = output_validator.validate_risk_score({
    "overall_risk_score": 0.75,
    "technical_risk": 0.8,
    "business_risk": 0.7,
    "regulatory_risk": 0.75,
    "temporal_risk": 0.7,
    "risk_factors": [...]
})

# Validate complete response
response_validator = ResponseValidator()
response_result = response_validator.validate_analysis_response(
    response_data,
    "risk_assessment",
    request_id="req_123"
)
```

## Integration with Analysis Service

The training and validation infrastructure integrates seamlessly with the Analysis Service:

1. **Model Training Pipeline**
   - Automated training data preparation from detector outputs
   - Model training with configurable LoRA parameters
   - Checkpoint management and version tracking
   - Performance evaluation and model validation

2. **Deployment Pipeline**
   - Model version management and registry
   - A/B testing and canary deployments
   - Rollback capabilities and health monitoring
   - Integration with serving infrastructure

3. **Runtime Validation**
   - Input validation for all analysis requests
   - Output validation for model responses
   - Response validation before client delivery
   - Continuous quality monitoring

## Configuration

Training and validation can be configured through:

1. **Training Configuration**
   - Model parameters (LoRA rank, learning rate, etc.)
   - Training parameters (batch size, epochs, etc.)
   - Data processing parameters
   - Checkpoint and versioning settings

2. **Validation Configuration**
   - Strict vs. lenient validation modes
   - Custom validation rules and thresholds
   - Schema validation settings
   - Performance metric thresholds

This infrastructure ensures high-quality, reliable risk assessment and compliance analysis capabilities for the Analysis Service.