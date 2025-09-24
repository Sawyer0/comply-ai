# Pipeline Orchestration System

## Overview

The Pipeline Orchestration System provides a maintainable, extensible framework for automating training, evaluation, and deployment workflows. Built with clean architecture principles and production-grade maintainability from day 1.

## Key Features

- **Clean Architecture**: Proper separation of concerns with domain/application/infrastructure layers
- **Extensible Design**: Easy to add new stages and pipeline types
- **Async Execution**: Non-blocking pipeline execution with monitoring
- **Dependency Resolution**: Automatic stage dependency resolution and parallel execution
- **Comprehensive Monitoring**: Built-in metrics collection and performance tracking
- **Error Handling**: Robust error handling with proper fallback mechanisms
- **Configuration Management**: Type-safe configuration with validation

## Quick Start

### 1. Train a Dual Model (Mapper + Analyst)

```bash
# Basic training
mapper train --model-type dual --output-dir ./training_output

# With custom configuration
mapper train --model-type dual --config-file examples/pipeline_config.json

# Async execution
mapper train --model-type dual --async
```

### 2. Train Individual Models

```bash
# Train only mapper model
mapper train --model-type mapper

# Train only analyst model  
mapper train --model-type analyst
```

### 3. Pipeline Management

```bash
# List all pipelines
mapper pipeline list

# Check pipeline status
mapper pipeline status dual-training-pipeline

# View pipeline metrics
mapper pipeline metrics dual-training-pipeline

# List active executions
mapper pipeline active
```

### 4. Standalone Operations

```bash
# Evaluate existing model
mapper pipeline evaluate my-model --model-path ./checkpoints/v1.0.0

# Deploy model
mapper pipeline deploy my-model v1.0.0 --environment staging --canary-percentage 10
```

## Architecture

### Core Components

1. **PipelineOrchestrator**: High-level orchestrator managing pipeline lifecycle
2. **Pipeline**: Orchestrates execution of multiple stages with dependency resolution
3. **PipelineStage**: Abstract base class for pipeline stages (Template Method pattern)
4. **PipelineRegistry**: Manages pipeline definitions and configurations
5. **PipelineMonitor**: Comprehensive monitoring and observability

### Stage Implementations

1. **DataPreparationStage**: Orchestrates training data generation
2. **TrainingStage**: Connects to existing LoRA/Phi-3 training infrastructure
3. **EvaluationStage**: Comprehensive model evaluation with quality gates
4. **DeploymentStage**: Canary deployment with KPI-based promotion

### Pipeline Flow

```
Data Preparation → Training → Evaluation → Deployment
       ↓              ↓           ↓           ↓
   Generate Data   Train Models  Run Tests   Deploy Canary
   Validate Data   Register      Quality     Monitor KPIs
                   Versions      Gates       Promote/Rollback
```

## Configuration

### Pipeline Configuration

```json
{
  "name": "dual-model-training",
  "description": "Training pipeline for dual model",
  "version": "1.0.0",
  "timeout_seconds": 7200,
  "max_parallel_stages": 2,
  "failure_strategy": "fail_fast",
  
  "stages": [
    {
      "name": "data_preparation",
      "enabled": true,
      "timeout_seconds": 1800,
      "dependencies": []
    },
    {
      "name": "training", 
      "enabled": true,
      "timeout_seconds": 5400,
      "dependencies": ["data_preparation"]
    }
  ],
  
  "global_config": {
    "model_type": "dual",
    "required_keys": ["output_dir", "training_data_path"],
    "type_constraints": {
      "output_dir": "string",
      "epochs": "integer"
    }
  }
}
```

### Training Configuration

```json
{
  "model_type": "dual",
  "output_dir": "./training_output",
  
  "mapper_learning_rate": 2e-4,
  "mapper_epochs": 2,
  "mapper_batch_size": 4,
  "mapper_lora_r": 256,
  "mapper_lora_alpha": 512,
  
  "analyst_learning_rate": 1e-4,
  "analyst_epochs": 2,
  "analyst_batch_size": 8,
  "analyst_lora_r": 128,
  "analyst_lora_alpha": 256
}
```

## Monitoring and Observability

### Pipeline Metrics

- **Execution Metrics**: Duration, success rate, failure rate
- **Performance Metrics**: P50/P95/P99 latencies, throughput
- **Quality Metrics**: Model accuracy, F1 scores, quality gate pass rates
- **Resource Metrics**: CPU/memory usage, cost tracking

### Monitoring Commands

```bash
# Get comprehensive pipeline metrics
mapper pipeline metrics dual-training-pipeline

# View execution history
mapper pipeline history --pipeline dual-training-pipeline --limit 10

# Check for anomalies
mapper pipeline anomalies dual-training-pipeline

# Generate performance report
mapper pipeline report dual-training-pipeline --days 7
```

## Extending the System

### Adding New Stages

1. **Create Stage Class**:
```python
class MyCustomStage(PipelineStage):
    async def _execute_stage(self, context: PipelineContext) -> PipelineContext:
        # Your stage logic here
        return context.with_artifact("my_result", result)
    
    def get_dependencies(self) -> List[str]:
        return ["previous_stage"]
```

2. **Register in Pipeline**:
```python
pipeline.add_stage(MyCustomStage())
```

### Creating Custom Pipelines

```python
from llama_mapper.pipeline import Pipeline, PipelineOrchestrator

# Create custom pipeline
stages = [DataPreparationStage(), MyCustomStage(), DeploymentStage()]
pipeline = Pipeline("my-custom-pipeline", stages)

# Register and execute
orchestrator = PipelineOrchestrator()
await orchestrator.register_pipeline(pipeline)
await orchestrator.execute_pipeline("my-custom-pipeline")
```

## Production Deployment

### Quality Gates

All models must pass quality gates before deployment:

- **Accuracy**: ≥ 85%
- **F1 Score**: ≥ 80%  
- **Precision**: ≥ 80%
- **Recall**: ≥ 75%
- **Schema Validation**: ≥ 98%

### Canary Deployment

1. **Deploy Canary**: 5% traffic by default
2. **Monitor KPIs**: P95 latency, error rate, F1 score
3. **Evaluate Performance**: Compare against baseline
4. **Promote/Rollback**: Based on KPI thresholds

### Rollback Procedures

```bash
# Emergency rollback
mapper pipeline rollback my-model v1.2.0 --to-version v1.1.0

# Check rollback status
mapper pipeline status my-model-rollback
```

## Troubleshooting

### Common Issues

1. **Training Failures**: Check data preparation logs and GPU memory
2. **Quality Gate Failures**: Review evaluation metrics and thresholds
3. **Deployment Issues**: Verify model registry and Kubernetes connectivity

### Debug Commands

```bash
# Verbose pipeline execution
mapper train --model-type dual --verbose

# Check pipeline validation
mapper pipeline validate dual-training-pipeline

# View detailed logs
mapper logs pipeline --pipeline dual-training-pipeline --level debug
```

## Best Practices

1. **Configuration Management**: Use version-controlled config files
2. **Monitoring**: Set up alerts for pipeline failures and quality degradation
3. **Testing**: Validate pipelines in staging before production
4. **Documentation**: Document custom stages and pipeline modifications
5. **Rollback Planning**: Always have rollback procedures ready

## Integration with Existing Systems

The pipeline orchestration system integrates seamlessly with existing infrastructure:

- **Training**: Uses existing LoRATrainer and Phi3Trainer classes
- **Data Generation**: Leverages HybridTrainingDataGenerator and AnalysisModuleDataGenerator
- **Model Versioning**: Integrates with ModelVersionManager for deployment
- **Evaluation**: Uses ComplianceModelEvaluator for quality assessment
- **CLI**: Extends existing mapper CLI with pipeline commands