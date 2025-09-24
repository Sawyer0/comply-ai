---
inclusion: fileMatch
fileMatchPattern: '**/training/**'
---

# Model Training Guidelines

Reference the training configuration schema:
#[[file:schemas/training-config.json]]

## LoRA Fine-tuning Process

### Training Data Preparation
```python
# Training data format
{
    "input": "Detector output: {detector_type: 'presidio', findings: [...]}",
    "output": "Canonical taxonomy: {category: 'pii', subcategory: 'person_name', confidence: 0.95}"
}
```

### LoRA Configuration
```yaml
# training/lora_config.yaml
model_name: "meta-llama/Llama-3-8b-instruct"
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 32          # LoRA scaling parameter
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

training_config:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_epochs: 3
  warmup_steps: 100
  max_seq_length: 2048
```

### Training Script
```python
# training/train_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_model(model_name: str, lora_config: dict):
    """Setup model with LoRA configuration"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **lora_config
    )
    
    model = get_peft_model(model, peft_config)
    return model

def train_model(model, train_dataset, eval_dataset, training_args):
    """Train the LoRA model"""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    trainer.train()
    return trainer
```

## Phi-3 Mini Training

### Model Configuration
```yaml
# training/phi3_config.yaml
model_name: "microsoft/Phi-3-mini-4k-instruct"
training_config:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 5
  max_seq_length: 4096
  gradient_checkpointing: true
```

### Training Data Format
```python
# Compliance analysis training data
{
    "input": "Context: {compliance_framework: 'SOC2', finding: {...}} Analyze compliance risk.",
    "output": "Risk Assessment: {risk_level: 'medium', remediation: [...], controls: [...]}"
}
```

## Training Pipeline

### Data Validation
```python
def validate_training_data(dataset):
    """Validate training dataset quality"""
    checks = [
        check_data_format,
        check_label_distribution,
        check_sequence_lengths,
        check_data_quality
    ]
    
    for check in checks:
        result = check(dataset)
        if not result.passed:
            raise ValidationError(f"Data validation failed: {result.message}")
```

### Model Evaluation
```python
def evaluate_model(model, eval_dataset):
    """Comprehensive model evaluation"""
    metrics = {
        'accuracy': calculate_accuracy(model, eval_dataset),
        'f1_score': calculate_f1_score(model, eval_dataset),
        'confidence_calibration': check_confidence_calibration(model, eval_dataset),
        'latency': measure_inference_latency(model)
    }
    return metrics
```

## Training Infrastructure

### Cloud Training Setup
```python
# notebooks/cloud_training.py
# Google Colab / Kaggle setup
!pip install transformers peft accelerate bitsandbytes

# Multi-GPU training configuration
from accelerate import Accelerator
accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

### Local Training
```bash
# Local training with GPU
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    training/train_lora.py \
    --config training/lora_config.yaml \
    --output_dir checkpoints/llama-3-8b-lora
```

## Model Versioning

### Checkpoint Management
```python
class ModelVersionManager:
    def save_checkpoint(self, model, version: str, metadata: dict):
        """Save model checkpoint with version and metadata"""
        checkpoint_path = f"checkpoints/{version}"
        
        # Save model
        model.save_pretrained(checkpoint_path)
        
        # Save metadata
        with open(f"{checkpoint_path}/metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Upload to S3
        self.upload_to_s3(checkpoint_path, version)
    
    def load_checkpoint(self, version: str):
        """Load specific model version"""
        checkpoint_path = f"checkpoints/{version}"
        
        if not os.path.exists(checkpoint_path):
            self.download_from_s3(version, checkpoint_path)
        
        return AutoModelForCausalLM.from_pretrained(checkpoint_path)
```

### Model Registry
```yaml
# Model registry configuration
models:
  llama-3-8b-compliance-v1.0:
    base_model: "meta-llama/Llama-3-8b-instruct"
    training_data: "compliance_mappings_v1.0"
    performance_metrics:
      accuracy: 0.94
      f1_score: 0.92
      latency_p95: "45ms"
    
  phi-3-mini-analyst-v1.0:
    base_model: "microsoft/Phi-3-mini-4k-instruct"
    training_data: "risk_analysis_v1.0"
    performance_metrics:
      accuracy: 0.89
      f1_score: 0.87
      latency_p95: "30ms"
```

## Quality Assurance

### Training Monitoring
```python
def setup_training_monitoring():
    """Setup monitoring for training process"""
    wandb.init(project="llama-mapper-training")
    
    # Log training metrics
    wandb.log({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model_parameters": model.num_parameters()
    })

def log_training_step(step, loss, metrics):
    """Log training step metrics"""
    wandb.log({
        "step": step,
        "train_loss": loss,
        **metrics
    })
```

### Model Testing
```python
def test_model_quality(model, test_dataset):
    """Comprehensive model quality testing"""
    tests = [
        test_accuracy_threshold,
        test_confidence_calibration,
        test_bias_detection,
        test_adversarial_robustness,
        test_inference_speed
    ]
    
    results = {}
    for test in tests:
        results[test.__name__] = test(model, test_dataset)
    
    return results
```

## Deployment Integration

### Model Serving Preparation
```python
def prepare_for_serving(model_path: str, serving_backend: str):
    """Prepare model for production serving"""
    if serving_backend == "vllm":
        # Convert to vLLM format
        convert_to_vllm_format(model_path)
    elif serving_backend == "tgi":
        # Prepare for Text Generation Inference
        prepare_tgi_config(model_path)
    
    # Run serving tests
    test_serving_performance(model_path, serving_backend)
```

### A/B Testing Setup
```python
def setup_ab_testing(new_model_version: str, traffic_split: float = 0.1):
    """Setup A/B testing for new model version"""
    config = {
        "models": {
            "control": "llama-3-8b-compliance-v1.0",
            "treatment": new_model_version
        },
        "traffic_split": {
            "control": 1.0 - traffic_split,
            "treatment": traffic_split
        },
        "metrics": ["accuracy", "latency", "user_satisfaction"]
    }
    
    deploy_ab_test_config(config)
```