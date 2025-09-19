# Development Workflow Guide

This guide explains how to develop the Llama Mapper project using a hybrid local/cloud approach.

## 🏗️ Architecture Overview

```
Local Development (Your Computer)
├── Code development & testing
├── Data preparation & validation  
├── Model serving & inference
├── API development
└── Unit testing

Cloud Training (Google Colab)
├── LoRA fine-tuning
├── Hyperparameter experiments
└── Large dataset training
```

## 🔄 Typical Workflow

### 1. Local Development
```bash
# Develop and test your code locally
python -m pytest tests/
python src/llama_mapper/cli.py --help

# Prepare training data
python examples/prepare_training_data.py

# Test inference pipeline (with pre-trained model)
python examples/test_inference.py
```

### 2. Cloud Training
```bash
# Upload notebook to Google Colab
# notebooks/llama_mapper_colab.ipynb

# Train model in Colab (15-30 minutes)
# Download trained model checkpoint
```

### 3. Local Integration
```bash
# Use trained model locally
python examples/load_trained_model.py

# Deploy to serving infrastructure
python src/llama_mapper/serving/server.py
```

## 📁 Project Structure

```
llama-mapper/
├── src/llama_mapper/           # Core library (develop locally)
│   ├── training/               # Training components
│   ├── serving/                # Inference & API
│   ├── data/                   # Data processing
│   └── models/                 # Model utilities
├── notebooks/                  # Colab training notebooks
│   └── llama_mapper_colab.ipynb
├── examples/                   # Usage examples
├── tests/                      # Unit tests (run locally)
└── checkpoints/                # Downloaded model checkpoints
```

## 🛠️ Local Development Setup

### Prerequisites
```bash
# Your system is fine for development (no GPU needed)
python -m pip install -e .
```

### What You Can Do Locally
- ✅ **Code Development**: Write and test all Python code
- ✅ **Data Processing**: Prepare training datasets
- ✅ **Unit Testing**: Run pytest on all components
- ✅ **Model Loading**: Load and use pre-trained models
- ✅ **Inference**: Run inference with trained checkpoints
- ✅ **API Development**: Build FastAPI serving endpoints
- ✅ **Integration Testing**: Test end-to-end workflows

### What Requires Cloud
- 🚀 **Model Training**: LoRA fine-tuning (GPU intensive)
- 🚀 **Large Experiments**: Hyperparameter sweeps
- 🚀 **Big Datasets**: Training on 1000s of examples

## 🔗 Integration Points

### 1. Checkpoint Management
```python
# After Colab training, download and use locally
from llama_mapper.training import CheckpointManager

manager = CheckpointManager("./downloaded_checkpoints")
model, tokenizer, metadata = manager.load_checkpoint("mapper-lora@v1.0.0")
```

### 2. Model Serving
```python
# Use trained model in your local API
from llama_mapper.serving import ModelServer

server = ModelServer(checkpoint_path="./downloaded_checkpoints/mapper-lora@v1.0.0")
server.start()
```

### 3. Data Pipeline
```python
# Prepare data locally, train in Colab
from llama_mapper.data import TrainingDataGenerator

generator = TrainingDataGenerator()
training_data = generator.generate_dataset(size=1000)
# Upload to Colab for training
```

## 🧪 Testing Strategy

### Local Testing (Fast)
```bash
# Unit tests for all components
python -m pytest tests/ -v

# Integration tests with mock models
python -m pytest tests/integration/ -v

# Data pipeline tests
python -m pytest tests/test_data_pipeline.py -v
```

### Cloud Testing (Slow)
```bash
# Full training pipeline test in Colab
# Use small dataset for quick validation
```

## 📦 Deployment Workflow

### 1. Development
```bash
# Local development and testing
git add .
git commit -m "Add new feature"
git push origin feature-branch
```

### 2. Training
```bash
# Train in Colab with production data
# Download checkpoint: model_v1.2.3.zip
```

### 3. Integration
```bash
# Unzip checkpoint locally
unzip model_v1.2.3.zip -d ./checkpoints/

# Test locally
python examples/test_checkpoint.py --checkpoint ./checkpoints/model_v1.2.3

# Deploy to production
python deploy.py --checkpoint ./checkpoints/model_v1.2.3
```

## 🎯 Best Practices

### Version Control
- ✅ **Code**: Git repository (all source code)
- ✅ **Models**: Checkpoint manager (versioned models)
- ✅ **Data**: Data versioning (training datasets)
- ✅ **Experiments**: Colab notebooks with results

### Development Cycle
1. **Develop locally** → Fast iteration on code
2. **Test locally** → Unit tests and integration tests  
3. **Train in cloud** → GPU-intensive fine-tuning
4. **Validate locally** → Test trained model
5. **Deploy** → Production serving

### Resource Optimization
- **Local**: CPU-only development and testing
- **Cloud**: GPU training when needed
- **Cost**: Free Colab for experiments, paid for production

## 🚀 Getting Started

### 1. Set Up Local Environment
```bash
# Clone and install
git clone <your-repo>
cd llama-mapper
python -m pip install -e .

# Run tests to verify setup
python -m pytest tests/ -v
```

### 2. Prepare Training Data
```bash
# Generate or prepare your training dataset
python examples/prepare_training_data.py
```

### 3. Train in Colab
- Open `notebooks/llama_mapper_colab.ipynb` in Google Colab
- Upload your training data
- Run training (15-30 minutes)
- Download checkpoint

### 4. Use Trained Model Locally
```bash
# Load and test your trained model
python examples/test_trained_model.py --checkpoint ./checkpoints/your_model
```

This workflow gives you the best of both worlds: fast local development and powerful cloud training!