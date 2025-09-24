# 📊 Comply-AI Training Datasets Documentation

## Overview

This document consolidates all training datasets used for our compliance intelligence models, including both currently integrated datasets and recommended additions from recent research.

## 🎯 Model-Specific Dataset Strategies

### **Llama-3-8B Training Datasets**
- **Primary Use**: Comprehensive compliance mapping and analysis
- **Memory Requirements**: ~16-20GB VRAM with QLoRA
- **Training Time**: ~3 hours on H100 GPU
- **Optimal LoRA Config**: r=256, alpha=512

### **Phi-3-Mini-4K Training Datasets** 
- **Primary Use**: Efficient compliance mapping with lower resource requirements
- **Memory Requirements**: ~8-12GB VRAM
- **Training Approach**: Supervised Fine-tuning + DPO
- **Strengths**: Exceptional instruction following and reasoning

## ✅ Currently Integrated Datasets

### **Hugging Face Datasets (Active)**

#### **Llama Mapper Model:**
- **ai4privacy/pii-masking-300k**: PII detection examples for privacy compliance
- **cgoosen/llm_guard_dataset**: Prompt injection detection for security analysis
- **ibm-research/AttaQ**: Attack pattern detection for security threats
- **GotThatData/nist-cybersecurity-framework**: NIST framework for security control mapping
- **allenai/wildguardmix**: Content toxicity detection for harmful content analysis
- **qa4pc/QA4PC**: Policy compliance Q&A for compliance scenario training
- **pile-of-law/pile-of-law**: Legal documents for regulatory analysis

#### **Compliance Analyst Model:**
- **AndreaSimeri/GDPR**: Complete GDPR regulation text for legal compliance training
- **pile-of-law/pile-of-law**: 256GB of legal documents for regulatory analysis
- **ai4privacy/pii-masking-300k**: PII detection examples for privacy compliance
- **qa4pc/QA4PC**: Policy compliance Q&A for compliance scenario training

### **Dataset Statistics (Current)**
- **Total Examples**: 242 validated examples
- **Taxonomy Coverage**: 80% (28/35 labels covered)
- **Quality Score**: 92.3% overall
- **Confidence Range**: 0.60-0.95 (avg: 0.79)

## 🚀 Recommended Additional Datasets

### **High-Priority Additions (Ready to Download)**

#### **Hugging Face - Immediate Access**
```python
from datasets import load_dataset

# Enhanced PII Detection (43k examples)
pii_enhanced = load_dataset("ai4privacy/pii-masking-43k")

# Legal Reasoning Tasks
legal_bench = load_dataset("nguha/legalbench")

# GDPR Complete Dataset
gdpr_complete = load_dataset("AndreaSimeri/GDPR")
```

#### **Kaggle Datasets - Direct Download**
- **GDPR Violations**: Real enforcement cases and penalties
- **Employee Policy Compliance Dataset**: 4,000 structured compliance records
- **FDA Enforcement Actions**: Regulatory enforcement data
- **Anti Money Laundering Dataset**: AML transaction monitoring
- **Audit Findings Dataset**: Data analytics in auditing practices

#### **GitHub Open Source Projects**
- **getprobo/probo**: Open-source SOC-2 compliance platform
- **trycompai/comp**: Multi-framework compliance platform (SOC 2, ISO 27001, HIPAA, GDPR)
- **compliance-framework**: Cloud compliance framework with OSCAL configurations
- **ThreatNG Security**: Governance and compliance data repositories

## 📋 Dataset Integration Plan

### **Phase 1: Enhanced PII Detection (Week 1)**
```python
# Priority: ai4privacy/pii-masking-43k
# Benefits: 43k examples, real-world patterns, confidence scores
# Integration: Extend existing PII detection capabilities
```

### **Phase 2: Legal Reasoning Enhancement (Week 2)**
```python
# Priority: nguha/legalbench
# Benefits: Structured legal reasoning tasks
# Integration: Improve compliance interpretation accuracy
```

### **Phase 3: Real-World Enforcement Cases (Week 3)**
```python
# Priority: Kaggle GDPR Violations, FDA Enforcement Actions
# Benefits: Real enforcement scenarios, penalty data
# Integration: Add enforcement prediction capabilities
```

### **Phase 4: Multi-Framework Compliance (Week 4)**
```python
# Priority: trycompai/comp, compliance-framework
# Benefits: Multi-framework patterns, OSCAL configurations
# Integration: Expand beyond current GDPR/SEC/HIPAA coverage
```

## 🔧 Training Configuration Updates

### **Optimal LoRA Configuration**
```python
lora_config = {
    "r": 256,                    # Rank - optimal for performance
    "lora_alpha": 512,          # Alpha = 2x rank rule
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none"
}
```

### **Memory-Optimized Training**
```python
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # Effective batch size = 32
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "fp16": True,  # Memory optimization
}
```

### **QLoRA Quantization**
```python
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
}
```

## 📊 Dataset Quality Metrics

### **Current Coverage Analysis**
- **Detector Coverage**: 100% (5/5 detectors covered)
- **Taxonomy Coverage**: 80% (28/35 labels covered)
- **Category Coverage**: 67% (4/6 categories covered)

### **Coverage Gaps to Address**
- **BIAS categories**: Gender, Race, Religion, Other
- **HARM.VIOLENCE.Suicide**: Self-harm detection
- **OTHER categories**: ModelError, Unknown

### **Quality Improvements Needed**
- **Label Balance**: Some categories over-represented
- **Edge Cases**: More complex compliance scenarios
- **Multi-Framework**: Cross-regulatory mapping examples

## 🎯 Training Data Format Standards

### **Compliance-Specific Format**
```json
{
  "instruction": "Map this PII detection to GDPR compliance requirements",
  "input": "Detected: email address john@company.com, phone number +1-555-0123",
  "output": "GDPR Article 6 (Lawfulness): Requires explicit consent or legitimate interest. Article 32 (Security): Implement pseudonymization and encryption. Article 17 (Right to erasure): Enable data deletion upon request.",
  "context": "User profile data processing",
  "framework": "GDPR",
  "confidence": 0.95
}
```

### **Multi-Turn Compliance Conversations**
```json
{
  "conversation": [
    {
      "role": "user",
      "content": "We detected PII in our customer database. What compliance actions are required?"
    },
    {
      "role": "assistant", 
      "content": "First, identify the regulatory frameworks applicable to your jurisdiction. For GDPR, you need to: 1) Document the legal basis for processing, 2) Implement appropriate security measures, 3) Ensure data subject rights are accessible..."
    }
  ],
  "metadata": {
    "task_type": "compliance_advisory",
    "frameworks": ["GDPR", "CCPA"]
  }
}
```

## 📈 Performance Validation

### **Compliance-Specific Evaluation Metrics**
- **Regulatory Accuracy**: Mapping consistency across frameworks
- **Structured Output Quality**: JSON formatting and confidence scores
- **Context Understanding**: Multi-detector input processing
- **Audit Trail Generation**: Provenance tracking and evidence linking

### **Expected Improvements with New Datasets**
- **Taxonomy Coverage**: 80% → 95%+ with additional datasets
- **Multi-Framework Support**: Current 3 → 8+ frameworks
- **Real-World Accuracy**: Enhanced with enforcement case data
- **Edge Case Handling**: Improved with complex scenario training

## 🔄 Dataset Versioning & Management

### **Version Control**
- **Current Version**: v1.2 (242 examples, 80% coverage)
- **Target Version**: v2.0 (1000+ examples, 95% coverage)
- **Update Schedule**: Weekly integration of new datasets

### **Quality Gates**
- **Format Validation**: 100% compliance with schema
- **Taxonomy Validation**: All labels must be in approved taxonomy
- **Confidence Thresholds**: Minimum 0.6 confidence for training examples
- **Balance Requirements**: No single category >40% of dataset

## 📚 References

### **Dataset Sources**
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Kaggle Compliance Datasets](https://www.kaggle.com/search?q=compliance)
- [GitHub Compliance Projects](https://github.com/topics/compliance)

### **Training Optimization**
- [LoRA Fine-tuning Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [QLoRA Memory Optimization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [Gradient Accumulation Best Practices](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

**Last Updated**: 2025-01-27  
**Next Review**: 2025-02-03  
**Maintainer**: Comply-AI Development Team
