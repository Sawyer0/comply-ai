# Fine-Tuning Preparation Guide

## 🚀 Night-Before Checklist for Model Fine-Tuning

This guide provides a comprehensive preparation checklist for your model fine-tuning session tomorrow. All configurations, scripts, and validations are ready to ensure optimal training performance.

## 📋 Quick Reference Checklist

- [ ] **Tokenizer & Chat Template**: Use each base model's own tokenizer and chat template
- [ ] **Sequence Length**: Set based on 95th percentile analysis (Mapper: 512-768, Analyst: 768-1024)
- [ ] **Packing**: Enable with tokens-per-step targets (Mapper: 32k-64k, Analyst: 16k-32k)
- [ ] **Output Caps**: Set max_new_tokens (Mapper: 32-64, Analyst: 128-256)
- [ ] **API Guardrails**: Add input token limits and logging
- [ ] **Evaluation Parity**: Ensure gold sets use same tokenization as training

## 🛠️ Preparation Scripts

### 1. Token Length Analysis
```bash
python scripts/analyze_token_lengths.py \
  --mapper-data llm/llm-reports/enhanced_hybrid_sample.jsonl \
  --analyst-data llm/llm-reports/enhanced_hybrid_sample.jsonl \
  --generate-plots
```

### 2. Packing Configuration
```bash
python scripts/configure_packing.py \
  --mapper-analysis analysis/token_lengths/mapper_analysis.json \
  --analyst-analysis analysis/token_lengths/analyst_analysis.json
```

### 3. Evaluation Parity Validation
```bash
python scripts/validate_evaluation_parity.py \
  --mapper-gold tests/golden_test_cases_comprehensive.json \
  --analyst-gold tests/golden_test_cases_comprehensive.json
```

### 4. Complete Pre-Training Validation
```bash
python scripts/pre_training_checklist.py
```

## ⚙️ Configuration Files

### Main Configuration
- **`config/fine_tuning_preparation.yaml`**: Complete preparation configuration
- **`config/dual_model_training_config.json`**: Training hyperparameters
- **`config/optimized_training_config.json`**: Generated optimized configuration

### Key Settings

#### Tokenizer Configuration
```yaml
tokenizer_config:
  mapper:
    model_name: "meta-llama/Llama-3-8B-Instruct"
    chat_template: "llama-3-instruct"
  analyst:
    model_name: "microsoft/Phi-3-mini-4k-instruct"
    chat_template: "phi-3-instruct"
```

#### Sequence Lengths
```yaml
sequence_lengths:
  mapper:
    max_sequence_length: 768  # 512-768 tokens
    target_95th_percentile: 512
  analyst:
    max_sequence_length: 1024  # 768-1024 tokens
    target_95th_percentile: 768
```

#### Packing Configuration
```yaml
packing_config:
  mapper:
    target_tokens_per_step: 48000  # 32k-64k tokens/step
    batch_size: 4
    gradient_accumulation: 8
  analyst:
    target_tokens_per_step: 24000  # 16k-32k tokens/step
    batch_size: 8
    gradient_accumulation: 4
```

#### Output Limits
```yaml
output_limits:
  mapper:
    max_new_tokens: 64  # Tiny JSON output
  analyst:
    max_new_tokens: 256  # Concise reasoning
```

## 🔧 Implementation Details

### 1. Tokenizer & Chat Template Setup
- **Mapper**: Uses Llama-3-8B-Instruct tokenizer with native chat template
- **Analyst**: Uses Phi-3-Mini tokenizer with native chat template
- **Consistency**: Same tokenization used for train/val/gold/eval

### 2. Sequence Length Optimization
- **Analysis**: 95th percentile token length measurement
- **Outlier Handling**: Drop or chunk sequences above 99th percentile
- **Memory Efficiency**: Avoid unnecessary high sequence lengths

### 3. Packing Strategy
- **Length Bucketing**: Group samples by similar lengths
- **Tokens per Step**: Target steady tokens per optimizer step
- **Gradient Accumulation**: Use to hit target without OOMs

### 4. Output Token Caps
- **Mapper**: 32-64 tokens for tiny JSON responses
- **Analyst**: 128-256 tokens for concise analysis
- **Schema Retry**: One retry with stricter prompt if decoding fails

### 5. API Input Guardrails
- **Mapper**: 1024 token input limit with rules-based fallback
- **Analyst**: 2048 token input limit with chunked analysis fallback
- **Logging**: Token counts (not content) for monitoring

### 6. Evaluation Parity
- **Same Tokenization**: Gold sets use identical tokenizer as training
- **Template Consistency**: Same chat templates for train and eval
- **Re-tokenization**: Required if templates change

## 📊 Performance Targets

### Throughput & Cost Planning
- **Mapper**: 1000 req/sec, 200ms p95 latency, 16GB memory
- **Analyst**: 500 req/sec, 500ms p95 latency, 8GB memory
- **Scaling**: Tokens-based scaling for autoscaling and cost optimization

### Memory Requirements
- **Llama-3-8B (QLoRA)**: ~16GB VRAM with optimized settings
- **Phi-3-Mini**: ~8GB VRAM with 4-bit quantization
- **Batch Sizing**: Optimized for target tokens per step

## 🚨 Critical Success Factors

### 1. Tokenization Consistency
- **Never mix tokenizers** between train/val/eval
- **Freeze tokenizer choice** for entire run
- **Re-tokenize datasets** if templates change

### 2. Sequence Length Management
- **Measure 95th percentile** after tokenization
- **Chunk or drop outliers** instead of pushing seq_len high
- **Keep memory stable** with reasonable limits

### 3. Packing Efficiency
- **Target tokens per step** rather than examples per batch
- **Use grad accumulation** to hit targets without OOMs
- **Monitor packing efficiency** during training

### 4. Output Control
- **Hard caps on max_new_tokens** for latency/compute
- **Schema validation** with retry mechanism
- **Strict system prompts** to prevent rambling

## 🔍 Monitoring & Validation

### Pre-Training Validation
Run the complete validation script to ensure everything is ready:
```bash
python scripts/pre_training_checklist.py
```

### Key Validation Points
1. **Environment**: Python 3.8+, required packages, GPU availability
2. **Configuration**: Valid YAML/JSON, proper file paths
3. **Data**: Training data files present and accessible
4. **Tokenizers**: Loadable with proper special tokens
5. **Sequence Lengths**: Within recommended ranges
6. **Packing**: Analysis completed and optimized
7. **Output Limits**: Set to recommended values
8. **API Guardrails**: Implemented and configured
9. **Evaluation Parity**: Gold sets validated
10. **Performance**: Targets achievable with current setup

### Monitoring During Training
- **Token Usage**: Log input/output token counts
- **Memory Usage**: Monitor VRAM utilization
- **Training Speed**: Track tokens per second
- **Packing Efficiency**: Monitor padding waste

## 🎯 Final Night-Before Actions

1. **Run Complete Validation**:
   ```bash
   python scripts/pre_training_checklist.py
   ```

2. **Verify All Configurations**:
   - Check `config/fine_tuning_preparation.yaml`
   - Verify `config/optimized_training_config.json`
   - Confirm training data paths

3. **Test Tokenizers**:
   ```bash
   python -c "from transformers import AutoTokenizer; print('Mapper:', AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B-Instruct')); print('Analyst:', AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct'))"
   ```

4. **Check GPU Resources**:
   ```bash
   nvidia-smi
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
   ```

5. **Validate Data Access**:
   - Confirm training data files exist
   - Check file sizes and formats
   - Verify gold test cases

## 🚀 Training Day Commands

### Start Training (Mapper)
```bash
python scripts/implement_dual_model_training.py --model mapper --config config/optimized_training_config.json
```

### Start Training (Analyst)
```bash
python scripts/implement_dual_model_training.py --model analyst --config config/optimized_training_config.json
```

### Monitor Training
```bash
# Watch logs
tail -f checkpoints/*/logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training metrics
python scripts/monitor_training.py --checkpoint-dir checkpoints/
```

## 📈 Success Metrics

### Training Metrics
- **Loss Convergence**: Steady decrease in training loss
- **Validation Accuracy**: >95% for mapper, >90% for analyst
- **Training Speed**: Target tokens per second achieved
- **Memory Efficiency**: No OOM errors, stable VRAM usage

### Performance Metrics
- **Latency**: Within target ranges (200ms mapper, 500ms analyst)
- **Throughput**: Meeting target requests per second
- **Token Efficiency**: Minimal padding waste, good packing
- **Output Quality**: Valid JSON, appropriate length

## 🆘 Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or sequence length
2. **Slow Training**: Check GPU utilization, optimize packing
3. **Poor Quality**: Verify tokenization consistency
4. **High Latency**: Reduce max_new_tokens, optimize generation

### Emergency Contacts
- **GPU Issues**: Check `nvidia-smi`, restart if needed
- **Data Issues**: Verify file paths and formats
- **Config Issues**: Use backup configurations in `config/`

---

## 🎉 You're Ready!

With this comprehensive preparation, you're fully equipped for successful fine-tuning tomorrow. The configurations are optimized, validations are in place, and monitoring is ready. Good luck with your training session!

**Remember**: Start with the validation script to ensure everything is perfect before beginning training.
