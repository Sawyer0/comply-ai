# Llama Mapper

Fine-tuned Llama model for mapping detector outputs to canonical taxonomy.

## Overview

Llama Mapper is a privacy-first, audit-ready service that normalizes raw detector outputs into a canonical taxonomy for compliance evidence generation. The system uses LoRA fine-tuning on Llama-3-8B-Instruct to create deterministic mappings from various AI safety detectors to standardized labels.

## Features

- **Privacy-First**: Metadata-only logging, no raw detector inputs persisted
- **Configurable**: YAML-based configuration with environment overrides
- **Audit-Ready**: Framework mapping for SOC 2, ISO 27001, and HIPAA compliance
- **High Performance**: vLLM/TGI serving with confidence-based fallback
- **Versioned**: Full version tracking for taxonomy, models, and frameworks

## Quick Start

### Installation

```bash
pip install -e .
```

### Configuration

Copy and customize the configuration file:

```bash
cp config.yaml my-config.yaml
# Edit my-config.yaml with your settings
```

### Validate Configuration

```bash
mapper validate-config --config my-config.yaml
```

### Set Confidence Thresholds

```bash
# Set threshold for specific detector
mapper set-threshold --detector deberta-toxicity --threshold 0.7

# Show all thresholds
mapper show-thresholds
```

## Configuration

The system supports hierarchical configuration:

1. **Default settings** from `Settings` class
2. **YAML configuration** file (config.yaml)
3. **Environment variable** overrides

### Environment Variables

All configuration can be overridden with environment variables using the prefix `LLAMA_MAPPER_`:

```bash
export LLAMA_MAPPER_MODEL__TEMPERATURE=0.2
export LLAMA_MAPPER_CONFIDENCE__DEFAULT_THRESHOLD=0.7
export LLAMA_MAPPER_SERVING__PORT=8080
```

### Key Configuration Sections

- **model**: Base model settings and generation parameters
- **lora**: LoRA fine-tuning configuration (r=16, α=32)
- **training**: Training hyperparameters (lr=2e-4, epochs=2)
- **confidence**: Confidence thresholds and calibration
- **serving**: API serving configuration (vLLM/TGI)
- **storage**: S3 and database settings
- **logging**: Privacy-first logging configuration
- **security**: API keys and encryption settings

## Privacy and Security

### Privacy-First Logging

The system implements metadata-only logging as required by compliance:

- ✅ Logs: tenant_id, detector_type, canonical_label, confidence
- ❌ Never logs: raw detector inputs, user content, prompts

### Data Protection

- AES256-KMS encryption with BYOK support
- Tenant isolation with scoped data access
- Immutable S3 storage with WORM configuration
- 90-day hot data retention with automated cleanup

## Development

### Project Structure

```
src/llama_mapper/
├── config/           # Configuration management
│   ├── manager.py    # ConfigManager class
│   └── settings.py   # Pydantic settings models
├── utils/            # Utility modules
│   └── logging.py    # Privacy-first logging
└── cli.py           # Command-line interface
```

### Requirements

- Python 3.11+
- FastAPI for API serving
- Transformers + PEFT for model fine-tuning
- vLLM/TGI for model serving
- Structured logging with privacy filters

## License

MIT License - see LICENSE file for details.