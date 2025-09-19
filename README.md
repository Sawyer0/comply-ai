# Llama Mapper

Note on Mapper API request schema deprecation:
- The /map and /map/batch endpoints now prefer the MapperPayload request shape (see docs/contracts/mapper_compliance.md and docs/release/mapper_migration.md).
- The legacy DetectorRequest shape is still accepted during a deprecation window; responses include `Deprecation: true` and metrics count usage. Sunset: Fri, 31 Oct 2025 00:00:00 GMT. Removal targeted for 0.3.0.
- Configure mapper payload limits and privacy checks via environment variables:
  - MAPPER_SERVING_MAPPER_TIMEOUT_MS (default 500)
  - MAPPER_SERVING_MAX_PAYLOAD_KB (default 64)
  - MAPPER_SERVING_REJECT_ON_RAW_CONTENT (default true)

Client SDK examples
- See docs/clients/ for Python, JavaScript/TypeScript, Go, Java, C#, and curl examples.

API request examples
- Preferred (MapperPayload):
```json
{
  "detector": "orchestrated-multi",
  "output": "toxic|hate|pii_detected",
  "tenant_id": "tenant-123",
  "metadata": {
    "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
    "aggregation_method": "weighted_average",
    "coverage_achieved": 1.0,
    "provenance": [{"detector":"deberta-toxicity","confidence":0.93}]
  }
}
```
- Deprecated (DetectorRequest):
```json
{
  "detector": "deberta-toxicity",
  "output": "toxic"
}
```

## Rate limiting

This service includes configurable rate limiting for the /map and /map/batch endpoints.

- Identity precedence: API key (X-API-Key) → tenant (X-Tenant-ID) → client IP
- Defaults: 600 req/min per API key or tenant; 120 req/min per IP; 60s window
- Headers returned: RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset and Retry-After (on 429). Legacy X-RateLimit-* headers are also supported.
- Configuration: see docs/runbook/rate-limits.md
- Backend: in-memory by default; set `RATE_LIMIT_BACKEND=redis` and `MAPPER_IDEMPOTENCY_REDIS_URL` to enable a Redis backend for cross-instance limits.

## Idempotency cache (Redis)

- Mapper supports Redis-backed idempotency caching when `MAPPER_IDEMPOTENCY_BACKEND=redis` and `MAPPER_IDEMPOTENCY_REDIS_URL=redis://...` are set. Fallback is in-memory.
- Orchestrator (detector-orchestration) supports Redis-backed idempotency and response caches via env:
  - `ORCH_CONFIG__cache_backend=redis`
  - `ORCH_CONFIG__redis_url=redis://localhost:6379/0`
  - `ORCH_CONFIG__redis_prefix=orch`

Regenerate the OpenAPI spec to capture rate limit headers:

- python scripts/export_openapi.py --output docs/openapi.yaml

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

### CLI quick reference

- Validate all configs (taxonomy/frameworks/detectors):
  ```bash
  mapper validate-config --data-dir ./.kiro/pillars-detectors
  ```
- Inspect effective config with overlays (secrets masked):
  ```bash
  mapper show-config --tenant acme --environment production
  # JSON output
  mapper show-config --tenant acme --environment production --format json
  ```
- Detectors:
  ```bash
  # Lint detector YAMLs against taxonomy
  mapper detectors lint --data-dir ./.kiro/pillars-detectors [--format json]

  # Scaffold a new detector YAML
  mapper detectors add --name sample-detector --output-dir ./pillars-detectors

  # Suggest and optionally apply fixes to invalid labels
  mapper detectors fix --data-dir ./.kiro/pillars-detectors --format json
  mapper detectors fix --data-dir ./.kiro/pillars-detectors --apply --threshold 0.9
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
