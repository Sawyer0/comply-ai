# Technology Stack

## Core Technologies

- **Python 3.11+**: Primary development language
- **FastAPI**: Web framework for API endpoints
- **Pydantic**: Data validation and settings management
- **PyTorch**: Deep learning framework
- **Transformers + PEFT**: Model fine-tuning with LoRA
- **vLLM/TGI**: High-performance model serving
- **Redis**: Caching, rate limiting, and idempotency backend
- **PostgreSQL**: Primary database (asyncpg driver)
- **ClickHouse**: Analytics database
- **S3/MinIO**: Object storage for models and artifacts
- **Vault**: Secrets management (HashiCorp Vault)
- **OPA**: Policy enforcement (Open Policy Agent)

## Development Tools

- **pytest**: Testing framework with asyncio support
- **black + isort**: Code formatting
- **mypy**: Type checking
- **flake8**: Linting
- **pre-commit**: Git hooks for code quality
- **Docker**: Containerization
- **Prometheus + Grafana**: Monitoring and observability

## Common Commands

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/
```

### CLI Operations
```bash
# Validate configuration
mapper validate-config --config config.yaml

# Show effective configuration
mapper show-config --tenant acme --environment production

# Detector management
mapper detectors lint --data-dir ./.kiro/pillars-detectors
mapper detectors add --name sample-detector --output-dir ./pillars-detectors
mapper detectors fix --data-dir ./.kiro/pillars-detectors --apply --threshold 0.9

# Confidence thresholds
mapper set-threshold --detector deberta-toxicity --threshold 0.7
mapper show-thresholds

# Orchestration service CLI
orch --help
orch registry list
orch policy validate
```

### Docker Operations
```bash
# Build image
docker build -t llama-mapper .

# Run with monitoring stack
docker-compose -f docker-compose.prometheus.yml up

# Performance testing (rules-only mode)
wsl.exe -e bash -lc '/mnt/c/Users/Dawan/comply-ai/scripts/serve_rules_only.sh start'

# Detector orchestration service
cd detector-orchestration
docker build -t detector-orchestration .
docker-compose -f docker-compose.dev.yaml up
```

### Testing
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v -m integration

# Performance tests
python -m pytest tests/performance/ -v -m performance

# Coverage report
python -m pytest --cov=src/llama_mapper --cov-report=html
```

## Configuration Management

- **Hierarchical Config**: Default settings → YAML → Environment variables
- **Environment Prefix**: `LLAMA_MAPPER_` for mapper service, `ORCH_` for orchestration
- **Nested Config**: Use `__` for nested keys (e.g., `LLAMA_MAPPER_MODEL__TEMPERATURE`)
- **Privacy Mode**: Enabled by default, prevents raw content logging
- **API Schema Migration**: Legacy DetectorRequest deprecated, use MapperPayload (sunset Oct 2025)
- **Rate Limiting**: Configurable per API key/tenant/IP with Redis backend support
- **Idempotency**: Redis-backed caching for consistent responses

## Enhanced Architecture Patterns

### Observability & Monitoring
- **Distributed Tracing**: OpenTelemetry with Jaeger for request flow tracking
- **Business Metrics**: Domain-specific metrics beyond technical monitoring
- **Anomaly Detection**: ML-based performance monitoring with automated alerting
- **Correlation IDs**: Request tracking across all services and logs

### Security Enhancements
- **Automated Secrets Rotation**: HashiCorp Vault integration with scheduled rotation
- **Multi-Layer Input Sanitization**: Comprehensive protection against injection attacks
- **Service Mesh Security**: Istio with mTLS and authorization policies
- **Field-Level Encryption**: Protect sensitive data with automatic encryption/decryption

### Scalability Patterns
- **Event Sourcing**: Immutable audit trail with event projections for compliance
- **CQRS**: Separate read/write models for optimal performance
- **Stream Processing**: Apache Kafka for real-time compliance analysis
- **Batch Optimization**: Parallel processing for large dataset operations
- **Database Scaling**: Read replicas with intelligent load balancing