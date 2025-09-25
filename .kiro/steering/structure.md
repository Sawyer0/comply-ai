# Project Structure

## Root Directory Layout

```
llama-mapper/
├── detector-orchestration/    # SERVICE 1: Detector Orchestration Service
├── analysis-service/          # SERVICE 2: Analysis Service  
├── mapper-service/            # SERVICE 3: Mapper Service
├── shared/                    # Shared libraries and utilities
├── tests/                     # Cross-service integration tests
├── docs/                      # Documentation
├── examples/                  # Usage examples and demos
├── scripts/                   # Utility and automation scripts
├── config/                    # Global configuration files
├── notebooks/                 # Jupyter notebooks for training
├── infra/                     # Infrastructure as code
├── perf/                      # Performance testing
├── schemas/                   # JSON schemas for validation
├── analysis/                  # Analysis results and data
├── checkpoints/               # Model checkpoints (gitignored)
├── htmlcov/                   # Coverage reports
└── .kiro/                     # Kiro IDE configuration
```

## Microservice Organization

### Detector Orchestration Service (`detector-orchestration/`)
```
detector-orchestration/
├── src/orchestration/
│   ├── api/                   # HTTP API endpoints
│   ├── core/                  # Core orchestration logic
│   ├── policy/                # Policy management
│   ├── discovery/             # Service discovery
│   ├── resilience/            # Circuit breakers, retry logic
│   ├── cache/                 # Caching layer
│   ├── security/              # Authentication, WAF, RBAC
│   ├── monitoring/            # Metrics and observability
│   ├── tenancy/               # Multi-tenancy support
│   ├── plugins/               # Plugin system
│   ├── pipelines/             # Pipeline management
│   ├── cli/                   # CLI commands
│   └── config/                # Configuration
├── tests/                     # Service-specific tests
└── docker/                    # Docker configurations
```

### Analysis Service (`analysis-service/`)
```
analysis-service/
├── src/analysis/
│   ├── api/                   # HTTP API endpoints
│   ├── engines/               # Analysis engines
│   │   ├── core/              # Primary engines
│   │   ├── statistical/       # Statistical components
│   │   └── optimization/      # Optimization engines
│   ├── ml/                    # ML components
│   ├── rag/                   # RAG system
│   ├── quality/               # Quality system
│   ├── privacy/               # Privacy controls
│   ├── security/              # Security components
│   ├── infrastructure/        # Infrastructure components
│   ├── resilience/            # Resilience patterns
│   ├── tenancy/               # Multi-tenancy support
│   ├── plugins/               # Plugin system
│   ├── pipelines/             # Pipeline management
│   ├── taxonomy/              # Taxonomy management
│   ├── schemas/               # Schema management
│   ├── cli/                   # CLI commands
│   └── config/                # Configuration
├── tests/                     # Service-specific tests
└── docker/                    # Docker configurations
```

### Mapper Service (`mapper-service/`)
```
mapper-service/
├── src/mapper/
│   ├── api/                   # HTTP API endpoints
│   ├── core/                  # Core mapping logic
│   ├── ml/                    # ML components
│   ├── serving/               # Model serving infrastructure
│   ├── taxonomy/              # Taxonomy management
│   ├── schemas/               # Schema management
│   ├── validation/            # Validation components
│   ├── fallback/              # Fallback mechanisms
│   ├── monitoring/            # Monitoring & observability
│   ├── deployment/            # Deployment management
│   ├── tenancy/               # Multi-tenancy support
│   ├── plugins/               # Plugin system
│   ├── pipelines/             # Pipeline management
│   ├── cli/                   # CLI commands
│   └── config/                # Configuration
├── tests/                     # Service-specific tests
└── docker/                    # Docker configurations
```

## Key Directories

### `/tests/`
- `unit/` - Unit tests for individual components
- `integration/` - Integration tests across components
- `performance/` - Performance and load tests
- `security/` - Security-focused tests
- `fixtures/` - Test data and fixtures

### `/docs/`
- `adr/` - Architecture Decision Records
- `deployment/` - Deployment guides
- `runbooks/` - Operational runbooks
- `contracts/` - API contracts and schemas
- `openapi.yaml` - OpenAPI specification

### `/examples/`
- Usage examples for different scenarios
- Demo scripts and sample data
- Client SDK examples (Python, JS, Go, Java, C#)

### `/scripts/`
- Automation and utility scripts
- Database backup/restore scripts
- Training data generation
- CI/CD helper scripts

### `/config/`
- Default configuration files
- Environment-specific configs
- Monitoring configurations (Prometheus, Grafana)

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Config files**: `kebab-case.yaml` or `snake_case.json`
- **Test files**: `test_*.py`

## Import Organization

```python
# Standard library imports
import os
from typing import Dict, List, Optional

# Third-party imports
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from llama_mapper.config import Settings
from llama_mapper.models import MapperModel
```

## Configuration File Locations

- **Main config**: `config.yaml` (root)
- **Environment configs**: `config/`
- **API keys**: `api_keys.json` (root, gitignored)
- **Detector configs**: `.kiro/pillars-detectors/`
- **Kiro settings**: `.kiro/steering/`
- **Orchestration config**: `detector-orchestration/` (separate service)

## Detector Orchestration Service

The `detector-orchestration/` directory contains a separate FastAPI service:

```
detector-orchestration/
├── src/detector_orchestration/
│   ├── api/                   # API endpoints
│   ├── service_discovery/     # Service discovery
│   ├── service_factory/       # Service factory patterns
│   ├── aggregator.py         # Result aggregation
│   ├── cache.py              # Caching layer
│   ├── circuit_breaker.py    # Circuit breaker pattern
│   ├── coordinator.py        # Request coordination
│   ├── mapper_client.py      # Mapper service client
│   ├── policy.py             # Policy enforcement
│   └── registry.py           # Detector registry
├── charts/                   # Helm charts
├── policies/                 # OPA policies
└── tests/                    # Service-specific tests
```

## Development Workflow

1. **Local Development**: Code in `src/`, test with `pytest`
2. **Cloud Training**: Use notebooks in `notebooks/` (Google Colab)
3. **Model Checkpoints**: Store in `checkpoints/` (gitignored)
4. **Docker Builds**: Multi-stage builds with caching
5. **CI/CD**: GitHub Actions with quality gates