# Llama Mapper Microservice Architecture

This document describes the microservice architecture for the llama-mapper system, consisting of 3 focused services with clear domain boundaries and distributed system best practices.

## Architecture Overview

The system has been refactored into exactly **3 microservices** following Single Responsibility Principle:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              Detector Orchestration Service                     │
│  Port: 8000 | Domain: Coordination & Policy                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   Analysis Service                              │
│  Port: 8001 | Domain: Advanced Analysis & Intelligence         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Mapper Service                               │
│  Port: 8002 | Domain: Core Mapping & Generation                │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### 1. Detector Orchestration Service (`detector-orchestration/`)
**Port**: 8000  
**Domain**: Detector coordination, policy enforcement, service discovery

**Responsibilities**:
- Detector health monitoring and circuit breakers
- Policy enforcement with OPA integration
- Multi-tenant routing and isolation
- Rate limiting and authentication
- Async job processing and pipeline orchestration

**Key Features**:
- WAF integration and security controls
- Redis-backed caching and idempotency
- Prometheus metrics and distributed tracing
- Plugin system for detectors and policies

### 2. Analysis Service (`analysis-service/`)
**Port**: 8001  
**Domain**: Advanced analysis, risk scoring, compliance intelligence

**Responsibilities**:
- Pattern recognition (temporal, frequency, correlation, anomaly)
- Risk scoring with ML enhancement (Phi-3 model)
- Compliance framework mapping (SOC2, ISO27001, HIPAA, GDPR)
- RAG system with regulatory knowledge
- Quality monitoring and evaluation

**Key Features**:
- Privacy-first architecture (metadata-only logging)
- ClickHouse for analytics data
- Weekly evaluation services
- Multi-tenant analytics and customization

### 3. Mapper Service (`mapper-service/`)
**Port**: 8002  
**Domain**: Core mapping, model serving, response generation

**Responsibilities**:
- Core mapping logic and taxonomy management
- ML model serving (Llama-3-8B with LoRA fine-tuning)
- Response validation and formatting
- Cost monitoring and optimization
- Model versioning and deployment

**Key Features**:
- vLLM/TGI high-performance model serving
- MinIO/S3 model storage
- Canary deployments and A/B testing
- Comprehensive cost tracking

## Shared Components (`shared/`)

Common libraries and utilities used across all services:

- **Interfaces**: Service contracts and data models
- **Utils**: Correlation IDs, logging, metrics
- **Exceptions**: Standardized error handling
- **Models**: Shared domain models

## Quick Start

### Development Setup

1. **Clone and setup each service**:
```bash
# Setup Detector Orchestration Service
cd detector-orchestration
pip install -e .
pip install -e ../shared

# Setup Analysis Service
cd ../analysis-service
pip install -e .
pip install -e ../shared

# Setup Mapper Service
cd ../mapper-service
pip install -e .
pip install -e ../shared
```

2. **Start all services with Docker Compose**:
```bash
# Start all microservices
docker-compose -f docker-compose.microservices.yml up --build

# Or start individual services
cd detector-orchestration && docker-compose up
cd analysis-service && docker-compose up
cd mapper-service && docker-compose up
```

3. **Verify services are running**:
```bash
# Health checks
curl http://localhost:8000/health  # Orchestration
curl http://localhost:8001/health  # Analysis
curl http://localhost:8002/health  # Mapper

# Metrics
curl http://localhost:8000/metrics  # Orchestration metrics
curl http://localhost:8001/metrics  # Analysis metrics
curl http://localhost:8002/metrics  # Mapper metrics
```

### Service Communication Flow

```bash
# Example: Complete processing flow
curl -X POST http://localhost:8000/api/v1/orchestrate \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "content": "Sample content for analysis",
    "detector_types": ["presidio", "deberta"],
    "processing_mode": "standard"
  }'
```

## Configuration

Each service has independent configuration:

- **Environment Variables**: Service-specific prefixes (`ORCHESTRATION_`, `ANALYSIS_`, `MAPPER_`)
- **YAML Configuration**: `config/settings.yaml` in each service
- **Docker Environment**: Environment-specific docker-compose files

### Key Configuration Areas

1. **Database**: Each service has its own database schema
2. **Redis**: Shared Redis with different database numbers
3. **Security**: Independent API keys and authentication
4. **Monitoring**: Service-specific metrics and tracing
5. **ML Models**: Service-specific model configuration

## Development Guidelines

### Service Boundaries
- **No Shared Code**: Each service is completely independent
- **HTTP APIs Only**: Services communicate via well-defined HTTP contracts
- **Database Isolation**: Each service has its own database schema
- **Independent Deployment**: Services can be deployed independently

### Adding New Features
1. **Identify Service Domain**: Determine which service owns the feature
2. **Design API Contract**: Define HTTP API with OpenAPI spec
3. **Implement with Fallbacks**: Always provide non-ML fallback mechanisms
4. **Add Monitoring**: Include metrics, logging, and health checks
5. **Test Thoroughly**: Unit, integration, and contract tests

### Code Quality Standards
- Type hints for all functions
- Comprehensive error handling
- Structured logging with correlation IDs
- 80%+ test coverage
- Security-first development

## Monitoring & Observability

### Metrics (Prometheus)
- **Service Metrics**: Request count, latency, errors
- **Business Metrics**: Domain-specific KPIs
- **Resource Metrics**: CPU, memory, GPU utilization
- **Cost Metrics**: Token usage, inference costs

### Tracing (Jaeger)
- **Distributed Tracing**: Request flow across services
- **Correlation IDs**: Request correlation tracking
- **Performance Analysis**: Detailed performance insights
- **Error Tracking**: Error propagation analysis

### Logging (Structured)
- **JSON Logging**: Structured log format
- **Privacy Controls**: Automatic PII scrubbing
- **Correlation IDs**: Request correlation in logs
- **Service Context**: Service-specific context

### Dashboards (Grafana)
- **Service Dashboards**: Per-service monitoring
- **Business Dashboards**: Domain-specific metrics
- **Infrastructure Dashboards**: Resource monitoring
- **Cost Dashboards**: Cost tracking and optimization

## Security

### Authentication & Authorization
- **API Keys**: Service-to-service authentication
- **JWT Tokens**: User authentication
- **RBAC**: Role-based access control
- **Multi-Factor**: Enhanced authentication

### Privacy Controls
- **Metadata-Only Logging**: No raw content persistence
- **Content Scrubbing**: Automatic PII removal
- **Data Minimization**: Minimal data retention
- **Field Encryption**: Sensitive data encryption

### Network Security
- **Service Mesh**: Istio with mTLS
- **Network Policies**: Kubernetes network policies
- **WAF Protection**: Web application firewall
- **Rate Limiting**: Request rate controls

## Deployment

### Deployment Strategies
- **Blue-Green**: Zero-downtime deployments
- **Canary**: Gradual rollout with monitoring
- **A/B Testing**: Model and algorithm testing
- **Feature Flags**: Runtime feature toggling

### Scaling
- **Horizontal Scaling**: Multiple service instances
- **Auto-scaling**: Dynamic scaling based on metrics
- **Load Balancing**: Intelligent request routing
- **Resource Management**: Optimal resource allocation

### Infrastructure
- **Kubernetes**: Container orchestration
- **Docker**: Containerization
- **Helm**: Package management
- **Terraform**: Infrastructure as code

## Testing

### Testing Strategy
- **Unit Tests**: Service-specific business logic
- **Integration Tests**: Service-to-service communication
- **Contract Tests**: API contract validation
- **End-to-End Tests**: Complete workflow validation
- **Performance Tests**: Load and stress testing

### Test Organization
```
tests/
├── unit/           # Service-specific unit tests
├── integration/    # Cross-service integration tests
├── contracts/      # API contract tests
├── e2e/           # End-to-end workflow tests
└── performance/   # Load and performance tests
```

## Migration from Monolith

The microservice architecture preserves all existing functionality while providing:

1. **Better Separation of Concerns**: Clear domain boundaries
2. **Independent Scaling**: Scale services based on demand
3. **Technology Flexibility**: Different tech stacks per service
4. **Team Autonomy**: Independent development and deployment
5. **Fault Isolation**: Service failures don't cascade
6. **Easier Maintenance**: Smaller, focused codebases

## Contributing

1. **Follow Service Boundaries**: Respect domain boundaries
2. **API-First Design**: Design APIs before implementation
3. **Comprehensive Testing**: Test all service interactions
4. **Security Compliance**: Follow security best practices
5. **Documentation**: Update documentation for changes

## Troubleshooting

### Common Issues

1. **Service Communication**: Check network connectivity and API contracts
2. **Database Connections**: Verify database URLs and credentials
3. **Model Loading**: Check model cache and storage configuration
4. **Resource Limits**: Monitor CPU, memory, and GPU usage
5. **Configuration**: Validate environment variables and YAML files

### Debugging Tools

- **Health Checks**: Service health endpoints
- **Metrics**: Prometheus metrics for monitoring
- **Logs**: Structured logs with correlation IDs
- **Tracing**: Jaeger for distributed tracing
- **Profiling**: Performance profiling tools

## License

MIT License - see LICENSE file for details.

---

For service-specific documentation, see the README files in each service directory:
- [Detector Orchestration Service](detector-orchestration/README.md)
- [Analysis Service](analysis-service/README.md)
- [Mapper Service](mapper-service/README.md)
- [Shared Libraries](shared/README.md)