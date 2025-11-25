# Mapper Service

The Mapper Service provides core mapping functionality, model serving, and response generation within the llama-mapper microservice architecture. It handles the transformation of analysis results into canonical taxonomy and compliance framework mappings.

## Features

- **Core Mapping**: Canonical taxonomy mapping with high accuracy
- **Model Serving**: Llama-3-8B model serving with vLLM/TGI backends
- **Training Infrastructure**: LoRA fine-tuning and model management
- **Validation**: Comprehensive input/output validation
- **Cost Monitoring**: Real-time cost tracking and optimization
- **Multi-Tenancy**: Tenant-specific models and data isolation
- **Deployment**: Canary deployments and A/B testing

## Quick Start

### Development Setup

```bash
# Setup shared components integration
python setup_shared.py

# Install dependencies (including shared components)
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Set up database environment variables
export MAPPER_DATABASE_HOST=localhost
export MAPPER_DATABASE_PORT=5432
export MAPPER_DATABASE_NAME=mapper_service
export MAPPER_DATABASE_USER=mapper_user
export MAPPER_DATABASE_PASSWORD=your_password

# Run database migrations (if needed)
python -c "from mapper.infrastructure.database_manager import create_database_manager_from_env; import asyncio; asyncio.run(create_database_manager_from_env().initialize())"

# Run tests
pytest tests/ -v

# Start development server
uvicorn mapper.main:app --reload --port 8002
```

### Shared Components Integration

The mapper service integrates with shared components from `../shared/`:

- **Interfaces**: Common request/response models and base classes
- **Utilities**: Logging, correlation IDs, metrics, circuit breakers
- **Exceptions**: Standardized error handling across services
- **Models**: Shared data models and validation

The integration is configured automatically when you run `python setup_shared.py`.

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run with GPU support
docker-compose -f docker-compose.gpu.yml up
```

## Configuration

Configuration is managed through YAML files and environment variables:

- `config/settings.yaml`: Base configuration
- Environment variables with `MAPPER_` prefix override YAML settings

Key configuration sections:

- `database`: PostgreSQL connection settings
- `storage`: Model storage configuration (MinIO/S3)
- `ml`: ML model serving configuration
- `training`: Model training parameters
- `cost_monitoring`: Cost tracking settings

## API Endpoints

- `POST /api/v1/map`: Core mapping functionality
- `POST /api/v1/batch-map`: Batch mapping operations
- `POST /api/v1/validate`: Input/output validation
- `POST /api/v1/train`: Model training endpoints
- `GET /api/v1/models`: List available models
- `GET /api/v1/health`: Health check endpoint
- `GET /api/v1/metrics`: Prometheus metrics

## Architecture

The service follows a clean architecture pattern:

```
API Layer (FastAPI endpoints)
├── Core Layer (mapping logic)
├── ML Layer (model serving)
├── Serving Layer (inference backends)
├── Validation Layer (I/O validation)
├── Storage Layer (model storage)
└── Infrastructure Layer (database, monitoring)
```

## Core Components

### Mapping Engine

- **Canonical Mapping**: Transform detector outputs to canonical taxonomy
- **Framework Mapping**: Map canonical results to compliance frameworks
- **Confidence Scoring**: Provide confidence metrics for all mappings
- **Fallback Mechanisms**: Rule-based fallbacks for low-confidence predictions

### Model Serving

- **vLLM Backend**: High-performance GPU inference
- **TGI Backend**: Text Generation Inference support
- **CPU Fallback**: CPU-based inference for non-GPU environments
- **Model Caching**: Intelligent model caching and loading

### Training Infrastructure

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning
- **Data Pipeline**: Training data preprocessing and validation
- **Checkpoint Management**: Model versioning and checkpoint storage
- **Evaluation**: Model performance evaluation and metrics

## Model Management

### Supported Models

- **Llama-3-8B**: Primary mapping model
- **LoRA Adapters**: Domain-specific adaptations
- **Custom Models**: Tenant-specific model support
- **Fallback Models**: Rule-based mapping alternatives

### Model Lifecycle

- **Training**: Automated training pipelines
- **Validation**: Model quality validation
- **Deployment**: Canary and blue-green deployments
- **Monitoring**: Performance and drift monitoring
- **Retirement**: Model lifecycle management

## Validation System

### Input Validation

- **Schema Validation**: JSON schema compliance
- **Content Validation**: Input content validation
- **Security Validation**: Malicious input detection
- **Rate Limiting**: Request rate validation

### Output Validation

- **Format Validation**: Output format compliance
- **Confidence Validation**: Confidence threshold checks
- **Schema Compliance**: Output schema validation
- **Quality Checks**: Output quality validation

## Cost Monitoring

### Real-time Tracking

- **Token Usage**: Input/output token counting
- **Inference Costs**: Per-request cost calculation
- **Storage Costs**: Model storage cost tracking
- **Total Cost**: Comprehensive cost aggregation

### Optimization

- **Batch Processing**: Efficient batch operations
- **Model Caching**: Reduce redundant model loading
- **Resource Management**: Optimal resource utilization
- **Cost Alerts**: Budget threshold alerts

## Multi-Tenancy

### Tenant Isolation

- **Data Isolation**: Tenant-specific data separation
- **Model Isolation**: Tenant-specific model serving
- **Resource Quotas**: Per-tenant resource limits
- **Cost Allocation**: Tenant-specific cost tracking

### Customization

- **Custom Models**: Tenant-specific model training
- **Custom Taxonomies**: Tenant-specific taxonomy support
- **Custom Validation**: Tenant-specific validation rules
- **Custom Policies**: Tenant-specific processing policies

## Development

### Code Quality

- Type hints for all functions
- Comprehensive error handling
- Structured logging with correlation IDs
- Performance optimization

### Testing

- Unit tests for core logic
- Integration tests for API endpoints
- Performance tests for model serving
- Load tests for scalability

### Documentation

- API documentation with OpenAPI
- Code documentation with docstrings
- Architecture decision records
- Deployment guides

## Deployment

### Environment Variables

Key environment variables:

- `MAPPER_ENV`: Environment (development, staging, production)
- `MAPPER_DATABASE_HOST`: Database host (default: localhost)
- `MAPPER_DATABASE_PORT`: Database port (default: 5432)
- `MAPPER_DATABASE_NAME`: Database name (default: mapper_service)
- `MAPPER_DATABASE_USER`: Database username (default: mapper_user)
- `MAPPER_DATABASE_PASSWORD`: Database password
- `MAPPER_DATABASE_SSL_MODE`: SSL mode (default: require)
- `MAPPER_DATABASE_POOL_MIN`: Minimum pool size (default: 5)
- `MAPPER_DATABASE_POOL_MAX`: Maximum pool size (default: 20)
- `MAPPER_MODEL_CACHE_DIR`: Model cache directory
- `MAPPER_STORAGE_BACKEND`: Storage backend (minio, s3, local)
- `MAPPER_GPU_ENABLED`: Enable GPU acceleration

### Scaling Strategies

- **Horizontal Scaling**: Multiple service instances
- **Model Sharding**: Distribute models across instances
- **Load Balancing**: Intelligent request routing
- **Auto-scaling**: Dynamic scaling based on load

### Deployment Patterns

- **Blue-Green**: Zero-downtime deployments
- **Canary**: Gradual rollout with monitoring
- **A/B Testing**: Model performance comparison
- **Feature Flags**: Runtime feature toggling

## Monitoring & Observability

### Metrics

- **Request Metrics**: Request count, latency, errors
- **Model Metrics**: Inference time, GPU utilization
- **Cost Metrics**: Token usage, inference costs
- **Quality Metrics**: Confidence scores, accuracy

### Tracing

- **Distributed Tracing**: Request flow across services
- **Correlation IDs**: Request correlation tracking
- **Performance Tracing**: Detailed performance analysis
- **Error Tracing**: Error propagation tracking

### Alerting

- **Performance Alerts**: Latency and error rate alerts
- **Cost Alerts**: Budget threshold notifications
- **Quality Alerts**: Model performance degradation
- **Resource Alerts**: Resource utilization warnings

## Security

### Authentication & Authorization

- **API Key Authentication**: Service-to-service auth
- **JWT Tokens**: User authentication
- **RBAC**: Role-based access control
- **Multi-Factor Auth**: Enhanced security

### Data Protection

- **Encryption**: Data encryption at rest and in transit
- **Privacy Controls**: PII detection and handling
- **Audit Logging**: Comprehensive audit trails
- **Secure Storage**: Encrypted model storage

## Performance Optimization

### Inference Optimization

- **Batch Processing**: Efficient batch inference
- **Model Quantization**: Reduced model size
- **Caching**: Intelligent response caching
- **GPU Optimization**: Optimal GPU utilization

### Resource Management

- **Memory Management**: Efficient memory usage
- **Connection Pooling**: Database connection optimization
- **Load Balancing**: Optimal request distribution
- **Resource Monitoring**: Real-time resource tracking

## Contributing

1. Follow the coding standards and best practices
2. Write comprehensive tests for new features
3. Update documentation for API changes
4. Ensure security and privacy compliance
5. Submit pull requests with detailed descriptions

## License

MIT License - see LICENSE file for details.
