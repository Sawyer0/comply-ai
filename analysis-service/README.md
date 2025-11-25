# Analysis Service

A learning implementation of an analysis service for AI safety and compliance, built to explore microservices architecture and AI integration patterns.

## Learning Focus

This service serves as an educational example of:

- Building analytical microservices in Python
- Implementing pattern recognition and risk scoring
- Working with AI/ML models in production-like environments
- Designing for privacy and compliance

## Current Implementation Status

### Implemented Features

- Basic pattern recognition (temporal/frequency analysis)
- Core risk scoring framework
- Basic API endpoints for analysis
- Integration with ML models

### Learning Areas

- Microservices communication patterns
- Asynchronous processing with FastAPI
- Containerized ML model serving
- Testing distributed systems

## Key Components

- `api/` - FastAPI endpoints for analysis operations
- `engines/` - Core analysis engines (pattern recognition, risk scoring)
- `ml/` - Machine learning model integration
- `monitoring/` - Basic observability setup

## Quick Start

### Development Setup

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start development server
uvicorn analysis.api.main:app --reload --port 8001
```

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
- Environment variables with `ANALYSIS_` prefix override YAML settings

Key configuration sections:

- `database`: PostgreSQL connection settings
- `clickhouse`: ClickHouse analytics database
- `ml`: ML model configuration
- `rag`: RAG system settings
- `quality`: Quality monitoring configuration

## API Endpoints

- `POST /api/v1/analyze`: Perform comprehensive analysis
- `POST /api/v1/patterns`: Pattern recognition analysis
- `POST /api/v1/risk-score`: Risk scoring analysis
- `POST /api/v1/compliance`: Compliance framework mapping
- `POST /api/v1/rag/query`: RAG-enhanced queries
- `GET /api/v1/health`: Health check endpoint
- `GET /api/v1/metrics`: Prometheus metrics

## Architecture

The service follows a domain-driven architecture:

```
API Layer (FastAPI endpoints)
├── Engines Layer (analysis engines)
│   ├── Pattern Recognition
│   ├── Risk Scoring
│   ├── Compliance Intelligence
│   └── Statistical Analysis
├── ML Layer (model serving)
├── RAG Layer (knowledge retrieval)
├── Quality Layer (monitoring)
└── Infrastructure Layer (database, cache)
```

## Analysis Engines

### Pattern Recognition

- **Temporal Analysis**: Time-based pattern detection
- **Frequency Analysis**: Occurrence pattern analysis
- **Correlation Analysis**: Cross-pattern correlation
- **Anomaly Detection**: Statistical anomaly identification

### Risk Scoring

- **Technical Risk**: System and security risks
- **Business Risk**: Business impact assessment
- **Regulatory Risk**: Compliance risk evaluation
- **Temporal Risk**: Time-sensitive risk factors

### Compliance Intelligence

- **Framework Mapping**: Automatic compliance framework mapping
- **Gap Analysis**: Compliance gap identification
- **Evidence Generation**: Audit evidence creation
- **Recommendation Engine**: Remediation recommendations

## ML Components

### Model Serving

- **Phi-3 Model**: Analysis-specific language model
- **Embeddings**: Sentence transformers for similarity
- **vLLM Backend**: High-performance GPU inference
- **CPU Fallback**: Fallback for non-GPU environments

### Training Pipeline

- **Data Preparation**: Training data preprocessing
- **Model Fine-tuning**: LoRA adaptation for domain-specific tasks
- **Evaluation**: Model performance evaluation
- **Deployment**: Model versioning and deployment

## RAG System

### Knowledge Base

- **Regulatory Documents**: Compliance framework documents
- **Best Practices**: Industry best practices
- **Case Studies**: Real-world examples
- **Updates**: Automatic knowledge base updates

### Retrieval

- **Semantic Search**: Vector-based document retrieval
- **Ranking**: Relevance-based result ranking
- **Context Building**: Context-aware response generation
- **Caching**: Intelligent caching for performance

## Quality Monitoring

### Automated Evaluation

- **Model Performance**: Continuous model evaluation
- **Drift Detection**: Data and concept drift monitoring
- **Quality Metrics**: Comprehensive quality scoring
- **Alerting**: Automated quality alerts

### Weekly Evaluation

- **Scheduled Evaluation**: Regular quality assessments
- **Trend Analysis**: Performance trend monitoring
- **Report Generation**: Automated quality reports
- **Improvement Recommendations**: Quality improvement suggestions

## Privacy & Security

### Privacy Controls

- **Metadata-Only Logging**: No raw content persistence
- **Content Scrubbing**: Automatic PII removal
- **Data Minimization**: Minimal data retention
- **Encryption**: Field-level encryption for sensitive data

### Security Features

- **Authentication**: Multi-factor authentication
- **Authorization**: Fine-grained access control
- **Audit Trails**: Comprehensive audit logging
- **Secure Communication**: TLS encryption

## Development

### Code Organization

- Domain-driven design with clear boundaries
- Dependency injection for testability
- Interface segregation for modularity
- Clean architecture principles

### Testing Strategy

- Unit tests for business logic
- Integration tests for API endpoints
- Performance tests for ML models
- Contract tests for service interfaces

## Deployment

### Environment Variables

Key environment variables:

- `ANALYSIS_ENV`: Environment (development, staging, production)
- `ANALYSIS_DATABASE_URL`: PostgreSQL connection string
- `ANALYSIS_CLICKHOUSE_URL`: ClickHouse connection string
- `ANALYSIS_MODEL_CACHE_DIR`: Model cache directory
- `ANALYSIS_GPU_ENABLED`: Enable GPU acceleration

### Scaling

- Horizontal scaling with load balancing
- GPU resource management
- Model caching and sharing
- Database read replicas

## Monitoring

### Metrics

- Request/response metrics
- Model performance metrics
- Quality metrics
- Resource utilization metrics

### Observability

- Distributed tracing
- Structured logging
- Health checks
- Performance monitoring

## Contributing

1. Follow coding standards and best practices
2. Write comprehensive tests
3. Update documentation
4. Ensure security compliance
5. Submit pull requests with clear descriptions

## License

MIT License - see LICENSE file for details.
