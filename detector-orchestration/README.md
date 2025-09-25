# Detector Orchestration Service

The Detector Orchestration Service is responsible for coordinating multiple detector executions, policy enforcement, and service discovery within the llama-mapper microservice architecture.

## Features

- **Detector Coordination**: Manages multiple detector executions with circuit breakers and health monitoring
- **Policy Enforcement**: OPA integration for policy validation and conflict resolution
- **Service Discovery**: Dynamic detector discovery and configuration reloading
- **Multi-Tenancy**: Tenant isolation, routing, and tenant-specific configurations
- **Security**: WAF integration, RBAC, API key management, and rate limiting
- **Monitoring**: Prometheus metrics, distributed tracing, and structured logging
- **Resilience**: Circuit breakers, retry logic, and graceful degradation

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
uvicorn orchestration.api.main:app --reload --port 8000
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in production mode
docker-compose -f docker-compose.prod.yml up
```

## Configuration

Configuration is managed through YAML files and environment variables:

- `config/settings.yaml`: Base configuration
- Environment variables with `ORCHESTRATION_` prefix override YAML settings

Key configuration sections:
- `database`: PostgreSQL connection settings
- `redis`: Redis cache configuration
- `security`: Authentication and authorization settings
- `orchestration`: Detector coordination settings
- `policy`: OPA policy engine configuration

## API Endpoints

- `POST /api/v1/orchestrate`: Orchestrate detector execution
- `GET /api/v1/detectors`: List available detectors
- `POST /api/v1/policies/validate`: Validate policy compliance
- `GET /api/v1/health`: Health check endpoint
- `GET /api/v1/metrics`: Prometheus metrics

## Architecture

The service follows a layered architecture:

```
API Layer (FastAPI endpoints)
├── Core Layer (orchestration logic)
├── Policy Layer (OPA integration)
├── Discovery Layer (service discovery)
├── Security Layer (auth, WAF, RBAC)
└── Infrastructure Layer (database, cache, monitoring)
```

## Development

### Code Style

- Use Black for code formatting
- Use isort for import sorting
- Use mypy for type checking
- Follow PEP 8 guidelines

### Testing

- Write unit tests for all business logic
- Include integration tests for API endpoints
- Use pytest fixtures for test data
- Maintain 80%+ test coverage

### Monitoring

The service includes comprehensive monitoring:

- **Metrics**: Prometheus metrics for requests, errors, and performance
- **Tracing**: OpenTelemetry distributed tracing
- **Logging**: Structured JSON logging with correlation IDs
- **Health Checks**: Comprehensive health endpoints

## Deployment

### Environment Variables

Key environment variables:

- `ORCHESTRATION_ENV`: Environment (development, staging, production)
- `ORCHESTRATION_DATABASE_URL`: PostgreSQL connection string
- `ORCHESTRATION_REDIS_URL`: Redis connection string
- `ORCHESTRATION_LOG_LEVEL`: Logging level
- `ORCHESTRATION_JWT_SECRET_KEY`: JWT signing key

### Scaling

The service supports horizontal scaling:

- Stateless design enables multiple instances
- Redis-backed caching for shared state
- Database connection pooling
- Load balancer compatible

## Security

Security features include:

- **Authentication**: API key and JWT token support
- **Authorization**: Role-based access control (RBAC)
- **WAF Protection**: SQL injection, XSS, and other attack prevention
- **Rate Limiting**: Configurable rate limits per tenant/API key
- **Audit Logging**: Comprehensive audit trails

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.