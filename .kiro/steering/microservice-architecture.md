---
inclusion: always
---

# Microservice Architecture Guidelines

## Overview

The llama-mapper system has been refactored into exactly **3 microservices** following Single Responsibility Principle and distributed system best practices. This document provides architectural guidance for development within this new structure.

## Service Architecture

### 1. Detector Orchestration Service
**Domain**: Detector coordination, policy enforcement, service discovery
**Location**: `detector-orchestration/`
**Responsibilities**:
- Detector health monitoring and circuit breakers
- Policy enforcement with OPA integration
- Multi-tenant routing and isolation
- Rate limiting and authentication
- Async job processing

### 2. Analysis Service
**Domain**: Advanced analysis, risk scoring, compliance intelligence
**Location**: `analysis-service/`
**Responsibilities**:
- Pattern recognition and statistical analysis
- Risk scoring with ML enhancement
- Compliance framework mapping
- RAG system and regulatory knowledge
- Quality monitoring and evaluation
- Privacy-first data processing

### 3. Mapper Service
**Domain**: Core mapping, model serving, response generation
**Location**: `mapper-service/`
**Responsibilities**:
- Core mapping logic and taxonomy management
- ML model serving (Llama-3-8B, LoRA fine-tuning)
- Response validation and formatting
- Cost monitoring and optimization
- Model versioning and deployment

## Development Principles

### Service Boundaries
- **No Shared Code**: Each service is completely independent
- **HTTP APIs Only**: Services communicate via well-defined HTTP contracts
- **Database Isolation**: Each service has its own database schema
- **Independent Deployment**: Services can be deployed independently

### Model Serving Strategy
- **Internal Implementation**: vLLM/TGI/CPU backends are internal to services, not separate services
- **Service-Specific Models**: Analysis Service uses Phi-3, Mapper Service uses Llama-3-8B
- **Fallback Mechanisms**: Rule-based alternatives for all ML components

### Multi-Tenancy
- **Row-Level Security**: Database-level tenant isolation
- **Tenant Routing**: Request routing based on tenant context
- **Resource Quotas**: Per-tenant resource limits and monitoring
- **Custom Configuration**: Tenant-specific settings and rules

## Implementation Guidelines

### Adding New Features
1. **Identify Service Domain**: Determine which service owns the feature
2. **Design API Contract**: Define HTTP API with OpenAPI spec
3. **Implement with Fallbacks**: Always provide non-ML fallback mechanisms
4. **Add Monitoring**: Include metrics, logging, and health checks
5. **Test Thoroughly**: Unit, integration, and contract tests

### Service Communication
```python
# Example: Analysis Service calling Mapper Service
async def call_mapper_service(analysis_result: AnalysisResponse) -> MappingResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAPPER_SERVICE_URL}/api/v1/map",
            json=analysis_result.dict(),
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        return MappingResponse(**response.json())
```

### Error Handling
- **Circuit Breakers**: Protect against cascading failures
- **Retry Logic**: Exponential backoff with jitter
- **Graceful Degradation**: Fallback to rule-based alternatives
- **Proper Logging**: Structured logging with correlation IDs

### Configuration Management
- **Service-Specific**: Each service has independent configuration
- **Environment Variables**: Use `SERVICE_NAME_` prefixes (e.g., `ANALYSIS_`, `MAPPER_`)
- **Hot Reload**: Support configuration updates without restart
- **Validation**: Fail fast with clear error messages

## Plugin System

### Plugin Architecture
Each service supports plugins for extensibility:
- **Detector Orchestration**: Detector plugins, policy plugins
- **Analysis Service**: Analysis engine plugins, ML model plugins
- **Mapper Service**: Mapping plugins, validation plugins

### Plugin Development
```python
# Example: Analysis Engine Plugin
class CustomAnalysisPlugin(IAnalysisEngine):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze(self, data: AnalysisRequest) -> AnalysisResult:
        # Custom analysis logic
        pass
    
    def get_capabilities(self) -> List[str]:
        return ["custom_pattern_detection"]
```

## Testing Strategy

### Service Testing
- **Unit Tests**: 80%+ coverage for business logic
- **Integration Tests**: Service-to-service communication
- **Contract Tests**: API contract validation
- **End-to-End Tests**: Complete workflow validation

### Test Organization
```
tests/
├── unit/           # Service-specific unit tests
├── integration/    # Cross-service integration tests
├── contracts/      # API contract tests
└── e2e/           # End-to-end workflow tests
```

## Deployment Guidelines

### Deployment Strategy
- **Blue-Green**: Zero-downtime deployments
- **Canary**: Gradual rollout with monitoring
- **Feature Flags**: Runtime feature toggling
- **A/B Testing**: Model and algorithm testing

### Monitoring Requirements
- **Service Metrics**: Prometheus metrics for each service
- **Distributed Tracing**: OpenTelemetry with correlation IDs
- **Health Checks**: Comprehensive health endpoints
- **Business Metrics**: Domain-specific KPIs

## Security Guidelines

### Authentication & Authorization
- **API Keys**: Service-to-service authentication
- **RBAC**: Role-based access control
- **Multi-Factor**: Enhanced authentication for admin operations
- **Audit Trails**: Comprehensive audit logging

### Privacy Controls
- **Metadata Only**: Never log raw content
- **Content Scrubbing**: Sanitize all logged data
- **Data Minimization**: Store only necessary metadata
- **Retention Policies**: Automatic data cleanup

## Common Patterns

### Circuit Breaker Pattern
```python
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
async def call_external_service():
    # Service call with circuit breaker protection
    pass
```

### Retry Pattern
```python
@retry(max_attempts=3, backoff=exponential_backoff)
async def resilient_service_call():
    # Service call with retry logic
    pass
```

### Correlation ID Pattern
```python
def get_correlation_id() -> str:
    return correlation_id.get() or str(uuid.uuid4())

# Include in all logs and service calls
logger.info("Processing request", correlation_id=get_correlation_id())
```

## Migration Guidelines

### Extracting Functionality
1. **Identify Dependencies**: Map all dependencies for the functionality
2. **Create Service Boundary**: Define clean API contracts
3. **Implement Service**: Build service with comprehensive tests
4. **Migrate Data**: Move data to service-specific schema
5. **Update Clients**: Switch to service API calls
6. **Validate**: Ensure functionality preservation

### Database Migration
- **Schema Separation**: Each service gets its own schema
- **Data Migration**: Preserve all existing data
- **Referential Integrity**: Handle cross-service references
- **Rollback Plan**: Ability to revert changes

## Best Practices

### Code Organization
- **Domain-Driven Design**: Organize by business domain
- **Clean Architecture**: Separate concerns with clear layers
- **Dependency Injection**: Use IoC containers for testability
- **Interface Segregation**: Small, focused interfaces

### Performance Optimization
- **Connection Pooling**: Efficient database connections
- **Caching Strategy**: Redis for frequently accessed data
- **Batch Processing**: Optimize for bulk operations
- **Resource Management**: Proper cleanup and resource limits

### Documentation
- **API Documentation**: Complete OpenAPI specifications
- **Architecture Decisions**: Document all major decisions
- **Runbooks**: Operational procedures and troubleshooting
- **Developer Guides**: Onboarding and development workflows

This architecture provides a solid foundation for scalable, maintainable, and reliable microservices while preserving all existing functionality.