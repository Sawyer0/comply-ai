# Enhanced Request/Response Models with Validation

This directory contains enhanced request/response models with comprehensive validation for the llama-mapper microservice architecture.

## Features

### 🔧 Enhanced Request/Response Models
- **Pydantic-based models** with comprehensive validation
- **Type safety** with proper enum definitions
- **Field validation** with custom validators
- **Nested model support** for complex data structures
- **Automatic serialization/deserialization**

### 🛡️ Comprehensive Validation
- **Request validation** with detailed error messages
- **Response validation** to ensure data integrity
- **Schema validation** using JSON Schema and Pydantic
- **Tenant validation** for multi-tenancy support
- **Confidence threshold validation** for ML models

### 🔄 Enhanced HTTP Clients
- **Circuit breaker pattern** for resilience
- **Retry logic** with exponential backoff
- **Correlation ID tracking** for distributed tracing
- **Structured error handling** with custom exceptions
- **Connection pooling** and timeout management

### 🎯 Validation Decorators
- **@validate_request_response** for automatic validation
- **@validate_tenant_access** for authorization
- **@validate_confidence_threshold** for ML quality control
- **Async/sync support** for all decorators

## Directory Structure

```
shared/
├── interfaces/           # Pydantic models for all services
│   ├── base.py          # Base request/response models
│   ├── orchestration.py # Orchestration service models
│   ├── analysis.py      # Analysis service models
│   └── mapper.py        # Mapper service models
├── clients/             # Enhanced HTTP clients
│   ├── orchestration_client.py
│   ├── analysis_client.py
│   ├── mapper_client.py
│   └── client_factory.py
├── validation/          # Validation utilities
│   ├── decorators.py    # Validation decorators
│   ├── schemas.py       # Schema validation
│   └── middleware.py    # FastAPI middleware
├── utils/               # Utility functions
│   ├── correlation.py   # Correlation ID management
│   ├── validation.py    # Validation helpers
│   ├── retry.py         # Retry logic
│   └── circuit_breaker.py
├── exceptions/          # Custom exceptions
│   └── base.py          # Base exception classes
└── examples/            # Usage examples
    └── enhanced_client_usage.py
```

## Quick Start

### 1. Basic Client Usage

```python
from shared.clients import create_orchestration_client
from shared.interfaces.orchestration import OrchestrationRequest, ProcessingMode

# Create client
client = create_orchestration_client()

# Create request with validation
request = OrchestrationRequest(
    content="Sensitive data to analyze",
    detector_types=["presidio", "deberta"],
    processing_mode=ProcessingMode.STANDARD
)

# Make request
response = await client.orchestrate_detectors(
    request=request,
    tenant_id="my-tenant"
)
```

### 2. Enhanced Client Factory

```python
from shared.clients.client_factory import ClientFactory, ClientConfig

# Configure client
config = ClientConfig(
    timeout=60.0,
    max_retries=3,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5
)

# Create factory
factory = ClientFactory(default_config=config)

# Create clients
orchestration_client = factory.create_orchestration_client()
analysis_client = factory.create_analysis_client()
mapper_client = factory.create_mapper_client()
```

### 3. Validation Decorators

```python
from shared.validation.decorators import validate_request_response
from shared.interfaces.orchestration import OrchestrationRequest, OrchestrationResponse

@validate_request_response(
    request_model=OrchestrationRequest,
    response_model=OrchestrationResponse,
    validate_tenant=True
)
async def orchestrate_detectors(request: OrchestrationRequest, tenant_id: str):
    # Function automatically validates request and response
    # Tenant validation is enforced
    pass
```

### 4. Schema Validation

```python
from shared.validation.schemas import default_validator, ValidationContext

# Create validation context
context = ValidationContext(
    tenant_id="my-tenant",
    strict_mode=True,
    allow_extra_fields=False
)

# Validate data
validated_data = default_validator.validate_request(
    data=request_data,
    model_name="OrchestrationRequest",
    context=context
)
```

## Model Features

### Base Models

All models inherit from enhanced base classes:

```python
class BaseRequest(BaseModel):
    correlation_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True

class BaseResponse(BaseModel):
    request_id: str
    success: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    correlation_id: Optional[str] = None
```

### Enhanced Validation

Models include comprehensive validation:

```python
class OrchestrationRequest(BaseRequest):
    content: str = Field(max_length=10000, min_length=1)
    detector_types: List[str] = Field(min_items=1, max_items=20)
    processing_mode: ProcessingMode = ProcessingMode.STANDARD
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()
    
    @validator('detector_types')
    def validate_detector_types(cls, v):
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for detector_type in v:
            if detector_type not in seen:
                seen.add(detector_type)
                unique_types.append(detector_type)
        return unique_types
```

### Enum Definitions

Comprehensive enums for type safety:

```python
class ProcessingMode(str, Enum):
    STANDARD = "standard"
    FAST = "fast"
    THOROUGH = "thorough"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

## Error Handling

### Custom Exceptions

```python
from shared.exceptions.base import ValidationError, ServiceUnavailableError

try:
    response = await client.orchestrate_detectors(request, tenant_id)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Details: {e.details}")
except ServiceUnavailableError as e:
    print(f"Service unavailable: {e.message}")
```

### Exception Hierarchy

```
BaseServiceException
├── ValidationError
├── AuthenticationError
├── AuthorizationError
├── RateLimitError
├── ServiceUnavailableError
├── TimeoutError
└── ConfigurationError
```

## Resilience Features

### Circuit Breaker

```python
from shared.utils.circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
async def call_external_service():
    # Service call protected by circuit breaker
    pass
```

### Retry Logic

```python
from shared.utils.retry import retry_with_backoff

@retry_with_backoff(max_attempts=3, base_delay=1.0)
async def resilient_service_call():
    # Service call with automatic retry
    pass
```

### Correlation ID Tracking

```python
from shared.utils.correlation import get_correlation_id, set_correlation_id

# Set correlation ID for request tracing
correlation_id = set_correlation_id("my-correlation-id")

# Get current correlation ID
current_id = get_correlation_id()
```

## FastAPI Integration

### Validation Middleware

```python
from fastapi import FastAPI
from shared.validation.middleware import ValidationMiddleware, TenantValidationMiddleware

app = FastAPI()

# Add validation middleware
app.add_middleware(
    ValidationMiddleware,
    validate_requests=True,
    validate_responses=True,
    strict_mode=True
)

# Add tenant validation middleware
app.add_middleware(
    TenantValidationMiddleware,
    require_tenant_id=True,
    allowed_tenants=["tenant1", "tenant2"]
)
```

### Route Validation

```python
from fastapi import FastAPI, Depends
from shared.interfaces.orchestration import OrchestrationRequest, OrchestrationResponse

app = FastAPI()

@app.post("/api/v1/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_detectors(
    request: OrchestrationRequest,  # Automatic validation
    tenant_id: str = Header(..., alias="X-Tenant-ID")
):
    # Request is automatically validated
    # Response will be validated against OrchestrationResponse
    pass
```

## Configuration

### Environment Variables

```bash
# Service URLs
ORCHESTRATION_SERVICE_URL=http://localhost:8000
ANALYSIS_SERVICE_URL=http://localhost:8001
MAPPER_SERVICE_URL=http://localhost:8002

# API Keys
ORCHESTRATION_API_KEY=your-api-key
ANALYSIS_API_KEY=your-api-key
MAPPER_API_KEY=your-api-key

# Client Configuration
CLIENT_TIMEOUT=60.0
CLIENT_MAX_RETRIES=3
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60.0
```

### Client Configuration

```python
from shared.clients.client_factory import ClientConfig

config = ClientConfig(
    timeout=60.0,
    max_retries=3,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0
)
```

## Testing

### Model Testing

```python
import pytest
from shared.interfaces.orchestration import OrchestrationRequest
from pydantic import ValidationError

def test_orchestration_request_validation():
    # Valid request
    request = OrchestrationRequest(
        content="Test content",
        detector_types=["presidio"]
    )
    assert request.content == "Test content"
    
    # Invalid request
    with pytest.raises(ValidationError):
        OrchestrationRequest(
            content="",  # Empty content should fail
            detector_types=[]  # Empty detector types should fail
        )
```

### Client Testing

```python
import pytest
from unittest.mock import AsyncMock
from shared.clients import create_orchestration_client

@pytest.mark.asyncio
async def test_orchestration_client():
    client = create_orchestration_client()
    
    # Mock the HTTP client
    client.client.post = AsyncMock()
    
    # Test client call
    response = await client.orchestrate_detectors(request, tenant_id)
    
    # Verify call was made
    client.client.post.assert_called_once()
```

## Best Practices

### 1. Always Use Type Hints

```python
from typing import List, Optional
from shared.interfaces.orchestration import OrchestrationRequest

async def process_request(
    request: OrchestrationRequest,
    tenant_id: str,
    options: Optional[dict] = None
) -> OrchestrationResponse:
    pass
```

### 2. Validate at Boundaries

```python
@validate_request_response(
    request_model=OrchestrationRequest,
    response_model=OrchestrationResponse
)
async def api_endpoint(request: OrchestrationRequest, tenant_id: str):
    # Validation happens automatically at the boundary
    pass
```

### 3. Use Correlation IDs

```python
from shared.utils.correlation import get_correlation_id

async def process_request():
    correlation_id = get_correlation_id()
    logger.info("Processing request", extra={"correlation_id": correlation_id})
```

### 4. Handle Errors Gracefully

```python
from shared.exceptions.base import BaseServiceException

try:
    result = await client.call_service()
except BaseServiceException as e:
    logger.error("Service call failed: %s", e.message)
    # Handle specific error types
    if isinstance(e, ValidationError):
        # Handle validation error
        pass
    elif isinstance(e, ServiceUnavailableError):
        # Handle service unavailable
        pass
```

### 5. Use Circuit Breakers for External Calls

```python
from shared.utils.circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=5)
async def call_external_api():
    # External API call protected by circuit breaker
    pass
```

## Migration Guide

### From Basic Models to Enhanced Models

1. **Update imports:**
   ```python
   # Old
   from some_module import BasicRequest
   
   # New
   from shared.interfaces.orchestration import OrchestrationRequest
   ```

2. **Add validation decorators:**
   ```python
   # Old
   async def my_function(data: dict):
       pass
   
   # New
   @validate_request_response(request_model=OrchestrationRequest)
   async def my_function(request: OrchestrationRequest):
       pass
   ```

3. **Update error handling:**
   ```python
   # Old
   except Exception as e:
       pass
   
   # New
   from shared.exceptions.base import BaseServiceException
   except BaseServiceException as e:
       logger.error("Service error: %s", e.message)
   ```

## Contributing

When adding new models or validation:

1. **Follow naming conventions:** Use descriptive names and proper enums
2. **Add comprehensive validation:** Include field validators and root validators
3. **Write tests:** Test both valid and invalid cases
4. **Update documentation:** Add examples and usage instructions
5. **Consider backwards compatibility:** Use optional fields for new features

## Performance Considerations

- **Model validation** adds overhead but ensures data integrity
- **Circuit breakers** prevent cascading failures
- **Connection pooling** improves performance for HTTP clients
- **Correlation IDs** enable efficient debugging without performance impact

## Security Features

- **Input validation** prevents injection attacks
- **Tenant isolation** ensures data security
- **API key management** for service authentication
- **Request sanitization** removes potentially harmful data

This enhanced validation system provides a robust foundation for microservice communication while maintaining high performance and reliability.