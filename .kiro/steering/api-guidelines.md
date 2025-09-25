---
inclusion: fileMatch
fileMatchPattern: '**/api/**'
---

# Microservice API Design Guidelines

## Service API Architecture

The system uses **3 independent microservices** with well-defined HTTP APIs:

### Service Endpoints
- **Detector Orchestration**: `/orchestration/api/v1/`
- **Analysis Service**: `/analysis/api/v1/`  
- **Mapper Service**: `/mapper/api/v1/`

### Inter-Service Communication
Services communicate via HTTP APIs with:
- OpenAPI 3.0 specifications for all endpoints
- Consistent authentication (API keys)
- Structured error responses
- Request/response validation

# API Design Guidelines

## RESTful API Principles

### Resource Naming
- Use nouns for resource names, not verbs
- Use plural nouns for collections: `/detectors`, `/mappings`
- Use kebab-case for multi-word resources: `/compliance-frameworks`
- Nest resources logically: `/tenants/{id}/detectors`

### HTTP Methods
- `GET` - Retrieve resources (idempotent)
- `POST` - Create new resources
- `PUT` - Update entire resource (idempotent)
- `PATCH` - Partial resource updates
- `DELETE` - Remove resources (idempotent)

### Status Codes
- `200 OK` - Successful GET, PUT, PATCH
- `201 Created` - Successful POST
- `204 No Content` - Successful DELETE
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource doesn't exist
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server errors

## Request/Response Format

### Request Structure
```json
{
  "data": {
    // Primary request data
  },
  "metadata": {
    "requestId": "uuid",
    "timestamp": "ISO8601",
    "version": "v1"
  }
}
```

### Response Structure
```json
{
  "data": {
    // Response payload
  },
  "metadata": {
    "requestId": "uuid", 
    "timestamp": "ISO8601",
    "processingTime": "100ms"
  },
  "pagination": {
    // For paginated responses
    "page": 1,
    "limit": 50,
    "total": 1000,
    "hasNext": true
  }
}
```

### Error Response Structure
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "detector_type",
        "message": "Invalid detector type"
      }
    ]
  },
  "metadata": {
    "requestId": "uuid",
    "timestamp": "ISO8601"
  }
}
```

## Authentication & Authorization

### API Key Authentication
- Include API key in `Authorization` header: `Bearer <api_key>`
- Implement tenant isolation based on API key
- Support API key rotation without service interruption

### Rate Limiting
- Implement per-tenant and per-endpoint rate limits
- Return rate limit headers:
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Window reset time

## Versioning Strategy

### URL Versioning
- Include version in URL path: `/api/v1/detectors`
- Support multiple versions simultaneously
- Provide clear migration paths between versions

### Deprecation Process
1. Announce deprecation with 6-month notice
2. Add deprecation headers to responses
3. Update documentation with migration guide
4. Remove deprecated endpoints after sunset period

## Pagination & Filtering

### Pagination Parameters
- `page`: Page number (1-based)
- `limit`: Items per page (default: 50, max: 1000)
- `cursor`: Cursor-based pagination for large datasets

### Filtering Parameters
- Use query parameters for filtering: `?detector_type=pii&confidence_min=0.8`
- Support common operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`
- Use consistent parameter naming across endpoints

### Sorting Parameters
- `sort`: Field to sort by
- `order`: Sort direction (`asc` or `desc`)
- Support multiple sort fields: `sort=created_at,name&order=desc,asc`

## Performance Guidelines

### Response Times
- Target sub-100ms for simple queries
- Target sub-500ms for complex operations
- Implement timeout handling for long-running operations

### Caching Strategy
- Use ETags for conditional requests
- Implement Redis caching for frequently accessed data
- Set appropriate cache headers for static content

### Async Operations
- Use async endpoints for long-running operations
- Return operation status and polling endpoints
- Implement webhooks for completion notifications

## Security Considerations

### Input Validation
- Validate all input parameters using Pydantic models
- Sanitize inputs to prevent injection attacks
- Implement request size limits

### Data Privacy
- Never log sensitive data in request/response logs
- Implement field-level encryption for sensitive data
- Support data deletion requests for compliance

### CORS Configuration
- Configure CORS headers appropriately
- Restrict origins in production environments
- Support preflight requests for complex operations