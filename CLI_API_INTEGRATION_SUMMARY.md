# CLI and API Integration Summary

## Overview

Yes, customers can absolutely use both the CLI and API together! The refactored CLI architecture now includes comprehensive API client commands that allow seamless integration between local CLI usage and API-based processing.

## What's Available

### 🔧 **CLI Commands for API Interaction**

The new `api_client.py` module provides CLI commands that interact with the Llama Mapper API:

```bash
# API Health and Status
mapper api health                    # Check API health
mapper api metrics                   # Get API metrics
mapper api alerts                    # Get quality alerts

# Single and Batch Processing
mapper api map --input data.json     # Map single detector output
mapper api batch-map --input batch.json  # Batch process multiple outputs

# Authentication
mapper auth rotate-key --tenant customer-123  # Generate API keys
```

### 🌐 **Direct API Endpoints**

The API provides RESTful endpoints for programmatic access:

```bash
# Core Endpoints
POST /map                    # Map single detector output
POST /map/batch             # Batch map multiple outputs
GET  /health                # Health check
GET  /metrics/summary       # Get metrics
GET  /metrics/alerts        # Get quality alerts
GET  /openapi.yaml          # API documentation
```

## Customer Use Cases

### 1. **Development and Testing**
```bash
# Start API server
mapper serve --host 0.0.0.0 --port 8000

# Test with CLI
mapper api map --input test_data.json --output result.json

# Verify with direct API call
curl -X POST http://localhost:8000/map \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### 2. **Production Integration**
```python
# Python application using API
import requests

response = requests.post(
    "https://api.your-domain.com/map",
    json=detector_output,
    headers={"X-API-Key": "your-api-key"}
)
result = response.json()
```

### 3. **CI/CD Pipeline**
```yaml
# GitHub Actions example
- name: Map Detector Outputs
  run: |
    mapper api map --input test_outputs.json --output mapped_results.json
    mapper api health  # Verify API is healthy
```

### 4. **Monitoring and Operations**
```bash
# Health monitoring
mapper api health

# Metrics collection
mapper api metrics --format json > metrics.json

# Quality alerts
mapper api alerts --output alerts.json
```

## Key Benefits

### 🚀 **Flexibility**
- **CLI**: Perfect for development, testing, and one-off tasks
- **API**: Ideal for application integration and automation
- **Both**: Can be used together for comprehensive workflows

### 🔄 **Seamless Integration**
- CLI commands can call the API
- API responses work with CLI tools
- Shared configuration and authentication
- Consistent error handling and logging

### 🛡️ **Security**
- API key authentication for both CLI and API
- Tenant isolation and access control
- Rate limiting and request validation
- Audit trails and monitoring

### 📊 **Observability**
- Health checks for both CLI and API
- Metrics collection and monitoring
- Quality alerts and notifications
- Comprehensive logging and tracing

## Sample Workflows

### Workflow 1: Development to Production
```bash
# 1. Local development with CLI
mapper api map --input local_test.json

# 2. Test with API directly
curl -X POST http://localhost:8000/map -d @local_test.json

# 3. Deploy to production
# Application uses API endpoints directly
```

### Workflow 2: Batch Processing
```bash
# 1. Prepare batch data
mapper api batch-map --input large_dataset.json --output results.json

# 2. Monitor progress
mapper api metrics --format table

# 3. Check for issues
mapper api alerts
```

### Workflow 3: Monitoring and Maintenance
```bash
# 1. Health check
mapper api health

# 2. Get metrics
mapper api metrics --format json > daily_metrics.json

# 3. Check alerts
mapper api alerts --output daily_alerts.json

# 4. Generate reports
# Process metrics and alerts data
```

## Configuration

### API Server Configuration
```yaml
# config.yaml
serving:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_payload_kb: 64

auth:
  enabled: true
  api_key_header: "X-API-Key"
  tenant_header: "X-Tenant-ID"
```

### CLI Configuration
```bash
# Use custom config
mapper --config custom-config.yaml api health

# Override settings
mapper --log-level DEBUG api map --input data.json
```

## Authentication

### API Key Management
```bash
# Generate API key
mapper auth rotate-key --tenant customer-123 --scope map:write

# Use API key with CLI
mapper api map --input data.json --api-key your-api-key

# Use API key with API
curl -X POST http://localhost:8000/map \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d @data.json
```

## Error Handling

### CLI Error Handling
```bash
# Commands provide clear error messages
mapper api map --input invalid.json
# Output: ✗ Failed to load input file: Invalid JSON in invalid.json
```

### API Error Handling
```json
// API returns structured error responses
{
  "error_code": "INVALID_REQUEST",
  "message": "Request body does not match MapperPayload",
  "request_id": "req-123",
  "retryable": false
}
```

## Examples and Documentation

### Sample Data Files
- `examples/sample_data/detector_output.json` - Single detector output
- `examples/sample_data/batch_input.json` - Batch processing input

### Demo Scripts
- `examples/scripts/demo_cli_api_integration.py` - Complete integration demo
- `examples/customer_usage_guide.md` - Comprehensive usage guide

### API Documentation
- OpenAPI specification available at `/openapi.yaml`
- Interactive docs at `/docs` (when server is running)

## Getting Started

### 1. Start the API Server
```bash
mapper serve --host 0.0.0.0 --port 8000
```

### 2. Test with CLI
```bash
mapper api health
mapper api map --input examples/sample_data/detector_output.json
```

### 3. Test with API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/map \
  -H "Content-Type: application/json" \
  -d @examples/sample_data/detector_output.json
```

### 4. Run the Demo
```bash
python examples/scripts/demo_cli_api_integration.py
```

## Conclusion

The refactored CLI architecture provides customers with the best of both worlds:

- **CLI**: For development, testing, monitoring, and automation
- **API**: For application integration and production use
- **Integration**: Seamless workflow between both interfaces

Customers can choose the right tool for each task, or use both together for comprehensive detector output mapping and analysis workflows.

The new architecture makes it easy to:
- Develop and test locally with the CLI
- Integrate into applications with the API
- Monitor and maintain with both interfaces
- Scale from development to production seamlessly
