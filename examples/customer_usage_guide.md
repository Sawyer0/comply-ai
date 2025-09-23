# Customer Usage Guide: CLI and API Integration

This guide shows how customers can use both the Llama Mapper CLI and API together for comprehensive detector output mapping and analysis.

## Overview

The Llama Mapper provides two complementary interfaces:

- **CLI (Command Line Interface)**: For local development, testing, and automation
- **API (Application Programming Interface)**: For integration into applications and services

Both interfaces share the same underlying functionality and can be used together seamlessly.

## Getting Started

### 1. Start the API Server

First, start the Llama Mapper API server:

```bash
# Using the CLI to start the server
python -m src.llama_mapper.cli.main serve --host 0.0.0.0 --port 8000

# Or using the serve command directly
mapper serve --host 0.0.0.0 --port 8000
```

### 2. Verify API Health

Check that the API is running and healthy:

```bash
# Using CLI to check API health
mapper api health

# Or using curl
curl http://localhost:8000/health
```

## Use Cases

### Use Case 1: Local Development and Testing

**Scenario**: Developer wants to test detector output mapping locally before deploying to production.

**CLI Approach**:
```bash
# Test with local files
mapper api map --input detector_output.json --output mapped_result.json

# Check results
cat mapped_result.json
```

**API Approach**:
```bash
# Using curl
curl -X POST http://localhost:8000/map \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d @detector_output.json
```

### Use Case 2: Batch Processing

**Scenario**: Process multiple detector outputs efficiently.

**CLI Approach**:
```bash
# Create batch input file
cat > batch_input.json << EOF
{
  "requests": [
    {"detector": "safety-detector-1", "output": {...}},
    {"detector": "safety-detector-2", "output": {...}},
    {"detector": "safety-detector-3", "output": {...}}
  ]
}
EOF

# Process batch
mapper api batch-map --input batch_input.json --output batch_results.json
```

**API Approach**:
```bash
# Using curl
curl -X POST http://localhost:8000/map/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d @batch_input.json
```

### Use Case 3: Integration with CI/CD Pipeline

**Scenario**: Integrate mapping into automated testing pipeline.

**CLI in CI/CD**:
```yaml
# GitHub Actions example
- name: Map Detector Outputs
  run: |
    mapper api map --input test_outputs.json --output mapped_results.json
    mapper api health  # Verify API is healthy
    
- name: Check Quality Alerts
  run: |
    mapper api alerts --output alerts.json
    # Process alerts in pipeline
```

**API in CI/CD**:
```yaml
# GitHub Actions example
- name: Map Detector Outputs
  run: |
    curl -X POST $API_URL/map \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $API_KEY" \
      -d @test_outputs.json > mapped_results.json
```

### Use Case 4: Monitoring and Observability

**Scenario**: Monitor API performance and quality metrics.

**CLI Monitoring**:
```bash
# Get current metrics
mapper api metrics --format table

# Get quality alerts
mapper api alerts

# Health check
mapper api health
```

**API Monitoring**:
```bash
# Metrics endpoint
curl http://localhost:8000/metrics/summary

# Alerts endpoint
curl http://localhost:8000/metrics/alerts

# Health endpoint
curl http://localhost:8000/health
```

## Sample Data Files

### Single Detector Output
```json
{
  "detector": "safety-detector-v1",
  "output": {
    "risk_level": "high",
    "confidence": 0.95,
    "categories": ["violence", "harassment"],
    "details": "Content contains explicit threats"
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0"
  },
  "tenant_id": "customer-123"
}
```

### Batch Input
```json
{
  "requests": [
    {
      "detector": "safety-detector-v1",
      "output": {"risk_level": "high", "confidence": 0.95},
      "tenant_id": "customer-123"
    },
    {
      "detector": "safety-detector-v2",
      "output": {"risk_level": "medium", "confidence": 0.78},
      "tenant_id": "customer-123"
    }
  ]
}
```

## Authentication

### API Key Setup

1. **Generate API Key**:
```bash
mapper auth rotate-key --tenant customer-123 --scope map:write
```

2. **Use API Key**:
```bash
# CLI
mapper api map --input data.json --api-key your-api-key

# API
curl -X POST http://localhost:8000/map \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d @data.json
```

## Error Handling

### CLI Error Handling
```bash
# Commands automatically handle errors and provide helpful messages
mapper api map --input invalid.json
# Output: ✗ Failed to load input file: Invalid JSON in invalid.json: Expecting value: line 1 column 1 (char 0)
```

### API Error Handling
```bash
# API returns structured error responses
curl -X POST http://localhost:8000/map \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'

# Response:
{
  "error_code": "INVALID_REQUEST",
  "message": "Request body does not match MapperPayload or DetectorRequest",
  "request_id": "req-123",
  "retryable": false
}
```

## Best Practices

### 1. Use CLI for Development
- Local testing and debugging
- One-off analysis tasks
- Configuration management
- Health checks and monitoring

### 2. Use API for Production
- Application integration
- High-volume processing
- Automated workflows
- Service-to-service communication

### 3. Combine Both Approaches
- Use CLI to test API endpoints
- Use API for bulk operations, CLI for individual checks
- Use CLI for monitoring, API for processing

### 4. Error Handling
- Always check API health before processing
- Implement retry logic for transient failures
- Monitor quality alerts regularly
- Use structured logging for debugging

## Advanced Usage

### Custom Configuration
```bash
# Use custom config file
mapper --config custom-config.yaml api map --input data.json

# Override specific settings
mapper --log-level DEBUG api health
```

### Plugin Development
```bash
# Load custom plugins
mapper --plugin-dir ./my-plugins api map --input data.json
```

### Output Formatting
```bash
# Different output formats
mapper api metrics --format json
mapper api metrics --format yaml
mapper api metrics --format table
```

## Troubleshooting

### Common Issues

1. **API Not Responding**:
```bash
# Check if server is running
mapper api health

# Check server logs
mapper serve --log-level DEBUG
```

2. **Authentication Errors**:
```bash
# Verify API key
mapper auth rotate-key --tenant your-tenant

# Check API key permissions
mapper config show --format json
```

3. **Input Validation Errors**:
```bash
# Validate input file
mapper config validate-config --data-dir ./data

# Check file format
file detector_output.json
```

### Getting Help
```bash
# CLI help
mapper --help
mapper api --help
mapper api map --help

# API documentation
curl http://localhost:8000/openapi.yaml
```

## Conclusion

The Llama Mapper CLI and API work together to provide a comprehensive solution for detector output mapping. Use the CLI for development and testing, and the API for production integration. Both interfaces share the same underlying functionality and can be used interchangeably based on your needs.
