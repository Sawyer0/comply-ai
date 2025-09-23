# Analysis Module CLI

The Analysis Module CLI provides command-line access to the automated analysis functionality for security metrics. This CLI allows you to analyze security findings, generate remediations, create policy recommendations, and manage the analysis module.

## Overview

The analysis CLI is integrated into the main Llama Mapper CLI and provides the following capabilities:

- **Single Analysis**: Analyze individual security metrics files
- **Batch Analysis**: Process multiple analysis requests in parallel
- **Health Monitoring**: Check the health status of the analysis module
- **Quality Evaluation**: Evaluate the quality of analysis outputs
- **Cache Management**: Manage analysis result caching
- **Configuration Validation**: Validate analysis module configuration

## Installation

The analysis CLI is automatically available when you install the Llama Mapper package:

```bash
pip install -e ".[dev]"
```

## Basic Usage

All analysis commands are available under the `analysis` subcommand:

```bash
mapper analysis --help
```

## Commands

### 1. Analyze Security Metrics

Analyze a single security metrics file to generate insights, remediations, and policy recommendations.

```bash
mapper analysis analyze --metrics-file <path> [options]
```

**Options:**
- `--metrics-file, -f`: Path to JSON file containing security metrics (required)
- `--analysis-type, -t`: Type of analysis (explanation|remediation|policy|comprehensive) [default: comprehensive]
- `--output, -o`: Output path for results (default: stdout)
- `--format`: Output format (json|yaml|text) [default: json]
- `--tenant-id`: Tenant ID for analysis context
- `--request-id`: Request ID for idempotency

**Example:**
```bash
# Analyze security metrics with comprehensive analysis
mapper analysis analyze --metrics-file examples/sample_metrics.json --format json

# Generate only remediation recommendations
mapper analysis analyze --metrics-file examples/sample_metrics.json --analysis-type remediation --output remediation.json

# Analyze with tenant context
mapper analysis analyze --metrics-file examples/sample_metrics.json --tenant-id "acme-corp" --format yaml
```

**Input Format:**
The metrics file should contain structured security findings in the following format:

```json
{
  "tenant_id": "example-tenant",
  "timestamp": "2024-01-15T10:30:00Z",
  "detector_results": [
    {
      "detector_name": "security-scan",
      "detector_version": "1.2.3",
      "scan_type": "vulnerability",
      "results": [
        {
          "finding_id": "CVE-2024-1234",
          "severity": "high",
          "category": "vulnerability",
          "description": "SQL injection vulnerability in user authentication",
          "affected_component": "auth-service",
          "confidence": 0.95
        }
      ]
    }
  ],
  "metadata": {
    "scan_duration_seconds": 45,
    "total_findings": 1,
    "high_severity_count": 1
  }
}
```

### 2. Batch Analysis

Process multiple analysis requests in parallel for improved efficiency.

```bash
mapper analysis batch-analyze --batch-file <path> [options]
```

**Options:**
- `--batch-file, -f`: Path to JSON file containing batch analysis requests (required)
- `--output, -o`: Output path for batch results (default: stdout)
- `--format`: Output format (json|yaml) [default: json]
- `--max-concurrent`: Maximum number of concurrent analyses [default: 5]

**Example:**
```bash
# Process batch analysis requests
mapper analysis batch-analyze --batch-file examples/sample_batch_analysis.json --max-concurrent 10

# Save results to file
mapper analysis batch-analyze --batch-file examples/sample_batch_analysis.json --output batch_results.json
```

**Batch Input Format:**
```json
{
  "requests": [
    {
      "metrics": {
        "tenant_id": "tenant-1",
        "detector_results": [...]
      },
      "analysis_type": "remediation",
      "tenant_id": "tenant-1",
      "request_id": "req-001"
    }
  ]
}
```

### 3. Health Check

Check the health status of the analysis module and its components.

```bash
mapper analysis health
```

**Example:**
```bash
# Check analysis module health
mapper analysis health
```

**Output:**
```
Analysis Module Health Check
==============================
Status: healthy
Version: 1.0.0
Uptime: 3600s

Component Status:
  ✓ model_server: healthy
  ✓ validator: healthy
  ✓ template_provider: healthy
```

### 4. Quality Evaluation

Evaluate the quality of analysis outputs using sample data or golden test cases.

```bash
mapper analysis quality-eval [options]
```

**Options:**
- `--sample-file, -f`: Path to sample analysis results for evaluation
- `--output, -o`: Output path for quality evaluation report

**Example:**
```bash
# Run quality evaluation
mapper analysis quality-eval

# Evaluate with sample data
mapper analysis quality-eval --sample-file sample_results.json --output quality_report.json
```

**Output:**
```
Analysis Quality Evaluation
============================
Overall Score: 8.50/10
Confidence: 0.90

Quality Metrics:
  accuracy: 0.90
  completeness: 0.85
  relevance: 0.92

Recommendations:
  • Improve accuracy in vulnerability classification
  • Add more comprehensive remediation steps
```

### 5. Cache Management

Manage the analysis module's result cache for idempotency and performance.

```bash
mapper analysis cache --action <action> [options]
```

**Actions:**
- `stats`: Show cache statistics (default)
- `clear`: Clear cache entries
- `list`: List cache entries

**Options:**
- `--pattern`: Pattern to match for cache operations (for clear/list actions)

**Examples:**
```bash
# Show cache statistics
mapper analysis cache --action stats

# Clear all cache entries
mapper analysis cache --action clear

# Clear cache entries matching a pattern
mapper analysis cache --action clear --pattern "tenant-*"

# List cache entries
mapper analysis cache --action list

# List cache entries matching a pattern
mapper analysis cache --action list --pattern "vulnerability-*"
```

**Cache Stats Output:**
```
Cache Statistics
==================
Total entries: 150
Memory usage: 25.50 MB
Hit rate: 85.20%
```

### 6. Configuration Validation

Validate the analysis module configuration to ensure proper setup.

```bash
mapper analysis validate-config [options]
```

**Options:**
- `--output, -o`: Output path for validation report

**Example:**
```bash
# Validate configuration
mapper analysis validate-config

# Save validation report
mapper analysis validate-config --output config_validation.json
```

**Output:**
```
Analysis Module Configuration Validation
=============================================
  ✓ Model name configured
  ✓ API host configured
  ✓ Confidence threshold valid
  ✓ Quality evaluation enabled

✓ All configuration checks passed
```

## Configuration

The analysis CLI uses the same configuration system as the main Llama Mapper. Configuration can be provided via:

1. **YAML Configuration File**: Use `--config` option to specify a custom config file
2. **Environment Variables**: Set analysis-specific environment variables
3. **Default Configuration**: Uses built-in defaults

### Environment Variables

Analysis-specific environment variables (prefixed with `ANALYSIS_`):

```bash
# Model configuration
ANALYSIS_MODEL_PATH=models/phi3-mini-3.8b
ANALYSIS_TEMPERATURE=0.1
ANALYSIS_CONFIDENCE_CUTOFF=0.3

# Processing configuration
ANALYSIS_MAX_CONCURRENT_REQUESTS=10
ANALYSIS_REQUEST_TIMEOUT_SECONDS=30
ANALYSIS_BATCH_SIZE_LIMIT=100

# Cache configuration
ANALYSIS_IDEMPOTENCY_CACHE_TTL_HOURS=24
ANALYSIS_CACHE_CLEANUP_INTERVAL_MINUTES=60

# Quality evaluation
ANALYSIS_QUALITY_EVALUATION_ENABLED=true
ANALYSIS_QUALITY_THRESHOLD=0.8
```

## Error Handling

The analysis CLI provides comprehensive error handling:

- **File Not Found**: Clear error messages for missing input files
- **Invalid JSON**: Validation of input file format
- **Configuration Errors**: Detailed configuration validation
- **Analysis Failures**: Graceful handling of analysis errors
- **Network Issues**: Retry logic for external service calls

## Examples

### Complete Workflow Example

```bash
# 1. Validate configuration
mapper analysis validate-config

# 2. Check health
mapper analysis health

# 3. Analyze security metrics
mapper analysis analyze --metrics-file security_scan_results.json --output analysis_results.json

# 4. Evaluate quality
mapper analysis quality-eval --sample-file analysis_results.json --output quality_report.json

# 5. Check cache statistics
mapper analysis cache --action stats
```

### Batch Processing Example

```bash
# 1. Create batch file with multiple analysis requests
cat > batch_requests.json << EOF
{
  "requests": [
    {
      "metrics": {"detector_results": [...]},
      "analysis_type": "remediation",
      "tenant_id": "tenant-1"
    },
    {
      "metrics": {"detector_results": [...]},
      "analysis_type": "policy",
      "tenant_id": "tenant-2"
    }
  ]
}
EOF

# 2. Process batch analysis
mapper analysis batch-analyze --batch-file batch_requests.json --max-concurrent 5 --output batch_results.json

# 3. Review results
cat batch_results.json | jq '.results[].analysis.explanation'
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Ensure the model path is correctly configured
   - Check that the model files exist at the specified location

2. **"Configuration validation failed"**:
   - Run `mapper analysis validate-config` to identify specific issues
   - Check environment variables and configuration file syntax

3. **"Analysis timeout"**:
   - Increase the timeout setting in configuration
   - Reduce batch size or concurrent requests

4. **"Cache errors"**:
   - Clear the cache: `mapper analysis cache --action clear`
   - Check cache configuration and storage permissions

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
mapper --log-level DEBUG analysis analyze --metrics-file sample.json
```

## Integration

The analysis CLI integrates with:

- **CI/CD Pipelines**: Use in automated testing and deployment
- **Monitoring Systems**: Health checks and quality metrics
- **Data Processing**: Batch analysis for large datasets
- **Compliance Workflows**: Automated policy generation and validation

## Performance

- **Single Analysis**: Typically 1-5 seconds per request
- **Batch Analysis**: Scales linearly with concurrent processing
- **Cache Hit Rate**: 80-90% for repeated analysis requests
- **Memory Usage**: ~100MB base + ~10MB per concurrent request

## Security

- **Input Validation**: All input files are validated before processing
- **Tenant Isolation**: Multi-tenant support with proper isolation
- **Audit Logging**: Comprehensive logging of all analysis operations
- **Idempotency**: Safe retry of analysis requests without side effects
