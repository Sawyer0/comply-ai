# Security Enhancements Implementation

This document describes the high-priority security enhancements implemented in the Llama Mapper system.

## Overview

The following security enhancements have been implemented:

1. **Correlation IDs** - Request tracing across all services
2. **Automated Secrets Rotation** - Scheduled rotation with rollback capabilities
3. **Advanced Input Sanitization** - Multi-layer protection against injection attacks

## 1. Correlation IDs

### Purpose
Correlation IDs provide distributed tracing capabilities, allowing you to track requests across all services and components.

### Implementation
- **Context Management**: Uses Python's `contextvars` for thread-safe correlation ID storage
- **Middleware Integration**: FastAPI middleware automatically extracts or generates correlation IDs
- **Logging Integration**: All log entries automatically include correlation IDs

### Usage

#### Automatic (Recommended)
The correlation middleware automatically handles correlation IDs:

```python
from fastapi import FastAPI
from llama_mapper.api.middleware.correlation import CorrelationMiddleware

app = FastAPI()
app.add_middleware(CorrelationMiddleware)
```

#### Manual
You can also work with correlation IDs manually:

```python
from llama_mapper.utils.correlation import (
    get_correlation_id, 
    set_correlation_id, 
    generate_correlation_id
)

# Generate a new correlation ID
corr_id = generate_correlation_id()

# Get current correlation ID
current_id = get_correlation_id()

# Set a specific correlation ID
set_correlation_id("custom-id")
```

### HTTP Headers
- **Request Header**: `X-Correlation-ID` (optional)
- **Response Header**: `X-Correlation-ID` (always present)

If no correlation ID is provided in the request, one is automatically generated.

## 2. Automated Secrets Rotation

### Purpose
Automated secrets rotation enhances security by regularly updating credentials and API keys with minimal manual intervention.

### Features
- **Scheduled Rotation**: Configurable schedules using cron expressions
- **Rollback Capability**: Automatic rollback on verification failures
- **Multiple Secret Types**: Database credentials, API keys, TLS certificates, encryption keys
- **Audit Trail**: Complete history of all rotation operations

### Configuration

Default rotation schedules:
- **Database Credentials**: Weekly (Sunday 2 AM)
- **API Keys**: Monthly (1st at 3 AM)
- **TLS Certificates**: Quarterly
- **Encryption Keys**: Semi-annually

### CLI Usage

#### Rotate Database Credentials
```bash
# Rotate specific database
mapper security rotate-db-credentials --database primary

# Rotate all databases
mapper security rotate-db-credentials --all-databases

# Dry run (show what would be rotated)
mapper security rotate-db-credentials --all-databases --dry-run
```

#### Rotate API Keys
```bash
# Rotate specific tenant
mapper security rotate-api-keys --tenant-id tenant1

# Rotate all tenants
mapper security rotate-api-keys --all-tenants

# Dry run
mapper security rotate-api-keys --all-tenants --dry-run
```

#### Start Automated Scheduler
```bash
# Start the rotation scheduler (runs continuously)
mapper security start-rotation-scheduler
```

#### View Rotation History
```bash
# Show all rotation history
mapper security rotation-history

# Show history for specific secret
mapper security rotation-history --secret-name "database/primary"
```

### Programmatic Usage

```python
from llama_mapper.security.secrets_manager import SecretsManager
from llama_mapper.security.rotation import SecretsRotationManager

# Initialize
secrets_manager = SecretsManager(settings)
rotation_manager = SecretsRotationManager(secrets_manager)

# Rotate database credentials
result = await rotation_manager.rotate_database_credentials("primary")

if result.status == RotationStatus.COMPLETED:
    print(f"✓ Rotation successful: {result.new_version}")
elif result.status == RotationStatus.ROLLED_BACK:
    print(f"⚠ Rolled back: {result.error_message}")
else:
    print(f"✗ Failed: {result.error_message}")

# Start scheduled rotations
rotation_manager.schedule_rotation_jobs()
```

### Rollback Process
1. **Verification Failure**: If new credentials fail verification, automatic rollback occurs
2. **Manual Rollback**: Previous credentials are maintained for manual rollback if needed
3. **Audit Trail**: All rollback operations are logged with correlation IDs

## 3. Advanced Input Sanitization

### Purpose
Multi-layer input sanitization provides comprehensive protection against various injection attacks and malicious input.

### Protection Against
- **SQL Injection**: Detects and prevents SQL injection attempts
- **XSS (Cross-Site Scripting)**: Blocks malicious JavaScript and HTML
- **Path Traversal**: Prevents directory traversal attacks
- **Command Injection**: Detects system command injection attempts
- **LDAP Injection**: Protects against LDAP query manipulation
- **XML Injection**: Prevents XML entity attacks

### Sanitization Levels

#### Basic
- Length validation
- Basic pattern detection
- Minimal sanitization

#### Strict (Recommended)
- HTML escaping
- Control character removal
- Comprehensive pattern detection
- Whitespace normalization

#### Paranoid
- Aggressive character filtering
- Strict length limits
- Rejects input with any malicious patterns

### Usage

#### Direct Sanitization
```python
from llama_mapper.security.input_sanitization import (
    SecuritySanitizer, 
    SanitizationLevel
)

# Create sanitizer
sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)

# Sanitize string input
clean_input = sanitizer.sanitize_string(user_input, "user_comment")

# Sanitize complex data structures
clean_data = sanitizer.sanitize_input(request_data, "api_request")

# Validate specific data types
clean_email = sanitizer.validate_email(email_input)
clean_url = sanitizer.validate_url(url_input)
clean_path = sanitizer.validate_file_path(file_path)
```

#### Pydantic Model Integration
```python
from llama_mapper.security.input_sanitization import SecureDetectorRequest

# Automatic validation and sanitization
request = SecureDetectorRequest(
    detector_type="presidio",
    content="User input content",
    metadata={"key": "value"},
    tenant_id="tenant-123"
)
# Input is automatically sanitized during validation
```

#### CLI Testing
```bash
# Test sanitization on input
mapper security test-sanitization "'; DROP TABLE users; --" --level strict

# Output:
# Original text: '; DROP TABLE users; --
# Sanitization level: strict
# --------------------------------------------------
# ⚠ Detected attacks: sql_injection
# Sanitized text: &#x27;; DROP TABLE users; --
# ⚠ Input was modified during sanitization
```

### Attack Detection
The sanitizer detects and logs various attack patterns:

```python
# Detect attacks without sanitizing
attacks = sanitizer.detect_malicious_patterns(suspicious_input)

for attack in attacks:
    print(f"Detected: {attack.value}")
    # Logs are automatically generated with correlation IDs
```

### Integration with FastAPI
```python
from fastapi import FastAPI, Depends
from llama_mapper.security.input_sanitization import (
    SecuritySanitizer, 
    SanitizationLevel
)

app = FastAPI()
sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)

@app.post("/api/detect")
async def detect_endpoint(request: SecureDetectorRequest):
    # Request is automatically sanitized by Pydantic validation
    return {"status": "processed", "detector": request.detector_type}
```

## Security Best Practices

### Correlation IDs
1. **Always Use**: Include correlation IDs in all log entries and external API calls
2. **Propagate**: Pass correlation IDs to downstream services
3. **Monitor**: Use correlation IDs for distributed tracing and debugging

### Secrets Rotation
1. **Regular Schedule**: Don't rely only on manual rotation
2. **Test Verification**: Always verify new credentials before committing
3. **Monitor History**: Regularly review rotation history for anomalies
4. **Backup Strategy**: Ensure rollback capabilities are tested

### Input Sanitization
1. **Defense in Depth**: Use multiple validation layers
2. **Appropriate Level**: Choose sanitization level based on risk
3. **Log Attacks**: Monitor and alert on detected attack patterns
4. **Regular Updates**: Keep attack patterns updated

## Monitoring and Alerting

### Correlation IDs
- Monitor for missing correlation IDs in logs
- Track request flows across services
- Alert on correlation ID propagation failures

### Secrets Rotation
- Alert on rotation failures
- Monitor rotation schedule adherence
- Track rollback frequency

### Input Sanitization
- Alert on detected attack patterns
- Monitor sanitization failure rates
- Track attack pattern trends

## Configuration

### Environment Variables
```bash
# Correlation ID header name (default: X-Correlation-ID)
CORRELATION_ID_HEADER=X-Correlation-ID

# Secrets rotation backend (vault, aws, env)
SECRETS_BACKEND=vault

# Input sanitization level (basic, strict, paranoid)
SANITIZATION_LEVEL=strict

# Maximum input length
MAX_INPUT_LENGTH=5000
```

### Configuration File
```yaml
security:
  correlation:
    header_name: "X-Correlation-ID"
  
  rotation:
    backend: "vault"
    schedules:
      database_credentials: "0 2 * * 0"  # Weekly
      api_keys: "0 3 1 * *"              # Monthly
  
  sanitization:
    level: "strict"
    max_length: 5000
    paranoid_mode: false
```

## Testing

### Unit Tests
```bash
# Run security enhancement tests
pytest tests/integration/test_security_enhancements.py -v
```

### Integration Tests
The implementation includes comprehensive integration tests covering:
- Correlation ID propagation
- Secrets rotation workflows
- Input sanitization effectiveness
- Attack pattern detection

### Manual Testing
```bash
# Test correlation IDs
curl -H "X-Correlation-ID: test-123" http://localhost:8000/api/health

# Test input sanitization
mapper security test-sanitization "<script>alert('xss')</script>" --level strict
```

## Troubleshooting

### Common Issues

#### Correlation IDs Not Appearing
- Ensure `CorrelationMiddleware` is added to FastAPI app
- Check that logging configuration includes correlation processor
- Verify context isolation in async operations

#### Secrets Rotation Failures
- Check Vault/AWS credentials and permissions
- Verify database connectivity for credential updates
- Review rotation history for error patterns

#### Input Sanitization Too Aggressive
- Adjust sanitization level (basic vs strict vs paranoid)
- Review and customize attack patterns
- Consider field-specific sanitization rules

### Debugging
```python
# Enable debug logging
import logging
logging.getLogger("llama_mapper.security").setLevel(logging.DEBUG)

# Check correlation ID context
from llama_mapper.utils.correlation import get_correlation_id
print(f"Current correlation ID: {get_correlation_id()}")

# Test sanitization patterns
sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
attacks = sanitizer.detect_malicious_patterns(test_input)
print(f"Detected attacks: {[a.value for a in attacks]}")
```

## Performance Impact

### Correlation IDs
- **Minimal**: Context variable access is very fast
- **Memory**: Small UUID string per request context
- **Network**: Additional HTTP header (~40 bytes)

### Secrets Rotation
- **Background**: Scheduled operations don't impact request processing
- **Vault/AWS**: Network calls only during rotation operations
- **Storage**: Minimal metadata storage for rotation history

### Input Sanitization
- **CPU**: Regex pattern matching has measurable but acceptable cost
- **Memory**: Compiled patterns cached for performance
- **Latency**: Adds ~1-5ms per request depending on input size and level

## Future Enhancements

### Planned Improvements
1. **OpenTelemetry Integration**: Full distributed tracing support
2. **Advanced Rotation Policies**: Custom rotation rules and conditions
3. **ML-based Attack Detection**: Behavioral analysis for input validation
4. **Performance Optimization**: Faster pattern matching algorithms

### Configuration Enhancements
1. **Dynamic Configuration**: Runtime configuration updates
2. **Tenant-specific Rules**: Per-tenant sanitization policies
3. **Custom Attack Patterns**: User-defined attack detection rules