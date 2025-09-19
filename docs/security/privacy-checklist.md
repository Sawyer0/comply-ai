# Security & Privacy Checklist

Do
- Use metadata-only logging (no raw inputs, prompts, or model outputs)
- Redact PII in any user-provided strings before logging or storing
- Prefer tenant-scoped identifiers over raw content references
- Use KMS-backed encryption keys (BYOK) where available
- Audit all secret access events (secret name, operation, success/failure)
- Enforce least-privilege credentials for AWS/Vault access
- Rotate secrets regularly and automate rotation where possible

Don't
- Log request bodies, prompts, responses, or raw detector outputs
- Include API keys, tokens, passwords, or PII in logs or exceptions
- Store unencrypted sensitive data at rest
- Use shared credentials across tenants or environments

PII Redaction
- Utilize llama_mapper.security.redaction to sanitize strings and dictionaries
- Redact: emails, phones, IPs, SSNs, credit cards, JWTs, API keys

Pre-commit Lints
- The repository includes a pre-commit hook to block raw-content logging patterns
- Run: pre-commit install; pre-commit run --all-files

BYOK Verification
- When storage.kms_key_id is configured, verify that:
  - S3 uploads use ServerSideEncryption=aws:kms with SSEKMSKeyId set
  - Local encryption uses a data key derived via KMS generate_data_key
