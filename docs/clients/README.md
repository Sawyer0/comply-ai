# Client Examples for Llama Mapper

This folder provides minimal client snippets for calling the Mapper /map endpoint using the preferred MapperPayload request schema, plus a batch example.

Base URL
- Default local: http://localhost:8000

Headers
- X-API-Key: your API key (if auth.enabled)
- X-Tenant-ID: your tenant ID (if required by your deployment policy)
- Idempotency-Key: optional; providing a unique value per logical request enables safe retries

Canonical errors
- On failure, responses include a JSON body under `detail` with:
  - error_code (e.g., INVALID_REQUEST, REQUEST_TIMEOUT)
  - message
  - request_id
  - retryable (true/false)

Examples
- See specific language guides:
  - python.md
  - javascript.md
  - typescript.md
  - go.md
  - java.md
  - csharp.md
  - curl.md

Example MapperPayload
{
  "detector": "orchestrated-multi",
  "output": "toxic|hate|pii_detected",
  "tenant_id": "YOUR_TENANT_ID",
  "metadata": {
    "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
    "aggregation_method": "weighted_average",
    "coverage_achieved": 1.0,
    "provenance": [
      {"detector": "deberta-toxicity", "confidence": 0.93}
    ]
  }
}

Batch payload shape
{
  "requests": [
    { /* MapperPayload item 1 */ },
    { /* MapperPayload item 2 */ }
  ]
}
