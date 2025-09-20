# curl examples (bash)

# Environment variables for secrets (avoid putting secrets inline)
export MAPPER_BASE_URL=${MAPPER_BASE_URL:-http://localhost:8000}
export MAPPER_API_KEY=${MAPPER_API_KEY:-YOUR_API_KEY}
export MAPPER_TENANT_ID=${MAPPER_TENANT_ID:-YOUR_TENANT_ID}

curl -sS -X POST "$MAPPER_BASE_URL/map" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MAPPER_API_KEY" \
  -H "X-Tenant-ID: $MAPPER_TENANT_ID" \
  -H "Idempotency-Key: example-req-123" \
  -d '{
    "detector": "orchestrated-multi",
    "output": "toxic|hate|pii_detected",
    "tenant_id": "'$MAPPER_TENANT_ID'",
    "metadata": {
      "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
      "aggregation_method": "weighted_average",
      "coverage_achieved": 1.0,
      "provenance": [{"detector":"deberta-toxicity","confidence":0.93}]
    }
  }'

# Windows PowerShell example
# $env:MAPPER_BASE_URL = "http://localhost:8000"
# $env:MAPPER_API_KEY = "YOUR_API_KEY"
# $env:MAPPER_TENANT_ID = "YOUR_TENANT_ID"
# Invoke-RestMethod -Method Post -Uri "$env:MAPPER_BASE_URL/map" -Headers @{
#   "Content-Type"="application/json";
#   "X-API-Key"=$env:MAPPER_API_KEY;
#   "X-Tenant-ID"=$env:MAPPER_TENANT_ID;
#   "Idempotency-Key"="example-req-123"
# } -Body (
#   '{"detector":"orchestrated-multi","output":"toxic|hate|pii_detected","tenant_id":"' + $env:MAPPER_TENANT_ID + '","metadata":{"contributing_detectors":["deberta-toxicity","openai-moderation"],"aggregation_method":"weighted_average","coverage_achieved":1.0,"provenance":[{"detector":"deberta-toxicity","confidence":0.93}]}}'
# )
