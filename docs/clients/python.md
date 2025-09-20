# Python (requests) example

import os
import json
import requests

BASE_URL = os.getenv("MAPPER_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("MAPPER_API_KEY", "YOUR_API_KEY")
TENANT_ID = os.getenv("MAPPER_TENANT_ID", "YOUR_TENANT_ID")

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    "X-Tenant-ID": TENANT_ID,
    "Idempotency-Key": "example-req-123",  # optional
}

payload = {
    "detector": "orchestrated-multi",
    "output": "toxic|hate|pii_detected",
    "tenant_id": TENANT_ID,
    "metadata": {
        "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
        "aggregation_method": "weighted_average",
        "coverage_achieved": 1.0,
        "provenance": [{"detector": "deberta-toxicity", "confidence": 0.93}],
    },
}

resp = requests.post(f"{BASE_URL}/map", headers=headers, data=json.dumps(payload), timeout=5)

if resp.status_code == 200:
    data = resp.json()
    print("taxonomy:", data.get("taxonomy"))
    print("confidence:", data.get("confidence"))
    vi = data.get("version_info", {})
    print("versions:", vi)
else:
    try:
        detail = resp.json().get("detail", {})
    except Exception:
        detail = {"message": resp.text}
    print("error_code:", detail.get("error_code"))
    print("retryable:", detail.get("retryable"))
    print("message:", detail.get("message"))
