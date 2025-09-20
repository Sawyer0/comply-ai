# JavaScript (fetch) example

const BASE_URL = process.env.MAPPER_BASE_URL || 'http://localhost:8000';
const API_KEY = process.env.MAPPER_API_KEY || 'YOUR_API_KEY';
const TENANT_ID = process.env.MAPPER_TENANT_ID || 'YOUR_TENANT_ID';

async function mapPayload() {
  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
    'X-Tenant-ID': TENANT_ID,
    'Idempotency-Key': 'example-req-123',
  };

  const payload = {
    detector: 'orchestrated-multi',
    output: 'toxic|hate|pii_detected',
    tenant_id: TENANT_ID,
    metadata: {
      contributing_detectors: ['deberta-toxicity', 'openai-moderation'],
      aggregation_method: 'weighted_average',
      coverage_achieved: 1.0,
      provenance: [{ detector: 'deberta-toxicity', confidence: 0.93 }],
    },
  };

  const resp = await fetch(`${BASE_URL}/map`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  if (resp.ok) {
    const data = await resp.json();
    console.log('taxonomy', data.taxonomy);
    console.log('confidence', data.confidence);
    console.log('version_info', data.version_info);
  } else {
    const body = await resp.json().catch(() => ({}));
    const detail = body.detail || {};
    console.error('error_code', detail.error_code);
    console.error('retryable', detail.retryable);
    console.error('message', detail.message);
  }
}

mapPayload().catch(console.error);
