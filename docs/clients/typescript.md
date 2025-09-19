# TypeScript (fetch) example

interface MapperPayload {
  detector: string;
  output: string;
  tenant_id: string;
  metadata?: {
    contributing_detectors?: string[];
    aggregation_method?: 'weighted_average' | 'majority_vote' | 'highest_confidence' | 'most_restrictive';
    coverage_achieved?: number;
    provenance?: Array<{ detector: string; confidence?: number; output?: string; processing_time_ms?: number }>;
  };
}

interface ErrorBody {
  error_code: string;
  message?: string;
  request_id?: string;
  retryable: boolean;
}

const BASE_URL = process.env.MAPPER_BASE_URL || 'http://localhost:8000';
const API_KEY = process.env.MAPPER_API_KEY || 'YOUR_API_KEY';
const TENANT_ID = process.env.MAPPER_TENANT_ID || 'YOUR_TENANT_ID';

async function mapPayloadTS() {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
    'X-Tenant-ID': TENANT_ID,
  };

  const payload: MapperPayload = {
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
  } else {
    const body = await resp.json().catch(() => ({}));
    const detail: ErrorBody = body.detail || { error_code: 'UNKNOWN', retryable: false };
    console.error('error_code', detail.error_code, 'retryable', detail.retryable, 'message', detail.message);
  }
}

mapPayloadTS().catch(console.error);
