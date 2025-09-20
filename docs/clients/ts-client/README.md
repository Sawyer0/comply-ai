# Llama Mapper TypeScript Client (minimal)

A tiny TypeScript wrapper for the Llama Mapper API using the preferred MapperPayload request schema.

Install locally (without publish)
- npm: npm i file:docs/clients/ts-client
- yarn: yarn add file:docs/clients/ts-client

Build
- npm run build

Usage (Node 18+ or browsers with fetch)
import { MapperClient } from '@your-org/llama-mapper-client';

const client = new MapperClient({
  baseUrl: process.env.MAPPER_BASE_URL || 'http://localhost:8000',
  apiKey: process.env.MAPPER_API_KEY || 'YOUR_API_KEY',
  tenantId: process.env.MAPPER_TENANT_ID || 'YOUR_TENANT_ID',
});

const payload = {
  detector: 'orchestrated-multi',
  output: 'toxic|hate|pii_detected',
  tenant_id: 'YOUR_TENANT_ID',
  metadata: {
    contributing_detectors: ['deberta-toxicity', 'openai-moderation'],
    aggregation_method: 'weighted_average',
    coverage_achieved: 1.0,
    provenance: [{ detector: 'deberta-toxicity', confidence: 0.93 }],
  },
};

(async () => {
  const res = await client.map(payload, { idempotencyKey: 'example-req-123' });
  console.log(res.taxonomy, res.confidence, res.version_info);
})();

Notes
- For Node < 18, you will need a fetch polyfill (e.g., node-fetch or cross-fetch) and pass it in options.
