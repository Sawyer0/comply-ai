export type AggregationMethod =
  | 'weighted_average'
  | 'majority_vote'
  | 'highest_confidence'
  | 'most_restrictive';

export interface HandoffProvenanceEntry {
  detector: string;
  confidence?: number;
  output?: string;
  processing_time_ms?: number;
}

export interface MapperPayload {
  detector: string;
  output: string;
  tenant_id: string;
  metadata?: {
    contributing_detectors?: string[];
    aggregation_method?: AggregationMethod;
    coverage_achieved?: number;
    provenance?: HandoffProvenanceEntry[];
  };
}

export interface VersionInfo {
  taxonomy: string;
  frameworks: string;
  model: string;
}

export interface MappingResponse {
  taxonomy: string[];
  scores: Record<string, number>;
  confidence: number;
  notes?: string;
  version_info?: VersionInfo;
}

export interface ErrorBody {
  error_code: string;
  message?: string;
  request_id?: string;
  retryable: boolean;
}

export interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
  tenantId?: string;
  fetchImpl?: typeof fetch;
}

export interface RequestOptions {
  idempotencyKey?: string;
}

export class MapperClient {
  private baseUrl: string;
  private apiKey?: string;
  private tenantId?: string;
  private fetchImpl: typeof fetch;

  constructor(opts: ClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '');
    this.apiKey = opts.apiKey;
    this.tenantId = opts.tenantId;
    this.fetchImpl = opts.fetchImpl || fetch;
  }

  async map(payload: MapperPayload, opts?: RequestOptions): Promise<MappingResponse> {
    const url = `${this.baseUrl}/map`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    if (this.tenantId) headers['X-Tenant-ID'] = this.tenantId;
    if (opts?.idempotencyKey) headers['Idempotency-Key'] = opts.idempotencyKey;

    const resp = await this.fetchImpl(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    const bodyText = await resp.text();

    let parsed: any = {};
    try {
      parsed = bodyText ? JSON.parse(bodyText) : {};
    } catch {
      // leave parsed as {}
    }

    if (!resp.ok) {
      const detail: ErrorBody = parsed?.detail || {
        error_code: 'UNKNOWN',
        message: bodyText,
        retryable: false,
      };
      const e = new Error(`Mapper error: ${detail.error_code} ${detail.message || ''}`);
      (e as any).detail = detail;
      throw e;
    }
    return parsed as MappingResponse;
  }

  async mapBatch(payloads: MapperPayload[], opts?: RequestOptions): Promise<{ results: MappingResponse[]; errors?: any[] }> {
    const url = `${this.baseUrl}/map/batch`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    if (this.tenantId) headers['X-Tenant-ID'] = this.tenantId;
    if (opts?.idempotencyKey) headers['Idempotency-Key'] = opts.idempotencyKey;

    const resp = await this.fetchImpl(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ requests: payloads }),
    });
    const bodyText = await resp.text();

    let parsed: any = {};
    try {
      parsed = bodyText ? JSON.parse(bodyText) : {};
    } catch {
      // leave parsed as {}
    }

    if (!resp.ok) {
      const detail: ErrorBody = parsed?.detail || {
        error_code: 'UNKNOWN',
        message: bodyText,
        retryable: false,
      };
      const e = new Error(`Mapper error: ${detail.error_code} ${detail.message || ''}`);
      (e as any).detail = detail;
      throw e;
    }
    return parsed as { results: MappingResponse[]; errors?: any[] };
  }
}
