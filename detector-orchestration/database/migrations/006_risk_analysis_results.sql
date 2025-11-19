-- Risk analysis results per orchestration request

CREATE TABLE IF NOT EXISTS risk_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    request_correlation_id UUID NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    risk_score DECIMAL(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    rules_evaluation JSONB NOT NULL,
    model_features JSONB NOT NULL,
    detector_ids TEXT[] NOT NULL,
    requested_by VARCHAR(100),
    requested_via_api_key VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_analysis_tenant_created
    ON risk_analysis_results(tenant_id, created_at DESC);

ALTER TABLE risk_analysis_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS tenant_isolation_risk_analysis_results ON risk_analysis_results
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));
