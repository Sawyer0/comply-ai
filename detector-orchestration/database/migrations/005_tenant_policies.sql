-- Tenant policies data store for OPA integration

CREATE TABLE IF NOT EXISTS tenant_policies (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE tenant_policies IS 'Stores tenant_policies JSON blob used to seed OPA data. Single-row logical store.';
