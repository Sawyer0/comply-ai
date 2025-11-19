-- Detector mapping configuration per detector type and tenant

CREATE TABLE IF NOT EXISTS detector_mapping_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    detector_type VARCHAR(100) NOT NULL,
    detector_version VARCHAR(50),
    version VARCHAR(50) NOT NULL,
    schema_version VARCHAR(50) NOT NULL,
    mapping_rules JSONB NOT NULL,
    validation_schema JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive', 'deprecated', 'rolled_back')),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    backward_compatible BOOLEAN DEFAULT TRUE,
    rollback_of_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    deactivated_at TIMESTAMPTZ,
    created_by VARCHAR(100),

    CONSTRAINT valid_mapping_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_mapping_detector_type CHECK (length(detector_type) > 0),
    CONSTRAINT valid_mapping_version CHECK (length(version) > 0),
    CONSTRAINT valid_mapping_schema_version CHECK (length(schema_version) > 0),

    UNIQUE (tenant_id, detector_type, version)
);

CREATE INDEX IF NOT EXISTS idx_detector_mapping_configs_active
    ON detector_mapping_configs(tenant_id, detector_type, is_active, created_at DESC);

ALTER TABLE detector_mapping_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS tenant_isolation_detector_mapping_configs ON detector_mapping_configs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));
