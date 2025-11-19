-- Migration 004: Persistent detector registry schema
-- Ensures detectors table exists with indexes optimized for lookups

CREATE TABLE IF NOT EXISTS detectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detector_type VARCHAR(50) NOT NULL,
    detector_name VARCHAR(100) NOT NULL,
    endpoint_url VARCHAR(500) NOT NULL,
    health_check_url VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'failed')),
    version VARCHAR(50) NOT NULL,
    capabilities TEXT[] DEFAULT '{}',
    configuration JSONB DEFAULT '{}',
    tenant_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_health_check TIMESTAMPTZ,
    health_status VARCHAR(20) DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    response_time_ms INTEGER,
    error_rate DECIMAL(5,4) DEFAULT 0.0,
    CONSTRAINT valid_detector_type CHECK (length(detector_type) > 0),
    CONSTRAINT valid_detector_name CHECK (length(detector_name) > 0),
    CONSTRAINT valid_endpoint_url CHECK (length(endpoint_url) > 0),
    CONSTRAINT valid_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_error_rate CHECK (error_rate >= 0 AND error_rate <= 1),
    UNIQUE(detector_type, detector_name, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_detectors_tenant_type
    ON detectors(tenant_id, detector_type);

CREATE INDEX IF NOT EXISTS idx_detectors_health_status
    ON detectors(health_status, last_health_check DESC);
