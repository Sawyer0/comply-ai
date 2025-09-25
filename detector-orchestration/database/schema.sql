-- Detector Orchestration Service Database Schema
-- This schema handles detector coordination, policy enforcement, and service discovery

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Detector registry and health monitoring
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

-- Detector execution requests and results
CREATE TABLE IF NOT EXISTS detector_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    detector_id UUID NOT NULL REFERENCES detectors(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    input_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for privacy
    execution_status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (execution_status IN ('pending', 'running', 'completed', 'failed', 'timeout')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    confidence_score DECIMAL(5,4),
    result_data JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    correlation_id UUID,
    circuit_breaker_state VARCHAR(20) DEFAULT 'closed' CHECK (circuit_breaker_state IN ('closed', 'open', 'half_open')),
    
    CONSTRAINT valid_exec_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_input_hash CHECK (length(input_hash) = 64),
    CONSTRAINT valid_execution_time CHECK (execution_time_ms IS NULL OR execution_time_ms >= 0),
    CONSTRAINT valid_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0)
);

-- Orchestration requests and aggregated results
CREATE TABLE IF NOT EXISTS orchestration_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    input_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for privacy
    detector_types TEXT[] NOT NULL,
    policy_bundle VARCHAR(100),
    processing_mode VARCHAR(20) DEFAULT 'standard' CHECK (processing_mode IN ('standard', 'fast', 'thorough')),
    max_detectors INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'partial')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_execution_time_ms INTEGER,
    detectors_executed INTEGER DEFAULT 0,
    detectors_successful INTEGER DEFAULT 0,
    coverage_achieved DECIMAL(5,4),
    aggregated_results JSONB,
    policy_violations JSONB DEFAULT '[]',
    recommendations TEXT[],
    correlation_id UUID,
    
    CONSTRAINT valid_orch_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_orch_input_hash CHECK (length(input_hash) = 64),
    CONSTRAINT valid_detector_types CHECK (array_length(detector_types, 1) > 0),
    CONSTRAINT valid_max_detectors CHECK (max_detectors IS NULL OR max_detectors > 0),
    CONSTRAINT valid_total_execution_time CHECK (total_execution_time_ms IS NULL OR total_execution_time_ms >= 0),
    CONSTRAINT valid_detectors_executed CHECK (detectors_executed >= 0),
    CONSTRAINT valid_detectors_successful CHECK (detectors_successful >= 0 AND detectors_successful <= detectors_executed),
    CONSTRAINT valid_coverage CHECK (coverage_achieved IS NULL OR (coverage_achieved >= 0 AND coverage_achieved <= 1))
);

-- Policy definitions and enforcement
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_name VARCHAR(100) NOT NULL,
    policy_bundle VARCHAR(100) NOT NULL,
    policy_version VARCHAR(50) NOT NULL,
    policy_content TEXT NOT NULL,
    policy_type VARCHAR(50) NOT NULL CHECK (policy_type IN ('detector_selection', 'conflict_resolution', 'quality_gate', 'security')),
    tenant_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_policy_name CHECK (length(policy_name) > 0),
    CONSTRAINT valid_policy_bundle CHECK (length(policy_bundle) > 0),
    CONSTRAINT valid_policy_version CHECK (length(policy_version) > 0),
    CONSTRAINT valid_policy_content CHECK (length(policy_content) > 0),
    CONSTRAINT valid_policy_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_priority CHECK (priority >= 0),
    
    UNIQUE(policy_name, policy_bundle, tenant_id)
);

-- Policy enforcement audit trail
CREATE TABLE IF NOT EXISTS policy_enforcements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    enforcement_result VARCHAR(20) NOT NULL CHECK (enforcement_result IN ('allowed', 'denied', 'modified', 'warning')),
    policy_decision JSONB NOT NULL,
    enforcement_time_ms INTEGER NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    correlation_id UUID,
    
    CONSTRAINT valid_enf_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_enforcement_time CHECK (enforcement_time_ms >= 0)
);

-- Rate limiting and throttling
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(64),
    ip_address INET,
    endpoint VARCHAR(200) NOT NULL,
    requests_count INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMPTZ NOT NULL,
    window_duration_seconds INTEGER NOT NULL,
    limit_per_window INTEGER NOT NULL,
    is_blocked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_rate_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_endpoint CHECK (length(endpoint) > 0),
    CONSTRAINT valid_requests_count CHECK (requests_count >= 0),
    CONSTRAINT valid_window_duration CHECK (window_duration_seconds > 0),
    CONSTRAINT valid_limit_per_window CHECK (limit_per_window > 0),
    
    UNIQUE(tenant_id, api_key_hash, ip_address, endpoint, window_start)
);

-- Service discovery and health monitoring
CREATE TABLE IF NOT EXISTS service_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    service_type VARCHAR(50) NOT NULL CHECK (service_type IN ('detector', 'analysis', 'mapper', 'external')),
    endpoint_url VARCHAR(500) NOT NULL,
    health_check_url VARCHAR(500),
    version VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'failed')),
    capabilities JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    tenant_id VARCHAR(100) NOT NULL,
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    last_heartbeat TIMESTAMPTZ,
    heartbeat_interval_seconds INTEGER DEFAULT 30,
    
    CONSTRAINT valid_service_name CHECK (length(service_name) > 0),
    CONSTRAINT valid_service_endpoint CHECK (length(endpoint_url) > 0),
    CONSTRAINT valid_service_version CHECK (length(version) > 0),
    CONSTRAINT valid_service_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_heartbeat_interval CHECK (heartbeat_interval_seconds > 0),
    
    UNIQUE(service_name, service_type, tenant_id)
);

-- Async job processing
CREATE TABLE IF NOT EXISTS async_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,
    job_data JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 100,
    tenant_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    result_data JSONB,
    correlation_id UUID,
    
    CONSTRAINT valid_job_type CHECK (length(job_type) > 0),
    CONSTRAINT valid_job_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_job_priority CHECK (priority >= 0),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0),
    CONSTRAINT valid_max_retries CHECK (max_retries >= 0)
);

-- Audit trail for orchestration operations
CREATE TABLE IF NOT EXISTS orchestration_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    old_values JSONB,
    new_values JSONB,
    correlation_id UUID,
    ip_address INET,
    user_agent TEXT,
    
    CONSTRAINT valid_audit_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_operation CHECK (length(operation) > 0),
    CONSTRAINT valid_resource_type CHECK (length(resource_type) > 0),
    CONSTRAINT valid_resource_id CHECK (length(resource_id) > 0)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_detectors_tenant_type ON detectors(tenant_id, detector_type);
CREATE INDEX IF NOT EXISTS idx_detectors_health ON detectors(health_status, last_health_check);
CREATE INDEX IF NOT EXISTS idx_detector_executions_request ON detector_executions(request_id, started_at);
CREATE INDEX IF NOT EXISTS idx_detector_executions_performance ON detector_executions(detector_id, execution_time_ms, started_at);
CREATE INDEX IF NOT EXISTS idx_orchestration_requests_tenant ON orchestration_requests(tenant_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_orchestration_requests_status ON orchestration_requests(status, started_at);
CREATE INDEX IF NOT EXISTS idx_policies_tenant_active ON policies(tenant_id, is_active, priority);
CREATE INDEX IF NOT EXISTS idx_policy_enforcements_request ON policy_enforcements(request_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_rate_limits_tenant_window ON rate_limits(tenant_id, window_start DESC);
CREATE INDEX IF NOT EXISTS idx_service_registry_type_status ON service_registry(service_type, status, last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_async_jobs_status_priority ON async_jobs(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_orchestration_audit_tenant_timestamp ON orchestration_audit(tenant_id, timestamp DESC);

-- Enable Row Level Security
ALTER TABLE detectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE detector_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orchestration_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_enforcements ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE service_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE async_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE orchestration_audit ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_detectors ON detectors
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_detector_executions ON detector_executions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_orchestration_requests ON orchestration_requests
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_policies ON policies
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_policy_enforcements ON policy_enforcements
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_rate_limits ON rate_limits
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_service_registry ON service_registry
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_async_jobs ON async_jobs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_orchestration_audit ON orchestration_audit
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));