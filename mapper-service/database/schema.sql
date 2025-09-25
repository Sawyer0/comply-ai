-- Mapper Service Database Schema
-- This schema handles core mapping, model serving, response generation, and training

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Mapping requests and results
CREATE TABLE IF NOT EXISTS mapping_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    analysis_response_id UUID NOT NULL,
    target_frameworks TEXT[] NOT NULL,
    mapping_mode VARCHAR(20) DEFAULT 'standard' CHECK (mapping_mode IN ('standard', 'fast', 'comprehensive')),
    include_validation BOOLEAN DEFAULT TRUE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    processing_time_ms INTEGER,
    correlation_id UUID,
    
    CONSTRAINT valid_mapping_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_target_frameworks CHECK (array_length(target_frameworks, 1) > 0),
    CONSTRAINT valid_mapping_processing_time CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- Individual mapping results
CREATE TABLE IF NOT EXISTS mapping_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mapping_request_id UUID NOT NULL REFERENCES mapping_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    canonical_category VARCHAR(100) NOT NULL,
    canonical_subcategory VARCHAR(100) NOT NULL,
    canonical_confidence DECIMAL(5,4) NOT NULL,
    canonical_risk_level VARCHAR(20) NOT NULL CHECK (canonical_risk_level IN ('low', 'medium', 'high', 'critical')),
    canonical_tags TEXT[] DEFAULT '{}',
    canonical_metadata JSONB DEFAULT '{}',
    framework_mappings JSONB NOT NULL,
    overall_confidence DECIMAL(5,4) NOT NULL,
    fallback_used BOOLEAN DEFAULT FALSE,
    validation_passed BOOLEAN DEFAULT TRUE,
    validation_errors TEXT[],
    validation_warnings TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_mapping_result_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_canonical_category CHECK (length(canonical_category) > 0),
    CONSTRAINT valid_canonical_subcategory CHECK (length(canonical_subcategory) > 0),
    CONSTRAINT valid_canonical_confidence CHECK (canonical_confidence >= 0 AND canonical_confidence <= 1),
    CONSTRAINT valid_overall_confidence CHECK (overall_confidence >= 0 AND overall_confidence <= 1)
);

-- Model serving and inference tracking
CREATE TABLE IF NOT EXISTS model_inferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mapping_request_id UUID NOT NULL REFERENCES mapping_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    inference_backend VARCHAR(50) NOT NULL CHECK (inference_backend IN ('vllm', 'tgi', 'cpu', 'custom')),
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    inference_time_ms INTEGER NOT NULL,
    gpu_utilization DECIMAL(5,4),
    memory_usage_mb INTEGER,
    batch_size INTEGER DEFAULT 1,
    temperature DECIMAL(3,2),
    max_tokens INTEGER,
    confidence_score DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_inference_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_model_name CHECK (length(model_name) > 0),
    CONSTRAINT valid_model_version CHECK (length(model_version) > 0),
    CONSTRAINT valid_input_tokens CHECK (input_tokens >= 0),
    CONSTRAINT valid_output_tokens CHECK (output_tokens >= 0),
    CONSTRAINT valid_inference_time CHECK (inference_time_ms >= 0),
    CONSTRAINT valid_gpu_utilization CHECK (gpu_utilization IS NULL OR (gpu_utilization >= 0 AND gpu_utilization <= 1)),
    CONSTRAINT valid_memory_usage CHECK (memory_usage_mb IS NULL OR memory_usage_mb >= 0),
    CONSTRAINT valid_batch_size CHECK (batch_size >= 1),
    CONSTRAINT valid_temperature CHECK (temperature IS NULL OR (temperature >= 0 AND temperature <= 2)),
    CONSTRAINT valid_max_tokens CHECK (max_tokens IS NULL OR max_tokens > 0),
    CONSTRAINT valid_inference_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1))
);

-- Cost tracking and monitoring
CREATE TABLE IF NOT EXISTS cost_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mapping_request_id UUID NOT NULL REFERENCES mapping_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    tokens_processed INTEGER NOT NULL,
    inference_cost DECIMAL(10,6) NOT NULL,
    storage_cost DECIMAL(10,6) DEFAULT 0.0,
    total_cost DECIMAL(10,6) NOT NULL,
    cost_per_request DECIMAL(10,6) NOT NULL,
    billing_period DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_cost_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_tokens_processed CHECK (tokens_processed >= 0),
    CONSTRAINT valid_inference_cost CHECK (inference_cost >= 0),
    CONSTRAINT valid_storage_cost CHECK (storage_cost >= 0),
    CONSTRAINT valid_total_cost CHECK (total_cost >= 0),
    CONSTRAINT valid_cost_per_request CHECK (cost_per_request >= 0)
);

-- Model training and fine-tuning
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    base_model VARCHAR(100) NOT NULL,
    training_type VARCHAR(50) NOT NULL CHECK (training_type IN ('lora', 'full_finetune', 'qlora')),
    training_data_path VARCHAR(500) NOT NULL,
    validation_data_path VARCHAR(500),
    output_model_path VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    configuration JSONB NOT NULL,
    metrics JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    training_time_minutes INTEGER,
    error_message TEXT,
    created_by VARCHAR(100),
    
    CONSTRAINT valid_training_job_name CHECK (length(job_name) > 0),
    CONSTRAINT valid_training_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_base_model CHECK (length(base_model) > 0),
    CONSTRAINT valid_training_data_path CHECK (length(training_data_path) > 0),
    CONSTRAINT valid_training_time CHECK (training_time_minutes IS NULL OR training_time_minutes >= 0),
    
    UNIQUE(job_name, tenant_id)
);

-- Model versions and deployment tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('base', 'lora', 'merged', 'quantized')),
    model_path VARCHAR(500) NOT NULL,
    parent_model_id UUID REFERENCES model_versions(id),
    training_job_id UUID REFERENCES training_jobs(id),
    configuration JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    validation_metrics JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'deprecated', 'archived')),
    deployment_status VARCHAR(20) DEFAULT 'not_deployed' CHECK (deployment_status IN ('not_deployed', 'staging', 'canary', 'production', 'rollback')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    created_by VARCHAR(100),
    
    CONSTRAINT valid_version_model_name CHECK (length(model_name) > 0),
    CONSTRAINT valid_version CHECK (length(version) > 0),
    CONSTRAINT valid_version_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_version_model_path CHECK (length(model_path) > 0),
    
    UNIQUE(model_name, version, tenant_id)
);

-- A/B testing and canary deployments
CREATE TABLE IF NOT EXISTS deployment_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    model_a_id UUID NOT NULL REFERENCES model_versions(id),
    model_b_id UUID NOT NULL REFERENCES model_versions(id),
    traffic_split_percentage INTEGER NOT NULL DEFAULT 50,
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'paused', 'completed', 'cancelled')),
    success_metrics JSONB DEFAULT '{}',
    current_metrics JSONB DEFAULT '{}',
    statistical_significance DECIMAL(5,4),
    winner_model_id UUID REFERENCES model_versions(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    created_by VARCHAR(100),
    
    CONSTRAINT valid_experiment_name CHECK (length(experiment_name) > 0),
    CONSTRAINT valid_experiment_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_traffic_split CHECK (traffic_split_percentage >= 0 AND traffic_split_percentage <= 100),
    CONSTRAINT valid_statistical_significance CHECK (statistical_significance IS NULL OR (statistical_significance >= 0 AND statistical_significance <= 1)),
    CONSTRAINT different_models CHECK (model_a_id != model_b_id),
    
    UNIQUE(experiment_name, tenant_id)
);

-- Taxonomy management and versioning
CREATE TABLE IF NOT EXISTS taxonomies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    taxonomy_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    taxonomy_data JSONB NOT NULL,
    schema_version VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    backward_compatible BOOLEAN DEFAULT TRUE,
    migration_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    created_by VARCHAR(100),
    
    CONSTRAINT valid_taxonomy_name CHECK (length(taxonomy_name) > 0),
    CONSTRAINT valid_taxonomy_version CHECK (length(version) > 0),
    CONSTRAINT valid_taxonomy_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_schema_version CHECK (length(schema_version) > 0),
    
    UNIQUE(taxonomy_name, version, tenant_id)
);

-- Framework mappings and configurations
CREATE TABLE IF NOT EXISTS framework_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_name VARCHAR(100) NOT NULL,
    framework_version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    mapping_rules JSONB NOT NULL,
    validation_schema JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_framework_name CHECK (length(framework_name) > 0),
    CONSTRAINT valid_framework_version CHECK (length(framework_version) > 0),
    CONSTRAINT valid_framework_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_framework_priority CHECK (priority >= 0),
    
    UNIQUE(framework_name, framework_version, tenant_id)
);

-- Validation rules and schemas
CREATE TABLE IF NOT EXISTS validation_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_name VARCHAR(100) NOT NULL,
    schema_version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    schema_type VARCHAR(50) NOT NULL CHECK (schema_type IN ('input', 'output', 'framework', 'taxonomy')),
    json_schema JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_schema_name CHECK (length(schema_name) > 0),
    CONSTRAINT valid_validation_schema_version CHECK (length(schema_version) > 0),
    CONSTRAINT valid_validation_tenant_id CHECK (length(tenant_id) > 0),
    
    UNIQUE(schema_name, schema_version, tenant_id)
);

-- Feature flags and configuration
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_name VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    is_enabled BOOLEAN DEFAULT FALSE,
    rollout_percentage INTEGER DEFAULT 0,
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_flag_name CHECK (length(flag_name) > 0),
    CONSTRAINT valid_flag_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_rollout_percentage CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    
    UNIQUE(flag_name, tenant_id)
);

-- Storage and artifact management
CREATE TABLE IF NOT EXISTS storage_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type VARCHAR(50) NOT NULL CHECK (artifact_type IN ('model', 'checkpoint', 'dataset', 'config', 'log')),
    artifact_name VARCHAR(200) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    storage_backend VARCHAR(50) NOT NULL CHECK (storage_backend IN ('s3', 'minio', 'local', 'azure')),
    size_bytes BIGINT,
    checksum VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_artifact_name CHECK (length(artifact_name) > 0),
    CONSTRAINT valid_artifact_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_storage_path CHECK (length(storage_path) > 0),
    CONSTRAINT valid_size_bytes CHECK (size_bytes IS NULL OR size_bytes >= 0),
    
    UNIQUE(artifact_name, tenant_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_mapping_requests_tenant_status ON mapping_requests(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_mapping_results_request ON mapping_results(mapping_request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_model_inferences_model_performance ON model_inferences(model_name, model_version, inference_time_ms);
CREATE INDEX IF NOT EXISTS idx_model_inferences_tenant_date ON model_inferences(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_metrics_tenant_period ON cost_metrics(tenant_id, billing_period DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_tenant_status ON training_jobs(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_versions_tenant_status ON model_versions(tenant_id, status, deployment_status);
CREATE INDEX IF NOT EXISTS idx_deployment_experiments_tenant_status ON deployment_experiments(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_taxonomies_tenant_active ON taxonomies(tenant_id, is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_framework_configs_tenant_active ON framework_configs(tenant_id, is_active, priority);
CREATE INDEX IF NOT EXISTS idx_validation_schemas_tenant_type ON validation_schemas(tenant_id, schema_type, is_active);
CREATE INDEX IF NOT EXISTS idx_feature_flags_tenant_enabled ON feature_flags(tenant_id, is_enabled, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_storage_artifacts_tenant_type ON storage_artifacts(tenant_id, artifact_type, created_at DESC);

-- API Keys for authentication and authorization
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id VARCHAR(100) NOT NULL UNIQUE,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    tenant_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    permissions TEXT[] NOT NULL DEFAULT '{}',
    scopes TEXT[] NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    usage_count INTEGER NOT NULL DEFAULT 0,
    rate_limit_per_minute INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_api_key_id CHECK (length(key_id) > 0),
    CONSTRAINT valid_api_key_hash CHECK (length(key_hash) = 64),
    CONSTRAINT valid_api_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_api_key_name CHECK (length(name) > 0),
    CONSTRAINT valid_rate_limit CHECK (rate_limit_per_minute IS NULL OR rate_limit_per_minute > 0),
    CONSTRAINT valid_usage_count CHECK (usage_count >= 0)
);

-- API Key usage tracking for analytics and billing
CREATE TABLE IF NOT EXISTS api_key_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    tokens_processed INTEGER DEFAULT 0,
    cost_cents DECIMAL(10,2) DEFAULT 0.0,
    user_agent TEXT,
    ip_address INET,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_usage_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_endpoint CHECK (length(endpoint) > 0),
    CONSTRAINT valid_method CHECK (method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH')),
    CONSTRAINT valid_status_code CHECK (status_code >= 100 AND status_code < 600),
    CONSTRAINT valid_response_time CHECK (response_time_ms >= 0),
    CONSTRAINT valid_tokens_processed CHECK (tokens_processed >= 0),
    CONSTRAINT valid_cost CHECK (cost_cents >= 0),
    CONSTRAINT valid_request_size CHECK (request_size_bytes IS NULL OR request_size_bytes >= 0),
    CONSTRAINT valid_response_size CHECK (response_size_bytes IS NULL OR response_size_bytes >= 0)
);

-- Database connection management
CREATE TABLE IF NOT EXISTS database_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_name VARCHAR(100) NOT NULL UNIQUE,
    connection_type VARCHAR(50) NOT NULL CHECK (connection_type IN ('postgresql', 'redis', 'clickhouse')),
    host VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    database_name VARCHAR(100) NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255),
    ssl_enabled BOOLEAN DEFAULT TRUE,
    pool_size INTEGER DEFAULT 10,
    max_overflow INTEGER DEFAULT 20,
    connection_timeout INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT TRUE,
    health_check_interval INTEGER DEFAULT 60,
    last_health_check TIMESTAMPTZ,
    health_status VARCHAR(20) DEFAULT 'unknown' CHECK (health_status IN ('healthy', 'unhealthy', 'unknown')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_connection_name CHECK (length(connection_name) > 0),
    CONSTRAINT valid_host CHECK (length(host) > 0),
    CONSTRAINT valid_port CHECK (port > 0 AND port <= 65535),
    CONSTRAINT valid_database_name CHECK (length(database_name) > 0),
    CONSTRAINT valid_username CHECK (length(username) > 0),
    CONSTRAINT valid_pool_size CHECK (pool_size > 0),
    CONSTRAINT valid_max_overflow CHECK (max_overflow >= 0),
    CONSTRAINT valid_connection_timeout CHECK (connection_timeout > 0),
    CONSTRAINT valid_health_check_interval CHECK (health_check_interval > 0)
);

-- Create indexes for API key management
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_active ON api_keys(tenant_id, is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_key_usage_key_date ON api_key_usage(api_key_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_tenant_date ON api_key_usage(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_endpoint_performance ON api_key_usage(endpoint, response_time_ms);
CREATE INDEX IF NOT EXISTS idx_database_connections_type_active ON database_connections(connection_type, is_active);
CREATE INDEX IF NOT EXISTS idx_database_connections_health ON database_connections(health_status, last_health_check);

-- Enable Row Level Security
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_key_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE mapping_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE mapping_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_inferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE cost_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE deployment_experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE taxonomies ENABLE ROW LEVEL SECURITY;
ALTER TABLE framework_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_schemas ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_flags ENABLE ROW LEVEL SECURITY;
ALTER TABLE storage_artifacts ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_api_keys ON api_keys
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_api_key_usage ON api_key_usage
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_mapping_requests ON mapping_requests
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_mapping_results ON mapping_results
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_model_inferences ON model_inferences
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_cost_metrics ON cost_metrics
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_training_jobs ON training_jobs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_model_versions ON model_versions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_deployment_experiments ON deployment_experiments
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_taxonomies ON taxonomies
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_framework_configs ON framework_configs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_validation_schemas ON validation_schemas
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_feature_flags ON feature_flags
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_storage_artifacts ON storage_artifacts
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));