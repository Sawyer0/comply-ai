-- Analysis Service Database Schema
-- This schema handles advanced analysis, risk scoring, compliance intelligence, and RAG system

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector"; -- For embeddings storage

-- Analysis requests and results
CREATE TABLE IF NOT EXISTS analysis_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    orchestration_response_id UUID NOT NULL,
    analysis_types TEXT[] NOT NULL,
    frameworks TEXT[],
    include_recommendations BOOLEAN DEFAULT TRUE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    processing_time_ms INTEGER,
    correlation_id UUID,
    
    CONSTRAINT valid_analysis_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_analysis_types CHECK (array_length(analysis_types, 1) > 0),
    CONSTRAINT valid_processing_time CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- Canonical taxonomy results
CREATE TABLE IF NOT EXISTS canonical_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID NOT NULL REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_canonical_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_category CHECK (length(category) > 0),
    CONSTRAINT valid_subcategory CHECK (length(subcategory) > 0),
    CONSTRAINT valid_canonical_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Pattern analysis results
CREATE TABLE IF NOT EXISTS pattern_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID NOT NULL REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    temporal_patterns JSONB DEFAULT '[]',
    frequency_patterns JSONB DEFAULT '[]',
    correlation_patterns JSONB DEFAULT '[]',
    anomaly_patterns JSONB DEFAULT '[]',
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_pattern_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_pattern_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Risk scoring results
CREATE TABLE IF NOT EXISTS risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID NOT NULL REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    overall_risk_score DECIMAL(5,4) NOT NULL,
    technical_risk DECIMAL(5,4) NOT NULL,
    business_risk DECIMAL(5,4) NOT NULL,
    regulatory_risk DECIMAL(5,4) NOT NULL,
    temporal_risk DECIMAL(5,4) NOT NULL,
    risk_factors JSONB DEFAULT '[]',
    mitigation_recommendations TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_risk_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_overall_risk CHECK (overall_risk_score >= 0 AND overall_risk_score <= 1),
    CONSTRAINT valid_technical_risk CHECK (technical_risk >= 0 AND technical_risk <= 1),
    CONSTRAINT valid_business_risk CHECK (business_risk >= 0 AND business_risk <= 1),
    CONSTRAINT valid_regulatory_risk CHECK (regulatory_risk >= 0 AND regulatory_risk <= 1),
    CONSTRAINT valid_temporal_risk CHECK (temporal_risk >= 0 AND temporal_risk <= 1)
);

-- Compliance framework mappings
CREATE TABLE IF NOT EXISTS compliance_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID NOT NULL REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    mappings JSONB NOT NULL,
    compliance_score DECIMAL(5,4) NOT NULL,
    gaps JSONB DEFAULT '[]',
    recommendations TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_compliance_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_framework CHECK (length(framework) > 0),
    CONSTRAINT valid_compliance_score CHECK (compliance_score >= 0 AND compliance_score <= 1)
);

-- RAG system knowledge base
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    document_title VARCHAR(500) NOT NULL,
    document_type VARCHAR(50) NOT NULL CHECK (document_type IN ('regulation', 'best_practice', 'case_study', 'framework')),
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    metadata JSONB DEFAULT '{}',
    framework VARCHAR(50),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_document_id CHECK (length(document_id) > 0),
    CONSTRAINT valid_document_title CHECK (length(document_title) > 0),
    CONSTRAINT valid_content CHECK (length(content) > 0),
    CONSTRAINT valid_content_hash CHECK (length(content_hash) = 64),
    
    UNIQUE(document_id, version)
);

-- RAG query results and insights
CREATE TABLE IF NOT EXISTS rag_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID NOT NULL REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    query_text TEXT NOT NULL,
    relevant_regulations JSONB DEFAULT '[]',
    compliance_guidance TEXT[],
    risk_context TEXT[],
    remediation_steps TEXT[],
    confidence DECIMAL(5,4) NOT NULL,
    retrieved_documents UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_rag_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_query_text CHECK (length(query_text) > 0),
    CONSTRAINT valid_rag_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Quality monitoring and evaluation
CREATE TABLE IF NOT EXISTS quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    model_version VARCHAR(100),
    evaluation_date DATE NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_quality_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_metric_type CHECK (length(metric_type) > 0),
    CONSTRAINT valid_metric_name CHECK (length(metric_name) > 0)
);

-- Quality alerts and degradation detection
CREATE TABLE IF NOT EXISTS quality_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    metric_name VARCHAR(100),
    current_value DECIMAL(10,6),
    threshold_value DECIMAL(10,6),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolution_notes TEXT,
    
    CONSTRAINT valid_alert_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_alert_type CHECK (length(alert_type) > 0),
    CONSTRAINT valid_alert_message CHECK (length(message) > 0)
);

-- Weekly evaluation results
CREATE TABLE IF NOT EXISTS weekly_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    evaluation_week DATE NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    accuracy_score DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    confidence_distribution JSONB,
    performance_trends JSONB,
    recommendations TEXT[],
    status VARCHAR(20) NOT NULL DEFAULT 'completed' CHECK (status IN ('running', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_eval_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_model_version CHECK (length(model_version) > 0),
    CONSTRAINT valid_accuracy CHECK (accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 1)),
    CONSTRAINT valid_precision CHECK (precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)),
    CONSTRAINT valid_recall CHECK (recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)),
    CONSTRAINT valid_f1 CHECK (f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)),
    
    UNIQUE(tenant_id, evaluation_week, model_version)
);

-- ML model metadata and performance tracking
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('phi3', 'embedding', 'custom')),
    model_path VARCHAR(500),
    configuration JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    tenant_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'training', 'deprecated')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_ml_model_name CHECK (length(model_name) > 0),
    CONSTRAINT valid_ml_model_version CHECK (length(model_version) > 0),
    CONSTRAINT valid_ml_tenant_id CHECK (length(tenant_id) > 0),
    
    UNIQUE(model_name, model_version, tenant_id)
);

-- Analysis pipeline configurations
CREATE TABLE IF NOT EXISTS analysis_pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_name VARCHAR(100) NOT NULL,
    pipeline_version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    configuration JSONB NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    
    CONSTRAINT valid_pipeline_name CHECK (length(pipeline_name) > 0),
    CONSTRAINT valid_pipeline_version CHECK (length(pipeline_version) > 0),
    CONSTRAINT valid_pipeline_tenant_id CHECK (length(tenant_id) > 0),
    
    UNIQUE(pipeline_name, pipeline_version, tenant_id)
);

-- Tenant configurations for multi-tenancy
CREATE TABLE IF NOT EXISTS tenant_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'trial')),
    configuration JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_tenant_config_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_tenant_name CHECK (length(name) > 0)
);

-- Resource quotas for tenants
CREATE TABLE IF NOT EXISTS tenant_resource_quotas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL REFERENCES tenant_configurations(tenant_id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL CHECK (resource_type IN ('analysis_requests', 'batch_requests', 'storage_mb', 'cpu_minutes', 'ml_inference_calls')),
    quota_limit INTEGER NOT NULL CHECK (quota_limit >= 0),
    current_usage INTEGER NOT NULL DEFAULT 0 CHECK (current_usage >= 0),
    period_hours INTEGER NOT NULL DEFAULT 24 CHECK (period_hours > 0),
    reset_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_quota_tenant_id CHECK (length(tenant_id) > 0),
    UNIQUE(tenant_id, resource_type)
);

-- Plugin registry for managing analysis plugins
CREATE TABLE IF NOT EXISTS plugin_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plugin_name VARCHAR(100) NOT NULL UNIQUE,
    plugin_version VARCHAR(50) NOT NULL,
    plugin_type VARCHAR(50) NOT NULL CHECK (plugin_type IN ('analysis_engine', 'ml_model', 'quality_evaluator', 'pattern_detector', 'risk_scorer', 'compliance_mapper')),
    status VARCHAR(20) NOT NULL DEFAULT 'inactive' CHECK (status IN ('active', 'inactive', 'error', 'loading')),
    metadata JSONB NOT NULL DEFAULT '{}',
    configuration JSONB NOT NULL DEFAULT '{}',
    capabilities TEXT[] DEFAULT '{}',
    supported_frameworks TEXT[] DEFAULT '{}',
    min_confidence_threshold DECIMAL(3,2) DEFAULT 0.0 CHECK (min_confidence_threshold >= 0 AND min_confidence_threshold <= 1),
    max_batch_size INTEGER,
    plugin_path VARCHAR(500),
    dependencies TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_health_check TIMESTAMPTZ,
    health_status JSONB DEFAULT '{}',
    error_log TEXT[] DEFAULT '{}',
    
    CONSTRAINT valid_plugin_name CHECK (length(plugin_name) > 0),
    CONSTRAINT valid_plugin_version CHECK (length(plugin_version) > 0)
);

-- Plugin execution history and performance tracking
CREATE TABLE IF NOT EXISTS plugin_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plugin_name VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    request_id VARCHAR(100) NOT NULL,
    execution_type VARCHAR(20) NOT NULL CHECK (execution_type IN ('single', 'batch')),
    input_hash VARCHAR(64) NOT NULL, -- Hash of input data for privacy
    confidence DECIMAL(5,4),
    processing_time_ms INTEGER NOT NULL CHECK (processing_time_ms >= 0),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_plugin_exec_name CHECK (length(plugin_name) > 0),
    CONSTRAINT valid_plugin_exec_tenant CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_plugin_exec_request CHECK (length(request_id) > 0),
    CONSTRAINT valid_input_hash CHECK (length(input_hash) = 64),
    CONSTRAINT valid_exec_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Tenant-specific analytics and customization
CREATE TABLE IF NOT EXISTS tenant_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL REFERENCES tenant_configurations(tenant_id) ON DELETE CASCADE,
    analytics_type VARCHAR(50) NOT NULL,
    time_period VARCHAR(20) NOT NULL CHECK (time_period IN ('daily', 'weekly', 'monthly', 'quarterly')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Request metrics
    total_requests INTEGER DEFAULT 0 CHECK (total_requests >= 0),
    successful_requests INTEGER DEFAULT 0 CHECK (successful_requests >= 0),
    failed_requests INTEGER DEFAULT 0 CHECK (failed_requests >= 0),
    
    -- Performance metrics
    avg_response_time_ms DECIMAL(10,2) DEFAULT 0.0 CHECK (avg_response_time_ms >= 0),
    p95_response_time_ms DECIMAL(10,2) DEFAULT 0.0 CHECK (p95_response_time_ms >= 0),
    
    -- Quality metrics
    avg_confidence_score DECIMAL(5,4) DEFAULT 0.0 CHECK (avg_confidence_score >= 0 AND avg_confidence_score <= 1),
    low_confidence_count INTEGER DEFAULT 0 CHECK (low_confidence_count >= 0),
    
    -- Resource usage
    cpu_minutes_used DECIMAL(10,2) DEFAULT 0.0 CHECK (cpu_minutes_used >= 0),
    storage_mb_used DECIMAL(10,2) DEFAULT 0.0 CHECK (storage_mb_used >= 0),
    ml_inference_calls INTEGER DEFAULT 0 CHECK (ml_inference_calls >= 0),
    
    -- Analysis breakdown
    pattern_recognition_count INTEGER DEFAULT 0 CHECK (pattern_recognition_count >= 0),
    risk_scoring_count INTEGER DEFAULT 0 CHECK (risk_scoring_count >= 0),
    compliance_mapping_count INTEGER DEFAULT 0 CHECK (compliance_mapping_count >= 0),
    
    -- Framework usage and error breakdown
    framework_usage JSONB DEFAULT '{}',
    error_types JSONB DEFAULT '{}',
    
    -- Additional insights
    insights TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_analytics_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_analytics_type CHECK (length(analytics_type) > 0),
    CONSTRAINT valid_period_range CHECK (period_end >= period_start),
    CONSTRAINT valid_success_rate CHECK (successful_requests <= total_requests),
    CONSTRAINT valid_failure_rate CHECK (failed_requests <= total_requests),
    
    UNIQUE(tenant_id, analytics_type, time_period, period_start)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_requests_tenant_status ON analysis_requests(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_canonical_results_request ON canonical_results(analysis_request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_request ON pattern_analysis(analysis_request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_risk_scores_request ON risk_scores(analysis_request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_compliance_mappings_framework ON compliance_mappings(framework, tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_embedding ON knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_hash ON knowledge_base(content_hash);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_framework_category ON knowledge_base(framework, category, subcategory);
CREATE INDEX IF NOT EXISTS idx_rag_insights_request ON rag_insights(analysis_request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_tenant_date ON quality_metrics(tenant_id, evaluation_date DESC, metric_type);
CREATE INDEX IF NOT EXISTS idx_quality_alerts_tenant_status ON quality_alerts(tenant_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_weekly_evaluations_tenant_week ON weekly_evaluations(tenant_id, evaluation_week DESC);
CREATE INDEX IF NOT EXISTS idx_ml_models_tenant_status ON ml_models(tenant_id, status, model_type);
CREATE INDEX IF NOT EXISTS idx_analysis_pipelines_tenant_enabled ON analysis_pipelines(tenant_id, enabled, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tenant_analytics_tenant_period ON tenant_analytics(tenant_id, time_period, period_start DESC);

-- Indexes for tenant management
CREATE INDEX IF NOT EXISTS idx_tenant_configurations_status ON tenant_configurations(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tenant_resource_quotas_tenant ON tenant_resource_quotas(tenant_id, resource_type);
CREATE INDEX IF NOT EXISTS idx_tenant_resource_quotas_reset ON tenant_resource_quotas(reset_at) WHERE current_usage > 0;

-- Indexes for plugin management
CREATE INDEX IF NOT EXISTS idx_plugin_registry_type_status ON plugin_registry(plugin_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_plugin_registry_name_version ON plugin_registry(plugin_name, plugin_version);
CREATE INDEX IF NOT EXISTS idx_plugin_executions_plugin_tenant ON plugin_executions(plugin_name, tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plugin_executions_success ON plugin_executions(success, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plugin_executions_performance ON plugin_executions(plugin_name, processing_time_ms) WHERE success = true;

-- Enable Row Level Security
ALTER TABLE analysis_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_pipelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_resource_quotas ENABLE ROW LEVEL SECURITY;
ALTER TABLE plugin_executions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_analysis_requests ON analysis_requests
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_canonical_results ON canonical_results
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_pattern_analysis ON pattern_analysis
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_risk_scores ON risk_scores
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_compliance_mappings ON compliance_mappings
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Knowledge base is shared across tenants but access controlled by application logic
CREATE POLICY IF NOT EXISTS knowledge_base_access ON knowledge_base
    FOR SELECT
    USING (true);

CREATE POLICY IF NOT EXISTS tenant_isolation_rag_insights ON rag_insights
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_quality_metrics ON quality_metrics
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_quality_alerts ON quality_alerts
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_weekly_evaluations ON weekly_evaluations
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_ml_models ON ml_models
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_analysis_pipelines ON analysis_pipelines
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_tenant_analytics ON tenant_analytics
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- RLS policies for tenant management
CREATE POLICY IF NOT EXISTS tenant_isolation_tenant_configurations ON tenant_configurations
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_tenant_resource_quotas ON tenant_resource_quotas
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_plugin_executions ON plugin_executions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Plugin registry is accessible to all tenants but managed by admin
CREATE POLICY IF NOT EXISTS plugin_registry_read_access ON plugin_registry
    FOR SELECT
    USING (true);

CREATE POLICY IF NOT EXISTS plugin_registry_admin_access ON plugin_registry
    FOR INSERT, UPDATE, DELETE
    USING (current_setting('app.current_user_role', true) = 'admin');
    
-- Privacy validation logs table
CREATE TABLE IF NOT EXISTS privacy_validation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    is_compliant BOOLEAN NOT NULL DEFAULT false,
    violations_count INTEGER NOT NULL DEFAULT 0,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    validation_details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_privacy_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_privacy_data_type CHECK (length(data_type) > 0),
    CONSTRAINT valid_violations_count CHECK (violations_count >= 0),
    CONSTRAINT valid_warnings_count CHECK (warnings_count >= 0)
);

-- Data retention compliance logs table
CREATE TABLE IF NOT EXISTS data_retention_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data_age_days INTEGER NOT NULL,
    retention_status VARCHAR(20) NOT NULL CHECK (retention_status IN ('compliant', 'warning', 'violation')),
    check_details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_retention_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_retention_data_type CHECK (length(data_type) > 0),
    CONSTRAINT valid_data_age CHECK (data_age_days >= 0)
);

-- Risk factors table (for detailed risk breakdown)
CREATE TABLE IF NOT EXISTS risk_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_score_id UUID REFERENCES risk_scores(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    weight DECIMAL(5,4) NOT NULL CHECK (weight >= 0 AND weight <= 1),
    value DECIMAL(5,4) NOT NULL CHECK (value >= 0 AND value <= 1),
    contribution DECIMAL(5,4) NOT NULL CHECK (contribution >= 0 AND contribution <= 1),
    justification TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_factor_name CHECK (length(factor_name) > 0)
);

-- Security findings table (for risk scoring input)
CREATE TABLE IF NOT EXISTS security_findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_request_id UUID REFERENCES analysis_requests(id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL,
    finding_id VARCHAR(255) NOT NULL,
    detector_id VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical', 'informational')),
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_findings_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_finding_id CHECK (length(finding_id) > 0),
    CONSTRAINT valid_detector_id CHECK (length(detector_id) > 0),
    CONSTRAINT valid_category CHECK (length(category) > 0),
    CONSTRAINT valid_description CHECK (length(description) > 0)
);

-- Additional indexes for new tables
CREATE INDEX IF NOT EXISTS idx_privacy_validation_logs_tenant_created ON privacy_validation_logs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_privacy_validation_logs_analysis_request ON privacy_validation_logs(analysis_request_id);
CREATE INDEX IF NOT EXISTS idx_privacy_validation_logs_compliance ON privacy_validation_logs(tenant_id, is_compliant, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_data_retention_logs_tenant_created ON data_retention_logs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_retention_logs_status ON data_retention_logs(tenant_id, retention_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_retention_logs_data_type ON data_retention_logs(data_type, retention_status);

CREATE INDEX IF NOT EXISTS idx_risk_factors_risk_score ON risk_factors(risk_score_id);
CREATE INDEX IF NOT EXISTS idx_risk_factors_contribution ON risk_factors(risk_score_id, contribution DESC);

CREATE INDEX IF NOT EXISTS idx_security_findings_analysis_request ON security_findings(analysis_request_id);
CREATE INDEX IF NOT EXISTS idx_security_findings_tenant_created ON security_findings(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_findings_severity ON security_findings(tenant_id, severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_findings_detector ON security_findings(detector_id, severity, created_at DESC);

-- Row-level security policies for multi-tenancy (new tables)
ALTER TABLE privacy_validation_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_retention_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_findings ENABLE ROW LEVEL SECURITY;

-- Privacy validation logs tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_privacy_logs ON privacy_validation_logs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Data retention logs tenant isolation  
CREATE POLICY IF NOT EXISTS tenant_isolation_retention_logs ON data_retention_logs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Risk factors inherit tenant isolation from risk_scores table
CREATE POLICY IF NOT EXISTS tenant_isolation_risk_factors ON risk_factors
    FOR ALL USING (
        risk_score_id IN (
            SELECT id FROM risk_scores 
            WHERE tenant_id = current_setting('app.current_tenant_id', true)
        )
    );

-- Security findings tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_security_findings ON security_findings
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));
-- Sec
urity Management Tables
-- These tables support the security components: authentication, authorization, 
-- rate limiting, content scanning, and audit logging

-- API Keys table for secure authentication
CREATE TABLE IF NOT EXISTS api_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash of the API key
    tenant_id VARCHAR(100) NOT NULL,
    permissions TEXT[] NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used TIMESTAMPTZ,
    created_by VARCHAR(100),
    description TEXT,
    
    CONSTRAINT valid_api_key_hash CHECK (length(key_hash) = 64),
    CONSTRAINT valid_api_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_permissions CHECK (array_length(permissions, 1) > 0)
);

-- Security audit log for tracking all security events
CREATE TABLE IF NOT EXISTS security_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    tenant_id VARCHAR(100),
    details JSONB NOT NULL DEFAULT '{}',
    source VARCHAR(50) NOT NULL DEFAULT 'analysis_service',
    correlation_id UUID,
    
    CONSTRAINT valid_audit_event_type CHECK (length(event_type) > 0),
    CONSTRAINT valid_audit_source CHECK (length(source) > 0)
);

-- Rate limiting buckets (could also use Redis, but DB provides persistence)
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bucket_key VARCHAR(255) NOT NULL UNIQUE, -- client_id:endpoint format
    tokens DECIMAL(10,2) NOT NULL DEFAULT 0,
    last_refill TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_bucket_key CHECK (length(bucket_key) > 0),
    CONSTRAINT valid_tokens CHECK (tokens >= 0)
);

-- Content security scan results
CREATE TABLE IF NOT EXISTS content_security_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of scanned content
    scan_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_safe BOOLEAN NOT NULL DEFAULT TRUE,
    risk_score DECIMAL(5,4) NOT NULL DEFAULT 0.0,
    threats_detected JSONB NOT NULL DEFAULT '[]',
    scan_duration_ms INTEGER NOT NULL DEFAULT 0,
    tenant_id VARCHAR(100),
    
    CONSTRAINT valid_content_hash CHECK (length(content_hash) = 64),
    CONSTRAINT valid_scan_risk_score CHECK (risk_score >= 0 AND risk_score <= 1),
    CONSTRAINT valid_scan_duration CHECK (scan_duration_ms >= 0)
);

-- JWT token blacklist for revoked tokens
CREATE TABLE IF NOT EXISTS jwt_token_blacklist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jti VARCHAR(255) NOT NULL UNIQUE, -- JWT ID claim
    user_id VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100),
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    reason VARCHAR(100),
    
    CONSTRAINT valid_jti CHECK (length(jti) > 0),
    CONSTRAINT valid_blacklist_user_id CHECK (length(user_id) > 0)
);

-- Security configuration per tenant
CREATE TABLE IF NOT EXISTS tenant_security_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL UNIQUE,
    rate_limit_per_minute INTEGER NOT NULL DEFAULT 100,
    rate_limit_burst_size INTEGER NOT NULL DEFAULT 20,
    max_input_size_bytes INTEGER NOT NULL DEFAULT 1048576, -- 1MB
    enable_content_scanning BOOLEAN NOT NULL DEFAULT TRUE,
    blocked_patterns TEXT[] DEFAULT '{}',
    allowed_content_types TEXT[] DEFAULT '{"application/json", "text/plain"}',
    jwt_expiration_minutes INTEGER NOT NULL DEFAULT 60,
    require_mfa BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_security_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_rate_limit CHECK (rate_limit_per_minute > 0),
    CONSTRAINT valid_burst_size CHECK (rate_limit_burst_size > 0),
    CONSTRAINT valid_max_input_size CHECK (max_input_size_bytes > 0),
    CONSTRAINT valid_jwt_expiration CHECK (jwt_expiration_minutes > 0)
);

-- Security metrics aggregation table
CREATE TABLE IF NOT EXISTS security_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    
    -- Authentication metrics
    total_auth_attempts INTEGER NOT NULL DEFAULT 0,
    successful_auth INTEGER NOT NULL DEFAULT 0,
    failed_auth INTEGER NOT NULL DEFAULT 0,
    
    -- Authorization metrics
    total_authz_checks INTEGER NOT NULL DEFAULT 0,
    successful_authz INTEGER NOT NULL DEFAULT 0,
    failed_authz INTEGER NOT NULL DEFAULT 0,
    
    -- Rate limiting metrics
    total_requests INTEGER NOT NULL DEFAULT 0,
    rate_limited_requests INTEGER NOT NULL DEFAULT 0,
    
    -- Content security metrics
    content_scans_performed INTEGER NOT NULL DEFAULT 0,
    threats_detected INTEGER NOT NULL DEFAULT 0,
    high_risk_content INTEGER NOT NULL DEFAULT 0,
    
    -- API key metrics
    active_api_keys INTEGER NOT NULL DEFAULT 0,
    expired_api_keys INTEGER NOT NULL DEFAULT 0,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_metrics_tenant_id CHECK (length(tenant_id) > 0),
    CONSTRAINT valid_auth_success_rate CHECK (successful_auth <= total_auth_attempts),
    CONSTRAINT valid_authz_success_rate CHECK (successful_authz <= total_authz_checks),
    CONSTRAINT valid_rate_limit_rate CHECK (rate_limited_requests <= total_requests),
    
    UNIQUE(tenant_id, metric_date)
);

-- Indexes for security tables
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_active ON api_keys(tenant_id, is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys(expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_security_audit_log_timestamp ON security_audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_security_audit_log_event_type ON security_audit_log(event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_security_audit_log_user ON security_audit_log(user_id, timestamp DESC) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_security_audit_log_tenant ON security_audit_log(tenant_id, timestamp DESC) WHERE tenant_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_security_audit_log_correlation ON security_audit_log(correlation_id) WHERE correlation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_rate_limit_buckets_key ON rate_limit_buckets(bucket_key);
CREATE INDEX IF NOT EXISTS idx_rate_limit_buckets_refill ON rate_limit_buckets(last_refill);

CREATE INDEX IF NOT EXISTS idx_content_security_scans_hash ON content_security_scans(content_hash);
CREATE INDEX IF NOT EXISTS idx_content_security_scans_timestamp ON content_security_scans(scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_content_security_scans_tenant ON content_security_scans(tenant_id, scan_timestamp DESC) WHERE tenant_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_content_security_scans_threats ON content_security_scans(is_safe, risk_score DESC);

CREATE INDEX IF NOT EXISTS idx_jwt_blacklist_jti ON jwt_token_blacklist(jti);
CREATE INDEX IF NOT EXISTS idx_jwt_blacklist_user ON jwt_token_blacklist(user_id, revoked_at DESC);
CREATE INDEX IF NOT EXISTS idx_jwt_blacklist_expires ON jwt_token_blacklist(expires_at);

CREATE INDEX IF NOT EXISTS idx_tenant_security_config_tenant ON tenant_security_config(tenant_id);

CREATE INDEX IF NOT EXISTS idx_security_metrics_tenant_date ON security_metrics(tenant_id, metric_date DESC);
CREATE INDEX IF NOT EXISTS idx_security_metrics_date ON security_metrics(metric_date DESC);

-- Row Level Security for security tables
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE rate_limit_buckets ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_security_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE jwt_token_blacklist ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_security_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_metrics ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY IF NOT EXISTS tenant_isolation_api_keys ON api_keys
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_security_audit_log ON security_audit_log
    FOR ALL
    USING (
        tenant_id = current_setting('app.current_tenant_id', true) OR
        tenant_id IS NULL OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- Rate limit buckets are shared but access controlled by application logic
CREATE POLICY IF NOT EXISTS rate_limit_buckets_access ON rate_limit_buckets
    FOR ALL
    USING (true);

CREATE POLICY IF NOT EXISTS tenant_isolation_content_security_scans ON content_security_scans
    FOR ALL
    USING (
        tenant_id = current_setting('app.current_tenant_id', true) OR
        tenant_id IS NULL
    );

CREATE POLICY IF NOT EXISTS tenant_isolation_jwt_blacklist ON jwt_token_blacklist
    FOR ALL
    USING (
        tenant_id = current_setting('app.current_tenant_id', true) OR
        tenant_id IS NULL OR
        current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY IF NOT EXISTS tenant_isolation_tenant_security_config ON tenant_security_config
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY IF NOT EXISTS tenant_isolation_security_metrics ON security_metrics
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Functions for security operations
CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM jwt_token_blacklist 
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security_audit_log 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_security_metrics()
RETURNS VOID AS $$
BEGIN
    INSERT INTO security_metrics (
        tenant_id,
        metric_date,
        total_auth_attempts,
        successful_auth,
        failed_auth,
        total_authz_checks,
        successful_authz,
        failed_authz,
        total_requests,
        rate_limited_requests,
        content_scans_performed,
        threats_detected,
        high_risk_content,
        active_api_keys,
        expired_api_keys
    )
    SELECT 
        COALESCE(sal.tenant_id, 'system') as tenant_id,
        CURRENT_DATE as metric_date,
        
        -- Authentication metrics
        COUNT(CASE WHEN sal.event_type IN ('authentication_success', 'authentication_failed') THEN 1 END) as total_auth_attempts,
        COUNT(CASE WHEN sal.event_type = 'authentication_success' THEN 1 END) as successful_auth,
        COUNT(CASE WHEN sal.event_type = 'authentication_failed' THEN 1 END) as failed_auth,
        
        -- Authorization metrics  
        COUNT(CASE WHEN sal.event_type IN ('authorization_success', 'authorization_failed') THEN 1 END) as total_authz_checks,
        COUNT(CASE WHEN sal.event_type = 'authorization_success' THEN 1 END) as successful_authz,
        COUNT(CASE WHEN sal.event_type = 'authorization_failed' THEN 1 END) as failed_authz,
        
        -- Rate limiting metrics
        COUNT(CASE WHEN sal.event_type = 'request_processed' THEN 1 END) as total_requests,
        COUNT(CASE WHEN sal.event_type = 'rate_limit_exceeded' THEN 1 END) as rate_limited_requests,
        
        -- Content security metrics
        COUNT(CASE WHEN sal.event_type = 'content_scanned' THEN 1 END) as content_scans_performed,
        COUNT(CASE WHEN sal.event_type = 'threat_detected' THEN 1 END) as threats_detected,
        COUNT(CASE WHEN sal.event_type = 'high_risk_content' THEN 1 END) as high_risk_content,
        
        -- API key metrics (from separate queries)
        0 as active_api_keys,
        0 as expired_api_keys
        
    FROM security_audit_log sal
    WHERE sal.timestamp >= CURRENT_DATE
      AND sal.timestamp < CURRENT_DATE + INTERVAL '1 day'
    GROUP BY COALESCE(sal.tenant_id, 'system')
    
    ON CONFLICT (tenant_id, metric_date) 
    DO UPDATE SET
        total_auth_attempts = EXCLUDED.total_auth_attempts,
        successful_auth = EXCLUDED.successful_auth,
        failed_auth = EXCLUDED.failed_auth,
        total_authz_checks = EXCLUDED.total_authz_checks,
        successful_authz = EXCLUDED.successful_authz,
        failed_authz = EXCLUDED.failed_authz,
        total_requests = EXCLUDED.total_requests,
        rate_limited_requests = EXCLUDED.rate_limited_requests,
        content_scans_performed = EXCLUDED.content_scans_performed,
        threats_detected = EXCLUDED.threats_detected,
        high_risk_content = EXCLUDED.high_risk_content,
        updated_at = NOW();
        
    -- Update API key metrics separately
    UPDATE security_metrics sm SET
        active_api_keys = (
            SELECT COUNT(*) FROM api_keys ak 
            WHERE ak.tenant_id = sm.tenant_id 
              AND ak.is_active = true
              AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
        ),
        expired_api_keys = (
            SELECT COUNT(*) FROM api_keys ak 
            WHERE ak.tenant_id = sm.tenant_id 
              AND (ak.is_active = false OR ak.expires_at <= NOW())
        ),
        updated_at = NOW()
    WHERE sm.metric_date = CURRENT_DATE;
END;
$$ LANGUAGE plpgsql;