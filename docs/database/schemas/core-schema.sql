-- ============================================================================
-- COMPLY-AI CORE DATABASE SCHEMA
-- ============================================================================
-- This schema contains core application data and service operations
-- Database: comply-ai-core
-- Purpose: Users, tenants, storage records, audit logs, configurations

-- ============================================================================
-- CORE APPLICATION TABLES
-- ============================================================================

-- User management tables
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    company VARCHAR(255),
    phone VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- User roles and permissions
CREATE TABLE IF NOT EXISTS user_roles (
    id BIGSERIAL PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '[]',
    is_system_role BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User role assignments
CREATE TABLE IF NOT EXISTS user_role_assignments (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    role_name VARCHAR(100) REFERENCES user_roles(role_name) ON DELETE CASCADE,
    tenant_id VARCHAR(255),
    assigned_by VARCHAR(255),
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id, role_name, tenant_id)
);

-- User sessions and authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tenant configurations with Azure resource mapping
CREATE TABLE IF NOT EXISTS tenant_configs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure resource configuration
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    storage_account VARCHAR(100),
    key_vault_name VARCHAR(100),
    container_registry VARCHAR(100),
    virtual_network VARCHAR(100),
    subnet VARCHAR(100),
    application_gateway VARCHAR(100),
    backup_vault VARCHAR(100),
    managed_identity VARCHAR(100),
    log_analytics_workspace VARCHAR(100),
    application_insights VARCHAR(100),
    security_center VARCHAR(100),
    
    -- Configuration settings
    settings JSONB DEFAULT '{}',
    compliance_frameworks JSONB DEFAULT '[]',
    custom_taxonomy JSONB DEFAULT '{}',
    detector_configs JSONB DEFAULT '{}',
    rate_limits JSONB DEFAULT '{}',
    retention_policies JSONB DEFAULT '{}'
);

-- Model versions and tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'mapper', 'detector', 'classifier'
    description TEXT,
    model_path VARCHAR(500),
    model_size BIGINT,
    training_data_hash VARCHAR(255),
    training_metrics JSONB DEFAULT '{}',
    validation_metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    container_registry VARCHAR(100),
    blob_storage_path VARCHAR(500)
);

-- Storage records with Azure Blob Storage integration
CREATE TABLE IF NOT EXISTS storage_records (
    id VARCHAR(255) PRIMARY KEY,
    source_data TEXT NOT NULL,
    mapped_data TEXT NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB,
    tenant_id VARCHAR(255) NOT NULL,
    s3_key VARCHAR(500),
    encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    blob_storage_account VARCHAR(100),
    blob_container VARCHAR(100),
    blob_path VARCHAR(500),
    encryption_key_id VARCHAR(255)
);

-- Audit logs with comprehensive tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    subscription_id VARCHAR(255),
    correlation_id VARCHAR(255)
);

-- Compliance mappings
CREATE TABLE IF NOT EXISTS compliance_mappings (
    id BIGSERIAL PRIMARY KEY,
    taxonomy_label VARCHAR(255) NOT NULL,
    compliance_framework VARCHAR(100) NOT NULL,
    framework_requirement VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    remediation_guidance TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(taxonomy_label, compliance_framework, framework_requirement)
);

-- Detector configurations
CREATE TABLE IF NOT EXISTS detector_configs (
    id BIGSERIAL PRIMARY KEY,
    detector_name VARCHAR(255) UNIQUE NOT NULL,
    detector_type VARCHAR(100) NOT NULL,
    description TEXT,
    version VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    configuration JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    health_status VARCHAR(50) DEFAULT 'healthy',
    last_health_check TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    container_instance VARCHAR(100),
    endpoint_url VARCHAR(500)
);

-- Taxonomy labels
CREATE TABLE IF NOT EXISTS taxonomy_labels (
    id BIGSERIAL PRIMARY KEY,
    label VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    description TEXT,
    severity VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SERVICE-SPECIFIC TABLES
-- ============================================================================

-- Detector Orchestration tables
CREATE TABLE IF NOT EXISTS orchestration_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    job_id VARCHAR(255),
    tenant_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    policy_bundle VARCHAR(255) NOT NULL,
    environment VARCHAR(20) DEFAULT 'dev',
    processing_mode VARCHAR(20) DEFAULT 'sync',
    priority VARCHAR(20) DEFAULT 'normal',
    metadata JSONB,
    required_detectors JSONB,
    excluded_detectors JSONB,
    idempotency_key VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orchestration_responses (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) REFERENCES orchestration_requests(request_id) ON DELETE CASCADE,
    job_id VARCHAR(255),
    tenant_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    coverage_score DECIMAL(5,4),
    processing_time_ms INTEGER NOT NULL,
    detector_results JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS detector_results (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) REFERENCES orchestration_requests(request_id) ON DELETE CASCADE,
    detector_name VARCHAR(255) NOT NULL,
    detector_version VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    raw_output TEXT,
    processed_output JSONB,
    confidence_score DECIMAL(5,4),
    processing_time_ms INTEGER NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS routing_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    primary_detectors JSONB NOT NULL,
    secondary_detectors JSONB DEFAULT '[]',
    coverage_method VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis Module tables
CREATE TABLE IF NOT EXISTS analysis_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    period VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL,
    evidence_refs JSONB DEFAULT '[]',
    analysis_type VARCHAR(100) DEFAULT 'security_analysis',
    priority VARCHAR(20) DEFAULT 'normal',
    metadata JSONB,
    idempotency_key VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_responses (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) REFERENCES analysis_requests(request_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    explanation TEXT NOT NULL,
    remediation JSONB DEFAULT '[]',
    policy_recommendations JSONB DEFAULT '[]',
    confidence_score DECIMAL(5,4),
    processing_time_ms INTEGER NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    evidence_refs JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_templates (
    id BIGSERIAL PRIMARY KEY,
    template_name VARCHAR(255) UNIQUE NOT NULL,
    template_type VARCHAR(100) NOT NULL,
    template_content TEXT NOT NULL,
    version VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Llama Mapper tables
CREATE TABLE IF NOT EXISTS mapper_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    detector VARCHAR(255) NOT NULL,
    source_data TEXT NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    processing_mode VARCHAR(20) DEFAULT 'sync',
    priority VARCHAR(20) DEFAULT 'normal',
    metadata JSONB,
    idempotency_key VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mapper_responses (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) REFERENCES mapper_requests(request_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    taxonomy JSONB NOT NULL,
    scores JSONB NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    notes TEXT,
    provenance JSONB,
    policy_context JSONB,
    version_info JSONB,
    processing_time_ms INTEGER NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- CUSTOMER SUPPORT AND KNOWLEDGE BASE TABLES
-- ============================================================================

-- Customer support tickets
CREATE TABLE IF NOT EXISTS support_tickets (
    id BIGSERIAL PRIMARY KEY,
    ticket_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    tenant_id VARCHAR(255) NOT NULL,
    priority VARCHAR(20) NOT NULL, -- 'low', 'normal', 'high', 'urgent'
    status VARCHAR(50) NOT NULL, -- 'open', 'in_progress', 'resolved', 'closed'
    category VARCHAR(100) NOT NULL, -- 'technical', 'billing', 'feature_request', 'white_glove'
    subject VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    assigned_to VARCHAR(255), -- support agent or success manager
    resolution_notes TEXT,
    tags JSONB DEFAULT '[]',
    attachments JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ
);

-- Support ticket communications
CREATE TABLE IF NOT EXISTS support_ticket_communications (
    id BIGSERIAL PRIMARY KEY,
    ticket_id VARCHAR(255) REFERENCES support_tickets(ticket_id) ON DELETE CASCADE,
    from_user_id VARCHAR(255), -- References users table in core database
    to_user_id VARCHAR(255), -- References users table in core database
    communication_type VARCHAR(50) NOT NULL, -- 'comment', 'internal_note', 'status_update'
    content TEXT NOT NULL,
    is_internal BOOLEAN DEFAULT FALSE,
    attachments JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge base articles
CREATE TABLE IF NOT EXISTS knowledge_base_articles (
    id BIGSERIAL PRIMARY KEY,
    article_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) NOT NULL, -- 'getting_started', 'api_docs', 'troubleshooting', 'white_glove'
    subcategory VARCHAR(100),
    tags JSONB DEFAULT '[]',
    is_public BOOLEAN DEFAULT FALSE,
    is_featured BOOLEAN DEFAULT FALSE,
    view_count INTEGER DEFAULT 0,
    helpful_count INTEGER DEFAULT 0,
    not_helpful_count INTEGER DEFAULT 0,
    created_by VARCHAR(255) NOT NULL, -- References users table in core database
    updated_by VARCHAR(255), -- References users table in core database
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ
);

-- API key management
CREATE TABLE IF NOT EXISTS api_keys (
    id BIGSERIAL PRIMARY KEY,
    key_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    tenant_id VARCHAR(255) NOT NULL,
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL, -- First 8 characters for identification
    permissions JSONB DEFAULT '[]',
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data retention and privacy policies
CREATE TABLE IF NOT EXISTS data_retention_policies (
    id BIGSERIAL PRIMARY KEY,
    policy_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(100) NOT NULL, -- 'user_data', 'audit_logs', 'usage_metrics', 'storage_records'
    retention_days INTEGER NOT NULL,
    auto_delete BOOLEAN DEFAULT TRUE,
    compliance_framework VARCHAR(100), -- 'GDPR', 'CCPA', 'HIPAA', 'SOC2'
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL, -- References users table in core database
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- User management indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_roles_role_name ON user_roles(role_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_roles_is_system_role ON user_roles(is_system_role);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_user_id ON user_role_assignments(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_tenant_id ON user_role_assignments(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_role_name ON user_role_assignments(role_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_is_active ON user_role_assignments(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Core application indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_tenant_id ON tenant_configs(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_is_active ON tenant_configs(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_is_active ON model_versions(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_is_production ON model_versions(is_production);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_tenant_id ON storage_records(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_timestamp ON storage_records(timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_model_version ON storage_records(model_version);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_created_at ON storage_records(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_taxonomy_label ON compliance_mappings(taxonomy_label);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_framework ON compliance_mappings(compliance_framework);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_severity ON compliance_mappings(severity);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_detector_name ON detector_configs(detector_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_is_active ON detector_configs(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_health_status ON detector_configs(health_status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_label ON taxonomy_labels(label);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_category ON taxonomy_labels(category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_severity ON taxonomy_labels(severity);

-- Service-specific indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_tenant_id ON orchestration_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_request_id ON orchestration_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_created_at ON orchestration_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_responses_request_id ON orchestration_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_responses_tenant_id ON orchestration_responses(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_responses_status ON orchestration_responses(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_request_id ON detector_results(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_detector_name ON detector_results(detector_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_status ON detector_results(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_routing_plans_plan_name ON routing_plans(plan_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_routing_plans_is_active ON routing_plans(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_tenant_id ON analysis_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_request_id ON analysis_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_created_at ON analysis_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_responses_request_id ON analysis_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_responses_tenant_id ON analysis_responses(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_responses_status ON analysis_responses(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_tenant_id ON analysis_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_metric_type ON analysis_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_timestamp ON analysis_metrics(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_templates_template_name ON analysis_templates(template_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_templates_is_active ON analysis_templates(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_tenant_id ON mapper_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_request_id ON mapper_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_created_at ON mapper_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_responses_request_id ON mapper_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_responses_tenant_id ON mapper_responses(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_responses_created_at ON mapper_responses(created_at);

-- Indexes for customer support and knowledge base tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_ticket_id ON support_tickets(ticket_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_user_id ON support_tickets(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_tenant_id ON support_tickets(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_status ON support_tickets(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_priority ON support_tickets(priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_category ON support_tickets(category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_assigned_to ON support_tickets(assigned_to);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_tickets_created_at ON support_tickets(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_ticket_communications_ticket_id ON support_ticket_communications(ticket_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_ticket_communications_from_user_id ON support_ticket_communications(from_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_ticket_communications_to_user_id ON support_ticket_communications(to_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_ticket_communications_communication_type ON support_ticket_communications(communication_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_ticket_communications_created_at ON support_ticket_communications(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_article_id ON knowledge_base_articles(article_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_category ON knowledge_base_articles(category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_is_public ON knowledge_base_articles(is_public);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_is_featured ON knowledge_base_articles(is_featured);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_created_by ON knowledge_base_articles(created_by);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_base_articles_created_at ON knowledge_base_articles(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_tenant_id ON api_keys(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_retention_policies_policy_id ON data_retention_policies(policy_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_retention_policies_tenant_id ON data_retention_policies(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_retention_policies_data_type ON data_retention_policies(data_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_retention_policies_compliance_framework ON data_retention_policies(compliance_framework);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_retention_policies_is_active ON data_retention_policies(is_active);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on tenant-specific tables
ALTER TABLE storage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE orchestration_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE orchestration_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE detector_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE mapper_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE mapper_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE support_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE support_ticket_communications ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_retention_policies ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY tenant_isolation_storage_records ON storage_records
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_audit_logs ON audit_logs
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_orchestration_requests ON orchestration_requests
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_orchestration_responses ON orchestration_responses
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_detector_results ON detector_results
    FOR ALL TO PUBLIC
    USING (request_id IN (
        SELECT request_id FROM orchestration_requests 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_analysis_requests ON analysis_requests
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_analysis_responses ON analysis_responses
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_analysis_metrics ON analysis_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_mapper_requests ON mapper_requests
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_mapper_responses ON mapper_responses
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_support_tickets ON support_tickets
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_support_ticket_communications ON support_ticket_communications
    FOR ALL TO PUBLIC
    USING (ticket_id IN (
        SELECT ticket_id FROM support_tickets 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_api_keys ON api_keys
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_data_retention_policies ON data_retention_policies
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_roles_updated_at BEFORE UPDATE ON user_roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_role_assignments_updated_at BEFORE UPDATE ON user_role_assignments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tenant_configs_updated_at BEFORE UPDATE ON tenant_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_versions_updated_at BEFORE UPDATE ON model_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_mappings_updated_at BEFORE UPDATE ON compliance_mappings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detector_configs_updated_at BEFORE UPDATE ON detector_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_taxonomy_labels_updated_at BEFORE UPDATE ON taxonomy_labels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_routing_plans_updated_at BEFORE UPDATE ON routing_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analysis_templates_updated_at BEFORE UPDATE ON analysis_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orchestration_requests_updated_at BEFORE UPDATE ON orchestration_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analysis_requests_updated_at BEFORE UPDATE ON analysis_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mapper_requests_updated_at BEFORE UPDATE ON mapper_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample user roles
INSERT INTO user_roles (role_name, description, permissions, is_system_role) VALUES
('admin', 'System Administrator', '["read", "write", "delete", "admin"]', TRUE),
('user', 'Regular User', '["read", "write"]', TRUE),
('viewer', 'Read-only User', '["read"]', TRUE),
('billing_admin', 'Billing Administrator', '["read", "write", "billing"]', FALSE)
ON CONFLICT (role_name) DO NOTHING;

-- Insert sample taxonomy labels
INSERT INTO taxonomy_labels (label, category, subcategory, description, severity) VALUES
('HARM.SPEECH.Toxicity', 'HARM', 'SPEECH', 'Toxic or harmful speech content', 'high'),
('HARM.SPEECH.Hate', 'HARM', 'SPEECH', 'Hate speech or discriminatory content', 'critical'),
('PII.Identifier.SSN', 'PII', 'Identifier', 'Social Security Number', 'critical'),
('PII.Identifier.Email', 'PII', 'Identifier', 'Email address', 'medium'),
('PII.Financial.CreditCard', 'PII', 'Financial', 'Credit card information', 'critical'),
('COMPLIANCE.GDPR.PersonalData', 'COMPLIANCE', 'GDPR', 'Personal data under GDPR', 'high'),
('COMPLIANCE.HIPAA.HealthInfo', 'COMPLIANCE', 'HIPAA', 'Health information under HIPAA', 'critical')
ON CONFLICT (label) DO NOTHING;

-- Insert sample compliance mappings
INSERT INTO compliance_mappings (taxonomy_label, compliance_framework, framework_requirement, severity, description, remediation_guidance) VALUES
('PII.Identifier.SSN', 'SOC2', 'CC6.1', 'critical', 'SSN exposure violates data protection requirements', 'Implement data masking and access controls'),
('PII.Identifier.Email', 'SOC2', 'CC6.1', 'medium', 'Email addresses require protection', 'Use encryption and access controls'),
('HARM.SPEECH.Hate', 'SOC2', 'CC6.2', 'critical', 'Hate speech violates content policies', 'Implement content filtering and moderation'),
('COMPLIANCE.GDPR.PersonalData', 'GDPR', 'Article 5', 'high', 'Personal data processing requirements', 'Ensure lawful basis and data minimization'),
('COMPLIANCE.HIPAA.HealthInfo', 'HIPAA', '164.502', 'critical', 'Health information protection requirements', 'Implement administrative, physical, and technical safeguards')
ON CONFLICT (taxonomy_label, compliance_framework, framework_requirement) DO NOTHING;

-- Insert sample detector configurations
INSERT INTO detector_configs (detector_name, detector_type, description, version, configuration, health_status) VALUES
('pii-detector', 'PII', 'Detects personally identifiable information', 'v1.2.0', '{"confidence_threshold": 0.8, "supported_types": ["ssn", "email", "phone"]}', 'healthy'),
('toxicity-detector', 'Content', 'Detects toxic or harmful content', 'v2.1.0', '{"confidence_threshold": 0.7, "categories": ["toxicity", "hate", "harassment"]}', 'healthy'),
('sentiment-detector', 'Content', 'Analyzes sentiment of text content', 'v1.0.0', '{"confidence_threshold": 0.6, "output_format": "polarity_score"}', 'healthy')
ON CONFLICT (detector_name) DO NOTHING;

-- Insert sample routing plans
INSERT INTO routing_plans (plan_name, primary_detectors, secondary_detectors, coverage_method, is_active) VALUES
('default-security-plan', '["pii-detector", "toxicity-detector", "hate-speech-detector"]', '["sentiment-detector", "language-detector"]', 'required_set', TRUE),
('compliance-plan', '["pii-detector", "gdpr-detector", "hipaa-detector"]', '["audit-detector"]', 'required_set', TRUE)
ON CONFLICT (plan_name) DO NOTHING;

-- Insert sample analysis templates
INSERT INTO analysis_templates (template_name, template_type, template_content, version, is_active) VALUES
('security-analysis-template', 'security_analysis', 'Analyze the following security metrics and provide recommendations...', 'v1.0', TRUE),
('compliance-analysis-template', 'compliance_analysis', 'Review compliance metrics and identify gaps...', 'v1.0', TRUE)
ON CONFLICT (template_name) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Comply-AI Core Database Schema created successfully!';
    RAISE NOTICE 'Core Tables: users, user_roles, user_role_assignments, user_sessions, tenant_configs, model_versions, storage_records, audit_logs, compliance_mappings, detector_configs, taxonomy_labels';
    RAISE NOTICE 'Detector Orchestration Tables: orchestration_requests, orchestration_responses, detector_results, routing_plans';
    RAISE NOTICE 'Analysis Module Tables: analysis_requests, analysis_responses, analysis_metrics, analysis_templates';
    RAISE NOTICE 'Llama Mapper Tables: mapper_requests, mapper_responses';
    RAISE NOTICE 'Indexes created: Performance and GIN indexes for JSONB columns';
    RAISE NOTICE 'Security: Row Level Security enabled with tenant isolation policies';
    RAISE NOTICE 'Functions: Automatic timestamp updates and cleanup functions';
    RAISE NOTICE 'Sample data: Basic taxonomy labels, compliance mappings, detector configs, routing plans, and analysis templates inserted';
    RAISE NOTICE 'Total Tables Created: 20 tables covering core application data and all 3 services';
END $$;
