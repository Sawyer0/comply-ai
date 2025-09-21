-- Azure Database Schema for Llama Mapper System
-- This script creates all necessary tables, indexes, and security policies
-- for the Azure-native implementation of the Llama Mapper system.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================================================
-- CORE STORAGE TABLES
-- ============================================================================

-- Primary storage table for mapping results
CREATE TABLE IF NOT EXISTS storage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_data TEXT NOT NULL,
    mapped_data JSONB NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    tenant_id VARCHAR(100) NOT NULL,
    s3_key VARCHAR(500),
    encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '90 days',
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    subscription_id VARCHAR(100),
    blob_url VARCHAR(500),
    container_name VARCHAR(100) DEFAULT 'mapper-outputs'
);

-- Performance indexes for storage_records
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_tenant_timestamp 
    ON storage_records(tenant_id, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_model_version 
    ON storage_records(model_version);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_expires_at 
    ON storage_records(expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_azure_region 
    ON storage_records(azure_region);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_created_at 
    ON storage_records(created_at);

-- GIN index for JSONB metadata
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_metadata_gin 
    ON storage_records USING GIN (metadata);

-- GIN index for JSONB mapped_data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_mapped_data_gin 
    ON storage_records USING GIN (mapped_data);

-- ============================================================================
-- AUDIT AND COMPLIANCE TABLES
-- ============================================================================

-- Comprehensive audit logging table
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
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    subscription_id VARCHAR(100),
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes for audit_logs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_tenant_created 
    ON audit_logs(tenant_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_action 
    ON audit_logs(action);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_resource_type 
    ON audit_logs(resource_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_azure_region 
    ON audit_logs(azure_region);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_created_at 
    ON audit_logs(created_at);

-- GIN index for JSONB details
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_details_gin 
    ON audit_logs USING GIN (details);

-- Compliance framework mappings
CREATE TABLE IF NOT EXISTS compliance_mappings (
    id BIGSERIAL PRIMARY KEY,
    taxonomy_label VARCHAR(255) NOT NULL,
    framework_name VARCHAR(100) NOT NULL,
    control_id VARCHAR(100) NOT NULL,
    control_description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    
    -- Unique constraint
    UNIQUE(taxonomy_label, framework_name, control_id)
);

-- Indexes for compliance_mappings
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_taxonomy 
    ON compliance_mappings(taxonomy_label);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_framework 
    ON compliance_mappings(framework_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_mappings_control 
    ON compliance_mappings(control_id);

-- ============================================================================
-- AZURE INFRASTRUCTURE TRACKING TABLES
-- ============================================================================

-- Azure Container Registry tracking
CREATE TABLE IF NOT EXISTS azure_container_registries (
    id BIGSERIAL PRIMARY KEY,
    registry_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    login_server VARCHAR(255) NOT NULL,
    sku VARCHAR(50) DEFAULT 'Basic',
    admin_enabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Virtual Network tracking
CREATE TABLE IF NOT EXISTS azure_virtual_networks (
    id BIGSERIAL PRIMARY KEY,
    vnet_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    address_space VARCHAR(255) NOT NULL,
    dns_servers JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Subnets tracking
CREATE TABLE IF NOT EXISTS azure_subnets (
    id BIGSERIAL PRIMARY KEY,
    subnet_name VARCHAR(255) NOT NULL,
    vnet_id BIGINT REFERENCES azure_virtual_networks(id) ON DELETE CASCADE,
    address_prefix VARCHAR(255) NOT NULL,
    network_security_group_id BIGINT,
    route_table_id BIGINT,
    service_endpoints JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(subnet_name, vnet_id)
);

-- Azure Network Security Groups tracking
CREATE TABLE IF NOT EXISTS azure_network_security_groups (
    id BIGSERIAL PRIMARY KEY,
    nsg_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    security_rules JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Application Gateway tracking
CREATE TABLE IF NOT EXISTS azure_application_gateways (
    id BIGSERIAL PRIMARY KEY,
    gateway_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    sku VARCHAR(50) DEFAULT 'Standard_v2',
    capacity INTEGER DEFAULT 2,
    public_ip_address VARCHAR(255),
    backend_pools JSONB,
    http_listeners JSONB,
    routing_rules JSONB,
    ssl_certificates JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Backup Vault tracking
CREATE TABLE IF NOT EXISTS azure_backup_vaults (
    id BIGSERIAL PRIMARY KEY,
    vault_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    storage_type VARCHAR(50) DEFAULT 'GeoRedundant',
    backup_policies JSONB,
    protected_items JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Managed Identities tracking
CREATE TABLE IF NOT EXISTS azure_managed_identities (
    id BIGSERIAL PRIMARY KEY,
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    principal_id VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    assigned_resources JSONB,
    role_assignments JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Log Analytics Workspaces tracking
CREATE TABLE IF NOT EXISTS azure_log_analytics_workspaces (
    id BIGSERIAL PRIMARY KEY,
    workspace_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    workspace_id VARCHAR(255) UNIQUE NOT NULL,
    retention_days INTEGER DEFAULT 30,
    sku VARCHAR(50) DEFAULT 'PerGB2018',
    daily_quota_gb DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Application Insights tracking
CREATE TABLE IF NOT EXISTS azure_application_insights (
    id BIGSERIAL PRIMARY KEY,
    app_name VARCHAR(255) UNIQUE NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    azure_region VARCHAR(50) NOT NULL,
    subscription_id VARCHAR(100) NOT NULL,
    app_id VARCHAR(255) UNIQUE NOT NULL,
    instrumentation_key VARCHAR(255) UNIQUE NOT NULL,
    connection_string VARCHAR(500),
    workspace_id BIGINT REFERENCES azure_log_analytics_workspaces(id),
    sampling_percentage DECIMAL(5,2) DEFAULT 100.00,
    retention_days INTEGER DEFAULT 90,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Security Center tracking
CREATE TABLE IF NOT EXISTS azure_security_center (
    id BIGSERIAL PRIMARY KEY,
    subscription_id VARCHAR(100) UNIQUE NOT NULL,
    pricing_tier VARCHAR(50) DEFAULT 'Free',
    auto_provisioning BOOLEAN DEFAULT FALSE,
    security_contacts JSONB,
    security_alerts JSONB,
    compliance_assessments JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SERVICE-SPECIFIC TABLES
-- ============================================================================

-- Detector Orchestration Service Tables
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
    request_id VARCHAR(255) REFERENCES orchestration_requests(request_id),
    job_id VARCHAR(255),
    processing_mode VARCHAR(20) NOT NULL,
    detector_results JSONB NOT NULL,
    aggregated_payload JSONB,
    mapping_result JSONB,
    total_processing_time_ms INTEGER NOT NULL,
    detectors_attempted INTEGER NOT NULL,
    detectors_succeeded INTEGER NOT NULL,
    detectors_failed INTEGER NOT NULL,
    coverage_achieved DECIMAL(5,4) NOT NULL,
    routing_decision JSONB,
    fallback_used BOOLEAN DEFAULT FALSE,
    error_code VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS detector_results (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) REFERENCES orchestration_requests(request_id),
    detector VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    output TEXT,
    metadata JSONB,
    error TEXT,
    processing_time_ms INTEGER DEFAULT 0,
    confidence DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS routing_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_name VARCHAR(255) UNIQUE NOT NULL,
    primary_detectors JSONB NOT NULL,
    secondary_detectors JSONB DEFAULT '[]',
    parallel_groups JSONB DEFAULT '[]',
    sequential_dependencies JSONB DEFAULT '{}',
    timeout_config JSONB DEFAULT '{}',
    retry_config JSONB DEFAULT '{}',
    coverage_method VARCHAR(100) DEFAULT 'required_set',
    weights JSONB DEFAULT '{}',
    required_taxonomy_categories JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis Module Service Tables
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
    request_id VARCHAR(255) REFERENCES analysis_requests(request_id),
    explanation TEXT NOT NULL,
    remediation_steps JSONB NOT NULL,
    policy_recommendations JSONB NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    analysis_metadata JSONB,
    version_info JSONB,
    processing_time_ms INTEGER NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    template_fallback_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50),
    time_period VARCHAR(100) NOT NULL,
    evidence_refs JSONB DEFAULT '[]',
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_templates (
    id BIGSERIAL PRIMARY KEY,
    template_name VARCHAR(255) UNIQUE NOT NULL,
    template_type VARCHAR(100) NOT NULL,
    template_content TEXT NOT NULL,
    version VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Llama Mapper Service Tables (additional to existing storage_records)
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
    request_id VARCHAR(255) REFERENCES mapper_requests(request_id),
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
-- USER MANAGEMENT AND BILLING TABLES
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

-- Billing and subscription management
CREATE TABLE IF NOT EXISTS billing_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_id VARCHAR(255) UNIQUE NOT NULL,
    plan_name VARCHAR(255) NOT NULL,
    plan_type VARCHAR(50) NOT NULL, -- 'free', 'trial', 'paid', 'enterprise'
    description TEXT,
    price_monthly DECIMAL(10,2) DEFAULT 0.00,
    price_yearly DECIMAL(10,2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    features JSONB DEFAULT '{}',
    limits JSONB DEFAULT '{}',
    stripe_price_id VARCHAR(255),
    stripe_product_id VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User subscriptions
CREATE TABLE IF NOT EXISTS user_subscriptions (
    id BIGSERIAL PRIMARY KEY,
    subscription_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    plan_id VARCHAR(255) REFERENCES billing_plans(plan_id),
    status VARCHAR(50) NOT NULL, -- 'active', 'canceled', 'past_due', 'unpaid', 'trialing'
    billing_cycle VARCHAR(20) DEFAULT 'monthly', -- 'monthly', 'yearly'
    current_period_start TIMESTAMPTZ NOT NULL,
    current_period_end TIMESTAMPTZ NOT NULL,
    trial_start TIMESTAMPTZ,
    trial_end TIMESTAMPTZ,
    canceled_at TIMESTAMPTZ,
    stripe_subscription_id VARCHAR(255),
    stripe_customer_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Usage tracking for billing
CREATE TABLE IF NOT EXISTS usage_records (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id VARCHAR(255) REFERENCES user_subscriptions(subscription_id),
    usage_type VARCHAR(100) NOT NULL, -- 'api_calls', 'storage_gb', 'detector_runs', 'analysis_runs'
    usage_amount DECIMAL(15,4) NOT NULL,
    usage_unit VARCHAR(50) NOT NULL,
    billing_period_start TIMESTAMPTZ NOT NULL,
    billing_period_end TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- Billing invoices and payments
CREATE TABLE IF NOT EXISTS billing_invoices (
    id BIGSERIAL PRIMARY KEY,
    invoice_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id VARCHAR(255) REFERENCES user_subscriptions(subscription_id),
    invoice_number VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'draft', 'open', 'paid', 'void', 'uncollectible'
    amount_due DECIMAL(10,2) NOT NULL,
    amount_paid DECIMAL(10,2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    invoice_date TIMESTAMPTZ NOT NULL,
    due_date TIMESTAMPTZ,
    paid_at TIMESTAMPTZ,
    stripe_invoice_id VARCHAR(255),
    stripe_payment_intent_id VARCHAR(255),
    line_items JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Payment methods
CREATE TABLE IF NOT EXISTS payment_methods (
    id BIGSERIAL PRIMARY KEY,
    payment_method_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'card', 'bank_account', 'paypal'
    is_default BOOLEAN DEFAULT FALSE,
    stripe_payment_method_id VARCHAR(255),
    card_last_four VARCHAR(4),
    card_brand VARCHAR(50),
    card_exp_month INTEGER,
    card_exp_year INTEGER,
    billing_address JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Free tier limits and tracking
CREATE TABLE IF NOT EXISTS free_tier_usage (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    usage_type VARCHAR(100) NOT NULL,
    current_usage DECIMAL(15,4) DEFAULT 0.00,
    usage_limit DECIMAL(15,4) NOT NULL,
    reset_period VARCHAR(20) DEFAULT 'monthly', -- 'daily', 'weekly', 'monthly', 'yearly'
    last_reset_at TIMESTAMPTZ DEFAULT NOW(),
    next_reset_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, tenant_id, usage_type)
);

-- Promotional codes and discounts
CREATE TABLE IF NOT EXISTS promotional_codes (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    discount_type VARCHAR(50) NOT NULL, -- 'percentage', 'fixed_amount', 'free_trial'
    discount_value DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    max_uses INTEGER,
    current_uses INTEGER DEFAULT 0,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ,
    applicable_plans JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Promotional code usage tracking
CREATE TABLE IF NOT EXISTS promotional_code_usage (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(100) REFERENCES promotional_codes(code) ON DELETE CASCADE,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    subscription_id VARCHAR(255) REFERENCES user_subscriptions(subscription_id),
    used_at TIMESTAMPTZ DEFAULT NOW(),
    discount_applied DECIMAL(10,2) NOT NULL
);

-- White-glove service management
CREATE TABLE IF NOT EXISTS white_glove_services (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id VARCHAR(255) REFERENCES user_subscriptions(subscription_id),
    service_type VARCHAR(100) NOT NULL, -- 'implementation', 'integration', 'customization', 'support'
    status VARCHAR(50) NOT NULL, -- 'requested', 'in_progress', 'completed', 'cancelled'
    priority VARCHAR(20) DEFAULT 'normal', -- 'low', 'normal', 'high', 'urgent'
    description TEXT NOT NULL,
    requirements JSONB DEFAULT '{}',
    deliverables JSONB DEFAULT '[]',
    estimated_hours INTEGER,
    actual_hours INTEGER,
    estimated_cost DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    assigned_to VARCHAR(255), -- success manager or engineer
    start_date TIMESTAMPTZ,
    target_completion_date TIMESTAMPTZ,
    actual_completion_date TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- White-glove service milestones
CREATE TABLE IF NOT EXISTS white_glove_milestones (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) REFERENCES white_glove_services(service_id) ON DELETE CASCADE,
    milestone_name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL, -- 'pending', 'in_progress', 'completed', 'blocked'
    priority INTEGER DEFAULT 1,
    estimated_hours INTEGER,
    actual_hours INTEGER,
    start_date TIMESTAMPTZ,
    target_date TIMESTAMPTZ,
    completed_date TIMESTAMPTZ,
    assigned_to VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- White-glove service communications
CREATE TABLE IF NOT EXISTS white_glove_communications (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) REFERENCES white_glove_services(service_id) ON DELETE CASCADE,
    communication_type VARCHAR(50) NOT NULL, -- 'email', 'call', 'meeting', 'documentation', 'status_update'
    subject VARCHAR(255),
    content TEXT NOT NULL,
    from_user_id VARCHAR(255) REFERENCES users(user_id),
    to_user_id VARCHAR(255) REFERENCES users(user_id),
    is_internal BOOLEAN DEFAULT FALSE,
    attachments JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- White-glove service deliverables
CREATE TABLE IF NOT EXISTS white_glove_deliverables (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) REFERENCES white_glove_services(service_id) ON DELETE CASCADE,
    deliverable_name VARCHAR(255) NOT NULL,
    deliverable_type VARCHAR(100) NOT NULL, -- 'documentation', 'code', 'configuration', 'training', 'integration'
    description TEXT,
    status VARCHAR(50) NOT NULL, -- 'pending', 'in_progress', 'completed', 'delivered'
    file_path VARCHAR(500),
    file_size BIGINT,
    mime_type VARCHAR(100),
    version VARCHAR(50) DEFAULT '1.0',
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- White-glove service feedback and satisfaction
CREATE TABLE IF NOT EXISTS white_glove_feedback (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) REFERENCES white_glove_services(service_id) ON DELETE CASCADE,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    categories JSONB DEFAULT '{}', -- 'communication', 'technical_quality', 'timeliness', 'value'
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- CONFIGURATION AND METADATA TABLES
-- ============================================================================

-- Tenant configurations with Azure resource mapping
CREATE TABLE IF NOT EXISTS tenant_configs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    config_data JSONB NOT NULL,
    
    -- Azure-specific fields
    azure_subscription_id VARCHAR(100),
    azure_resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    azure_region VARCHAR(50) DEFAULT 'eastus',
    azure_key_vault_url VARCHAR(500),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for tenant_configs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_tenant_id 
    ON tenant_configs(tenant_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_azure_subscription 
    ON tenant_configs(azure_subscription_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_azure_region 
    ON tenant_configs(azure_region);

-- GIN index for JSONB config_data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_config_data_gin 
    ON tenant_configs USING GIN (config_data);

-- Model version tracking with Azure Blob Storage integration
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(255) NOT NULL,
    model_path TEXT NOT NULL,
    
    -- Azure-specific fields
    azure_blob_url VARCHAR(500),
    azure_container_name VARCHAR(100) DEFAULT 'model-artifacts',
    azure_region VARCHAR(50) DEFAULT 'eastus',
    
    checksum VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ
);

-- Indexes for model_versions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_model_name 
    ON model_versions(model_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_version 
    ON model_versions(version);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_is_active 
    ON model_versions(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_azure_region 
    ON model_versions(azure_region);

-- Detector configurations
CREATE TABLE IF NOT EXISTS detector_configs (
    id BIGSERIAL PRIMARY KEY,
    detector_name VARCHAR(255) UNIQUE NOT NULL,
    version VARCHAR(255) NOT NULL,
    mapping_config JSONB NOT NULL,
    notes TEXT,
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for detector_configs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_name 
    ON detector_configs(detector_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_version 
    ON detector_configs(version);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_azure_region 
    ON detector_configs(azure_region);

-- GIN index for JSONB mapping_config
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_configs_mapping_config_gin 
    ON detector_configs USING GIN (mapping_config);

-- Taxonomy labels
CREATE TABLE IF NOT EXISTS taxonomy_labels (
    id BIGSERIAL PRIMARY KEY,
    label_name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    aliases JSONB,
    version VARCHAR(50) NOT NULL,
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for taxonomy_labels
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_name 
    ON taxonomy_labels(label_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_category 
    ON taxonomy_labels(category);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_version 
    ON taxonomy_labels(version);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_azure_region 
    ON taxonomy_labels(azure_region);

-- GIN index for JSONB aliases
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_taxonomy_labels_aliases_gin 
    ON taxonomy_labels USING GIN (aliases);

-- ============================================================================
-- SECURITY AND ROW LEVEL SECURITY
-- ============================================================================

-- Enable Row Level Security on storage_records
ALTER TABLE storage_records ENABLE ROW LEVEL SECURITY;

-- Policy to ensure users can only see their own tenant's data
CREATE POLICY IF NOT EXISTS tenant_isolation_policy ON storage_records
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Enable Row Level Security on audit_logs
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Policy for audit logs (admin access or own tenant)
CREATE POLICY IF NOT EXISTS audit_tenant_policy ON audit_logs
    FOR ALL
    USING (
        tenant_id = current_setting('app.current_tenant_id', true) OR
        current_setting('app.user_role', true) = 'admin'
    );

-- Enable Row Level Security on tenant_configs
ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;

-- Policy for tenant configs (own tenant only)
CREATE POLICY IF NOT EXISTS tenant_config_policy ON tenant_configs
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- ============================================================================
-- PARTITIONING FOR LARGE-SCALE DEPLOYMENTS
-- ============================================================================

-- Partition storage_records by tenant_id for large-scale multi-tenancy
CREATE TABLE IF NOT EXISTS storage_records_partitioned (
    LIKE storage_records INCLUDING ALL
) PARTITION BY HASH (tenant_id);

-- Create partitions (example for 4 partitions)
CREATE TABLE IF NOT EXISTS storage_records_p0 PARTITION OF storage_records_partitioned
    FOR VALUES WITH (modulus 4, remainder 0);

CREATE TABLE IF NOT EXISTS storage_records_p1 PARTITION OF storage_records_partitioned
    FOR VALUES WITH (modulus 4, remainder 1);

CREATE TABLE IF NOT EXISTS storage_records_p2 PARTITION OF storage_records_partitioned
    FOR VALUES WITH (modulus 4, remainder 2);

CREATE TABLE IF NOT EXISTS storage_records_p3 PARTITION OF storage_records_partitioned
    FOR VALUES WITH (modulus 4, remainder 3);

-- Partition audit_logs by month for time-based partitioning
CREATE TABLE IF NOT EXISTS audit_logs_partitioned (
    LIKE audit_logs INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions (example for 2024)
CREATE TABLE IF NOT EXISTS audit_logs_2024_01 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_02 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_03 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_04 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_05 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_06 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_07 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_08 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_09 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_10 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_11 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_12 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- Indexes for user management and billing tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email 
    ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_user_id 
    ON users(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_is_active 
    ON users(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at 
    ON users(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_roles_role_name 
    ON user_roles(role_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_roles_is_system_role 
    ON user_roles(is_system_role);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_user_id 
    ON user_role_assignments(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_tenant_id 
    ON user_role_assignments(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_role_name 
    ON user_role_assignments(role_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_role_assignments_is_active 
    ON user_role_assignments(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_session_id 
    ON user_sessions(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id 
    ON user_sessions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_at 
    ON user_sessions(expires_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_plan_id 
    ON billing_plans(plan_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_plan_type 
    ON billing_plans(plan_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_is_active 
    ON billing_plans(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_stripe_price_id 
    ON billing_plans(stripe_price_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_subscription_id 
    ON user_subscriptions(subscription_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_user_id 
    ON user_subscriptions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_tenant_id 
    ON user_subscriptions(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_status 
    ON user_subscriptions(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_stripe_subscription_id 
    ON user_subscriptions(stripe_subscription_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_current_period_end 
    ON user_subscriptions(current_period_end);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_user_id 
    ON usage_records(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_tenant_id 
    ON usage_records(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_usage_type 
    ON usage_records(usage_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_billing_period_start 
    ON usage_records(billing_period_start);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_billing_period_end 
    ON usage_records(billing_period_end);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_invoice_id 
    ON billing_invoices(invoice_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_user_id 
    ON billing_invoices(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_tenant_id 
    ON billing_invoices(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_status 
    ON billing_invoices(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_due_date 
    ON billing_invoices(due_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_stripe_invoice_id 
    ON billing_invoices(stripe_invoice_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_payment_method_id 
    ON payment_methods(payment_method_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_user_id 
    ON payment_methods(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_tenant_id 
    ON payment_methods(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_is_default 
    ON payment_methods(is_default);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_stripe_payment_method_id 
    ON payment_methods(stripe_payment_method_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_user_id 
    ON free_tier_usage(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_tenant_id 
    ON free_tier_usage(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_usage_type 
    ON free_tier_usage(usage_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_next_reset_at 
    ON free_tier_usage(next_reset_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_code 
    ON promotional_codes(code);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_is_active 
    ON promotional_codes(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_valid_from 
    ON promotional_codes(valid_from);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_valid_until 
    ON promotional_codes(valid_until);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_code 
    ON promotional_code_usage(code);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_user_id 
    ON promotional_code_usage(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_used_at 
    ON promotional_code_usage(used_at);

-- Indexes for white-glove service tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_service_id 
    ON white_glove_services(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_user_id 
    ON white_glove_services(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_tenant_id 
    ON white_glove_services(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_status 
    ON white_glove_services(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_service_type 
    ON white_glove_services(service_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_assigned_to 
    ON white_glove_services(assigned_to);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_created_at 
    ON white_glove_services(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_service_id 
    ON white_glove_milestones(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_status 
    ON white_glove_milestones(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_priority 
    ON white_glove_milestones(priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_assigned_to 
    ON white_glove_milestones(assigned_to);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_target_date 
    ON white_glove_milestones(target_date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_service_id 
    ON white_glove_communications(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_communication_type 
    ON white_glove_communications(communication_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_from_user_id 
    ON white_glove_communications(from_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_to_user_id 
    ON white_glove_communications(to_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_created_at 
    ON white_glove_communications(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_service_id 
    ON white_glove_deliverables(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_deliverable_type 
    ON white_glove_deliverables(deliverable_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_status 
    ON white_glove_deliverables(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_delivered_at 
    ON white_glove_deliverables(delivered_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_service_id 
    ON white_glove_feedback(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_user_id 
    ON white_glove_feedback(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_rating 
    ON white_glove_feedback(rating);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_created_at 
    ON white_glove_feedback(created_at);

-- Indexes for service-specific tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_tenant_id 
    ON orchestration_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_request_id 
    ON orchestration_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_requests_created_at 
    ON orchestration_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_responses_request_id 
    ON orchestration_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orchestration_responses_created_at 
    ON orchestration_responses(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_request_id 
    ON detector_results(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_detector 
    ON detector_results(detector);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_results_status 
    ON detector_results(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_routing_plans_name 
    ON routing_plans(plan_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_routing_plans_is_active 
    ON routing_plans(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_tenant_id 
    ON analysis_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_request_id 
    ON analysis_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_requests_created_at 
    ON analysis_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_responses_request_id 
    ON analysis_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_responses_created_at 
    ON analysis_responses(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_tenant_id 
    ON analysis_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_metric_name 
    ON analysis_metrics(metric_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_metrics_time_period 
    ON analysis_metrics(time_period);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_templates_name 
    ON analysis_templates(template_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_templates_type 
    ON analysis_templates(template_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analysis_templates_is_active 
    ON analysis_templates(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_tenant_id 
    ON mapper_requests(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_request_id 
    ON mapper_requests(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_detector 
    ON mapper_requests(detector);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_requests_created_at 
    ON mapper_requests(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_responses_request_id 
    ON mapper_responses(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mapper_responses_created_at 
    ON mapper_responses(created_at);

-- Indexes for Azure infrastructure tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_container_registries_name 
    ON azure_container_registries(registry_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_virtual_networks_name 
    ON azure_virtual_networks(vnet_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_subnets_vnet_id 
    ON azure_subnets(vnet_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_network_security_groups_name 
    ON azure_network_security_groups(nsg_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_gateways_name 
    ON azure_application_gateways(gateway_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_backup_vaults_name 
    ON azure_backup_vaults(vault_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_managed_identities_name 
    ON azure_managed_identities(identity_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_log_analytics_workspaces_name 
    ON azure_log_analytics_workspaces(workspace_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_insights_name 
    ON azure_application_insights(app_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_security_center_subscription 
    ON azure_security_center(subscription_id);

-- ============================================================================
-- PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Azure Database for PostgreSQL performance optimizations
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- Optimize for Azure workloads
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Enable connection pooling
ALTER SYSTEM SET max_connections = 200;

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_tenant_configs_updated_at 
    BEFORE UPDATE ON tenant_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detector_configs_updated_at 
    BEFORE UPDATE ON detector_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_taxonomy_labels_updated_at 
    BEFORE UPDATE ON taxonomy_labels 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_mappings_updated_at 
    BEFORE UPDATE ON compliance_mappings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired records
CREATE OR REPLACE FUNCTION cleanup_expired_records()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM storage_records WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for active model versions
CREATE OR REPLACE VIEW active_model_versions AS
SELECT 
    model_name,
    version,
    model_path,
    azure_blob_url,
    azure_region,
    created_at,
    activated_at
FROM model_versions
WHERE is_active = TRUE
ORDER BY activated_at DESC;

-- View for tenant storage statistics
CREATE OR REPLACE VIEW tenant_storage_stats AS
SELECT 
    tenant_id,
    azure_region,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours') as records_last_24h,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as records_last_7d,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as records_last_30d,
    MIN(created_at) as first_record,
    MAX(created_at) as last_record
FROM storage_records
GROUP BY tenant_id, azure_region
ORDER BY total_records DESC;

-- View for audit log summary
CREATE OR REPLACE VIEW audit_log_summary AS
SELECT 
    tenant_id,
    action,
    resource_type,
    azure_region,
    DATE(created_at) as audit_date,
    COUNT(*) as event_count
FROM audit_logs
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY tenant_id, action, resource_type, azure_region, DATE(created_at)
ORDER BY audit_date DESC, event_count DESC;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Insert sample taxonomy labels
INSERT INTO taxonomy_labels (label_name, description, category, subcategory, aliases, version) VALUES
('PII.Identifier.SSN', 'Social Security Number', 'PII', 'Identifier', '["ssn", "social_security"]', '2024.01'),
('PII.Identifier.Email', 'Email Address', 'PII', 'Identifier', '["email", "email_address"]', '2024.01'),
('HARM.SPEECH.Toxicity', 'Toxic Speech Content', 'HARM', 'SPEECH', '["toxic", "toxicity"]', '2024.01'),
('HARM.SPEECH.Hate', 'Hate Speech Content', 'HARM', 'SPEECH', '["hate", "hate_speech"]', '2024.01')
ON CONFLICT (label_name) DO NOTHING;

-- Insert sample compliance mappings
INSERT INTO compliance_mappings (taxonomy_label, framework_name, control_id, control_description) VALUES
('PII.Identifier.SSN', 'SOC2', 'CC6.1', 'Logical and Physical Access Controls'),
('PII.Identifier.Email', 'SOC2', 'CC6.1', 'Logical and Physical Access Controls'),
('HARM.SPEECH.Toxicity', 'SOC2', 'CC7.2', 'System Communications and Data'),
('HARM.SPEECH.Hate', 'SOC2', 'CC7.2', 'System Communications and Data')
ON CONFLICT (taxonomy_label, framework_name, control_id) DO NOTHING;

-- Insert sample model version
INSERT INTO model_versions (model_name, version, model_path, is_active, azure_blob_url) VALUES
('llama-3-8b-instruct', 'v1.0', '/models/llama-3-8b-instruct-v1.0', TRUE, 'https://complyaistorage.blob.core.windows.net/model-artifacts/llama-3-8b-instruct/v1.0/model.bin')
ON CONFLICT DO NOTHING;

-- Insert sample Azure infrastructure data
INSERT INTO azure_container_registries (registry_name, resource_group, azure_region, subscription_id, login_server, sku, admin_enabled) VALUES
('complyaiacr', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'complyaiacr.azurecr.io', 'Basic', TRUE)
ON CONFLICT (registry_name) DO NOTHING;

INSERT INTO azure_virtual_networks (vnet_name, resource_group, azure_region, subscription_id, address_space, dns_servers) VALUES
('comply-ai-vnet', 'comply-ai-rg', 'eastus', 'your-subscription-id', '10.0.0.0/16', '["168.63.129.16"]')
ON CONFLICT (vnet_name) DO NOTHING;

INSERT INTO azure_network_security_groups (nsg_name, resource_group, azure_region, subscription_id, security_rules) VALUES
('comply-ai-nsg', 'comply-ai-rg', 'eastus', 'your-subscription-id', 
 '{"inbound_rules": [{"name": "AllowHTTPS", "priority": 100, "source": "Internet", "destination": "443", "access": "Allow"}], "outbound_rules": []}')
ON CONFLICT (nsg_name) DO NOTHING;

INSERT INTO azure_application_gateways (gateway_name, resource_group, azure_region, subscription_id, sku, capacity, public_ip_address) VALUES
('comply-ai-gateway', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'Standard_v2', 2, 'comply-ai-public-ip')
ON CONFLICT (gateway_name) DO NOTHING;

INSERT INTO azure_backup_vaults (vault_name, resource_group, azure_region, subscription_id, storage_type) VALUES
('comply-ai-backup-vault', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'GeoRedundant')
ON CONFLICT (vault_name) DO NOTHING;

INSERT INTO azure_managed_identities (identity_name, resource_group, azure_region, subscription_id, principal_id, client_id, tenant_id) VALUES
('comply-ai-identity', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'your-principal-id', 'your-client-id', 'your-tenant-id')
ON CONFLICT (identity_name) DO NOTHING;

INSERT INTO azure_log_analytics_workspaces (workspace_name, resource_group, azure_region, subscription_id, workspace_id, retention_days, sku) VALUES
('comply-ai-logs', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'your-workspace-id', 30, 'PerGB2018')
ON CONFLICT (workspace_name) DO NOTHING;

INSERT INTO azure_application_insights (app_name, resource_group, azure_region, subscription_id, app_id, instrumentation_key, sampling_percentage, retention_days) VALUES
('comply-ai-insights', 'comply-ai-rg', 'eastus', 'your-subscription-id', 'your-app-id', 'your-instrumentation-key', 100.00, 90)
ON CONFLICT (app_name) DO NOTHING;

INSERT INTO azure_security_center (subscription_id, pricing_tier, auto_provisioning) VALUES
('your-subscription-id', 'Standard', TRUE)
ON CONFLICT (subscription_id) DO NOTHING;

-- Insert sample service-specific data
INSERT INTO routing_plans (plan_name, primary_detectors, secondary_detectors, coverage_method, is_active) VALUES
('default-security-plan', '["pii-detector", "toxicity-detector", "hate-speech-detector"]', '["sentiment-detector", "language-detector"]', 'required_set', TRUE),
('compliance-plan', '["pii-detector", "gdpr-detector", "hipaa-detector"]', '["audit-detector"]', 'required_set', TRUE)
ON CONFLICT (plan_name) DO NOTHING;

INSERT INTO analysis_templates (template_name, template_type, template_content, version, is_active) VALUES
('security-analysis-template', 'security_analysis', 'Analyze the following security metrics and provide recommendations...', 'v1.0', TRUE),
('compliance-analysis-template', 'compliance_analysis', 'Review compliance metrics and identify gaps...', 'v1.0', TRUE)
ON CONFLICT (template_name) DO NOTHING;

-- Insert sample user management and billing data
INSERT INTO user_roles (role_name, description, permissions, is_system_role) VALUES
('admin', 'System Administrator', '["read", "write", "delete", "admin"]', TRUE),
('user', 'Regular User', '["read", "write"]', TRUE),
('viewer', 'Read-only User', '["read"]', TRUE),
('billing_admin', 'Billing Administrator', '["read", "write", "billing"]', FALSE)
ON CONFLICT (role_name) DO NOTHING;

INSERT INTO billing_plans (plan_id, plan_name, plan_type, description, price_monthly, price_yearly, currency, features, limits, stripe_price_id, stripe_product_id) VALUES
('free', 'Free Tier', 'free', 'Free tier with basic features and usage limits', 0.00, 0.00, 'USD', 
 '{"api_calls": true, "basic_detectors": true, "standard_support": true}', 
 '{"api_calls_per_month": 1000, "storage_gb": 1, "detector_runs_per_month": 500, "analysis_runs_per_month": 10}', 
 NULL, NULL),
('white_glove_basic', 'White-Glove Basic', 'white_glove', 'White-glove service with dedicated support and custom implementation', 0.00, 0.00, 'USD',
 '{"api_calls": true, "all_detectors": true, "white_glove_support": true, "custom_taxonomy": true, "custom_integration": true, "dedicated_success_manager": true, "priority_support": true, "custom_sla": true}',
 '{"api_calls_per_month": 50000, "storage_gb": 100, "detector_runs_per_month": 25000, "analysis_runs_per_month": 500, "custom_limits": true}',
 NULL, NULL),
('white_glove_standard', 'White-Glove Standard', 'white_glove', 'White-glove service with advanced features and custom compliance frameworks', 0.00, 0.00, 'USD',
 '{"api_calls": true, "all_detectors": true, "white_glove_support": true, "custom_taxonomy": true, "custom_integration": true, "dedicated_success_manager": true, "priority_support": true, "custom_sla": true, "custom_compliance_frameworks": true, "advanced_analytics": true, "api_access": true}',
 '{"api_calls_per_month": 100000, "storage_gb": 250, "detector_runs_per_month": 50000, "analysis_runs_per_month": 1000, "custom_limits": true}',
 NULL, NULL),
('white_glove_premium', 'White-Glove Premium', 'white_glove', 'White-glove service with full customization and enterprise features', 0.00, 0.00, 'USD',
 '{"api_calls": true, "all_detectors": true, "white_glove_support": true, "custom_taxonomy": true, "custom_integration": true, "dedicated_success_manager": true, "priority_support": true, "custom_sla": true, "custom_compliance_frameworks": true, "advanced_analytics": true, "api_access": true, "sso": true, "audit_logs": true, "custom_detectors": true, "on_premise_deployment": true}',
 '{"api_calls_per_month": 500000, "storage_gb": 1000, "detector_runs_per_month": 250000, "analysis_runs_per_month": 5000, "custom_limits": true}',
 NULL, NULL),
('enterprise', 'Enterprise Plan', 'enterprise', 'Enterprise plan for large organizations', 299.00, 2990.00, 'USD',
 '{"api_calls": true, "all_detectors": true, "dedicated_support": true, "custom_taxonomy": true, "advanced_analytics": true, "api_access": true, "sso": true, "audit_logs": true}',
 '{"api_calls_per_month": 200000, "storage_gb": 200, "detector_runs_per_month": 100000, "analysis_runs_per_month": 2000}',
 'price_enterprise_monthly', 'prod_enterprise')
ON CONFLICT (plan_id) DO NOTHING;

INSERT INTO promotional_codes (code, description, discount_type, discount_value, currency, max_uses, valid_from, valid_until, applicable_plans, is_active) VALUES
('WELCOME20', 'Welcome discount for new users', 'percentage', 20.00, 'USD', 1000, NOW(), NOW() + INTERVAL '1 year', '["starter", "professional"]', TRUE),
('STARTUP50', 'Startup discount', 'percentage', 50.00, 'USD', 100, NOW(), NOW() + INTERVAL '6 months', '["starter", "professional", "enterprise"]', TRUE),
('FREETRIAL', 'Free trial extension', 'free_trial', 30.00, 'USD', 500, NOW(), NOW() + INTERVAL '3 months', '["starter", "professional"]', TRUE)
ON CONFLICT (code) DO NOTHING;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Create application user (replace with actual username)
-- CREATE USER llama_mapper_app WITH PASSWORD 'your_secure_password';

-- Grant necessary permissions
-- GRANT CONNECT ON DATABASE llama_mapper TO llama_mapper_app;
-- GRANT USAGE ON SCHEMA public TO llama_mapper_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO llama_mapper_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO llama_mapper_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO llama_mapper_app;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE storage_records IS 'Primary table for storing mapping results with Azure-specific optimizations';
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for compliance reporting and security monitoring';
COMMENT ON TABLE tenant_configs IS 'Tenant-specific configuration with Azure resource mapping';
COMMENT ON TABLE model_versions IS 'Model version tracking with Azure Blob Storage integration';
COMMENT ON TABLE compliance_mappings IS 'Maps taxonomy labels to compliance framework controls (SOC2, ISO27001, HIPAA)';

COMMENT ON COLUMN storage_records.azure_region IS 'Azure region where the record is stored';
COMMENT ON COLUMN storage_records.resource_group IS 'Azure resource group for the record';
COMMENT ON COLUMN storage_records.subscription_id IS 'Azure subscription ID';
COMMENT ON COLUMN storage_records.blob_url IS 'Azure Blob Storage URL for the record';

COMMENT ON COLUMN audit_logs.azure_region IS 'Azure region where the audit event occurred';
COMMENT ON COLUMN audit_logs.subscription_id IS 'Azure subscription ID';
COMMENT ON COLUMN audit_logs.resource_group IS 'Azure resource group';

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Azure Database Schema for Comply-AI Platform created successfully!';
    RAISE NOTICE 'Core Tables: storage_records, audit_logs, tenant_configs, model_versions, compliance_mappings, detector_configs, taxonomy_labels';
    RAISE NOTICE 'Detector Orchestration Tables: orchestration_requests, orchestration_responses, detector_results, routing_plans';
    RAISE NOTICE 'Analysis Module Tables: analysis_requests, analysis_responses, analysis_metrics, analysis_templates';
    RAISE NOTICE 'Llama Mapper Tables: mapper_requests, mapper_responses (additional to storage_records)';
    RAISE NOTICE 'User Management Tables: users, user_roles, user_role_assignments, user_sessions';
    RAISE NOTICE 'Billing Tables: billing_plans, user_subscriptions, usage_records, billing_invoices, payment_methods';
    RAISE NOTICE 'Free Tier Tables: free_tier_usage, promotional_codes, promotional_code_usage';
    RAISE NOTICE 'White-Glove Service Tables: white_glove_services, white_glove_milestones, white_glove_communications, white_glove_deliverables, white_glove_feedback';
    RAISE NOTICE 'Azure Infrastructure Tables: azure_container_registries, azure_virtual_networks, azure_subnets, azure_network_security_groups, azure_application_gateways, azure_backup_vaults, azure_managed_identities, azure_log_analytics_workspaces, azure_application_insights, azure_security_center';
    RAISE NOTICE 'Indexes created: Performance and GIN indexes for JSONB columns';
    RAISE NOTICE 'Security: Row Level Security enabled with tenant isolation policies';
    RAISE NOTICE 'Partitioning: Hash partitioning for storage_records, range partitioning for audit_logs';
    RAISE NOTICE 'Performance: Azure-optimized PostgreSQL settings applied';
    RAISE NOTICE 'Functions: Automatic timestamp updates and cleanup functions';
    RAISE NOTICE 'Views: Common query views for statistics and reporting';
    RAISE NOTICE 'Sample data: Basic taxonomy labels, compliance mappings, Azure infrastructure data, service-specific data, user roles, billing plans, and promotional codes inserted';
    RAISE NOTICE 'Total Tables Created: 40 tables covering all 3 services, complete Azure infrastructure, user management, Stripe billing integration, and white-glove service management';
END $$;
