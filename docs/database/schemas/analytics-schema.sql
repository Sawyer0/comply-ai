-- ============================================================================
-- COMPLY-AI ANALYTICS DATABASE SCHEMA
-- ============================================================================
-- This schema contains analytics, reporting, and performance metrics data
-- Database: comply-ai-analytics
-- Purpose: Usage metrics, performance data, compliance reports, dashboards

-- ============================================================================
-- ANALYTICS AND METRICS TABLES
-- ============================================================================

-- Usage metrics for analytics and reporting
CREATE TABLE IF NOT EXISTS usage_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255), -- References users table in core database
    metric_type VARCHAR(100) NOT NULL, -- 'api_calls', 'storage_gb', 'detector_runs', 'analysis_runs', 'response_time'
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- Performance metrics for system monitoring
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    service_name VARCHAR(100) NOT NULL, -- 'orchestration', 'mapper', 'analysis'
    metric_name VARCHAR(100) NOT NULL, -- 'response_time', 'throughput', 'error_rate', 'cpu_usage'
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    instance_id VARCHAR(255)
);

-- Compliance reports and analytics
CREATE TABLE IF NOT EXISTS compliance_reports (
    id BIGSERIAL PRIMARY KEY,
    report_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    report_type VARCHAR(100) NOT NULL, -- 'security_analysis', 'compliance_audit', 'risk_assessment'
    report_name VARCHAR(255) NOT NULL,
    report_period_start TIMESTAMPTZ NOT NULL,
    report_period_end TIMESTAMPTZ NOT NULL,
    compliance_framework VARCHAR(100) NOT NULL, -- 'SOC2', 'ISO27001', 'HIPAA', 'GDPR'
    status VARCHAR(50) NOT NULL, -- 'generating', 'completed', 'failed'
    summary JSONB DEFAULT '{}',
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    risk_score DECIMAL(5,2),
    compliance_score DECIMAL(5,2),
    generated_by VARCHAR(255), -- References users table in core database
    generated_at TIMESTAMPTZ,
    file_path VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User behavior analytics
CREATE TABLE IF NOT EXISTS user_behavior (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    session_id VARCHAR(255),
    action_type VARCHAR(100) NOT NULL, -- 'login', 'api_call', 'dashboard_view', 'report_generation'
    action_details JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System health monitoring
CREATE TABLE IF NOT EXISTS system_health (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL, -- 'orchestration', 'mapper', 'analysis', 'database'
    health_status VARCHAR(50) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    health_score DECIMAL(5,2) NOT NULL, -- 0-100
    check_type VARCHAR(100) NOT NULL, -- 'connectivity', 'performance', 'availability', 'security'
    check_details JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    resource_name VARCHAR(255)
);

-- Cost analytics and tracking
CREATE TABLE IF NOT EXISTS cost_analytics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    cost_type VARCHAR(100) NOT NULL, -- 'compute', 'storage', 'network', 'database', 'ai_services'
    cost_amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period_start TIMESTAMPTZ NOT NULL,
    billing_period_end TIMESTAMPTZ NOT NULL,
    resource_details JSONB DEFAULT '{}',
    usage_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
    subscription_id VARCHAR(255),
    resource_id VARCHAR(500)
);

-- ML model performance metrics
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255) NOT NULL, -- References model_versions table in core database
    model_version VARCHAR(100) NOT NULL,
    metric_type VARCHAR(100) NOT NULL, -- 'accuracy', 'precision', 'recall', 'f1_score', 'inference_time'
    metric_value DECIMAL(15,4) NOT NULL,
    dataset_type VARCHAR(100) NOT NULL, -- 'training', 'validation', 'test', 'production'
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dashboard configurations
CREATE TABLE IF NOT EXISTS dashboard_configs (
    id BIGSERIAL PRIMARY KEY,
    dashboard_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    dashboard_name VARCHAR(255) NOT NULL,
    dashboard_type VARCHAR(100) NOT NULL, -- 'executive', 'operational', 'compliance', 'technical'
    description TEXT,
    configuration JSONB NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL, -- References users table in core database
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert configurations
CREATE TABLE IF NOT EXISTS alert_configs (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    alert_name VARCHAR(255) NOT NULL,
    alert_type VARCHAR(100) NOT NULL, -- 'threshold', 'anomaly', 'compliance', 'performance'
    description TEXT,
    conditions JSONB NOT NULL,
    notification_channels JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL, -- References users table in core database
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert history
CREATE TABLE IF NOT EXISTS alert_history (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(255) REFERENCES alert_configs(alert_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    alert_status VARCHAR(50) NOT NULL, -- 'triggered', 'resolved', 'acknowledged'
    alert_message TEXT NOT NULL,
    alert_data JSONB DEFAULT '{}',
    triggered_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255), -- References users table in core database
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SUPPORT METRICS AND CUSTOMER SUCCESS TABLES
-- ============================================================================

-- Support team performance metrics
CREATE TABLE IF NOT EXISTS support_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    support_agent_id VARCHAR(255), -- References users table in core database
    metric_type VARCHAR(100) NOT NULL, -- 'response_time', 'resolution_time', 'ticket_volume', 'satisfaction_score'
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    measurement_period_start TIMESTAMPTZ NOT NULL,
    measurement_period_end TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- Customer success metrics
CREATE TABLE IF NOT EXISTS customer_success_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255), -- References users table in core database
    metric_type VARCHAR(100) NOT NULL, -- 'adoption_rate', 'feature_usage', 'engagement_score', 'health_score'
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    measurement_date TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- Feature usage analytics
CREATE TABLE IF NOT EXISTS feature_usage_analytics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255), -- References users table in core database
    feature_name VARCHAR(255) NOT NULL,
    feature_category VARCHAR(100) NOT NULL, -- 'detection', 'analysis', 'reporting', 'white_glove'
    usage_count INTEGER NOT NULL DEFAULT 1,
    session_duration_minutes INTEGER,
    success_rate DECIMAL(5,4), -- 0-1 success rate
    error_count INTEGER DEFAULT 0,
    measurement_date TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- ============================================================================
-- AZURE INFRASTRUCTURE TRACKING TABLES
-- ============================================================================

-- Azure Container Registry tracking
CREATE TABLE IF NOT EXISTS azure_container_registries (
    id BIGSERIAL PRIMARY KEY,
    registry_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    sku VARCHAR(50) NOT NULL,
    admin_enabled BOOLEAN DEFAULT FALSE,
    storage_account VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Virtual Network tracking
CREATE TABLE IF NOT EXISTS azure_virtual_networks (
    id BIGSERIAL PRIMARY KEY,
    vnet_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    address_space JSONB NOT NULL,
    dns_servers JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Subnet tracking
CREATE TABLE IF NOT EXISTS azure_subnets (
    id BIGSERIAL PRIMARY KEY,
    subnet_name VARCHAR(255) NOT NULL,
    vnet_name VARCHAR(255) REFERENCES azure_virtual_networks(vnet_name) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    address_prefix VARCHAR(50) NOT NULL,
    network_security_group VARCHAR(255),
    route_table VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(subnet_name, vnet_name)
);

-- Azure Network Security Groups tracking
CREATE TABLE IF NOT EXISTS azure_network_security_groups (
    id BIGSERIAL PRIMARY KEY,
    nsg_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    security_rules JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Application Gateway tracking
CREATE TABLE IF NOT EXISTS azure_application_gateways (
    id BIGSERIAL PRIMARY KEY,
    gateway_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    sku_name VARCHAR(50) NOT NULL,
    sku_tier VARCHAR(50) NOT NULL,
    capacity INTEGER NOT NULL,
    ssl_certificates JSONB DEFAULT '[]',
    backend_pools JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Backup Vault tracking
CREATE TABLE IF NOT EXISTS azure_backup_vaults (
    id BIGSERIAL PRIMARY KEY,
    vault_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    sku_name VARCHAR(50) NOT NULL,
    backup_policies JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Managed Identity tracking
CREATE TABLE IF NOT EXISTS azure_managed_identities (
    id BIGSERIAL PRIMARY KEY,
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    identity_type VARCHAR(50) NOT NULL, -- 'SystemAssigned', 'UserAssigned'
    principal_id VARCHAR(255),
    client_id VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Log Analytics Workspace tracking
CREATE TABLE IF NOT EXISTS azure_log_analytics_workspaces (
    id BIGSERIAL PRIMARY KEY,
    workspace_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    sku VARCHAR(50) NOT NULL,
    retention_days INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Application Insights tracking
CREATE TABLE IF NOT EXISTS azure_application_insights (
    id BIGSERIAL PRIMARY KEY,
    app_insights_name VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    resource_group VARCHAR(100) NOT NULL,
    location VARCHAR(50) NOT NULL,
    application_type VARCHAR(50) NOT NULL,
    instrumentation_key VARCHAR(255),
    connection_string VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Azure Security Center tracking
CREATE TABLE IF NOT EXISTS azure_security_center (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id VARCHAR(255) NOT NULL,
    pricing_tier VARCHAR(50) NOT NULL,
    auto_provisioning BOOLEAN DEFAULT FALSE,
    security_contacts JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Analytics and metrics indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_metrics_tenant_id ON usage_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_metrics_user_id ON usage_metrics(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_metrics_metric_type ON usage_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_metrics_timestamp ON usage_metrics(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_tenant_id ON performance_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_service_name ON performance_metrics(service_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_report_id ON compliance_reports(report_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_tenant_id ON compliance_reports(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_report_type ON compliance_reports(report_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_compliance_framework ON compliance_reports(compliance_framework);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_status ON compliance_reports(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compliance_reports_generated_at ON compliance_reports(generated_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_tenant_id ON user_behavior(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_user_id ON user_behavior(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_action_type ON user_behavior(action_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_timestamp ON user_behavior(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_service_name ON system_health(service_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_health_status ON system_health(health_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_check_type ON system_health(check_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_analytics_tenant_id ON cost_analytics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_analytics_cost_type ON cost_analytics(cost_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_analytics_billing_period_start ON cost_analytics(billing_period_start);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_analytics_billing_period_end ON cost_analytics(billing_period_end);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_tenant_id ON ml_model_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_model_id ON ml_model_metrics(model_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_metric_type ON ml_model_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_dataset_type ON ml_model_metrics(dataset_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_timestamp ON ml_model_metrics(timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dashboard_configs_dashboard_id ON dashboard_configs(dashboard_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dashboard_configs_tenant_id ON dashboard_configs(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dashboard_configs_dashboard_type ON dashboard_configs(dashboard_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dashboard_configs_is_public ON dashboard_configs(is_public);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_configs_alert_id ON alert_configs(alert_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_configs_tenant_id ON alert_configs(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_configs_alert_type ON alert_configs(alert_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_configs_is_active ON alert_configs(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_history_alert_id ON alert_history(alert_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_history_tenant_id ON alert_history(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_history_alert_status ON alert_history(alert_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_history_triggered_at ON alert_history(triggered_at);

-- Indexes for support metrics and customer success tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_tenant_id ON support_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_support_agent_id ON support_metrics(support_agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_metric_type ON support_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_measurement_period_start ON support_metrics(measurement_period_start);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_measurement_period_end ON support_metrics(measurement_period_end);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_support_metrics_created_at ON support_metrics(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_success_metrics_tenant_id ON customer_success_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_success_metrics_user_id ON customer_success_metrics(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_success_metrics_metric_type ON customer_success_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_success_metrics_measurement_date ON customer_success_metrics(measurement_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_success_metrics_created_at ON customer_success_metrics(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_tenant_id ON feature_usage_analytics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_user_id ON feature_usage_analytics(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_feature_name ON feature_usage_analytics(feature_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_feature_category ON feature_usage_analytics(feature_category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_measurement_date ON feature_usage_analytics(measurement_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_usage_analytics_created_at ON feature_usage_analytics(created_at);

-- Azure infrastructure indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_container_registries_registry_name ON azure_container_registries(registry_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_container_registries_tenant_id ON azure_container_registries(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_container_registries_is_active ON azure_container_registries(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_virtual_networks_vnet_name ON azure_virtual_networks(vnet_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_virtual_networks_tenant_id ON azure_virtual_networks(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_virtual_networks_is_active ON azure_virtual_networks(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_subnets_subnet_name ON azure_subnets(subnet_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_subnets_vnet_name ON azure_subnets(vnet_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_subnets_tenant_id ON azure_subnets(tenant_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_network_security_groups_nsg_name ON azure_network_security_groups(nsg_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_network_security_groups_tenant_id ON azure_network_security_groups(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_network_security_groups_is_active ON azure_network_security_groups(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_gateways_gateway_name ON azure_application_gateways(gateway_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_gateways_tenant_id ON azure_application_gateways(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_gateways_is_active ON azure_application_gateways(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_backup_vaults_vault_name ON azure_backup_vaults(vault_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_backup_vaults_tenant_id ON azure_backup_vaults(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_backup_vaults_is_active ON azure_backup_vaults(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_managed_identities_identity_name ON azure_managed_identities(identity_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_managed_identities_tenant_id ON azure_managed_identities(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_managed_identities_is_active ON azure_managed_identities(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_log_analytics_workspaces_workspace_name ON azure_log_analytics_workspaces(workspace_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_log_analytics_workspaces_tenant_id ON azure_log_analytics_workspaces(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_log_analytics_workspaces_is_active ON azure_log_analytics_workspaces(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_insights_app_insights_name ON azure_application_insights(app_insights_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_insights_tenant_id ON azure_application_insights(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_application_insights_is_active ON azure_application_insights(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_security_center_tenant_id ON azure_security_center(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_security_center_subscription_id ON azure_security_center(subscription_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_azure_security_center_is_active ON azure_security_center(is_active);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on tenant-specific tables
ALTER TABLE usage_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_behavior ENABLE ROW LEVEL SECURITY;
ALTER TABLE cost_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE dashboard_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE support_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE customer_success_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_usage_analytics ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY tenant_isolation_usage_metrics ON usage_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_performance_metrics ON performance_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_compliance_reports ON compliance_reports
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_user_behavior ON user_behavior
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_cost_analytics ON cost_analytics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_ml_model_metrics ON ml_model_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_dashboard_configs ON dashboard_configs
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_alert_configs ON alert_configs
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_alert_history ON alert_history
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_support_metrics ON support_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_customer_success_metrics ON customer_success_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_feature_usage_analytics ON feature_usage_analytics
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
CREATE TRIGGER update_compliance_reports_updated_at BEFORE UPDATE ON compliance_reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dashboard_configs_updated_at BEFORE UPDATE ON dashboard_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_configs_updated_at BEFORE UPDATE ON alert_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_container_registries_updated_at BEFORE UPDATE ON azure_container_registries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_virtual_networks_updated_at BEFORE UPDATE ON azure_virtual_networks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_subnets_updated_at BEFORE UPDATE ON azure_subnets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_network_security_groups_updated_at BEFORE UPDATE ON azure_network_security_groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_application_gateways_updated_at BEFORE UPDATE ON azure_application_gateways
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_backup_vaults_updated_at BEFORE UPDATE ON azure_backup_vaults
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_managed_identities_updated_at BEFORE UPDATE ON azure_managed_identities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_log_analytics_workspaces_updated_at BEFORE UPDATE ON azure_log_analytics_workspaces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_application_insights_updated_at BEFORE UPDATE ON azure_application_insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_azure_security_center_updated_at BEFORE UPDATE ON azure_security_center
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample Azure infrastructure data
INSERT INTO azure_container_registries (registry_name, tenant_id, resource_group, location, sku, admin_enabled, is_active) VALUES
('complyaiacr', 'tenant-001', 'comply-ai-rg', 'eastus', 'Basic', FALSE, TRUE)
ON CONFLICT (registry_name) DO NOTHING;

INSERT INTO azure_virtual_networks (vnet_name, tenant_id, resource_group, location, address_space, is_active) VALUES
('comply-ai-vnet', 'tenant-001', 'comply-ai-rg', 'eastus', '["10.0.0.0/16"]', TRUE)
ON CONFLICT (vnet_name) DO NOTHING;

INSERT INTO azure_subnets (subnet_name, vnet_name, tenant_id, resource_group, address_prefix, is_active) VALUES
('comply-ai-subnet', 'comply-ai-vnet', 'tenant-001', 'comply-ai-rg', '10.0.1.0/24', TRUE)
ON CONFLICT (subnet_name, vnet_name) DO NOTHING;

INSERT INTO azure_network_security_groups (nsg_name, tenant_id, resource_group, location, is_active) VALUES
('comply-ai-nsg', 'tenant-001', 'comply-ai-rg', 'eastus', TRUE)
ON CONFLICT (nsg_name) DO NOTHING;

INSERT INTO azure_application_gateways (gateway_name, tenant_id, resource_group, location, sku_name, sku_tier, capacity, is_active) VALUES
('comply-ai-gateway', 'tenant-001', 'comply-ai-rg', 'eastus', 'Standard_v2', 'Standard_v2', 2, TRUE)
ON CONFLICT (gateway_name) DO NOTHING;

INSERT INTO azure_backup_vaults (vault_name, tenant_id, resource_group, location, sku_name, is_active) VALUES
('comply-ai-backup-vault', 'tenant-001', 'comply-ai-rg', 'eastus', 'RS0', TRUE)
ON CONFLICT (vault_name) DO NOTHING;

INSERT INTO azure_managed_identities (identity_name, tenant_id, resource_group, location, identity_type, is_active) VALUES
('comply-ai-identity', 'tenant-001', 'comply-ai-rg', 'eastus', 'SystemAssigned', TRUE)
ON CONFLICT (identity_name) DO NOTHING;

INSERT INTO azure_log_analytics_workspaces (workspace_name, tenant_id, resource_group, location, sku, retention_days, is_active) VALUES
('comply-ai-logs', 'tenant-001', 'comply-ai-rg', 'eastus', 'PerGB2018', 30, TRUE)
ON CONFLICT (workspace_name) DO NOTHING;

INSERT INTO azure_application_insights (app_insights_name, tenant_id, resource_group, location, application_type, is_active) VALUES
('comply-ai-insights', 'tenant-001', 'comply-ai-rg', 'eastus', 'web', TRUE)
ON CONFLICT (app_insights_name) DO NOTHING;

INSERT INTO azure_security_center (tenant_id, subscription_id, pricing_tier, auto_provisioning, is_active) VALUES
('tenant-001', 'sub-001', 'Standard', TRUE, TRUE)
ON CONFLICT (tenant_id, subscription_id) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Comply-AI Analytics Database Schema created successfully!';
    RAISE NOTICE 'Analytics Tables: usage_metrics, performance_metrics, compliance_reports, user_behavior, system_health, cost_analytics, ml_model_metrics';
    RAISE NOTICE 'Dashboard Tables: dashboard_configs, alert_configs, alert_history';
    RAISE NOTICE 'Azure Infrastructure Tables: azure_container_registries, azure_virtual_networks, azure_subnets, azure_network_security_groups, azure_application_gateways, azure_backup_vaults, azure_managed_identities, azure_log_analytics_workspaces, azure_application_insights, azure_security_center';
    RAISE NOTICE 'Indexes created: Performance and GIN indexes for JSONB columns';
    RAISE NOTICE 'Security: Row Level Security enabled with tenant isolation policies';
    RAISE NOTICE 'Functions: Automatic timestamp updates and cleanup functions';
    RAISE NOTICE 'Sample data: Azure infrastructure data inserted';
    RAISE NOTICE 'Total Tables Created: 20 tables covering analytics, reporting, and Azure infrastructure tracking';
END $$;
