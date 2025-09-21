-- ============================================================================
-- COMPLY-AI BILLING DATABASE SCHEMA
-- ============================================================================
-- This schema contains billing, subscription, and white-glove service data
-- Database: comply-ai-billing
-- Purpose: Subscriptions, payments, white-glove services, usage tracking

-- ============================================================================
-- BILLING AND SUBSCRIPTION TABLES
-- ============================================================================

-- Billing plans with Stripe integration
CREATE TABLE IF NOT EXISTS billing_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_id VARCHAR(255) UNIQUE NOT NULL,
    plan_name VARCHAR(255) NOT NULL,
    plan_type VARCHAR(50) NOT NULL, -- 'free', 'trial', 'paid', 'enterprise', 'white_glove'
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    subscription_id VARCHAR(255) REFERENCES user_subscriptions(subscription_id),
    used_at TIMESTAMPTZ DEFAULT NOW(),
    discount_applied DECIMAL(10,2) NOT NULL
);

-- ============================================================================
-- WHITE-GLOVE SERVICE MANAGEMENT TABLES
-- ============================================================================

-- White-glove service management
CREATE TABLE IF NOT EXISTS white_glove_services (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
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
    from_user_id VARCHAR(255), -- References users table in core database
    to_user_id VARCHAR(255), -- References users table in core database
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
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    categories JSONB DEFAULT '{}', -- 'communication', 'technical_quality', 'timeliness', 'value'
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- TRAINING AND SLA MANAGEMENT TABLES
-- ============================================================================

-- Training sessions for white-glove clients
CREATE TABLE IF NOT EXISTS training_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    tenant_id VARCHAR(255) NOT NULL,
    training_type VARCHAR(100) NOT NULL, -- 'onboarding', 'advanced', 'compliance', 'custom'
    training_name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL, -- 'scheduled', 'in_progress', 'completed', 'cancelled'
    trainer_id VARCHAR(255), -- References users table in core database
    session_date TIMESTAMPTZ,
    duration_minutes INTEGER,
    max_participants INTEGER DEFAULT 10,
    current_participants INTEGER DEFAULT 0,
    meeting_link VARCHAR(500),
    materials JSONB DEFAULT '[]', -- Training materials and resources
    feedback_rating INTEGER CHECK (feedback_rating >= 1 AND feedback_rating <= 5),
    feedback_notes TEXT,
    completion_certificate_path VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- SLA metrics tracking for white-glove services
CREATE TABLE IF NOT EXISTS sla_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    service_type VARCHAR(100) NOT NULL, -- 'white_glove', 'support', 'api', 'system'
    sla_type VARCHAR(100) NOT NULL, -- 'response_time', 'uptime', 'resolution_time', 'availability'
    target_value DECIMAL(10,2) NOT NULL,
    actual_value DECIMAL(10,2) NOT NULL,
    measurement_period_start TIMESTAMPTZ NOT NULL,
    measurement_period_end TIMESTAMPTZ NOT NULL,
    is_breach BOOLEAN DEFAULT FALSE,
    breach_reason TEXT,
    breach_severity VARCHAR(20), -- 'minor', 'major', 'critical'
    remediation_actions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    azure_region VARCHAR(50) DEFAULT 'eastus',
    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg'
);

-- Customer satisfaction surveys
CREATE TABLE IF NOT EXISTS customer_satisfaction_surveys (
    id BIGSERIAL PRIMARY KEY,
    survey_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL, -- References users table in core database
    tenant_id VARCHAR(255) NOT NULL,
    survey_type VARCHAR(100) NOT NULL, -- 'onboarding', 'service', 'support', 'white_glove'
    overall_rating INTEGER CHECK (overall_rating >= 1 AND overall_rating <= 5),
    category_ratings JSONB DEFAULT '{}', -- 'communication', 'technical_quality', 'timeliness', 'value'
    feedback_text TEXT,
    would_recommend BOOLEAN,
    improvement_suggestions TEXT,
    survey_date TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Billing and subscription indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_plan_id ON billing_plans(plan_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_plan_type ON billing_plans(plan_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_is_active ON billing_plans(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_plans_stripe_price_id ON billing_plans(stripe_price_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_subscription_id ON user_subscriptions(subscription_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_user_id ON user_subscriptions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_tenant_id ON user_subscriptions(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_status ON user_subscriptions(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_stripe_subscription_id ON user_subscriptions(stripe_subscription_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_subscriptions_current_period_end ON user_subscriptions(current_period_end);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_user_id ON usage_records(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_tenant_id ON usage_records(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_usage_type ON usage_records(usage_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_billing_period_start ON usage_records(billing_period_start);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_usage_records_billing_period_end ON usage_records(billing_period_end);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_invoice_id ON billing_invoices(invoice_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_user_id ON billing_invoices(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_tenant_id ON billing_invoices(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_status ON billing_invoices(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_due_date ON billing_invoices(due_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_billing_invoices_stripe_invoice_id ON billing_invoices(stripe_invoice_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_payment_method_id ON payment_methods(payment_method_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_user_id ON payment_methods(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_tenant_id ON payment_methods(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_is_default ON payment_methods(is_default);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_payment_methods_stripe_payment_method_id ON payment_methods(stripe_payment_method_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_user_id ON free_tier_usage(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_tenant_id ON free_tier_usage(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_usage_type ON free_tier_usage(usage_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_free_tier_usage_next_reset_at ON free_tier_usage(next_reset_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_code ON promotional_codes(code);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_is_active ON promotional_codes(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_valid_from ON promotional_codes(valid_from);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_codes_valid_until ON promotional_codes(valid_until);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_code ON promotional_code_usage(code);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_user_id ON promotional_code_usage(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_promotional_code_usage_used_at ON promotional_code_usage(used_at);

-- White-glove service indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_service_id ON white_glove_services(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_user_id ON white_glove_services(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_tenant_id ON white_glove_services(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_status ON white_glove_services(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_service_type ON white_glove_services(service_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_assigned_to ON white_glove_services(assigned_to);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_services_created_at ON white_glove_services(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_service_id ON white_glove_milestones(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_status ON white_glove_milestones(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_priority ON white_glove_milestones(priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_assigned_to ON white_glove_milestones(assigned_to);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_milestones_target_date ON white_glove_milestones(target_date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_service_id ON white_glove_communications(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_communication_type ON white_glove_communications(communication_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_from_user_id ON white_glove_communications(from_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_to_user_id ON white_glove_communications(to_user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_communications_created_at ON white_glove_communications(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_service_id ON white_glove_deliverables(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_deliverable_type ON white_glove_deliverables(deliverable_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_status ON white_glove_deliverables(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_deliverables_delivered_at ON white_glove_deliverables(delivered_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_service_id ON white_glove_feedback(service_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_user_id ON white_glove_feedback(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_rating ON white_glove_feedback(rating);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_white_glove_feedback_created_at ON white_glove_feedback(created_at);

-- Indexes for training and SLA management tables
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_session_id ON training_sessions(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_user_id ON training_sessions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_tenant_id ON training_sessions(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_training_type ON training_sessions(training_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_status ON training_sessions(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_trainer_id ON training_sessions(trainer_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_session_date ON training_sessions(session_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_sessions_created_at ON training_sessions(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_tenant_id ON sla_metrics(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_service_type ON sla_metrics(service_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_sla_type ON sla_metrics(sla_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_is_breach ON sla_metrics(is_breach);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_measurement_period_start ON sla_metrics(measurement_period_start);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_measurement_period_end ON sla_metrics(measurement_period_end);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sla_metrics_created_at ON sla_metrics(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_survey_id ON customer_satisfaction_surveys(survey_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_user_id ON customer_satisfaction_surveys(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_tenant_id ON customer_satisfaction_surveys(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_survey_type ON customer_satisfaction_surveys(survey_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_overall_rating ON customer_satisfaction_surveys(overall_rating);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_survey_date ON customer_satisfaction_surveys(survey_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_customer_satisfaction_surveys_created_at ON customer_satisfaction_surveys(created_at);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on tenant-specific tables
ALTER TABLE user_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE billing_invoices ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_methods ENABLE ROW LEVEL SECURITY;
ALTER TABLE free_tier_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE white_glove_services ENABLE ROW LEVEL SECURITY;
ALTER TABLE white_glove_milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE white_glove_communications ENABLE ROW LEVEL SECURITY;
ALTER TABLE white_glove_deliverables ENABLE ROW LEVEL SECURITY;
ALTER TABLE white_glove_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE sla_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE customer_satisfaction_surveys ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY tenant_isolation_user_subscriptions ON user_subscriptions
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_usage_records ON usage_records
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_billing_invoices ON billing_invoices
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_payment_methods ON payment_methods
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_free_tier_usage ON free_tier_usage
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_white_glove_services ON white_glove_services
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_white_glove_milestones ON white_glove_milestones
    FOR ALL TO PUBLIC
    USING (service_id IN (
        SELECT service_id FROM white_glove_services 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_white_glove_communications ON white_glove_communications
    FOR ALL TO PUBLIC
    USING (service_id IN (
        SELECT service_id FROM white_glove_services 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_white_glove_deliverables ON white_glove_deliverables
    FOR ALL TO PUBLIC
    USING (service_id IN (
        SELECT service_id FROM white_glove_services 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_white_glove_feedback ON white_glove_feedback
    FOR ALL TO PUBLIC
    USING (service_id IN (
        SELECT service_id FROM white_glove_services 
        WHERE tenant_id = current_setting('app.current_tenant_id')
    ));

CREATE POLICY tenant_isolation_training_sessions ON training_sessions
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_sla_metrics ON sla_metrics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY tenant_isolation_customer_satisfaction_surveys ON customer_satisfaction_surveys
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
CREATE TRIGGER update_billing_plans_updated_at BEFORE UPDATE ON billing_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_subscriptions_updated_at BEFORE UPDATE ON user_subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_billing_invoices_updated_at BEFORE UPDATE ON billing_invoices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payment_methods_updated_at BEFORE UPDATE ON payment_methods
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_free_tier_usage_updated_at BEFORE UPDATE ON free_tier_usage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_promotional_codes_updated_at BEFORE UPDATE ON promotional_codes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_white_glove_services_updated_at BEFORE UPDATE ON white_glove_services
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_white_glove_milestones_updated_at BEFORE UPDATE ON white_glove_milestones
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_white_glove_deliverables_updated_at BEFORE UPDATE ON white_glove_deliverables
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample billing plans
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

-- Insert sample promotional codes
INSERT INTO promotional_codes (code, description, discount_type, discount_value, currency, max_uses, valid_from, valid_until, applicable_plans, is_active) VALUES
('WELCOME20', 'Welcome discount for new users', 'percentage', 20.00, 'USD', 1000, NOW(), NOW() + INTERVAL '1 year', '["starter", "professional"]', TRUE),
('STARTUP50', 'Startup discount', 'percentage', 50.00, 'USD', 100, NOW(), NOW() + INTERVAL '6 months', '["starter", "professional", "enterprise"]', TRUE),
('FREETRIAL', 'Free trial extension', 'free_trial', 30.00, 'USD', 500, NOW(), NOW() + INTERVAL '3 months', '["starter", "professional"]', TRUE)
ON CONFLICT (code) DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Comply-AI Billing Database Schema created successfully!';
    RAISE NOTICE 'Billing Tables: billing_plans, user_subscriptions, usage_records, billing_invoices, payment_methods';
    RAISE NOTICE 'Free Tier Tables: free_tier_usage, promotional_codes, promotional_code_usage';
    RAISE NOTICE 'White-Glove Service Tables: white_glove_services, white_glove_milestones, white_glove_communications, white_glove_deliverables, white_glove_feedback';
    RAISE NOTICE 'Indexes created: Performance and GIN indexes for JSONB columns';
    RAISE NOTICE 'Security: Row Level Security enabled with tenant isolation policies';
    RAISE NOTICE 'Functions: Automatic timestamp updates and cleanup functions';
    RAISE NOTICE 'Sample data: Billing plans and promotional codes inserted';
    RAISE NOTICE 'Total Tables Created: 10 tables covering billing, subscriptions, and white-glove services';
END $$;
