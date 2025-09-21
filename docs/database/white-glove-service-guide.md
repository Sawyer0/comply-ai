# White-Glove Service Management Guide

This guide provides comprehensive documentation for the white-glove service management system integrated into the Comply-AI platform's database schema.

## Table of Contents

1. [Overview](#overview)
2. [White-Glove Service Model](#white-glove-service-model)
3. [Database Schema](#database-schema)
4. [Service Types](#service-types)
5. [Workflow Management](#workflow-management)
6. [Implementation Examples](#implementation-examples)
7. [Best Practices](#best-practices)
8. [Monitoring and Analytics](#monitoring-and-analytics)

## Overview

The white-glove service model provides a comprehensive framework for delivering premium, customized services to enterprise clients. This system supports:

- **Custom Implementation**: Tailored deployment and configuration
- **Dedicated Support**: Assigned success managers and engineers
- **Custom Integration**: API integrations and third-party connections
- **Custom Compliance**: Industry-specific compliance frameworks
- **Project Management**: Milestone tracking and deliverable management
- **Communication Tracking**: All client interactions and documentation
- **Feedback Collection**: Service quality and satisfaction monitoring

## White-Glove Service Model

### Service Tiers

| Tier | Description | Features | Target Audience |
|------|-------------|----------|-----------------|
| **White-Glove Basic** | Custom implementation with dedicated support | Custom taxonomy, integration, success manager, priority support, custom SLA | Small to medium enterprises |
| **White-Glove Standard** | Advanced features with custom compliance frameworks | All Basic features + custom compliance frameworks, advanced analytics, API access | Growing enterprises |
| **White-Glove Premium** | Full customization with enterprise features | All Standard features + SSO, audit logs, custom detectors, on-premise deployment | Large enterprises |

### Key Features

- **Dedicated Success Manager**: Assigned account manager for each client
- **Custom SLA**: Tailored service level agreements
- **Custom Integration**: API integrations and third-party connections
- **Custom Compliance**: Industry-specific compliance frameworks (HIPAA, SOX, GDPR)
- **Custom Detectors**: Specialized detection models for specific use cases
- **On-Premise Deployment**: Private cloud or on-premise installation options
- **Priority Support**: 24/7 support with guaranteed response times

## Database Schema

### Core Tables

#### 1. white_glove_services
Primary table for managing white-glove service requests and projects.

```sql
CREATE TABLE white_glove_services (
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
```

#### 2. white_glove_milestones
Track project milestones and deliverables.

```sql
CREATE TABLE white_glove_milestones (
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
```

#### 3. white_glove_communications
Track all client communications and interactions.

```sql
CREATE TABLE white_glove_communications (
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
```

#### 4. white_glove_deliverables
Manage project deliverables and documentation.

```sql
CREATE TABLE white_glove_deliverables (
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
```

#### 5. white_glove_feedback
Collect and manage client feedback and satisfaction ratings.

```sql
CREATE TABLE white_glove_feedback (
    id BIGSERIAL PRIMARY KEY,
    service_id VARCHAR(255) REFERENCES white_glove_services(service_id) ON DELETE CASCADE,
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    categories JSONB DEFAULT '{}', -- 'communication', 'technical_quality', 'timeliness', 'value'
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Service Types

### 1. Implementation Services
**Purpose**: Custom deployment and configuration of the Comply-AI platform.

**Typical Milestones**:
- Requirements gathering and analysis
- Custom taxonomy configuration
- Detector setup and configuration
- Integration with existing systems
- User training and documentation
- Go-live support and monitoring

**Deliverables**:
- Implementation plan and timeline
- Custom configuration files
- Integration documentation
- User training materials
- Post-implementation support plan

### 2. Integration Services
**Purpose**: Connect Comply-AI with existing enterprise systems and workflows.

**Typical Milestones**:
- API integration development
- Data pipeline configuration
- Authentication and authorization setup
- Testing and validation
- Production deployment
- Monitoring and maintenance

**Deliverables**:
- Integration code and configuration
- API documentation
- Testing reports
- Deployment guides
- Monitoring dashboards

### 3. Customization Services
**Purpose**: Tailor the platform to specific industry or compliance requirements.

**Typical Milestones**:
- Custom compliance framework development
- Specialized detector training
- Custom reporting and analytics
- Workflow customization
- User interface modifications
- Testing and validation

**Deliverables**:
- Custom compliance frameworks
- Trained detection models
- Custom reports and dashboards
- Modified user interfaces
- Documentation and training materials

### 4. Support Services
**Purpose**: Ongoing support and maintenance for white-glove clients.

**Typical Milestones**:
- Support plan development
- Monitoring setup
- Issue resolution procedures
- Performance optimization
- Regular health checks
- Continuous improvement

**Deliverables**:
- Support procedures and documentation
- Monitoring and alerting setup
- Performance reports
- Optimization recommendations
- Regular status updates

## Workflow Management

### 1. Service Request Creation

```python
async def create_white_glove_service(
    user_id: str,
    tenant_id: str,
    subscription_id: str,
    service_type: str,
    description: str,
    requirements: dict,
    priority: str = "normal"
) -> str:
    """Create a new white-glove service request."""
    
    service_id = str(uuid.uuid4())
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_glove_services (
                service_id, user_id, tenant_id, subscription_id,
                service_type, description, requirements, priority, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """, service_id, user_id, tenant_id, subscription_id,
            service_type, description, json.dumps(requirements), priority, 'requested')
        
        # Create initial communication
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, service_id, 'status_update', 'Service Request Created',
            f'White-glove service request created for {service_type}', user_id)
    
    return service_id
```

### 2. Milestone Management

```python
async def create_milestone(
    service_id: str,
    milestone_name: str,
    description: str,
    estimated_hours: int,
    target_date: datetime,
    assigned_to: str
) -> str:
    """Create a new milestone for a white-glove service."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_glove_milestones (
                service_id, milestone_name, description, estimated_hours,
                target_date, assigned_to, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, service_id, milestone_name, description, estimated_hours,
            target_date, assigned_to, 'pending')
        
        # Log milestone creation
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, service_id, 'status_update', 'Milestone Created',
            f'Milestone "{milestone_name}" created and assigned to {assigned_to}', assigned_to)

async def update_milestone_status(
    milestone_id: int,
    status: str,
    actual_hours: int = None,
    notes: str = None
):
    """Update milestone status and progress."""
    
    async with db_pool.acquire() as conn:
        # Get milestone details
        milestone = await conn.fetchrow("""
            SELECT * FROM white_glove_milestones WHERE id = $1
        """, milestone_id)
        
        # Update milestone
        await conn.execute("""
            UPDATE white_glove_milestones 
            SET status = $1, actual_hours = COALESCE($2, actual_hours), 
                notes = COALESCE($3, notes), updated_at = NOW()
            WHERE id = $4
        """, status, actual_hours, notes, milestone_id)
        
        # Log status update
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, milestone['service_id'], 'status_update', 'Milestone Status Updated',
            f'Milestone "{milestone["milestone_name"]}" status updated to {status}', milestone['assigned_to'])
```

### 3. Communication Tracking

```python
async def log_communication(
    service_id: str,
    communication_type: str,
    subject: str,
    content: str,
    from_user_id: str,
    to_user_id: str = None,
    attachments: list = None
):
    """Log client communication."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content,
                from_user_id, to_user_id, attachments
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, service_id, communication_type, subject, content,
            from_user_id, to_user_id, json.dumps(attachments or []))

async def get_communication_history(service_id: str) -> list:
    """Get communication history for a service."""
    
    async with db_pool.acquire() as conn:
        communications = await conn.fetch("""
            SELECT wc.*, u1.email as from_email, u2.email as to_email
            FROM white_glove_communications wc
            LEFT JOIN users u1 ON wc.from_user_id = u1.user_id
            LEFT JOIN users u2 ON wc.to_user_id = u2.user_id
            WHERE wc.service_id = $1
            ORDER BY wc.created_at DESC
        """, service_id)
        
        return [dict(comm) for comm in communications]
```

### 4. Deliverable Management

```python
async def create_deliverable(
    service_id: str,
    deliverable_name: str,
    deliverable_type: str,
    description: str,
    file_path: str = None
) -> str:
    """Create a new deliverable."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_glove_deliverables (
                service_id, deliverable_name, deliverable_type, description, file_path, status
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, service_id, deliverable_name, deliverable_type, description, file_path, 'pending')
        
        # Log deliverable creation
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, service_id, 'documentation', 'Deliverable Created',
            f'Deliverable "{deliverable_name}" created', 'system')

async def deliver_deliverable(
    deliverable_id: int,
    file_path: str,
    file_size: int,
    mime_type: str
):
    """Mark deliverable as delivered."""
    
    async with db_pool.acquire() as conn:
        # Get deliverable details
        deliverable = await conn.fetchrow("""
            SELECT * FROM white_glove_deliverables WHERE id = $1
        """, deliverable_id)
        
        # Update deliverable
        await conn.execute("""
            UPDATE white_glove_deliverables 
            SET status = 'delivered', file_path = $1, file_size = $2, 
                mime_type = $3, delivered_at = NOW(), updated_at = NOW()
            WHERE id = $4
        """, file_path, file_size, mime_type, deliverable_id)
        
        # Log delivery
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, deliverable['service_id'], 'documentation', 'Deliverable Delivered',
            f'Deliverable "{deliverable["deliverable_name"]}" has been delivered', 'system')
```

## Implementation Examples

### 1. Custom Compliance Framework Implementation

```python
async def implement_custom_compliance_framework(
    user_id: str,
    tenant_id: str,
    framework_name: str,
    requirements: dict
) -> str:
    """Implement a custom compliance framework."""
    
    # Create service request
    service_id = await create_white_glove_service(
        user_id=user_id,
        tenant_id=tenant_id,
        subscription_id=await get_active_subscription(user_id),
        service_type='customization',
        description=f'Custom {framework_name} compliance framework implementation',
        requirements=requirements,
        priority='high'
    )
    
    # Create milestones
    milestones = [
        ('Requirements Analysis', 'Analyze compliance requirements and map to existing framework', 8),
        ('Framework Design', 'Design custom compliance framework structure', 16),
        ('Implementation', 'Implement custom compliance framework', 24),
        ('Testing', 'Test framework with sample data', 8),
        ('Documentation', 'Create user documentation and training materials', 12),
        ('Deployment', 'Deploy framework to production environment', 8),
        ('Training', 'Provide user training and support', 16)
    ]
    
    for milestone_name, description, hours in milestones:
        await create_milestone(
            service_id=service_id,
            milestone_name=milestone_name,
            description=description,
            estimated_hours=hours,
            target_date=datetime.utcnow() + timedelta(days=7),
            assigned_to='compliance_engineer'
        )
    
    # Create deliverables
    deliverables = [
        ('Compliance Framework Design', 'documentation'),
        ('Implementation Code', 'code'),
        ('Configuration Files', 'configuration'),
        ('User Documentation', 'documentation'),
        ('Training Materials', 'training')
    ]
    
    for deliverable_name, deliverable_type in deliverables:
        await create_deliverable(
            service_id=service_id,
            deliverable_name=deliverable_name,
            deliverable_type=deliverable_type,
            description=f'{deliverable_name} for {framework_name} compliance framework'
        )
    
    return service_id
```

### 2. API Integration Service

```python
async def implement_api_integration(
    user_id: str,
    tenant_id: str,
    integration_type: str,
    target_system: str,
    requirements: dict
) -> str:
    """Implement API integration with external system."""
    
    # Create service request
    service_id = await create_white_glove_service(
        user_id=user_id,
        tenant_id=tenant_id,
        subscription_id=await get_active_subscription(user_id),
        service_type='integration',
        description=f'API integration with {target_system}',
        requirements=requirements,
        priority='normal'
    )
    
    # Create milestones
    milestones = [
        ('API Analysis', 'Analyze target system API and requirements', 8),
        ('Authentication Setup', 'Configure authentication and authorization', 4),
        ('Data Mapping', 'Map data between systems', 8),
        ('Integration Development', 'Develop integration code', 16),
        ('Testing', 'Test integration with sample data', 8),
        ('Documentation', 'Create integration documentation', 8),
        ('Deployment', 'Deploy integration to production', 4)
    ]
    
    for milestone_name, description, hours in milestones:
        await create_milestone(
            service_id=service_id,
            milestone_name=milestone_name,
            description=description,
            estimated_hours=hours,
            target_date=datetime.utcnow() + timedelta(days=5),
            assigned_to='integration_engineer'
        )
    
    return service_id
```

## Best Practices

### 1. Service Request Management

- **Clear Requirements**: Always gather detailed requirements before starting
- **Realistic Timelines**: Set achievable milestones and deadlines
- **Regular Communication**: Maintain frequent communication with clients
- **Documentation**: Document all decisions and changes
- **Quality Assurance**: Implement thorough testing and validation

### 2. Milestone Management

- **SMART Goals**: Make milestones Specific, Measurable, Achievable, Relevant, and Time-bound
- **Regular Updates**: Update milestone status regularly
- **Risk Management**: Identify and mitigate potential risks early
- **Resource Allocation**: Ensure adequate resources for each milestone
- **Client Involvement**: Keep clients informed of progress

### 3. Communication Best Practices

- **Regular Updates**: Provide weekly status updates
- **Clear Documentation**: Document all communications
- **Proactive Communication**: Reach out before issues become problems
- **Client Feedback**: Regularly solicit and act on client feedback
- **Escalation Procedures**: Have clear escalation procedures for issues

### 4. Deliverable Management

- **Quality Standards**: Maintain high quality standards for all deliverables
- **Version Control**: Use proper version control for all deliverables
- **Client Approval**: Get client approval before finalizing deliverables
- **Documentation**: Provide comprehensive documentation for all deliverables
- **Training**: Provide training on how to use deliverables

## Monitoring and Analytics

### 1. Service Performance Metrics

```python
async def get_service_metrics(service_id: str) -> dict:
    """Get performance metrics for a white-glove service."""
    
    async with db_pool.acquire() as conn:
        # Get service details
        service = await conn.fetchrow("""
            SELECT * FROM white_glove_services WHERE service_id = $1
        """, service_id)
        
        # Get milestone statistics
        milestone_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_milestones,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_milestones,
                COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress_milestones,
                COUNT(*) FILTER (WHERE status = 'blocked') as blocked_milestones,
                SUM(estimated_hours) as total_estimated_hours,
                SUM(actual_hours) as total_actual_hours
            FROM white_glove_milestones 
            WHERE service_id = $1
        """, service_id)
        
        # Get deliverable statistics
        deliverable_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_deliverables,
                COUNT(*) FILTER (WHERE status = 'delivered') as delivered_deliverables,
                COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress_deliverables
            FROM white_glove_deliverables 
            WHERE service_id = $1
        """, service_id)
        
        # Get communication statistics
        communication_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_communications,
                COUNT(*) FILTER (WHERE communication_type = 'email') as emails,
                COUNT(*) FILTER (WHERE communication_type = 'call') as calls,
                COUNT(*) FILTER (WHERE communication_type = 'meeting') as meetings
            FROM white_glove_communications 
            WHERE service_id = $1
        """, service_id)
        
        return {
            'service': dict(service),
            'milestones': dict(milestone_stats),
            'deliverables': dict(deliverable_stats),
            'communications': dict(communication_stats)
        }
```

### 2. Client Satisfaction Tracking

```python
async def collect_feedback(
    service_id: str,
    user_id: str,
    rating: int,
    feedback_text: str,
    categories: dict
):
    """Collect client feedback for a service."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO white_glove_feedback (
                service_id, user_id, rating, feedback_text, categories
            ) VALUES ($1, $2, $3, $4, $5)
        """, service_id, user_id, rating, feedback_text, json.dumps(categories))
        
        # Log feedback collection
        await conn.execute("""
            INSERT INTO white_glove_communications (
                service_id, communication_type, subject, content, from_user_id
            ) VALUES ($1, $2, $3, $4, $5)
        """, service_id, 'status_update', 'Feedback Collected',
            f'Client feedback collected: {rating}/5 stars', user_id)

async def get_satisfaction_metrics() -> dict:
    """Get overall satisfaction metrics."""
    
    async with db_pool.acquire() as conn:
        # Get overall satisfaction
        overall_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as average_rating,
                COUNT(*) FILTER (WHERE rating >= 4) as satisfied_clients,
                COUNT(*) FILTER (WHERE rating <= 2) as dissatisfied_clients
            FROM white_glove_feedback
        """)
        
        # Get satisfaction by service type
        service_type_stats = await conn.fetch("""
            SELECT 
                wgs.service_type,
                COUNT(wgf.id) as feedback_count,
                AVG(wgf.rating) as average_rating
            FROM white_glove_services wgs
            LEFT JOIN white_glove_feedback wgf ON wgs.service_id = wgf.service_id
            GROUP BY wgs.service_type
        """)
        
        return {
            'overall': dict(overall_stats),
            'by_service_type': [dict(stat) for stat in service_type_stats]
        }
```

### 3. Resource Utilization Tracking

```python
async def track_resource_utilization(assigned_to: str, start_date: datetime, end_date: datetime) -> dict:
    """Track resource utilization for a team member."""
    
    async with db_pool.acquire() as conn:
        # Get assigned services
        services = await conn.fetch("""
            SELECT 
                service_id, service_type, status, estimated_hours, actual_hours
            FROM white_glove_services 
            WHERE assigned_to = $1 AND created_at BETWEEN $2 AND $3
        """, assigned_to, start_date, end_date)
        
        # Get milestone hours
        milestone_hours = await conn.fetchrow("""
            SELECT 
                SUM(estimated_hours) as total_estimated_hours,
                SUM(actual_hours) as total_actual_hours,
                COUNT(*) as total_milestones
            FROM white_glove_milestones 
            WHERE assigned_to = $1 AND created_at BETWEEN $2 AND $3
        """, assigned_to, start_date, end_date)
        
        return {
            'services': [dict(service) for service in services],
            'milestone_hours': dict(milestone_hours),
            'utilization_rate': (milestone_hours['total_actual_hours'] or 0) / (milestone_hours['total_estimated_hours'] or 1) * 100
        }
```

## Summary

The white-glove service management system provides:

- **Comprehensive Service Management**: Complete lifecycle management for white-glove services
- **Project Tracking**: Milestone and deliverable management
- **Communication Tracking**: All client interactions and documentation
- **Quality Assurance**: Feedback collection and satisfaction monitoring
- **Resource Management**: Team member utilization and workload tracking
- **Analytics and Reporting**: Performance metrics and insights

This system enables the delivery of premium, customized services while maintaining high quality standards and client satisfaction. The database schema supports the complete white-glove service lifecycle from initial request through project completion and ongoing support.
