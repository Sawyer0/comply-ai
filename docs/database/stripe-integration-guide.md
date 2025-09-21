# Stripe Integration Guide for Comply-AI Platform

This guide provides comprehensive instructions for integrating Stripe billing with the Comply-AI platform's database schema.

## Table of Contents

1. [Overview](#overview)
2. [Stripe Setup](#stripe-setup)
3. [Database Integration](#database-integration)
4. [Free Tier Implementation](#free-tier-implementation)
5. [Billing Workflows](#billing-workflows)
6. [Usage Tracking](#usage-tracking)
7. [Promotional Codes](#promotional-codes)
8. [Webhook Handling](#webhook-handling)
9. [Testing](#testing)
10. [Production Deployment](#production-deployment)

## Overview

The Comply-AI platform uses Stripe for subscription billing with the following features:

- **Free Tier**: 1,000 API calls/month, 1GB storage, 500 detector runs, 10 analysis runs
- **Paid Plans**: Starter ($29/month), Professional ($99/month), Enterprise ($299/month)
- **Stripe Integration**: Full subscription management, payment processing, webhooks
- **Usage Tracking**: Real-time usage monitoring and billing
- **Promotional Codes**: Discount codes and free trials

## Stripe Setup

### 1. Create Stripe Account and Products

```bash
# Install Stripe CLI
curl -s https://packages.stripe.dev/api/security/keypair/stripe-cli-gpg/public | gpg --dearmor | sudo tee /usr/share/keyrings/stripe.gpg
echo "deb [signed-by=/usr/share/keyrings/stripe.gpg] https://packages.stripe.dev/stripe-cli-debian-local stable main" | sudo tee -a /etc/apt/sources.list.d/stripe.list
sudo apt update
sudo apt install stripe

# Login to Stripe
stripe login

# Create products and prices
stripe products create --name "Comply-AI Starter Plan" --description "Starter plan for small teams"
stripe products create --name "Comply-AI Professional Plan" --description "Professional plan for growing teams"
stripe products create --name "Comply-AI Enterprise Plan" --description "Enterprise plan for large organizations"
```

### 2. Create Stripe Prices

```bash
# Starter Plan - Monthly
stripe prices create \
  --product prod_starter \
  --unit-amount 2900 \
  --currency usd \
  --recurring interval=month

# Starter Plan - Yearly
stripe prices create \
  --product prod_starter \
  --unit-amount 29000 \
  --currency usd \
  --recurring interval=year

# Professional Plan - Monthly
stripe prices create \
  --product prod_professional \
  --unit-amount 9900 \
  --currency usd \
  --recurring interval=month

# Professional Plan - Yearly
stripe prices create \
  --product prod_professional \
  --unit-amount 99000 \
  --currency usd \
  --recurring interval=year

# Enterprise Plan - Monthly
stripe prices create \
  --product prod_enterprise \
  --unit-amount 29900 \
  --currency usd \
  --recurring interval=month

# Enterprise Plan - Yearly
stripe prices create \
  --product prod_enterprise \
  --unit-amount 299000 \
  --currency usd \
  --recurring interval=year
```

### 3. Update Database with Stripe IDs

```sql
-- Update billing_plans with Stripe IDs
UPDATE billing_plans SET 
    stripe_price_id = 'price_starter_monthly',
    stripe_product_id = 'prod_starter'
WHERE plan_id = 'starter';

UPDATE billing_plans SET 
    stripe_price_id = 'price_professional_monthly',
    stripe_product_id = 'prod_professional'
WHERE plan_id = 'professional';

UPDATE billing_plans SET 
    stripe_price_id = 'price_enterprise_monthly',
    stripe_product_id = 'prod_enterprise'
WHERE plan_id = 'enterprise';
```

## Database Integration

### 1. User Registration Flow

```python
import stripe
from datetime import datetime, timedelta
import uuid

async def create_user_with_free_tier(email: str, password_hash: str, tenant_id: str):
    """Create user with free tier subscription."""
    
    # Create user
    user_id = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO users (user_id, email, password_hash, is_active, is_verified)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, email, password_hash, True, False)
        
        # Create Stripe customer
        stripe_customer = stripe.Customer.create(
            email=email,
            metadata={
                'user_id': user_id,
                'tenant_id': tenant_id
            }
        )
        
        # Create free tier subscription
        subscription_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO user_subscriptions (
                subscription_id, user_id, tenant_id, plan_id, status,
                current_period_start, current_period_end,
                stripe_customer_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, subscription_id, user_id, tenant_id, 'free', 'active',
            datetime.utcnow(), datetime.utcnow() + timedelta(days=30),
            stripe_customer.id)
        
        # Initialize free tier usage limits
        free_tier_limits = [
            ('api_calls', 1000),
            ('storage_gb', 1),
            ('detector_runs', 500),
            ('analysis_runs', 10)
        ]
        
        for usage_type, limit in free_tier_limits:
            await conn.execute("""
                INSERT INTO free_tier_usage (
                    user_id, tenant_id, usage_type, usage_limit,
                    next_reset_at
                ) VALUES ($1, $2, $3, $4, $5)
            """, user_id, tenant_id, usage_type, limit,
                datetime.utcnow() + timedelta(days=30))
        
        # Assign default user role
        await conn.execute("""
            INSERT INTO user_role_assignments (user_id, role_name, tenant_id)
            VALUES ($1, $2, $3)
        """, user_id, 'user', tenant_id)
    
    return user_id
```

### 2. Subscription Upgrade Flow

```python
async def upgrade_subscription(user_id: str, plan_id: str, payment_method_id: str):
    """Upgrade user subscription to paid plan."""
    
    async with db_pool.acquire() as conn:
        # Get current subscription
        current_sub = await conn.fetchrow("""
            SELECT * FROM user_subscriptions 
            WHERE user_id = $1 AND status = 'active'
        """, user_id)
        
        if not current_sub:
            raise ValueError("No active subscription found")
        
        # Get billing plan details
        plan = await conn.fetchrow("""
            SELECT * FROM billing_plans WHERE plan_id = $1
        """, plan_id)
        
        # Create Stripe subscription
        stripe_subscription = stripe.Subscription.create(
            customer=current_sub['stripe_customer_id'],
            items=[{
                'price': plan['stripe_price_id']
            }],
            default_payment_method=payment_method_id,
            metadata={
                'user_id': user_id,
                'tenant_id': current_sub['tenant_id'],
                'plan_id': plan_id
            }
        )
        
        # Update subscription in database
        new_subscription_id = str(uuid.uuid4())
        await conn.execute("""
            UPDATE user_subscriptions SET 
                status = 'canceled',
                canceled_at = NOW()
            WHERE subscription_id = $1
        """, current_sub['subscription_id'])
        
        await conn.execute("""
            INSERT INTO user_subscriptions (
                subscription_id, user_id, tenant_id, plan_id, status,
                billing_cycle, current_period_start, current_period_end,
                stripe_subscription_id, stripe_customer_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, new_subscription_id, user_id, current_sub['tenant_id'], 
            plan_id, 'active', 'monthly',
            datetime.fromtimestamp(stripe_subscription.current_period_start),
            datetime.fromtimestamp(stripe_subscription.current_period_end),
            stripe_subscription.id, current_sub['stripe_customer_id'])
        
        # Remove free tier usage limits
        await conn.execute("""
            DELETE FROM free_tier_usage WHERE user_id = $1
        """, user_id)
    
    return new_subscription_id
```

## Free Tier Implementation

### 1. Usage Tracking

```python
async def track_usage(user_id: str, tenant_id: str, usage_type: str, amount: float):
    """Track usage for billing and free tier limits."""
    
    async with db_pool.acquire() as conn:
        # Check if user is on free tier
        subscription = await conn.fetchrow("""
            SELECT plan_id FROM user_subscriptions 
            WHERE user_id = $1 AND status = 'active'
        """, user_id)
        
        if subscription['plan_id'] == 'free':
            # Check free tier limits
            usage = await conn.fetchrow("""
                SELECT current_usage, usage_limit, next_reset_at
                FROM free_tier_usage 
                WHERE user_id = $1 AND tenant_id = $2 AND usage_type = $3
            """, user_id, tenant_id, usage_type)
            
            if usage:
                # Check if limit exceeded
                if usage['current_usage'] + amount > usage['usage_limit']:
                    raise ValueError(f"Free tier limit exceeded for {usage_type}")
                
                # Update usage
                await conn.execute("""
                    UPDATE free_tier_usage 
                    SET current_usage = current_usage + $1
                    WHERE user_id = $2 AND tenant_id = $3 AND usage_type = $4
                """, amount, user_id, tenant_id, usage_type)
        
        # Record usage for billing
        await conn.execute("""
            INSERT INTO usage_records (
                user_id, tenant_id, subscription_id, usage_type,
                usage_amount, usage_unit, billing_period_start, billing_period_end
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, user_id, tenant_id, subscription['subscription_id'], usage_type,
            amount, 'count', 
            datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + 
             timedelta(days=32)).replace(day=1) - timedelta(days=1))
```

### 2. Free Tier Reset

```python
async def reset_free_tier_usage():
    """Reset free tier usage at the beginning of each month."""
    
    async with db_pool.acquire() as conn:
        # Get all free tier users
        free_users = await conn.fetch("""
            SELECT DISTINCT user_id, tenant_id FROM user_subscriptions 
            WHERE plan_id = 'free' AND status = 'active'
        """)
        
        for user in free_users:
            # Reset usage for each usage type
            await conn.execute("""
                UPDATE free_tier_usage 
                SET current_usage = 0, last_reset_at = NOW(), 
                    next_reset_at = NOW() + INTERVAL '1 month'
                WHERE user_id = $1 AND tenant_id = $2
            """, user['user_id'], user['tenant_id'])
```

## Billing Workflows

### 1. Invoice Generation

```python
async def generate_monthly_invoices():
    """Generate monthly invoices for paid subscriptions."""
    
    async with db_pool.acquire() as conn:
        # Get active paid subscriptions
        subscriptions = await conn.fetch("""
            SELECT us.*, bp.price_monthly, bp.currency
            FROM user_subscriptions us
            JOIN billing_plans bp ON us.plan_id = bp.plan_id
            WHERE us.status = 'active' AND bp.plan_type = 'paid'
            AND us.current_period_end <= NOW()
        """)
        
        for sub in subscriptions:
            # Create Stripe invoice
            stripe_invoice = stripe.Invoice.create(
                customer=sub['stripe_customer_id'],
                subscription=sub['stripe_subscription_id'],
                auto_advance=True
            )
            
            # Record invoice in database
            invoice_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO billing_invoices (
                    invoice_id, user_id, tenant_id, subscription_id,
                    invoice_number, status, amount_due, currency,
                    invoice_date, due_date, stripe_invoice_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, invoice_id, sub['user_id'], sub['tenant_id'], sub['subscription_id'],
                stripe_invoice.number, 'open', sub['price_monthly'], sub['currency'],
                datetime.utcnow(), datetime.utcnow() + timedelta(days=30),
                stripe_invoice.id)
```

### 2. Payment Processing

```python
async def process_payment(invoice_id: str):
    """Process payment for an invoice."""
    
    async with db_pool.acquire() as conn:
        # Get invoice details
        invoice = await conn.fetchrow("""
            SELECT * FROM billing_invoices WHERE invoice_id = $1
        """, invoice_id)
        
        # Finalize and pay Stripe invoice
        stripe_invoice = stripe.Invoice.finalize_invoice(invoice['stripe_invoice_id'])
        stripe_invoice = stripe.Invoice.pay(invoice['stripe_invoice_id'])
        
        # Update invoice status
        await conn.execute("""
            UPDATE billing_invoices 
            SET status = 'paid', amount_paid = $1, paid_at = NOW(),
                stripe_payment_intent_id = $2
            WHERE invoice_id = $3
        """, invoice['amount_due'], stripe_invoice.payment_intent, invoice_id)
```

## Usage Tracking

### 1. API Call Tracking

```python
async def track_api_call(user_id: str, tenant_id: str, endpoint: str):
    """Track API call usage."""
    await track_usage(user_id, tenant_id, 'api_calls', 1)
    
    # Log API call for audit
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO audit_logs (
                tenant_id, user_id, action, resource_type, resource_id,
                details, ip_address, user_agent
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, tenant_id, user_id, 'api_call', 'endpoint', endpoint,
            json.dumps({'endpoint': endpoint}), None, None)
```

### 2. Storage Usage Tracking

```python
async def track_storage_usage(user_id: str, tenant_id: str, bytes_used: int):
    """Track storage usage in GB."""
    gb_used = bytes_used / (1024 ** 3)
    await track_usage(user_id, tenant_id, 'storage_gb', gb_used)
```

### 3. Detector Run Tracking

```python
async def track_detector_run(user_id: str, tenant_id: str, detector_name: str):
    """Track detector run usage."""
    await track_usage(user_id, tenant_id, 'detector_runs', 1)
```

## Promotional Codes

### 1. Apply Promotional Code

```python
async def apply_promotional_code(user_id: str, code: str, subscription_id: str):
    """Apply promotional code to subscription."""
    
    async with db_pool.acquire() as conn:
        # Validate promotional code
        promo_code = await conn.fetchrow("""
            SELECT * FROM promotional_codes 
            WHERE code = $1 AND is_active = TRUE 
            AND valid_from <= NOW() AND (valid_until IS NULL OR valid_until >= NOW())
            AND (max_uses IS NULL OR current_uses < max_uses)
        """, code)
        
        if not promo_code:
            raise ValueError("Invalid or expired promotional code")
        
        # Apply discount to Stripe subscription
        stripe_subscription = stripe.Subscription.retrieve(subscription_id)
        
        if promo_code['discount_type'] == 'percentage':
            discount = stripe.Coupon.create(
                percent_off=promo_code['discount_value'],
                duration='once'
            )
        elif promo_code['discount_type'] == 'fixed_amount':
            discount = stripe.Coupon.create(
                amount_off=int(promo_code['discount_value'] * 100),
                currency='usd',
                duration='once'
            )
        
        stripe.Subscription.modify(
            subscription_id,
            coupon=discount.id
        )
        
        # Record usage
        await conn.execute("""
            INSERT INTO promotional_code_usage (
                code, user_id, subscription_id, used_at, discount_applied
            ) VALUES ($1, $2, $3, $4, $5)
        """, code, user_id, subscription_id, datetime.utcnow(), 
            promo_code['discount_value'])
        
        # Update usage count
        await conn.execute("""
            UPDATE promotional_codes 
            SET current_uses = current_uses + 1
            WHERE code = $1
        """, code)
```

## Webhook Handling

### 1. Stripe Webhook Setup

```python
from fastapi import FastAPI, Request
import stripe

app = FastAPI()

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks."""
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return {"error": "Invalid payload"}, 400
    except stripe.error.SignatureVerificationError:
        return {"error": "Invalid signature"}, 400
    
    # Handle different event types
    if event['type'] == 'invoice.payment_succeeded':
        await handle_invoice_payment_succeeded(event['data']['object'])
    elif event['type'] == 'invoice.payment_failed':
        await handle_invoice_payment_failed(event['data']['object'])
    elif event['type'] == 'customer.subscription.updated':
        await handle_subscription_updated(event['data']['object'])
    elif event['type'] == 'customer.subscription.deleted':
        await handle_subscription_deleted(event['data']['object'])
    
    return {"status": "success"}
```

### 2. Webhook Handlers

```python
async def handle_invoice_payment_succeeded(invoice):
    """Handle successful invoice payment."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE billing_invoices 
            SET status = 'paid', amount_paid = $1, paid_at = NOW()
            WHERE stripe_invoice_id = $2
        """, invoice['amount_paid'] / 100, invoice['id'])

async def handle_invoice_payment_failed(invoice):
    """Handle failed invoice payment."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE billing_invoices 
            SET status = 'uncollectible'
            WHERE stripe_invoice_id = $1
        """, invoice['id'])
        
        # Update subscription status
        await conn.execute("""
            UPDATE user_subscriptions 
            SET status = 'past_due'
            WHERE stripe_subscription_id = $1
        """, invoice['subscription'])

async def handle_subscription_updated(subscription):
    """Handle subscription updates."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE user_subscriptions 
            SET status = $1, current_period_start = $2, current_period_end = $3
            WHERE stripe_subscription_id = $4
        """, subscription['status'],
            datetime.fromtimestamp(subscription['current_period_start']),
            datetime.fromtimestamp(subscription['current_period_end']),
            subscription['id'])

async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation."""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE user_subscriptions 
            SET status = 'canceled', canceled_at = NOW()
            WHERE stripe_subscription_id = $1
        """, subscription['id'])
```

## Testing

### 1. Test Stripe Integration

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_user_registration_with_free_tier():
    """Test user registration with free tier setup."""
    
    with patch('stripe.Customer.create') as mock_customer:
        mock_customer.return_value = MagicMock(id='cus_test123')
        
        user_id = await create_user_with_free_tier(
            'test@example.com', 'hashed_password', 'tenant123'
        )
        
        assert user_id is not None
        
        # Verify user was created
        async with db_pool.acquire() as conn:
            user = await conn.fetchrow("""
                SELECT * FROM users WHERE user_id = $1
            """, user_id)
            assert user['email'] == 'test@example.com'
            
            # Verify free tier subscription
            subscription = await conn.fetchrow("""
                SELECT * FROM user_subscriptions WHERE user_id = $1
            """, user_id)
            assert subscription['plan_id'] == 'free'
            assert subscription['status'] == 'active'
            
            # Verify free tier usage limits
            usage_limits = await conn.fetch("""
                SELECT * FROM free_tier_usage WHERE user_id = $1
            """, user_id)
            assert len(usage_limits) == 4  # api_calls, storage_gb, detector_runs, analysis_runs

@pytest.mark.asyncio
async def test_usage_tracking():
    """Test usage tracking for free tier."""
    
    # Create test user
    user_id = await create_test_user()
    
    # Track API call
    await track_usage(user_id, 'tenant123', 'api_calls', 1)
    
    # Verify usage was recorded
    async with db_pool.acquire() as conn:
        usage = await conn.fetchrow("""
            SELECT current_usage FROM free_tier_usage 
            WHERE user_id = $1 AND usage_type = 'api_calls'
        """, user_id)
        assert usage['current_usage'] == 1

@pytest.mark.asyncio
async def test_usage_limit_exceeded():
    """Test free tier usage limit enforcement."""
    
    user_id = await create_test_user()
    
    # Exceed API call limit
    with pytest.raises(ValueError, match="Free tier limit exceeded"):
        await track_usage(user_id, 'tenant123', 'api_calls', 1001)
```

### 2. Test Promotional Codes

```python
@pytest.mark.asyncio
async def test_promotional_code_application():
    """Test promotional code application."""
    
    user_id = await create_test_user()
    subscription_id = await create_test_subscription(user_id)
    
    with patch('stripe.Coupon.create') as mock_coupon:
        mock_coupon.return_value = MagicMock(id='coupon_test123')
        
        await apply_promotional_code(user_id, 'WELCOME20', subscription_id)
        
        # Verify usage was recorded
        async with db_pool.acquire() as conn:
            usage = await conn.fetchrow("""
                SELECT * FROM promotional_code_usage 
                WHERE code = 'WELCOME20' AND user_id = $1
            """, user_id)
            assert usage is not None
            assert usage['discount_applied'] == 20.00
```

## Production Deployment

### 1. Environment Variables

```bash
# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database

# Azure Configuration
AZURE_STORAGE_ACCOUNT=your-storage-account
AZURE_STORAGE_KEY=your-storage-key
```

### 2. Webhook Endpoint Setup

```bash
# Install Stripe CLI
stripe listen --forward-to https://your-domain.com/webhooks/stripe

# Get webhook secret
stripe webhook_endpoints create \
  --url https://your-domain.com/webhooks/stripe \
  --enabled-events invoice.payment_succeeded \
  --enabled-events invoice.payment_failed \
  --enabled-events customer.subscription.updated \
  --enabled-events customer.subscription.deleted
```

### 3. Monitoring and Alerting

```python
# Set up monitoring for billing events
async def monitor_billing_health():
    """Monitor billing system health."""
    
    async with db_pool.acquire() as conn:
        # Check for failed payments
        failed_payments = await conn.fetch("""
            SELECT COUNT(*) as count FROM billing_invoices 
            WHERE status = 'uncollectible' AND created_at >= NOW() - INTERVAL '24 hours'
        """)
        
        if failed_payments[0]['count'] > 10:
            # Send alert
            await send_alert("High number of failed payments detected")
        
        # Check for expired subscriptions
        expired_subs = await conn.fetch("""
            SELECT COUNT(*) as count FROM user_subscriptions 
            WHERE status = 'past_due' AND current_period_end < NOW() - INTERVAL '7 days'
        """)
        
        if expired_subs[0]['count'] > 5:
            # Send alert
            await send_alert("Multiple expired subscriptions need attention")
```

## Summary

This Stripe integration provides:

- **Complete Billing System**: Free tier, paid plans, usage tracking
- **Stripe Integration**: Full subscription management, payment processing
- **Usage Monitoring**: Real-time usage tracking and limit enforcement
- **Promotional Codes**: Discount codes and free trials
- **Webhook Handling**: Automated billing event processing
- **Testing**: Comprehensive test coverage
- **Production Ready**: Monitoring, alerting, and error handling

The system supports the complete billing lifecycle from user registration through subscription management, usage tracking, and payment processing, all integrated with the comprehensive database schema.
