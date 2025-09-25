"""
Billing Management System for Mapper Service

Provides comprehensive billing functionality including:
- Usage-based billing calculations
- Invoice generation
- Payment tracking
- Billing analytics and reporting
- Integration with cost monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
from decimal import Decimal, ROUND_HALF_UP
import json
from calendar import monthrange

from shared.interfaces.cost_monitoring import CostEvent
from .tenant_manager import MapperTenantManager
from .cost_tracker import MapperCostTracker

logger = logging.getLogger(__name__)


class BillingPeriod(str, Enum):
    """Billing period types"""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    USAGE_BASED = "usage_based"


class InvoiceStatus(str, Enum):
    """Invoice status types"""

    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    """Payment method types"""

    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    INVOICE = "invoice"
    CREDITS = "credits"


@dataclass
class BillingTier:
    """Billing tier configuration"""

    tier_name: str
    monthly_base_fee: Decimal
    included_requests: int
    overage_rate_per_request: Decimal
    included_tokens: int
    overage_rate_per_token: Decimal
    included_storage_gb: int
    overage_rate_per_gb: Decimal
    support_level: str
    features: List[str]


@dataclass
class UsageLineItem:
    """Individual usage line item for billing"""

    item_type: str
    description: str
    quantity: Decimal
    unit_price: Decimal
    total_amount: Decimal
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any] = None


@dataclass
class Invoice:
    """Invoice data structure"""

    invoice_id: str
    tenant_id: str
    billing_period: BillingPeriod
    period_start: datetime
    period_end: datetime
    line_items: List[UsageLineItem]
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    currency: str
    status: InvoiceStatus
    due_date: datetime
    created_at: datetime
    paid_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class BillingManager:
    """Comprehensive billing management system"""

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        tenant_manager: MapperTenantManager,
        cost_tracker: MapperCostTracker,
    ):
        self.db_pool = db_pool
        self.tenant_manager = tenant_manager
        self.cost_tracker = cost_tracker
        self.billing_tiers = self._load_billing_tiers()
        self.tax_rates = self._load_tax_rates()

    def _load_billing_tiers(self) -> Dict[str, BillingTier]:
        """Load billing tier configurations"""
        return {
            "free": BillingTier(
                tier_name="free",
                monthly_base_fee=Decimal("0.00"),
                included_requests=1000,
                overage_rate_per_request=Decimal("0.001"),
                included_tokens=50000,
                overage_rate_per_token=Decimal("0.0001"),
                included_storage_gb=1,
                overage_rate_per_gb=Decimal("0.10"),
                support_level="community",
                features=["basic_mapping", "standard_models"],
            ),
            "basic": BillingTier(
                tier_name="basic",
                monthly_base_fee=Decimal("29.00"),
                included_requests=10000,
                overage_rate_per_request=Decimal("0.0008"),
                included_tokens=500000,
                overage_rate_per_token=Decimal("0.00008"),
                included_storage_gb=10,
                overage_rate_per_gb=Decimal("0.08"),
                support_level="email",
                features=["basic_mapping", "standard_models", "custom_thresholds"],
            ),
            "premium": BillingTier(
                tier_name="premium",
                monthly_base_fee=Decimal("99.00"),
                included_requests=50000,
                overage_rate_per_request=Decimal("0.0006"),
                included_tokens=2500000,
                overage_rate_per_token=Decimal("0.00006"),
                included_storage_gb=50,
                overage_rate_per_gb=Decimal("0.06"),
                support_level="priority",
                features=[
                    "advanced_mapping",
                    "all_models",
                    "custom_models",
                    "analytics",
                ],
            ),
            "enterprise": BillingTier(
                tier_name="enterprise",
                monthly_base_fee=Decimal("299.00"),
                included_requests=200000,
                overage_rate_per_request=Decimal("0.0004"),
                included_tokens=10000000,
                overage_rate_per_token=Decimal("0.00004"),
                included_storage_gb=200,
                overage_rate_per_gb=Decimal("0.04"),
                support_level="dedicated",
                features=[
                    "enterprise_mapping",
                    "all_models",
                    "custom_models",
                    "advanced_analytics",
                    "sla",
                ],
            ),
        }

    def _load_tax_rates(self) -> Dict[str, Decimal]:
        """Load tax rates by region"""
        return {
            "US": Decimal("0.08"),  # Average US sales tax
            "EU": Decimal("0.20"),  # EU VAT
            "UK": Decimal("0.20"),  # UK VAT
            "CA": Decimal("0.13"),  # Canadian GST/HST
            "default": Decimal("0.00"),
        }

    async def calculate_monthly_bill(
        self, tenant_id: str, billing_month: datetime
    ) -> Invoice:
        """Calculate monthly bill for tenant"""
        # Get tenant configuration
        tenant = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Get billing tier
        tier = self.billing_tiers.get(tenant.tier, self.billing_tiers["free"])

        # Calculate billing period
        year = billing_month.year
        month = billing_month.month
        period_start = datetime(year, month, 1)
        _, last_day = monthrange(year, month)
        period_end = datetime(year, month, last_day, 23, 59, 59)

        # Get usage data for the period
        usage_data = await self._get_usage_data(tenant_id, period_start, period_end)

        # Calculate line items
        line_items = []

        # Base subscription fee
        if tier.monthly_base_fee > 0:
            line_items.append(
                UsageLineItem(
                    item_type="subscription",
                    description=f"{tier.tier_name.title()} Plan - Monthly Subscription",
                    quantity=Decimal("1"),
                    unit_price=tier.monthly_base_fee,
                    total_amount=tier.monthly_base_fee,
                    period_start=period_start,
                    period_end=period_end,
                    metadata={"tier": tier.tier_name},
                )
            )

        # Request overage charges
        total_requests = usage_data.get("total_requests", 0)
        if total_requests > tier.included_requests:
            overage_requests = total_requests - tier.included_requests
            overage_amount = (
                Decimal(str(overage_requests)) * tier.overage_rate_per_request
            )
            line_items.append(
                UsageLineItem(
                    item_type="request_overage",
                    description=f"Request Overage ({overage_requests:,} requests)",
                    quantity=Decimal(str(overage_requests)),
                    unit_price=tier.overage_rate_per_request,
                    total_amount=overage_amount,
                    period_start=period_start,
                    period_end=period_end,
                    metadata={
                        "included_requests": tier.included_requests,
                        "total_requests": total_requests,
                        "overage_requests": overage_requests,
                    },
                )
            )

        # Token overage charges
        total_tokens = usage_data.get("total_tokens", 0)
        if total_tokens > tier.included_tokens:
            overage_tokens = total_tokens - tier.included_tokens
            overage_amount = Decimal(str(overage_tokens)) * tier.overage_rate_per_token
            line_items.append(
                UsageLineItem(
                    item_type="token_overage",
                    description=f"Token Overage ({overage_tokens:,} tokens)",
                    quantity=Decimal(str(overage_tokens)),
                    unit_price=tier.overage_rate_per_token,
                    total_amount=overage_amount,
                    period_start=period_start,
                    period_end=period_end,
                    metadata={
                        "included_tokens": tier.included_tokens,
                        "total_tokens": total_tokens,
                        "overage_tokens": overage_tokens,
                    },
                )
            )

        # Storage overage charges
        avg_storage_gb = usage_data.get("avg_storage_gb", 0)
        if avg_storage_gb > tier.included_storage_gb:
            overage_storage = avg_storage_gb - tier.included_storage_gb
            overage_amount = Decimal(str(overage_storage)) * tier.overage_rate_per_gb
            line_items.append(
                UsageLineItem(
                    item_type="storage_overage",
                    description=f"Storage Overage ({overage_storage:.2f} GB)",
                    quantity=Decimal(str(overage_storage)),
                    unit_price=tier.overage_rate_per_gb,
                    total_amount=overage_amount,
                    period_start=period_start,
                    period_end=period_end,
                    metadata={
                        "included_storage_gb": tier.included_storage_gb,
                        "avg_storage_gb": avg_storage_gb,
                        "overage_storage_gb": overage_storage,
                    },
                )
            )

        # Calculate totals
        subtotal = sum(item.total_amount for item in line_items)

        # Calculate tax
        tax_rate = self._get_tax_rate(tenant_id)
        tax_amount = (subtotal * tax_rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        total_amount = subtotal + tax_amount

        # Create invoice
        invoice_id = f"INV-{tenant_id}-{year:04d}{month:02d}"
        due_date = period_end + timedelta(days=30)  # 30 days payment terms

        invoice = Invoice(
            invoice_id=invoice_id,
            tenant_id=tenant_id,
            billing_period=BillingPeriod.MONTHLY,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items,
            subtotal=subtotal,
            tax_amount=tax_amount,
            total_amount=total_amount,
            currency="USD",
            status=InvoiceStatus.DRAFT,
            due_date=due_date,
            created_at=datetime.utcnow(),
            metadata={
                "tier": tier.tier_name,
                "tax_rate": str(tax_rate),
                "usage_data": usage_data,
            },
        )

        return invoice

    async def generate_invoice(self, tenant_id: str, billing_month: datetime) -> str:
        """Generate and store invoice"""
        invoice = await self.calculate_monthly_bill(tenant_id, billing_month)

        # Store invoice in database
        query = """
        INSERT INTO invoices (
            invoice_id, tenant_id, billing_period, period_start, period_end,
            line_items, subtotal, tax_amount, total_amount, currency,
            status, due_date, created_at, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        RETURNING invoice_id
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow(
                query,
                invoice.invoice_id,
                invoice.tenant_id,
                invoice.billing_period.value,
                invoice.period_start,
                invoice.period_end,
                json.dumps([asdict(item) for item in invoice.line_items]),
                str(invoice.subtotal),
                str(invoice.tax_amount),
                str(invoice.total_amount),
                invoice.currency,
                invoice.status.value,
                invoice.due_date,
                invoice.created_at,
                json.dumps(invoice.metadata),
            )

        logger.info(f"Generated invoice {invoice.invoice_id} for tenant {tenant_id}")
        return result["invoice_id"]

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID"""
        query = """
        SELECT * FROM invoices WHERE invoice_id = $1
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, invoice_id)

        if not row:
            return None

        # Parse line items
        line_items = []
        for item_data in json.loads(row["line_items"]):
            # Convert datetime strings back to datetime objects
            item_data["period_start"] = datetime.fromisoformat(
                item_data["period_start"]
            )
            item_data["period_end"] = datetime.fromisoformat(item_data["period_end"])
            # Convert decimal strings back to Decimal objects
            item_data["quantity"] = Decimal(item_data["quantity"])
            item_data["unit_price"] = Decimal(item_data["unit_price"])
            item_data["total_amount"] = Decimal(item_data["total_amount"])
            line_items.append(UsageLineItem(**item_data))

        return Invoice(
            invoice_id=row["invoice_id"],
            tenant_id=row["tenant_id"],
            billing_period=BillingPeriod(row["billing_period"]),
            period_start=row["period_start"],
            period_end=row["period_end"],
            line_items=line_items,
            subtotal=Decimal(row["subtotal"]),
            tax_amount=Decimal(row["tax_amount"]),
            total_amount=Decimal(row["total_amount"]),
            currency=row["currency"],
            status=InvoiceStatus(row["status"]),
            due_date=row["due_date"],
            created_at=row["created_at"],
            paid_at=row["paid_at"],
            metadata=json.loads(row["metadata"]),
        )

    async def mark_invoice_paid(
        self, invoice_id: str, payment_method: PaymentMethod, transaction_id: str = None
    ) -> bool:
        """Mark invoice as paid"""
        query = """
        UPDATE invoices 
        SET status = $2, paid_at = $3, 
            metadata = jsonb_set(metadata, '{payment}', $4)
        WHERE invoice_id = $1 AND status != 'paid'
        """

        payment_info = {
            "payment_method": payment_method.value,
            "transaction_id": transaction_id,
            "paid_at": datetime.utcnow().isoformat(),
        }

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                query,
                invoice_id,
                InvoiceStatus.PAID.value,
                datetime.utcnow(),
                json.dumps(payment_info),
            )

        success = result != "UPDATE 0"
        if success:
            logger.info(
                f"Marked invoice {invoice_id} as paid via {payment_method.value}"
            )

        return success

    async def get_tenant_invoices(
        self, tenant_id: str, limit: int = 50
    ) -> List[Invoice]:
        """Get invoices for tenant"""
        query = """
        SELECT * FROM invoices 
        WHERE tenant_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id, limit)

        invoices = []
        for row in rows:
            invoice = await self.get_invoice(row["invoice_id"])
            if invoice:
                invoices.append(invoice)

        return invoices

    async def get_billing_analytics(
        self, tenant_id: str, months: int = 12
    ) -> Dict[str, Any]:
        """Get billing analytics for tenant"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=months * 30)

        # Get invoice summary
        invoice_query = """
        SELECT 
            DATE_TRUNC('month', period_start) as month,
            COUNT(*) as invoice_count,
            SUM(total_amount::decimal) as total_amount,
            AVG(total_amount::decimal) as avg_amount,
            status
        FROM invoices 
        WHERE tenant_id = $1 AND created_at >= $2
        GROUP BY DATE_TRUNC('month', period_start), status
        ORDER BY month DESC
        """

        # Get usage trends
        usage_query = """
        SELECT 
            DATE_TRUNC('month', timestamp) as month,
            event_type,
            COUNT(*) as event_count,
            SUM(cost_amount::decimal) as total_cost,
            SUM((metadata->>'total_tokens')::int) as total_tokens
        FROM mapper_cost_events 
        WHERE tenant_id = $1 AND timestamp >= $2
        GROUP BY DATE_TRUNC('month', timestamp), event_type
        ORDER BY month DESC
        """

        async with self.db_pool.acquire() as conn:
            invoice_rows = await conn.fetch(invoice_query, tenant_id, start_date)
            usage_rows = await conn.fetch(usage_query, tenant_id, start_date)

        # Calculate metrics
        total_billed = sum(Decimal(row["total_amount"]) for row in invoice_rows)
        total_paid = sum(
            Decimal(row["total_amount"])
            for row in invoice_rows
            if row["status"] == "paid"
        )
        outstanding_amount = total_billed - total_paid

        return {
            "tenant_id": tenant_id,
            "period_months": months,
            "summary": {
                "total_billed": float(total_billed),
                "total_paid": float(total_paid),
                "outstanding_amount": float(outstanding_amount),
                "payment_rate": (
                    float(total_paid / total_billed) if total_billed > 0 else 0
                ),
            },
            "monthly_invoices": [dict(row) for row in invoice_rows],
            "monthly_usage": [dict(row) for row in usage_rows],
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def _get_usage_data(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage data for billing period"""
        query = """
        SELECT 
            COUNT(*) as total_requests,
            SUM((metadata->>'total_tokens')::int) as total_tokens,
            AVG((metadata->>'storage_gb')::float) as avg_storage_gb,
            SUM(cost_amount::decimal) as total_cost
        FROM mapper_cost_events 
        WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id, start_date, end_date)

        return {
            "total_requests": row["total_requests"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "avg_storage_gb": float(row["avg_storage_gb"] or 0),
            "total_cost": float(row["total_cost"] or 0),
        }

    def _get_tax_rate(self, tenant_id: str) -> Decimal:
        """Get tax rate for tenant (simplified - would use actual billing address)"""
        # In a real implementation, this would look up the tenant's billing address
        # and determine the appropriate tax rate
        return self.tax_rates.get("default", Decimal("0.00"))

    async def initialize_billing_schema(self) -> None:
        """Initialize billing database schema"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS invoices (
            invoice_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            billing_period VARCHAR(50) NOT NULL,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            line_items JSONB NOT NULL,
            subtotal DECIMAL(12,2) NOT NULL,
            tax_amount DECIMAL(12,2) NOT NULL,
            total_amount DECIMAL(12,2) NOT NULL,
            currency VARCHAR(3) DEFAULT 'USD',
            status VARCHAR(50) NOT NULL,
            due_date TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            paid_at TIMESTAMP NULL,
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX IF NOT EXISTS idx_invoices_tenant_id ON invoices(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
        CREATE INDEX IF NOT EXISTS idx_invoices_due_date ON invoices(due_date);
        CREATE INDEX IF NOT EXISTS idx_invoices_period ON invoices(period_start, period_end);
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)
