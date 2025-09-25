"""
Mapper Service Cost Tracking

Single Responsibility: Track costs for mapping operations only.

This component is responsible ONLY for:
- Recording mapping operation costs
- Calculating mapping-specific cost metrics
- Storing cost events for mapping operations

It does NOT handle:
- Cross-service cost aggregation (handled by analysis service)
- Billing calculations (handled by billing manager)
- Cost optimization recommendations (handled by analysis service)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import asyncpg
import json

from shared.interfaces.cost_monitoring import CostEvent, CostCategory

logger = logging.getLogger(__name__)


@dataclass
class MappingCostEvent:
    """Cost event specific to mapping operations"""

    tenant_id: str
    operation_type: str  # "mapping", "validation", "training"
    model_name: str
    input_tokens: int
    output_tokens: int
    inference_time_ms: float
    cost_amount: Decimal
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_cost_event(self) -> CostEvent:
        """Convert to generic cost event"""
        return CostEvent(
            tenant_id=self.tenant_id,
            event_type="mapping_operation",
            resource_id=f"{self.operation_type}_{self.model_name}",
            cost_amount=self.cost_amount,
            currency="USD",
            timestamp=self.timestamp,
            metadata={
                "operation_type": self.operation_type,
                "model_name": self.model_name,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                **self.metadata,
            },
            performance_metrics={
                "inference_time_ms": self.inference_time_ms,
                "tokens_per_second": (
                    (self.input_tokens + self.output_tokens)
                    / (self.inference_time_ms / 1000)
                    if self.inference_time_ms > 0
                    else 0
                ),
                "cost_per_token": (
                    float(self.cost_amount / (self.input_tokens + self.output_tokens))
                    if (self.input_tokens + self.output_tokens) > 0
                    else 0
                ),
            },
        )


class MapperCostTracker:
    """
    Mapper Service Cost Tracking

    Single Responsibility: Track and store costs for mapping operations only.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.cost_rates = self._load_mapping_cost_rates()

    def _load_mapping_cost_rates(self) -> Dict[str, Decimal]:
        """Load cost rates specific to mapping operations"""
        return {
            # Model inference costs (per token)
            "llama_3_8b_inference": Decimal("0.0002"),
            "phi_3_mini_inference": Decimal("0.0001"),
            "embedding_generation": Decimal("0.00005"),
            # Operation costs
            "mapping_request_base": Decimal("0.001"),
            "validation_request": Decimal("0.0005"),
            "model_loading": Decimal("0.05"),
            # Training costs (per hour)
            "lora_training_hour": Decimal("1.20"),
            "full_training_hour": Decimal("8.50"),
        }

    async def track_mapping_cost(
        self,
        tenant_id: str,
        operation_type: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        inference_time_ms: float,
        metadata: Dict[str, Any] = None,
    ) -> MappingCostEvent:
        """
        Track cost for a mapping operation.

        Single Responsibility: Calculate and store mapping operation cost only.
        """
        # Calculate costs
        total_tokens = input_tokens + output_tokens

        # Token costs based on model
        if "llama" in model_name.lower():
            token_cost = self.cost_rates["llama_3_8b_inference"] * total_tokens
        elif "phi" in model_name.lower():
            token_cost = self.cost_rates["phi_3_mini_inference"] * total_tokens
        else:
            token_cost = (
                self.cost_rates["llama_3_8b_inference"] * total_tokens
            )  # Default

        # Base operation cost
        base_cost = self.cost_rates["mapping_request_base"]

        # Additional costs based on operation type
        additional_cost = Decimal("0")
        if operation_type == "validation":
            additional_cost = self.cost_rates["validation_request"]
        elif metadata and metadata.get("model_loaded", False):
            additional_cost = self.cost_rates["model_loading"]

        total_cost = token_cost + base_cost + additional_cost

        # Create cost event
        cost_event = MappingCostEvent(
            tenant_id=tenant_id,
            operation_type=operation_type,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inference_time_ms=inference_time_ms,
            cost_amount=total_cost,
            timestamp=datetime.utcnow(),
            metadata={
                "token_cost": str(token_cost),
                "base_cost": str(base_cost),
                "additional_cost": str(additional_cost),
                **(metadata or {}),
            },
        )

        # Store cost event
        await self._store_cost_event(cost_event)

        logger.debug(f"Tracked mapping cost: {total_cost} for tenant {tenant_id}")
        return cost_event

    async def track_training_cost(
        self,
        tenant_id: str,
        training_job_id: str,
        training_type: str,
        duration_hours: float,
        model_size: str = "8b",
    ) -> MappingCostEvent:
        """
        Track cost for model training operations.

        Single Responsibility: Calculate and store training operation cost only.
        """
        # Base cost based on training type
        if training_type == "lora":
            base_rate = self.cost_rates["lora_training_hour"]
        else:
            base_rate = self.cost_rates["full_training_hour"]

        # Adjust for model size
        size_multiplier = {"3b": 0.5, "8b": 1.0, "13b": 1.5, "70b": 3.0}.get(
            model_size, 1.0
        )
        total_cost = (
            base_rate * Decimal(str(duration_hours)) * Decimal(str(size_multiplier))
        )

        # Create cost event
        cost_event = MappingCostEvent(
            tenant_id=tenant_id,
            operation_type="training",
            model_name=f"{model_size}_{training_type}",
            input_tokens=0,  # Not applicable for training
            output_tokens=0,  # Not applicable for training
            inference_time_ms=duration_hours
            * 3600
            * 1000,  # Convert to ms for consistency
            cost_amount=total_cost,
            timestamp=datetime.utcnow(),
            metadata={
                "training_job_id": training_job_id,
                "training_type": training_type,
                "duration_hours": duration_hours,
                "model_size": model_size,
                "size_multiplier": size_multiplier,
                "base_rate": str(base_rate),
            },
        )

        # Store cost event
        await self._store_cost_event(cost_event)

        logger.info(f"Tracked training cost: {total_cost} for tenant {tenant_id}")
        return cost_event

    async def get_tenant_cost_summary(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get cost summary for tenant.

        Single Responsibility: Return mapping-specific cost data only.
        """
        query = """
        SELECT 
            operation_type,
            COUNT(*) as operation_count,
            SUM(cost_amount::decimal) as total_cost,
            AVG(cost_amount::decimal) as avg_cost,
            SUM(input_tokens + output_tokens) as total_tokens,
            AVG(inference_time_ms) as avg_inference_time
        FROM mapper_cost_events 
        WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
        GROUP BY operation_type
        ORDER BY total_cost DESC
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id, start_date, end_date)

        # Calculate totals
        total_cost = sum(row["total_cost"] for row in rows)
        total_operations = sum(row["operation_count"] for row in rows)

        return {
            "tenant_id": tenant_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_cost": float(total_cost),
            "total_operations": total_operations,
            "cost_by_operation": [dict(row) for row in rows],
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def get_recent_cost_events(
        self, tenant_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent cost events for tenant.

        Single Responsibility: Return recent mapping cost events only.
        """
        query = """
        SELECT * FROM mapper_cost_events 
        WHERE tenant_id = $1 
        ORDER BY timestamp DESC 
        LIMIT $2
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id, limit)

        return [dict(row) for row in rows]

    async def _store_cost_event(self, cost_event: MappingCostEvent) -> None:
        """
        Store cost event in database.

        Single Responsibility: Persist mapping cost event only.
        """
        query = """
        INSERT INTO mapper_cost_events (
            tenant_id, operation_type, model_name, input_tokens, output_tokens,
            inference_time_ms, cost_amount, timestamp, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                cost_event.tenant_id,
                cost_event.operation_type,
                cost_event.model_name,
                cost_event.input_tokens,
                cost_event.output_tokens,
                cost_event.inference_time_ms,
                str(cost_event.cost_amount),
                cost_event.timestamp,
                json.dumps(cost_event.metadata),
            )

    async def initialize_database_schema(self) -> None:
        """
        Initialize database schema for cost tracking.

        Single Responsibility: Create mapping cost tracking tables only.
        """
        schema_sql = """
        CREATE TABLE IF NOT EXISTS mapper_cost_events (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            operation_type VARCHAR(50) NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            inference_time_ms DECIMAL(10,2) NOT NULL DEFAULT 0,
            cost_amount DECIMAL(12,6) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX IF NOT EXISTS idx_mapper_cost_events_tenant_id ON mapper_cost_events(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_mapper_cost_events_timestamp ON mapper_cost_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_mapper_cost_events_operation ON mapper_cost_events(operation_type);
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)
