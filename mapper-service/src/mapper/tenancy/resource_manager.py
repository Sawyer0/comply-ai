"""
Resource Management System for Mapper Service

Provides comprehensive resource management including:
- Resource allocation and monitoring
- Auto-scaling based on usage patterns
- Resource optimization recommendations
- Performance tracking and alerting
- Integration with cost monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
from decimal import Decimal
import json
import psutil
import GPUtil

from .tenant_manager import MapperTenantManager, MapperResourceType

logger = logging.getLogger(__name__)


class ResourceStatus(str, Enum):
    """Resource status types"""

    AVAILABLE = "available"
    ALLOCATED = "allocated"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ScalingAction(str, Enum):
    """Auto-scaling action types"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    ALERT = "alert"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""

    timestamp: datetime
    tenant_id: str
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    active_requests: int
    queue_length: int
    response_time_ms: float
    error_rate_percent: float


@dataclass
class ResourceAllocation:
    """Resource allocation for tenant"""

    tenant_id: str
    allocated_cpu_cores: float
    allocated_memory_gb: float
    allocated_gpu_count: int
    allocated_storage_gb: float
    max_concurrent_requests: int
    priority_level: int
    created_at: datetime
    updated_at: datetime


@dataclass
class ScalingRecommendation:
    """Auto-scaling recommendation"""

    tenant_id: str
    resource_type: str
    current_allocation: float
    recommended_allocation: float
    action: ScalingAction
    confidence: float
    reasoning: str
    estimated_cost_impact: Decimal
    timestamp: datetime


class ResourceManager:
    """Comprehensive resource management system"""

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        tenant_manager: MapperTenantManager,
    ):
        self.db_pool = db_pool
        self.tenant_manager = tenant_manager
        self.resource_limits = self._load_resource_limits()
        self.scaling_policies = self._load_scaling_policies()
        self._monitoring_active = False

    def _load_resource_limits(self) -> Dict[str, Dict[str, float]]:
        """Load resource limits by tenant tier"""
        return {
            "free": {
                "max_cpu_cores": 0.5,
                "max_memory_gb": 1.0,
                "max_gpu_count": 0,
                "max_storage_gb": 1.0,
                "max_concurrent_requests": 10,
                "max_queue_length": 50,
            },
            "basic": {
                "max_cpu_cores": 2.0,
                "max_memory_gb": 4.0,
                "max_gpu_count": 0,
                "max_storage_gb": 10.0,
                "max_concurrent_requests": 50,
                "max_queue_length": 200,
            },
            "premium": {
                "max_cpu_cores": 8.0,
                "max_memory_gb": 16.0,
                "max_gpu_count": 1,
                "max_storage_gb": 50.0,
                "max_concurrent_requests": 200,
                "max_queue_length": 500,
            },
            "enterprise": {
                "max_cpu_cores": 32.0,
                "max_memory_gb": 64.0,
                "max_gpu_count": 4,
                "max_storage_gb": 200.0,
                "max_concurrent_requests": 1000,
                "max_queue_length": 2000,
            },
        }

    def _load_scaling_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load auto-scaling policies"""
        return {
            "cpu_scaling": {
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "scale_up_factor": 1.5,
                "scale_down_factor": 0.8,
                "cooldown_minutes": 5,
            },
            "memory_scaling": {
                "scale_up_threshold": 85.0,
                "scale_down_threshold": 40.0,
                "scale_up_factor": 1.3,
                "scale_down_factor": 0.9,
                "cooldown_minutes": 10,
            },
            "request_scaling": {
                "scale_up_threshold": 90.0,  # % of max concurrent requests
                "scale_down_threshold": 20.0,
                "scale_up_factor": 1.2,
                "scale_down_factor": 0.9,
                "cooldown_minutes": 3,
            },
        }

    async def allocate_resources(self, tenant_id: str) -> ResourceAllocation:
        """Allocate resources for tenant based on tier"""
        tenant = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Get resource limits for tenant tier
        limits = self.resource_limits.get(tenant.tier, self.resource_limits["free"])

        # Create resource allocation
        allocation = ResourceAllocation(
            tenant_id=tenant_id,
            allocated_cpu_cores=limits["max_cpu_cores"],
            allocated_memory_gb=limits["max_memory_gb"],
            allocated_gpu_count=int(limits["max_gpu_count"]),
            allocated_storage_gb=limits["max_storage_gb"],
            max_concurrent_requests=int(limits["max_concurrent_requests"]),
            priority_level=self._get_priority_level(tenant.tier),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Store allocation in database
        await self._store_resource_allocation(allocation)

        logger.info(f"Allocated resources for tenant {tenant_id} (tier: {tenant.tier})")
        return allocation

    async def get_resource_allocation(
        self, tenant_id: str
    ) -> Optional[ResourceAllocation]:
        """Get current resource allocation for tenant"""
        query = """
        SELECT * FROM resource_allocations WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id)

        if not row:
            return None

        return ResourceAllocation(
            tenant_id=row["tenant_id"],
            allocated_cpu_cores=float(row["allocated_cpu_cores"]),
            allocated_memory_gb=float(row["allocated_memory_gb"]),
            allocated_gpu_count=row["allocated_gpu_count"],
            allocated_storage_gb=float(row["allocated_storage_gb"]),
            max_concurrent_requests=row["max_concurrent_requests"],
            priority_level=row["priority_level"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def collect_resource_metrics(self, tenant_id: str) -> ResourceMetrics:
        """Collect current resource metrics for tenant"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        # Get GPU metrics if available
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_usage = gpu.load * 100
                gpu_memory_usage = gpu.memoryUtil * 100
        except Exception:
            pass  # GPU monitoring not available

        # Get tenant-specific metrics from database
        tenant_metrics = await self._get_tenant_specific_metrics(tenant_id)

        metrics = ResourceMetrics(
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_usage_percent=gpu_memory_usage,
            disk_usage_percent=disk.percent,
            network_io_mbps=tenant_metrics.get("network_io_mbps", 0.0),
            active_requests=tenant_metrics.get("active_requests", 0),
            queue_length=tenant_metrics.get("queue_length", 0),
            response_time_ms=tenant_metrics.get("response_time_ms", 0.0),
            error_rate_percent=tenant_metrics.get("error_rate_percent", 0.0),
        )

        # Store metrics
        await self._store_resource_metrics(metrics)

        return metrics

    async def analyze_scaling_needs(
        self, tenant_id: str
    ) -> List[ScalingRecommendation]:
        """Analyze resource usage and generate scaling recommendations"""
        # Get recent metrics (last hour)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        metrics_history = await self._get_metrics_history(
            tenant_id, start_time, end_time
        )
        if not metrics_history:
            return []

        # Get current allocation
        allocation = await self.get_resource_allocation(tenant_id)
        if not allocation:
            return []

        recommendations = []

        # Analyze CPU usage
        avg_cpu = sum(m.cpu_usage_percent for m in metrics_history) / len(
            metrics_history
        )
        cpu_recommendation = self._analyze_cpu_scaling(tenant_id, avg_cpu, allocation)
        if cpu_recommendation:
            recommendations.append(cpu_recommendation)

        # Analyze memory usage
        avg_memory = sum(m.memory_usage_percent for m in metrics_history) / len(
            metrics_history
        )
        memory_recommendation = self._analyze_memory_scaling(
            tenant_id, avg_memory, allocation
        )
        if memory_recommendation:
            recommendations.append(memory_recommendation)

        # Analyze request capacity
        avg_requests = sum(m.active_requests for m in metrics_history) / len(
            metrics_history
        )
        request_recommendation = self._analyze_request_scaling(
            tenant_id, avg_requests, allocation
        )
        if request_recommendation:
            recommendations.append(request_recommendation)

        # Store recommendations
        for rec in recommendations:
            await self._store_scaling_recommendation(rec)

        return recommendations

    def _analyze_cpu_scaling(
        self, tenant_id: str, avg_cpu: float, allocation: ResourceAllocation
    ) -> Optional[ScalingRecommendation]:
        """Analyze CPU scaling needs"""
        policy = self.scaling_policies["cpu_scaling"]

        if avg_cpu > policy["scale_up_threshold"]:
            new_allocation = allocation.allocated_cpu_cores * policy["scale_up_factor"]
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="cpu",
                current_allocation=allocation.allocated_cpu_cores,
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_UP,
                confidence=min(0.9, (avg_cpu - policy["scale_up_threshold"]) / 20),
                reasoning=f"CPU usage at {avg_cpu:.1f}% exceeds threshold of {policy['scale_up_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "cpu", new_allocation - allocation.allocated_cpu_cores
                ),
                timestamp=datetime.utcnow(),
            )
        elif avg_cpu < policy["scale_down_threshold"]:
            new_allocation = (
                allocation.allocated_cpu_cores * policy["scale_down_factor"]
            )
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="cpu",
                current_allocation=allocation.allocated_cpu_cores,
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_DOWN,
                confidence=min(0.8, (policy["scale_down_threshold"] - avg_cpu) / 20),
                reasoning=f"CPU usage at {avg_cpu:.1f}% below threshold of {policy['scale_down_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "cpu", new_allocation - allocation.allocated_cpu_cores
                ),
                timestamp=datetime.utcnow(),
            )

        return None

    def _analyze_memory_scaling(
        self, tenant_id: str, avg_memory: float, allocation: ResourceAllocation
    ) -> Optional[ScalingRecommendation]:
        """Analyze memory scaling needs"""
        policy = self.scaling_policies["memory_scaling"]

        if avg_memory > policy["scale_up_threshold"]:
            new_allocation = allocation.allocated_memory_gb * policy["scale_up_factor"]
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="memory",
                current_allocation=allocation.allocated_memory_gb,
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_UP,
                confidence=min(0.9, (avg_memory - policy["scale_up_threshold"]) / 15),
                reasoning=f"Memory usage at {avg_memory:.1f}% exceeds threshold of {policy['scale_up_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "memory", new_allocation - allocation.allocated_memory_gb
                ),
                timestamp=datetime.utcnow(),
            )
        elif avg_memory < policy["scale_down_threshold"]:
            new_allocation = (
                allocation.allocated_memory_gb * policy["scale_down_factor"]
            )
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="memory",
                current_allocation=allocation.allocated_memory_gb,
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_DOWN,
                confidence=min(0.8, (policy["scale_down_threshold"] - avg_memory) / 30),
                reasoning=f"Memory usage at {avg_memory:.1f}% below threshold of {policy['scale_down_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "memory", new_allocation - allocation.allocated_memory_gb
                ),
                timestamp=datetime.utcnow(),
            )

        return None

    def _analyze_request_scaling(
        self, tenant_id: str, avg_requests: float, allocation: ResourceAllocation
    ) -> Optional[ScalingRecommendation]:
        """Analyze request capacity scaling needs"""
        policy = self.scaling_policies["request_scaling"]
        request_utilization = (avg_requests / allocation.max_concurrent_requests) * 100

        if request_utilization > policy["scale_up_threshold"]:
            new_allocation = (
                allocation.max_concurrent_requests * policy["scale_up_factor"]
            )
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="request_capacity",
                current_allocation=float(allocation.max_concurrent_requests),
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_UP,
                confidence=min(
                    0.9, (request_utilization - policy["scale_up_threshold"]) / 10
                ),
                reasoning=f"Request utilization at {request_utilization:.1f}% exceeds threshold of {policy['scale_up_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "requests", new_allocation - allocation.max_concurrent_requests
                ),
                timestamp=datetime.utcnow(),
            )
        elif request_utilization < policy["scale_down_threshold"]:
            new_allocation = (
                allocation.max_concurrent_requests * policy["scale_down_factor"]
            )
            return ScalingRecommendation(
                tenant_id=tenant_id,
                resource_type="request_capacity",
                current_allocation=float(allocation.max_concurrent_requests),
                recommended_allocation=new_allocation,
                action=ScalingAction.SCALE_DOWN,
                confidence=min(
                    0.7, (policy["scale_down_threshold"] - request_utilization) / 15
                ),
                reasoning=f"Request utilization at {request_utilization:.1f}% below threshold of {policy['scale_down_threshold']}%",
                estimated_cost_impact=self._estimate_cost_impact(
                    "requests", new_allocation - allocation.max_concurrent_requests
                ),
                timestamp=datetime.utcnow(),
            )

        return None

    def _estimate_cost_impact(self, resource_type: str, delta: float) -> Decimal:
        """Estimate cost impact of resource changes"""
        cost_per_unit = {
            "cpu": Decimal("0.048"),  # per core per hour
            "memory": Decimal("0.006"),  # per GB per hour
            "requests": Decimal("0.0001"),  # per request capacity per hour
        }

        rate = cost_per_unit.get(resource_type, Decimal("0"))
        return rate * Decimal(str(abs(delta))) * 24 * 30  # Monthly estimate

    def _get_priority_level(self, tier: str) -> int:
        """Get priority level for tenant tier"""
        priority_map = {"free": 1, "basic": 2, "premium": 3, "enterprise": 4}
        return priority_map.get(tier, 1)

    async def _get_tenant_specific_metrics(self, tenant_id: str) -> Dict[str, float]:
        """Get tenant-specific metrics from application layer"""
        # This would integrate with the actual application metrics
        # For now, return mock data
        return {
            "network_io_mbps": 10.5,
            "active_requests": 25,
            "queue_length": 5,
            "response_time_ms": 150.0,
            "error_rate_percent": 0.5,
        }

    async def _store_resource_allocation(self, allocation: ResourceAllocation) -> None:
        """Store resource allocation in database"""
        query = """
        INSERT INTO resource_allocations (
            tenant_id, allocated_cpu_cores, allocated_memory_gb, allocated_gpu_count,
            allocated_storage_gb, max_concurrent_requests, priority_level,
            created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (tenant_id) DO UPDATE SET
            allocated_cpu_cores = EXCLUDED.allocated_cpu_cores,
            allocated_memory_gb = EXCLUDED.allocated_memory_gb,
            allocated_gpu_count = EXCLUDED.allocated_gpu_count,
            allocated_storage_gb = EXCLUDED.allocated_storage_gb,
            max_concurrent_requests = EXCLUDED.max_concurrent_requests,
            priority_level = EXCLUDED.priority_level,
            updated_at = EXCLUDED.updated_at
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                allocation.tenant_id,
                allocation.allocated_cpu_cores,
                allocation.allocated_memory_gb,
                allocation.allocated_gpu_count,
                allocation.allocated_storage_gb,
                allocation.max_concurrent_requests,
                allocation.priority_level,
                allocation.created_at,
                allocation.updated_at,
            )

    async def _store_resource_metrics(self, metrics: ResourceMetrics) -> None:
        """Store resource metrics in database"""
        query = """
        INSERT INTO resource_metrics (
            timestamp, tenant_id, cpu_usage_percent, memory_usage_percent,
            gpu_usage_percent, gpu_memory_usage_percent, disk_usage_percent,
            network_io_mbps, active_requests, queue_length,
            response_time_ms, error_rate_percent
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                metrics.timestamp,
                metrics.tenant_id,
                metrics.cpu_usage_percent,
                metrics.memory_usage_percent,
                metrics.gpu_usage_percent,
                metrics.gpu_memory_usage_percent,
                metrics.disk_usage_percent,
                metrics.network_io_mbps,
                metrics.active_requests,
                metrics.queue_length,
                metrics.response_time_ms,
                metrics.error_rate_percent,
            )

    async def _store_scaling_recommendation(
        self, recommendation: ScalingRecommendation
    ) -> None:
        """Store scaling recommendation in database"""
        query = """
        INSERT INTO scaling_recommendations (
            tenant_id, resource_type, current_allocation, recommended_allocation,
            action, confidence, reasoning, estimated_cost_impact, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                recommendation.tenant_id,
                recommendation.resource_type,
                recommendation.current_allocation,
                recommendation.recommended_allocation,
                recommendation.action.value,
                recommendation.confidence,
                recommendation.reasoning,
                str(recommendation.estimated_cost_impact),
                recommendation.timestamp,
            )

    async def _get_metrics_history(
        self, tenant_id: str, start_time: datetime, end_time: datetime
    ) -> List[ResourceMetrics]:
        """Get metrics history for tenant"""
        query = """
        SELECT * FROM resource_metrics 
        WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
        ORDER BY timestamp ASC
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id, start_time, end_time)

        return [
            ResourceMetrics(
                timestamp=row["timestamp"],
                tenant_id=row["tenant_id"],
                cpu_usage_percent=row["cpu_usage_percent"],
                memory_usage_percent=row["memory_usage_percent"],
                gpu_usage_percent=row["gpu_usage_percent"],
                gpu_memory_usage_percent=row["gpu_memory_usage_percent"],
                disk_usage_percent=row["disk_usage_percent"],
                network_io_mbps=row["network_io_mbps"],
                active_requests=row["active_requests"],
                queue_length=row["queue_length"],
                response_time_ms=row["response_time_ms"],
                error_rate_percent=row["error_rate_percent"],
            )
            for row in rows
        ]

    async def initialize_resource_schema(self) -> None:
        """Initialize resource management database schema"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS resource_allocations (
            tenant_id VARCHAR(255) PRIMARY KEY,
            allocated_cpu_cores DECIMAL(8,2) NOT NULL,
            allocated_memory_gb DECIMAL(8,2) NOT NULL,
            allocated_gpu_count INTEGER NOT NULL DEFAULT 0,
            allocated_storage_gb DECIMAL(10,2) NOT NULL,
            max_concurrent_requests INTEGER NOT NULL,
            priority_level INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS resource_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            tenant_id VARCHAR(255) NOT NULL,
            cpu_usage_percent DECIMAL(5,2) NOT NULL,
            memory_usage_percent DECIMAL(5,2) NOT NULL,
            gpu_usage_percent DECIMAL(5,2) NOT NULL DEFAULT 0,
            gpu_memory_usage_percent DECIMAL(5,2) NOT NULL DEFAULT 0,
            disk_usage_percent DECIMAL(5,2) NOT NULL,
            network_io_mbps DECIMAL(10,2) NOT NULL DEFAULT 0,
            active_requests INTEGER NOT NULL DEFAULT 0,
            queue_length INTEGER NOT NULL DEFAULT 0,
            response_time_ms DECIMAL(10,2) NOT NULL DEFAULT 0,
            error_rate_percent DECIMAL(5,2) NOT NULL DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS scaling_recommendations (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            resource_type VARCHAR(50) NOT NULL,
            current_allocation DECIMAL(10,2) NOT NULL,
            recommended_allocation DECIMAL(10,2) NOT NULL,
            action VARCHAR(20) NOT NULL,
            confidence DECIMAL(3,2) NOT NULL,
            reasoning TEXT NOT NULL,
            estimated_cost_impact DECIMAL(12,2) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_resource_metrics_tenant_time ON resource_metrics(tenant_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_scaling_recommendations_tenant ON scaling_recommendations(tenant_id);
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)
