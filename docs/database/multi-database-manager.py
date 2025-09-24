"""
Multi-Database Manager for Comply-AI Platform

This module provides a comprehensive database manager for the multi-database architecture
used in the Comply-AI platform, supporting Core, Billing, and Analytics databases.
"""

import asyncio
import asyncpg
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for a single database."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class MultiDatabaseConfig:
    """Configuration for all databases in the multi-database architecture."""
    core: DatabaseConfig
    billing: DatabaseConfig
    analytics: DatabaseConfig

class MultiDatabaseManager:
    """
    Manages connections to multiple databases in the Comply-AI platform.
    
    This class provides a unified interface for accessing the three databases:
    - Core Database: Application data and service operations
    - Billing Database: Subscriptions, payments, and white-glove services
    - Analytics Database: Usage metrics, performance data, and reports
    """
    
    def __init__(self, config: MultiDatabaseConfig):
        self.config = config
        self._pools: Dict[str, asyncpg.Pool] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection pools for all databases."""
        if self._initialized:
            return
        
        try:
            # Initialize Core Database pool
            self._pools['core'] = await asyncpg.create_pool(
                host=self.config.core.host,
                port=self.config.core.port,
                database=self.config.core.database,
                user=self.config.core.username,
                password=self.config.core.password,
                ssl=self.config.core.ssl_mode,
                min_size=5,
                max_size=self.config.core.pool_size,
                command_timeout=60
            )
            logger.info("Core database pool initialized")
            
            # Initialize Billing Database pool
            self._pools['billing'] = await asyncpg.create_pool(
                host=self.config.billing.host,
                port=self.config.billing.port,
                database=self.config.billing.database,
                user=self.config.billing.username,
                password=self.config.billing.password,
                ssl=self.config.billing.ssl_mode,
                min_size=3,
                max_size=self.config.billing.pool_size,
                command_timeout=60
            )
            logger.info("Billing database pool initialized")
            
            # Initialize Analytics Database pool
            self._pools['analytics'] = await asyncpg.create_pool(
                host=self.config.analytics.host,
                port=self.config.analytics.port,
                database=self.config.analytics.database,
                user=self.config.analytics.username,
                password=self.config.analytics.password,
                ssl=self.config.analytics.ssl_mode,
                min_size=3,
                max_size=self.config.analytics.pool_size,
                command_timeout=60
            )
            logger.info("Analytics database pool initialized")
            
            self._initialized = True
            logger.info("All database pools initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database pools: %s", e)
            await self.close()
            raise
    
    @asynccontextmanager
    async def get_connection(self, database: str):
        """
        Get database connection from pool.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            
        Yields:
            asyncpg.Connection: Database connection
            
        Raises:
            ValueError: If database name is invalid
            RuntimeError: If database pool is not initialized
        """
        if database not in self._pools:
            raise ValueError(f"Unknown database: {database}")
        
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self._pools[database].acquire() as conn:
            yield conn
    
    async def execute_query(self, database: str, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute query on specified database.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            query: SQL query string
            *args: Query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        async with self.get_connection(database) as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, database: str, command: str, *args) -> str:
        """
        Execute command on specified database.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            command: SQL command string
            *args: Command parameters
            
        Returns:
            Command result string
        """
        async with self.get_connection(database) as conn:
            return await conn.execute(command, *args)
    
    async def fetch_one(self, database: str, query: str, *args) -> Optional[Dict[str, Any]]:
        """
        Fetch single row from specified database.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            query: SQL query string
            *args: Query parameters
            
        Returns:
            Single row as dictionary or None
        """
        async with self.get_connection(database) as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetch_scalar(self, database: str, query: str, *args) -> Any:
        """
        Fetch scalar value from specified database.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            query: SQL query string
            *args: Query parameters
            
        Returns:
            Scalar value
        """
        async with self.get_connection(database) as conn:
            return await conn.fetchval(query, *args)
    
    async def execute_transaction(self, database: str, operations: List[tuple]) -> List[Any]:
        """
        Execute multiple operations in a transaction.
        
        Args:
            database: Database name ('core', 'billing', 'analytics')
            operations: List of (query, args) tuples
            
        Returns:
            List of operation results
        """
        async with self.get_connection(database) as conn:
            async with conn.transaction():
                results = []
                for query, args in operations:
                    if query.strip().upper().startswith('SELECT'):
                        rows = await conn.fetch(query, *args)
                        results.append([dict(row) for row in rows])
                    else:
                        result = await conn.execute(query, *args)
                        results.append(result)
                return results
    
    async def close(self):
        """Close all database connections."""
        for name, pool in self._pools.items():
            try:
                await pool.close()
                logger.info("Closed %s database pool", name)
            except Exception as e:
                logger.error("Error closing %s database pool: %s", name, e)
        
        self._pools.clear()
        self._initialized = False
        logger.info("All database connections closed")

class CoreStorageManager:
    """Manages core database operations."""
    
    def __init__(self, db_manager: MultiDatabaseManager):
        self.db = db_manager
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user from core database."""
        return await self.db.fetch_one(
            'core',
            "SELECT * FROM users WHERE user_id = $1",
            user_id
        )
    
    async def get_tenant_config(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant configuration from core database."""
        return await self.db.fetch_one(
            'core',
            "SELECT * FROM tenant_configs WHERE tenant_id = $1",
            tenant_id
        )
    
    async def store_mapping_result(self, record: Dict[str, Any]):
        """Store mapping result in core database."""
        await self.db.execute_command(
            'core',
            """
            INSERT INTO storage_records (
                id, source_data, mapped_data, model_version, 
                timestamp, metadata, tenant_id, s3_key, encrypted
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            record['id'], record['source_data'], record['mapped_data'],
            record['model_version'], record['timestamp'], json.dumps(record['metadata']),
            record['tenant_id'], record.get('s3_key'), record.get('encrypted', False)
        )
    
    async def log_audit_event(self, event: Dict[str, Any]):
        """Log audit event in core database."""
        await self.db.execute_command(
            'core',
            """
            INSERT INTO audit_logs (
                tenant_id, user_id, action, resource_type, resource_id,
                details, ip_address, user_agent
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            event['tenant_id'], event.get('user_id'), event['action'],
            event['resource_type'], event.get('resource_id'),
            json.dumps(event.get('details', {})), event.get('ip_address'),
            event.get('user_agent')
        )

class BillingStorageManager:
    """Manages billing database operations."""
    
    def __init__(self, db_manager: MultiDatabaseManager):
        self.db = db_manager
    
    async def get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user subscription from billing database."""
        return await self.db.fetch_one(
            'billing',
            "SELECT * FROM user_subscriptions WHERE user_id = $1 AND status = 'active'",
            user_id
        )
    
    async def track_usage(self, user_id: str, tenant_id: str, usage_type: str, amount: float):
        """Track usage in billing database."""
        subscription = await self.get_user_subscription(user_id)
        if not subscription:
            raise ValueError("No active subscription found")
        
        await self.db.execute_command(
            'billing',
            """
            INSERT INTO usage_records (
                user_id, tenant_id, subscription_id, usage_type,
                usage_amount, usage_unit, billing_period_start, billing_period_end
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            user_id, tenant_id, subscription['subscription_id'], usage_type,
            amount, 'count',
            datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + 
             timedelta(days=32)).replace(day=1) - timedelta(days=1)
        )
    
    async def create_white_glove_service(self, service_data: Dict[str, Any]) -> str:
        """Create white-glove service in billing database."""
        service_id = str(uuid.uuid4())
        await self.db.execute_command(
            'billing',
            """
            INSERT INTO white_glove_services (
                service_id, user_id, tenant_id, subscription_id,
                service_type, description, requirements, priority, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            service_id, service_data['user_id'], service_data['tenant_id'],
            service_data['subscription_id'], service_data['service_type'],
            service_data['description'], json.dumps(service_data['requirements']),
            service_data['priority'], 'requested'
        )
        return service_id
    
    async def get_billing_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get billing plan from billing database."""
        return await self.db.fetch_one(
            'billing',
            "SELECT * FROM billing_plans WHERE plan_id = $1",
            plan_id
        )
    
    async def check_free_tier_limits(self, user_id: str, tenant_id: str, usage_type: str, amount: float) -> bool:
        """Check if free tier usage limits are exceeded."""
        usage = await self.db.fetch_one(
            'billing',
            """
            SELECT current_usage, usage_limit FROM free_tier_usage 
            WHERE user_id = $1 AND tenant_id = $2 AND usage_type = $3
            """,
            user_id, tenant_id, usage_type
        )
        
        if not usage:
            return True  # No limit set
        
        return usage['current_usage'] + amount <= usage['usage_limit']
    
    async def update_free_tier_usage(self, user_id: str, tenant_id: str, usage_type: str, amount: float):
        """Update free tier usage."""
        await self.db.execute_command(
            'billing',
            """
            UPDATE free_tier_usage 
            SET current_usage = current_usage + $1
            WHERE user_id = $2 AND tenant_id = $3 AND usage_type = $4
            """,
            amount, user_id, tenant_id, usage_type
        )

class AnalyticsStorageManager:
    """Manages analytics database operations."""
    
    def __init__(self, db_manager: MultiDatabaseManager):
        self.db = db_manager
    
    async def record_usage_metric(self, metric_data: Dict[str, Any]):
        """Record usage metric in analytics database."""
        await self.db.execute_command(
            'analytics',
            """
            INSERT INTO usage_metrics (
                tenant_id, user_id, metric_type, metric_value, 
                timestamp, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            metric_data['tenant_id'], metric_data.get('user_id'), metric_data['metric_type'],
            metric_data['metric_value'], metric_data['timestamp'],
            json.dumps(metric_data.get('metadata', {}))
        )
    
    async def record_performance_metric(self, metric_data: Dict[str, Any]):
        """Record performance metric in analytics database."""
        await self.db.execute_command(
            'analytics',
            """
            INSERT INTO performance_metrics (
                tenant_id, service_name, metric_name, metric_value,
                timestamp, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            metric_data['tenant_id'], metric_data['service_name'], metric_data['metric_name'],
            metric_data['metric_value'], metric_data['timestamp'],
            json.dumps(metric_data.get('metadata', {}))
        )
    
    async def get_performance_report(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get performance report from analytics database."""
        return await self.db.execute_query(
            'analytics',
            """
            SELECT * FROM performance_metrics 
            WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp DESC
            """,
            tenant_id, start_date, end_date
        )
    
    async def create_compliance_report(self, report_data: Dict[str, Any]) -> str:
        """Create compliance report in analytics database."""
        report_id = str(uuid.uuid4())
        await self.db.execute_command(
            'analytics',
            """
            INSERT INTO compliance_reports (
                report_id, tenant_id, report_type, report_name,
                report_period_start, report_period_end, compliance_framework,
                status, summary, findings, recommendations
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            report_id, report_data['tenant_id'], report_data['report_type'],
            report_data['report_name'], report_data['report_period_start'],
            report_data['report_period_end'], report_data['compliance_framework'],
            'generating', json.dumps(report_data.get('summary', {})),
            json.dumps(report_data.get('findings', [])),
            json.dumps(report_data.get('recommendations', []))
        )
        return report_id

class UnifiedStorageManager:
    """
    Unified storage manager that provides access to all three databases.
    
    This class combines the functionality of CoreStorageManager, BillingStorageManager,
    and AnalyticsStorageManager to provide a single interface for all database operations.
    """
    
    def __init__(self, config: MultiDatabaseConfig):
        self.db_manager = MultiDatabaseManager(config)
        self.core = CoreStorageManager(self.db_manager)
        self.billing = BillingStorageManager(self.db_manager)
        self.analytics = AnalyticsStorageManager(self.db_manager)
    
    async def initialize(self):
        """Initialize all database connections."""
        await self.db_manager.initialize()
    
    async def close(self):
        """Close all database connections."""
        await self.db_manager.close()
    
    async def track_api_call(self, user_id: str, tenant_id: str, endpoint: str):
        """Track API call usage across all relevant databases."""
        try:
            # Check free tier limits
            can_proceed = await self.billing.check_free_tier_limits(
                user_id, tenant_id, 'api_calls', 1
            )
            
            if not can_proceed:
                raise ValueError("Free tier limit exceeded for API calls")
            
            # Update free tier usage
            await self.billing.update_free_tier_usage(
                user_id, tenant_id, 'api_calls', 1
            )
            
            # Record usage in billing database
            await self.billing.track_usage(user_id, tenant_id, 'api_calls', 1)
            
            # Record usage metric in analytics database
            await self.analytics.record_usage_metric({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'metric_type': 'api_calls',
                'metric_value': 1,
                'timestamp': datetime.utcnow(),
                'metadata': {'endpoint': endpoint}
            })
            
            # Log audit event in core database
            await self.core.log_audit_event({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'action': 'api_call',
                'resource_type': 'endpoint',
                'resource_id': endpoint,
                'details': {'endpoint': endpoint}
            })
            
        except Exception as e:
            logger.error("Error tracking API call: %s", e)
            raise
    
    async def process_mapping_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mapping request across all databases."""
        try:
            # Store mapping result in core database
            await self.core.store_mapping_result(request_data)
            
            # Track usage in billing database
            await self.billing.track_usage(
                request_data['user_id'], request_data['tenant_id'], 'detector_runs', 1
            )
            
            # Record performance metric in analytics database
            await self.analytics.record_performance_metric({
                'tenant_id': request_data['tenant_id'],
                'service_name': 'mapper',
                'metric_name': 'processing_time',
                'metric_value': request_data.get('processing_time_ms', 0),
                'timestamp': datetime.utcnow(),
                'metadata': {'model_version': request_data['model_version']}
            })
            
            return {'status': 'success', 'message': 'Mapping request processed successfully'}
            
        except Exception as e:
            logger.error("Error processing mapping request: %s", e)
            raise

# Example usage and configuration
def get_database_config() -> MultiDatabaseConfig:
    """Get database configuration from environment variables."""
    import os
    
    return MultiDatabaseConfig(
        core=DatabaseConfig(
            host=os.getenv("CORE_DB_HOST", "comply-ai-core-db.postgres.database.azure.com"),
            port=int(os.getenv("CORE_DB_PORT", 5432)),
            database=os.getenv("CORE_DB_NAME", "comply-ai-core"),
            username=os.getenv("CORE_DB_USER", "core_admin"),
            password=os.getenv("CORE_DB_PASSWORD", ""),
            pool_size=int(os.getenv("CORE_DB_POOL_SIZE", 10))
        ),
        billing=DatabaseConfig(
            host=os.getenv("BILLING_DB_HOST", "comply-ai-billing-db.postgres.database.azure.com"),
            port=int(os.getenv("BILLING_DB_PORT", 5432)),
            database=os.getenv("BILLING_DB_NAME", "comply-ai-billing"),
            username=os.getenv("BILLING_DB_USER", "billing_admin"),
            password=os.getenv("BILLING_DB_PASSWORD", ""),
            pool_size=int(os.getenv("BILLING_DB_POOL_SIZE", 5))
        ),
        analytics=DatabaseConfig(
            host=os.getenv("ANALYTICS_DB_HOST", "comply-ai-analytics-db.postgres.database.azure.com"),
            port=int(os.getenv("ANALYTICS_DB_PORT", 5432)),
            database=os.getenv("ANALYTICS_DB_NAME", "comply-ai-analytics"),
            username=os.getenv("ANALYTICS_DB_USER", "analytics_admin"),
            password=os.getenv("ANALYTICS_DB_PASSWORD", ""),
            pool_size=int(os.getenv("ANALYTICS_DB_POOL_SIZE", 5))
        )
    )

# Example usage
async def main():
    """Example usage of the multi-database manager."""
    config = get_database_config()
    storage_manager = UnifiedStorageManager(config)
    
    try:
        await storage_manager.initialize()
        
        # Example: Track API call
        await storage_manager.track_api_call(
            user_id="user-123",
            tenant_id="tenant-456",
            endpoint="/api/v1/map"
        )
        
        # Example: Process mapping request
        await storage_manager.process_mapping_request({
            'id': 'mapping-789',
            'user_id': 'user-123',
            'tenant_id': 'tenant-456',
            'source_data': 'sample text',
            'mapped_data': '{"taxonomy": ["HARM.SPEECH.Toxicity"]}',
            'model_version': 'v1.2.0',
            'timestamp': datetime.utcnow(),
            'metadata': {},
            'processing_time_ms': 150
        })
        
        print("Operations completed successfully")
        
    finally:
        await storage_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
