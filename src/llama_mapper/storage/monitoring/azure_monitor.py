"""Azure Database monitoring with native Azure Monitor integration."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from azure.identity import DefaultAzureCredential
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# Prometheus metrics
DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['pool_type', 'azure_region']
)

DATABASE_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query execution time',
    ['operation', 'table', 'tenant_id']
)

DATABASE_ERRORS = Counter(
    'database_errors_total',
    'Total database errors',
    ['error_type', 'operation']
)

AZURE_METRICS = Gauge(
    'azure_database_metric',
    'Azure Database metrics from Azure Monitor',
    ['metric_name', 'server_name']
)


class AzureDatabaseMonitor:
    """Azure Database monitoring with native Azure Monitor integration."""
    
    def __init__(self, connection_manager, azure_config):
        self.connection_manager = connection_manager
        self.azure_config = azure_config
        self.credential = DefaultAzureCredential()
        self.azure_monitor_client = None
        self._monitoring_active = False
        
        # Initialize Azure Monitor client if available
        try:
            from azure.monitor.query import MetricsQueryClient
            self.azure_monitor_client = MetricsQueryClient(self.credential)
        except ImportError:
            logger.warning("Azure Monitor client not available - install azure-monitor-query")
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        self._monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_connection_health()),
            asyncio.create_task(self._monitor_database_performance()),
            asyncio.create_task(self._monitor_azure_metrics()),
        ]
        
        logger.info("Azure Database monitoring started")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Monitoring tasks cancelled")
        except Exception as e:
            logger.error("Monitoring error", error=str(e))
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring_active = False
        logger.info("Azure Database monitoring stopped")
    
    async def _monitor_connection_health(self):
        """Monitor database connection health."""
        while self._monitoring_active:
            try:
                health_status = await self.connection_manager.health_check()
                
                for pool_name, is_healthy in health_status.items():
                    DATABASE_CONNECTIONS.labels(
                        pool_type=pool_name,
                        azure_region=self.azure_config.azure_region
                    ).set(1 if is_healthy else 0)
                
                if not all(health_status.values()):
                    logger.warning("Database connection health issues detected", status=health_status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Connection health monitoring error", error=str(e))
                DATABASE_ERRORS.labels(error_type="connection_health", operation="monitor").inc()
                await asyncio.sleep(60)  # Back off on error
    
    async def _monitor_database_performance(self):
        """Monitor database performance metrics."""
        while self._monitoring_active:
            try:
                # Get connection info
                connection_info = await self.connection_manager.get_azure_connection_info()
                
                # Update Prometheus metrics
                if 'active_connections' in connection_info:
                    DATABASE_CONNECTIONS.labels(
                        pool_type="active",
                        azure_region=self.azure_config.azure_region
                    ).set(int(connection_info['active_connections']))
                
                # Collect slow queries
                await self._collect_slow_queries()
                
                # Collect table statistics
                await self._collect_table_statistics()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Performance monitoring error", error=str(e))
                DATABASE_ERRORS.labels(error_type="performance", operation="monitor").inc()
                await asyncio.sleep(120)  # Back off on error
    
    async def _collect_slow_queries(self):
        """Collect slow query statistics."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Check if pg_stat_statements is available
                ext_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    )
                """)
                
                if not ext_exists:
                    return
                
                # Get slow queries (> 1 second)
                slow_queries = await conn.fetch("""
                    SELECT 
                        query,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        rows,
                        100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements 
                    WHERE mean_exec_time > 1000  -- Queries slower than 1 second
                    ORDER BY mean_exec_time DESC 
                    LIMIT 10
                """)
                
                for query_stat in slow_queries:
                    # Extract table name from query (simplified)
                    table_name = self._extract_table_name(query_stat['query'])
                    
                    DATABASE_QUERY_DURATION.labels(
                        operation="slow_query",
                        table=table_name,
                        tenant_id="system"
                    ).observe(query_stat['mean_exec_time'] / 1000.0)  # Convert ms to seconds
                
        except Exception as e:
            logger.warning("Failed to collect slow queries", error=str(e))
    
    async def _collect_table_statistics(self):
        """Collect table-level statistics."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Get table sizes and activity
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        pg_total_relation_size(relid) as total_size
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                """)
                
                for stat in table_stats:
                    # Log significant table activity
                    total_ops = stat['inserts'] + stat['updates'] + stat['deletes']
                    if total_ops > 1000:  # Arbitrary threshold
                        logger.debug(
                            "High table activity",
                            table=stat['tablename'],
                            total_operations=total_ops,
                            live_tuples=stat['live_tuples'],
                            size_bytes=stat['total_size']
                        )
                
        except Exception as e:
            logger.warning("Failed to collect table statistics", error=str(e))
    
    async def _monitor_azure_metrics(self):
        """Monitor Azure Database metrics from Azure Monitor."""
        if not self.azure_monitor_client:
            logger.debug("Azure Monitor client not available, skipping Azure metrics")
            return
        
        while self._monitoring_active:
            try:
                await self.collect_azure_metrics()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Azure metrics monitoring error", error=str(e))
                DATABASE_ERRORS.labels(error_type="azure_metrics", operation="monitor").inc()
                await asyncio.sleep(600)  # Back off on error
    
    async def collect_azure_metrics(self):
        """Collect Azure Database specific metrics."""
        if not self.azure_monitor_client:
            return
        
        try:
            # Azure Database resource ID
            resource_id = (
                f"/subscriptions/{self.azure_config.subscription_id}"
                f"/resourceGroups/{self.azure_config.resource_group}"
                f"/providers/Microsoft.DBforPostgreSQL/flexibleServers/{self.azure_config.server_name}"
            )
            
            # Query Azure Monitor metrics
            metrics_to_collect = [
                "cpu_percent",
                "memory_percent", 
                "storage_percent",
                "active_connections",
                "connections_failed",
                "network_bytes_ingress",
                "network_bytes_egress",
                "io_consumption_percent"
            ]
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            for metric_name in metrics_to_collect:
                try:
                    response = self.azure_monitor_client.query_metrics(
                        resource_id=resource_id,
                        metric_names=[metric_name],
                        timespan=(start_time, end_time),
                        granularity=timedelta(minutes=1)
                    )
                    
                    # Process and store metrics
                    for metric in response.metrics:
                        for time_series in metric.timeseries:
                            for data_point in time_series.data:
                                if data_point.average is not None:
                                    AZURE_METRICS.labels(
                                        metric_name=metric_name,
                                        server_name=self.azure_config.server_name
                                    ).set(data_point.average)
                                    
                                    # Log critical thresholds
                                    if metric_name == "cpu_percent" and data_point.average > 80:
                                        logger.warning(
                                            "High CPU usage detected",
                                            cpu_percent=data_point.average,
                                            timestamp=data_point.timestamp
                                        )
                                    elif metric_name == "storage_percent" and data_point.average > 90:
                                        logger.critical(
                                            "Storage almost full",
                                            storage_percent=data_point.average,
                                            timestamp=data_point.timestamp
                                        )
                
                except Exception as e:
                    logger.warning(f"Failed to collect Azure metric {metric_name}", error=str(e))
            
        except Exception as e:
            logger.error("Failed to collect Azure metrics", error=str(e))
    
    async def setup_azure_alerts(self):
        """Configure Azure Monitor alerts for database health."""
        # This would typically be done via ARM templates or Azure CLI
        # For now, we'll log the alert configuration that should be created
        
        alert_rules = [
            {
                'name': 'High CPU Usage',
                'metric': 'cpu_percent',
                'threshold': 80,
                'operator': 'GreaterThan',
                'severity': 2,
                'description': 'CPU usage is above 80%'
            },
            {
                'name': 'High Memory Usage', 
                'metric': 'memory_percent',
                'threshold': 85,
                'operator': 'GreaterThan',
                'severity': 2,
                'description': 'Memory usage is above 85%'
            },
            {
                'name': 'Storage Almost Full',
                'metric': 'storage_percent', 
                'threshold': 90,
                'operator': 'GreaterThan',
                'severity': 1,
                'description': 'Storage usage is above 90%'
            },
            {
                'name': 'Connection Failures',
                'metric': 'connections_failed',
                'threshold': 10,
                'operator': 'GreaterThan',
                'severity': 2,
                'description': 'More than 10 connection failures in 5 minutes'
            },
            {
                'name': 'High Active Connections',
                'metric': 'active_connections',
                'threshold': 80,  # Percentage of max_connections
                'operator': 'GreaterThan',
                'severity': 3,
                'description': 'Active connections approaching limit'
            }
        ]
        
        logger.info(
            "Azure Monitor alert rules should be configured",
            alert_count=len(alert_rules),
            rules=[rule['name'] for rule in alert_rules]
        )
        
        return alert_rules
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from SQL query (simplified)."""
        import re
        
        # Simple regex to extract table names from common patterns
        patterns = [
            r'FROM\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'INSERT\s+INTO\s+(\w+)',
            r'DELETE\s+FROM\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary and health status."""
        try:
            # Get connection health
            connection_health = await self.connection_manager.health_check()
            
            # Get Azure connection info
            azure_info = await self.connection_manager.get_azure_connection_info()
            
            # Calculate health score
            healthy_connections = sum(1 for h in connection_health.values() if h)
            total_connections = len(connection_health)
            health_score = (healthy_connections / total_connections) if total_connections > 0 else 0
            
            return {
                "monitoring_active": self._monitoring_active,
                "connection_health": connection_health,
                "azure_info": azure_info,
                "health_score": health_score,
                "azure_monitor_available": self.azure_monitor_client is not None,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get monitoring summary", error=str(e))
            return {
                "monitoring_active": self._monitoring_active,
                "error": str(e),
                "health_score": 0,
                "last_check": datetime.utcnow().isoformat()
            }


class DatabasePerformanceAnalyzer:
    """Analyze database performance trends and anomalies."""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
    
    async def analyze_query_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze query performance over time period."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Check if pg_stat_statements is available
                ext_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    )
                """)
                
                if not ext_exists:
                    return {"error": "pg_stat_statements extension not available"}
                
                # Get query performance statistics
                query_stats = await conn.fetch("""
                    SELECT 
                        LEFT(query, 100) as query_sample,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        max_exec_time,
                        stddev_exec_time,
                        rows,
                        100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS cache_hit_ratio
                    FROM pg_stat_statements 
                    WHERE calls > 10  -- Filter out one-off queries
                    ORDER BY total_exec_time DESC 
                    LIMIT 20
                """)
                
                # Calculate performance metrics
                total_queries = sum(stat['calls'] for stat in query_stats)
                total_time = sum(stat['total_exec_time'] for stat in query_stats)
                avg_cache_hit_ratio = sum(
                    stat['cache_hit_ratio'] for stat in query_stats 
                    if stat['cache_hit_ratio'] is not None
                ) / len(query_stats) if query_stats else 0
                
                return {
                    "analysis_period_hours": hours,
                    "total_queries_analyzed": total_queries,
                    "total_execution_time_ms": total_time,
                    "average_cache_hit_ratio": avg_cache_hit_ratio,
                    "top_queries_by_total_time": [dict(stat) for stat in query_stats[:10]],
                    "performance_summary": {
                        "queries_over_1s": len([s for s in query_stats if s['mean_exec_time'] > 1000]),
                        "high_variance_queries": len([s for s in query_stats if s['stddev_exec_time'] > s['mean_exec_time']]),
                        "low_cache_hit_queries": len([s for s in query_stats if s['cache_hit_ratio'] and s['cache_hit_ratio'] < 95])
                    }
                }
                
        except Exception as e:
            logger.error("Failed to analyze query performance", error=str(e))
            return {"error": str(e)}
    
    async def analyze_table_growth(self) -> Dict[str, Any]:
        """Analyze table growth patterns."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Get table size information
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(relid)) as size,
                        pg_total_relation_size(relid) as size_bytes,
                        n_live_tup as row_count,
                        n_tup_ins as total_inserts,
                        n_tup_upd as total_updates,
                        n_tup_del as total_deletes
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(relid) DESC
                """)
                
                # Calculate growth metrics
                total_size = sum(stat['size_bytes'] for stat in table_sizes)
                total_rows = sum(stat['row_count'] for stat in table_sizes)
                
                # Identify high-activity tables
                high_activity_tables = [
                    stat for stat in table_sizes
                    if (stat['total_inserts'] + stat['total_updates'] + stat['total_deletes']) > 1000
                ]
                
                return {
                    "total_database_size_bytes": total_size,
                    "total_rows": total_rows,
                    "table_count": len(table_sizes),
                    "largest_tables": [dict(stat) for stat in table_sizes[:5]],
                    "high_activity_tables": [dict(stat) for stat in high_activity_tables],
                    "growth_indicators": {
                        "tables_over_1gb": len([s for s in table_sizes if s['size_bytes'] > 1024**3]),
                        "tables_over_1m_rows": len([s for s in table_sizes if s['row_count'] > 1000000]),
                        "tables_with_high_churn": len(high_activity_tables)
                    }
                }
                
        except Exception as e:
            logger.error("Failed to analyze table growth", error=str(e))
            return {"error": str(e)}
