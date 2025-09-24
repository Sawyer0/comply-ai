"""Azure Monitor integration for database performance and backup monitoring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

try:
    from azure.identity import DefaultAzureCredential
    from azure.monitor.query import LogsQueryClient, MetricsQueryClient
    from azure.mgmt.monitor import MonitorManagementClient
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    LogsQueryClient = None
    MetricsQueryClient = None
    MonitorManagementClient = None
    AZURE_MONITOR_AVAILABLE = False

from ..storage.database.azure_config import AzureDatabaseConnectionManager

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseMetric:
    """Database performance metric."""
    
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    resource_id: str
    metadata: Dict[str, Any] = None


@dataclass
class AlertRule:
    """Azure Monitor alert rule configuration."""
    
    name: str
    description: str
    metric_name: str
    threshold: float
    operator: str  # GreaterThan, LessThan, etc.
    severity: int  # 0-4
    enabled: bool = True


class AzureMonitorIntegration:
    """Azure Monitor integration for database monitoring."""
    
    def __init__(self, subscription_id: str, resource_group: str, 
                 server_name: str, log_analytics_workspace_id: Optional[str] = None):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.server_name = server_name
        self.log_analytics_workspace_id = log_analytics_workspace_id
        
        if not AZURE_MONITOR_AVAILABLE:
            logger.warning("Azure Monitor SDK not available")
            return
            
        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential)
        self.metrics_client = MetricsQueryClient(self.credential)
        self.monitor_client = MonitorManagementClient(self.credential, subscription_id)
        
        self.resource_id = (
            f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/"
            f"providers/Microsoft.DBforPostgreSQL/flexibleServers/{server_name}"
        )
    
    async def get_database_metrics(self, 
                                  start_time: datetime, 
                                  end_time: datetime,
                                  metrics: List[str] = None) -> List[DatabaseMetric]:
        """Get database performance metrics from Azure Monitor."""
        if not AZURE_MONITOR_AVAILABLE:
            logger.warning("Azure Monitor not available, returning empty metrics")
            return []
        
        if metrics is None:
            metrics = [
                "cpu_percent",
                "memory_percent", 
                "storage_percent",
                "active_connections",
                "connections_failed",
                "network_bytes_ingress",
                "network_bytes_egress"
            ]
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.metrics_client.query_resource,
                self.resource_id,
                metrics,
                timespan=(start_time, end_time),
                granularity=timedelta(minutes=5)
            )
            
            database_metrics = []
            for metric in response.metrics:
                for timeseries in metric.timeseries:
                    for data_point in timeseries.data:
                        if data_point.average is not None:
                            database_metrics.append(DatabaseMetric(
                                metric_name=metric.name.value,
                                value=data_point.average,
                                timestamp=data_point.timestamp,
                                unit=metric.unit.value,
                                resource_id=self.resource_id,
                                metadata={
                                    "resource_group": self.resource_group,
                                    "server_name": self.server_name
                                }
                            ))
            
            logger.info("Retrieved database metrics", 
                       metric_count=len(database_metrics),
                       time_range=f"{start_time} to {end_time}")
            
            return database_metrics
            
        except Exception as e:
            logger.error("Failed to get database metrics", error=str(e))
            return []
    
    async def get_query_performance_insights(self, 
                                           start_time: datetime,
                                           end_time: datetime) -> List[Dict[str, Any]]:
        """Get Query Performance Insights data."""
        if not AZURE_MONITOR_AVAILABLE or not self.log_analytics_workspace_id:
            logger.warning("Azure Monitor or Log Analytics not available")
            return []
        
        try:
            # KQL query for Query Performance Insights
            kql_query = """
            AzureDiagnostics
            | where ResourceProvider == "MICROSOFT.DBFORPOSTGRESQL"
            | where Category == "PostgreSQLLogs"
            | where TimeGenerated between (datetime({start_time}) .. datetime({end_time}))
            | where Message contains "duration:"
            | extend QueryDuration = extract(@"duration: ([0-9.]+) ms", 1, Message)
            | extend QueryText = extract(@"statement: (.+)", 1, Message)
            | where isnotempty(QueryDuration)
            | project TimeGenerated, QueryDuration=todouble(QueryDuration), QueryText, Resource
            | summarize 
                AvgDuration=avg(QueryDuration),
                MaxDuration=max(QueryDuration),
                QueryCount=count()
                by QueryText
            | order by AvgDuration desc
            | limit 50
            """.format(
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs_client.query_workspace,
                self.log_analytics_workspace_id,
                kql_query,
                timespan=(start_time, end_time)
            )
            
            query_insights = []
            for table in response.tables:
                for row in table.rows:
                    query_insights.append({
                        "query_text": row[0] if len(row) > 0 else "",
                        "avg_duration_ms": row[1] if len(row) > 1 else 0,
                        "max_duration_ms": row[2] if len(row) > 2 else 0,
                        "query_count": row[3] if len(row) > 3 else 0
                    })
            
            logger.info("Retrieved query performance insights", 
                       query_count=len(query_insights))
            
            return query_insights
            
        except Exception as e:
            logger.error("Failed to get query performance insights", error=str(e))
            return []
    
    async def create_database_alerts(self, alert_rules: List[AlertRule]) -> Dict[str, bool]:
        """Create Azure Monitor alert rules for database monitoring."""
        if not AZURE_MONITOR_AVAILABLE:
            logger.warning("Azure Monitor not available, cannot create alerts")
            return {}
        
        results = {}
        
        for rule in alert_rules:
            try:
                # Create metric alert rule
                alert_rule_params = {
                    "location": "global",
                    "description": rule.description,
                    "severity": rule.severity,
                    "enabled": rule.enabled,
                    "scopes": [self.resource_id],
                    "evaluation_frequency": "PT5M",  # 5 minutes
                    "window_size": "PT15M",  # 15 minutes
                    "criteria": {
                        "odata.type": "Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria",
                        "allOf": [{
                            "name": f"{rule.name}_condition",
                            "metric_name": rule.metric_name,
                            "operator": rule.operator,
                            "threshold": rule.threshold,
                            "time_aggregation": "Average"
                        }]
                    },
                    "actions": [{
                        "action_group_id": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/microsoft.insights/actionGroups/backup-alerts"
                    }]
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.monitor_client.metric_alerts.create_or_update,
                    self.resource_group,
                    rule.name,
                    alert_rule_params
                )
                
                results[rule.name] = True
                logger.info("Created alert rule", rule_name=rule.name)
                
            except Exception as e:
                logger.error("Failed to create alert rule", 
                           rule_name=rule.name, error=str(e))
                results[rule.name] = False
        
        return results
    
    async def get_backup_status_from_logs(self, 
                                        start_time: datetime,
                                        end_time: datetime) -> Dict[str, Any]:
        """Get backup status from Azure Monitor logs."""
        if not AZURE_MONITOR_AVAILABLE or not self.log_analytics_workspace_id:
            logger.warning("Azure Monitor or Log Analytics not available")
            return {}
        
        try:
            # KQL query for backup operations
            kql_query = """
            AzureDiagnostics
            | where ResourceProvider == "MICROSOFT.DBFORPOSTGRESQL"
            | where Category == "PostgreSQLLogs"
            | where TimeGenerated between (datetime({start_time}) .. datetime({end_time}))
            | where Message contains "backup" or Message contains "restore"
            | project TimeGenerated, Message, Resource, Level
            | order by TimeGenerated desc
            """.format(
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.logs_client.query_workspace,
                self.log_analytics_workspace_id,
                kql_query,
                timespan=(start_time, end_time)
            )
            
            backup_events = []
            for table in response.tables:
                for row in table.rows:
                    backup_events.append({
                        "timestamp": row[0] if len(row) > 0 else None,
                        "message": row[1] if len(row) > 1 else "",
                        "resource": row[2] if len(row) > 2 else "",
                        "level": row[3] if len(row) > 3 else ""
                    })
            
            # Analyze backup status
            successful_backups = len([e for e in backup_events if "success" in e["message"].lower()])
            failed_backups = len([e for e in backup_events if "fail" in e["message"].lower()])
            
            return {
                "total_backup_events": len(backup_events),
                "successful_backups": successful_backups,
                "failed_backups": failed_backups,
                "backup_success_rate": successful_backups / len(backup_events) if backup_events else 0,
                "recent_events": backup_events[:10]  # Last 10 events
            }
            
        except Exception as e:
            logger.error("Failed to get backup status from logs", error=str(e))
            return {}


class EnhancedAzureBackupMonitor:
    """Enhanced Azure backup monitoring with database performance integration."""
    
    def __init__(self, 
                 connection_manager: AzureDatabaseConnectionManager,
                 monitor_integration: AzureMonitorIntegration):
        self.connection_manager = connection_manager
        self.monitor_integration = monitor_integration
    
    async def get_comprehensive_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status including database performance."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "database_health": {},
            "performance_metrics": {},
            "backup_status": {},
            "alerts": {}
        }
        
        try:
            # Get database health
            status["database_health"] = await self.connection_manager.health_check()
            
            # Get Azure connection info
            connection_info = await self.connection_manager.get_azure_connection_info()
            status["database_health"]["connection_info"] = connection_info
            
            # Get performance metrics (last hour)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            metrics = await self.monitor_integration.get_database_metrics(start_time, end_time)
            if metrics:
                # Aggregate metrics
                cpu_metrics = [m for m in metrics if m.metric_name == "cpu_percent"]
                memory_metrics = [m for m in metrics if m.metric_name == "memory_percent"]
                connection_metrics = [m for m in metrics if m.metric_name == "active_connections"]
                
                status["performance_metrics"] = {
                    "avg_cpu_percent": sum(m.value for m in cpu_metrics) / len(cpu_metrics) if cpu_metrics else 0,
                    "avg_memory_percent": sum(m.value for m in memory_metrics) / len(memory_metrics) if memory_metrics else 0,
                    "avg_active_connections": sum(m.value for m in connection_metrics) / len(connection_metrics) if connection_metrics else 0,
                    "metric_count": len(metrics)
                }
            
            # Get backup status from logs
            backup_status = await self.monitor_integration.get_backup_status_from_logs(start_time, end_time)
            status["backup_status"] = backup_status
            
            # Get query performance insights
            query_insights = await self.monitor_integration.get_query_performance_insights(start_time, end_time)
            status["query_performance"] = {
                "slow_queries_count": len(query_insights),
                "top_slow_queries": query_insights[:5] if query_insights else []
            }
            
        except Exception as e:
            logger.error("Failed to get comprehensive backup status", error=str(e))
            status["error"] = str(e)
        
        return status
    
    async def setup_production_alerts(self) -> Dict[str, bool]:
        """Set up production-ready alert rules."""
        alert_rules = [
            AlertRule(
                name="high-cpu-usage",
                description="Database CPU usage is high",
                metric_name="cpu_percent",
                threshold=80.0,
                operator="GreaterThan",
                severity=2
            ),
            AlertRule(
                name="high-memory-usage", 
                description="Database memory usage is high",
                metric_name="memory_percent",
                threshold=85.0,
                operator="GreaterThan",
                severity=2
            ),
            AlertRule(
                name="storage-space-low",
                description="Database storage space is running low",
                metric_name="storage_percent",
                threshold=90.0,
                operator="GreaterThan",
                severity=1
            ),
            AlertRule(
                name="connection-failures",
                description="Database connection failures detected",
                metric_name="connections_failed",
                threshold=10.0,
                operator="GreaterThan",
                severity=2
            ),
            AlertRule(
                name="too-many-connections",
                description="Too many active database connections",
                metric_name="active_connections",
                threshold=80.0,
                operator="GreaterThan",
                severity=3
            )
        ]
        
        return await self.monitor_integration.create_database_alerts(alert_rules)


def create_default_monitor_integration(subscription_id: str, 
                                     resource_group: str,
                                     server_name: str,
                                     log_analytics_workspace_id: Optional[str] = None) -> AzureMonitorIntegration:
    """Create default Azure Monitor integration."""
    return AzureMonitorIntegration(
        subscription_id=subscription_id,
        resource_group=resource_group,
        server_name=server_name,
        log_analytics_workspace_id=log_analytics_workspace_id
    )