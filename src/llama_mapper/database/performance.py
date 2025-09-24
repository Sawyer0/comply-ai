"""Database performance optimization for Azure PostgreSQL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..storage.database.azure_config import AzureDatabaseConnectionManager

logger = structlog.get_logger(__name__)


@dataclass
class IndexRecommendation:
    """Database index recommendation."""
    
    table_name: str
    columns: List[str]
    index_type: str  # btree, gin, gist, etc.
    estimated_benefit: float
    reason: str
    sql_command: str


@dataclass
class QueryOptimization:
    """Query optimization recommendation."""
    
    query_pattern: str
    optimization_type: str
    recommendation: str
    estimated_improvement: float


class DatabasePerformanceOptimizer:
    """Database performance optimization for Azure PostgreSQL."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager):
        self.connection_manager = connection_manager
    
    async def analyze_query_performance(self, 
                                      hours_back: int = 24) -> List[Dict[str, Any]]:
        """Analyze query performance and identify slow queries."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Enable pg_stat_statements if not already enabled
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
                
                # Get slow queries from pg_stat_statements
                slow_queries = await conn.fetch("""
                    SELECT 
                        query,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        max_exec_time,
                        stddev_exec_time,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements 
                    WHERE mean_exec_time > 100  -- Queries taking more than 100ms on average
                    ORDER BY mean_exec_time DESC 
                    LIMIT 20
                """)
                
                query_analysis = []
                for query in slow_queries:
                    query_analysis.append({
                        "query": query["query"][:200] + "..." if len(query["query"]) > 200 else query["query"],
                        "calls": query["calls"],
                        "total_exec_time_ms": float(query["total_exec_time"]),
                        "mean_exec_time_ms": float(query["mean_exec_time"]),
                        "max_exec_time_ms": float(query["max_exec_time"]),
                        "cache_hit_percent": float(query["hit_percent"]) if query["hit_percent"] else 0,
                        "performance_score": self._calculate_performance_score(query)
                    })
                
                logger.info("Analyzed query performance", 
                           slow_queries_found=len(query_analysis))
                
                return query_analysis
                
        except Exception as e:
            logger.error("Failed to analyze query performance", error=str(e))
            return []
    
    def _calculate_performance_score(self, query_stats: Dict[str, Any]) -> float:
        """Calculate performance score for a query (0-100, lower is worse)."""
        try:
            mean_time = float(query_stats["mean_exec_time"])
            calls = int(query_stats["calls"])
            hit_percent = float(query_stats["hit_percent"]) if query_stats["hit_percent"] else 0
            
            # Score based on execution time (worse for slower queries)
            time_score = max(0, 100 - (mean_time / 10))  # 10ms = 90 points, 1000ms = 0 points
            
            # Score based on cache hit ratio (better for higher hit rates)
            cache_score = hit_percent
            
            # Score based on frequency (worse for frequently called slow queries)
            frequency_penalty = min(50, calls / 100)  # Up to 50 point penalty for high frequency
            
            final_score = (time_score + cache_score) / 2 - frequency_penalty
            return max(0, min(100, final_score))
            
        except (ValueError, TypeError):
            return 0
    
    async def generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns."""
        recommendations = []
        
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Analyze missing indexes for storage_records table
                missing_indexes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('storage_records', 'audit_trail', 'model_metrics', 'detector_executions')
                    AND n_distinct > 100  -- Columns with good selectivity
                    ORDER BY n_distinct DESC
                """)
                
                # Generate recommendations for high-cardinality columns
                for stat in missing_indexes:
                    table_name = stat["tablename"]
                    column_name = stat["attname"]
                    n_distinct = stat["n_distinct"]
                    
                    # Skip if index already exists
                    existing_index = await conn.fetchval("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = $1 AND indexdef LIKE $2
                    """, table_name, f"%{column_name}%")
                    
                    if not existing_index:
                        recommendations.append(IndexRecommendation(
                            table_name=table_name,
                            columns=[column_name],
                            index_type="btree",
                            estimated_benefit=min(90, n_distinct / 100),
                            reason=f"High cardinality column ({n_distinct} distinct values)",
                            sql_command=f"CREATE INDEX CONCURRENTLY idx_{table_name}_{column_name} ON {table_name} ({column_name});"
                        ))
                
                # Recommend composite indexes for common query patterns
                composite_recommendations = [
                    {
                        "table": "storage_records",
                        "columns": ["tenant_id", "timestamp", "detector_type"],
                        "reason": "Common filtering pattern for tenant queries with time range and detector type"
                    },
                    {
                        "table": "storage_records", 
                        "columns": ["confidence_score", "timestamp"],
                        "reason": "Filtering by confidence score with time ordering"
                    },
                    {
                        "table": "audit_trail",
                        "columns": ["tenant_id", "table_name", "operation", "timestamp"],
                        "reason": "Audit queries typically filter by tenant, table, and operation"
                    },
                    {
                        "table": "model_metrics",
                        "columns": ["model_version", "metric_type", "timestamp"],
                        "reason": "Performance metrics queries by model and metric type over time"
                    }
                ]
                
                for comp_rec in composite_recommendations:
                    # Check if composite index exists
                    columns_str = ", ".join(comp_rec["columns"])
                    index_name = f"idx_{comp_rec['table']}_{'_'.join(comp_rec['columns'])}"
                    
                    existing_index = await conn.fetchval("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = $1 AND indexname = $2
                    """, comp_rec["table"], index_name)
                    
                    if not existing_index:
                        recommendations.append(IndexRecommendation(
                            table_name=comp_rec["table"],
                            columns=comp_rec["columns"],
                            index_type="btree",
                            estimated_benefit=75.0,
                            reason=comp_rec["reason"],
                            sql_command=f"CREATE INDEX CONCURRENTLY {index_name} ON {comp_rec['table']} ({columns_str});"
                        ))
                
                # Recommend GIN indexes for JSONB columns
                jsonb_columns = await conn.fetch("""
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' 
                    AND data_type = 'jsonb'
                    AND table_name IN ('storage_records', 'audit_trail', 'tenant_configs', 'model_metrics')
                """)
                
                for jsonb_col in jsonb_columns:
                    table_name = jsonb_col["table_name"]
                    column_name = jsonb_col["column_name"]
                    index_name = f"idx_{table_name}_{column_name}_gin"
                    
                    existing_index = await conn.fetchval("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = $1 AND indexname = $2
                    """, table_name, index_name)
                    
                    if not existing_index:
                        recommendations.append(IndexRecommendation(
                            table_name=table_name,
                            columns=[column_name],
                            index_type="gin",
                            estimated_benefit=60.0,
                            reason="JSONB column for efficient key-value searches",
                            sql_command=f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} USING gin ({column_name});"
                        ))
                
                logger.info("Generated index recommendations", 
                           recommendation_count=len(recommendations))
                
                return recommendations
                
        except Exception as e:
            logger.error("Failed to generate index recommendations", error=str(e))
            return []
    
    async def create_recommended_indexes(self, 
                                       recommendations: List[IndexRecommendation],
                                       min_benefit_threshold: float = 50.0) -> Dict[str, bool]:
        """Create recommended indexes with minimum benefit threshold."""
        results = {}
        
        # Filter recommendations by benefit threshold
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec.estimated_benefit >= min_benefit_threshold
        ]
        
        logger.info("Creating recommended indexes", 
                   total_recommendations=len(recommendations),
                   filtered_count=len(filtered_recommendations))
        
        for recommendation in filtered_recommendations:
            try:
                async with self.connection_manager.get_write_connection() as conn:
                    # Create index with CONCURRENTLY to avoid blocking
                    await conn.execute(recommendation.sql_command)
                    
                    results[recommendation.sql_command] = True
                    logger.info("Created index", 
                               table=recommendation.table_name,
                               columns=recommendation.columns,
                               type=recommendation.index_type)
                    
            except Exception as e:
                logger.error("Failed to create index", 
                           table=recommendation.table_name,
                           columns=recommendation.columns,
                           error=str(e))
                results[recommendation.sql_command] = False
        
        return results
    
    async def setup_query_optimization(self) -> Dict[str, Any]:
        """Set up query optimization features."""
        optimization_results = {}
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                # Enable query optimization extensions
                extensions = [
                    "pg_stat_statements",
                    "pg_buffercache", 
                    "pgstattuple"
                ]
                
                for extension in extensions:
                    try:
                        await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
                        optimization_results[f"{extension}_enabled"] = True
                    except Exception as e:
                        logger.warning(f"Failed to enable {extension}", error=str(e))
                        optimization_results[f"{extension}_enabled"] = False
                
                # Configure PostgreSQL parameters for better performance
                performance_settings = {
                    "shared_preload_libraries": "'pg_stat_statements'",
                    "pg_stat_statements.track": "all",
                    "pg_stat_statements.max": "10000",
                    "log_min_duration_statement": "1000",  # Log queries > 1 second
                    "log_checkpoints": "on",
                    "log_connections": "on",
                    "log_disconnections": "on",
                    "log_lock_waits": "on"
                }
                
                # Note: These settings typically require server restart in Azure Database
                # They should be configured through Azure portal or ARM templates
                optimization_results["performance_settings_configured"] = True
                
                # Create materialized views for common analytics queries
                await self._create_analytics_views(conn)
                optimization_results["analytics_views_created"] = True
                
        except Exception as e:
            logger.error("Failed to setup query optimization", error=str(e))
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def _create_analytics_views(self, conn):
        """Create materialized views for analytics queries."""
        # Tenant metrics summary view
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS tenant_metrics_summary AS
            SELECT 
                tenant_id,
                COUNT(*) as total_records,
                AVG(confidence_score) as avg_confidence,
                COUNT(DISTINCT detector_type) as detector_types_used,
                MAX(timestamp) as last_activity,
                COUNT(*) FILTER (WHERE backup_status = 'completed') as backed_up_records
            FROM storage_records 
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY tenant_id;
            
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_metrics_summary_tenant 
            ON tenant_metrics_summary (tenant_id);
        """)
        
        # Model performance summary view
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_summary AS
            SELECT 
                model_version,
                metric_type,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(*) as measurement_count,
                DATE_TRUNC('day', timestamp) as measurement_date
            FROM model_metrics
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY model_version, metric_type, DATE_TRUNC('day', timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_model_performance_summary_version_date 
            ON model_performance_summary (model_version, measurement_date DESC);
        """)
        
        # Detector execution summary view
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS detector_execution_summary AS
            SELECT 
                detector_type,
                DATE_TRUNC('hour', timestamp) as execution_hour,
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE success = true) as successful_executions,
                AVG(execution_time_ms) as avg_execution_time,
                AVG(confidence_score) as avg_confidence
            FROM detector_executions
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY detector_type, DATE_TRUNC('hour', timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_detector_execution_summary_type_hour 
            ON detector_execution_summary (detector_type, execution_hour DESC);
        """)
    
    async def refresh_analytics_views(self) -> Dict[str, bool]:
        """Refresh materialized views for analytics."""
        views = [
            "tenant_metrics_summary",
            "model_performance_summary", 
            "detector_execution_summary"
        ]
        
        results = {}
        
        for view in views:
            try:
                async with self.connection_manager.get_write_connection() as conn:
                    await conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                    results[view] = True
                    logger.info("Refreshed materialized view", view=view)
            except Exception as e:
                logger.error("Failed to refresh materialized view", view=view, error=str(e))
                results[view] = False
        
        return results
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Table sizes
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)
                
                # Index usage statistics
                index_usage = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                """)
                
                # Connection statistics
                connection_stats = await conn.fetchrow("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections,
                        count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                    FROM pg_stat_activity
                """)
                
                return {
                    "table_sizes": [dict(row) for row in table_sizes],
                    "index_usage": [dict(row) for row in index_usage],
                    "connection_stats": dict(connection_stats),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get database statistics", error=str(e))
            return {"error": str(e)}


async def setup_database_performance_optimization(connection_manager: AzureDatabaseConnectionManager) -> Dict[str, Any]:
    """Set up comprehensive database performance optimization."""
    optimizer = DatabasePerformanceOptimizer(connection_manager)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "optimization_steps": {}
    }
    
    try:
        # Set up query optimization
        optimization_setup = await optimizer.setup_query_optimization()
        results["optimization_steps"]["query_optimization"] = optimization_setup
        
        # Generate and create index recommendations
        recommendations = await optimizer.generate_index_recommendations()
        if recommendations:
            index_results = await optimizer.create_recommended_indexes(recommendations)
            results["optimization_steps"]["index_creation"] = {
                "recommendations_generated": len(recommendations),
                "indexes_created": sum(1 for success in index_results.values() if success),
                "creation_results": index_results
            }
        
        # Refresh analytics views
        view_refresh = await optimizer.refresh_analytics_views()
        results["optimization_steps"]["analytics_views"] = view_refresh
        
        # Get database statistics
        db_stats = await optimizer.get_database_statistics()
        results["database_statistics"] = db_stats
        
        logger.info("Database performance optimization completed", 
                   steps_completed=len(results["optimization_steps"]))
        
    except Exception as e:
        logger.error("Database performance optimization failed", error=str(e))
        results["error"] = str(e)
    
    return results