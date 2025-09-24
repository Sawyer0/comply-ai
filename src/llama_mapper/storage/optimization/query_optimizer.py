"""Database query optimization and performance tuning for Azure PostgreSQL."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import structlog

from ..database.azure_config import AzureDatabaseConnectionManager

logger = structlog.get_logger(__name__)


class QueryOptimizer:
    """Database query optimization and performance tuning."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.materialized_views: List[str] = []
        
    async def create_materialized_views(self) -> Dict[str, Any]:
        """Create materialized views for analytics queries."""
        views_created = []
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                # Tenant performance summary view
                await conn.execute("""
                    DROP MATERIALIZED VIEW IF EXISTS tenant_performance_summary;
                    
                    CREATE MATERIALIZED VIEW tenant_performance_summary AS
                    SELECT 
                        tenant_id,
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as total_requests,
                        AVG(confidence_score) as avg_confidence,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY confidence_score) as p95_confidence,
                        COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_count,
                        COUNT(DISTINCT detector_type) as unique_detectors,
                        AVG(CASE WHEN backup_status = 'completed' THEN 1 ELSE 0 END) as backup_success_rate
                    FROM storage_records 
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                    AND confidence_score IS NOT NULL
                    GROUP BY tenant_id, DATE_TRUNC('hour', timestamp);
                    
                    CREATE UNIQUE INDEX ON tenant_performance_summary (tenant_id, hour);
                    CREATE INDEX ON tenant_performance_summary (hour DESC);
                """)
                views_created.append("tenant_performance_summary")
                
                # Detector performance summary view
                await conn.execute("""
                    DROP MATERIALIZED VIEW IF EXISTS detector_performance_summary;
                    
                    CREATE MATERIALIZED VIEW detector_performance_summary AS
                    SELECT 
                        detector_type,
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as execution_count,
                        AVG(execution_time_ms) as avg_execution_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
                        COUNT(CASE WHEN success THEN 1 END)::float / COUNT(*) as success_rate,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as error_count
                    FROM detector_executions 
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY detector_type, DATE_TRUNC('hour', timestamp);
                    
                    CREATE UNIQUE INDEX ON detector_performance_summary (detector_type, hour);
                    CREATE INDEX ON detector_performance_summary (hour DESC);
                    CREATE INDEX ON detector_performance_summary (success_rate);
                """)
                views_created.append("detector_performance_summary")
                
                # Model metrics summary view
                await conn.execute("""
                    DROP MATERIALIZED VIEW IF EXISTS model_metrics_summary;
                    
                    CREATE MATERIALIZED VIEW model_metrics_summary AS
                    SELECT 
                        model_version,
                        metric_type,
                        DATE_TRUNC('day', timestamp) as day,
                        AVG(metric_value) as avg_value,
                        MIN(metric_value) as min_value,
                        MAX(metric_value) as max_value,
                        STDDEV(metric_value) as stddev_value,
                        COUNT(*) as sample_count
                    FROM model_metrics 
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                    GROUP BY model_version, metric_type, DATE_TRUNC('day', timestamp);
                    
                    CREATE UNIQUE INDEX ON model_metrics_summary (model_version, metric_type, day);
                    CREATE INDEX ON model_metrics_summary (day DESC);
                """)
                views_created.append("model_metrics_summary")
                
                # Audit trail summary view
                await conn.execute("""
                    DROP MATERIALIZED VIEW IF EXISTS audit_trail_summary;
                    
                    CREATE MATERIALIZED VIEW audit_trail_summary AS
                    SELECT 
                        tenant_id,
                        table_name,
                        operation,
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as operation_count,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT correlation_id) as unique_requests
                    FROM audit_trail 
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY tenant_id, table_name, operation, DATE_TRUNC('hour', timestamp);
                    
                    CREATE UNIQUE INDEX ON audit_trail_summary (tenant_id, table_name, operation, hour);
                    CREATE INDEX ON audit_trail_summary (hour DESC);
                """)
                views_created.append("audit_trail_summary")
                
                self.materialized_views.extend(views_created)
                
                logger.info("Materialized views created", views=views_created)
                
        except Exception as e:
            logger.error("Failed to create materialized views", error=str(e))
            raise
        
        return {
            "views_created": views_created,
            "total_views": len(views_created),
            "creation_time": datetime.utcnow().isoformat()
        }
    
    async def refresh_materialized_views(self) -> Dict[str, Any]:
        """Refresh all materialized views."""
        refreshed_views = []
        failed_views = []
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                for view_name in self.materialized_views:
                    try:
                        start_time = time.time()
                        await conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view_name}")
                        refresh_time = time.time() - start_time
                        
                        refreshed_views.append({
                            "view": view_name,
                            "refresh_time_seconds": refresh_time
                        })
                        
                        logger.debug("Materialized view refreshed", view=view_name, time=refresh_time)
                        
                    except Exception as e:
                        failed_views.append({
                            "view": view_name,
                            "error": str(e)
                        })
                        logger.warning("Failed to refresh materialized view", view=view_name, error=str(e))
                
        except Exception as e:
            logger.error("Failed to refresh materialized views", error=str(e))
            raise
        
        return {
            "refreshed_views": refreshed_views,
            "failed_views": failed_views,
            "refresh_time": datetime.utcnow().isoformat()
        }
    
    async def create_performance_indexes(self) -> Dict[str, Any]:
        """Create performance-optimized indexes."""
        indexes_created = []
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                # Composite indexes for common query patterns
                index_definitions = [
                    # Storage records performance indexes
                    {
                        "name": "idx_storage_records_tenant_detector_timestamp",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_tenant_detector_timestamp 
                            ON storage_records (tenant_id, detector_type, timestamp DESC)
                        """
                    },
                    {
                        "name": "idx_storage_records_confidence_timestamp",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_confidence_timestamp 
                            ON storage_records (confidence_score DESC, timestamp DESC) 
                            WHERE confidence_score IS NOT NULL
                        """
                    },
                    {
                        "name": "idx_storage_records_backup_status_timestamp",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_backup_status_timestamp 
                            ON storage_records (backup_status, timestamp DESC)
                        """
                    },
                    {
                        "name": "idx_storage_records_source_hash",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_storage_records_source_hash 
                            ON storage_records (source_data_hash) 
                            WHERE source_data_hash IS NOT NULL
                        """
                    },
                    
                    # Detector executions performance indexes
                    {
                        "name": "idx_detector_executions_tenant_detector_success",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_executions_tenant_detector_success 
                            ON detector_executions (tenant_id, detector_type, success, timestamp DESC)
                        """
                    },
                    {
                        "name": "idx_detector_executions_execution_time",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detector_executions_execution_time 
                            ON detector_executions (execution_time_ms DESC, timestamp DESC)
                        """
                    },
                    
                    # Model metrics performance indexes
                    {
                        "name": "idx_model_metrics_tenant_model_type",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_metrics_tenant_model_type 
                            ON model_metrics (tenant_id, model_version, metric_type, timestamp DESC)
                        """
                    },
                    
                    # Audit trail performance indexes
                    {
                        "name": "idx_audit_trail_tenant_table_operation",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_trail_tenant_table_operation 
                            ON audit_trail (tenant_id, table_name, operation, timestamp DESC)
                        """
                    },
                    {
                        "name": "idx_audit_trail_user_timestamp",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_trail_user_timestamp 
                            ON audit_trail (user_id, timestamp DESC) 
                            WHERE user_id IS NOT NULL
                        """
                    },
                    
                    # Tenant configs indexes
                    {
                        "name": "idx_tenant_configs_threshold_encryption",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_configs_threshold_encryption 
                            ON tenant_configs (confidence_threshold, encryption_enabled)
                        """
                    }
                ]
                
                for index_def in index_definitions:
                    try:
                        start_time = time.time()
                        await conn.execute(index_def["sql"])
                        creation_time = time.time() - start_time
                        
                        indexes_created.append({
                            "name": index_def["name"],
                            "creation_time_seconds": creation_time
                        })
                        
                        logger.debug("Performance index created", index=index_def["name"], time=creation_time)
                        
                    except Exception as e:
                        logger.warning("Failed to create index", index=index_def["name"], error=str(e))
                
                logger.info("Performance indexes created", count=len(indexes_created))
                
        except Exception as e:
            logger.error("Failed to create performance indexes", error=str(e))
            raise
        
        return {
            "indexes_created": indexes_created,
            "total_indexes": len(indexes_created),
            "creation_time": datetime.utcnow().isoformat()
        }
    
    async def create_partitions(self) -> Dict[str, Any]:
        """Create time-based partitions for large tables."""
        partitions_created = []
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                # Create partitions for the next 12 months
                current_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                for i in range(12):
                    start_date = current_date + timedelta(days=32 * i)  # Rough month calculation
                    start_date = start_date.replace(day=1)  # First day of month
                    
                    end_date = start_date + timedelta(days=32)
                    end_date = end_date.replace(day=1)  # First day of next month
                    
                    partition_name = f"storage_records_y{start_date.year}m{start_date.month:02d}"
                    
                    try:
                        await conn.execute(f"""
                            CREATE TABLE IF NOT EXISTS {partition_name} 
                            PARTITION OF storage_records
                            FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}')
                        """)
                        
                        # Create indexes on partition
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS {partition_name}_tenant_timestamp 
                            ON {partition_name} (tenant_id, timestamp DESC)
                        """)
                        
                        await conn.execute(f"""
                            CREATE INDEX IF NOT EXISTS {partition_name}_detector_confidence 
                            ON {partition_name} (detector_type, confidence_score DESC) 
                            WHERE confidence_score IS NOT NULL
                        """)
                        
                        partitions_created.append({
                            "name": partition_name,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat()
                        })
                        
                        logger.debug("Partition created", partition=partition_name)
                        
                    except Exception as e:
                        logger.warning("Failed to create partition", partition=partition_name, error=str(e))
                
                logger.info("Partitions created", count=len(partitions_created))
                
        except Exception as e:
            logger.error("Failed to create partitions", error=str(e))
            raise
        
        return {
            "partitions_created": partitions_created,
            "total_partitions": len(partitions_created),
            "creation_time": datetime.utcnow().isoformat()
        }
    
    async def optimize_azure_parameters(self) -> Dict[str, Any]:
        """Optimize Azure Database parameters for workload."""
        parameters_set = []
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                # Check current parameter values
                current_params = await conn.fetch("""
                    SELECT name, setting, unit, category 
                    FROM pg_settings 
                    WHERE name IN (
                        'shared_preload_libraries',
                        'work_mem',
                        'maintenance_work_mem',
                        'effective_cache_size',
                        'random_page_cost',
                        'effective_io_concurrency',
                        'checkpoint_completion_target',
                        'wal_buffers',
                        'default_statistics_target',
                        'max_connections'
                    )
                """)
                
                # Recommended optimizations for OLTP workload
                optimizations = {
                    'work_mem': '8MB',  # Increased for complex queries
                    'maintenance_work_mem': '256MB',  # For maintenance operations
                    'effective_cache_size': '1GB',  # Assuming 2GB available memory
                    'random_page_cost': '1.1',  # SSD storage
                    'effective_io_concurrency': '200',  # SSD concurrency
                    'checkpoint_completion_target': '0.9',  # Spread checkpoints
                    'default_statistics_target': '100',  # Better query planning
                }
                
                for param, value in optimizations.items():
                    try:
                        # Note: Some parameters require server restart on Azure
                        await conn.execute(f"SET {param} = '{value}'")
                        parameters_set.append({
                            "parameter": param,
                            "value": value,
                            "scope": "session"
                        })
                        
                        logger.debug("Parameter optimized", parameter=param, value=value)
                        
                    except Exception as e:
                        logger.warning("Could not set parameter", parameter=param, error=str(e))
                
                logger.info("Azure parameters optimized", count=len(parameters_set))
                
        except Exception as e:
            logger.error("Failed to optimize Azure parameters", error=str(e))
            raise
        
        return {
            "parameters_set": parameters_set,
            "optimization_time": datetime.utcnow().isoformat(),
            "note": "Some parameters may require server restart to take effect"
        }
    
    async def enable_azure_extensions(self) -> Dict[str, Any]:
        """Enable required PostgreSQL extensions on Azure."""
        extensions_enabled = []
        
        extensions_to_enable = [
            'pg_stat_statements',  # Query performance monitoring
            'pg_buffercache',      # Buffer cache monitoring
            'pgcrypto',           # Encryption functions
            'uuid-ossp',          # UUID generation
            'btree_gin',          # GIN indexes for better performance
            'pg_trgm'             # Trigram matching for text search
        ]
        
        try:
            async with self.connection_manager.get_write_connection() as conn:
                for extension in extensions_to_enable:
                    try:
                        await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}"')
                        
                        # Verify extension is installed
                        result = await conn.fetchrow(
                            "SELECT extname, extversion FROM pg_extension WHERE extname = $1",
                            extension
                        )
                        
                        if result:
                            extensions_enabled.append({
                                "name": extension,
                                "version": result['extversion']
                            })
                            logger.debug("Extension enabled", extension=extension, version=result['extversion'])
                        
                    except Exception as e:
                        logger.warning("Could not enable extension", extension=extension, error=str(e))
                
                logger.info("Azure extensions enabled", count=len(extensions_enabled))
                
        except Exception as e:
            logger.error("Failed to enable Azure extensions", error=str(e))
            raise
        
        return {
            "extensions_enabled": extensions_enabled,
            "total_extensions": len(extensions_enabled),
            "enable_time": datetime.utcnow().isoformat()
        }
    
    async def analyze_query_plans(self) -> Dict[str, Any]:
        """Analyze query execution plans for optimization opportunities."""
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
                
                # Get slow queries with their plans
                slow_queries = await conn.fetch("""
                    SELECT 
                        LEFT(query, 200) as query_sample,
                        calls,
                        total_exec_time,
                        mean_exec_time,
                        max_exec_time,
                        rows,
                        100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS cache_hit_ratio
                    FROM pg_stat_statements 
                    WHERE mean_exec_time > 100  -- Queries slower than 100ms
                    AND calls > 5  -- Called multiple times
                    ORDER BY total_exec_time DESC 
                    LIMIT 10
                """)
                
                query_analysis = []
                for query_stat in slow_queries:
                    # Analyze specific query patterns
                    query_sample = query_stat['query_sample']
                    
                    analysis = {
                        "query_sample": query_sample,
                        "performance_metrics": {
                            "calls": query_stat['calls'],
                            "mean_exec_time_ms": query_stat['mean_exec_time'],
                            "max_exec_time_ms": query_stat['max_exec_time'],
                            "cache_hit_ratio": query_stat['cache_hit_ratio']
                        },
                        "optimization_suggestions": []
                    }
                    
                    # Add optimization suggestions based on patterns
                    if query_stat['cache_hit_ratio'] and query_stat['cache_hit_ratio'] < 95:
                        analysis["optimization_suggestions"].append("Low cache hit ratio - consider adding indexes")
                    
                    if query_stat['mean_exec_time'] > 1000:
                        analysis["optimization_suggestions"].append("Very slow query - review query structure and indexes")
                    
                    if "ORDER BY" in query_sample.upper() and "LIMIT" in query_sample.upper():
                        analysis["optimization_suggestions"].append("Consider composite index for ORDER BY + LIMIT queries")
                    
                    query_analysis.append(analysis)
                
                return {
                    "slow_queries_analyzed": len(query_analysis),
                    "query_analysis": query_analysis,
                    "analysis_time": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to analyze query plans", error=str(e))
            return {"error": str(e)}
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations."""
        recommendations = {
            "immediate_actions": [],
            "performance_improvements": [],
            "monitoring_setup": [],
            "maintenance_tasks": []
        }
        
        try:
            async with self.connection_manager.get_read_connection() as conn:
                # Check table sizes
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(relid)) as size,
                        pg_total_relation_size(relid) as size_bytes,
                        n_live_tup as row_count
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(relid) DESC
                """)
                
                # Analyze table sizes and suggest optimizations
                for table in table_sizes:
                    if table['size_bytes'] > 1024**3:  # > 1GB
                        recommendations["immediate_actions"].append(
                            f"Large table detected: {table['tablename']} ({table['size']}) - consider partitioning"
                        )
                    
                    if table['row_count'] > 1000000:  # > 1M rows
                        recommendations["performance_improvements"].append(
                            f"High row count table: {table['tablename']} ({table['row_count']:,} rows) - verify indexes"
                        )
                
                # Check for missing indexes
                missing_indexes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    AND n_distinct > 100
                    AND correlation < 0.1
                """)
                
                if missing_indexes:
                    recommendations["performance_improvements"].append(
                        f"Consider adding indexes on columns with high cardinality: {len(missing_indexes)} candidates found"
                    )
                
                # Check extension status
                extensions = await conn.fetch("""
                    SELECT extname FROM pg_extension 
                    WHERE extname IN ('pg_stat_statements', 'pg_buffercache')
                """)
                
                extension_names = [ext['extname'] for ext in extensions]
                
                if 'pg_stat_statements' not in extension_names:
                    recommendations["monitoring_setup"].append("Enable pg_stat_statements extension for query monitoring")
                
                if 'pg_buffercache' not in extension_names:
                    recommendations["monitoring_setup"].append("Enable pg_buffercache extension for buffer analysis")
                
                # Check for maintenance needs
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_dead_tup,
                        n_live_tup,
                        CASE 
                            WHEN n_live_tup > 0 
                            THEN (n_dead_tup::float / n_live_tup::float) * 100 
                            ELSE 0 
                        END as dead_tuple_percent
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    AND n_live_tup > 0
                """)
                
                for stat in table_stats:
                    if stat['dead_tuple_percent'] > 20:
                        recommendations["maintenance_tasks"].append(
                            f"High dead tuple ratio in {stat['tablename']} ({stat['dead_tuple_percent']:.1f}%) - schedule VACUUM"
                        )
                
                # Add materialized view recommendations
                if not self.materialized_views:
                    recommendations["performance_improvements"].append(
                        "Create materialized views for frequently accessed analytics queries"
                    )
                
                # Add partitioning recommendations
                recommendations["performance_improvements"].append(
                    "Implement time-based partitioning for storage_records table"
                )
                
                return {
                    "recommendations": recommendations,
                    "total_recommendations": sum(len(category) for category in recommendations.values()),
                    "analysis_time": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get optimization recommendations", error=str(e))
            return {"error": str(e)}


class CacheManager:
    """Redis cache management for database query caching."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour default TTL
    
    async def cache_query_result(self, query_key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result in Redis."""
        if not self.redis_client:
            return False
        
        try:
            import json
            import pickle
            
            # Try JSON first, fall back to pickle
            try:
                serialized_result = json.dumps(result)
                cache_type = "json"
            except (TypeError, ValueError):
                serialized_result = pickle.dumps(result)
                cache_type = "pickle"
            
            cache_value = {
                "data": serialized_result,
                "type": cache_type,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"query_cache:{query_key}",
                ttl or self.cache_ttl,
                json.dumps(cache_value)
            )
            
            return True
            
        except Exception as e:
            logger.warning("Failed to cache query result", key=query_key, error=str(e))
            return False
    
    async def get_cached_result(self, query_key: str) -> Optional[Any]:
        """Get cached query result from Redis."""
        if not self.redis_client:
            return None
        
        try:
            import json
            import pickle
            
            cached_value = await self.redis_client.get(f"query_cache:{query_key}")
            if not cached_value:
                return None
            
            cache_data = json.loads(cached_value)
            
            if cache_data["type"] == "json":
                return json.loads(cache_data["data"])
            else:
                return pickle.loads(cache_data["data"])
                
        except Exception as e:
            logger.warning("Failed to get cached result", key=query_key, error=str(e))
            return None
    
    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(f"query_cache:{pattern}")
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning("Failed to invalidate cache pattern", pattern=pattern, error=str(e))
            return 0
