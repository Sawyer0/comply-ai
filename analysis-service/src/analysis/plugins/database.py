"""
Plugin database operations for the Analysis Service.

This module provides database operations for plugin management.
Single Responsibility: Handle plugin database operations only.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json

from .interfaces import PluginMetadata, PluginType, PluginStatus
from shared.database.connection_manager import get_service_db

logger = logging.getLogger(__name__)


class PluginDatabaseManager:
    """
    Manages plugin database operations.

    Single Responsibility: Handle plugin data persistence only.
    """

    def __init__(self, db_service_name: str = "analysis"):
        """
        Initialize the plugin database manager.

        Args:
            db_service_name: Database service name for plugin data
        """
        self.db_service_name = db_service_name

    async def register_plugin(
        self, plugin_name: str, metadata: PluginMetadata, config: Dict[str, Any] = None
    ) -> bool:
        """
        Register a plugin in the database.

        Args:
            plugin_name: Unique plugin name
            metadata: Plugin metadata
            config: Plugin configuration

        Returns:
            True if registration successful
        """
        try:
            db = get_service_db(self.db_service_name)

            query = """
            INSERT INTO plugin_registry (
                plugin_name, plugin_version, plugin_type, status, metadata, configuration,
                capabilities, supported_frameworks, min_confidence_threshold, max_batch_size,
                plugin_path, dependencies, created_at, updated_at, health_status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            ON CONFLICT (plugin_name) DO UPDATE SET
                plugin_version = EXCLUDED.plugin_version,
                plugin_type = EXCLUDED.plugin_type,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                configuration = EXCLUDED.configuration,
                capabilities = EXCLUDED.capabilities,
                supported_frameworks = EXCLUDED.supported_frameworks,
                min_confidence_threshold = EXCLUDED.min_confidence_threshold,
                max_batch_size = EXCLUDED.max_batch_size,
                plugin_path = EXCLUDED.plugin_path,
                dependencies = EXCLUDED.dependencies,
                updated_at = EXCLUDED.updated_at,
                health_status = EXCLUDED.health_status
            """

            await db.execute(
                query,
                plugin_name,
                metadata.version,
                metadata.plugin_type.value,
                PluginStatus.INACTIVE.value,
                json.dumps(metadata.dict()),
                json.dumps(config or {}),
                [cap.value for cap in metadata.capabilities],
                metadata.supported_frameworks,
                metadata.min_confidence_threshold,
                metadata.max_batch_size,
                None,  # plugin_path - set when loading from file
                metadata.dependencies,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
                json.dumps({"status": "registered"}),
            )

            logger.info("Registered plugin in database", plugin_name=plugin_name)
            return True

        except Exception as e:
            logger.error(
                "Failed to register plugin in database",
                plugin_name=plugin_name,
                error=str(e),
            )
            return False

    async def update_plugin_status(
        self, plugin_name: str, status: PluginStatus, health_info: Dict[str, Any] = None
    ) -> bool:
        """
        Update plugin status in the database.

        Args:
            plugin_name: Plugin name
            status: New plugin status
            health_info: Health information

        Returns:
            True if update successful
        """
        try:
            db = get_service_db(self.db_service_name)

            query = """
            UPDATE plugin_registry 
            SET status = $2, last_health_check = $3, health_status = $4, updated_at = $5
            WHERE plugin_name = $1
            """

            await db.execute(
                query,
                plugin_name,
                status.value,
                datetime.now(timezone.utc),
                json.dumps(health_info or {}),
                datetime.now(timezone.utc),
            )

            logger.debug(
                "Updated plugin status", plugin_name=plugin_name, status=status.value
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update plugin status", plugin_name=plugin_name, error=str(e)
            )
            return False

    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information from database.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin information if found
        """
        try:
            db = get_service_db(self.db_service_name)

            query = """
            SELECT plugin_name, plugin_version, plugin_type, status, metadata, configuration,
                   capabilities, supported_frameworks, min_confidence_threshold, max_batch_size,
                   plugin_path, dependencies, created_at, updated_at, last_health_check, 
                   health_status, error_log
            FROM plugin_registry
            WHERE plugin_name = $1
            """

            row = await db.fetchrow(query, plugin_name)

            if not row:
                return None

            return {
                "plugin_name": row["plugin_name"],
                "plugin_version": row["plugin_version"],
                "plugin_type": row["plugin_type"],
                "status": row["status"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "configuration": (
                    json.loads(row["configuration"]) if row["configuration"] else {}
                ),
                "capabilities": row["capabilities"] or [],
                "supported_frameworks": row["supported_frameworks"] or [],
                "min_confidence_threshold": row["min_confidence_threshold"],
                "max_batch_size": row["max_batch_size"],
                "plugin_path": row["plugin_path"],
                "dependencies": row["dependencies"] or [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_health_check": row["last_health_check"],
                "health_status": (
                    json.loads(row["health_status"]) if row["health_status"] else {}
                ),
                "error_log": row["error_log"] or [],
            }

        except Exception as e:
            logger.error(
                "Failed to get plugin info", plugin_name=plugin_name, error=str(e)
            )
            return None

    async def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        status: Optional[PluginStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        List plugins from database with optional filters.

        Args:
            plugin_type: Optional plugin type filter
            status: Optional status filter

        Returns:
            List of plugin information
        """
        try:
            db = get_service_db(self.db_service_name)

            # Build query with filters
            where_conditions = []
            params = []
            param_count = 0

            if plugin_type:
                param_count += 1
                where_conditions.append(f"plugin_type = ${param_count}")
                params.append(plugin_type.value)

            if status:
                param_count += 1
                where_conditions.append(f"status = ${param_count}")
                params.append(status.value)

            where_clause = (
                "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            )

            query = f"""
            SELECT plugin_name, plugin_version, plugin_type, status, metadata, configuration,
                   capabilities, supported_frameworks, min_confidence_threshold, max_batch_size,
                   created_at, updated_at, last_health_check, health_status, error_log
            FROM plugin_registry
            {where_clause}
            ORDER BY created_at DESC
            """

            rows = await db.fetch(query, *params)

            plugins = []
            for row in rows:
                plugins.append(
                    {
                        "plugin_name": row["plugin_name"],
                        "plugin_version": row["plugin_version"],
                        "plugin_type": row["plugin_type"],
                        "status": row["status"],
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                        "configuration": (
                            json.loads(row["configuration"])
                            if row["configuration"]
                            else {}
                        ),
                        "capabilities": row["capabilities"] or [],
                        "supported_frameworks": row["supported_frameworks"] or [],
                        "min_confidence_threshold": row["min_confidence_threshold"],
                        "max_batch_size": row["max_batch_size"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "last_health_check": row["last_health_check"],
                        "health_status": (
                            json.loads(row["health_status"])
                            if row["health_status"]
                            else {}
                        ),
                        "error_log": row["error_log"] or [],
                    }
                )

            return plugins

        except Exception as e:
            logger.error("Failed to list plugins", error=str(e))
            return []

    async def record_plugin_execution(
        self,
        plugin_name: str,
        tenant_id: str,
        request_id: str,
        execution_type: str,
        input_hash: str,
        confidence: Optional[float],
        processing_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Record plugin execution in database.

        Args:
            plugin_name: Plugin name
            tenant_id: Tenant identifier
            request_id: Request identifier
            execution_type: Type of execution (single, batch)
            input_hash: Hash of input data
            confidence: Confidence score
            processing_time_ms: Processing time in milliseconds
            success: Whether execution was successful
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            True if recording successful
        """
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            query = """
            INSERT INTO plugin_executions (
                plugin_name, tenant_id, request_id, execution_type, input_hash,
                confidence, processing_time_ms, success, error_message, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            """

            await db.execute(
                query,
                plugin_name,
                tenant_id,
                request_id,
                execution_type,
                input_hash,
                confidence,
                processing_time_ms,
                success,
                error_message,
                json.dumps(metadata or {}),
            )

            logger.debug(
                "Recorded plugin execution",
                plugin_name=plugin_name,
                tenant_id=tenant_id,
                request_id=request_id,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to record plugin execution",
                plugin_name=plugin_name,
                tenant_id=tenant_id,
                error=str(e),
            )
            return False

    async def get_plugin_performance_stats(
        self, plugin_name: str, days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Get plugin performance statistics.

        Args:
            plugin_name: Plugin name
            days_back: Number of days to look back

        Returns:
            Performance statistics
        """
        try:
            db = get_service_db(self.db_service_name)

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

            query = """
            SELECT 
                COUNT(*) as total_executions,
                COUNT(CASE WHEN success = true THEN 1 END) as successful_executions,
                COUNT(CASE WHEN success = false THEN 1 END) as failed_executions,
                AVG(processing_time_ms) as avg_processing_time_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_processing_time_ms,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM plugin_executions
            WHERE plugin_name = $1 
                AND created_at >= $2
                AND success = true
            """

            row = await db.fetchrow(query, plugin_name, cutoff_date)

            if not row or not row["total_executions"]:
                return {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "success_rate": 0.0,
                    "avg_processing_time_ms": 0.0,
                    "p95_processing_time_ms": 0.0,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "max_confidence": 0.0,
                }

            total = row["total_executions"]
            successful = row["successful_executions"]
            success_rate = (successful / total * 100) if total > 0 else 0.0

            return {
                "total_executions": total,
                "successful_executions": successful,
                "failed_executions": row["failed_executions"],
                "success_rate": success_rate,
                "avg_processing_time_ms": float(row["avg_processing_time_ms"] or 0.0),
                "p95_processing_time_ms": float(row["p95_processing_time_ms"] or 0.0),
                "avg_confidence": float(row["avg_confidence"] or 0.0),
                "min_confidence": float(row["min_confidence"] or 0.0),
                "max_confidence": float(row["max_confidence"] or 0.0),
            }

        except Exception as e:
            logger.error(
                "Failed to get plugin performance stats",
                plugin_name=plugin_name,
                error=str(e),
            )
            return {}

    async def add_plugin_error(self, plugin_name: str, error_message: str) -> bool:
        """
        Add error message to plugin error log.

        Args:
            plugin_name: Plugin name
            error_message: Error message to add

        Returns:
            True if successful
        """
        try:
            db = get_service_db(self.db_service_name)

            # Get current error log
            query = "SELECT error_log FROM plugin_registry WHERE plugin_name = $1"
            row = await db.fetchrow(query, plugin_name)

            if not row:
                return False

            current_errors = row["error_log"] or []

            # Add new error with timestamp
            error_entry = f"{datetime.now(timezone.utc).isoformat()}: {error_message}"
            current_errors.append(error_entry)

            # Keep only last 50 errors
            if len(current_errors) > 50:
                current_errors = current_errors[-50:]

            # Update error log
            update_query = """
            UPDATE plugin_registry 
            SET error_log = $2, updated_at = $3
            WHERE plugin_name = $1
            """

            await db.execute(
                update_query, plugin_name, current_errors, datetime.now(timezone.utc)
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to add plugin error", plugin_name=plugin_name, error=str(e)
            )
            return False

    async def cleanup_old_executions(self, retention_days: int = 30) -> int:
        """
        Clean up old plugin execution records.

        Args:
            retention_days: Number of days to retain execution records

        Returns:
            Number of records cleaned up
        """
        try:
            db = get_service_db(self.db_service_name)

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            query = """
            DELETE FROM plugin_executions
            WHERE created_at < $1
            """

            result = await db.execute(query, cutoff_date)

            # Extract number of deleted rows from result
            deleted_count = 0
            if hasattr(result, "split"):
                # PostgreSQL returns "DELETE n" format
                parts = result.split()
                if len(parts) == 2 and parts[0] == "DELETE":
                    deleted_count = int(parts[1])

            logger.info(
                "Cleaned up old plugin executions",
                cutoff_date=cutoff_date,
                deleted_count=deleted_count,
            )

            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup old plugin executions", error=str(e))
            return 0
