"""
Audit logging module for the Analysis Service.

Handles security event logging and audit trail management.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .config import SecurityConfig
from ..shared_integration import get_shared_database

logger = structlog.get_logger(__name__)


class AuditLogger:
    """Manages security audit logging and event tracking."""

    def __init__(self, config: SecurityConfig, db_pool=None):
        self.config = config
        self.db = db_pool or get_shared_database()
        self.logger = logger.bind(component="audit_logger")

    async def log_security_event(
        self, event_type: str, details: Dict[str, Any], user_id: str = None
    ) -> None:
        """
        Log security event for audit purposes.

        Args:
            event_type: Type of security event
            details: Event details and context
            user_id: Optional user ID associated with event
        """
        if not self.config.enable_audit_logging:
            return

        audit_entry = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "source": "analysis_service",
        }

        try:
            # Store in database
            await self._store_audit_entry(audit_entry)

            # Log to structured logger
            self.logger.info(
                "Security event logged",
                event_type=event_type,
                user_id=user_id,
                **details,
            )

        except Exception as e:
            self.logger.error(
                "Failed to log security event", event_type=event_type, error=str(e)
            )

    async def _store_audit_entry(self, entry: Dict[str, Any]) -> None:
        """Store audit entry in database."""
        query = """
            INSERT INTO security_audit_log (
                timestamp, event_type, user_id, details, source
            ) VALUES ($1, $2, $3, $4, $5)
        """

        await self.db.execute(
            query,
            entry["timestamp"],
            entry["event_type"],
            entry["user_id"],
            entry["details"],
            entry["source"],
        )

    async def get_audit_logs(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with filtering.

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of logs to return

        Returns:
            List of audit log entries
        """
        conditions = []
        params = []
        param_count = 0

        # Build WHERE clause dynamically
        if event_type:
            param_count += 1
            conditions.append(f"event_type = ${param_count}")
            params.append(event_type)

        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)

        if start_time:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)

        if end_time:
            param_count += 1
            conditions.append(f"timestamp <= ${param_count}")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT timestamp, event_type, user_id, details, source
            FROM security_audit_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        try:
            rows = await self.db.fetch(query, *params)
            return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error("Failed to retrieve audit logs", error=str(e))
            return []

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics from audit logs."""
        try:
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)

            # Count events in different time periods
            metrics_query = """
                SELECT 
                    event_type,
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN timestamp > $1 THEN 1 END) as count_24h,
                    COUNT(CASE WHEN timestamp > $2 THEN 1 END) as count_7d
                FROM security_audit_log
                WHERE timestamp > $2
                GROUP BY event_type
            """

            rows = await self.db.fetch(metrics_query, last_24h, last_7d)

            event_metrics = {}
            for row in rows:
                event_metrics[row["event_type"]] = {
                    "total": row["total_count"],
                    "last_24h": row["count_24h"],
                    "last_7d": row["count_7d"],
                }

            # Get total log count
            total_query = "SELECT COUNT(*) as total FROM security_audit_log"
            total_result = await self.db.fetchrow(total_query)

            return {
                "total_events": total_result["total"],
                "event_metrics": event_metrics,
                "retention_days": self.config.audit_log_retention_days,
                "last_updated": now.isoformat(),
            }

        except Exception as e:
            self.logger.error("Failed to get security metrics", error=str(e))
            return {
                "total_events": 0,
                "event_metrics": {},
                "error": str(e),
            }

    async def cleanup_old_logs(self) -> int:
        """
        Clean up old audit logs based on retention policy.

        Returns:
            Number of logs deleted
        """
        if not self.config.enable_audit_logging:
            return 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(
                days=self.config.audit_log_retention_days
            )

            delete_query = """
                DELETE FROM security_audit_log 
                WHERE timestamp < $1
            """

            result = await self.db.execute(delete_query, cutoff_date)

            # Extract number of deleted rows from result
            deleted_count = int(result.split()[-1]) if result else 0

            self.logger.info(
                "Audit log cleanup completed",
                deleted_count=deleted_count,
                cutoff_date=cutoff_date.isoformat(),
            )

            return deleted_count

        except Exception as e:
            self.logger.error("Failed to cleanup audit logs", error=str(e))
            return 0

    async def log_authentication_event(
        self,
        success: bool,
        user_id: str = None,
        method: str = "unknown",
        details: Dict[str, Any] = None,
    ) -> None:
        """Log authentication-specific events."""
        event_type = "authentication_success" if success else "authentication_failed"
        event_details = {"method": method, **(details or {})}

        await self.log_security_event(event_type, event_details, user_id)

    async def log_authorization_event(
        self,
        success: bool,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any] = None,
    ) -> None:
        """Log authorization-specific events."""
        event_type = "authorization_success" if success else "authorization_failed"
        event_details = {"action": action, "resource": resource, **(details or {})}

        await self.log_security_event(event_type, event_details, user_id)

    async def log_rate_limit_event(
        self, client_id: str, endpoint: str, details: Dict[str, Any] = None
    ) -> None:
        """Log rate limiting events."""
        event_details = {
            "client_id": client_id,
            "endpoint": endpoint,
            **(details or {}),
        }

        await self.log_security_event("rate_limit_exceeded", event_details)
