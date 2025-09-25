"""
Tenant analytics management for the Analysis Service.

This module provides analytics tracking and reporting for tenants.
Single Responsibility: Manage tenant analytics data only.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from .models import TenantAnalytics
from shared.database.connection_manager import get_service_db

logger = logging.getLogger(__name__)


class AnalyticsManager:
    """
    Manages tenant analytics collection, aggregation, and reporting.

    Single Responsibility: Handle analytics data operations only.
    """

    def __init__(self, db_service_name: str = "analysis"):
        """
        Initialize the analytics manager.

        Args:
            db_service_name: Database service name for analytics storage
        """
        self.db_service_name = db_service_name

    async def record_request_metrics(
        self, tenant_id: str, metrics: Dict[str, Any]
    ) -> bool:
        """
        Record request-level metrics for a tenant.

        Args:
            tenant_id: Tenant identifier
            metrics: Request metrics to record

        Returns:
            True if successful, False otherwise
        """
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            # Get or create today's analytics record
            today = datetime.now(timezone.utc).date()

            # Update request metrics
            request_count = metrics.get("request_count", 1)
            success = metrics.get("success", True)
            response_time = metrics.get("response_time_ms", 0.0)
            confidence = metrics.get("confidence", 0.0)
            cpu_minutes = metrics.get("cpu_minutes", 0.0)
            storage_mb = metrics.get("storage_mb", 0.0)
            ml_calls = metrics.get("ml_inference_calls", 0)

            # Determine analysis type counts
            analysis_type = metrics.get("analysis_type", "")
            pattern_count = 1 if "pattern" in analysis_type else 0
            risk_count = 1 if "risk" in analysis_type else 0
            compliance_count = 1 if "compliance" in analysis_type else 0

            # Framework usage
            framework = metrics.get("framework", "")
            framework_usage = {framework: 1} if framework else {}

            # Error types
            error_type = metrics.get("error_type", "")
            error_types = {error_type: 1} if error_type and not success else {}

            # Insert or update daily analytics
            query = """
            INSERT INTO tenant_analytics (
                tenant_id, analytics_type, time_period, period_start, period_end,
                total_requests, successful_requests, failed_requests,
                avg_response_time_ms, p95_response_time_ms,
                avg_confidence_score, low_confidence_count,
                cpu_minutes_used, storage_mb_used, ml_inference_calls,
                pattern_recognition_count, risk_scoring_count, compliance_mapping_count,
                framework_usage, error_types
            ) VALUES (
                $1, 'daily', 'daily', $2, $2,
                $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (tenant_id, analytics_type, time_period, period_start)
            DO UPDATE SET
                total_requests = tenant_analytics.total_requests + EXCLUDED.total_requests,
                successful_requests = tenant_analytics.successful_requests + EXCLUDED.successful_requests,
                failed_requests = tenant_analytics.failed_requests + EXCLUDED.failed_requests,
                avg_response_time_ms = CASE 
                    WHEN (tenant_analytics.total_requests + EXCLUDED.total_requests) > 0 
                    THEN ((tenant_analytics.avg_response_time_ms * tenant_analytics.total_requests) + 
                          (EXCLUDED.avg_response_time_ms * EXCLUDED.total_requests)) / 
                         (tenant_analytics.total_requests + EXCLUDED.total_requests)
                    ELSE 0 
                END,
                p95_response_time_ms = GREATEST(tenant_analytics.p95_response_time_ms, EXCLUDED.p95_response_time_ms),
                avg_confidence_score = CASE 
                    WHEN (tenant_analytics.total_requests + EXCLUDED.total_requests) > 0 
                    THEN ((tenant_analytics.avg_confidence_score * tenant_analytics.total_requests) + 
                          (EXCLUDED.avg_confidence_score * EXCLUDED.total_requests)) / 
                         (tenant_analytics.total_requests + EXCLUDED.total_requests)
                    ELSE 0 
                END,
                low_confidence_count = tenant_analytics.low_confidence_count + EXCLUDED.low_confidence_count,
                cpu_minutes_used = tenant_analytics.cpu_minutes_used + EXCLUDED.cpu_minutes_used,
                storage_mb_used = tenant_analytics.storage_mb_used + EXCLUDED.storage_mb_used,
                ml_inference_calls = tenant_analytics.ml_inference_calls + EXCLUDED.ml_inference_calls,
                pattern_recognition_count = tenant_analytics.pattern_recognition_count + EXCLUDED.pattern_recognition_count,
                risk_scoring_count = tenant_analytics.risk_scoring_count + EXCLUDED.risk_scoring_count,
                compliance_mapping_count = tenant_analytics.compliance_mapping_count + EXCLUDED.compliance_mapping_count,
                framework_usage = COALESCE(tenant_analytics.framework_usage, '{}')::jsonb || COALESCE(EXCLUDED.framework_usage, '{}')::jsonb,
                error_types = COALESCE(tenant_analytics.error_types, '{}')::jsonb || COALESCE(EXCLUDED.error_types, '{}')::jsonb,
                updated_at = NOW()
            """

            await db.execute(
                query,
                tenant_id,
                today,
                request_count,
                1 if success else 0,
                0 if success else 1,
                response_time,
                response_time,  # Use same value for P95 in simple case
                confidence,
                1 if confidence > 0 and confidence < 0.7 else 0,
                cpu_minutes,
                storage_mb,
                ml_calls,
                pattern_count,
                risk_count,
                compliance_count,
                json.dumps(framework_usage),
                json.dumps(error_types),
            )

            logger.debug("Recorded request metrics", tenant_id=tenant_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to record request metrics", tenant_id=tenant_id, error=str(e)
            )
            return False

    async def get_analytics(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        analytics_type: str = "daily",
    ) -> Optional[TenantAnalytics]:
        """
        Get tenant analytics for a specific period.

        Args:
            tenant_id: Tenant identifier
            period_start: Analytics period start
            period_end: Analytics period end
            analytics_type: Type of analytics (daily, weekly, monthly)

        Returns:
            Aggregated tenant analytics if available
        """
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            # Query analytics records for the period
            query = """
            SELECT 
                SUM(total_requests) as total_requests,
                SUM(successful_requests) as successful_requests,
                SUM(failed_requests) as failed_requests,
                AVG(avg_response_time_ms) as avg_response_time_ms,
                MAX(p95_response_time_ms) as p95_response_time_ms,
                AVG(avg_confidence_score) as avg_confidence_score,
                SUM(low_confidence_count) as low_confidence_count,
                SUM(cpu_minutes_used) as cpu_minutes_used,
                SUM(storage_mb_used) as storage_mb_used,
                SUM(ml_inference_calls) as ml_inference_calls,
                SUM(pattern_recognition_count) as pattern_recognition_count,
                SUM(risk_scoring_count) as risk_scoring_count,
                SUM(compliance_mapping_count) as compliance_mapping_count
            FROM tenant_analytics
            WHERE tenant_id = $1 
                AND analytics_type = $2
                AND period_start >= $3::date 
                AND period_start <= $4::date
            """

            row = await db.fetchrow(
                query, tenant_id, analytics_type, period_start.date(), period_end.date()
            )

            if not row or not row["total_requests"]:
                # Return empty analytics if no data found
                return TenantAnalytics(
                    tenant_id=tenant_id,
                    period_start=period_start,
                    period_end=period_end,
                )

            # Get framework usage and error types separately
            framework_query = """
            SELECT framework_usage, error_types
            FROM tenant_analytics
            WHERE tenant_id = $1 
                AND analytics_type = $2
                AND period_start >= $3::date 
                AND period_start <= $4::date
                AND (framework_usage IS NOT NULL OR error_types IS NOT NULL)
            """

            framework_rows = await db.fetch(
                framework_query,
                tenant_id,
                analytics_type,
                period_start.date(),
                period_end.date(),
            )

            # Aggregate framework usage and error types
            aggregated_framework_usage = {}
            aggregated_error_types = {}

            for fr in framework_rows:
                if fr["framework_usage"]:
                    framework_data = (
                        json.loads(fr["framework_usage"])
                        if isinstance(fr["framework_usage"], str)
                        else fr["framework_usage"]
                    )
                    for framework, count in framework_data.items():
                        aggregated_framework_usage[framework] = (
                            aggregated_framework_usage.get(framework, 0) + count
                        )

                if fr["error_types"]:
                    error_data = (
                        json.loads(fr["error_types"])
                        if isinstance(fr["error_types"], str)
                        else fr["error_types"]
                    )
                    for error_type, count in error_data.items():
                        aggregated_error_types[error_type] = (
                            aggregated_error_types.get(error_type, 0) + count
                        )

            return TenantAnalytics(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
                total_requests=row["total_requests"] or 0,
                successful_requests=row["successful_requests"] or 0,
                failed_requests=row["failed_requests"] or 0,
                avg_response_time_ms=float(row["avg_response_time_ms"] or 0.0),
                p95_response_time_ms=float(row["p95_response_time_ms"] or 0.0),
                avg_confidence_score=float(row["avg_confidence_score"] or 0.0),
                low_confidence_count=row["low_confidence_count"] or 0,
                cpu_minutes_used=float(row["cpu_minutes_used"] or 0.0),
                storage_mb_used=float(row["storage_mb_used"] or 0.0),
                ml_inference_calls=row["ml_inference_calls"] or 0,
                pattern_recognition_count=row["pattern_recognition_count"] or 0,
                risk_scoring_count=row["risk_scoring_count"] or 0,
                compliance_mapping_count=row["compliance_mapping_count"] or 0,
                framework_usage=aggregated_framework_usage,
                error_types=aggregated_error_types,
            )

        except Exception as e:
            logger.error(
                "Failed to get tenant analytics", tenant_id=tenant_id, error=str(e)
            )
            return None

    async def generate_weekly_analytics(
        self, tenant_id: str, week_start: datetime
    ) -> bool:
        """
        Generate weekly analytics by aggregating daily analytics.

        Args:
            tenant_id: Tenant identifier
            week_start: Start of the week

        Returns:
            True if successful, False otherwise
        """
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            week_end = week_start + timedelta(days=6)

            # Aggregate daily analytics for the week
            query = """
            SELECT 
                SUM(total_requests) as total_requests,
                SUM(successful_requests) as successful_requests,
                SUM(failed_requests) as failed_requests,
                AVG(avg_response_time_ms) as avg_response_time_ms,
                MAX(p95_response_time_ms) as p95_response_time_ms,
                AVG(avg_confidence_score) as avg_confidence_score,
                SUM(low_confidence_count) as low_confidence_count,
                SUM(cpu_minutes_used) as cpu_minutes_used,
                SUM(storage_mb_used) as storage_mb_used,
                SUM(ml_inference_calls) as ml_inference_calls,
                SUM(pattern_recognition_count) as pattern_recognition_count,
                SUM(risk_scoring_count) as risk_scoring_count,
                SUM(compliance_mapping_count) as compliance_mapping_count
            FROM tenant_analytics
            WHERE tenant_id = $1 
                AND analytics_type = 'daily'
                AND period_start >= $2::date 
                AND period_start <= $3::date
            """

            row = await db.fetchrow(
                query, tenant_id, week_start.date(), week_end.date()
            )

            if not row or not row["total_requests"]:
                logger.info(
                    "No daily analytics found for weekly aggregation",
                    tenant_id=tenant_id,
                    week_start=week_start.date(),
                )
                return True

            # Aggregate framework usage and error types
            framework_query = """
            SELECT framework_usage, error_types
            FROM tenant_analytics
            WHERE tenant_id = $1 
                AND analytics_type = 'daily'
                AND period_start >= $2::date 
                AND period_start <= $3::date
                AND (framework_usage IS NOT NULL OR error_types IS NOT NULL)
            """

            framework_rows = await db.fetch(
                framework_query, tenant_id, week_start.date(), week_end.date()
            )

            aggregated_framework_usage = {}
            aggregated_error_types = {}

            for fr in framework_rows:
                if fr["framework_usage"]:
                    framework_data = (
                        json.loads(fr["framework_usage"])
                        if isinstance(fr["framework_usage"], str)
                        else fr["framework_usage"]
                    )
                    for framework, count in framework_data.items():
                        aggregated_framework_usage[framework] = (
                            aggregated_framework_usage.get(framework, 0) + count
                        )

                if fr["error_types"]:
                    error_data = (
                        json.loads(fr["error_types"])
                        if isinstance(fr["error_types"], str)
                        else fr["error_types"]
                    )
                    for error_type, count in error_data.items():
                        aggregated_error_types[error_type] = (
                            aggregated_error_types.get(error_type, 0) + count
                        )

            # Insert or update weekly analytics
            insert_query = """
            INSERT INTO tenant_analytics (
                tenant_id, analytics_type, time_period, period_start, period_end,
                total_requests, successful_requests, failed_requests,
                avg_response_time_ms, p95_response_time_ms,
                avg_confidence_score, low_confidence_count,
                cpu_minutes_used, storage_mb_used, ml_inference_calls,
                pattern_recognition_count, risk_scoring_count, compliance_mapping_count,
                framework_usage, error_types
            ) VALUES (
                $1, 'weekly', 'weekly', $2, $3,
                $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            ON CONFLICT (tenant_id, analytics_type, time_period, period_start)
            DO UPDATE SET
                period_end = EXCLUDED.period_end,
                total_requests = EXCLUDED.total_requests,
                successful_requests = EXCLUDED.successful_requests,
                failed_requests = EXCLUDED.failed_requests,
                avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                p95_response_time_ms = EXCLUDED.p95_response_time_ms,
                avg_confidence_score = EXCLUDED.avg_confidence_score,
                low_confidence_count = EXCLUDED.low_confidence_count,
                cpu_minutes_used = EXCLUDED.cpu_minutes_used,
                storage_mb_used = EXCLUDED.storage_mb_used,
                ml_inference_calls = EXCLUDED.ml_inference_calls,
                pattern_recognition_count = EXCLUDED.pattern_recognition_count,
                risk_scoring_count = EXCLUDED.risk_scoring_count,
                compliance_mapping_count = EXCLUDED.compliance_mapping_count,
                framework_usage = EXCLUDED.framework_usage,
                error_types = EXCLUDED.error_types,
                updated_at = NOW()
            """

            await db.execute(
                insert_query,
                tenant_id,
                week_start.date(),
                week_end.date(),
                row["total_requests"] or 0,
                row["successful_requests"] or 0,
                row["failed_requests"] or 0,
                float(row["avg_response_time_ms"] or 0.0),
                float(row["p95_response_time_ms"] or 0.0),
                float(row["avg_confidence_score"] or 0.0),
                row["low_confidence_count"] or 0,
                float(row["cpu_minutes_used"] or 0.0),
                float(row["storage_mb_used"] or 0.0),
                row["ml_inference_calls"] or 0,
                row["pattern_recognition_count"] or 0,
                row["risk_scoring_count"] or 0,
                row["compliance_mapping_count"] or 0,
                json.dumps(aggregated_framework_usage),
                json.dumps(aggregated_error_types),
            )

            logger.info(
                "Generated weekly analytics",
                tenant_id=tenant_id,
                week_start=week_start.date(),
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to generate weekly analytics", tenant_id=tenant_id, error=str(e)
            )
            return False

    async def get_analytics_trends(
        self,
        tenant_id: str,
        metric_name: str,
        period_type: str = "daily",
        days_back: int = 30,
    ) -> List[Tuple[datetime, float]]:
        """
        Get analytics trends for a specific metric over time.

        Args:
            tenant_id: Tenant identifier
            metric_name: Name of the metric to track
            period_type: Type of period (daily, weekly, monthly)
            days_back: Number of days to look back

        Returns:
            List of (date, value) tuples for the trend
        """
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days_back)

            # Map metric names to database columns
            metric_column_map = {
                "total_requests": "total_requests",
                "success_rate": "CASE WHEN total_requests > 0 THEN (successful_requests::float / total_requests * 100) ELSE 0 END",
                "avg_response_time": "avg_response_time_ms",
                "avg_confidence": "avg_confidence_score",
                "cpu_usage": "cpu_minutes_used",
                "storage_usage": "storage_mb_used",
            }

            if metric_name not in metric_column_map:
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_column = metric_column_map[metric_name]

            query = f"""
            SELECT period_start, {metric_column} as metric_value
            FROM tenant_analytics
            WHERE tenant_id = $1 
                AND analytics_type = $2
                AND period_start >= $3
                AND period_start <= $4
            ORDER BY period_start ASC
            """

            rows = await db.fetch(query, tenant_id, period_type, start_date, end_date)

            return [
                (row["period_start"], float(row["metric_value"] or 0.0)) for row in rows
            ]

        except Exception as e:
            logger.error(
                "Failed to get analytics trends",
                tenant_id=tenant_id,
                metric_name=metric_name,
                error=str(e),
            )
            return []

    async def cleanup_old_analytics(self, retention_days: int = 365) -> int:
        """
        Clean up old analytics records beyond retention period.

        Args:
            retention_days: Number of days to retain analytics

        Returns:
            Number of records cleaned up
        """
        try:
            db = get_service_db(self.db_service_name)

            cutoff_date = datetime.now(timezone.utc).date() - timedelta(
                days=retention_days
            )

            # Delete old daily analytics (keep weekly and monthly longer)
            query = """
            DELETE FROM tenant_analytics
            WHERE analytics_type = 'daily' 
                AND period_start < $1
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
                "Cleaned up old analytics",
                cutoff_date=cutoff_date,
                deleted_count=deleted_count,
            )

            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup old analytics", error=str(e))
            return 0
