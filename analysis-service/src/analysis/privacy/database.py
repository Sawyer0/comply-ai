"""
Database integration for privacy validation.

This module provides database operations for privacy validation components,
following SRP by focusing only on privacy data persistence.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from shared.database.connection_manager import get_connection

logger = logging.getLogger(__name__)


class PrivacyDatabaseManager:
    """
    Database manager for privacy validation operations.

    Handles persistence of privacy validation results and audit trails
    following SRP by focusing only on privacy database operations.
    """

    def __init__(self, db_service_name: str = "analysis"):
        self.db_service_name = db_service_name

    async def log_privacy_validation(
        self,
        analysis_request_id: UUID,
        tenant_id: str,
        validation_result: Dict[str, Any],
        data_type: str = "analysis_data",
    ) -> UUID:
        """
        Log privacy validation result to database.

        Args:
            analysis_request_id: ID of the analysis request
            tenant_id: Tenant ID for multi-tenancy
            validation_result: Privacy validation result
            data_type: Type of data validated

        Returns:
            UUID of the logged privacy validation record
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                # Insert privacy validation log
                result = await conn.fetchrow(
                    """
                    INSERT INTO privacy_validation_logs (
                        analysis_request_id, tenant_id, data_type,
                        is_compliant, violations_count, warnings_count,
                        validation_details, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    ) RETURNING id
                    """,
                    analysis_request_id,
                    tenant_id,
                    data_type,
                    validation_result.get("is_compliant", False),
                    len(validation_result.get("violations", [])),
                    len(validation_result.get("warnings", [])),
                    validation_result,
                    datetime.utcnow(),
                )

                privacy_log_id = result["id"]

                logger.info(
                    "Privacy validation logged to database",
                    privacy_log_id=privacy_log_id,
                    analysis_request_id=analysis_request_id,
                    tenant_id=tenant_id,
                    is_compliant=validation_result.get("is_compliant", False),
                )

                return privacy_log_id

        except Exception as e:
            logger.error(
                "Failed to log privacy validation to database",
                error=str(e),
                analysis_request_id=analysis_request_id,
                tenant_id=tenant_id,
            )
            raise

    async def log_data_retention_check(
        self,
        tenant_id: str,
        data_type: str,
        data_age_days: int,
        retention_status: str,
        details: Dict[str, Any],
    ) -> UUID:
        """
        Log data retention compliance check.

        Args:
            tenant_id: Tenant ID
            data_type: Type of data checked
            data_age_days: Age of data in days
            retention_status: Compliance status (compliant, warning, violation)
            details: Additional details about the check

        Returns:
            UUID of the logged retention check record
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                result = await conn.fetchrow(
                    """
                    INSERT INTO data_retention_logs (
                        tenant_id, data_type, data_age_days,
                        retention_status, check_details, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6
                    ) RETURNING id
                    """,
                    tenant_id,
                    data_type,
                    data_age_days,
                    retention_status,
                    details,
                    datetime.utcnow(),
                )

                retention_log_id = result["id"]

                logger.info(
                    "Data retention check logged to database",
                    retention_log_id=retention_log_id,
                    tenant_id=tenant_id,
                    data_type=data_type,
                    retention_status=retention_status,
                )

                return retention_log_id

        except Exception as e:
            logger.error(
                "Failed to log data retention check to database",
                error=str(e),
                tenant_id=tenant_id,
                data_type=data_type,
            )
            raise

    async def get_privacy_compliance_summary(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get privacy compliance summary for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Privacy compliance summary
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                # Get privacy validation statistics
                privacy_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_validations,
                        COUNT(CASE WHEN is_compliant = true THEN 1 END) as compliant_count,
                        COUNT(CASE WHEN is_compliant = false THEN 1 END) as violation_count,
                        AVG(violations_count) as avg_violations_per_check,
                        AVG(warnings_count) as avg_warnings_per_check
                    FROM privacy_validation_logs 
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                    """,
                    tenant_id,
                )

                # Get retention compliance statistics
                retention_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_retention_checks,
                        COUNT(CASE WHEN retention_status = 'compliant' THEN 1 END) as compliant_retention,
                        COUNT(CASE WHEN retention_status = 'violation' THEN 1 END) as retention_violations,
                        COUNT(CASE WHEN retention_status = 'warning' THEN 1 END) as retention_warnings
                    FROM data_retention_logs 
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                    """,
                    tenant_id,
                )

                return {
                    "privacy_validation": dict(privacy_stats) if privacy_stats else {},
                    "data_retention": dict(retention_stats) if retention_stats else {},
                    "overall_compliance_rate": (
                        privacy_stats["compliant_count"]
                        / privacy_stats["total_validations"]
                        if privacy_stats and privacy_stats["total_validations"] > 0
                        else 0.0
                    ),
                }

        except Exception as e:
            logger.error(
                "Failed to get privacy compliance summary",
                error=str(e),
                tenant_id=tenant_id,
            )
            raise
