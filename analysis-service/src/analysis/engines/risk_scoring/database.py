"""
Database integration for risk scoring engine.

This module provides database operations for risk scoring components,
following SRP by focusing only on risk scoring data persistence.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from shared.database.connection_manager import get_connection

from ...schemas.domain_models import RiskScore, SecurityFinding, RiskLevel

logger = logging.getLogger(__name__)


class RiskScoringDatabaseManager:
    """
    Database manager for risk scoring operations.

    Handles persistence of risk scores, findings, and related data
    following SRP by focusing only on risk scoring database operations.
    """

    def __init__(self, db_service_name: str = "analysis"):
        self.db_service_name = db_service_name

    async def save_risk_score(
        self, analysis_request_id: UUID, risk_score: RiskScore, tenant_id: str
    ) -> UUID:
        """
        Save risk score to database.

        Args:
            analysis_request_id: ID of the analysis request
            risk_score: Risk score to save
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            UUID of the saved risk score record
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                # Insert risk score
                result = await conn.fetchrow(
                    """
                    INSERT INTO risk_scores (
                        analysis_request_id, tenant_id, composite_score, 
                        risk_level, confidence, technical_risk, business_risk,
                        regulatory_risk, temporal_risk, methodology,
                        created_at, validity_start, validity_end
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    ) RETURNING id
                    """,
                    analysis_request_id,
                    tenant_id,
                    risk_score.composite_score,
                    risk_score.risk_level.value,
                    risk_score.confidence,
                    risk_score.breakdown.technical_risk,
                    risk_score.breakdown.business_risk,
                    risk_score.breakdown.regulatory_risk,
                    risk_score.breakdown.temporal_risk,
                    risk_score.breakdown.methodology,
                    risk_score.timestamp,
                    (
                        risk_score.validity_period.start
                        if risk_score.validity_period
                        else None
                    ),
                    (
                        risk_score.validity_period.end
                        if risk_score.validity_period
                        else None
                    ),
                )

                risk_score_id = result["id"]

                # Save contributing factors
                if risk_score.breakdown.contributing_factors:
                    await self._save_risk_factors(
                        conn, risk_score_id, risk_score.breakdown.contributing_factors
                    )

                logger.info(
                    "Risk score saved to database",
                    risk_score_id=risk_score_id,
                    analysis_request_id=analysis_request_id,
                    tenant_id=tenant_id,
                )

                return risk_score_id

        except Exception as e:
            logger.error(
                "Failed to save risk score to database",
                error=str(e),
                analysis_request_id=analysis_request_id,
                tenant_id=tenant_id,
            )
            raise

    async def _save_risk_factors(self, conn, risk_score_id: UUID, factors: List[Any]):
        """Save risk factors to database."""
        for factor in factors:
            await conn.execute(
                """
                INSERT INTO risk_factors (
                    risk_score_id, factor_name, weight, value, 
                    contribution, justification, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                risk_score_id,
                factor.factor_name,
                factor.weight,
                factor.value,
                factor.contribution,
                factor.justification,
                datetime.utcnow(),
            )

    async def get_risk_score(
        self, risk_score_id: UUID, tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get risk score from database.

        Args:
            risk_score_id: ID of the risk score
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Risk score data or None if not found
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                # Get risk score
                risk_score_row = await conn.fetchrow(
                    """
                    SELECT * FROM risk_scores 
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    risk_score_id,
                    tenant_id,
                )

                if not risk_score_row:
                    return None

                # Get risk factors
                factors_rows = await conn.fetch(
                    """
                    SELECT * FROM risk_factors 
                    WHERE risk_score_id = $1
                    ORDER BY created_at
                    """,
                    risk_score_id,
                )

                # Convert to dict
                risk_score_data = dict(risk_score_row)
                risk_score_data["contributing_factors"] = [
                    dict(row) for row in factors_rows
                ]

                return risk_score_data

        except Exception as e:
            logger.error(
                "Failed to get risk score from database",
                error=str(e),
                risk_score_id=risk_score_id,
                tenant_id=tenant_id,
            )
            raise

    async def get_risk_scores_by_analysis_request(
        self, analysis_request_id: UUID, tenant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all risk scores for an analysis request.

        Args:
            analysis_request_id: ID of the analysis request
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            List of risk score data
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM risk_scores 
                    WHERE analysis_request_id = $1 AND tenant_id = $2
                    ORDER BY created_at DESC
                    """,
                    analysis_request_id,
                    tenant_id,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(
                "Failed to get risk scores from database",
                error=str(e),
                analysis_request_id=analysis_request_id,
                tenant_id=tenant_id,
            )
            raise

    async def get_tenant_risk_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get risk scoring statistics for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Risk scoring statistics
        """
        try:
            async with get_connection(self.db_service_name) as conn:
                # Get overall statistics
                stats_row = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_risk_scores,
                        AVG(composite_score) as avg_composite_score,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_count,
                        COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_count,
                        COUNT(CASE WHEN risk_level = 'medium' THEN 1 END) as medium_count,
                        COUNT(CASE WHEN risk_level = 'low' THEN 1 END) as low_count
                    FROM risk_scores 
                    WHERE tenant_id = $1
                    """,
                    tenant_id,
                )

                return dict(stats_row) if stats_row else {}

        except Exception as e:
            logger.error(
                "Failed to get tenant risk statistics",
                error=str(e),
                tenant_id=tenant_id,
            )
            raise
