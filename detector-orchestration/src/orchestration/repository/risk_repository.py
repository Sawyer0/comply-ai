from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from shared.database.connection_manager import get_service_db


@dataclass
class RiskAnalysisRecord:
    id: str
    tenant_id: str
    request_correlation_id: str
    risk_level: str
    risk_score: float
    rules_evaluation: Dict[str, Any]
    model_features: Dict[str, Any]
    detector_ids: List[str]
    requested_by: Optional[str]
    requested_via_api_key: Optional[str]
    created_at: str


class RiskAnalysisRepository:
    def __init__(self, *, service_name: str = "orchestration") -> None:
        self._db = get_service_db(service_name)

    async def create_risk_analysis(self, *, fields: Mapping[str, Any]) -> str:
        insert_sql = (
            """
            INSERT INTO risk_analysis_results (
                id,
                tenant_id,
                request_correlation_id,
                risk_level,
                risk_score,
                rules_evaluation,
                model_features,
                detector_ids,
                requested_by,
                requested_via_api_key
            ) VALUES (
                $1,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10
            )
            RETURNING id
            """
        )
        new_id = await self._db.fetchval(
            insert_sql,
            fields.get("id") or str(uuid4()),
            fields["tenant_id"],
            fields["request_correlation_id"],
            fields["risk_level"],
            fields["risk_score"],
            fields["rules_evaluation"],
            fields["model_features"],
            list(fields.get("detector_ids", [])),
            fields.get("requested_by"),
            fields.get("requested_via_api_key"),
        )
        return new_id
